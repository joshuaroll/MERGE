#!/usr/bin/env python
"""
Training script for CheMoE on PDGrapher data (10,716 genes).

CheMoE adapted for PDGrapher: 4 experts, top-k=2 sparse routing, gene-aware processing.
Uses same data loading and evaluation pattern as train_transigen_pdg.py.

Usage:
    python train_chemoe_pdg.py --cell_line A549 --fold 0 --gpu 0
    python train_chemoe_pdg.py --cell_line A549 --fold 0 --gpu 0 --mol_encoder morgan
    python train_chemoe_pdg.py --cell_line A549 --fold 0 --gpu 0 --epochs 2  # Quick test
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr
import wandb
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.chemoe_pdg import CheMoE_PDG, CELL_LINE_TO_IDX


# Default paths (same as train_transigen_pdg.py)
DATA_PICKLE = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl"
DISEASED_PICKLE = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl"
SPLITS_BASE = "/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical"
MOLECULE_EMBEDDINGS = "/raid/home/joshua/projects/cbio_032024/zenodo_TranSiGen/TranSiGen/data/LINCS2020/KPGT_emb2304.pickle"
RESULTS_CSV = "/raid/home/joshua/projects/PDGrapher_Baseline_Models/data/topk_r2_results.csv"


def save_results_to_csv(model_name, cell_line, fold, metrics, csv_path=RESULTS_CSV):
    """
    Save top-k R2 results to a shared CSV file.
    Appends new results or updates existing ones.
    Thread-safe with file locking.
    """
    import pandas as pd
    from datetime import datetime
    import filelock
    from pathlib import Path

    # Prepare new row
    new_row = {
        'model': model_name,
        'cell_line': cell_line,
        'fold': fold,
        'r2_top20': metrics.get('r2_top20', 0.0),
        'r2_top40': metrics.get('r2_top40', 0.0),
        'r2_top80': metrics.get('r2_top80', 0.0),
        'pearson_de': metrics.get('pearson', metrics.get('pearson_de', 0.0)),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_path.with_suffix('.csv.lock')

    # Use file lock for concurrent access safety
    lock = filelock.FileLock(lock_path, timeout=30)

    with lock:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Check if this model/cell_line/fold already exists
            mask = (df['model'] == model_name) & (df['cell_line'] == cell_line) & (df['fold'] == fold)
            if mask.any():
                # Update existing row
                for col, val in new_row.items():
                    if col in df.columns:
                        df.loc[mask, col] = val
            else:
                # Append new row
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        # Sort by model, cell_line, fold for readability
        df = df.sort_values(['model', 'cell_line', 'fold']).reset_index(drop=True)
        df.to_csv(csv_path, index=False)

    print(f"Results saved to {csv_path}")


class PDGrapherDataset(Dataset):
    """PyTorch Dataset for PDGrapher data with cell line index support."""

    def __init__(self, treated_df, diseased_df, gene_cols, smiles_to_embedding,
                 default_embedding, cell_line):
        """
        Args:
            treated_df: DataFrame with treated expression
            diseased_df: DataFrame with diseased expression
            gene_cols: List of gene column names
            smiles_to_embedding: Dict mapping SMILES to molecular embeddings
            default_embedding: Default embedding for unknown molecules
            cell_line: Cell line name (for cell index)
        """
        self.treated = treated_df[gene_cols].values.astype(np.float32)
        self.diseased = diseased_df[gene_cols].values.astype(np.float32)
        self.smiles = treated_df['smiles'].values if 'smiles' in treated_df.columns else None
        self.smiles_to_embedding = smiles_to_embedding
        self.default_embedding = default_embedding
        self.gene_cols = gene_cols
        self.cell_idx = CELL_LINE_TO_IDX[cell_line]

    def __len__(self):
        return len(self.treated)

    def __getitem__(self, idx):
        x1 = self.diseased[idx]  # diseased (basal)
        x2 = self.treated[idx]   # treated (target)

        # Get molecular embedding
        if self.smiles is not None and self.smiles[idx] in self.smiles_to_embedding:
            mol_embed = self.smiles_to_embedding[self.smiles[idx]]
        else:
            mol_embed = self.default_embedding

        mol_embed = np.array(mol_embed, dtype=np.float32)

        return (
            torch.FloatTensor(x1),
            torch.FloatTensor(x2),
            torch.FloatTensor(mol_embed),
            self.cell_idx
        )


def load_data(cell_line, fold):
    """Load PDGrapher data for a specific cell line and fold."""
    print(f"Loading data for {cell_line} fold {fold}...")

    # Load expression data
    with open(DATA_PICKLE, 'rb') as f:
        treated_df = pickle.load(f)
    with open(DISEASED_PICKLE, 'rb') as f:
        diseased_df = pickle.load(f)

    # Filter by cell line
    mask = treated_df['cell_id'] == cell_line
    treated_df = treated_df[mask].reset_index(drop=True)
    diseased_df = diseased_df[mask].reset_index(drop=True)

    # Get gene columns
    metadata_cols = ['sig_id', 'idx', 'pert_id', 'pert_type', 'cell_id', 'pert_idose',
                     'cell_type', 'dose', 'smiles']
    gene_cols = [c for c in treated_df.columns if c not in metadata_cols]

    print(f"  Samples: {len(treated_df)}, Genes: {len(gene_cols)}")

    # Load PDGrapher splits (same as MultiDCP_CheMoE_pdg)
    splits_path = os.path.join(SPLITS_BASE, cell_line, "random", "5fold", "splits.pt")
    splits = torch.load(splits_path, weights_only=False)

    # PDGrapher uses 1-indexed folds
    fold_key = fold + 1
    fold_splits = splits[fold_key]

    # Get backward indices (same as MultiDCP_CheMoE_pdg)
    train_indices = fold_splits['train_index_backward']
    val_indices = fold_splits['val_index_backward']
    test_indices = fold_splits['test_index_backward']

    if isinstance(train_indices, torch.Tensor):
        train_indices = train_indices.numpy()
    if isinstance(val_indices, torch.Tensor):
        val_indices = val_indices.numpy()
    if isinstance(test_indices, torch.Tensor):
        test_indices = test_indices.numpy()

    # Map split indices to dataframe rows via 'idx' column
    idx_col = treated_df['idx'].values
    train_idx_set = set(train_indices.tolist())
    val_idx_set = set(val_indices.tolist())
    test_idx_set = set(test_indices.tolist())

    train_mask = np.array([idx in train_idx_set for idx in idx_col])
    val_mask = np.array([idx in val_idx_set for idx in idx_col])
    test_mask = np.array([idx in test_idx_set for idx in idx_col])

    train_idx = np.where(train_mask | val_mask)[0]  # Combine train+val
    test_idx = np.where(test_mask)[0]

    print(f"  Using PDGrapher splits (fold {fold_key})")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    return treated_df, diseased_df, gene_cols, train_idx, test_idx


def load_molecule_embeddings(mol_encoder='kpgt'):
    """Load molecular embeddings (KPGT or Morgan FP)."""
    smiles_to_embedding = {}

    if mol_encoder == 'kpgt':
        if os.path.exists(MOLECULE_EMBEDDINGS):
            with open(MOLECULE_EMBEDDINGS, 'rb') as f:
                smiles_to_embedding = pickle.load(f)
            print(f"Loaded {len(smiles_to_embedding)} KPGT molecular embeddings")

        # Determine embedding dimension
        if smiles_to_embedding:
            sample_emb = next(iter(smiles_to_embedding.values()))
            emb_dim = len(sample_emb) if hasattr(sample_emb, '__len__') else 2304
        else:
            emb_dim = 2304
    else:  # morgan
        # For Morgan FP, we'll compute on-the-fly (cache in Dataset if needed)
        emb_dim = 1024
        print(f"Using Morgan fingerprints (1024-dim)")

        # Try to load pre-computed Morgan FPs if available
        morgan_path = "/raid/home/joshua/projects/PDGrapher_Baseline_Models/data/morgan_fps.pkl"
        if os.path.exists(morgan_path):
            with open(morgan_path, 'rb') as f:
                smiles_to_embedding = pickle.load(f)
            print(f"Loaded {len(smiles_to_embedding)} pre-computed Morgan fingerprints")
        else:
            # Compute Morgan FPs from drug SMILES
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem

                # Load all SMILES from data
                drug_csv = "/raid/home/joshua/projects/MultiDCP_pdg/data/all_drugs_pdg.csv"
                if os.path.exists(drug_csv):
                    drug_df = pd.read_csv(drug_csv)
                    smiles_list = drug_df['smiles'].dropna().unique()

                    for smiles in smiles_list:
                        try:
                            mol = Chem.MolFromSmiles(smiles)
                            if mol:
                                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                                smiles_to_embedding[smiles] = np.array(fp, dtype=np.float32)
                        except:
                            pass

                    print(f"Computed {len(smiles_to_embedding)} Morgan fingerprints")

                    # Cache for future use
                    Path(morgan_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(morgan_path, 'wb') as f:
                        pickle.dump(smiles_to_embedding, f)
                    print(f"Cached Morgan fingerprints to {morgan_path}")
            except ImportError:
                print("Warning: RDKit not available. Using zero vectors for missing Morgan FPs.")

    default_embedding = np.zeros(emb_dim, dtype=np.float32)

    return smiles_to_embedding, default_embedding, emb_dim


def evaluate_model(model, dataloader, device):
    """Evaluate model on test data."""
    model.eval()

    all_diseased = []
    all_treated = []
    all_pred = []

    with torch.no_grad():
        for diseased, treated, mol_embed, cell_idx in dataloader:
            diseased = diseased.to(device)
            mol_embed = mol_embed.to(device)

            # Handle cell_idx (could be int or tensor)
            if isinstance(cell_idx, int):
                cell_idx = torch.tensor([cell_idx] * diseased.size(0), device=device)
            else:
                cell_idx = cell_idx.to(device)

            pred = model(diseased, mol_embed, cell_idx)

            all_diseased.append(diseased.cpu().numpy())
            all_treated.append(treated.numpy())
            all_pred.append(pred.cpu().numpy())

    diseased = np.vstack(all_diseased)
    treated = np.vstack(all_treated)
    pred = np.vstack(all_pred)

    # Compute metrics on differential expression
    true_de = treated - diseased
    pred_de = pred - diseased

    results = {}

    # Top-k R2 scores (using Pearson^2 for consistency with other models)
    for k in [20, 40, 80]:
        r2_scores = []
        for i in range(len(treated)):
            de_mag = np.abs(true_de[i])
            top_k_idx = np.argsort(de_mag)[-k:]
            try:
                r, _ = pearsonr(true_de[i, top_k_idx], pred_de[i, top_k_idx])
                r2 = r ** 2
                if not np.isnan(r2) and not np.isinf(r2):
                    r2_scores.append(r2)
            except:
                pass
        results[f'r2_top{k}'] = np.mean(r2_scores) if r2_scores else 0

    # Overall Pearson correlation
    pearson_scores = []
    for i in range(len(treated)):
        try:
            r, _ = pearsonr(true_de[i], pred_de[i])
            if not np.isnan(r):
                pearson_scores.append(r)
        except:
            pass
    results['pearson'] = np.mean(pearson_scores) if pearson_scores else 0

    return results


def train_model(model, train_loader, test_loader, device, args):
    """Train the CheMoE_PDG model."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float('inf')
    best_epoch = 0
    history = defaultdict(list)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_mse = 0
        train_aux = 0
        train_samples = 0

        for diseased, treated, mol_embed, cell_idx in train_loader:
            diseased = diseased.to(device)
            treated = treated.to(device)
            mol_embed = mol_embed.to(device)

            # Handle cell_idx (could be int or tensor)
            if isinstance(cell_idx, int):
                cell_idx = torch.tensor([cell_idx] * diseased.size(0), device=device)
            else:
                cell_idx = cell_idx.to(device)

            if diseased.shape[0] == 1:
                continue

            optimizer.zero_grad()

            # Forward pass
            pred_treated = model(diseased, mol_embed, cell_idx)

            # MSE loss on treated expression
            mse_loss = torch.nn.functional.mse_loss(pred_treated, treated)

            # Load balancing auxiliary loss
            aux_loss = model.compute_load_balance_loss(weight=args.aux_loss_weight)

            # Total loss
            total_loss = mse_loss + aux_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item() * diseased.shape[0]
            train_mse += mse_loss.item() * diseased.shape[0]
            train_aux += aux_loss.item() * diseased.shape[0]
            train_samples += diseased.shape[0]

        train_loss /= train_samples
        train_mse /= train_samples
        train_aux /= train_samples

        # Evaluate
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            test_metrics = evaluate_model(model, test_loader, device)

            # Get expert usage stats
            _ = model(diseased.to(device), mol_embed.to(device), cell_idx)  # Forward pass to get stats
            expert_stats = model.get_expert_usage_stats()

            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, Aux: {train_aux:.6f}) | "
                  f"R2 Top-20: {test_metrics['r2_top20']:.4f} | "
                  f"R2 Top-40: {test_metrics['r2_top40']:.4f} | "
                  f"Pearson: {test_metrics['pearson']:.4f}")

            if expert_stats:
                usage_str = ", ".join([f"E{i}:{u:.2f}" for i, u in enumerate(expert_stats['usage_per_expert'])])
                print(f"        Expert usage: {usage_str}")

            # Log to WandB
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_mse': train_mse,
                'train_aux_loss': train_aux,
                'test_r2_top20': test_metrics['r2_top20'],
                'test_r2_top40': test_metrics['r2_top40'],
                'test_r2_top80': test_metrics['r2_top80'],
                'test_pearson': test_metrics['pearson'],
            })

            if expert_stats:
                for i, usage in enumerate(expert_stats['usage_per_expert']):
                    wandb.log({f'expert_{i}_usage': usage})

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            for k, v in test_metrics.items():
                history[k].append(v)

            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                torch.save(model.state_dict(), args.output_path / 'best_model.pt')

    print(f"\nBest epoch: {best_epoch} with loss: {best_loss:.4f}")
    return history


def main(args):
    """Main training function."""
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    args.output_path = Path(args.output_dir) / f"chemoe_{args.mol_encoder}_{args.cell_line}_fold{args.fold}"
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    project_name = "CheMoE_PDG_AE_DE"
    run_name = f"chemoe_{args.mol_encoder}_{args.cell_line}_fold{args.fold}"

    wandb.init(
        project=project_name,
        name=run_name,
        config=vars(args),
        reinit=True
    )

    # Load data
    treated_df, diseased_df, gene_cols, train_idx, test_idx = load_data(args.cell_line, args.fold)
    smiles_to_embedding, default_embedding, emb_dim = load_molecule_embeddings(args.mol_encoder)

    n_genes = len(gene_cols)
    print(f"Number of genes: {n_genes}")
    print(f"Molecular embedding dimension: {emb_dim}")

    # Create datasets
    train_treated = treated_df.iloc[train_idx].reset_index(drop=True)
    train_diseased = diseased_df.iloc[train_idx].reset_index(drop=True)
    test_treated = treated_df.iloc[test_idx].reset_index(drop=True)
    test_diseased = diseased_df.iloc[test_idx].reset_index(drop=True)

    train_dataset = PDGrapherDataset(
        train_treated, train_diseased, gene_cols,
        smiles_to_embedding, default_embedding, args.cell_line
    )
    test_dataset = PDGrapherDataset(
        test_treated, test_diseased, gene_cols,
        smiles_to_embedding, default_embedding, args.cell_line
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4)

    # Create model
    model = CheMoE_PDG(
        n_genes=n_genes,
        embed_dim=args.embed_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        mol_encoder_type=args.mol_encoder,
        num_cell_lines=10,  # PDGrapher has 10 cell lines
        dropout=args.dropout,
    )
    model = model.to(device)

    print(f"\nModel: CheMoE_PDG")
    print(f"  mol_encoder: {args.mol_encoder}")
    print(f"  num_experts: {args.num_experts}")
    print(f"  top_k: {args.top_k}")
    print(f"  embed_dim: {args.embed_dim}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    history = train_model(model, train_loader, test_loader, device, args)

    # Save final model and history
    torch.save(model.state_dict(), args.output_path / 'final_model.pt')
    with open(args.output_path / 'history.pkl', 'wb') as f:
        pickle.dump(dict(history), f)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load(args.output_path / 'best_model.pt', weights_only=True))
    final_metrics = evaluate_model(model, test_loader, device)
    print(f"R2 Top-20: {final_metrics['r2_top20']:.4f}")
    print(f"R2 Top-40: {final_metrics['r2_top40']:.4f}")
    print(f"R2 Top-80: {final_metrics['r2_top80']:.4f}")
    print(f"Pearson:   {final_metrics['pearson']:.4f}")

    # Save predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for diseased, treated, mol_embed, cell_idx in test_loader:
            diseased = diseased.to(device)
            mol_embed = mol_embed.to(device)
            if isinstance(cell_idx, int):
                cell_idx = torch.tensor([cell_idx] * diseased.size(0), device=device)
            else:
                cell_idx = cell_idx.to(device)
            pred = model(diseased, mol_embed, cell_idx)
            all_preds.append(pred.cpu().numpy())

    predictions = np.vstack(all_preds)
    np.savez(
        args.output_path / 'predictions.npz',
        predictions=predictions,
        treated_test=test_treated[gene_cols].values,
        diseased_test=test_diseased[gene_cols].values,
    )

    print(f"\nResults saved to: {args.output_path}")

    # Log final metrics and finish WandB
    wandb.log({
        'final_r2_top20': final_metrics['r2_top20'],
        'final_r2_top40': final_metrics['r2_top40'],
        'final_r2_top80': final_metrics['r2_top80'],
        'final_pearson': final_metrics['pearson'],
    })

    # Save to shared CSV results file
    model_name = 'CheMoE'
    save_results_to_csv(model_name, args.cell_line, args.fold, final_metrics)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CheMoE on PDGrapher data')

    # Data args
    parser.add_argument('--cell_line', type=str, required=True,
                        help='Cell line to train on (A375, A549, BT20, HELA, HT29, MCF7, MDAMB231, PC3, VCAP)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Cross-validation fold (0-4)')

    # Model args
    parser.add_argument('--mol_encoder', type=str, default='kpgt', choices=['kpgt', 'morgan'],
                        help='Molecular encoder type (kpgt=2304-dim, morgan=1024-dim)')
    parser.add_argument('--num_experts', type=int, default=4,
                        help='Number of experts (default: 4)')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Number of experts to select per sample (default: 2)')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embedding dimension for each modality (default: 128)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')

    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay (default: 1e-3)')
    parser.add_argument('--aux_loss_weight', type=float, default=0.01,
                        help='Load balancing auxiliary loss weight (default: 0.01)')

    # Other args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                        help='Output directory (default: trained_models)')

    args = parser.parse_args()
    main(args)
