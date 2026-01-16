#!/usr/bin/env python
"""
Training script for TranSiGen and TranSiGen MoE on PDGrapher data.

Usage:
    python train_transigen_pdg.py --model transigen --cell_line A549 --fold 0 --gpu 0
    python train_transigen_pdg.py --model transigen_moe --moe_style balanced --cell_line A549 --fold 0 --gpu 0
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
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import wandb
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.transigen import TranSiGen_PDG
from models.transigen_moe import TranSiGen_MoE_PDG


# Default paths
DATA_PICKLE = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl"
DISEASED_PICKLE = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl"
SPLITS_BASE = "/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical"
MOLECULE_EMBEDDINGS = "/raid/home/joshua/projects/cbio_032024/zenodo_TranSiGen/TranSiGen/data/LINCS2020/KPGT_emb2304.pickle"
DRUG_SMILES_CSV = "/raid/home/joshua/projects/MultiDCP_pdg/data/all_drugs_pdg.csv"


class PDGrapherDataset(Dataset):
    """PyTorch Dataset for PDGrapher data."""

    def __init__(self, treated_df, diseased_df, gene_cols, smiles_to_embedding, default_embedding):
        """
        Args:
            treated_df: DataFrame with treated expression
            diseased_df: DataFrame with diseased expression
            gene_cols: List of gene column names
            smiles_to_embedding: Dict mapping SMILES to molecular embeddings
            default_embedding: Default embedding for unknown molecules
        """
        self.treated = treated_df[gene_cols].values.astype(np.float32)
        self.diseased = diseased_df[gene_cols].values.astype(np.float32)
        self.smiles = treated_df['smiles'].values if 'smiles' in treated_df.columns else None
        self.smiles_to_embedding = smiles_to_embedding
        self.default_embedding = default_embedding
        self.gene_cols = gene_cols

    def __len__(self):
        return len(self.treated)

    def __getitem__(self, idx):
        x1 = self.diseased[idx]  # diseased (control)
        x2 = self.treated[idx]   # treated

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
            idx
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


def load_molecule_embeddings():
    """Load molecular embeddings."""
    smiles_to_embedding = {}

    if os.path.exists(MOLECULE_EMBEDDINGS):
        with open(MOLECULE_EMBEDDINGS, 'rb') as f:
            smiles_to_embedding = pickle.load(f)
        print(f"Loaded {len(smiles_to_embedding)} molecular embeddings")

    # Determine embedding dimension
    if smiles_to_embedding:
        sample_emb = next(iter(smiles_to_embedding.values()))
        emb_dim = len(sample_emb) if hasattr(sample_emb, '__len__') else 2304
    else:
        emb_dim = 2304

    default_embedding = np.zeros(emb_dim, dtype=np.float32)

    return smiles_to_embedding, default_embedding, emb_dim


def evaluate_model(model, dataloader, device):
    """Evaluate model on test data."""
    model.eval()

    all_x1 = []
    all_x2 = []
    all_x2_pred = []

    with torch.no_grad():
        for x1, x2, mol_embed, _ in dataloader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            mol_embed = mol_embed.to(device)

            outputs = model(x1, mol_embed)
            x2_pred = outputs[3]

            all_x1.append(x1.cpu().numpy())
            all_x2.append(x2.cpu().numpy())
            all_x2_pred.append(x2_pred.cpu().numpy())

    x1 = np.vstack(all_x1)
    x2 = np.vstack(all_x2)
    x2_pred = np.vstack(all_x2_pred)

    # Compute metrics on differential expression
    true_de = x2 - x1
    pred_de = x2_pred - x1

    results = {}

    # Top-k R² scores
    for k in [20, 40, 80]:
        r2_scores = []
        for i in range(len(x2)):
            de_mag = np.abs(true_de[i])
            top_k_idx = np.argsort(de_mag)[-k:]
            try:
                r2 = r2_score(true_de[i, top_k_idx], pred_de[i, top_k_idx])
                r2_scores.append(r2)
            except:
                pass
        results[f'r2_top{k}'] = np.mean(r2_scores) if r2_scores else 0

    # Pearson correlation
    pearson_scores = []
    for i in range(len(x2)):
        try:
            r, _ = pearsonr(true_de[i], pred_de[i])
            if not np.isnan(r):
                pearson_scores.append(r)
        except:
            pass
    results['pearson'] = np.mean(pearson_scores) if pearson_scores else 0

    return results


def train_model(model, train_loader, test_loader, device, args):
    """Train the model."""
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float('inf')
    best_epoch = 0
    history = defaultdict(list)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_samples = 0

        for x1, x2, mol_embed, _ in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            mol_embed = mol_embed.to(device)

            if x1.shape[0] == 1:
                continue

            optimizer.zero_grad()

            # Forward pass
            x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred = model(x1, mol_embed)
            z2, mu2, logvar2 = model.encode_x2(x2)
            x2_rec = model.decode_x2(z2)

            # Compute loss
            loss_tuple = model.loss(x1, x1_rec, mu1, logvar1, x2, x2_rec, mu2, logvar2,
                                    x2_pred, mu_pred, logvar_pred)
            loss = loss_tuple[0]

            # Add MoE auxiliary loss if applicable
            if hasattr(model, 'get_auxiliary_loss'):
                aux_loss = model.get_auxiliary_loss()
                if aux_loss.device != loss.device:
                    aux_loss = aux_loss.to(loss.device)
                loss = loss + args.aux_loss_coef * aux_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_samples += x1.shape[0]

        train_loss /= train_samples

        # Evaluate
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            test_metrics = evaluate_model(model, test_loader, device)

            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"R² Top-20: {test_metrics['r2_top20']:.4f} | "
                  f"R² Top-40: {test_metrics['r2_top40']:.4f} | "
                  f"Pearson: {test_metrics['pearson']:.4f}")

            # Log to WandB
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'test_r2_top20': test_metrics['r2_top20'],
                'test_r2_top40': test_metrics['r2_top40'],
                'test_r2_top80': test_metrics['r2_top80'],
                'test_pearson': test_metrics['pearson'],
            })

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            for k, v in test_metrics.items():
                history[k].append(v)

            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                torch.save(model, args.output_path / 'best_model.pt')

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

    # Create output directory (include moe_style for MoE models)
    if args.model == 'transigen_moe':
        args.output_path = Path(args.output_dir) / f"transigen_{args.moe_style}_moe_{args.cell_line}_fold{args.fold}"
    else:
        args.output_path = Path(args.output_dir) / f"{args.model}_{args.cell_line}_fold{args.fold}"
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    if args.model == 'transigen_moe':
        project_name = f"TranSiGen_MoE_{args.moe_style.capitalize()}_AE_DE"
        run_name = f"transigen_{args.moe_style}_moe_{args.cell_line}_fold{args.fold}"
    else:
        project_name = "TranSiGen_AE_DE"
        run_name = f"transigen_{args.cell_line}_fold{args.fold}"

    wandb.init(
        project=project_name,
        name=run_name,
        config=vars(args),
        reinit=True
    )

    # Load data
    treated_df, diseased_df, gene_cols, train_idx, test_idx = load_data(args.cell_line, args.fold)
    smiles_to_embedding, default_embedding, emb_dim = load_molecule_embeddings()

    n_genes = len(gene_cols)
    print(f"Number of genes: {n_genes}")
    print(f"Embedding dimension: {emb_dim}")

    # Create datasets
    train_treated = treated_df.iloc[train_idx].reset_index(drop=True)
    train_diseased = diseased_df.iloc[train_idx].reset_index(drop=True)
    test_treated = treated_df.iloc[test_idx].reset_index(drop=True)
    test_diseased = diseased_df.iloc[test_idx].reset_index(drop=True)

    train_dataset = PDGrapherDataset(train_treated, train_diseased, gene_cols,
                                      smiles_to_embedding, default_embedding)
    test_dataset = PDGrapherDataset(test_treated, test_diseased, gene_cols,
                                     smiles_to_embedding, default_embedding)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4)

    # Create model
    model_kwargs = {
        'n_genes': n_genes,
        'n_latent': args.n_latent,
        'n_en_hidden': [1200, 600],
        'n_de_hidden': [600, 1200],
        'features_dim': emb_dim,
        'features_embed_dim': [400],
        'beta': args.beta,
        'dropout': args.dropout,
        'device': device,
        'path_model': str(args.output_path) + '/',
    }

    if args.model == 'transigen':
        model = TranSiGen_PDG(**model_kwargs)
    elif args.model == 'transigen_moe':
        model = TranSiGen_MoE_PDG(
            moe_style=args.moe_style,
            num_experts=args.num_experts,
            **model_kwargs
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    print(f"\nModel: {args.model}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    history = train_model(model, train_loader, test_loader, device, args)

    # Save final model and history
    torch.save(model, args.output_path / 'final_model.pt')
    with open(args.output_path / 'history.pkl', 'wb') as f:
        pickle.dump(dict(history), f)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    model = torch.load(args.output_path / 'best_model.pt', map_location=device, weights_only=False)
    final_metrics = evaluate_model(model, test_loader, device)
    print(f"R² Top-20: {final_metrics['r2_top20']:.4f}")
    print(f"R² Top-40: {final_metrics['r2_top40']:.4f}")
    print(f"R² Top-80: {final_metrics['r2_top80']:.4f}")
    print(f"Pearson:   {final_metrics['pearson']:.4f}")

    # Save predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x1, x2, mol_embed, _ in test_loader:
            x1 = x1.to(device)
            mol_embed = mol_embed.to(device)
            outputs = model(x1, mol_embed)
            all_preds.append(outputs[3].cpu().numpy())

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
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TranSiGen on PDGrapher data')

    # Model args
    parser.add_argument('--model', type=str, default='transigen',
                        choices=['transigen', 'transigen_moe'],
                        help='Model type')
    parser.add_argument('--moe_style', type=str, default='balanced',
                        choices=['sparse', 'balanced'],
                        help='MoE style (for transigen_moe)')
    parser.add_argument('--num_experts', type=int, default=4,
                        help='Number of experts (for transigen_moe)')

    # Data args
    parser.add_argument('--cell_line', type=str, default='A549',
                        help='Cell line to train on')
    parser.add_argument('--fold', type=int, default=0,
                        help='Cross-validation fold')

    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='KLD loss weight')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--aux_loss_coef', type=float, default=0.1,
                        help='Auxiliary loss coefficient (for MoE)')

    # Architecture args
    parser.add_argument('--n_latent', type=int, default=100,
                        help='Latent dimension')

    # Other args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                        help='Output directory')

    args = parser.parse_args()
    main(args)
