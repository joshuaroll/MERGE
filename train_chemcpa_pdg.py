#!/usr/bin/env python
"""
Training script for ChemCPA on PDGrapher data.

ChemCPA uses a compositional perturbation autoencoder that models drug perturbations
as additive effects in latent space.

Usage:
    python train_chemcpa_pdg.py --cell_line A549 --fold 0 --gpu 0
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import wandb
import warnings
warnings.filterwarnings('ignore')

# Default paths (same as other training scripts)
DATA_PICKLE = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl"
DISEASED_PICKLE = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl"
SPLITS_BASE = "/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical"


# ============================================================================
# ChemCPA Model Components (from wrapper)
# ============================================================================

class SimpleChemCPADataset(Dataset):
    """Dataset for ChemCPA-style training."""
    def __init__(self, diseased: np.ndarray, treated: np.ndarray,
                 drug_indices: np.ndarray, dosages: np.ndarray):
        self.diseased = torch.FloatTensor(diseased)
        self.treated = torch.FloatTensor(treated)
        self.drug_indices = torch.LongTensor(drug_indices)
        self.dosages = torch.FloatTensor(dosages)

    def __len__(self):
        return len(self.diseased)

    def __getitem__(self, idx):
        return (
            self.diseased[idx],
            self.treated[idx],
            self.drug_indices[idx],
            self.dosages[idx],
        )


class DrugEncoder(nn.Module):
    """Drug embedding + MLP encoder."""
    def __init__(self, num_drugs: int, drug_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.drug_embedding = nn.Embedding(num_drugs, drug_dim)
        self.dose_encoder = nn.Linear(1, 32)
        self.mlp = nn.Sequential(
            nn.Linear(drug_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, drug_idx, dosage):
        drug_emb = self.drug_embedding(drug_idx)
        dose_emb = self.dose_encoder(dosage.unsqueeze(-1))
        combined = torch.cat([drug_emb, dose_emb], dim=-1)
        return self.mlp(combined)


class GeneEncoder(nn.Module):
    """Encoder for gene expression."""
    def __init__(self, num_genes: int, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class GeneDecoder(nn.Module):
    """Decoder for gene expression."""
    def __init__(self, latent_dim: int, num_genes: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_genes),
        )

    def forward(self, x):
        return self.decoder(x)


class SimpleChemCPA(nn.Module):
    """
    Simplified ChemCPA model.

    Architecture:
    1. Encode diseased expression to latent space
    2. Encode drug + dose to perturbation embedding
    3. Add perturbation to latent
    4. Decode to predicted treated expression
    """
    def __init__(self, num_genes: int, num_drugs: int,
                 latent_dim: int = 256, drug_dim: int = 128):
        super().__init__()

        self.gene_encoder = GeneEncoder(num_genes, latent_dim)
        self.drug_encoder = DrugEncoder(num_drugs, drug_dim, latent_dim)

        # Combine latent + drug perturbation
        self.combiner = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.gene_decoder = GeneDecoder(latent_dim, num_genes)

    def forward(self, diseased, drug_idx, dosage):
        # Encode diseased expression
        latent = self.gene_encoder(diseased)

        # Encode drug perturbation
        drug_latent = self.drug_encoder(drug_idx, dosage)

        # Combine
        combined = torch.cat([latent, drug_latent], dim=-1)
        perturbed_latent = self.combiner(combined)

        # Decode
        pred_treated = self.gene_decoder(perturbed_latent)

        return pred_treated


# ============================================================================
# Data Loading Functions
# ============================================================================

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

    # Load PDGrapher splits
    splits_path = os.path.join(SPLITS_BASE, cell_line, "random", "5fold", "splits.pt")
    splits = torch.load(splits_path, weights_only=False)

    # PDGrapher uses 1-indexed folds
    fold_key = fold + 1
    fold_splits = splits[fold_key]

    # Get backward indices
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


def prepare_drug_indices(df, smiles_to_idx=None):
    """Create drug indices from SMILES strings."""
    if 'smiles' not in df.columns:
        return np.zeros(len(df), dtype=np.int64), {'dummy': 0}

    smiles_list = df['smiles'].values
    unique_smiles = list(set(smiles_list))

    if smiles_to_idx is None:
        smiles_to_idx = {s: i for i, s in enumerate(unique_smiles)}

    drug_indices = np.array([smiles_to_idx.get(s, 0) for s in smiles_list], dtype=np.int64)

    return drug_indices, smiles_to_idx


def prepare_doses(df):
    """Extract and normalize doses."""
    if 'dose' not in df.columns:
        return np.ones(len(df), dtype=np.float32)

    doses = df['dose'].values

    # Convert to numeric, handling non-numeric values
    numeric_doses = []
    for d in doses:
        try:
            numeric_doses.append(float(d))
        except (ValueError, TypeError):
            numeric_doses.append(1.0)

    doses = np.array(numeric_doses, dtype=np.float32)

    # Normalize doses to [0, 1] range
    if doses.max() > doses.min():
        doses = (doses - doses.min()) / (doses.max() - doses.min())

    return doses


def compute_de_metrics(true_treated, pred_treated, diseased):
    """
    Compute evaluation metrics on differential expression.

    Args:
        true_treated: True treated expression (n_samples, n_genes)
        pred_treated: Predicted treated expression (n_samples, n_genes)
        diseased: Diseased/control expression (n_samples, n_genes)

    Returns:
        Dictionary of metrics
    """
    # Compute differential expression
    true_de = true_treated - diseased
    pred_de = pred_treated - diseased

    results = {}

    # Per-sample top-k R² scores
    for k in [20, 40, 80]:
        r2_scores = []
        for i in range(len(true_de)):
            de_mag = np.abs(true_de[i])
            top_k_idx = np.argsort(de_mag)[-k:]
            try:
                r2 = r2_score(true_de[i, top_k_idx], pred_de[i, top_k_idx])
                if not np.isnan(r2) and not np.isinf(r2):
                    r2_scores.append(r2)
            except:
                pass
        results[f'r2_top{k}'] = np.mean(r2_scores) if r2_scores else 0.0

    # Per-sample Pearson correlation on DE
    pearson_scores = []
    for i in range(len(true_de)):
        try:
            r, _ = pearsonr(true_de[i], pred_de[i])
            if not np.isnan(r):
                pearson_scores.append(r)
        except:
            pass
    results['pearson_de'] = np.mean(pearson_scores) if pearson_scores else 0.0

    # Per-sample Spearman correlation on DE
    spearman_scores = []
    for i in range(len(true_de)):
        try:
            r, _ = spearmanr(true_de[i], pred_de[i])
            if not np.isnan(r):
                spearman_scores.append(r)
        except:
            pass
    results['spearman_de'] = np.mean(spearman_scores) if spearman_scores else 0.0

    # Global metrics
    true_de_flat = true_de.flatten()
    pred_de_flat = pred_de.flatten()

    try:
        results['global_r2'] = r2_score(true_de_flat, pred_de_flat)
    except:
        results['global_r2'] = 0.0

    try:
        results['global_pearson'], _ = pearsonr(true_de_flat, pred_de_flat)
    except:
        results['global_pearson'] = 0.0

    return results


def evaluate_model(model, dataloader, device):
    """Evaluate model on test data."""
    model.eval()

    all_diseased = []
    all_treated = []
    all_pred = []

    with torch.no_grad():
        for diseased, treated, drug_idx, dosage in dataloader:
            diseased = diseased.to(device)
            drug_idx = drug_idx.to(device)
            dosage = dosage.to(device)

            pred = model(diseased, drug_idx, dosage)

            all_diseased.append(diseased.cpu().numpy())
            all_treated.append(treated.numpy())
            all_pred.append(pred.cpu().numpy())

    diseased = np.vstack(all_diseased)
    treated = np.vstack(all_treated)
    pred = np.vstack(all_pred)

    return compute_de_metrics(treated, pred, diseased)


def train_model(model, train_loader, test_loader, device, args):
    """Train the ChemCPA model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        train_samples = 0

        for diseased, treated, drug_idx, dosage in train_loader:
            diseased = diseased.to(device)
            treated = treated.to(device)
            drug_idx = drug_idx.to(device)
            dosage = dosage.to(device)

            if diseased.shape[0] == 1:
                continue

            optimizer.zero_grad()
            pred = model(diseased, drug_idx, dosage)
            loss = criterion(pred, treated)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * diseased.shape[0]
            train_samples += diseased.shape[0]

        train_loss /= train_samples

        # Evaluate periodically
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            test_metrics = evaluate_model(model, test_loader, device)

            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                  f"R² Top-20: {test_metrics['r2_top20']:.4f} | "
                  f"R² Top-40: {test_metrics['r2_top40']:.4f} | "
                  f"Pearson: {test_metrics['pearson_de']:.4f}")

            # Log to WandB
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'test_r2_top20': test_metrics['r2_top20'],
                'test_r2_top40': test_metrics['r2_top40'],
                'test_r2_top80': test_metrics['r2_top80'],
                'test_pearson_de': test_metrics['pearson_de'],
            })

            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                best_epoch = epoch
                torch.save(model.state_dict(), args.output_path / 'best_model.pt')

    print(f"\nBest epoch: {best_epoch} with loss: {best_loss:.4f}")
    return best_epoch


def main(args):
    """Main training function."""
    print(f"\n{'='*60}")
    print(f"ChemCPA Training - {args.cell_line} Fold {args.fold}")
    print(f"{'='*60}\n")

    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    args.output_path = Path(args.output_dir) / f"chemcpa_{args.cell_line}_fold{args.fold}"
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    wandb.init(
        project="ChemCPA_AE_DE",
        name=f"chemcpa_{args.cell_line}_fold{args.fold}",
        config=vars(args),
        reinit=True
    )

    # Load data
    treated_df, diseased_df, gene_cols, train_idx, test_idx = load_data(args.cell_line, args.fold)

    n_genes = len(gene_cols)
    print(f"Number of genes: {n_genes}")

    # Create datasets
    train_treated = treated_df.iloc[train_idx].reset_index(drop=True)
    train_diseased = diseased_df.iloc[train_idx].reset_index(drop=True)
    test_treated = treated_df.iloc[test_idx].reset_index(drop=True)
    test_diseased = diseased_df.iloc[test_idx].reset_index(drop=True)

    # Prepare drug indices (shared mapping between train and test)
    train_drug_idx, smiles_to_idx = prepare_drug_indices(train_treated)
    test_drug_idx, _ = prepare_drug_indices(test_treated, smiles_to_idx)
    num_drugs = len(smiles_to_idx)
    print(f"Number of unique drugs: {num_drugs}")

    # Prepare doses
    train_doses = prepare_doses(train_treated)
    test_doses = prepare_doses(test_treated)

    # Create datasets
    train_dataset = SimpleChemCPADataset(
        train_diseased[gene_cols].values,
        train_treated[gene_cols].values,
        train_drug_idx,
        train_doses
    )
    test_dataset = SimpleChemCPADataset(
        test_diseased[gene_cols].values,
        test_treated[gene_cols].values,
        test_drug_idx,
        test_doses
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4)

    # Create model
    model = SimpleChemCPA(
        num_genes=n_genes,
        num_drugs=num_drugs,
        latent_dim=args.latent_dim,
        drug_dim=args.drug_dim,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    best_epoch = train_model(model, train_loader, test_loader, device, args)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load(args.output_path / 'best_model.pt'))
    final_metrics = evaluate_model(model, test_loader, device)

    print(f"R² Top-20:   {final_metrics['r2_top20']:.4f}")
    print(f"R² Top-40:   {final_metrics['r2_top40']:.4f}")
    print(f"R² Top-80:   {final_metrics['r2_top80']:.4f}")
    print(f"Pearson DE:  {final_metrics['pearson_de']:.4f}")
    print(f"Spearman DE: {final_metrics['spearman_de']:.4f}")
    print(f"Global R²:   {final_metrics['global_r2']:.4f}")

    # Log final metrics
    wandb.log({
        'final_r2_top20': final_metrics['r2_top20'],
        'final_r2_top40': final_metrics['r2_top40'],
        'final_r2_top80': final_metrics['r2_top80'],
        'final_pearson_de': final_metrics['pearson_de'],
        'final_spearman_de': final_metrics['spearman_de'],
        'final_global_r2': final_metrics['global_r2'],
    })

    # Save predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for diseased, _, drug_idx, dosage in test_loader:
            diseased = diseased.to(device)
            drug_idx = drug_idx.to(device)
            dosage = dosage.to(device)
            pred = model(diseased, drug_idx, dosage)
            all_preds.append(pred.cpu().numpy())

    predictions = np.vstack(all_preds)
    np.savez(
        args.output_path / 'predictions.npz',
        predictions=predictions,
        treated_test=test_treated[gene_cols].values,
        diseased_test=test_diseased[gene_cols].values,
    )

    # Save full model
    torch.save(model.state_dict(), args.output_path / 'final_model.pt')

    with open(args.output_path / 'metrics.pkl', 'wb') as f:
        pickle.dump(final_metrics, f)

    print(f"\nResults saved to: {args.output_path}")

    wandb.finish()

    return final_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ChemCPA on PDGrapher data')

    # Data args
    parser.add_argument('--cell_line', type=str, default='A549',
                        help='Cell line to train on')
    parser.add_argument('--fold', type=int, default=0,
                        help='Cross-validation fold (0-indexed)')

    # Model args
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--drug_dim', type=int, default=128,
                        help='Drug embedding dimension')

    # Training args
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str,
                        default='/raid/home/joshua/projects/PDGrapher_Baseline_Models/output/chemcpa',
                        help='Output directory')

    args = parser.parse_args()
    main(args)
