#!/usr/bin/env python
"""
Training script for scGen on PDGrapher data.

scGen uses VAE with vector arithmetic in latent space for perturbation prediction.

Usage:
    python train_scgen_pdg.py --cell_line A549 --fold 0 --gpu 0

Note: Use scgen_env conda environment:
    /home/joshua/miniforge3/envs/scgen_env/bin/python train_scgen_pdg.py ...
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import wandb
import warnings
warnings.filterwarnings('ignore')

# scGen dependencies
import anndata as ad
import scanpy as sc
import scgen

# Default paths (same as other training scripts)
DATA_PICKLE = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl"
DISEASED_PICKLE = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl"
SPLITS_BASE = "/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical"


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


def compute_de_metrics(true_treated, pred_treated, diseased, gene_cols=None):
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

    # Global metrics (all samples flattened)
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


def train_scgen(treated_train, diseased_train, gene_names, args):
    """
    Train scGen model.

    Args:
        treated_train: Treated expression array (n_samples, n_genes)
        diseased_train: Diseased expression array (n_samples, n_genes)
        gene_names: List of gene names
        args: Training arguments

    Returns:
        Trained scGen model
    """
    n_samples = len(treated_train)

    # Stack control and treated as separate cells
    X_train = np.vstack([diseased_train, treated_train])

    adata_train = ad.AnnData(X=X_train.astype(np.float32))
    adata_train.var_names = gene_names

    # Add metadata
    adata_train.obs['condition'] = ['control'] * n_samples + ['treated'] * n_samples
    adata_train.obs['cell_type'] = 'cell'

    # Preprocess (scGen's expected normalization)
    sc.pp.normalize_total(adata_train, target_sum=1e4)
    sc.pp.log1p(adata_train)

    # Setup and train
    print("Setting up scGen model...")
    scgen.SCGEN.setup_anndata(adata_train, batch_key='condition', labels_key='cell_type')

    model = scgen.SCGEN(adata_train, n_latent=args.n_latent)

    print(f"Training scGen for {args.n_epochs} epochs...")
    model.train(
        max_epochs=args.n_epochs,
        batch_size=args.batch_size,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
    )

    return model, adata_train


def predict_scgen(model, diseased_test, gene_names):
    """
    Generate predictions using trained scGen model.

    Args:
        model: Trained scGen model
        diseased_test: Diseased expression array for test samples
        gene_names: List of gene names

    Returns:
        Predicted treated expression array
    """
    # Create test AnnData
    adata_test = ad.AnnData(X=diseased_test.astype(np.float32))
    adata_test.var_names = gene_names
    adata_test.obs['condition'] = 'control'
    adata_test.obs['cell_type'] = 'cell'

    # Apply same normalization as training
    sc.pp.normalize_total(adata_test, target_sum=1e4)
    sc.pp.log1p(adata_test)

    # Predict
    result = model.predict(
        ctrl_key='control',
        stim_key='treated',
        adata_to_predict=adata_test
    )

    # predict returns (corrected_adata, delta) tuple
    if isinstance(result, tuple):
        pred_adata, delta = result
    else:
        pred_adata = result

    return pred_adata.X


def main(args):
    """Main training function."""
    print(f"\n{'='*60}")
    print(f"scGen Training - {args.cell_line} Fold {args.fold}")
    print(f"{'='*60}\n")

    # Set GPU visibility (scGen uses scvi which can use GPU)
    if args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    args.output_path = Path(args.output_dir) / f"scgen_{args.cell_line}_fold{args.fold}"
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    wandb.init(
        project="scGen_AE_DE",
        name=f"scgen_{args.cell_line}_fold{args.fold}",
        config=vars(args),
        reinit=True
    )

    # Load data
    treated_df, diseased_df, gene_cols, train_idx, test_idx = load_data(args.cell_line, args.fold)

    # Split into train/test
    treated_train = treated_df.iloc[train_idx][gene_cols].values
    diseased_train = diseased_df.iloc[train_idx][gene_cols].values
    treated_test = treated_df.iloc[test_idx][gene_cols].values
    diseased_test = diseased_df.iloc[test_idx][gene_cols].values

    print(f"Train samples: {len(treated_train)}")
    print(f"Test samples: {len(treated_test)}")
    print(f"Genes: {len(gene_cols)}")

    # Train model
    print("\n--- Training scGen ---")
    model, adata_train = train_scgen(treated_train, diseased_train, gene_cols, args)

    # Generate predictions
    print("\n--- Generating Predictions ---")
    pred_treated = predict_scgen(model, diseased_test, gene_cols)

    # Normalize test data for fair comparison (same preprocessing as scGen uses)
    treated_test_norm = np.log1p(treated_test * 1e4 / (treated_test.sum(axis=1, keepdims=True) + 1e-8))
    diseased_test_norm = np.log1p(diseased_test * 1e4 / (diseased_test.sum(axis=1, keepdims=True) + 1e-8))

    # Compute metrics
    print("\n--- Evaluation ---")
    metrics = compute_de_metrics(treated_test_norm, pred_treated, diseased_test_norm, gene_cols)

    print(f"\nResults:")
    print(f"  R² Top-20:   {metrics['r2_top20']:.4f}")
    print(f"  R² Top-40:   {metrics['r2_top40']:.4f}")
    print(f"  R² Top-80:   {metrics['r2_top80']:.4f}")
    print(f"  Pearson DE:  {metrics['pearson_de']:.4f}")
    print(f"  Spearman DE: {metrics['spearman_de']:.4f}")
    print(f"  Global R²:   {metrics['global_r2']:.4f}")

    # Log to WandB
    wandb.log({
        'final_r2_top20': metrics['r2_top20'],
        'final_r2_top40': metrics['r2_top40'],
        'final_r2_top80': metrics['r2_top80'],
        'final_pearson_de': metrics['pearson_de'],
        'final_spearman_de': metrics['spearman_de'],
        'final_global_r2': metrics['global_r2'],
    })

    # Save predictions and metrics
    np.savez(
        args.output_path / 'predictions.npz',
        predictions=pred_treated,
        treated_test=treated_test_norm,
        diseased_test=diseased_test_norm,
    )

    with open(args.output_path / 'metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    # Save model reference path (scGen saves internally)
    model_dir = args.output_path / 'scgen_model'
    model.save(str(model_dir), overwrite=True)

    print(f"\nResults saved to: {args.output_path}")

    wandb.finish()

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train scGen on PDGrapher data')

    # Data args
    parser.add_argument('--cell_line', type=str, default='A549',
                        help='Cell line to train on')
    parser.add_argument('--fold', type=int, default=0,
                        help='Cross-validation fold (0-indexed)')

    # Model args
    parser.add_argument('--n_latent', type=int, default=100,
                        help='Latent dimension')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--early_stopping', action='store_true', default=True,
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')

    # Training args
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use (-1 for CPU)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str,
                        default='/raid/home/joshua/projects/PDGrapher_Baseline_Models/output/scgen',
                        help='Output directory')

    args = parser.parse_args()
    main(args)
