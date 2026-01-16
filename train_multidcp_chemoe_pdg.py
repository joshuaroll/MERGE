#!/usr/bin/env python3
"""
MultiDCP-CheMoE training script with Differential Expression (DE) metrics.

This integrates CheMoE's Mixture-of-Experts architecture with MultiDCP's
multimodal data fusion approach. Key features:
- 4 expert networks with top-k=2 routing
- Global gating on drug+cell+dose features
- Gene-aware expert processing
- Maintains MultiDCP's DE evaluation pipeline

KEY CHANGE: Computes R² on DIFFERENTIAL EXPRESSION (treated - diseased),
not raw expression. Top-k genes are selected by |true_treated - diseased|.

This matches the biolord evaluation methodology for fair comparison.

Usage:
cd /raid/home/joshua/projects/MultiDCP_CheMoE_pdg/src/
    python multidcp_chemoe_ae_de_pdg.py --test_cell A375 --fold 1 --gpu 0
    python multidcp_chemoe_ae_de_pdg.py --test_cell MCF7 --fold 1 --gpu 0
    python multidcp_chemoe_ae_de_pdg.py --test_cell A549 --fold 1 --gpu 1
    python multidcp_chemoe_ae_de_pdg.py --test_cell HT29 --fold 1 --gpu 2
    python multidcp_chemoe_ae_de_pdg.py --test_cell PC3 --fold 1 --gpu 3
    python multidcp_chemoe_ae_de_pdg.py --test_cell VCAP --fold 1 --gpu 4
    python multidcp_chemoe_ae_de_pdg.py --test_cell HELA --fold 1 --gpu 5
    python multidcp_chemoe_ae_de_pdg.py --test_cell MDAMB231 --fold 1 --gpu 6
    python multidcp_chemoe_ae_de_pdg.py --test_cell BT20 --fold 1 --gpu 7


"""

import os
import sys
import argparse

# =============================================================================
# MUST SET CUDA_VISIBLE_DEVICES BEFORE IMPORTING TORCH
# =============================================================================
def get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None

gpu_arg = get_gpu_arg()
if gpu_arg is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_arg
    print(f"Setting CUDA_VISIBLE_DEVICES={gpu_arg}")
elif 'CUDA_VISIBLE_DEVICES' in os.environ:
    print(f"Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} from environment")
else:
    print("WARNING: No GPU specified. Will use default GPU 0.")

# NOW import torch after setting the environment variable
import pandas as pd
import gc
from datetime import datetime
import torch
import random
from torch import save
import numpy as np
from collections import defaultdict

# Add project root to path for imports
project_root = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, project_root)

# Import from new folder structure
from models.multidcp_chemoe import model as multidcp_chemoe
from utils import datareader_pdg_12122025 as datareader
from utils import metric
from utils.multidcp_ae_pdg_utils import *

import wandb
import pdb
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr, linregress

# check cuda
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Use GPU: %s" % torch.cuda.is_available())

if torch.cuda.is_available():
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    print(f"torch.cuda.current_device() = {torch.cuda.current_device()}")
    print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory at start: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

# Extra variables
GEX_SIZE = 10716
data_save = False


# =============================================================================
# DIFFERENTIAL EXPRESSION DIAGNOSTIC FUNCTIONS
# =============================================================================

def compute_topk_by_differential_expression(true_de, k=80):
    """
    Select top-k differentially expressed genes per sample.
    
    Args:
        true_de: True differential expression (treated - diseased), shape [n_samples, n_genes]
        k: Number of top genes to select
    
    Returns:
        top_indices: Shape [n_samples, k], indices of top-k DEGs per sample
    """
    # Sort by absolute differential expression, descending
    top_indices = np.argsort(np.abs(true_de), axis=1)[:, ::-1][:, :k]
    return top_indices


def compute_global_metrics(true_de, pred_de):
    """Compute metrics over ALL values (flattened across samples and genes)."""
    true_flat = true_de.ravel()
    pred_flat = pred_de.ravel()
    
    mask = np.isfinite(true_flat) & np.isfinite(pred_flat)
    true_masked = true_flat[mask]
    pred_masked = pred_flat[mask]
    
    if len(true_masked) < 2:
        return {'pearson': np.nan, 'spearman': np.nan, 'r2': np.nan, 
                'r2_from_pearson': np.nan, 'rmse': np.nan, 'mae': np.nan, 'n_values': 0}
    
    pearson_r = pearsonr(true_masked, pred_masked).statistic
    spearman_r = spearmanr(true_masked, pred_masked).correlation
    
    # True R² (coefficient of determination)
    ss_res = np.sum((true_masked - pred_masked) ** 2)
    ss_tot = np.sum((true_masked - np.mean(true_masked)) ** 2)
    true_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    rmse = np.sqrt(np.mean((true_masked - pred_masked) ** 2))
    mae = np.mean(np.abs(true_masked - pred_masked))
    
    return {
        'pearson': pearson_r,
        'spearman': spearman_r,
        'r2': true_r2,
        'r2_from_pearson': pearson_r ** 2,
        'rmse': rmse,
        'mae': mae,
        'n_values': len(true_masked)
    }


def compute_persample_metrics(true_de, pred_de, gene_indices=None, show_progress=False):
    """
    Compute metrics per sample on DIFFERENTIAL EXPRESSION, then aggregate.
    
    Args:
        true_de: True differential expression (treated - diseased)
        pred_de: Predicted differential expression (pred_treated - diseased)
        gene_indices: If provided, compute metrics only on these genes per sample
    """
    n_samples = true_de.shape[0]
    
    if show_progress:
        print(f"    Processing {n_samples} samples...")
    
    # If using specific gene indices, extract them
    if gene_indices is not None:
        true_selected = np.array([true_de[i, gene_indices[i]] for i in range(n_samples)])
        pred_selected = np.array([pred_de[i, gene_indices[i]] for i in range(n_samples)])
    else:
        true_selected = true_de
        pred_selected = pred_de
    
    # Compute statistics per sample (vectorized)
    true_mean = np.mean(true_selected, axis=1, keepdims=True)
    pred_mean = np.mean(pred_selected, axis=1, keepdims=True)
    
    true_centered = true_selected - true_mean
    pred_centered = pred_selected - pred_mean
    
    true_std = np.std(true_selected, axis=1)
    pred_std = np.std(pred_selected, axis=1)
    
    var_true = true_std ** 2
    var_pred = pred_std ** 2
    
    n_genes_per_sample = true_selected.shape[1]
    
    # Avoid division by zero
    valid_mask = (true_std > 1e-10) & (pred_std > 1e-10)
    
    # Pearson correlation (vectorized)
    covariance = np.sum(true_centered * pred_centered, axis=1) / n_genes_per_sample
    
    pearson_vals = np.full(n_samples, np.nan)
    pearson_vals[valid_mask] = covariance[valid_mask] / (true_std[valid_mask] * pred_std[valid_mask])
    
    r2_pearson_vals = pearson_vals ** 2
    
    # True R² (coefficient of determination)
    ss_res = np.sum((true_selected - pred_selected) ** 2, axis=1)
    ss_tot = np.sum(true_centered ** 2, axis=1)
    
    r2_true_vals = np.full(n_samples, np.nan)
    valid_ss = ss_tot > 1e-10
    r2_true_vals[valid_ss] = 1 - (ss_res[valid_ss] / ss_tot[valid_ss])
    
    # RMSE per sample
    rmse_vals = np.sqrt(np.mean((true_selected - pred_selected) ** 2, axis=1))
    
    # Spearman correlation (sample for large datasets)
    spearman_vals = np.full(n_samples, np.nan)
    
    sample_indices = range(n_samples) if n_samples <= 5000 else np.random.choice(n_samples, 2000, replace=False)
    for i in sample_indices:
        if valid_mask[i]:
            try:
                rho = spearmanr(true_selected[i], pred_selected[i]).correlation
                if np.isfinite(rho):
                    spearman_vals[i] = rho
            except:
                pass
    
    if show_progress:
        print(f"    Done. Valid samples: {np.sum(valid_mask)}/{n_samples}")
    
    return {
        'pearson_mean': np.nanmean(pearson_vals),
        'pearson_median': np.nanmedian(pearson_vals),
        'pearson_std': np.nanstd(pearson_vals),
        'spearman_mean': np.nanmean(spearman_vals),
        'r2_true_mean': np.nanmean(r2_true_vals),
        'r2_from_pearson_mean': np.nanmean(r2_pearson_vals),
        'rmse_mean': np.nanmean(rmse_vals),
        'var_true_mean': np.nanmean(var_true),
        'var_pred_mean': np.nanmean(var_pred),
        'n_valid': np.sum(valid_mask),
    }


def compute_prediction_bias(true_de, pred_de):
    """Check for systematic prediction biases in differential expression."""
    errors = pred_de - true_de
    true_flat = true_de.ravel()
    error_flat = errors.ravel()
    
    mask = np.isfinite(true_flat) & np.isfinite(error_flat)
    true_masked = true_flat[mask]
    error_masked = error_flat[mask]
    
    if len(true_masked) < 10:
        return {'mean_error': np.nan, 'high_value_bias': np.nan, 'low_value_bias': np.nan}
    
    high_true = true_masked > np.percentile(true_masked, 90)
    low_true = true_masked < np.percentile(true_masked, 10)
    
    high_bias = np.mean(error_masked[high_true]) if np.sum(high_true) > 0 else np.nan
    low_bias = np.mean(error_masked[low_true]) if np.sum(low_true) > 0 else np.nan
    
    return {
        'mean_error': np.mean(error_masked),
        'std_error': np.std(error_masked),
        'high_value_bias': high_bias,
        'low_value_bias': low_bias,
    }


def run_diagnostics_de(true_de, pred_de, split_name, epoch, verbose=True):
    """
    Run comprehensive diagnostics on DIFFERENTIAL EXPRESSION.
    
    Args:
        true_de: True differential expression (treated - diseased)
        pred_de: Predicted differential expression (pred_treated - diseased)
        split_name: Name for logging
        epoch: Current epoch
    """
    results = {}
    
    n_samples = true_de.shape[0]
    n_genes = true_de.shape[1]
    
    if verbose:
        print(f"  Running DE diagnostics on {n_samples} samples x {n_genes} genes...")
        print(f"  True DE range: [{true_de.min():.3f}, {true_de.max():.3f}], mean: {true_de.mean():.3f}")
        print(f"  Pred DE range: [{pred_de.min():.3f}, {pred_de.max():.3f}], mean: {pred_de.mean():.3f}")
    
    # Select top-k DEGs based on TRUE differential expression
    top_80_idx = compute_topk_by_differential_expression(true_de, k=80)
    
    if verbose:
        print(f"  Computing global DE metrics...")
    global_metrics = compute_global_metrics(true_de, pred_de)
    results['global_pearson'] = global_metrics['pearson']
    results['global_spearman'] = global_metrics['spearman']
    results['global_r2_true'] = global_metrics['r2']
    results['global_r2_from_pearson'] = global_metrics['r2_from_pearson']
    results['global_rmse'] = global_metrics['rmse']
    
    if verbose:
        print(f"  Computing all-genes per-sample DE metrics...")
    allgenes_metrics = compute_persample_metrics(true_de, pred_de, gene_indices=None, show_progress=verbose)
    results['allgenes_pearson_mean'] = allgenes_metrics['pearson_mean']
    results['allgenes_pearson_median'] = allgenes_metrics['pearson_median']
    results['allgenes_spearman_mean'] = allgenes_metrics['spearman_mean']
    results['allgenes_r2_true_mean'] = allgenes_metrics['r2_true_mean']
    results['allgenes_r2_from_pearson_mean'] = allgenes_metrics['r2_from_pearson_mean']
    results['allgenes_rmse_mean'] = allgenes_metrics['rmse_mean']
    
    for k in [20, 40, 80]:
        if verbose:
            print(f"  Computing top-{k} DEG per-sample metrics...")
        top_k_idx = top_80_idx[:, :k]
        topk_metrics = compute_persample_metrics(true_de, pred_de, gene_indices=top_k_idx, show_progress=verbose)
        results[f'top{k}_pearson_mean'] = topk_metrics['pearson_mean']
        results[f'top{k}_pearson_median'] = topk_metrics['pearson_median']
        results[f'top{k}_spearman_mean'] = topk_metrics['spearman_mean']
        results[f'top{k}_r2_true_mean'] = topk_metrics['r2_true_mean']
        results[f'top{k}_r2_from_pearson_mean'] = topk_metrics['r2_from_pearson_mean']
        results[f'top{k}_rmse_mean'] = topk_metrics['rmse_mean']
        results[f'top{k}_var_true_mean'] = topk_metrics['var_true_mean']
    
    bias = compute_prediction_bias(true_de, pred_de)
    results['prediction_bias'] = bias['mean_error']
    results['high_value_bias'] = bias['high_value_bias']
    results['low_value_bias'] = bias['low_value_bias']
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"DE DIAGNOSTICS: {split_name.upper()} (Epoch {epoch})")
        print(f"{'='*70}")
        print(f"{'Metric':<40} {'Value':>12}")
        print(f"{'-'*55}")
        print(f"{'Global Pearson (DE)':<40} {results['global_pearson']:>12.4f}")
        print(f"{'Global True R² (DE)':<40} {results['global_r2_true']:>12.4f}")
        print(f"{'Global R² from Pearson² (DE)':<40} {results['global_r2_from_pearson']:>12.4f}")
        print(f"{'-'*55}")
        print(f"{'All-genes Pearson (DE, per-sample)':<40} {results['allgenes_pearson_mean']:>12.4f}")
        print(f"{'All-genes True R² (DE, per-sample)':<40} {results['allgenes_r2_true_mean']:>12.4f}")
        print(f"{'All-genes R² from Pearson² (DE)':<40} {results['allgenes_r2_from_pearson_mean']:>12.4f}")
        print(f"{'-'*55}")
        for k in [20, 40, 80]:
            print(f"{'Top-{} DEG Pearson (per-sample mean)':<40} {results[f'top{k}_pearson_mean']:>12.4f}".format(k))
            print(f"{'Top-{} DEG R² from Pearson²':<40} {results[f'top{k}_r2_from_pearson_mean']:>12.4f}".format(k))
            print(f"{'Top-{} DEG True R²':<40} {results[f'top{k}_r2_true_mean']:>12.4f}".format(k))
        print(f"{'-'*55}")
        print(f"{'Prediction Bias (mean error)':<40} {results['prediction_bias']:>12.4f}")
        print(f"{'='*70}\n")
    
    return results


def evaluate_topk_genes_persample_de(true_de, pred_de, top_indices):
    """
    Per-sample correlation metrics using precomputed top-k gene indices on DE.
    Returns R² from Pearson² for compatibility with existing code.
    """
    n_samples, k = top_indices.shape

    r2_values = np.full(n_samples, np.nan, dtype=float)
    pearson_values = np.full(n_samples, np.nan, dtype=float)
    spearman_values = np.full(n_samples, np.nan, dtype=float)

    for i in range(n_samples):
        idx = top_indices[i]
        true_de_sub = true_de[i, idx]
        pred_de_sub = pred_de[i, idx]

        if np.std(true_de_sub) == 0 or np.std(pred_de_sub) == 0:
            continue

        lr = linregress(true_de_sub, pred_de_sub)
        r = lr.rvalue
        s = spearmanr(true_de_sub, pred_de_sub).statistic

        if np.isfinite(r):
            pearson_values[i] = r
            r2_values[i] = r ** 2

        if np.isfinite(s):
            spearman_values[i] = s

    mean_r2 = np.nanmean(r2_values)
    mean_pearson = np.nanmean(pearson_values)
    mean_spearman = np.nanmean(spearman_values)

    return mean_r2, mean_pearson, mean_spearman


# =============================================================================
# DATA LOADING AND MATCHING FUNCTIONS
# =============================================================================

def load_and_match_diseased_data(args, data):
    """
    Load diseased data and match it to the treated data samples.
    
    Returns:
        dict with 'train', 'dev', 'test' keys, each containing:
            - 'diseased': numpy array of diseased expression matching treated samples
    """
    print("\n" + "=" * 70)
    print("LOADING DISEASED DATA FOR DIFFERENTIAL EXPRESSION")
    print("=" * 70)
    
    # Load diseased pickle
    print(f"Loading diseased data from: {args.diseased_pickle}")
    diseased_df = pd.read_pickle(args.diseased_pickle)
    print(f"Diseased data shape: {diseased_df.shape}")
    
    # Also need to load treated data to get matching info
    print(f"Loading treated data from: {args.data_pickle}")
    treated_df = pd.read_pickle(args.data_pickle)
    print(f"Treated data shape: {treated_df.shape}")
    
    # Identify gene columns (same logic as in biolord script)
    METADATA_COLS = ['cell_id', 'pert_id', 'pert_idose', 'idx', 'sig_id', 'pert_type', 'pert_iname']
    gene_cols = [col for col in treated_df.columns if col not in METADATA_COLS]
    print(f"Number of gene columns: {len(gene_cols)}")
    
    # Filter to cell type
    print(f"\nFiltering to cell type: {args.test_cell}")
    treated_cell = treated_df[treated_df['cell_id'] == args.test_cell].copy()
    diseased_cell = diseased_df[diseased_df['cell_id'] == args.test_cell].copy()
    print(f"Treated samples for {args.test_cell}: {len(treated_cell)}")
    print(f"Diseased samples for {args.test_cell}: {len(diseased_cell)}")
    
    if len(treated_cell) == 0:
        raise ValueError(f"No treated data for {args.test_cell}")
    if len(diseased_cell) == 0:
        raise ValueError(f"No diseased data for {args.test_cell}")
    
    # Reset indices
    treated_cell = treated_cell.reset_index(drop=True)
    diseased_cell = diseased_cell.reset_index(drop=True)
    
    # Match samples
    print("\nMatching treated and diseased samples...")
    
    if 'sig_id' in treated_cell.columns and 'sig_id' in diseased_cell.columns:
        print("Using 'sig_id' for matching")
        diseased_lookup = diseased_cell.set_index('sig_id')[gene_cols]
        
        matched_diseased = []
        valid_indices = []
        for i, sig_id in enumerate(treated_cell['sig_id']):
            if sig_id in diseased_lookup.index:
                matched_diseased.append(diseased_lookup.loc[sig_id].values)
                valid_indices.append(i)
        
        if len(valid_indices) < len(treated_cell):
            print(f"  WARNING: Only {len(valid_indices)}/{len(treated_cell)} samples matched")
            treated_cell = treated_cell.iloc[valid_indices].reset_index(drop=True)
        
        diseased_expr_full = np.array(matched_diseased).astype(np.float64)
        
    elif len(treated_cell) == len(diseased_cell):
        print("Same number of samples - assuming 1:1 correspondence")
        diseased_expr_full = diseased_cell[gene_cols].values.astype(np.float64)
    else:
        raise ValueError(
            f"Cannot match samples: treated has {len(treated_cell)}, "
            f"diseased has {len(diseased_cell)}. Need 'sig_id' column or same count."
        )
    
    print(f"Matched samples: {len(treated_cell)}")
    print(f"Diseased expression shape: {diseased_expr_full.shape}")
    
    # Now we need to split diseased data according to train/val/test splits
    # Load the split file
    from pathlib import Path
    splits_path = Path(args.splits_base_path) / args.test_cell / "random" / "5fold" / "splits.pt"
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    
    splits = torch.load(splits_path, weights_only=False)
    fold_splits = splits[args.fold]
    
    train_indices = fold_splits['train_index_backward']
    val_indices = fold_splits['val_index_backward']
    test_indices = fold_splits['test_index_backward']
    
    if isinstance(train_indices, torch.Tensor):
        train_indices = train_indices.numpy()
    if isinstance(val_indices, torch.Tensor):
        val_indices = val_indices.numpy()
    if isinstance(test_indices, torch.Tensor):
        test_indices = test_indices.numpy()
    
    train_idx_set = set(train_indices.tolist())
    val_idx_set = set(val_indices.tolist())
    test_idx_set = set(test_indices.tolist())
    
    # Get the idx column for matching
    if 'idx' in treated_cell.columns:
        idx_col = treated_cell['idx'].values
    else:
        idx_col = np.arange(len(treated_cell))
    
    # Create masks for each split
    train_mask = np.array([idx in train_idx_set for idx in idx_col])
    val_mask = np.array([idx in val_idx_set for idx in idx_col])
    test_mask = np.array([idx in test_idx_set for idx in idx_col])
    
    print(f"\nSplit sizes - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
    diseased_data = {
        'train': diseased_expr_full[train_mask],
        'dev': diseased_expr_full[val_mask],
        'test': diseased_expr_full[test_mask],
    }
    
    print(f"Diseased train shape: {diseased_data['train'].shape}")
    print(f"Diseased dev shape: {diseased_data['dev'].shape}")
    print(f"Diseased test shape: {diseased_data['test'].shape}")
    print("=" * 70 + "\n")
    
    return diseased_data, gene_cols


def precompute_topk_indices_by_de(data, diseased_data, k=80):
    """
    Precompute top-k gene indices for each sample based on |treated - diseased|.
    This is the KEY FIX - we select genes by differential expression, not raw expression.
    """
    precomputed = {}
    
    print("=" * 80)
    print("Gene Selection Strategy: Top-k by |DIFFERENTIAL EXPRESSION| (per sample)")
    print("  Formula: DE = treated - diseased")
    print("  Selection: top-k genes by |DE|")
    print("=" * 80)
    
    for split_name, dataloader_fn, diseased_key in [
        ('train', data.train_dataloader, 'train'), 
        ('dev', data.val_dataloader, 'dev'), 
        ('test', data.test_dataloader, 'test')
    ]:
        lb_list = []
        
        with torch.no_grad():
            for ft, lb, _ in dataloader_fn():
                lb_list.append(lb.detach().cpu().numpy())
                del lb, ft
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        treated_np = np.concatenate(lb_list, axis=0)
        del lb_list
        
        # Get corresponding diseased expression
        diseased_np = diseased_data[diseased_key]
        
        # Verify shapes match
        if treated_np.shape != diseased_np.shape:
            print(f"  WARNING: Shape mismatch for {split_name}!")
            print(f"    Treated: {treated_np.shape}, Diseased: {diseased_np.shape}")
            # Try to handle common case of length mismatch
            min_len = min(treated_np.shape[0], diseased_np.shape[0])
            treated_np = treated_np[:min_len]
            diseased_np = diseased_np[:min_len]
            print(f"    Truncated to {min_len} samples")
        
        # Compute differential expression
        de_np = treated_np - diseased_np
        
        # Select top-k by |DE|
        top_k_idx = compute_topk_by_differential_expression(de_np, k=k)
        
        precomputed[split_name] = {
            'treated': treated_np,
            'diseased': diseased_np,
            'de': de_np,
            'top_k_idx': top_k_idx
        }
        
        mean_de = np.mean(np.abs(de_np))
        mean_de_topk = np.mean(np.abs(de_np[np.arange(de_np.shape[0])[:, None], top_k_idx]))
        
        print(f"\n{split_name.capitalize()} split:")
        print(f"  Shape: {de_np.shape}")
        print(f"  Mean |DE| (all genes): {mean_de:.4f}")
        print(f"  Mean |DE| (top-{k} genes): {mean_de_topk:.4f}")
        print(f"  Enrichment ratio: {mean_de_topk/mean_de:.2f}x")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("=" * 80)
    print("Precomputation complete! Each sample has its own top-k DEG set.")
    print("=" * 80)
    
    return precomputed


def clear_memory():
    """Clear CUDA memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def model_training(args, model, data, ae_data, metrics_summary, topk_precomputed):
    """
    Training loop with DIFFERENTIAL EXPRESSION diagnostics.
    """
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory at training start: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    best_dev_pearson = float("-inf")
    
    all_diagnostics = {}

    for epoch in range(args.max_epoch):
        
        clear_memory()
        
        run_verbose_diag = (epoch == 0 or epoch == args.max_epoch - 1 or epoch % 5 == 0)

        # -----------------------------
        # TRAIN LOOP
        # -----------------------------
        model.train()
        epoch_loss = 0
        train_pred_list = [] if not args.skip_train_diagnostics else None

        for i, (ft, lb, _) in enumerate(data.train_dataloader()):
            drug = ft['drug']
            mask = ft['mask']
            cell_feature = ft['cell_id']
            pert_idose = ft['pert_idose']

            optimizer.zero_grad()
            predict, cell_hidden_ = model(
                input_cell_gex=cell_feature,
                input_drug=drug,
                input_gene=data.gene,
                mask=mask,
                input_pert_idose=pert_idose,
                job_id='perturbed',
                epoch=epoch,
            )

            loss_t = model.loss(lb, predict)
            loss_t.backward()
            optimizer.step()

            epoch_loss += loss_t.item()
            
            if not args.skip_train_diagnostics:
                train_pred_list.append(predict.detach().cpu().numpy())
            
            del predict, cell_hidden_

        print('Perturbed gene expression profile Train loss:')
        print(epoch_loss / (i + 1))
        if USE_WANDB:
            wandb.log({'Perturbed gene expression profile Train loss': epoch_loss / (i + 1)}, step=epoch)

        # --- TRAIN DIAGNOSTICS (DE-based) ---
        if not args.skip_train_diagnostics:
            train_pred_np = np.concatenate(train_pred_list, axis=0)
            train_diseased_np = topk_precomputed['train']['diseased']
            train_de_true = topk_precomputed['train']['de']
            train_top_k_idx = topk_precomputed['train']['top_k_idx']
            
            # Handle potential shape mismatch
            min_len = min(train_pred_np.shape[0], train_diseased_np.shape[0])
            train_pred_np = train_pred_np[:min_len]
            train_diseased_np = train_diseased_np[:min_len]
            train_de_true = train_de_true[:min_len]
            train_top_k_idx = train_top_k_idx[:min_len]
            
            # Compute predicted DE
            train_de_pred = train_pred_np - train_diseased_np

            train_diag = run_diagnostics_de(
                train_de_true, train_de_pred, 
                split_name="train", epoch=epoch, verbose=run_verbose_diag
            )
            
            for key, value in train_diag.items():
                metrics_summary[f'{key}_train'].append(value)
            
            if USE_WANDB:
                for key, value in train_diag.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        wandb.log({f'train_de_{key}': value}, step=epoch)

            # Legacy format metrics
            metrics_summary['pearson_list_perturbed_train'].append(train_diag['global_pearson'])
            metrics_summary['spearman_list_perturbed_train'].append(train_diag['global_spearman'])
            metrics_summary['rmse_list_perturbed_train'].append(train_diag['global_rmse'])

            for k in [20, 40, 80]:
                top_k_idx = train_top_k_idx[:, :k]
                r2_k, p_k, s_k = evaluate_topk_genes_persample_de(
                    train_de_true, train_de_pred, top_k_idx
                )
                metrics_summary[f"r2_top{k}_perturbed_train"].append(r2_k)
                metrics_summary[f"pearson_top{k}_perturbed_train"].append(p_k)
                metrics_summary[f"spearman_top{k}_perturbed_train"].append(s_k)

                if not run_verbose_diag:
                    print(f"Train DE top{k} -> R^2: {r2_k:.4f}, Pearson: {p_k:.4f}, Spearman: {s_k:.4f}")
                if USE_WANDB:
                    wandb.log({
                        f"Perturbed Train DE R2 top{k}": r2_k,
                        f"Perturbed Train DE Pearson top{k}": p_k,
                        f"Perturbed Train DE Spearman top{k}": s_k,
                    }, step=epoch)

            del train_pred_list, train_pred_np
        else:
            print("(Skipping train diagnostics to save memory)")
        
        clear_memory()

        # -----------------------------
        # DEV LOOP
        # -----------------------------
        model.eval()
        epoch_loss = 0
        predict_list = []
        
        with torch.no_grad():
            for i, (ft, lb, _) in enumerate(data.val_dataloader()):
                drug = ft['drug']
                mask = ft['mask']
                cell_feature = ft['cell_id']
                pert_idose = ft['pert_idose']
                predict, _ = model(
                    input_cell_gex=cell_feature,
                    input_drug=drug,
                    input_gene=data.gene,
                    mask=mask,
                    input_pert_idose=pert_idose,
                    job_id='perturbed',
                    epoch=epoch
                )
                loss = model.loss(lb, predict)
                epoch_loss += loss.item()
                predict_list.append(predict.cpu().numpy())

            predict_np = np.concatenate(predict_list, axis=0)
            dev_diseased_np = topk_precomputed['dev']['diseased']
            dev_de_true = topk_precomputed['dev']['de']
            dev_top_k_idx = topk_precomputed['dev']['top_k_idx']
            
            # Handle potential shape mismatch
            min_len = min(predict_np.shape[0], dev_diseased_np.shape[0])
            predict_np = predict_np[:min_len]
            dev_diseased_np = dev_diseased_np[:min_len]
            dev_de_true = dev_de_true[:min_len]
            dev_top_k_idx = dev_top_k_idx[:min_len]
            dev_treated_np = topk_precomputed['dev']['treated'][:min_len]

            # Compute predicted DE
            dev_de_pred = predict_np - dev_diseased_np

            print("DEV DE true finite fraction:", np.isfinite(dev_de_true).mean())
            print("DEV DE pred finite fraction:", np.isfinite(dev_de_pred).mean())

            # Also run legacy validation epoch end for raw expression (for backward compat)
            validation_epoch_end(
                epoch_loss=epoch_loss,
                lb_np=dev_treated_np,
                predict_np=predict_np,
                steps_per_epoch=i + 1,
                epoch=epoch,
                metrics_summary=metrics_summary,
                job='perturbed',
                USE_WANDB=USE_WANDB,
            )

            # --- COMPREHENSIVE DE DIAGNOSTICS FOR DEV ---
            dev_diag = run_diagnostics_de(
                dev_de_true, dev_de_pred,
                split_name="dev", epoch=epoch, verbose=run_verbose_diag
            )
            
            for key, value in dev_diag.items():
                metrics_summary[f'{key}_dev'].append(value)
            
            if USE_WANDB:
                for key, value in dev_diag.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        wandb.log({f'dev_de_{key}': value}, step=epoch)

            # Top-k metrics
            for k in [20, 40, 80]:
                top_k_idx = dev_top_k_idx[:, :k]
                r2_k, p_k, s_k = evaluate_topk_genes_persample_de(
                    dev_de_true, dev_de_pred, top_k_idx
                )
                metrics_summary[f"r2_top{k}_perturbed_dev"].append(r2_k)
                metrics_summary[f"pearson_top{k}_perturbed_dev"].append(p_k)
                metrics_summary[f"spearman_top{k}_perturbed_dev"].append(s_k)

                if not run_verbose_diag:
                    print(f"Dev DE top{k} -> R^2: {r2_k:.4f}, Pearson: {p_k:.4f}, Spearman: {s_k:.4f}")
                if USE_WANDB:
                    wandb.log({
                        f"perturbed Dev DE R2 top{k}": r2_k,
                        f"perturbed Dev DE Pearson top{k}": p_k,
                        f"perturbed Dev DE Spearman top{k}": s_k,
                    }, step=epoch)

            # Model selection: use DE all-genes pearson
            current_dev_pearson = dev_diag['allgenes_pearson_mean']
            if best_dev_pearson < current_dev_pearson or epoch == 1:
                best_dev_pearson = current_dev_pearson
                torch.save(model.state_dict(), args.model_name)
                print(f"  → Saved best model (dev DE all-genes Pearson: {best_dev_pearson:.4f})")

        del predict_list, predict_np
        clear_memory()

        # -----------------------------
        # TEST LOOP
        # -----------------------------
        epoch_loss = 0
        predict_list = []
        hidden_list = []
        
        with torch.no_grad():
            for i, (ft, lb, _) in enumerate(tqdm(data.test_dataloader())):
                drug = ft['drug']
                mask = ft['mask']
                cell_feature = ft['cell_id']
                pert_idose = ft['pert_idose']
                predict, cells_hidden_repr = model(
                    input_cell_gex=cell_feature,
                    input_drug=drug,
                    input_gene=data.gene,
                    mask=mask,
                    input_pert_idose=pert_idose,
                    job_id='perturbed'
                )
                loss = model.loss(lb, predict)
                epoch_loss += loss.item()
                predict_list.append(predict.cpu().numpy())
                hidden_list.append(cells_hidden_repr.cpu().numpy())

            predict_np = np.concatenate(predict_list, axis=0)
            hidden_np = np.concatenate(hidden_list, axis=0)
            test_diseased_np = topk_precomputed['test']['diseased']
            test_de_true = topk_precomputed['test']['de']
            test_top_k_idx = topk_precomputed['test']['top_k_idx']
            
            # Handle potential shape mismatch
            min_len = min(predict_np.shape[0], test_diseased_np.shape[0])
            predict_np = predict_np[:min_len]
            test_diseased_np = test_diseased_np[:min_len]
            test_de_true = test_de_true[:min_len]
            test_top_k_idx = test_top_k_idx[:min_len]
            test_treated_np = topk_precomputed['test']['treated'][:min_len]

            # Compute predicted DE
            test_de_pred = predict_np - test_diseased_np

            # Legacy test epoch end
            test_epoch_end(
                epoch_loss=epoch_loss,
                lb_np=test_treated_np,
                predict_np=predict_np,
                steps_per_epoch=i + 1,
                epoch=epoch,
                metrics_summary=metrics_summary,
                job='perturbed',
                USE_WANDB=USE_WANDB
            )

            # --- COMPREHENSIVE DE DIAGNOSTICS FOR TEST ---
            test_diag = run_diagnostics_de(
                test_de_true, test_de_pred,
                split_name="test", epoch=epoch, verbose=run_verbose_diag
            )
            
            for key, value in test_diag.items():
                metrics_summary[f'{key}_test'].append(value)
            
            if USE_WANDB:
                for key, value in test_diag.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        wandb.log({f'test_de_{key}': value}, step=epoch)
            
            if run_verbose_diag:
                all_diagnostics[f'epoch{epoch}'] = {
                    'train': train_diag if not args.skip_train_diagnostics else {},
                    'dev': dev_diag,
                    'test': test_diag
                }

            # Top-k metrics
            for k in [20, 40, 80]:
                top_k_idx = test_top_k_idx[:, :k]
                r2_k, p_k, s_k = evaluate_topk_genes_persample_de(
                    test_de_true, test_de_pred, top_k_idx
                )
                metrics_summary[f"r2_top{k}_perturbed_test"].append(r2_k)
                metrics_summary[f"pearson_top{k}_perturbed_test"].append(p_k)
                metrics_summary[f"spearman_top{k}_perturbed_test"].append(s_k)

                if not run_verbose_diag:
                    print(f"Test DE top{k} -> R^2: {r2_k:.4f}, Pearson: {p_k:.4f}, Spearman: {s_k:.4f}")
                if USE_WANDB:
                    wandb.log({
                        f"perturbed Test DE R2 top{k}": r2_k,
                        f"perturbed Test DE Pearson top{k}": p_k,
                        f"perturbed Test DE Spearman top{k}": s_k,
                    }, step=epoch)
        
        del predict_list, predict_np, hidden_list, hidden_np
        clear_memory()
    
    return all_diagnostics


if __name__ == '__main__':
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description='MultiDCP AE with Differential Expression Metrics')
    parser.add_argument('--drug_file', default="/raid/home/joshua/projects/MultiDCP/MultiDCP/data/all_drugs_pdg.csv")
    parser.add_argument('--gene_file', default="/raid/home/yoyowu/MultiDCP/data/gene_vector.csv")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--ae_input_file', default="/raid/home/yoyowu/MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4")
    parser.add_argument('--ae_label_file', default="/raid/home/yoyowu/MultiDCP/data/gene_expression_for_ae/gene_expression_combat_norm_978_split4")
    parser.add_argument('--cell_ge_file', help='the file which used to map cell line to gene expression file',
                        default="/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv")
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--predicted_result_for_testset', help="the file directory to save the predicted test dataframe", default=None)
    parser.add_argument('--hidden_repr_result_for_testset', help="the file directory to save the test data hidden representation dataframe", default=None)
    parser.add_argument('--all_cells', default="/raid/home/yoyowu/MultiDCP/data/ccle_tcga_ad_cells.p")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=343)
    parser.add_argument('--pretrained_model', type=str, default=None, help='add pretrained model path here if using pretrained model')
    parser.add_argument('--linear_encoder_flag', dest='linear_encoder_flag', action='store_true', default=False,
                        help='whether the cell embedding layer only have linear layers')
    parser.add_argument('--model_name', type=str, default='best_model.pt')
    
    # Data pickle paths - NOW REQUIRES BOTH TREATED AND DISEASED
    #parser.add_argument('--data_pickle', type=str, default='/raid/home/public/chemoe_collab_102025/PDGrapher/data/full_downloads/chemical/real_lognorm/chemical_PDGrapher_df_10302025.pkl',
    parser.add_argument('--data_pickle', type=str, default='/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_brddrugfiltered.pkl',
                        help='Path to treated data pickle file')



    #parser.add_argument('--diseased_pickle', type=str, default='/raid/home/public/chemoe_collab_102025/PDGrapher/data/full_downloads/chemical/real_lognorm/chemical_PDGrapher_diseased_df_10302025.pkl',
    parser.add_argument('--diseased_pickle', type=str, default='/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_diseased_brddrugfiltered.pkl',
                        help='Path to diseased/control data pickle file')
    parser.add_argument('--dev_cell', type=str, default=None,
                        help='Cell ID to hold out as development set (only for cell-type splitting mode)')
    parser.add_argument('--test_cell', type=str, required=True,
                        help='Cell type to train on')
    
    # Split file arguments
    parser.add_argument('--use_split_file', action='store_true', default=True,
                        help='Use PDGrapher-style split files')
    parser.add_argument('--splits_base_path', type=str, default='/raid/home/public/chemoe_collab_102025/PDGrapher/data/full_downloads/splits/chemical/',
                        help='Base path to splits directory')
    parser.add_argument('--fold', type=int, default=1,
                        help='Which fold to use (1-5)')
    parser.add_argument('--skip_train_diagnostics', action='store_true',
                        help='Skip computing diagnostics on training set to save memory')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device ID to use')

    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Validation
    if not os.path.exists(args.data_pickle):
        raise FileNotFoundError(f"Treated data pickle not found: {args.data_pickle}")
    if not os.path.exists(args.diseased_pickle):
        raise FileNotFoundError(f"Diseased data pickle not found: {args.diseased_pickle}")
    
    print(f"Using DE metrics mode: cell={args.test_cell}, fold={args.fold}")
    print(f"Treated pickle: {args.data_pickle}")
    print(f"Diseased pickle: {args.diseased_pickle}")
    
    all_cells = list(pickle.load(open(args.all_cells, 'rb')))

    DATA_FILTER = {"pert_type": ["trt_cp"]}

    ae_data = datareader.AEDataLoader(device, args)
    data = datareader.PerturbedDataLoader(DATA_FILTER, device, args)
    ae_data.setup()
    data.setup()
    print('#Train: %d' % len(data.train_data))
    print('#Dev: %d' % len(data.dev_data))
    print('#Test: %d' % len(data.test_data))
    print('#Train AE: %d' % len(ae_data.train_data))
    print('#Dev AE: %d' % len(ae_data.dev_data))
    print('#Test AE: %d' % len(ae_data.test_data))

    # Load diseased data and match to treated samples
    diseased_data, gene_cols = load_and_match_diseased_data(args, data)

    print("=" * 80)
    print("Precomputing top-k gene indices by |DIFFERENTIAL EXPRESSION| (per sample)...")
    print("=" * 80)
    topk_precomputed = precompute_topk_indices_by_de(data, diseased_data, k=80)
    print("Precomputation complete!")
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after precomputation: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
              f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
    
    print("=" * 80)

    # parameters initialization
    pert_idose_dim = data.train_data.feature['pert_idose'].shape[1]
    model_param_registry = initialize_model_registry()
    model_param_registry.update({
        'num_gene': GEX_SIZE,
        'pert_idose_input_dim': pert_idose_dim,
        'dropout': args.dropout,
        'linear_encoder_flag': args.linear_encoder_flag
    })

    print('--------------with linear encoder: {0!r}--------------'.format(args.linear_encoder_flag))
    model = multidcp_chemoe.MultiDCP_CheMoE_AE(device=device, model_param_registry=model_param_registry)
    model.init_weights(pretrained=args.pretrained_model)
    model.to(device)
    model = model.double()
    USE_WANDB = True
    
    wandb_name = f'multidcp_chemoe_de_{args.test_cell}_fold{args.fold}_sd{args.seed}'
    
    if USE_WANDB:
        wandb.init(project="MultiDCP_CheMoE_AE_DE", config=args, name=wandb_name)
        wandb.watch(model, log="all")
    else:
        os.environ["WANDB_MODE"] = "dryrun"

    # training
    metrics_summary = defaultdict(
        list,
        pearson_list_ae_dev=[],
        pearson_list_ae_test=[],
        pearson_list_perturbed_dev=[],
        pearson_list_perturbed_test=[],
        spearman_list_ae_dev=[],
        spearman_list_ae_test=[],
        spearman_list_perturbed_dev=[],
        spearman_list_perturbed_test=[],
        rmse_list_ae_dev=[],
        rmse_list_ae_test=[],
        rmse_list_perturbed_dev=[],
        rmse_list_perturbed_test=[],
        precisionk_list_ae_dev=[],
        precisionk_list_ae_test=[],
        precisionk_list_perturbed_dev=[],
        precisionk_list_perturbed_test=[],
        pearson_list_perturbed_train=[],
        spearman_list_perturbed_train=[],
        rmse_list_perturbed_train=[],
        r2_top20_perturbed_train=[],
        r2_top40_perturbed_train=[],
        r2_top80_perturbed_train=[],
        r2_top20_perturbed_dev=[],
        r2_top40_perturbed_dev=[],
        r2_top80_perturbed_dev=[],
        r2_top20_perturbed_test=[],
        r2_top40_perturbed_test=[],
        r2_top80_perturbed_test=[],
        pearson_top20_perturbed_train=[],
        pearson_top40_perturbed_train=[],
        pearson_top80_perturbed_train=[],
        spearman_top20_perturbed_train=[],
        spearman_top40_perturbed_train=[],
        spearman_top80_perturbed_train=[],
        pearson_top20_perturbed_dev=[],
        pearson_top40_perturbed_dev=[],
        pearson_top80_perturbed_dev=[],
        spearman_top20_perturbed_dev=[],
        spearman_top40_perturbed_dev=[],
        spearman_top80_perturbed_dev=[],
        pearson_top20_perturbed_test=[],
        pearson_top40_perturbed_test=[],
        pearson_top80_perturbed_test=[],
        spearman_top20_perturbed_test=[],
        spearman_top40_perturbed_test=[],
        spearman_top80_perturbed_test=[],
    )

    all_diagnostics = model_training(args, model, data, ae_data, metrics_summary, topk_precomputed)

    report_final_results(metrics_summary, ae=False, perturbed=True)

    # --- Final Summary of DE Diagnostics ---
    print("\n" + "=" * 80)
    print("FINAL DE DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Metric':<40} {'Train':>12} {'Dev':>12} {'Test':>12}")
    print("-" * 80)
    
    diagnostic_keys = [
        'global_pearson',
        'global_r2_true',
        'allgenes_pearson_mean',
        'allgenes_r2_true_mean',
        'top80_pearson_mean',
        'top80_r2_true_mean',
        'top80_r2_from_pearson_mean',
        'prediction_bias',
        'high_value_bias',
        'low_value_bias',
    ]
    
    for key in diagnostic_keys:
        train_val = metrics_summary.get(f'{key}_train', [np.nan])[-1]
        dev_val = metrics_summary.get(f'{key}_dev', [np.nan])[-1]
        test_val = metrics_summary.get(f'{key}_test', [np.nan])[-1]
        print(f"{key:<40} {train_val:>12.4f} {dev_val:>12.4f} {test_val:>12.4f}")

    # --- Build R^2 timeseries DataFrame ---
    r2_keys = [
        "r2_top20_perturbed_dev",
        "r2_top40_perturbed_dev",
        "r2_top80_perturbed_dev",
        "r2_top20_perturbed_test",
        "r2_top40_perturbed_test",
        "r2_top80_perturbed_test",
    ]

    r2_lengths = {k: len(metrics_summary[k]) for k in r2_keys}
    print("\nDE R2 list lengths:", r2_lengths)

    common_len = min(r2_lengths.values()) if r2_lengths else 0
    print("Using common length:", common_len)

    if common_len > 0:
        r2_df = pd.DataFrame({
            "epoch": np.arange(common_len),
            "r2_top20_dev": metrics_summary["r2_top20_perturbed_dev"][:common_len],
            "r2_top40_dev": metrics_summary["r2_top40_perturbed_dev"][:common_len],
            "r2_top80_dev": metrics_summary["r2_top80_perturbed_dev"][:common_len],
            "r2_top20_test": metrics_summary["r2_top20_perturbed_test"][:common_len],
            "r2_top40_test": metrics_summary["r2_top40_perturbed_test"][:common_len],
            "r2_top80_test": metrics_summary["r2_top80_perturbed_test"][:common_len],
            "allgenes_r2_dev": metrics_summary.get("allgenes_r2_true_mean_dev", [np.nan] * common_len)[:common_len],
            "allgenes_r2_test": metrics_summary.get("allgenes_r2_true_mean_test", [np.nan] * common_len)[:common_len],
            "allgenes_pearson_dev": metrics_summary.get("allgenes_pearson_mean_dev", [np.nan] * common_len)[:common_len],
            "allgenes_pearson_test": metrics_summary.get("allgenes_pearson_mean_test", [np.nan] * common_len)[:common_len],
        })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"diagnostics_de_{args.test_cell}_fold{args.fold}_{timestamp}.csv"
        r2_df.to_csv(out_name, index=False)
        print(f"\nSaved DE diagnostics CSV to: {out_name}")

    end_time = datetime.now()
    print(f"\nTotal runtime: {end_time - start_time}")