#!/usr/bin/env python3
"""
Biolord training script for PDGrapher data.
Uses the same data format and split files as MultiDCP.
Includes comprehensive evaluation metrics matching MultiDCP diagnostics.

KEY: Computes R² on DIFFERENTIAL EXPRESSION (treated - diseased), not raw expression.
Top-k genes are selected by |true_treated - diseased| (true differential expression).

Usage:
    python train_bl_pdg_de.py --test_cell A375 --fold 1 --gpu 6
"""

import os
import sys
import argparse

# =============================================================================
# SET GPU BEFORE IMPORTING TORCH
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

# Now import everything else
import pandas as pd
import numpy as np
import scanpy as sc
import biolord
import torch
import warnings
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

warnings.simplefilter("ignore", UserWarning)

# Set scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# =============================================================================
# WANDB SETUP
# =============================================================================
WANDB_AVAILABLE = False
WANDB_RUN = None
wandb = None


def init_wandb_and_start_run(project, name, config, tags):
    """Initialize wandb and start a run."""
    global WANDB_AVAILABLE, WANDB_RUN, wandb
    
    print("\n" + "=" * 70)
    print("INITIALIZING WEIGHTS & BIASES")
    print("=" * 70)
    
    try:
        import wandb as wb
        wandb = wb
        print(f"wandb version: {wb.__version__}")
    except ImportError:
        print("ERROR: wandb not installed.")
        print("=" * 70 + "\n")
        return False
    
    try:
        api_key = os.environ.get('WANDB_API_KEY') or wb.api.api_key
        if not api_key:
            print("ERROR: wandb API key not found.")
            print("=" * 70 + "\n")
            return False
        print("API key found ✓")
    except Exception as e:
        print(f"ERROR checking API key: {e}")
        print("=" * 70 + "\n")
        return False
    
    try:
        WANDB_RUN = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            reinit=True,
        )
        
        if WANDB_RUN is None:
            print("ERROR: wandb.init() returned None")
            print("=" * 70 + "\n")
            return False
        
        WANDB_AVAILABLE = True
        
        run_url = wandb.run.get_url()
        print("\n" + "*" * 70)
        print("*  WANDB RUN STARTED SUCCESSFULLY!" + " " * 33 + "*")
        print(f"*  Project: {project:<56} *")
        print(f"*  Run Name: {name:<55} *")
        print(f"*  {run_url:<66} *")
        print("*" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"ERROR: wandb.init() failed: {e}")
        WANDB_AVAILABLE = False
        return False


def wandb_log(metrics_dict, step=None):
    """Log metrics to wandb if available."""
    if not WANDB_AVAILABLE or wandb is None:
        return
    try:
        clean_metrics = {}
        for k, v in metrics_dict.items():
            if k.startswith('_'):
                continue
            if isinstance(v, (int, float)) and np.isfinite(v):
                clean_metrics[k] = v
        if clean_metrics:
            if step is not None:
                wandb.log(clean_metrics, step=step)
            else:
                wandb.log(clean_metrics)
    except Exception as e:
        print(f"WARNING: wandb.log() failed: {e}")


def wandb_config_update(config_dict):
    """Update wandb config if available."""
    if not WANDB_AVAILABLE or wandb is None:
        return
    try:
        wandb.config.update(config_dict, allow_val_change=True)
    except Exception as e:
        print(f"WARNING: wandb.config.update() failed: {e}")


def wandb_summary_update(key, value):
    """Update wandb run summary if available."""
    if not WANDB_AVAILABLE or wandb is None or wandb.run is None:
        return
    try:
        wandb.run.summary[key] = value
    except Exception as e:
        print(f"WARNING: wandb summary update failed: {e}")


def wandb_finish():
    """Finish wandb run if available."""
    if not WANDB_AVAILABLE or wandb is None:
        return
    try:
        run_url = wandb.run.get_url() if wandb.run else "N/A"
        wandb.finish()
        print("\n" + "=" * 70)
        print("WANDB RUN COMPLETED")
        print(f"Final dashboard: {run_url}")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"WARNING: wandb.finish() failed: {e}")


# =============================================================================
# DIAGNOSTIC FUNCTIONS - DIFFERENTIAL EXPRESSION BASED
# =============================================================================

def compute_topk_by_differential_expression(true_de, k=20):
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


def compute_persample_metrics(true_de, pred_de, gene_indices=None, show_progress=True):
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
    
    # Spearman correlation
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


def run_diagnostics_de(true_de, pred_de, split_name, verbose=True):
    """
    Run comprehensive diagnostics on DIFFERENTIAL EXPRESSION.
    
    Args:
        true_de: True differential expression (treated - diseased)
        pred_de: Predicted differential expression (pred_treated - diseased)
        split_name: Name for logging
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
        print(f"DE DIAGNOSTICS: {split_name.upper()}")
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


def get_biolord_predictions(model, adata, split=None):
    """Get reconstructed gene expression predictions from biolord model."""
    if split is not None:
        mask = adata.obs['split'] == split
        adata_subset = adata[mask].copy()
    else:
        adata_subset = adata
    
    true_vals = adata_subset.X
    if hasattr(true_vals, 'toarray'):
        true_vals = true_vals.toarray()
    true_vals = np.array(true_vals)
    
    n_samples, n_genes = true_vals.shape
    print(f"  True values shape: {true_vals.shape}")
    
    # Get predictions via module inference + generative
    try:
        model.module.eval()
        model._validate_anndata(adata_subset)
        
        scdl = model._make_data_loader(
            adata=adata_subset,
            batch_size=256,
            shuffle=False,
        )
        
        all_preds = []
        
        with torch.no_grad():
            for batch_idx, tensors in enumerate(scdl):
                inference_inputs = model.module._get_inference_input(tensors)
                inference_outputs = model.module.inference(**inference_inputs)
                generative_inputs = model.module._get_generative_input(tensors, inference_outputs)
                generative_outputs = model.module.generative(**generative_inputs)
                
                if batch_idx == 0:
                    print(f"    Generative output keys: {list(generative_outputs.keys())}")
                
                # Find reconstruction
                pred = None
                for key in ['px_m', 'px_rate', 'px', 'reconstruction', 'x_rec', 'mean', 'x_hat']:
                    if key in generative_outputs:
                        pred = generative_outputs[key]
                        if batch_idx == 0:
                            print(f"    Using '{key}' with shape {pred.shape}")
                        break
                
                if pred is None:
                    for key, val in generative_outputs.items():
                        if hasattr(val, 'shape') and len(val.shape) == 2 and val.shape[1] == n_genes:
                            pred = val
                            if batch_idx == 0:
                                print(f"    Using '{key}' (by shape) with shape {pred.shape}")
                            break
                
                if pred is None:
                    raise ValueError(f"Could not find reconstruction: {list(generative_outputs.keys())}")
                
                all_preds.append(pred.cpu().numpy())
        
        pred_vals = np.vstack(all_preds)
        print(f"  Predictions shape: {pred_vals.shape}")
        
        return true_vals, pred_vals
        
    except Exception as e:
        print(f"  Prediction extraction failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def evaluate_model_de(model, adata, diseased_expr, save_dir, test_cell, fold):
    """
    Evaluate biolord model using DIFFERENTIAL EXPRESSION metrics.
    
    Args:
        model: Trained biolord model
        adata: AnnData with treated expression
        diseased_expr: Diseased/control expression array matching adata samples
        save_dir: Output directory
        test_cell: Cell type name
        fold: Fold number
    """
    print("\n" + "=" * 70)
    print("MODEL EVALUATION (Differential Expression)")
    print("=" * 70)
    
    all_results = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\nEvaluating {split} split...")
        
        try:
            # Get predictions for this split
            mask = adata.obs['split'] == split
            split_indices = np.where(mask)[0]
            
            true_treated, pred_treated = get_biolord_predictions(model, adata, split=split)
            
            # Get corresponding diseased values
            diseased_split = diseased_expr[split_indices]
            
            print(f"  True treated shape: {true_treated.shape}")
            print(f"  Pred treated shape: {pred_treated.shape}")
            print(f"  Diseased shape: {diseased_split.shape}")
            
            # Compute differential expression
            true_de = true_treated - diseased_split  # True perturbation effect
            pred_de = pred_treated - diseased_split  # Predicted perturbation effect
            
            print(f"  True DE range: [{true_de.min():.3f}, {true_de.max():.3f}]")
            print(f"  Pred DE range: [{pred_de.min():.3f}, {pred_de.max():.3f}]")
            
            # Run diagnostics on differential expression
            results = run_diagnostics_de(true_de, pred_de, split_name=split, verbose=True)
            
            for key, value in results.items():
                all_results[f'{split}_{key}'] = value
            
            # Log to wandb
            wandb_metrics = {f'eval_de/{split}/{key}': value 
                           for key, value in results.items() 
                           if isinstance(value, (int, float)) and np.isfinite(value)}
            wandb_log(wandb_metrics)
                
        except Exception as e:
            print(f"  ERROR evaluating {split}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    summary_data = {
        'cell_type': test_cell,
        'fold': fold,
    }
    summary_data.update(all_results)
    
    summary_df = pd.DataFrame([summary_data])
    
    metrics_path = save_dir / f"metrics_de_{test_cell}_fold{fold}.csv"
    summary_df.to_csv(metrics_path, index=False)
    print(f"\nDE Metrics saved to: {metrics_path}")
    
    # Log to wandb summary
    if WANDB_AVAILABLE and wandb is not None:
        for k in [20, 40, 80]:
            for split in ['train', 'val', 'test']:
                for metric_type, suffix in [('r2_from_pearson_mean', 'r2'), 
                                            ('r2_true_mean', 'r2_true'),
                                            ('pearson_mean', 'pearson')]:
                    key = f'{split}_top{k}_{metric_type}'
                    if key in all_results and np.isfinite(all_results[key]):
                        wandb_summary_update(f'{split}/top{k}_de_{suffix}', all_results[key])
        
        # Log final metrics for easy comparison
        final_metrics = {}
        for k in [20, 40, 80]:
            for split in ['train', 'val', 'test']:
                key = f'{split}_top{k}_r2_from_pearson_mean'
                if key in all_results and np.isfinite(all_results[key]):
                    final_metrics[f'final/{split}_top{k}_r2'] = all_results[key]
        wandb_log(final_metrics)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: TOP-K DEG R² (from Pearson²) - DIFFERENTIAL EXPRESSION")
    print("=" * 70)
    print(f"{'K':<8} {'Train':>15} {'Val':>15} {'Test':>15}")
    print("-" * 70)
    for k in [20, 40, 80]:
        train_r2 = all_results.get(f'train_top{k}_r2_from_pearson_mean', np.nan)
        val_r2 = all_results.get(f'val_top{k}_r2_from_pearson_mean', np.nan)
        test_r2 = all_results.get(f'test_top{k}_r2_from_pearson_mean', np.nan)
        print(f"Top-{k:<4} {train_r2:>15.4f} {val_r2:>15.4f} {test_r2:>15.4f}")
    print("-" * 70)
    train_all = all_results.get('train_allgenes_r2_from_pearson_mean', np.nan)
    val_all = all_results.get('val_allgenes_r2_from_pearson_mean', np.nan)
    test_all = all_results.get('test_allgenes_r2_from_pearson_mean', np.nan)
    print(f"{'All':<8} {train_all:>15.4f} {val_all:>15.4f} {test_all:>15.4f}")
    print("=" * 70)
    
    return summary_df, all_results


def main(args):
    """Main training function."""
    
    start_time = datetime.now()
    
    TEST_CELL = args.test_cell
    FOLD = args.fold
    
    SAVE_DIR = Path(args.output_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb
    if args.use_wandb:
        wandb_name = f'biolord_de_{TEST_CELL}_fold{FOLD}_sd{args.seed}'
        wandb_ready = init_wandb_and_start_run(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args),
            tags=[TEST_CELL, f'fold{FOLD}', 'biolord', 'DE'],
        )
        if not wandb_ready:
            print("Continuing without wandb logging.\n")
    else:
        print("\nwandb logging disabled by user.\n")
    
    print("=" * 70)
    print(f"BIOLORD TRAINING (DE metrics): {TEST_CELL} fold {FOLD}")
    print("=" * 70)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        wandb_config_update({'gpu_name': gpu_name})
    else:
        print("WARNING: No GPU available, using CPU")
    
    # ==========================================================================
    # LOAD BOTH TREATED AND DISEASED DATA
    # ==========================================================================
    print("\n--- Loading data ---")
    
    print(f"Loading treated data from: {args.treated_pickle}")
    treated_df = pd.read_pickle(args.treated_pickle)
    print(f"Treated data shape: {treated_df.shape}")
    
    print(f"Loading diseased data from: {args.diseased_pickle}")
    diseased_df = pd.read_pickle(args.diseased_pickle)
    print(f"Diseased data shape: {diseased_df.shape}")
    
    # Identify columns
    METADATA_COLS = ['cell_id', 'pert_id', 'pert_idose', 'idx', 'sig_id', 'pert_type', 'pert_iname']
    gene_cols = [col for col in treated_df.columns if col not in METADATA_COLS]
    print(f"Number of gene columns: {len(gene_cols)}")
    
    # Filter to cell type
    print(f"\n--- Filtering to cell type: {TEST_CELL} ---")
    treated_cell = treated_df[treated_df['cell_id'] == TEST_CELL].copy()
    diseased_cell = diseased_df[diseased_df['cell_id'] == TEST_CELL].copy()
    print(f"Treated samples for {TEST_CELL}: {len(treated_cell)}")
    print(f"Diseased samples for {TEST_CELL}: {len(diseased_cell)}")
    
    if len(treated_cell) == 0:
        raise ValueError(f"No treated data for {TEST_CELL}")
    if len(diseased_cell) == 0:
        raise ValueError(f"No diseased data for {TEST_CELL}")
    
    # ==========================================================================
    # MATCH TREATED AND DISEASED SAMPLES
    # ==========================================================================
    print("\n--- Matching treated and diseased samples ---")
    
    # Reset indices
    treated_cell = treated_cell.reset_index(drop=True)
    diseased_cell = diseased_cell.reset_index(drop=True)
    
    # Check if they have matching sig_id or idx columns
    if 'sig_id' in treated_cell.columns and 'sig_id' in diseased_cell.columns:
        print("Using 'sig_id' for matching")
        # Create lookup from diseased
        diseased_lookup = diseased_cell.set_index('sig_id')[gene_cols]
        
        # Match
        matched_diseased = []
        valid_indices = []
        for i, sig_id in enumerate(treated_cell['sig_id']):
            if sig_id in diseased_lookup.index:
                matched_diseased.append(diseased_lookup.loc[sig_id].values)
                valid_indices.append(i)
        
        if len(valid_indices) < len(treated_cell):
            print(f"  WARNING: Only {len(valid_indices)}/{len(treated_cell)} samples matched")
            treated_cell = treated_cell.iloc[valid_indices].reset_index(drop=True)
        
        diseased_expr = np.array(matched_diseased)
        
    elif len(treated_cell) == len(diseased_cell):
        print("Same number of samples - assuming 1:1 correspondence")
        diseased_expr = diseased_cell[gene_cols].values
    else:
        raise ValueError(
            f"Cannot match samples: treated has {len(treated_cell)}, "
            f"diseased has {len(diseased_cell)}. Need 'sig_id' column or same count."
        )
    
    print(f"Matched samples: {len(treated_cell)}")
    print(f"Diseased expression shape: {diseased_expr.shape}")
    
    # ==========================================================================
    # LOAD SPLITS
    # ==========================================================================
    print(f"\n--- Loading split file ---")
    splits_path = Path(args.splits_base_path) / TEST_CELL / "random" / "5fold" / "splits.pt"
    if not splits_path.exists():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    
    splits = torch.load(splits_path, weights_only=False)
    print(f"Loaded splits from: {splits_path}")
    
    if FOLD not in splits:
        raise ValueError(f"Fold {FOLD} not found. Available: {list(splits.keys())}")
    
    fold_splits = splits[FOLD]
    
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
    
    print(f"Split sizes - Train: {len(train_idx_set)}, Val: {len(val_idx_set)}, Test: {len(test_idx_set)}")
    
    # Assign splits
    treated_cell['split'] = 'unknown'
    
    if 'idx' in treated_cell.columns:
        print("Using 'idx' column for split assignment")
        treated_cell.loc[treated_cell['idx'].isin(train_idx_set), 'split'] = 'train'
        treated_cell.loc[treated_cell['idx'].isin(val_idx_set), 'split'] = 'val'
        treated_cell.loc[treated_cell['idx'].isin(test_idx_set), 'split'] = 'test'
    else:
        print("Using row indices for split assignment")
        all_indices = train_idx_set | val_idx_set | test_idx_set
        max_idx = max(all_indices)
        
        if max_idx >= len(treated_cell):
            for i in range(len(treated_cell)):
                if i in train_idx_set:
                    treated_cell.iloc[i, treated_cell.columns.get_loc('split')] = 'train'
                elif i in val_idx_set:
                    treated_cell.iloc[i, treated_cell.columns.get_loc('split')] = 'val'
                elif i in test_idx_set:
                    treated_cell.iloc[i, treated_cell.columns.get_loc('split')] = 'test'
        else:
            for idx in train_idx_set:
                if idx < len(treated_cell):
                    treated_cell.iloc[idx, treated_cell.columns.get_loc('split')] = 'train'
            for idx in val_idx_set:
                if idx < len(treated_cell):
                    treated_cell.iloc[idx, treated_cell.columns.get_loc('split')] = 'val'
            for idx in test_idx_set:
                if idx < len(treated_cell):
                    treated_cell.iloc[idx, treated_cell.columns.get_loc('split')] = 'test'
    
    print(f"\nSplit distribution:\n{treated_cell['split'].value_counts()}")
    
    # Remove unknown
    n_unknown = (treated_cell['split'] == 'unknown').sum()
    if n_unknown > 0:
        print(f"WARNING: {n_unknown} samples unassigned, removing them")
        valid_mask = treated_cell['split'] != 'unknown'
        treated_cell = treated_cell[valid_mask].reset_index(drop=True)
        diseased_expr = diseased_expr[valid_mask.values]
        print(f"Final sample count: {len(treated_cell)}")
    
    # ==========================================================================
    # CREATE ANNDATA
    # ==========================================================================
    print("\n--- Creating AnnData ---")
    X = treated_cell[gene_cols].values.astype(np.float32)
    print(f"Gene expression matrix shape: {X.shape}")
    
    obs_dict = {
        'cell_id': treated_cell['cell_id'].astype('category').values,
        'pert_id': treated_cell['pert_id'].astype('category').values,
        'split': pd.Categorical(treated_cell['split'].values),
    }
    
    if 'pert_idose' in treated_cell.columns:
        dose_str = treated_cell['pert_idose'].astype(str)
        dose_numeric = pd.to_numeric(dose_str.str.extract(r'([\d.]+)')[0], errors='coerce').fillna(0)
        obs_dict['dose'] = dose_numeric.values
    
    obs = pd.DataFrame(obs_dict).reset_index(drop=True)
    
    adata = sc.AnnData(X=X, obs=obs)
    adata.var_names = gene_cols
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    
    print(f"AnnData created: {adata.shape}")
    print(f"Diseased expression shape: {diseased_expr.shape}")
    
    # ==========================================================================
    # SETUP AND TRAIN BIOLORD
    # ==========================================================================
    print("\n--- Setting up Biolord ---")
    categorical_attributes = ['pert_id']
    ordered_attributes = []
    if 'dose' in adata.obs.columns and adata.obs['dose'].nunique() > 1:
        ordered_attributes.append('dose')
    
    print(f"Categorical attributes: {categorical_attributes}")
    print(f"Ordered attributes: {ordered_attributes}")
    
    for col in categorical_attributes:
        adata.obs[col] = adata.obs[col].astype('category')
    
    biolord.Biolord.setup_anndata(
        adata,
        ordered_attributes_keys=ordered_attributes if ordered_attributes else None,
        categorical_attributes_keys=categorical_attributes,
    )
    
    print("\n--- Instantiating Biolord model ---")
    module_params = {
        "decoder_width": args.decoder_width,
        "decoder_depth": args.decoder_depth,
        "attribute_nn_width": args.attribute_nn_width,
        "attribute_nn_depth": args.attribute_nn_depth,
        "n_latent_attribute_categorical": args.n_latent_attribute_categorical,
        "gene_likelihood": "normal",
        "reconstruction_penalty": args.reconstruction_penalty,
        "unknown_attribute_penalty": args.unknown_attribute_penalty,
        "unknown_attribute_noise_param": args.unknown_attribute_noise_param,
        "attribute_dropout_rate": args.attribute_dropout_rate,
        "use_batch_norm": False,
        "use_layer_norm": False,
        "seed": args.seed,
    }
    
    model = biolord.Biolord(
        adata=adata,
        n_latent=args.n_latent,
        model_name=f"biolord_{TEST_CELL}_fold{FOLD}",
        module_params=module_params,
        train_classifiers=False,
        split_key="split",
    )
    
    print(f"Model created with n_latent={args.n_latent}")
    
    print("\n--- Training model ---")
    trainer_params = {
        "n_epochs_warmup": 0,
        "latent_lr": args.latent_lr,
        "latent_wd": args.latent_wd,
        "decoder_lr": args.decoder_lr,
        "decoder_wd": args.decoder_wd,
        "attribute_nn_lr": args.attribute_nn_lr,
        "attribute_nn_wd": args.attribute_nn_wd,
        "step_size_lr": 45,
        "cosine_scheduler": True,
        "scheduler_final_lr": 1e-5,
    }
    
    train_kwargs = {
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "plan_kwargs": trainer_params,
        "early_stopping": True,
        "early_stopping_patience": args.early_stopping_patience,
        "check_val_every_n_epoch": args.check_val_every_n_epoch,
        "num_workers": args.num_workers,
        "enable_checkpointing": False,
    }
    
    if WANDB_AVAILABLE:
        try:
            from pytorch_lightning.loggers import WandbLogger
            wandb_logger = WandbLogger(experiment=wandb.run, log_model=False)
            train_kwargs["logger"] = wandb_logger
            print("Using WandbLogger for live training metrics")
        except Exception as e:
            print(f"Could not setup WandbLogger: {e}")
    
    model.train(**train_kwargs)
    print("\nTraining completed!")
    
    # ==========================================================================
    # EVALUATE WITH DIFFERENTIAL EXPRESSION METRICS
    # ==========================================================================
    summary_df, all_results = evaluate_model_de(
        model, adata, diseased_expr, SAVE_DIR, TEST_CELL, FOLD
    )
    
    # ==========================================================================
    # SAVE OUTPUTS
    # ==========================================================================
    print("\n--- Saving outputs ---")
    model_save_path = SAVE_DIR / f"biolord_{TEST_CELL}_fold{FOLD}"
    model.save(str(model_save_path), overwrite=True)
    print(f"Model saved to: {model_save_path}")
    
    adata_save_path = SAVE_DIR / f"adata_{TEST_CELL}_fold{FOLD}.h5ad"
    adata.write(adata_save_path)
    print(f"AnnData saved to: {adata_save_path}")
    
    # Also save diseased expression for reference
    diseased_path = SAVE_DIR / f"diseased_{TEST_CELL}_fold{FOLD}.npy"
    np.save(diseased_path, diseased_expr)
    print(f"Diseased expression saved to: {diseased_path}")
    
    # Summary
    end_time = datetime.now()
    runtime = end_time - start_time
    
    print("\n" + "=" * 70)
    print(f"COMPLETED: {TEST_CELL} fold {FOLD}")
    print("=" * 70)
    print(f"Total runtime: {runtime}")
    print(f"Output directory: {SAVE_DIR}")
    print("=" * 70)
    
    wandb_summary_update('runtime_seconds', runtime.total_seconds())
    wandb_finish()
    
    return model, adata, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Biolord training with DE metrics')
    
    # Data paths
    parser.add_argument('--treated_pickle', type=str,
                        default='/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl',
                        help='Path to treated data pickle')
    parser.add_argument('--diseased_pickle', type=str,
                        default='/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl',
                        help='Path to diseased/control data pickle')
    parser.add_argument('--splits_base_path', type=str,
                        default='/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical',
                        help='Base path to splits directory')
    parser.add_argument('--output_dir', type=str,
                        default='/raid/home/joshua/projects/biolord_comp_122025',
                        help='Output directory')
    
    parser.add_argument('--test_cell', type=str, required=True,
                        help='Cell type to train on')
    parser.add_argument('--fold', type=int, default=1,
                        help='Which fold to use (1-5)')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device ID')
    
    # wandb
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb')
    parser.add_argument('--wandb_project', type=str, default='biolord_pdgrapher')
    
    # Training params (official biolord)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    
    # Model params (official biolord)
    parser.add_argument('--n_latent', type=int, default=32)
    parser.add_argument('--decoder_width', type=int, default=1024)
    parser.add_argument('--decoder_depth', type=int, default=4)
    parser.add_argument('--attribute_nn_width', type=int, default=512)
    parser.add_argument('--attribute_nn_depth', type=int, default=2)
    parser.add_argument('--n_latent_attribute_categorical', type=int, default=4)
    
    # Loss weights (official biolord)
    parser.add_argument('--reconstruction_penalty', type=float, default=1e2)
    parser.add_argument('--unknown_attribute_penalty', type=float, default=1e1)
    parser.add_argument('--unknown_attribute_noise_param', type=float, default=1e-1)
    parser.add_argument('--attribute_dropout_rate', type=float, default=0.1)
    
    # Learning rates (official biolord)
    parser.add_argument('--latent_lr', type=float, default=1e-4)
    parser.add_argument('--latent_wd', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_wd', type=float, default=1e-4)
    parser.add_argument('--attribute_nn_lr', type=float, default=1e-2)
    parser.add_argument('--attribute_nn_wd', type=float, default=4e-8)
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args)