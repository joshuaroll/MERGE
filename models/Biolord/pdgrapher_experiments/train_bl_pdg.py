#!/usr/bin/env python3
"""
Biolord training script for PDGrapher data.
Uses the same data format and split files as MultiDCP.
Includes comprehensive evaluation metrics matching MultiDCP diagnostics.
Logs all metrics to Weights & Biases (if available and logged in).

Usage:
    python train_biolord_pdgrapher.py --test_cell A375 --fold 1 --gpu 6
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
    """
    Initialize wandb and start a run. Returns True if successful.
    Prints the run URL prominently.
    """
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
        print("  To install: pip install wandb")
        print("=" * 70 + "\n")
        return False
    
    try:
        api_key = os.environ.get('WANDB_API_KEY') or wb.api.api_key
        if not api_key:
            print("ERROR: wandb API key not found.")
            print("  To login, run: wandb login")
            print("=" * 70 + "\n")
            return False
        print("API key found ✓")
    except Exception as e:
        print(f"ERROR checking API key: {e}")
        print("  To login, run: wandb login")
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
        print("*" + " " * 68 + "*")
        print("*  WANDB RUN STARTED SUCCESSFULLY!" + " " * 33 + "*")
        print("*" + " " * 68 + "*")
        print(f"*  Project: {project:<56} *")
        print(f"*  Run Name: {name:<55} *")
        print("*" + " " * 68 + "*")
        print("*  LIVE DASHBOARD:" + " " * 49 + "*")
        print(f"*  {run_url:<66} *")
        print("*" + " " * 68 + "*")
        print("*" * 70 + "\n")
        
        wandb.log({"_init_test": 1.0}, step=0)
        
        return True
        
    except Exception as e:
        print(f"ERROR: wandb.init() failed: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70 + "\n")
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
            elif isinstance(v, str):
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
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def compute_topk_by_treated_values(treated_values, k=80):
    """Select top-k genes by highest |treated| expression values per sample."""
    n_samples, n_genes = treated_values.shape
    top_indices = np.argsort(np.abs(treated_values), axis=1)[:, ::-1][:, :k]
    return top_indices


def compute_global_metrics(true_vals, pred_vals):
    """Compute metrics over ALL values (flattened across samples and genes)."""
    true_flat = true_vals.ravel()
    pred_flat = pred_vals.ravel()
    
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


def compute_persample_metrics(true_vals, pred_vals, gene_indices=None):
    """Compute metrics per sample, then aggregate."""
    n_samples = true_vals.shape[0]
    
    pearson_vals = np.full(n_samples, np.nan)
    spearman_vals = np.full(n_samples, np.nan)
    r2_true_vals = np.full(n_samples, np.nan)
    r2_pearson_vals = np.full(n_samples, np.nan)
    rmse_vals = np.full(n_samples, np.nan)
    var_true = np.full(n_samples, np.nan)
    var_pred = np.full(n_samples, np.nan)
    
    for i in range(n_samples):
        if gene_indices is not None:
            idx = gene_indices[i]
            true_sub = true_vals[i, idx]
            pred_sub = pred_vals[i, idx]
        else:
            true_sub = true_vals[i, :]
            pred_sub = pred_vals[i, :]
        
        mask = np.isfinite(true_sub) & np.isfinite(pred_sub)
        true_sub = true_sub[mask]
        pred_sub = pred_sub[mask]
        
        if len(true_sub) < 2:
            continue
        
        std_true = np.std(true_sub)
        std_pred = np.std(pred_sub)
        var_true[i] = std_true ** 2
        var_pred[i] = std_pred ** 2
        
        if std_true == 0 or std_pred == 0:
            continue
        
        r = pearsonr(true_sub, pred_sub).statistic
        if np.isfinite(r):
            pearson_vals[i] = r
            r2_pearson_vals[i] = r ** 2
        
        rho = spearmanr(true_sub, pred_sub).correlation
        if np.isfinite(rho):
            spearman_vals[i] = rho
        
        ss_res = np.sum((true_sub - pred_sub) ** 2)
        ss_tot = np.sum((true_sub - np.mean(true_sub)) ** 2)
        if ss_tot > 0:
            r2_true_vals[i] = 1 - (ss_res / ss_tot)
        
        rmse_vals[i] = np.sqrt(np.mean((true_sub - pred_sub) ** 2))
    
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
        'n_valid': np.sum(np.isfinite(pearson_vals)),
        '_raw_pearson': pearson_vals,
        '_raw_r2_true': r2_true_vals,
        '_raw_r2_pearson': r2_pearson_vals,
    }


def compute_prediction_bias(true_vals, pred_vals):
    """Check for systematic prediction biases."""
    errors = pred_vals - true_vals
    true_flat = true_vals.ravel()
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


def run_diagnostics(true_vals, pred_vals, split_name, verbose=True):
    """Run comprehensive diagnostics and return summary dict."""
    results = {}
    
    top_80_idx = compute_topk_by_treated_values(true_vals, k=80)
    
    global_metrics = compute_global_metrics(true_vals, pred_vals)
    results['global_pearson'] = global_metrics['pearson']
    results['global_spearman'] = global_metrics['spearman']
    results['global_r2_true'] = global_metrics['r2']
    results['global_r2_from_pearson'] = global_metrics['r2_from_pearson']
    results['global_rmse'] = global_metrics['rmse']
    
    allgenes_metrics = compute_persample_metrics(true_vals, pred_vals, gene_indices=None)
    results['allgenes_pearson_mean'] = allgenes_metrics['pearson_mean']
    results['allgenes_pearson_median'] = allgenes_metrics['pearson_median']
    results['allgenes_spearman_mean'] = allgenes_metrics['spearman_mean']
    results['allgenes_r2_true_mean'] = allgenes_metrics['r2_true_mean']
    results['allgenes_r2_from_pearson_mean'] = allgenes_metrics['r2_from_pearson_mean']
    results['allgenes_rmse_mean'] = allgenes_metrics['rmse_mean']
    
    for k in [20, 40, 80]:
        top_k_idx = top_80_idx[:, :k]
        topk_metrics = compute_persample_metrics(true_vals, pred_vals, gene_indices=top_k_idx)
        results[f'top{k}_pearson_mean'] = topk_metrics['pearson_mean']
        results[f'top{k}_pearson_median'] = topk_metrics['pearson_median']
        results[f'top{k}_spearman_mean'] = topk_metrics['spearman_mean']
        results[f'top{k}_r2_true_mean'] = topk_metrics['r2_true_mean']
        results[f'top{k}_r2_from_pearson_mean'] = topk_metrics['r2_from_pearson_mean']
        results[f'top{k}_rmse_mean'] = topk_metrics['rmse_mean']
        results[f'top{k}_var_true_mean'] = topk_metrics['var_true_mean']
    
    bias = compute_prediction_bias(true_vals, pred_vals)
    results['prediction_bias'] = bias['mean_error']
    results['high_value_bias'] = bias['high_value_bias']
    results['low_value_bias'] = bias['low_value_bias']
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"DIAGNOSTICS: {split_name.upper()}")
        print(f"{'='*70}")
        print(f"{'Metric':<40} {'Value':>12}")
        print(f"{'-'*55}")
        print(f"{'Global Pearson':<40} {results['global_pearson']:>12.4f}")
        print(f"{'Global True R²':<40} {results['global_r2_true']:>12.4f}")
        print(f"{'Global R² (Pearson²)':<40} {results['global_r2_from_pearson']:>12.4f}")
        print(f"{'-'*55}")
        print(f"{'All-genes Pearson (per-sample mean)':<40} {results['allgenes_pearson_mean']:>12.4f}")
        print(f"{'All-genes True R² (per-sample mean)':<40} {results['allgenes_r2_true_mean']:>12.4f}")
        print(f"{'All-genes R² from Pearson (mean)':<40} {results['allgenes_r2_from_pearson_mean']:>12.4f}")
        print(f"{'-'*55}")
        for k in [20, 40, 80]:
            print(f"{'Top-{} Pearson (per-sample mean)':<40} {results[f'top{k}_pearson_mean']:>12.4f}".format(k))
            print(f"{'Top-{} R² from Pearson (mean)':<40} {results[f'top{k}_r2_from_pearson_mean']:>12.4f}".format(k))
            print(f"{'Top-{} True R² (per-sample mean)':<40} {results[f'top{k}_r2_true_mean']:>12.4f}".format(k))
        print(f"{'-'*55}")
        print(f"{'Prediction Bias (mean error)':<40} {results['prediction_bias']:>12.4f}")
        print(f"{'High-value Bias (>90th %ile)':<40} {results['high_value_bias']:>12.4f}")
        print(f"{'Low-value Bias (<10th %ile)':<40} {results['low_value_bias']:>12.4f}")
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
    
    # Method 1: Try compute_prediction_adata (biolord specific)
    try:
        pred_adata = model.compute_prediction_adata(adata=adata_subset)
        if hasattr(pred_adata, 'X'):
            pred_vals = pred_adata.X
            if hasattr(pred_vals, 'toarray'):
                pred_vals = pred_vals.toarray()
            print(f"  Method 1 (compute_prediction_adata) succeeded")
            return np.array(true_vals), np.array(pred_vals)
    except Exception as e:
        print(f"  Method 1 (compute_prediction_adata) failed: {e}")
    
    # Method 2: Try predict method
    try:
        pred_vals = model.predict(adata=adata_subset)
        if hasattr(pred_vals, 'X'):
            pred_vals = pred_vals.X
        if hasattr(pred_vals, 'toarray'):
            pred_vals = pred_vals.toarray()
        if hasattr(pred_vals, 'values'):
            pred_vals = pred_vals.values
        print(f"  Method 2 (predict) succeeded")
        return np.array(true_vals), np.array(pred_vals)
    except Exception as e:
        print(f"  Method 2 (predict) failed: {e}")
    
    # Method 3: Try module.get_expression directly
    try:
        import torch
        model.module.eval()
        
        scdl = model._make_data_loader(
            adata=adata_subset,
            batch_size=512,
            shuffle=False,
        )
        
        all_preds = []
        
        with torch.no_grad():
            for tensors in scdl:
                # Try get_expression method
                x = tensors['X']
                if hasattr(model.module, 'get_expression'):
                    pred = model.module.get_expression(tensors)
                    if hasattr(pred, 'cpu'):
                        pred = pred.cpu().numpy()
                    all_preds.append(pred)
        
        if all_preds:
            pred_vals = np.vstack(all_preds)
            print(f"  Method 3 (module.get_expression) succeeded")
            return np.array(true_vals), np.array(pred_vals)
    except Exception as e:
        print(f"  Method 3 (module.get_expression) failed: {e}")
    
    # Method 4: Manual forward pass through module
    try:
        import torch
        model.module.eval()
        
        scdl = model._make_data_loader(
            adata=adata_subset,
            batch_size=512,
            shuffle=False,
        )
        
        all_preds = []
        
        with torch.no_grad():
            for tensors in scdl:
                # Call forward directly
                outputs = model.module(tensors)
                
                # outputs might be a dict or tuple
                if isinstance(outputs, dict):
                    # Look for reconstruction keys
                    for key in ['px_m', 'px', 'x_rec', 'reconstruction', 'mean', 'x_hat', 'decoded']:
                        if key in outputs:
                            pred = outputs[key]
                            break
                    else:
                        print(f"    Forward output keys: {list(outputs.keys())}")
                        # Try the first tensor-like output
                        for key, val in outputs.items():
                            if hasattr(val, 'shape') and len(val.shape) == 2:
                                pred = val
                                print(f"    Using key '{key}' with shape {val.shape}")
                                break
                        else:
                            raise KeyError(f"No suitable reconstruction found in {list(outputs.keys())}")
                elif isinstance(outputs, tuple):
                    pred = outputs[0]  # Usually first element is reconstruction
                else:
                    pred = outputs
                
                if hasattr(pred, 'cpu'):
                    pred = pred.cpu().numpy()
                all_preds.append(pred)
        
        if all_preds:
            pred_vals = np.vstack(all_preds)
            print(f"  Method 4 (module.forward) succeeded")
            return np.array(true_vals), np.array(pred_vals)
    except Exception as e:
        print(f"  Method 4 (module.forward) failed: {e}")
    
    # Method 5: Use inference + generative manually
    try:
        import torch
        model.module.eval()
        
        scdl = model._make_data_loader(
            adata=adata_subset,
            batch_size=512,
            shuffle=False,
        )
        
        all_preds = []
        
        with torch.no_grad():
            for tensors in scdl:
                # Get inference input
                inference_kwargs = model.module.get_inference_input(tensors)
                
                # Run inference
                inference_outputs = model.module.inference(**inference_kwargs)
                
                # Print what we got from inference (first batch only for debug)
                if len(all_preds) == 0:
                    print(f"    Inference output keys: {list(inference_outputs.keys())}")
                
                # Run generative - need to figure out what inputs it needs
                # Usually it needs the latent z and possibly other things
                gen_kwargs = {}
                for key in ['z', 'latent', 'qz_m', 'z_mean']:
                    if key in inference_outputs:
                        gen_kwargs['z'] = inference_outputs[key]
                        break
                
                # Also pass categorical and ordered attribute info from tensors
                for key in tensors.keys():
                    if key not in ['X', 'batch_index'] and key not in gen_kwargs:
                        gen_kwargs[key] = tensors[key]
                
                generative_outputs = model.module.generative(**gen_kwargs)
                
                if len(all_preds) == 0:
                    print(f"    Generative output keys: {list(generative_outputs.keys())}")
                
                # Find the reconstruction
                for key in ['px_m', 'px', 'x_rec', 'reconstruction', 'mean', 'x_hat', 'decoded', 'y']:
                    if key in generative_outputs:
                        pred = generative_outputs[key]
                        break
                else:
                    raise KeyError(f"No reconstruction in generative outputs: {list(generative_outputs.keys())}")
                
                if hasattr(pred, 'cpu'):
                    pred = pred.cpu().numpy()
                all_preds.append(pred)
        
        if all_preds:
            pred_vals = np.vstack(all_preds)
            print(f"  Method 5 (inference+generative) succeeded")
            return np.array(true_vals), np.array(pred_vals)
    except Exception as e:
        print(f"  Method 5 (inference+generative) failed: {e}")
        import traceback
        traceback.print_exc()
    
    raise NotImplementedError(
        "Could not extract predictions from biolord model. "
        "Please check biolord version and available methods."
    )


def evaluate_model(model, adata, save_dir, test_cell, fold):
    """Evaluate biolord model on train/val/test splits and save metrics."""
    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    all_results = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\nEvaluating {split} split...")
        
        try:
            true_vals, pred_vals = get_biolord_predictions(model, adata, split=split)
            print(f"  True shape: {true_vals.shape}, Pred shape: {pred_vals.shape}")
            
            results = run_diagnostics(true_vals, pred_vals, split_name=split, verbose=True)
            
            for key, value in results.items():
                all_results[f'{split}_{key}'] = value
            
            wandb_metrics = {f'eval/{split}/{key}': value 
                           for key, value in results.items() 
                           if isinstance(value, (int, float)) and np.isfinite(value)}
            wandb_log(wandb_metrics)
                
        except Exception as e:
            print(f"  ERROR evaluating {split}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    summary_data = {
        'cell_type': test_cell,
        'fold': fold,
    }
    summary_data.update(all_results)
    
    summary_df = pd.DataFrame([summary_data])
    
    metrics_path = save_dir / f"metrics_{test_cell}_fold{fold}.csv"
    summary_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to: {metrics_path}")
    
    if WANDB_AVAILABLE and wandb is not None:
        try:
            summary_table = wandb.Table(dataframe=summary_df)
            wandb_log({"summary_table": summary_table})
        except Exception as e:
            print(f"WARNING: Could not log wandb summary table: {e}")
        
        # Log ALL key metrics to wandb summary for easy comparison
        for k in [20, 40, 80]:
            for split in ['train', 'val', 'test']:
                for metric_type, suffix in [('r2_from_pearson_mean', 'r2'), 
                                            ('r2_true_mean', 'r2_true'),
                                            ('pearson_mean', 'pearson'),
                                            ('spearman_mean', 'spearman'),
                                            ('rmse_mean', 'rmse')]:
                    key = f'{split}_top{k}_{metric_type}'
                    if key in all_results and np.isfinite(all_results[key]):
                        wandb_summary_update(f'{split}/top{k}_{suffix}', all_results[key])
        
        # Also log all-genes and global metrics to summary
        for split in ['train', 'val', 'test']:
            for metric in ['allgenes_pearson_mean', 'allgenes_r2_from_pearson_mean', 'allgenes_r2_true_mean',
                          'global_pearson', 'global_r2_from_pearson', 'global_r2_true', 'global_rmse']:
                key = f'{split}_{metric}'
                if key in all_results and np.isfinite(all_results[key]):
                    wandb_summary_update(f'{split}/{metric}', all_results[key])
        
        # Log a single comprehensive metrics dict for easy access
        final_metrics = {}
        for k in [20, 40, 80]:
            for split in ['train', 'val', 'test']:
                key = f'{split}_top{k}_r2_from_pearson_mean'
                if key in all_results and np.isfinite(all_results[key]):
                    final_metrics[f'final/{split}_top{k}_r2'] = all_results[key]
        wandb_log(final_metrics)
    
    print("\n" + "=" * 70)
    print("SUMMARY: TOP-K R² (from Pearson²) BY SPLIT")
    print("=" * 70)
    print(f"{'Metric':<30} {'Train':>12} {'Val':>12} {'Test':>12}")
    print("-" * 70)
    
    for k in [20, 40, 80]:
        train_val = all_results.get(f'train_top{k}_r2_from_pearson_mean', np.nan)
        val_val = all_results.get(f'val_top{k}_r2_from_pearson_mean', np.nan)
        test_val = all_results.get(f'test_top{k}_r2_from_pearson_mean', np.nan)
        print(f"{'Top-{} R² (Pearson²)':<30} {train_val:>12.4f} {val_val:>12.4f} {test_val:>12.4f}".format(k))
    
    print("-" * 70)
    
    for k in [20, 40, 80]:
        train_val = all_results.get(f'train_top{k}_r2_true_mean', np.nan)
        val_val = all_results.get(f'val_top{k}_r2_true_mean', np.nan)
        test_val = all_results.get(f'test_top{k}_r2_true_mean', np.nan)
        print(f"{'Top-{} True R²':<30} {train_val:>12.4f} {val_val:>12.4f} {test_val:>12.4f}".format(k))
    
    print("-" * 70)
    
    train_val = all_results.get('train_allgenes_r2_from_pearson_mean', np.nan)
    val_val = all_results.get('val_allgenes_r2_from_pearson_mean', np.nan)
    test_val = all_results.get('test_allgenes_r2_from_pearson_mean', np.nan)
    print(f"{'All-genes R² (Pearson²)':<30} {train_val:>12.4f} {val_val:>12.4f} {test_val:>12.4f}")
    
    train_val = all_results.get('train_global_r2_from_pearson', np.nan)
    val_val = all_results.get('val_global_r2_from_pearson', np.nan)
    test_val = all_results.get('test_global_r2_from_pearson', np.nan)
    print(f"{'Global R² (Pearson²)':<30} {train_val:>12.4f} {val_val:>12.4f} {test_val:>12.4f}")
    
    print("=" * 70)
    
    # Print key metrics comparison for all splits
    print("\n" + "=" * 70)
    print("KEY METRICS SUMMARY (Top-K R² from Pearson²)")
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
    print(f"{'All-genes':<8} {train_all:>15.4f} {val_all:>15.4f} {test_all:>15.4f}")
    print("=" * 70)
    
    return summary_df, all_results


def main(args):
    """Main training function."""
    
    start_time = datetime.now()
    
    DATA_PICKLE = args.data_pickle
    SPLITS_BASE_PATH = args.splits_base_path
    TEST_CELL = args.test_cell
    FOLD = args.fold
    
    SAVE_DIR = Path(args.output_dir)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb FIRST
    if args.use_wandb:
        wandb_name = f'biolord_{TEST_CELL}_fold{FOLD}_sd{args.seed}'
        wandb_ready = init_wandb_and_start_run(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args),
            tags=[TEST_CELL, f'fold{FOLD}', 'biolord'],
        )
        if not wandb_ready:
            print("Continuing without wandb logging.\n")
    else:
        print("\nwandb logging disabled by user.\n")
    
    print("=" * 70)
    print(f"BIOLORD TRAINING: {TEST_CELL} fold {FOLD}")
    print("=" * 70)
    print(f"Data pickle: {DATA_PICKLE}")
    print(f"Splits base path: {SPLITS_BASE_PATH}")
    print(f"Output directory: {SAVE_DIR}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 70)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
        print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
        wandb_config_update({'gpu_name': gpu_name, 'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9})
    else:
        print("WARNING: No GPU available, using CPU")
    
    # Load data
    print("\n--- Loading data ---")
    treated_df = pd.read_pickle(DATA_PICKLE)
    print(f"Treated data shape: {treated_df.shape}")
    
    # Identify columns
    METADATA_COLS = ['cell_id', 'pert_id', 'pert_idose', 'idx', 'sig_id', 'pert_type', 'pert_iname']
    metadata_cols = [col for col in METADATA_COLS if col in treated_df.columns]
    gene_cols = [col for col in treated_df.columns if col not in METADATA_COLS]
    print(f"Metadata columns found: {metadata_cols}")
    print(f"All columns in DataFrame: {[c for c in treated_df.columns if not c.startswith('ENSG')][:15]}...")  # First 15 non-gene columns
    print(f"Number of gene columns: {len(gene_cols)}")
    
    wandb_config_update({'n_genes': len(gene_cols), 'metadata_cols': metadata_cols})
    
    # Filter to cell type
    print(f"\n--- Filtering to cell type: {TEST_CELL} ---")
    df = treated_df[treated_df['cell_id'] == TEST_CELL].copy()
    print(f"Filtered data shape: {df.shape}")
    
    if len(df) == 0:
        raise ValueError(f"No data for {TEST_CELL}. Available: {treated_df['cell_id'].unique().tolist()}")
    
    # Load split file
    print(f"\n--- Loading split file ---")
    splits_path = Path(SPLITS_BASE_PATH) / TEST_CELL / "random" / "5fold" / "splits.pt"
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
    
    # Create split labels
    # The split indices are ROW INDICES (positions), not values in a column
    # Reset index to ensure we have clean integer indices
    df = df.reset_index(drop=True)
    
    df['split'] = 'unknown'
    
    # Check if 'idx' column exists, otherwise use row positions
    if 'idx' in df.columns:
        print("Using 'idx' column for split assignment")
        df.loc[df['idx'].isin(train_idx_set), 'split'] = 'train'
        df.loc[df['idx'].isin(val_idx_set), 'split'] = 'val'
        df.loc[df['idx'].isin(test_idx_set), 'split'] = 'test'
    else:
        print("Using row indices for split assignment (no 'idx' column found)")
        # The indices from split file correspond to row positions in the cell-type filtered data
        # But we need to check if they're global indices or local indices
        
        # First, let's see what range the indices are in
        all_indices = train_idx_set | val_idx_set | test_idx_set
        max_idx = max(all_indices)
        min_idx = min(all_indices)
        print(f"  Index range in split file: {min_idx} to {max_idx}")
        print(f"  DataFrame row count: {len(df)}")
        
        # If max index > len(df), indices are likely global (from full dataset)
        # In that case, we need a different approach
        if max_idx >= len(df):
            print("  WARNING: Split indices exceed DataFrame size - these may be global indices")
            print("  Attempting to match by position in original order...")
            # Create a mapping from the original filtered indices
            # The split file indices likely refer to the full cell-type dataset order
            for i, row_idx in enumerate(df.index):
                if i in train_idx_set:
                    df.loc[row_idx, 'split'] = 'train'
                elif i in val_idx_set:
                    df.loc[row_idx, 'split'] = 'val'
                elif i in test_idx_set:
                    df.loc[row_idx, 'split'] = 'test'
        else:
            # Indices are within range, use them directly as row positions
            for idx in train_idx_set:
                if idx < len(df):
                    df.iloc[idx, df.columns.get_loc('split')] = 'train'
            for idx in val_idx_set:
                if idx < len(df):
                    df.iloc[idx, df.columns.get_loc('split')] = 'val'
            for idx in test_idx_set:
                if idx < len(df):
                    df.iloc[idx, df.columns.get_loc('split')] = 'test'
    
    print(f"\nSplit distribution:\n{df['split'].value_counts()}")
    
    n_unknown = (df['split'] == 'unknown').sum()
    if n_unknown > 0:
        print(f"WARNING: {n_unknown} samples could not be assigned to a split")
        df = df[df['split'] != 'unknown'].copy()
        print(f"Final sample count after removing unassigned: {len(df)}")
    
    split_counts = df['split'].value_counts().to_dict()
    wandb_config_update({
        'n_samples_total': len(df),
        'n_samples_train': split_counts.get('train', 0),
        'n_samples_val': split_counts.get('val', 0),
        'n_samples_test': split_counts.get('test', 0),
        'n_perturbations': df['pert_id'].nunique(),
    })
    
    wandb_log({
        'data/n_samples_total': len(df),
        'data/n_samples_train': split_counts.get('train', 0),
        'data/n_samples_val': split_counts.get('val', 0),
        'data/n_samples_test': split_counts.get('test', 0),
    })
    
    # Create AnnData
    print("\n--- Creating AnnData ---")
    X = df[gene_cols].values.astype(np.float32)
    print(f"Gene expression matrix shape: {X.shape}")
    
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    if n_nan > 0 or n_inf > 0:
        print(f"WARNING: Found {n_nan} NaN and {n_inf} Inf values")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    obs_dict = {
        'cell_id': df['cell_id'].astype('category').values,
        'pert_id': df['pert_id'].astype('category').values,
        'split': pd.Categorical(df['split'].values),
    }
    
    if 'pert_idose' in df.columns:
        dose_str = df['pert_idose'].astype(str)
        dose_numeric = pd.to_numeric(dose_str.str.extract(r'([\d.]+)')[0], errors='coerce').fillna(0)
        obs_dict['dose'] = dose_numeric.values
    
    # Store original index if available, otherwise use row number
    if 'idx' in df.columns:
        obs_dict['original_idx'] = df['idx'].values
    else:
        obs_dict['original_idx'] = np.arange(len(df))
    obs = pd.DataFrame(obs_dict).reset_index(drop=True)
    
    adata = sc.AnnData(X=X, obs=obs)
    adata.var_names = gene_cols
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    
    print(f"AnnData created: {adata.shape}")
    print(f"\nSplit in AnnData:\n{adata.obs['split'].value_counts()}")
    
    # Setup Biolord
    print("\n--- Setting up Biolord ---")
    categorical_attributes = ['pert_id']
    ordered_attributes = []
    if 'dose' in adata.obs.columns and adata.obs['dose'].nunique() > 1:
        ordered_attributes.append('dose')
    
    print(f"Categorical attributes: {categorical_attributes}")
    print(f"  pert_id has {adata.obs['pert_id'].nunique()} unique values")
    print(f"Ordered attributes: {ordered_attributes}")
    
    for col in categorical_attributes:
        adata.obs[col] = adata.obs[col].astype('category')
    
    biolord.Biolord.setup_anndata(
        adata,
        ordered_attributes_keys=ordered_attributes if ordered_attributes else None,
        categorical_attributes_keys=categorical_attributes,
    )
    
    # Create model
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
    wandb_config_update({'module_params': module_params, 'n_latent': args.n_latent})
    
    # Train
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
    
    # Use WandbLogger for live metrics
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
    
    # Log training history
    epoch_history = None
    if hasattr(model, 'training_plan') and hasattr(model.training_plan, 'epoch_history'):
        epoch_history = pd.DataFrame.from_dict(model.training_plan.epoch_history)
        model.epoch_history = epoch_history
        
        print("\nTraining metrics (last 5 validation epochs):")
        val_history = epoch_history[epoch_history["mode"] == "valid"]
        print(val_history.tail())
        
        # Note: WandbLogger already logs live training metrics, so we skip manual history logging
        # to avoid step conflicts. The history is still saved to CSV below.
    
    # Evaluate
    # First, let's see what methods the model has for getting predictions
    print("\n--- Checking model prediction methods ---")
    model_methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m, None))]
    prediction_methods = [m for m in model_methods if any(x in m.lower() for x in ['predict', 'reconstruct', 'express', 'sample', 'generate', 'decode'])]
    print(f"Potentially relevant model methods: {prediction_methods}")
    
    # Check module methods too
    if hasattr(model, 'module'):
        module_methods = [m for m in dir(model.module) if not m.startswith('_') and callable(getattr(model.module, m, None))]
        module_pred_methods = [m for m in module_methods if any(x in m.lower() for x in ['predict', 'reconstruct', 'express', 'sample', 'generate', 'decode', 'forward', 'inference', 'generative'])]
        print(f"Potentially relevant module methods: {module_pred_methods}")
    
    summary_df, all_results = evaluate_model(model, adata, SAVE_DIR, TEST_CELL, FOLD)
    
    # Save
    print("\n--- Saving outputs ---")
    model_save_path = SAVE_DIR / f"biolord_{TEST_CELL}_fold{FOLD}"
    model.save(str(model_save_path), overwrite=True)
    print(f"Model saved to: {model_save_path}")
    
    adata_save_path = SAVE_DIR / f"adata_{TEST_CELL}_fold{FOLD}.h5ad"
    adata.write(adata_save_path)
    print(f"AnnData saved to: {adata_save_path}")
    
    if epoch_history is not None:
        history_save_path = SAVE_DIR / f"history_{TEST_CELL}_fold{FOLD}.csv"
        epoch_history.to_csv(history_save_path, index=False)
        print(f"Training history saved to: {history_save_path}")
    
    # Summary
    end_time = datetime.now()
    runtime = end_time - start_time
    
    print("\n" + "=" * 70)
    print(f"COMPLETED: {TEST_CELL} fold {FOLD}")
    print("=" * 70)
    print(f"Total runtime: {runtime}")
    print(f"Output directory: {SAVE_DIR}")
    print(f"wandb logging: {'enabled' if WANDB_AVAILABLE else 'disabled'}")
    print("=" * 70)
    
    wandb_summary_update('runtime_seconds', runtime.total_seconds())
    wandb_summary_update('runtime_str', str(runtime))
    
    if args.save_wandb_artifact and WANDB_AVAILABLE:
        try:
            artifact = wandb.Artifact(
                name=f"biolord_{TEST_CELL}_fold{FOLD}",
                type="model",
            )
            artifact.add_dir(str(model_save_path))
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"WARNING: Could not save wandb artifact: {e}")
    
    wandb_finish()
    
    return model, adata, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Biolord training for PDGrapher data')
    
    parser.add_argument('--data_pickle', type=str,
                        default='/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_brddrugfiltered.pkl',
                        help='Path to treated data pickle')
    parser.add_argument('--splits_base_path', type=str,
                        default='/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical',
                        help='Base path to splits directory')
    parser.add_argument('--output_dir', type=str,
                        default='/raid/home/joshua/projects/biolord_comp_122025',
                        help='Output directory')
    
    parser.add_argument('--test_cell', type=str, required=True,
                        help='Cell type to train on (e.g., A375, MCF7, PC3)')
    parser.add_argument('--fold', type=int, default=1,
                        help='Which fold to use (1-5)')
    parser.add_argument('--gpu', type=str, default=None,
                        help='GPU device ID')
    
    parser.add_argument('--use_wandb', action='store_true', default=True,
                        help='Use wandb logging (default: True)')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='biolord_pdgrapher',
                        help='Wandb project name')
    parser.add_argument('--save_wandb_artifact', action='store_true', default=False,
                        help='Save model as wandb artifact')
    
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    parser.add_argument('--n_latent', type=int, default=32)
    parser.add_argument('--decoder_width', type=int, default=1024)
    parser.add_argument('--decoder_depth', type=int, default=4)
    parser.add_argument('--attribute_nn_width', type=int, default=512)
    parser.add_argument('--attribute_nn_depth', type=int, default=2)
    parser.add_argument('--n_latent_attribute_categorical', type=int, default=4)
    
    parser.add_argument('--reconstruction_penalty', type=float, default=1e2)
    parser.add_argument('--unknown_attribute_penalty', type=float, default=1e1)
    parser.add_argument('--unknown_attribute_noise_param', type=float, default=1e-1)
    parser.add_argument('--attribute_dropout_rate', type=float, default=0.1)
    
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