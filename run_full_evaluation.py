#!/usr/bin/env python
"""
Comprehensive evaluation script for all models on a single cell line.
Supports GPU selection for models that need it.

Usage:
    python run_full_evaluation.py --cell_line A549 --gpu 1 --fold 0
    python run_full_evaluation.py --cell_line A549 --all_folds --gpu 1
"""
import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj

# Parse GPU argument early to set CUDA_VISIBLE_DEVICES before importing torch
def get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == '--gpu' and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "0"

gpu_arg = get_gpu_arg()
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_arg
print(f"Setting CUDA_VISIBLE_DEVICES={gpu_arg}")

from data_loader import PDGrapherDataLoader, TopKEvaluator
from models.base_model import (
    NoChangeBaseline,
    MeanShiftBaseline,
    PerGeneLinearBaseline,
)

# Track which models successfully import
AVAILABLE_MODELS = {
    'baselines': True,
    'cellot': False,
    'scgen': False,
    'chemcpa': False,
    'biolord': False,
    'multidcp': False,
}

# Try importing optional models
try:
    from models.cellot_wrapper import CellOTModel
    AVAILABLE_MODELS['cellot'] = True
except ImportError as e:
    print(f"CellOT not available: {e}")

try:
    from models.scgen_wrapper import ScGenModel
    AVAILABLE_MODELS['scgen'] = True
except ImportError as e:
    print(f"scGen not available: {e}")

try:
    from models.chemcpa_wrapper import ChemCPAModel
    AVAILABLE_MODELS['chemcpa'] = True
except ImportError as e:
    print(f"ChemCPA not available: {e}")

try:
    from models.biolord_wrapper import BiolordModel
    AVAILABLE_MODELS['biolord'] = True
except ImportError as e:
    print(f"Biolord not available: {e}")

try:
    from models.multidcp_wrapper import MultiDCPModel
    AVAILABLE_MODELS['multidcp'] = True
except ImportError as e:
    print(f"MultiDCP not available: {e}")


def evaluate_single_fold(cell_line: str, fold: int, n_samples: int = None):
    """
    Run full evaluation on a single cell line and fold.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {cell_line} - Fold {fold}")
    print(f"{'='*70}")

    # Load data
    loader = PDGrapherDataLoader()
    train_idx, test_idx = loader.get_train_test_split(cell_line, fold=fold)

    # Optionally limit samples
    if n_samples:
        if len(train_idx) > n_samples:
            train_idx = train_idx[:n_samples]
        if len(test_idx) > n_samples // 4:
            test_idx = test_idx[:n_samples // 4]

    print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    treated_train, diseased_train = loader.get_expression_arrays(train_idx)
    treated_test, diseased_test = loader.get_expression_arrays(test_idx)

    # Get metadata for models that need it
    train_df = loader.treated_df.iloc[train_idx]
    test_df = loader.treated_df.iloc[test_idx]

    train_metadata = {
        'smiles': train_df['smiles'].tolist() if 'smiles' in train_df.columns else None,
        'drug_names': train_df['pert_id'].tolist() if 'pert_id' in train_df.columns else None,
        'dose': train_df['pert_idose'].tolist() if 'pert_idose' in train_df.columns else None,
    }
    test_metadata = {
        'smiles': test_df['smiles'].tolist() if 'smiles' in test_df.columns else None,
        'drug_names': test_df['pert_id'].tolist() if 'pert_id' in test_df.columns else None,
        'dose': test_df['pert_idose'].tolist() if 'pert_idose' in test_df.columns else None,
    }

    evaluator = TopKEvaluator()
    all_results = {}

    # ========== Base Models ==========
    print("\n--- Base Models ---")
    base_models = [
        ('NoChange', NoChangeBaseline()),
        ('MeanShift', MeanShiftBaseline()),
        ('PerGeneLinear', PerGeneLinearBaseline()),
    ]

    for name, model in base_models:
        try:
            print(f"  Training {name}...")
            model.train(diseased_train, treated_train)
            pred = model.predict(diseased_test)
            results = evaluator.compute_metrics(treated_test, pred, diseased_test)
            all_results[name] = results
            print(f"    R² Top-20: {results['r2_top20']:.4f}, Pearson: {results['pearson_top20']:.4f}")
        except Exception as e:
            print(f"    {name} failed: {e}")

    # ========== CellOT ==========
    if AVAILABLE_MODELS['cellot']:
        print("\n--- CellOT ---")
        try:
            model = CellOTModel(n_epochs=100, n_components=100)
            print("  Training CellOT (PCA100)...")
            model.train(diseased_train, treated_train)
            pred = model.predict(diseased_test)
            results = evaluator.compute_metrics(treated_test, pred, diseased_test)
            all_results['CellOT'] = results
            print(f"    R² Top-20: {results['r2_top20']:.4f}, Pearson: {results['pearson_top20']:.4f}")
        except Exception as e:
            print(f"    CellOT failed: {e}")

    # ========== scGen ==========
    if AVAILABLE_MODELS['scgen']:
        print("\n--- scGen ---")
        try:
            model = ScGenModel()
            print("  Training scGen...")
            model.train(diseased_train, treated_train)
            pred = model.predict(diseased_test)
            results = evaluator.compute_metrics(treated_test, pred, diseased_test)
            all_results['scGen'] = results
            print(f"    R² Top-20: {results['r2_top20']:.4f}, Pearson: {results['pearson_top20']:.4f}")
        except Exception as e:
            print(f"    scGen failed: {e}")

    # ========== ChemCPA ==========
    if AVAILABLE_MODELS['chemcpa']:
        print("\n--- ChemCPA ---")
        try:
            model = ChemCPAModel()
            print("  Training ChemCPA...")
            model.train(diseased_train, treated_train, metadata=train_metadata)
            pred = model.predict(diseased_test, metadata=test_metadata)
            results = evaluator.compute_metrics(treated_test, pred, diseased_test)
            all_results['ChemCPA'] = results
            print(f"    R² Top-20: {results['r2_top20']:.4f}, Pearson: {results['pearson_top20']:.4f}")
        except Exception as e:
            print(f"    ChemCPA failed: {e}")

    # ========== Biolord ==========
    if AVAILABLE_MODELS['biolord']:
        print("\n--- Biolord ---")
        try:
            model = BiolordModel()
            print("  Training Biolord...")
            model.train(diseased_train, treated_train, metadata=train_metadata)
            pred = model.predict(diseased_test, metadata=test_metadata)
            results = evaluator.compute_metrics(treated_test, pred, diseased_test)
            all_results['Biolord'] = results
            print(f"    R² Top-20: {results['r2_top20']:.4f}, Pearson: {results['pearson_top20']:.4f}")
        except Exception as e:
            print(f"    Biolord failed: {e}")

    # ========== MultiDCP ==========
    if AVAILABLE_MODELS['multidcp']:
        print("\n--- MultiDCP ---")
        try:
            model = MultiDCPModel()
            print("  Loading MultiDCP (pretrained)...")
            model.train(diseased_train, treated_train, metadata=train_metadata)
            pred = model.predict(diseased_test, metadata=test_metadata)
            results = evaluator.compute_metrics(treated_test, pred, diseased_test)
            all_results['MultiDCP'] = results
            print(f"    R² Top-20: {results['r2_top20']:.4f}, Pearson: {results['pearson_top20']:.4f}")
        except Exception as e:
            print(f"    MultiDCP failed: {e}")

    return all_results


def run_evaluation(cell_line: str, folds: list, n_samples: int = None, output_dir: str = "results"):
    """
    Run evaluation across specified folds for a cell line.
    """
    os.makedirs(output_dir, exist_ok=True)

    fold_results = {}
    for fold in folds:
        try:
            fold_results[f"fold_{fold}"] = evaluate_single_fold(cell_line, fold, n_samples)
        except Exception as e:
            print(f"Error on fold {fold}: {e}")
            import traceback
            traceback.print_exc()

    # Compute summary statistics
    summary = {}
    if fold_results:
        model_names = list(fold_results.get('fold_0', fold_results[list(fold_results.keys())[0]]).keys())

        for model_name in model_names:
            metrics = {}
            for metric in ['r2_top20', 'r2_top40', 'r2_top80', 'r2_all', 'pearson_top20', 'pearson_all']:
                values = []
                for fold_key, fold_data in fold_results.items():
                    if model_name in fold_data and metric in fold_data[model_name]:
                        values.append(fold_data[model_name][metric])
                if values:
                    metrics[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'values': [float(v) for v in values]
                    }
            summary[model_name] = metrics

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"results_{cell_line}_{timestamp}.json")

    output_data = {
        'cell_line': cell_line,
        'folds': folds,
        'fold_results': fold_results,
        'summary': summary,
        'available_models': AVAILABLE_MODELS,
        'timestamp': timestamp
    }

    with open(results_path, 'w') as f:
        json.dump(convert_to_native(output_data), f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY: {cell_line}")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'R² Top20':>12} {'R² Top40':>12} {'R² Top80':>12} {'Pearson':>12}")
    print("-" * 70)

    for model_name in sorted(summary.keys()):
        m = summary[model_name]
        r2_20 = m.get('r2_top20', {})
        r2_40 = m.get('r2_top40', {})
        r2_80 = m.get('r2_top80', {})
        pear = m.get('pearson_top20', {})

        print(f"{model_name:<20} "
              f"{r2_20.get('mean', 0):>10.4f}±{r2_20.get('std', 0):>4.3f} "
              f"{r2_40.get('mean', 0):>10.4f}±{r2_40.get('std', 0):>4.3f} "
              f"{r2_80.get('mean', 0):>10.4f}±{r2_80.get('std', 0):>4.3f} "
              f"{pear.get('mean', 0):>10.4f}±{pear.get('std', 0):>4.3f}")

    print(f"\nResults saved to: {results_path}")
    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comprehensive model evaluation")
    parser.add_argument("--cell_line", required=True, help="Cell line to evaluate")
    parser.add_argument("--fold", type=int, default=0, help="Single fold to evaluate")
    parser.add_argument("--all_folds", action="store_true", help="Evaluate all 5 folds")
    parser.add_argument("--gpu", default="0", help="GPU device to use")
    parser.add_argument("--n_samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--output_dir", default="results", help="Output directory")

    args = parser.parse_args()

    if args.all_folds:
        folds = [0, 1, 2, 3, 4]
    else:
        folds = [args.fold]

    print(f"\n{'#'*80}")
    print(f"# Comprehensive Model Evaluation")
    print(f"# Cell line: {args.cell_line}")
    print(f"# Folds: {folds}")
    print(f"# GPU: {args.gpu}")
    print(f"{'#'*80}")

    run_evaluation(args.cell_line, folds, args.n_samples, args.output_dir)
