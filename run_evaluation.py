#!/usr/bin/env python
"""
Full evaluation script for all baseline models on all cell lines.
Runs ensemble of baseline models and saves results.
"""
import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import PDGrapherDataLoader, TopKEvaluator
from models.base_model import (
    NoChangeBaseline,
    MeanShiftBaseline,
    PerGeneLinearBaseline,
)
from models.cellot_wrapper import CellOTModel
from ensemble import EnsembleModel, LearnedEnsemble, StackingEnsemble


def run_cell_line_evaluation(cell_line: str, fold: int = 0, n_samples: int = 10000):
    """
    Run full evaluation on a single cell line.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating on {cell_line} (fold {fold})")
    print(f"{'='*70}")

    # Load data
    loader = PDGrapherDataLoader()
    train_idx, test_idx = loader.get_train_test_split(cell_line, fold=fold)

    # Limit samples
    if len(train_idx) > n_samples:
        train_idx = train_idx[:n_samples]
    if len(test_idx) > n_samples // 4:
        test_idx = test_idx[:n_samples // 4]

    print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    treated_train, diseased_train = loader.get_expression_arrays(train_idx)
    treated_test, diseased_test = loader.get_expression_arrays(test_idx)

    evaluator = TopKEvaluator()
    all_results = {}

    # Base models
    base_models_classes = [
        NoChangeBaseline,
        MeanShiftBaseline,
        PerGeneLinearBaseline,
    ]

    # Evaluate base models
    print("\n--- Base Models ---")
    for ModelClass in base_models_classes:
        model = ModelClass()
        print(f"Training {model.name}...")
        model.train(diseased_train, treated_train)
        pred = model.predict(diseased_test)
        results = evaluator.compute_metrics(treated_test, pred, diseased_test)
        all_results[model.name] = results
        print(f"  R² Top-20: {results['r2_top20']:.4f}, R² All: {results['r2_all']:.4f}")

    # Stacking ensemble (best performer)
    print("\n--- Stacking Ensemble ---")
    ensemble_models = [
        NoChangeBaseline(),
        MeanShiftBaseline(),
        PerGeneLinearBaseline(),
    ]
    ensemble = StackingEnsemble(ensemble_models)
    ensemble.train(diseased_train, treated_train)
    pred = ensemble.predict(diseased_test)
    results = evaluator.compute_metrics(treated_test, pred, diseased_test)
    all_results['StackingEnsemble'] = results
    print(f"  R² Top-20: {results['r2_top20']:.4f}, R² All: {results['r2_all']:.4f}")

    # CellOT (optimal transport)
    print("\n--- CellOT ---")
    try:
        cellot = CellOTModel(n_epochs=100, n_components=100)
        cellot.train(diseased_train, treated_train)
        pred = cellot.predict(diseased_test)
        results = evaluator.compute_metrics(treated_test, pred, diseased_test)
        all_results['CellOT'] = results
        print(f"  R² Top-20: {results['r2_top20']:.4f}, R² All: {results['r2_all']:.4f}")
    except Exception as e:
        print(f"  CellOT failed: {e}")

    return all_results


def run_full_evaluation(n_samples: int = 10000, n_folds: int = 5, output_dir: str = "results"):
    """
    Run evaluation across all cell lines and folds.
    """
    os.makedirs(output_dir, exist_ok=True)

    cell_lines = PDGrapherDataLoader.CELL_LINES
    all_results = {}

    for cell_line in cell_lines:
        cell_results = {}
        for fold in range(n_folds):
            try:
                fold_results = run_cell_line_evaluation(cell_line, fold, n_samples)
                cell_results[f"fold_{fold}"] = fold_results
            except Exception as e:
                print(f"Error on {cell_line} fold {fold}: {e}")
                continue
        all_results[cell_line] = cell_results

    # Compute average across folds
    summary = {}
    for cell_line, cell_results in all_results.items():
        summary[cell_line] = {}
        model_names = list(cell_results.get('fold_0', {}).keys())

        for model_name in model_names:
            metrics = {}
            for metric in ['r2_top20', 'r2_top40', 'r2_top80', 'r2_all']:
                values = []
                for fold_key, fold_results in cell_results.items():
                    if model_name in fold_results:
                        values.append(fold_results[model_name][metric])
                if values:
                    metrics[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values))
                    }
            summary[cell_line][model_name] = metrics

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'raw_results': all_results,
            'summary': summary,
            'config': {
                'n_samples': n_samples,
                'n_folds': n_folds,
                'cell_lines': cell_lines
            }
        }, f, indent=2)

    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    # Print summary table
    print(f"\n{'Cell Line':<12} {'Model':<20} {'R² Top20':>12} {'R² Top40':>12} {'R² Top80':>12} {'R² All':>12}")
    print("-" * 80)

    for cell_line in cell_lines:
        if cell_line not in summary:
            continue
        for model_name in ['NoChange', 'PerGeneLinear', 'StackingEnsemble', 'CellOT']:
            if model_name not in summary[cell_line]:
                continue
            m = summary[cell_line][model_name]
            print(f"{cell_line:<12} {model_name:<20} "
                  f"{m['r2_top20']['mean']:>10.4f}±{m['r2_top20']['std']:>5.3f} "
                  f"{m['r2_top40']['mean']:>10.4f}±{m['r2_top40']['std']:>5.3f} "
                  f"{m['r2_top80']['mean']:>10.4f}±{m['r2_top80']['std']:>5.3f} "
                  f"{m['r2_all']['mean']:>10.4f}±{m['r2_all']['std']:>5.3f}")

    print(f"\nResults saved to: {results_path}")
    return summary


def quick_evaluation(cell_line: str = "A549"):
    """Quick evaluation on a single cell line for testing."""
    results = run_cell_line_evaluation(cell_line, fold=0, n_samples=5000)

    print(f"\n{'='*60}")
    print("Quick Evaluation Summary")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'R² Top20':>10} {'R² Top40':>10} {'R² Top80':>10} {'R² All':>10}")
    print("-" * 60)
    for model_name, r in results.items():
        print(f"{model_name:<20} {r['r2_top20']:>10.4f} {r['r2_top40']:>10.4f} "
              f"{r['r2_top80']:>10.4f} {r['r2_all']:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baseline model evaluation")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="Evaluation mode: quick (single cell line) or full (all cell lines)")
    parser.add_argument("--cell_line", default="A549",
                        help="Cell line for quick evaluation")
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Max samples per cell line")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Number of folds for full evaluation")
    parser.add_argument("--output_dir", default="results",
                        help="Output directory for results")

    args = parser.parse_args()

    if args.mode == "quick":
        quick_evaluation(args.cell_line)
    else:
        run_full_evaluation(args.n_samples, args.n_folds, args.output_dir)
