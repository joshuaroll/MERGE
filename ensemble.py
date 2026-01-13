#!/usr/bin/env python
"""
Ensemble model for PDGrapher baseline evaluation.
Combines predictions from multiple models using various strategies.
"""
import os
import sys
import numpy as np
from typing import List, Dict, Optional
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

from data_loader import PDGrapherDataLoader, TopKEvaluator
from models.base_model import (
    BasePerturbationModel,
    NoChangeBaseline,
    MeanShiftBaseline,
    PerGeneLinearBaseline,
)

# Try to import ChemCPA - may not be available in all environments
try:
    from models.chemcpa_wrapper import ChemCPAModel
    CHEMCPA_AVAILABLE = True
except ImportError:
    CHEMCPA_AVAILABLE = False

# Try to import CellOT
try:
    from models.cellot_wrapper import CellOTModel
    CELLOT_AVAILABLE = True
except ImportError:
    CELLOT_AVAILABLE = False

# Try to import scGen
try:
    from models.scgen_wrapper import ScGenModel
    SCGEN_AVAILABLE = True
except ImportError:
    SCGEN_AVAILABLE = False


class EnsembleModel(BasePerturbationModel):
    """
    Ensemble model that combines predictions from multiple base models.
    """

    def __init__(self, models: List[BasePerturbationModel], weights: Optional[List[float]] = None):
        """
        Initialize ensemble.

        Args:
            models: List of base models
            weights: Optional weights for each model (default: equal weights)
        """
        super().__init__()
        self.models = models
        self.weights = weights if weights else [1.0 / len(models)] * len(models)

    @property
    def name(self) -> str:
        return "Ensemble"

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """Train all base models."""
        for model in self.models:
            print(f"  Training {model.name}...")
            model.train(diseased, treated, metadata)
        self._trained = True

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """Combine predictions from all models using weighted average."""
        if not self._trained:
            raise RuntimeError("Ensemble must be trained first")

        predictions = []
        for model in self.models:
            pred = model.predict(diseased, metadata)
            predictions.append(pred)

        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred

        return weighted_pred


class LearnedEnsemble(EnsembleModel):
    """
    Ensemble with learned per-gene weights based on validation performance.
    """

    def __init__(self, models: List[BasePerturbationModel]):
        super().__init__(models)
        self._gene_weights = None

    @property
    def name(self) -> str:
        return "LearnedEnsemble"

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """Train base models and learn per-gene weights."""
        n_samples, n_genes = diseased.shape
        n_models = len(self.models)

        # Split into train/val for weight learning
        val_size = min(500, n_samples // 5)
        train_diseased = diseased[:-val_size]
        train_treated = treated[:-val_size]
        val_diseased = diseased[-val_size:]
        val_treated = treated[-val_size:]

        # Train all models
        for model in self.models:
            print(f"  Training {model.name}...")
            model.train(train_diseased, train_treated, metadata)

        # Get validation predictions from each model
        val_preds = []
        for model in self.models:
            pred = model.predict(val_diseased, metadata)
            val_preds.append(pred)

        # Learn per-gene weights based on validation MSE
        self._gene_weights = np.zeros((n_models, n_genes))

        for g in range(n_genes):
            mse_scores = []
            for pred in val_preds:
                mse = np.mean((pred[:, g] - val_treated[:, g]) ** 2)
                mse_scores.append(mse)

            # Convert MSE to weights (inverse MSE, normalized)
            mse_scores = np.array(mse_scores)
            # Add small epsilon to avoid division by zero
            inv_mse = 1.0 / (mse_scores + 1e-10)
            weights = inv_mse / np.sum(inv_mse)
            self._gene_weights[:, g] = weights

        self._trained = True
        print(f"  Learned per-gene weights for {n_genes} genes")

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """Combine predictions using learned per-gene weights."""
        if not self._trained:
            raise RuntimeError("Ensemble must be trained first")

        predictions = []
        for model in self.models:
            pred = model.predict(diseased, metadata)
            predictions.append(pred)

        # Per-gene weighted average
        n_samples, n_genes = diseased.shape
        weighted_pred = np.zeros((n_samples, n_genes))

        for g in range(n_genes):
            for i, pred in enumerate(predictions):
                weighted_pred[:, g] += self._gene_weights[i, g] * pred[:, g]

        return weighted_pred


class StackingEnsemble(EnsembleModel):
    """
    Stacking ensemble that uses a meta-learner to combine base predictions.
    """

    def __init__(self, models: List[BasePerturbationModel]):
        super().__init__(models)
        self._meta_weights = None
        self._meta_bias = None

    @property
    def name(self) -> str:
        return "StackingEnsemble"

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """Train base models and meta-learner."""
        n_samples, n_genes = diseased.shape
        n_models = len(self.models)

        # Split into train/val for meta-learning
        val_size = min(500, n_samples // 5)
        train_diseased = diseased[:-val_size]
        train_treated = treated[:-val_size]
        val_diseased = diseased[-val_size:]
        val_treated = treated[-val_size:]

        # Train all models
        for model in self.models:
            print(f"  Training {model.name}...")
            model.train(train_diseased, train_treated, metadata)

        # Get validation predictions
        val_preds = np.stack([model.predict(val_diseased, metadata) for model in self.models])

        # Learn meta-weights using linear regression per gene
        self._meta_weights = np.zeros((n_models, n_genes))
        self._meta_bias = np.zeros(n_genes)

        for g in range(n_genes):
            X = val_preds[:, :, g].T  # (n_val_samples, n_models)
            y = val_treated[:, g]

            # Ridge regression for stability
            lambda_reg = 0.01
            XtX = X.T @ X + lambda_reg * np.eye(n_models)
            Xty = X.T @ y

            try:
                weights = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                weights = np.ones(n_models) / n_models

            self._meta_weights[:, g] = weights
            self._meta_bias[g] = np.mean(y) - np.mean(X @ weights)

        self._trained = True
        print(f"  Trained meta-learner for {n_genes} genes")

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """Combine predictions using meta-learner."""
        if not self._trained:
            raise RuntimeError("Ensemble must be trained first")

        predictions = np.stack([model.predict(diseased, metadata) for model in self.models])

        n_samples, n_genes = diseased.shape
        weighted_pred = np.zeros((n_samples, n_genes))

        for g in range(n_genes):
            X = predictions[:, :, g].T  # (n_samples, n_models)
            weighted_pred[:, g] = X @ self._meta_weights[:, g] + self._meta_bias[g]

        return weighted_pred


def evaluate_all_models(cell_line: str = "A549", fold: int = 0, n_samples: int = 5000,
                        include_deep: bool = True):
    """
    Evaluate all baseline models and ensembles on a cell line.

    Args:
        cell_line: Cell line to evaluate
        fold: Cross-validation fold
        n_samples: Maximum training samples
        include_deep: Whether to include deep learning models (ChemCPA, CellOT)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating all models on {cell_line} (fold {fold})")
    print(f"{'='*60}")

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
    meta_train = loader.get_metadata(train_idx)
    meta_test = loader.get_metadata(test_idx)

    evaluator = TopKEvaluator()
    all_results = {}

    # Base models
    print("\n--- Base Models ---")
    base_models = [
        NoChangeBaseline(),
        MeanShiftBaseline(),
        PerGeneLinearBaseline(),
    ]

    for model in base_models:
        print(f"\nTraining {model.name}...")
        model.train(diseased_train, treated_train)
        pred = model.predict(diseased_test)

        results = evaluator.compute_metrics(treated_test, pred, diseased_test)
        evaluator.print_results(results, model.name)
        all_results[model.name] = results

    # Deep learning models (optional)
    if include_deep:
        print("\n--- Deep Learning Models ---")

        # CellOT
        if CELLOT_AVAILABLE:
            print("\nTraining CellOT...")
            try:
                cellot = CellOTModel(hidden_units=[256, 256, 256, 256], n_epochs=100, n_components=100)
                cellot.train(diseased_train, treated_train)
                pred = cellot.predict(diseased_test)
                results = evaluator.compute_metrics(treated_test, pred, diseased_test)
                evaluator.print_results(results, "CellOT")
                all_results["CellOT"] = results
            except Exception as e:
                print(f"  CellOT failed: {e}")

        # ChemCPA
        if CHEMCPA_AVAILABLE:
            print("\nTraining ChemCPA...")
            try:
                # Prepare metadata for ChemCPA
                chemcpa_meta = {
                    'smiles': meta_train['smiles'].tolist() if 'smiles' in meta_train.columns else None,
                    'cell_type': meta_train['cell_type'].tolist() if 'cell_type' in meta_train.columns else None,
                    'dose': meta_train['dose'].tolist() if 'dose' in meta_train.columns else None,
                }
                chemcpa = ChemCPAModel(latent_dim=256, n_epochs=100, batch_size=64)
                chemcpa.train(diseased_train, treated_train, chemcpa_meta)

                test_meta = {
                    'smiles': meta_test['smiles'].tolist() if 'smiles' in meta_test.columns else None,
                    'cell_type': meta_test['cell_type'].tolist() if 'cell_type' in meta_test.columns else None,
                    'dose': meta_test['dose'].tolist() if 'dose' in meta_test.columns else None,
                }
                pred = chemcpa.predict(diseased_test, test_meta)
                results = evaluator.compute_metrics(treated_test, pred, diseased_test)
                evaluator.print_results(results, "ChemCPA")
                all_results["ChemCPA"] = results
            except Exception as e:
                print(f"  ChemCPA failed: {e}")

    # Create ensembles with available models
    print("\n--- Ensemble Models ---")

    # Base ensemble (always available)
    print("\nTraining StackingEnsemble (base models)...")
    ensemble_base = [
        NoChangeBaseline(),
        MeanShiftBaseline(),
        PerGeneLinearBaseline(),
    ]
    stacking_base = StackingEnsemble(ensemble_base)
    stacking_base.train(diseased_train, treated_train)
    pred = stacking_base.predict(diseased_test)
    results = evaluator.compute_metrics(treated_test, pred, diseased_test)
    evaluator.print_results(results, "StackingEnsemble")
    all_results["StackingEnsemble"] = results

    # Full ensemble with deep learning models
    if include_deep and (CELLOT_AVAILABLE or CHEMCPA_AVAILABLE):
        print("\nTraining FullStackingEnsemble (with deep models)...")
        full_models = [
            NoChangeBaseline(),
            MeanShiftBaseline(),
            PerGeneLinearBaseline(),
        ]
        if CELLOT_AVAILABLE:
            full_models.append(CellOTModel(n_epochs=100, n_components=100))
        if CHEMCPA_AVAILABLE:
            full_models.append(ChemCPAModel(n_epochs=100))

        try:
            # Prepare metadata for full ensemble
            full_meta = {
                'smiles': meta_train['smiles'].tolist() if 'smiles' in meta_train.columns else None,
                'cell_type': meta_train['cell_type'].tolist() if 'cell_type' in meta_train.columns else None,
                'dose': meta_train['dose'].tolist() if 'dose' in meta_train.columns else None,
            }
            full_ensemble = StackingEnsemble(full_models)
            full_ensemble.train(diseased_train, treated_train, full_meta)

            test_full_meta = {
                'smiles': meta_test['smiles'].tolist() if 'smiles' in meta_test.columns else None,
                'cell_type': meta_test['cell_type'].tolist() if 'cell_type' in meta_test.columns else None,
                'dose': meta_test['dose'].tolist() if 'dose' in meta_test.columns else None,
            }
            pred = full_ensemble.predict(diseased_test, test_full_meta)
            results = evaluator.compute_metrics(treated_test, pred, diseased_test)
            evaluator.print_results(results, "FullStackingEnsemble")
            all_results["FullStackingEnsemble"] = results
        except Exception as e:
            print(f"  FullStackingEnsemble failed: {e}")

    # Summary table
    print(f"\n{'='*80}")
    print("Summary Table")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'R² Top20':>10} {'R² Top40':>10} {'R² Top80':>10} {'R² All':>10}")
    print("-" * 80)
    for model_name, results in all_results.items():
        print(f"{model_name:<25} {results['r2_top20']:>10.4f} {results['r2_top40']:>10.4f} "
              f"{results['r2_top80']:>10.4f} {results['r2_all']:>10.4f}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_line', type=str, default='A549')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--n_samples', type=int, default=5000)
    parser.add_argument('--no_deep', action='store_true', help='Skip deep learning models')
    args = parser.parse_args()

    results = evaluate_all_models(
        cell_line=args.cell_line,
        fold=args.fold,
        n_samples=args.n_samples,
        include_deep=not args.no_deep
    )
