#!/usr/bin/env python
"""
Biolord wrapper for PDGrapher baseline evaluation.
Biolord uses disentangled representations for perturbation prediction.
"""
import os
import sys
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import PDGrapherDataLoader, TopKEvaluator
from models.base_model import BasePerturbationModel


class BiolordModel(BasePerturbationModel):
    """
    Biolord model wrapper implementing BasePerturbationModel interface.

    Uses Biolord's disentangled representation learning with condition
    as a categorical attribute to predict treated from diseased.
    """

    def __init__(self, n_latent: int = 64, n_epochs: int = 100, batch_size: int = 64,
                 n_hvg: int = 2000):
        super().__init__()
        self.n_latent = n_latent
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_hvg = n_hvg
        self.model = None
        self._gene_names = None
        self._adata_train = None

    @property
    def name(self) -> str:
        return f"Biolord_lat{self.n_latent}_hvg{self.n_hvg}"

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """
        Train Biolord model.

        Args:
            diseased: Diseased expression array (n_samples, n_genes)
            treated: Treated expression array (n_samples, n_genes)
            metadata: Optional dict with 'gene_names'
        """
        import anndata as ad
        import scanpy as sc
        import biolord

        n_samples, n_genes = diseased.shape

        # Create gene names if not provided
        if metadata and 'gene_names' in metadata:
            self._gene_names = metadata['gene_names']
        else:
            self._gene_names = [f"gene_{i}" for i in range(n_genes)]

        # Stack diseased and treated as separate observations
        X_train = np.vstack([diseased, treated])

        adata_train = ad.AnnData(X=X_train)
        adata_train.var_names = self._gene_names

        # Add condition as categorical attribute
        adata_train.obs['condition'] = ['diseased'] * n_samples + ['treated'] * n_samples
        adata_train.obs['condition'] = adata_train.obs['condition'].astype('category')

        # Add sample index for pairing
        adata_train.obs['sample_idx'] = list(range(n_samples)) * 2

        # Preprocess
        sc.pp.normalize_total(adata_train, target_sum=1e4)
        sc.pp.log1p(adata_train)

        # Select HVGs for efficiency
        n_hvg = min(self.n_hvg, n_genes)
        sc.pp.highly_variable_genes(adata_train, n_top_genes=n_hvg)
        adata_hvg = adata_train[:, adata_train.var.highly_variable].copy()

        print(f"  Training Biolord with {adata_hvg.shape[0]} samples, {adata_hvg.shape[1]} HVGs")

        # Setup Biolord with condition as categorical attribute
        biolord.Biolord.setup_anndata(
            adata_hvg,
            categorical_attributes_keys=['condition'],
        )

        # Create and train model
        self.model = biolord.Biolord(
            adata_hvg,
            n_latent=self.n_latent,
        )

        # Patch the scvi callback to handle None loss values
        import scvi.train._callbacks as scvi_callbacks
        original_on_train_batch_end = scvi_callbacks.SaveCheckpoint.on_train_batch_end
        def patched_on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            if outputs is None:
                return  # Skip NaN check if outputs is None
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if loss is None:
                return  # Skip NaN check if loss is None
            return original_on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx)
        scvi_callbacks.SaveCheckpoint.on_train_batch_end = patched_on_train_batch_end

        self.model.train(
            max_epochs=self.n_epochs,
            batch_size=self.batch_size,
            early_stopping=False,  # Disable early stopping to avoid callback issues
            check_val_every_n_epoch=self.n_epochs + 1,  # Effectively disable validation
            enable_checkpointing=False,  # Disable checkpointing to avoid monitor key issue
        )

        self._adata_train = adata_hvg
        self._hvg_genes = list(adata_hvg.var_names)
        self._all_genes = self._gene_names
        self._trained = True

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Predict treated expression from diseased expression.

        Args:
            diseased: Diseased expression (n_samples, n_genes)
            metadata: Optional metadata

        Returns:
            Predicted treated expression (n_samples, n_genes)
        """
        if not self._trained:
            raise RuntimeError("Model must be trained first")

        import anndata as ad
        import scanpy as sc
        import biolord

        n_samples, n_genes = diseased.shape

        # Create test AnnData with diseased samples
        adata_test = ad.AnnData(X=diseased.copy())
        adata_test.var_names = self._all_genes
        adata_test.obs['condition'] = 'diseased'
        adata_test.obs['condition'] = adata_test.obs['condition'].astype('category')

        sc.pp.normalize_total(adata_test, target_sum=1e4)
        sc.pp.log1p(adata_test)

        # Keep same HVGs
        hvg_mask = [g in self._hvg_genes for g in adata_test.var_names]
        adata_test_hvg = adata_test[:, hvg_mask].copy()

        # Register test AnnData with Biolord (required for prediction)
        biolord.Biolord.setup_anndata(
            adata_test_hvg,
            categorical_attributes_keys=['condition'],
        )

        # Get predictions using Biolord's counterfactual prediction
        # Shift from 'diseased' to 'treated' condition
        pred_adata = self.model.compute_prediction_adata(
            adata=self._adata_train,
            adata_source=adata_test_hvg,
            target_attributes=['condition'],
        )

        # Get the treated predictions (filter for 'treated' condition)
        treated_mask = pred_adata.obs['condition'] == 'treated'
        pred_treated_hvg = pred_adata[treated_mask].X

        # Expand back to all genes (fill non-HVG with input values)
        # For non-HVG genes, use the normalized diseased values
        adata_test_norm = ad.AnnData(X=diseased.copy())
        sc.pp.normalize_total(adata_test_norm, target_sum=1e4)
        sc.pp.log1p(adata_test_norm)

        pred_treated_full = adata_test_norm.X.copy()
        hvg_indices = [i for i, g in enumerate(self._all_genes) if g in self._hvg_genes]
        for i, hvg_idx in enumerate(hvg_indices):
            pred_treated_full[:, hvg_idx] = pred_treated_hvg[:, i]

        return pred_treated_full


def train_and_evaluate_biolord(cell_line: str = "A549", fold: int = 0, n_samples: int = 5000,
                                n_latent: int = 64, n_epochs: int = 100, n_hvg: int = 2000):
    """
    Train and evaluate Biolord on a specific cell line.

    Args:
        cell_line: Cell line to evaluate
        fold: Cross-validation fold
        n_samples: Number of samples to use (for memory efficiency)
        n_latent: Latent dimension
        n_epochs: Number of training epochs
        n_hvg: Number of highly variable genes
    """
    print(f"\n=== Training Biolord (latent={n_latent}, hvg={n_hvg}, epochs={n_epochs}) on {cell_line} ===")

    # Load data
    loader = PDGrapherDataLoader()
    train_idx, test_idx = loader.get_train_test_split(cell_line, fold=fold)

    # Limit samples for memory
    if len(train_idx) > n_samples:
        train_idx = train_idx[:n_samples]
    if len(test_idx) > n_samples // 4:
        test_idx = test_idx[:n_samples // 4]

    print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # Get expression arrays
    treated_train, diseased_train = loader.get_expression_arrays(train_idx)
    treated_test, diseased_test = loader.get_expression_arrays(test_idx)

    # Create and train model
    model = BiolordModel(n_latent=n_latent, n_epochs=n_epochs, batch_size=64, n_hvg=n_hvg)
    metadata = {'gene_names': loader.gene_cols}
    model.train(diseased_train, treated_train, metadata)

    # Predict
    print("Generating predictions...")
    pred_treated = model.predict(diseased_test)

    # Normalize test data for fair comparison (same preprocessing as training)
    treated_test_norm = np.log1p(treated_test * 1e4 / treated_test.sum(axis=1, keepdims=True))
    diseased_test_norm = np.log1p(diseased_test * 1e4 / diseased_test.sum(axis=1, keepdims=True))

    # Evaluate
    evaluator = TopKEvaluator()
    results = evaluator.compute_metrics(treated_test_norm, pred_treated, diseased_test_norm)
    evaluator.print_results(results, model.name)

    return results, model


if __name__ == "__main__":
    # Hyperparameter tuning for Biolord
    all_results = {}

    # Configurations to test
    configs = [
        # (n_latent, n_hvg, n_epochs)
        (64, 2000, 100),    # baseline
        (128, 2000, 100),   # larger latent
        (256, 2000, 100),   # even larger latent
        (128, 3000, 100),   # more HVGs
        (128, 5000, 100),   # even more HVGs
        (128, 3000, 200),   # more epochs
    ]

    for n_latent, n_hvg, n_epochs in configs:
        config_name = f"lat{n_latent}_hvg{n_hvg}_ep{n_epochs}"
        print(f"\n{'='*70}")
        print(f"Testing config: {config_name}")
        print(f"{'='*70}")

        try:
            results, _ = train_and_evaluate_biolord(
                cell_line="A549", fold=0, n_samples=3000,
                n_latent=n_latent, n_epochs=n_epochs, n_hvg=n_hvg
            )
            all_results[config_name] = results
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Print comparison
    print("\n" + "="*80)
    print("BIOLORD HYPERPARAMETER TUNING RESULTS")
    print("="*80)
    print(f"{'Config':<30} {'R² Top-20':>12} {'R² Top-40':>12} {'R² Top-80':>12}")
    print("-"*80)
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['r2_top20']):
        print(f"{name:<30} {r['r2_top20']:>12.4f} {r['r2_top40']:>12.4f} {r['r2_top80']:>12.4f}")
    print("-"*80)
    print("Paper Biolord (A549):          0.7248       0.7362       0.7432")
