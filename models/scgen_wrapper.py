#!/usr/bin/env python
"""
scGen wrapper for PDGrapher baseline evaluation.
scGen uses a VAE with vector arithmetic for perturbation prediction.
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


class ScGenModel(BasePerturbationModel):
    """
    scGen model wrapper implementing BasePerturbationModel interface.

    scGen predicts perturbation effects using VAE latent space arithmetic:
    1. Train VAE on control (diseased) and treated cells
    2. Compute perturbation vector as mean(treated) - mean(control) in latent space
    3. Predict by adding perturbation vector to control cells
    """

    def __init__(self, n_latent: int = 100, n_epochs: int = 50, batch_size: int = 64):
        super().__init__()
        self.n_latent = n_latent
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model = None
        self._gene_names = None
        self._adata_train = None

    @property
    def name(self) -> str:
        return "scGen"

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """
        Train scGen model.

        Args:
            diseased: Diseased expression array (n_samples, n_genes)
            treated: Treated expression array (n_samples, n_genes)
            metadata: Optional dict with 'gene_names'
        """
        import anndata as ad
        import scanpy as sc
        import scgen

        n_samples, n_genes = diseased.shape

        # Create gene names if not provided
        if metadata and 'gene_names' in metadata:
            self._gene_names = metadata['gene_names']
        else:
            self._gene_names = [f"gene_{i}" for i in range(n_genes)]

        # Stack control and treated as separate cells
        X_train = np.vstack([diseased, treated])

        adata_train = ad.AnnData(X=X_train)
        adata_train.var_names = self._gene_names

        # Add metadata
        adata_train.obs['condition'] = ['control'] * n_samples + ['treated'] * n_samples
        adata_train.obs['cell_type'] = 'cell'

        # Preprocess
        sc.pp.normalize_total(adata_train, target_sum=1e4)
        sc.pp.log1p(adata_train)

        # Setup and train
        scgen.SCGEN.setup_anndata(adata_train, batch_key='condition', labels_key='cell_type')
        self.model = scgen.SCGEN(adata_train, n_latent=self.n_latent)
        self.model.train(max_epochs=self.n_epochs, batch_size=self.batch_size, early_stopping=True)

        self._adata_train = adata_train
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

        # Create test AnnData
        adata_test = ad.AnnData(X=diseased.copy())
        adata_test.var_names = self._gene_names
        adata_test.obs['condition'] = 'control'
        adata_test.obs['cell_type'] = 'cell'

        sc.pp.normalize_total(adata_test, target_sum=1e4)
        sc.pp.log1p(adata_test)

        # Predict
        result = self.model.predict(
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


def train_and_evaluate_scgen(cell_line: str = "A549", fold: int = 0, n_samples: int = 5000):
    """
    Train and evaluate scGen on a specific cell line.

    Args:
        cell_line: Cell line to evaluate
        fold: Cross-validation fold
        n_samples: Number of samples to use (for memory efficiency)
    """
    print(f"\n=== Training scGen on {cell_line} (fold {fold}) ===")

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
    model = ScGenModel(n_latent=100, n_epochs=50, batch_size=64)
    metadata = {'gene_names': loader.gene_cols}
    model.train(diseased_train, treated_train, metadata)

    # Predict
    print("Generating predictions...")
    pred_treated = model.predict(diseased_test)

    # Normalize test data for fair comparison
    import scanpy as sc
    import anndata as ad
    treated_test_norm = np.log1p(treated_test * 1e4 / treated_test.sum(axis=1, keepdims=True))
    diseased_test_norm = np.log1p(diseased_test * 1e4 / diseased_test.sum(axis=1, keepdims=True))

    # Evaluate
    evaluator = TopKEvaluator()
    results = evaluator.compute_metrics(treated_test_norm, pred_treated, diseased_test_norm)
    evaluator.print_results(results, f"scGen ({cell_line})")

    return results


if __name__ == "__main__":
    # Test scGen on A549
    results = train_and_evaluate_scgen(cell_line="A549", fold=0, n_samples=2000)
