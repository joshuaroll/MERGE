#!/usr/bin/env python
"""
Base class for all perturbation prediction models.
"""
import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BasePerturbationModel(ABC):
    """
    Abstract base class for perturbation prediction models.

    All models should implement:
    - train(): Train the model on training data
    - predict(): Predict treated expression from diseased expression
    - name: Model name property
    """

    def __init__(self):
        self._trained = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass

    @abstractmethod
    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """
        Train the model.

        Args:
            diseased: Diseased expression array (n_samples, n_genes)
            treated: Treated expression array (n_samples, n_genes)
            metadata: Optional metadata dict with keys like 'smiles', 'dose', 'cell_type'
        """
        pass

    @abstractmethod
    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Predict treated expression from diseased expression.

        Args:
            diseased: Diseased expression array (n_samples, n_genes)
            metadata: Optional metadata dict

        Returns:
            Predicted treated expression (n_samples, n_genes)
        """
        pass

    def get_embeddings(self, diseased: np.ndarray,
                       metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Extract embeddings/latent representations from the model.

        This is used for embedding-level stacking ensembles where
        intermediate representations are combined rather than final outputs.

        Args:
            diseased: Diseased expression array (n_samples, n_genes)
            metadata: Optional metadata dict

        Returns:
            Embeddings array (n_samples, embedding_dim)
            Returns None if model doesn't support embedding extraction
        """
        # Default implementation returns None (not supported)
        return None

    @property
    def embedding_dim(self) -> Optional[int]:
        """Return the embedding dimension, or None if not supported."""
        return None

    @property
    def is_trained(self) -> bool:
        return self._trained


class NoChangeBaseline(BasePerturbationModel):
    """
    No-change baseline: predicts treated = diseased.
    """

    @property
    def name(self) -> str:
        return "NoChange"

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        # No training needed
        self._trained = True

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        return diseased.copy()


class MeanShiftBaseline(BasePerturbationModel):
    """
    Mean-shift baseline: predicts treated = diseased + mean(treated - diseased).
    """

    def __init__(self):
        super().__init__()
        self._mean_shift = None

    @property
    def name(self) -> str:
        return "MeanShift"

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        # Compute mean shift
        self._mean_shift = np.mean(treated - diseased, axis=0)
        self._trained = True

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model must be trained first")
        return diseased + self._mean_shift


class PerGeneLinearBaseline(BasePerturbationModel):
    """
    Per-gene linear regression: fits y = ax + b for each gene.
    """

    def __init__(self):
        super().__init__()
        self._slopes = None
        self._intercepts = None

    @property
    def name(self) -> str:
        return "PerGeneLinear"

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        n_genes = diseased.shape[1]
        self._slopes = np.zeros(n_genes)
        self._intercepts = np.zeros(n_genes)

        for g in range(n_genes):
            x = diseased[:, g]
            y = treated[:, g]

            # Simple linear regression
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator > 1e-10:
                self._slopes[g] = numerator / denominator
                self._intercepts[g] = y_mean - self._slopes[g] * x_mean
            else:
                # If x is constant, predict mean of y
                self._slopes[g] = 0
                self._intercepts[g] = y_mean

        self._trained = True

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model must be trained first")
        return diseased * self._slopes + self._intercepts


if __name__ == "__main__":
    # Test baselines
    from data_loader import PDGrapherDataLoader, TopKEvaluator

    loader = PDGrapherDataLoader()
    train_idx, test_idx = loader.get_train_test_split("A549", fold=0)

    # Limit for testing
    train_idx = train_idx[:2000]
    test_idx = test_idx[:500]

    treated_train, diseased_train = loader.get_expression_arrays(train_idx)
    treated_test, diseased_test = loader.get_expression_arrays(test_idx)

    evaluator = TopKEvaluator()

    # Test each baseline
    for ModelClass in [NoChangeBaseline, MeanShiftBaseline, PerGeneLinearBaseline]:
        model = ModelClass()
        model.train(diseased_train, treated_train)
        pred = model.predict(diseased_test)

        results = evaluator.compute_metrics(treated_test, pred, diseased_test)
        evaluator.print_results(results, model.name)
