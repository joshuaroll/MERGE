#!/usr/bin/env python
"""
CellOT wrapper for PDGrapher baseline evaluation.
CellOT uses optimal transport with Input Convex Neural Networks (ICNN).
Base implementation without PCA - matches paper methodology.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import autograd
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import PDGrapherDataLoader, TopKEvaluator
from models.base_model import BasePerturbationModel


class NonNegativeLinear(nn.Linear):
    """Linear layer with non-negative weights (using softplus)."""
    def __init__(self, *args, beta=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta

    def forward(self, x):
        return nn.functional.linear(x, self.kernel(), self.bias)

    def kernel(self):
        return nn.functional.softplus(self.weight, beta=self.beta)


class ICNN(nn.Module):
    """
    Input Convex Neural Network for optimal transport.
    The network is convex in its input, enabling transport via gradient.
    """
    def __init__(self, input_dim: int, hidden_units: list = [256, 256, 256, 256],
                 activation: str = "leakyrelu", softplus_W_kernels: bool = True):
        super().__init__()
        self.softplus_W_kernels = softplus_W_kernels

        if activation.lower() == "leakyrelu":
            self.sigma = nn.LeakyReLU
        else:
            self.sigma = nn.ReLU

        units = hidden_units + [1]

        # W layers (non-negative weights)
        if self.softplus_W_kernels:
            WLinear = lambda *args, **kwargs: NonNegativeLinear(*args, **kwargs, beta=1.0)
        else:
            WLinear = nn.Linear

        self.W = nn.ModuleList([
            WLinear(idim, odim, bias=False)
            for idim, odim in zip(units[:-1], units[1:])
        ])

        # A layers (input skip connections)
        self.A = nn.ModuleList([
            nn.Linear(input_dim, odim, bias=True) for odim in units
        ])

        # Initialize
        for layer in self.A:
            nn.init.normal_(layer.weight, std=0.1)
            nn.init.zeros_(layer.bias)
        for layer in self.W:
            nn.init.normal_(layer.weight, std=0.1)

    def forward(self, x):
        z = self.sigma(0.2)(self.A[0](x))
        z = z * z  # Square activation for convexity

        for W, A in zip(self.W[:-1], self.A[1:-1]):
            z = self.sigma(0.2)(W(z) + A(x))

        y = self.W[-1](z) + self.A[-1](x)
        return y

    def transport(self, x):
        """Compute optimal transport map via gradient of convex function."""
        assert x.requires_grad
        output, = autograd.grad(
            self.forward(x),
            x,
            create_graph=True,
            only_inputs=True,
            grad_outputs=torch.ones((x.size()[0], 1), device=x.device).float(),
        )
        return output

    def clamp_w(self):
        """Clamp W weights to be non-negative (for non-softplus version)."""
        if self.softplus_W_kernels:
            return
        for w in self.W:
            w.weight.data = w.weight.data.clamp(min=0)


class CellOTModel(BasePerturbationModel):
    """
    CellOT model implementing BasePerturbationModel interface.
    Uses optimal transport via dual ICNN networks.
    Supports optional PCA dimensionality reduction.
    """

    def __init__(self, hidden_units: list = [256, 256, 256, 256],
                 n_epochs: int = 100, batch_size: int = 64,
                 learning_rate: float = 1e-4, n_inner_iters: int = 10,
                 n_components: int = None,  # None = no PCA, int = use PCA
                 device: str = 'cuda'):
        super().__init__()
        self.hidden_units = hidden_units
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_inner_iters = n_inner_iters
        self.n_components = n_components
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.f = None  # Potential function
        self.g = None  # Transport network
        self._input_dim = None
        self._pca = None  # PCA model if used

    @property
    def name(self) -> str:
        if self.n_components:
            return f"CellOT_PCA{self.n_components}"
        return "CellOT"

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """
        Train CellOT model using optimal transport.
        Optionally uses PCA for dimensionality reduction.
        """
        from sklearn.decomposition import PCA

        n_samples, n_genes = diseased.shape
        self._n_genes = n_genes

        # Z-score normalize
        self._mean = diseased.mean(axis=0)
        self._std = diseased.std(axis=0) + 1e-8
        diseased_norm = (diseased - self._mean) / self._std
        treated_norm = (treated - self._mean) / self._std

        # Apply PCA if specified
        if self.n_components is not None:
            self._pca = PCA(n_components=self.n_components)
            diseased_norm = self._pca.fit_transform(diseased_norm)
            treated_norm = self._pca.transform(treated_norm)
            self._input_dim = self.n_components
            var_explained = self._pca.explained_variance_ratio_.sum() * 100
            print(f"  Using PCA: {n_genes} genes -> {self.n_components} components (var: {var_explained:.2f}%)")
        else:
            self._input_dim = n_genes
            print(f"  Training on full {n_genes} genes (no PCA)")

        # Convert to tensors
        source_tensor = torch.FloatTensor(diseased_norm)
        target_tensor = torch.FloatTensor(treated_norm)

        # Create dataloaders
        source_dataset = TensorDataset(source_tensor)
        target_dataset = TensorDataset(target_tensor)
        source_loader = DataLoader(source_dataset, batch_size=self.batch_size, shuffle=True)
        target_loader = DataLoader(target_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize networks with input dimension (full genes or PCA components)
        self.f = ICNN(self._input_dim, self.hidden_units).to(self.device)
        self.g = ICNN(self._input_dim, self.hidden_units).to(self.device)

        # Optimizers
        opt_f = torch.optim.Adam(self.f.parameters(), lr=self.learning_rate)
        opt_g = torch.optim.Adam(self.g.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.n_epochs):
            total_loss_f = 0
            total_loss_g = 0
            n_batches = 0

            source_iter = iter(source_loader)
            target_iter = iter(target_loader)

            while True:
                try:
                    target_batch = next(target_iter)[0].to(self.device)
                except StopIteration:
                    break

                # Train g (transport) for n_inner_iters
                for _ in range(self.n_inner_iters):
                    try:
                        source_batch = next(source_iter)[0].to(self.device)
                    except StopIteration:
                        source_iter = iter(source_loader)
                        source_batch = next(source_iter)[0].to(self.device)

                    source_batch.requires_grad_(True)

                    opt_g.zero_grad()
                    transport = self.g.transport(source_batch)
                    loss_g = self.f(transport) - torch.multiply(source_batch, transport).sum(-1, keepdim=True)
                    loss_g = loss_g.mean()
                    loss_g.backward()
                    opt_g.step()
                    total_loss_g += loss_g.item()

                # Train f (potential)
                try:
                    source_batch = next(source_iter)[0].to(self.device)
                except StopIteration:
                    source_iter = iter(source_loader)
                    source_batch = next(source_iter)[0].to(self.device)

                source_batch.requires_grad_(True)

                opt_f.zero_grad()
                transport = self.g.transport(source_batch)
                loss_f = -self.f(transport) + self.f(target_batch)
                loss_f = loss_f.mean()
                loss_f.backward()
                opt_f.step()
                self.f.clamp_w()

                total_loss_f += loss_f.item()
                n_batches += 1

            if (epoch + 1) % 20 == 0:
                avg_f = total_loss_f / max(n_batches, 1)
                avg_g = total_loss_g / max(n_batches * self.n_inner_iters, 1)
                print(f"  Epoch {epoch+1}/{self.n_epochs}, Loss F: {avg_f:.4f}, Loss G: {avg_g:.4f}")

        self._trained = True

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Predict treated expression using optimal transport.
        """
        if not self._trained:
            raise RuntimeError("Model must be trained first")

        # Apply same preprocessing: z-score normalize
        diseased_norm = (diseased - self._mean) / self._std

        # Apply PCA if used during training
        if self._pca is not None:
            diseased_norm = self._pca.transform(diseased_norm)

        # Convert to tensor
        source_tensor = torch.FloatTensor(diseased_norm).to(self.device)
        source_tensor.requires_grad_(True)

        # Transport
        self.g.eval()
        with torch.enable_grad():  # Need gradients for transport
            transported_norm = self.g.transport(source_tensor)

        transported_np = transported_norm.detach().cpu().numpy()

        # Inverse PCA if used
        if self._pca is not None:
            transported_np = self._pca.inverse_transform(transported_np)

        # Inverse z-score
        transported = transported_np * self._std + self._mean

        return transported


def train_and_evaluate_cellot(cell_line: str = "A549", fold: int = 0, n_samples: int = 5000,
                               n_components: int = None):
    """
    Train and evaluate CellOT on a specific cell line.

    Args:
        cell_line: Cell line to evaluate
        fold: Cross-validation fold
        n_samples: Number of samples to use
        n_components: PCA components (None = no PCA)
    """
    pca_str = f" with PCA {n_components}" if n_components else " (no PCA)"
    print(f"\n=== Training CellOT{pca_str} on {cell_line} (fold {fold}) ===")

    # Load data
    loader = PDGrapherDataLoader()
    train_idx, test_idx = loader.get_train_test_split(cell_line, fold=fold)

    # Limit samples
    if len(train_idx) > n_samples:
        train_idx = train_idx[:n_samples]
    if len(test_idx) > n_samples // 4:
        test_idx = test_idx[:n_samples // 4]

    print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # Get data
    treated_train, diseased_train = loader.get_expression_arrays(train_idx)
    treated_test, diseased_test = loader.get_expression_arrays(test_idx)

    # Train model
    model = CellOTModel(hidden_units=[256, 256, 256, 256], n_epochs=100, batch_size=64,
                        n_components=n_components)
    model.train(diseased_train, treated_train)

    # Predict
    print("Generating predictions...")
    pred_treated = model.predict(diseased_test)

    # Evaluate
    evaluator = TopKEvaluator()
    results = evaluator.compute_metrics(treated_test, pred_treated, diseased_test)
    evaluator.print_results(results, model.name)

    return results, model


if __name__ == "__main__":
    import sys

    # Test with different PCA settings
    all_results = {}

    # PCA 100
    results_100, _ = train_and_evaluate_cellot(cell_line="A549", fold=0, n_samples=5000,
                                                n_components=100)
    all_results['CellOT_PCA100'] = results_100

    # PCA 200
    results_200, _ = train_and_evaluate_cellot(cell_line="A549", fold=0, n_samples=5000,
                                                n_components=200)
    all_results['CellOT_PCA200'] = results_200

    # Print comparison
    print("\n" + "="*70)
    print("CellOT PCA Comparison")
    print("="*70)
    print(f"{'Model':<20} {'R² Top-20':>12} {'R² Top-40':>12} {'R² Top-80':>12}")
    print("-"*70)
    for name, r in all_results.items():
        print(f"{name:<20} {r['r2_top20']:>12.4f} {r['r2_top40']:>12.4f} {r['r2_top80']:>12.4f}")
