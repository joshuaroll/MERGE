#!/usr/bin/env python
"""
Unified Mixture-of-Experts modules for MERGE ensemble.

Two MoE styles available:
1. Sparse MoE: Top-k routing with LayerNorm and re-normalization (from CheMoE/HeirCheMoE)
2. MultiDCP MoE: Top-k routing with softmax over selected + load balancing loss

Both can be used with TranSiGen and MultiDCP models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SparseGatingNetwork(nn.Module):
    """
    Sparse gating network with LayerNorm (CheMoE/SparseCheMoE style).

    Features:
    - Top-k expert selection with configurable k
    - LayerNorm on gate logits for stable training
    - Re-normalization of selected expert weights
    """

    def __init__(self, input_dim: int, num_experts: int = 4,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.num_experts = num_experts

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
            nn.LayerNorm(num_experts)
        )

    def forward(self, x: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim] input features
            top_k: number of experts to select

        Returns:
            weights: [batch, num_experts] sparse expert weights (only top-k non-zero)
            indices: [batch, top_k] indices of selected experts
        """
        # Compute gate logits
        logits = self.gate(x)  # [batch, num_experts]

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        # Select top-k
        top_k_weights, top_k_indices = torch.topk(probs, top_k, dim=-1)

        # Re-normalize selected weights to sum to 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Create sparse weight tensor
        weights = torch.zeros_like(probs)
        weights.scatter_(1, top_k_indices, top_k_weights)

        return weights, top_k_indices


class MultiDCPGatingNetwork(nn.Module):
    """
    MultiDCP-style gating network.

    Features:
    - Top-k expert selection (default k=2)
    - Softmax only over selected experts (not re-normalized from full distribution)
    - Supports load balancing loss computation
    """

    def __init__(self, input_dim: int, num_experts: int = 4,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.num_experts = num_experts

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts)
        )

    def forward(self, x: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim] input features
            top_k: number of experts to select

        Returns:
            weights: [batch, num_experts] sparse expert weights
            indices: [batch, top_k] indices of selected experts
        """
        # Compute raw gating scores
        logits = self.gate(x)  # [batch, num_experts]

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=1)

        # Softmax over selected experts only
        top_k_weights = F.softmax(top_k_logits, dim=1)

        # Create sparse gate tensor
        weights = torch.zeros_like(logits)
        weights.scatter_(1, top_k_indices, top_k_weights)

        return weights, top_k_indices


class ExpertNetwork(nn.Module):
    """
    Single expert network (MLP).

    Configurable architecture with LayerNorm option.
    """

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dims: list = None, dropout: float = 0.1,
                 use_layernorm: bool = True):
        super().__init__()

        hidden_dims = hidden_dims or [256]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoELayer(nn.Module):
    """
    Unified Mixture-of-Experts layer supporting both Sparse and MultiDCP styles.

    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        num_experts: Number of expert networks
        top_k: Number of experts to select per sample
        moe_style: 'sparse' (LayerNorm + re-normalize) or 'multidcp' (softmax over selected)
        expert_hidden_dims: Hidden dimensions for expert networks
        dropout: Dropout rate
        use_layernorm: Whether to use LayerNorm in experts (only for sparse style)
    """

    def __init__(self, input_dim: int, output_dim: int, num_experts: int = 4,
                 top_k: int = 2, moe_style: str = 'sparse',
                 expert_hidden_dims: list = None, dropout: float = 0.1,
                 use_layernorm: bool = True):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_style = moe_style

        # Select gating network based on style
        if moe_style == 'sparse':
            self.gating = SparseGatingNetwork(
                input_dim, num_experts, hidden_dim=128, dropout=dropout
            )
            expert_layernorm = use_layernorm
        elif moe_style == 'multidcp':
            self.gating = MultiDCPGatingNetwork(
                input_dim, num_experts, hidden_dim=128, dropout=dropout
            )
            expert_layernorm = False  # MultiDCP style doesn't use LayerNorm
        else:
            raise ValueError(f"Unknown moe_style: {moe_style}. Use 'sparse' or 'multidcp'")

        # Create expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim, output_dim,
                hidden_dims=expert_hidden_dims,
                dropout=dropout,
                use_layernorm=expert_layernorm
            )
            for _ in range(num_experts)
        ])

        # Store last weights for load balancing loss
        self._last_expert_weights = None
        self._last_expert_indices = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim] or [batch, seq_len, input_dim] input features

        Returns:
            output: [batch, output_dim] or [batch, seq_len, output_dim] expert outputs
        """
        # Handle 3D input (batch, seq_len, input_dim)
        if x.dim() == 3:
            batch_size, seq_len, input_dim = x.shape
            x_flat = x.view(batch_size * seq_len, input_dim)
            output_flat = self._forward_2d(x_flat)
            return output_flat.view(batch_size, seq_len, -1)
        else:
            return self._forward_2d(x)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 2D input [batch, input_dim]."""
        # Get expert weights
        weights, indices = self.gating(x, top_k=self.top_k)

        # Store for load balancing loss
        self._last_expert_weights = weights
        self._last_expert_indices = indices

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # [batch, num_experts, output_dim]

        # Weighted combination
        weights_expanded = weights.unsqueeze(-1)  # [batch, num_experts, 1]
        output = (expert_outputs * weights_expanded).sum(dim=1)  # [batch, output_dim]

        return output

    def compute_load_balance_loss(self, weight: float = 0.01) -> torch.Tensor:
        """
        Compute load balancing loss to encourage equal expert usage.

        Args:
            weight: Loss weight multiplier

        Returns:
            load_balance_loss: Scalar loss value
        """
        if self._last_expert_weights is None:
            return torch.tensor(0.0)

        # Mean importance per expert across batch
        importance = self._last_expert_weights.mean(dim=0)  # [num_experts]

        # Target: equal usage
        target = torch.ones_like(importance) / self.num_experts

        # L2 distance from uniform
        loss = F.mse_loss(importance, target)

        return weight * loss


class FusionMoE(nn.Module):
    """
    MoE for feature fusion (TranSiGen style).

    Takes two input tensors (e.g., latent z and feature embedding)
    and fuses them through the MoE layer.
    """

    def __init__(self, latent_dim: int, embed_dim: int, num_experts: int = 4,
                 top_k: int = 2, moe_style: str = 'sparse', dropout: float = 0.1):
        super().__init__()

        self.input_dim = latent_dim + embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_style = moe_style

        self.moe = MoELayer(
            input_dim=self.input_dim,
            output_dim=self.input_dim,  # Same as input for fusion
            num_experts=num_experts,
            top_k=top_k,
            moe_style=moe_style,
            expert_hidden_dims=[self.input_dim],
            dropout=dropout
        )

    def forward(self, z: torch.Tensor, feat_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, latent_dim] latent representation
            feat_embed: [batch, embed_dim] feature embedding

        Returns:
            fused: [batch, latent_dim + embed_dim] fused representation
        """
        combined = torch.cat([z, feat_embed], dim=-1)
        return self.moe(combined)

    def compute_load_balance_loss(self, weight: float = 0.01) -> torch.Tensor:
        return self.moe.compute_load_balance_loss(weight)


def get_moe_layer(input_dim: int, output_dim: int, moe_style: str = 'sparse',
                  num_experts: int = 4, top_k: int = 2, dropout: float = 0.1,
                  expert_hidden_dims: list = None) -> MoELayer:
    """
    Factory function to create an MoE layer with specified style.

    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension
        moe_style: 'sparse' or 'multidcp'
        num_experts: Number of experts
        top_k: Number of experts to select
        dropout: Dropout rate
        expert_hidden_dims: Hidden layer dimensions for experts

    Returns:
        MoELayer instance
    """
    return MoELayer(
        input_dim=input_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        top_k=top_k,
        moe_style=moe_style,
        expert_hidden_dims=expert_hidden_dims,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test both MoE styles
    batch_size = 32
    input_dim = 256
    output_dim = 128

    x = torch.randn(batch_size, input_dim)

    print("Testing Sparse MoE...")
    sparse_moe = MoELayer(input_dim, output_dim, moe_style='sparse', num_experts=4, top_k=2)
    out_sparse = sparse_moe(x)
    print(f"  Input: {x.shape}, Output: {out_sparse.shape}")
    print(f"  Load balance loss: {sparse_moe.compute_load_balance_loss():.4f}")

    print("\nTesting MultiDCP MoE...")
    mdcp_moe = MoELayer(input_dim, output_dim, moe_style='multidcp', num_experts=4, top_k=2)
    out_mdcp = mdcp_moe(x)
    print(f"  Input: {x.shape}, Output: {out_mdcp.shape}")
    print(f"  Load balance loss: {mdcp_moe.compute_load_balance_loss():.4f}")

    print("\nTesting Fusion MoE (TranSiGen style)...")
    latent_dim, embed_dim = 128, 128
    z = torch.randn(batch_size, latent_dim)
    feat = torch.randn(batch_size, embed_dim)

    fusion_sparse = FusionMoE(latent_dim, embed_dim, moe_style='sparse')
    fusion_mdcp = FusionMoE(latent_dim, embed_dim, moe_style='multidcp')

    fused_sparse = fusion_sparse(z, feat)
    fused_mdcp = fusion_mdcp(z, feat)

    print(f"  Sparse fusion: z={z.shape}, feat={feat.shape} -> {fused_sparse.shape}")
    print(f"  MultiDCP fusion: z={z.shape}, feat={feat.shape} -> {fused_mdcp.shape}")

    print("\nAll tests passed!")
