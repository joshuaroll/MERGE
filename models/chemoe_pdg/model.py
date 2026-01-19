#!/usr/bin/env python3
"""
CheMoE adapted for PDGrapher's 10,716-gene prediction task.

Architecture (per CONTEXT.md and RESEARCH.md):
- Drug: KPGT (2304) or Morgan FP (1024) -> embed_dim (128)
- Cell: Learned categorical embedding (10 cell lines) -> embed_dim (128)
- Basal: MLP encoder (10716 genes) -> embed_dim (128)
- (No dose - PDGrapher data lacks variation)

Gating: Global features (384-dim) -> 4 experts, top-k=2 sparse routing
Experts: Shared MLP per gene position (gene-aware processing)
Output: 10,716 genes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# Cell line to index mapping (matching existing code in the repo)
CELL_LINE_TO_IDX = {
    'A375': 0, 'A549': 1, 'BT20': 2, 'HA1E': 3, 'HELA': 4,
    'HT29': 5, 'MCF7': 6, 'MDAMB231': 7, 'PC3': 8, 'VCAP': 9
}

IDX_TO_CELL_LINE = {v: k for k, v in CELL_LINE_TO_IDX.items()}


class MolecularEncoder(nn.Module):
    """
    Molecular encoder supporting both KPGT and Morgan fingerprint inputs.

    Args:
        encoder_type: 'kpgt' (2304-dim) or 'morgan' (1024-dim)
        embed_dim: Output embedding dimension
        dropout: Dropout rate
    """

    def __init__(self, encoder_type: str = 'kpgt', embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder_type = encoder_type

        # Input dimension depends on encoder type
        if encoder_type == 'morgan':
            input_dim = 1024
        else:  # kpgt
            input_dim = 2304

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class BasalEncoder(nn.Module):
    """
    Encodes basal gene expression (10,716 genes) to embedding space.

    Args:
        n_genes: Number of input genes
        embed_dim: Output embedding dimension
        dropout: Dropout rate
    """

    def __init__(self, n_genes: int = 10716, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_genes, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class GatingNetwork(nn.Module):
    """
    Global gating network for expert selection with top-k sparse routing.

    Args:
        input_dim: Input feature dimension (global_dim = 3 * embed_dim)
        num_experts: Number of expert networks
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
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
            nn.Linear(hidden_dim, num_experts)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim] global features
            top_k: Number of experts to select

        Returns:
            weights: [batch, num_experts] sparse expert weights (only top-k non-zero)
            indices: [batch, top_k] indices of selected experts
        """
        # Compute gate logits
        logits = self.gate(x)  # [batch, num_experts]

        # Top-k selection
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

        # Softmax only over selected experts
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Create sparse weight tensor
        weights = torch.zeros_like(logits)
        weights.scatter_(1, top_k_indices, top_k_weights)

        return weights, top_k_indices


class ExpertNetwork(nn.Module):
    """
    Single expert network - shared MLP across all genes.

    Processes [batch, n_genes, input_dim] and outputs [batch, n_genes, 1].

    Args:
        input_dim: Input feature dimension (global_dim + gene_embed_dim)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
    """

    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()

        hidden_dims = hidden_dims or [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Output layer: 1 value per gene
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, n_genes, input_dim] gene-aware features

        Returns:
            [batch, n_genes, 1] predictions
        """
        return self.net(x)


class CheMoE_PDG(nn.Module):
    """
    CheMoE adapted for PDGrapher's 10,716-gene prediction.

    Architecture:
    - Drug: KPGT (2304) or Morgan FP (1024) -> 128-dim
    - Cell: Categorical embedding (10 cell lines) -> 128-dim
    - Basal: MLP encoder (10716 genes) -> 128-dim

    Gating: Global features (384-dim) -> 4 experts, top-k=2 sparse routing
    Experts: Shared MLP per gene, input = global (384) + gene_emb (128) = 512-dim
    Output: 10,716 genes

    Args:
        n_genes: Number of output genes (default: 10716)
        embed_dim: Embedding dimension for each modality (default: 128)
        num_experts: Number of expert networks (default: 4)
        top_k: Number of experts to select per sample (default: 2)
        mol_encoder_type: 'kpgt' (2304-dim) or 'morgan' (1024-dim)
        num_cell_lines: Number of cell lines (default: 10)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        n_genes: int = 10716,
        embed_dim: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        mol_encoder_type: str = 'kpgt',
        num_cell_lines: int = 10,
        dropout: float = 0.1,
        **kwargs  # Accept extra kwargs for compatibility
    ):
        super().__init__()

        self.n_genes = n_genes
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.mol_encoder_type = mol_encoder_type

        # === Encoders ===
        # Molecular encoder (KPGT or Morgan)
        self.mol_encoder = MolecularEncoder(
            encoder_type=mol_encoder_type,
            embed_dim=embed_dim,
            dropout=dropout
        )

        # Cell line embedding (categorical)
        self.cell_embedding = nn.Embedding(num_cell_lines, embed_dim)
        nn.init.normal_(self.cell_embedding.weight, mean=0.0, std=0.02)

        # Basal expression encoder
        self.basal_encoder = BasalEncoder(
            n_genes=n_genes,
            embed_dim=embed_dim,
            dropout=dropout
        )

        # Learnable gene embeddings
        self.gene_embedding = nn.Embedding(n_genes, embed_dim)
        nn.init.normal_(self.gene_embedding.weight, mean=0.0, std=0.02)

        # === MoE Components ===
        # Global dim: drug (128) + cell (128) + basal (128) = 384
        self.global_dim = embed_dim * 3

        # Gating network
        self.gating = GatingNetwork(
            input_dim=self.global_dim,
            num_experts=num_experts,
            hidden_dim=embed_dim,
            dropout=dropout
        )

        # Expert input: global (384) + gene_emb (128) = 512
        expert_input_dim = self.global_dim + embed_dim

        # Expert networks (4 by default, each is a shared MLP across genes)
        self.experts = nn.ModuleList([
            ExpertNetwork(
                input_dim=expert_input_dim,
                hidden_dims=[256, 128],
                dropout=dropout
            )
            for _ in range(num_experts)
        ])

        # Store last expert weights for load balancing loss
        self._last_expert_weights: Optional[torch.Tensor] = None
        self._last_expert_indices: Optional[torch.Tensor] = None

    def forward(
        self,
        basal_expr: torch.Tensor,
        mol_embed: torch.Tensor,
        cell_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            basal_expr: [batch, n_genes] diseased expression
            mol_embed: [batch, mol_dim] KPGT (2304) or Morgan (1024)
            cell_idx: [batch] cell line indices (0-9)

        Returns:
            pred_treated: [batch, n_genes] predicted treated expression
        """
        batch_size = basal_expr.size(0)
        device = basal_expr.device

        # === Encode modalities ===
        drug_embed = self.mol_encoder(mol_embed)        # [batch, 128]
        cell_embed = self.cell_embedding(cell_idx)      # [batch, 128]
        basal_embed = self.basal_encoder(basal_expr)    # [batch, 128]

        # Global features for gating
        global_features = torch.cat([drug_embed, cell_embed, basal_embed], dim=-1)  # [batch, 384]

        # === Sparse gating (top-k=2) ===
        expert_weights, expert_indices = self.gating(global_features, top_k=self.top_k)
        # expert_weights: [batch, num_experts], expert_indices: [batch, top_k]

        # Store for load balancing loss
        self._last_expert_weights = expert_weights
        self._last_expert_indices = expert_indices

        # === Gene embeddings ===
        gene_indices = torch.arange(self.n_genes, device=device)
        gene_embed = self.gene_embedding(gene_indices)  # [n_genes, 128]
        gene_embed = gene_embed.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n_genes, 128]

        # === Expand global features for gene-aware processing ===
        global_expanded = global_features.unsqueeze(1).expand(-1, self.n_genes, -1)  # [batch, n_genes, 384]
        expert_input = torch.cat([global_expanded, gene_embed], dim=-1)  # [batch, n_genes, 512]

        # === Process through all experts ===
        expert_outputs = torch.stack([
            expert(expert_input).squeeze(-1) for expert in self.experts
        ], dim=1)  # [batch, num_experts, n_genes]

        # === Weighted combination ===
        expert_weights_expanded = expert_weights.unsqueeze(-1)  # [batch, num_experts, 1]
        pred_treated = (expert_outputs * expert_weights_expanded).sum(dim=1)  # [batch, n_genes]

        return pred_treated

    def compute_load_balance_loss(self, weight: float = 0.01) -> torch.Tensor:
        """
        Compute load balancing loss to encourage equal expert usage.

        Auxiliary loss = weight * MSE(mean_expert_usage, 1/num_experts)

        Args:
            weight: Loss weight multiplier (default: 0.01)

        Returns:
            Scalar tensor with load balancing loss
        """
        if self._last_expert_weights is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Mean importance per expert across batch
        importance = self._last_expert_weights.mean(dim=0)  # [num_experts]

        # Target: equal usage (1/num_experts for each)
        target = torch.ones_like(importance) / self.num_experts

        # L2 distance from uniform distribution
        loss = F.mse_loss(importance, target)

        return weight * loss

    def get_expert_usage_stats(self) -> Optional[Dict]:
        """
        Get expert usage statistics for logging/analysis.

        Returns:
            Dict with expert weights, indices, per-expert usage, or None if not available
        """
        if self._last_expert_weights is None:
            return None

        weights = self._last_expert_weights.detach()
        indices = self._last_expert_indices.detach()

        # Per-expert mean usage (should be ~0.5 for 4 experts with top-k=2)
        usage_per_expert = weights.mean(dim=0).cpu().numpy()

        return {
            'weights': weights.cpu(),
            'indices': indices.cpu(),
            'usage_per_expert': usage_per_expert,
            'num_experts': self.num_experts,
            'top_k': self.top_k,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CheMoE_PDG Model")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Test parameters
    batch_size = 4
    n_genes = 10716

    # Create dummy data
    basal_expr = torch.randn(batch_size, n_genes)
    cell_idx = torch.randint(0, 10, (batch_size,))

    # ============================================
    # Test 1: KPGT embeddings (2304-dim)
    # ============================================
    print("\n" + "-" * 40)
    print("Test 1: KPGT embeddings (2304-dim)")
    print("-" * 40)

    kpgt_embed = torch.randn(batch_size, 2304)

    model_kpgt = CheMoE_PDG(
        n_genes=n_genes,
        mol_encoder_type='kpgt',
        num_experts=4,
        top_k=2
    )
    model_kpgt.to(device)

    # Move data to device
    basal_expr_d = basal_expr.to(device)
    kpgt_embed_d = kpgt_embed.to(device)
    cell_idx_d = cell_idx.to(device)

    # Forward pass
    pred = model_kpgt(basal_expr_d, kpgt_embed_d, cell_idx_d)

    print(f"Input: basal_expr={basal_expr_d.shape}, mol_embed={kpgt_embed_d.shape}, cell_idx={cell_idx_d.shape}")
    print(f"Output: pred_treated={pred.shape}")
    assert pred.shape == (batch_size, n_genes), f"Expected ({batch_size}, {n_genes}), got {pred.shape}"
    print("Shape verification: PASSED")

    # Load balance loss
    lb_loss = model_kpgt.compute_load_balance_loss()
    print(f"Load balance loss: {lb_loss.item():.6f}")
    assert lb_loss.item() >= 0, "Load balance loss should be non-negative"
    print("Load balance loss verification: PASSED")

    # Expert usage stats
    stats = model_kpgt.get_expert_usage_stats()
    if stats:
        print(f"Expert usage: {stats['usage_per_expert']}")
        assert len(stats['usage_per_expert']) == 4, "Should have 4 experts"
        print("Expert usage stats verification: PASSED")

    # Parameter count
    params_kpgt = sum(p.numel() for p in model_kpgt.parameters())
    print(f"Parameters: {params_kpgt:,}")

    # ============================================
    # Test 2: Morgan fingerprints (1024-dim)
    # ============================================
    print("\n" + "-" * 40)
    print("Test 2: Morgan fingerprints (1024-dim)")
    print("-" * 40)

    morgan_embed = torch.randn(batch_size, 1024).to(device)

    model_morgan = CheMoE_PDG(
        n_genes=n_genes,
        mol_encoder_type='morgan',
        num_experts=4,
        top_k=2
    ).to(device)

    pred_morgan = model_morgan(basal_expr_d, morgan_embed, cell_idx_d)

    print(f"Input: mol_embed={morgan_embed.shape}")
    print(f"Output: pred_treated={pred_morgan.shape}")
    assert pred_morgan.shape == (batch_size, n_genes), f"Expected ({batch_size}, {n_genes}), got {pred_morgan.shape}"
    print("Shape verification: PASSED")

    params_morgan = sum(p.numel() for p in model_morgan.parameters())
    print(f"Parameters: {params_morgan:,}")

    # ============================================
    # Test 3: Gating mechanism verification
    # ============================================
    print("\n" + "-" * 40)
    print("Test 3: Gating mechanism verification")
    print("-" * 40)

    # Run forward pass
    _ = model_kpgt(basal_expr_d, kpgt_embed_d, cell_idx_d)
    stats = model_kpgt.get_expert_usage_stats()

    # Check exactly top_k experts selected per sample
    weights = stats['weights']
    non_zero_per_sample = (weights > 0).sum(dim=1)
    print(f"Non-zero experts per sample: {non_zero_per_sample.tolist()}")
    assert all(n == 2 for n in non_zero_per_sample.tolist()), "Should have exactly 2 non-zero experts per sample"
    print("Top-k selection verification: PASSED")

    # Check weights sum to 1.0 per sample
    weight_sums = weights.sum(dim=1)
    print(f"Expert weight sums per sample: {weight_sums.tolist()}")
    assert all(abs(s - 1.0) < 1e-5 for s in weight_sums.tolist()), "Expert weights should sum to 1.0"
    print("Weight normalization verification: PASSED")

    # ============================================
    # Test 4: Different configurations
    # ============================================
    print("\n" + "-" * 40)
    print("Test 4: Different configurations")
    print("-" * 40)

    # Test with num_experts=8
    model_8exp = CheMoE_PDG(
        n_genes=n_genes,
        mol_encoder_type='kpgt',
        num_experts=8,
        top_k=2
    ).to(device)

    pred_8exp = model_8exp(basal_expr_d, kpgt_embed_d, cell_idx_d)
    assert pred_8exp.shape == (batch_size, n_genes)
    print(f"num_experts=8: output shape={pred_8exp.shape} - PASSED")

    # Test with top_k=1
    model_topk1 = CheMoE_PDG(
        n_genes=n_genes,
        mol_encoder_type='kpgt',
        num_experts=4,
        top_k=1
    ).to(device)

    pred_topk1 = model_topk1(basal_expr_d, kpgt_embed_d, cell_idx_d)
    assert pred_topk1.shape == (batch_size, n_genes)
    stats_topk1 = model_topk1.get_expert_usage_stats()
    non_zero_topk1 = (stats_topk1['weights'] > 0).sum(dim=1)
    assert all(n == 1 for n in non_zero_topk1.tolist()), "top_k=1 should select exactly 1 expert"
    print(f"top_k=1: output shape={pred_topk1.shape}, experts selected={non_zero_topk1.tolist()} - PASSED")

    # ============================================
    # Test 5: CUDA test (if available)
    # ============================================
    if torch.cuda.is_available():
        print("\n" + "-" * 40)
        print("Test 5: CUDA device test")
        print("-" * 40)

        model_cuda = CheMoE_PDG(n_genes=n_genes).to('cuda')
        basal_cuda = torch.randn(batch_size, n_genes).cuda()
        mol_cuda = torch.randn(batch_size, 2304).cuda()
        cell_cuda = torch.randint(0, 10, (batch_size,)).cuda()

        pred_cuda = model_cuda(basal_cuda, mol_cuda, cell_cuda)
        assert pred_cuda.is_cuda, "Output should be on CUDA"
        assert pred_cuda.shape == (batch_size, n_genes)
        print(f"CUDA test: output device={pred_cuda.device}, shape={pred_cuda.shape} - PASSED")
    else:
        print("\n(Skipping CUDA test - not available)")

    # ============================================
    # Summary
    # ============================================
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print(f"\nModel Summary (KPGT):")
    print(f"  - Parameters: {params_kpgt:,}")
    print(f"  - Experts: 4, Top-k: 2")
    print(f"  - Global dim: 384 (3 x 128)")
    print(f"  - Expert input: 512 (384 + 128)")
    print(f"  - Output: {n_genes} genes")
