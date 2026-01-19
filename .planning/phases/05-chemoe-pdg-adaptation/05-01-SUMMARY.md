---
phase: 05-chemoe-pdg-adaptation
plan: 01
subsystem: model-architecture
tags: [moe, chemoe, pdgrapher, gene-expression, pytorch]

dependency-graph:
  requires: []
  provides: [chemoe_pdg_model, moe_gating, gene_aware_experts]
  affects: [05-02, 05-03]

tech-stack:
  added: []
  patterns: [sparse-gating, gene-aware-processing, load-balancing-loss]

key-files:
  created:
    - models/chemoe_pdg/__init__.py
    - models/chemoe_pdg/model.py
  modified: []

decisions:
  - id: mol-encoder-dual-support
    choice: "Support both KPGT (2304-dim) and Morgan FP (1024-dim)"
    rationale: "Flexibility for experiments; KPGT is preferred for similarity tasks"
  - id: gene-aware-processing
    choice: "Shared MLP across all genes with gene embeddings"
    rationale: "Parameter efficient; avoids 10,716x memory explosion"
  - id: sparse-gating
    choice: "Top-k=2 selection with softmax over selected experts only"
    rationale: "Matches CheMoE original; provides conditional computation"

metrics:
  duration: "~5 minutes"
  completed: "2026-01-19"
---

# Phase 05 Plan 01: CheMoE_PDG Model Architecture Summary

**One-liner:** CheMoE MoE architecture (4 experts, top-k=2 routing) adapted for PDGrapher's 10,716-gene prediction with dual molecular encoder support.

## What Was Built

### CheMoE_PDG Model (`models/chemoe_pdg/model.py`)

A complete Mixture-of-Experts model for predicting gene expression responses to chemical perturbations, featuring:

**Encoders:**
- `MolecularEncoder`: Dual support for KPGT (2304-dim) and Morgan FP (1024-dim) inputs
  - Architecture: input -> 256 -> 128 with LayerNorm + ReLU + Dropout
- `BasalEncoder`: Compresses 10,716-gene basal expression to 128-dim
  - Architecture: 10716 -> 256 -> 128 with LayerNorm
- `CellEmbedding`: Learned embeddings for 10 cell lines (128-dim)
- `GeneEmbedding`: Learnable per-gene embeddings (128-dim each)

**MoE Components:**
- `GatingNetwork`: Global features (384-dim) -> 4 expert weights
  - Uses top-k=2 sparse selection with softmax over selected only
  - Stores weights for load balancing loss computation
- `ExpertNetwork`: 4 expert MLPs (512 -> 256 -> 128 -> 1)
  - Shared weights across all 10,716 genes (parameter efficient)
  - Gene-aware processing via concatenated gene embeddings

**Key Methods:**
- `forward(basal_expr, mol_embed, cell_idx)`: Returns [batch, 10716] predictions
- `compute_load_balance_loss(weight=0.01)`: MSE between expert usage and uniform target
- `get_expert_usage_stats()`: Dict with per-expert usage for logging

### Parameter Counts
| Configuration | Parameters |
|--------------|------------|
| KPGT (2304-dim) | 5,484,424 |
| Morgan (1024-dim) | 5,156,744 |

### Architecture Diagram
```
Input:
  basal_expr [batch, 10716] -----> BasalEncoder -----> [batch, 128]
  mol_embed [batch, 2304/1024] --> MolecularEncoder -> [batch, 128]  --> Global Features [batch, 384]
  cell_idx [batch] -------------> CellEmbedding ----> [batch, 128]
                                                           |
                                                           v
                                                    GatingNetwork
                                                           |
                                                    [batch, 4] expert weights
                                                    (top-k=2 sparse)
                                                           |
           +-----------------------------------------------+
           |               |               |               |
           v               v               v               v
      Expert 0        Expert 1        Expert 2        Expert 3
           |               |               |               |
           +-----------------------------------------------+
                                   |
                                   v
                         Weighted Sum (sparse)
                                   |
                                   v
                         pred_treated [batch, 10716]
```

## Verification Results

All inline tests passed:
1. Shape tests: KPGT and Morgan inputs produce [batch, 10716] output
2. Gating tests: Exactly 2 experts selected per sample, weights sum to 1.0
3. Configuration tests: Works with num_experts=8 and top_k=1
4. Device tests: CUDA compatibility verified

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Molecular encoder | Dual KPGT/Morgan support | Flexibility for experiments |
| Gene processing | Shared MLP + gene embeddings | Parameter efficient (5.5M vs ~60B) |
| Gating style | Softmax over selected top-k | Matches CheMoE original |
| Expert count | 4 experts, top-k=2 | Proven configuration from CheMoE |

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

### Ready for Plan 02 (Training Script)
- Model can be imported: `from models.chemoe_pdg import CheMoE_PDG`
- Forward pass returns correct shape
- Load balancing loss available for training loop
- Cell line mapping provided: `CELL_LINE_TO_IDX`

### Dependencies Provided
- CheMoE_PDG class exportable
- Supports both KPGT and Morgan molecular features
- Load balance loss for regularization
- Expert usage stats for WandB logging

## Commits

| Commit | Type | Description |
|--------|------|-------------|
| c19fa11 | feat | Create CheMoE_PDG model for 10,716-gene prediction |

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| models/chemoe_pdg/__init__.py | Created | 13 |
| models/chemoe_pdg/model.py | Created | 481 |
