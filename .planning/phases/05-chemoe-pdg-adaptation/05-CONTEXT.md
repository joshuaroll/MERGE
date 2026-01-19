# Phase 5: CheMoE PDGrapher Adaptation - Context

**Gathered:** 2026-01-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Adapt the original CheMoE model (978 genes, L1000) to work with PDGrapher's 10,716-gene data. Create wrapper/training script, train all 9 cell lines on fold 1, and extract top-k R² metrics for baseline comparison.

Source: `/raid/home/joshua/projects/CheMoE`

</domain>

<decisions>
## Implementation Decisions

### Output Scaling
- **Optimization target**: Optimize for PDGrapher performance, not strict CheMoE architecture fidelity
- Allow architecture changes that improve 10,716-gene prediction

### Input Encoders
- **Molecular features**: Support BOTH Morgan fingerprints (CheMoE original) AND KPGT embeddings (TranSiGen style) via CLI flag
- **Basal expression**: Include as input modality (CheMoE style) — model takes diseased expression, predicts treated
- **Cell encoding**: Learned categorical embedding per cell line
- **Dose encoding**: Claude's discretion based on PDGrapher data format

### Expert Configuration
- **Number of experts**: 4 (match CheMoE default)
- **Training approach**: End-to-end (no expert pretraining) — simpler, matches other baselines
- **Gating style**: Sparse top-k=2 with weight re-normalization
- **Load balancing**: Yes, include auxiliary loss to encourage equal expert usage

### Data Pipeline
- **Data loader**: Reuse existing `PDGrapherDataset` from `train_transigen_pdg.py`
- **Splits**: PDGrapher splits (same as all other baselines for fair comparison)
- **Metrics**: TopKEvaluator standard (k=20/40/80) for comparison
- **WandB project**: `CheMoE_PDG_AE_DE`

### Claude's Discretion
- Expert network depth and architecture (scale appropriately for 10,716 output)
- Gating network capacity adjustments
- Dose encoding approach (categorical bins vs continuous MLP)
- Embedding dimensions for each modality

</decisions>

<specifics>
## Specific Ideas

- Should feel like a proper CheMoE adaptation, not a completely new model
- Core MoE philosophy preserved: 4 experts, sparse routing, conditional computation
- Use existing code patterns from this repo (TranSiGen adaptation, MultiDCP_CheMoE) as reference
- Training script should follow same CLI pattern as `train_transigen_pdg.py`

</specifics>

<deferred>
## Deferred Ideas

- Expert pretraining with diversity mechanisms — could be Phase 6 if end-to-end underperforms
- Cross-modal attention variant (CheMoE's CrossModalMoE) — future enhancement
- Ensemble integration with other models — belongs in M3 milestone

</deferred>

---

*Phase: 05-chemoe-pdg-adaptation*
*Context gathered: 2026-01-19*
