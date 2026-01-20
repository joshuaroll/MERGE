# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-01-19)

**Core value:** Standardized, reproducible baseline comparisons
**Current focus:** v1.0 complete — ready for next milestone

## Current Position

**Milestone:** v1.0 Complete
**Phase:** Ready for M2 planning
**Status:** Milestone shipped

Last activity: 2026-01-19 — v1.0 milestone complete

Progress: [##########] 100% (5/5 phases)

## v1.0 Summary

All 10 models evaluated across 9 cell lines:

| Model | Mean R² Top-20 |
|-------|----------------|
| Biolord | **0.7957** |
| MultiDCP | 0.7694 |
| PDGrapher | 0.7689 |
| MultiDCP_CheMoE | 0.7601 |
| ChemCPA | 0.7460 |
| TranSiGen_MoE_Sparse | 0.7454 |
| TranSiGen | 0.6720 |
| CheMoE_PDG* | 0.6555 |
| TranSiGen_MoE_Balanced | 0.6013 |
| scGen | 0.5701 |

*fold 0

## Results File

`data/topk_r2_results.csv` — 124 results (10 models × 9 cells + extras)

## Decisions Made (v1.0)

| ID | Decision | Outcome |
|----|----------|---------|
| metric-standard | TopKEvaluator (Pearson² on DE) | ✓ Consistent |
| output-location | Single CSV | ✓ Consolidated |
| chemoe-architecture | 4 experts, top-k=2 | ✓ Working |

## Session Continuity

- **Last session:** 2026-01-19
- **Completed:** v1.0 milestone
- **Resume file:** None

## Next Actions

1. `/gsd:new-milestone` — start M2 (Fold 2-5 Expansion)
2. Consider retraining CheMoE on fold 1 for proper comparison

## Last Updated

2026-01-19 — v1.0 milestone shipped
