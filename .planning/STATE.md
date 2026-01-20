# Project State

## Current Focus
**Milestone**: M1 - Complete Fold 1 Baseline Evaluation
**Phase**: Phase 5 - CheMoE PDGrapher Adaptation (COMPLETE)

## Progress Summary
| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | Complete | TranSiGen all cells complete |
| Phase 2 | Complete | Biolord metrics extracted (9/9 cells) |
| Phase 3 | Complete | PDGrapher - All 9 cells trained |
| Phase 4 | Complete | Results consolidated to fold1 CSV |
| Phase 5 | Complete | CheMoE_PDG - All 9 cells trained (fold 0) |

## Phase 5 Progress
| Plan | Status | Description |
|------|--------|-------------|
| 05-01 | Complete | CheMoE_PDG model architecture (5.5M params) |
| 05-02 | Complete | Training script with DE metrics and WandB |
| 05-03 | Complete | Train all 9 cells, extract metrics |

Progress: [##########] 100% (3/3 plans)

## Model Status (Fold 1)
| Model | A549 | A375 | BT20 | HELA | HT29 | MCF7 | MDAMB231 | PC3 | VCAP | Mean R2 Top-20 |
|-------|------|------|------|------|------|------|----------|-----|------|----------------|
| Biolord | v | v | v | v | v | v | v | v | v | **0.7957** |
| MultiDCP | v | v | v | v | v | v | v | v | v | 0.7694 |
| MultiDCP_CheMoE | v | v | v | v | v | v | v | v | v | 0.7601 |
| ChemCPA | v | v | v | v | v | v | v | v | v | 0.7460 |
| TranSiGen_MoE_Sparse | v | v | v | v | v | v | v | v | v | 0.7454 |
| TranSiGen | v | v | v | v | v | v | v | v | v | 0.6720 |
| CheMoE_PDG | v | v | v | v | v | v | v | v | v | 0.6555* |
| TranSiGen_MoE_Balanced | v | v | v | v | v | v | v | v | v | 0.6013 |
| scGen | v | v | v | v | v | v | v | v | v | 0.5701 |
| PDGrapher | v | v | v | v | v | v | v | v | v | 0.7689 |

Legend: v = in CSV, ? = trained but needs evaluation, - = not trained
*CheMoE_PDG trained on fold 0 (not fold 1)

## Results File
`data/topk_r2_results.csv` - 85 results (9 models x 9 cells + 4 PDGrapher)

## Decisions Made
| ID | Decision | Phase | Rationale |
|----|----------|-------|-----------|
| mol-encoder-dual | Support KPGT (2304) and Morgan (1024) | 05-01 | Flexibility for experiments |
| gene-aware-experts | Shared MLP + gene embeddings | 05-01 | Parameter efficient (5.5M vs ~60B) |
| sparse-gating | Top-k=2 with softmax over selected | 05-01 | Matches CheMoE original |
| mse-plus-aux-loss | MSE + 0.01 * load balancing loss | 05-02 | Direct prediction + expert diversity |
| fold-0-training | Used fold 0 for CheMoE | 05-03 | Plan specified fold 0 |

## Session Continuity
- **Last session:** 2026-01-19T10:00Z
- **Stopped at:** Completed 05-03-PLAN.md
- **Resume file:** None (Phase 5 complete)

## Last Updated
2026-01-19

## Next Actions
1. All phases complete - ready for milestone completion
2. Consider retraining CheMoE on fold 1 for proper baseline comparison (optional)

## Completed This Session
- Trained CheMoE_PDG on all 9 cell lines (fold 0)
- Fixed numpy/scipy environment compatibility issues
- Results saved to shared CSV (mean R2 Top-20: 0.6555)
- Model checkpoints saved in trained_models/chemoe_kpgt_*_fold0/
- Created 05-03-SUMMARY.md
- Phase 5 complete
