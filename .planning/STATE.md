# Project State

## Current Focus
**Milestone**: M1 - Complete Fold 1 Baseline Evaluation
**Phase**: Phase 5 - CheMoE PDGrapher Adaptation

## Progress Summary
| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | Complete | TranSiGen all cells complete |
| Phase 2 | Complete | Biolord metrics extracted (9/9 cells) |
| Phase 3 | In Progress | PDGrapher - A549 trained, 8 cells need training |
| Phase 4 | Complete | Results consolidated to fold1 CSV |
| Phase 5 | In Progress | CheMoE_PDG - Plan 02 complete (training script) |

## Phase 5 Progress
| Plan | Status | Description |
|------|--------|-------------|
| 05-01 | Complete | CheMoE_PDG model architecture (5.5M params) |
| 05-02 | Complete | Training script with DE metrics and WandB |
| 05-03 | Pending | Train all 9 cells, extract metrics |

Progress: [####------] 67% (2/3 plans)

## Model Status (Fold 1)
| Model | A549 | A375 | BT20 | HELA | HT29 | MCF7 | MDAMB231 | PC3 | VCAP | Mean R2 Top-20 |
|-------|------|------|------|------|------|------|----------|-----|------|----------------|
| Biolord | v | v | v | v | v | v | v | v | v | **0.7957** |
| MultiDCP | v | v | v | v | v | v | v | v | v | 0.7694 |
| MultiDCP_CheMoE | v | v | v | v | v | v | v | v | v | 0.7601 |
| ChemCPA | v | v | v | v | v | v | v | v | v | 0.7460 |
| TranSiGen_MoE_Sparse | v | v | v | v | v | v | v | v | v | 0.7454 |
| TranSiGen | v | v | v | v | v | v | v | v | v | 0.6720 |
| TranSiGen_MoE_Balanced | v | v | v | v | v | v | v | v | v | 0.6013 |
| scGen | v | v | v | v | v | v | v | v | v | 0.5701 |
| PDGrapher | - | - | - | - | - | v | v | v | v | 0.7689* |
| CheMoE_PDG | - | - | - | - | - | - | - | - | - | pending |

Legend: v = in CSV, ? = trained but needs evaluation, - = not trained
*PDGrapher mean based on 4 cells only (MCF7, MDAMB231, PC3, VCAP)

## Results File
`data/topk_r2_results_fold1_01172026.csv` - 76 results (8 models x 9 cells + 4 PDGrapher)

## Decisions Made
| ID | Decision | Phase | Rationale |
|----|----------|-------|-----------|
| mol-encoder-dual | Support KPGT (2304) and Morgan (1024) | 05-01 | Flexibility for experiments |
| gene-aware-experts | Shared MLP + gene embeddings | 05-01 | Parameter efficient (5.5M vs ~60B) |
| sparse-gating | Top-k=2 with softmax over selected | 05-01 | Matches CheMoE original |
| mse-plus-aux-loss | MSE + 0.01 * load balancing loss | 05-02 | Direct prediction + expert diversity |

## Session Continuity
- **Last session:** 2026-01-19T05:10Z
- **Stopped at:** Completed 05-02-PLAN.md
- **Resume file:** .planning/phases/05-chemoe-pdg-adaptation/05-03-PLAN.md

## Last Updated
2026-01-19

## Next Actions
1. Execute 05-03-PLAN.md: Train all 9 cell lines, extract metrics
2. Complete PDGrapher training for 5 remaining cell lines

## Completed This Session
- Created train_chemoe_pdg.py training pipeline
- Verified training with 2-epoch dry run on MCF7 fold 0
- Loss decreased from 0.0375 to 0.0220 (41% reduction)
- R2 Top-20 = 0.69 after 2 epochs
- Results saved to shared CSV successfully
- Created 05-02-SUMMARY.md
