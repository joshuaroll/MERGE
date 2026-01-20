# PROJECT.md

## What This Is

A comprehensive baseline evaluation and ensemble framework for differential gene expression prediction, comparing 10 perturbation models across 9 cell lines and combining them via learned ensemble methods.

## Core Value

Standardized, reproducible baseline comparisons and ensemble combinations for perturbation prediction research.

## Current Milestone: v2.0 Ensemble Architecture

**Goal:** Build ensemble models that combine all 10 baseline models to outperform the best individual model (Biolord at 0.7957 mean R² Top-20).

**Target features:**
- Simple ensemble methods (averaging, top score, weighted average)
- Multi-fold OOF predictions (3 folds) for stacking
- Stacked meta-learners (multiple architectures: ridge, MLP, etc.)
- Embedding-based stacking (explore vs prediction-based)
- Comparison table of all methods vs individual models

## Current State (v1.0 Shipped)

**Shipped:** 2026-01-19

All 10 models evaluated with 9/9 cell lines:
- 124 results in `data/topk_r2_results.csv`
- Consistent TopKEvaluator methodology (Pearson² on DE for top-k genes)
- PascalCase naming standardized

**Top Performers (Mean R² Top-20):**
1. Biolord: 0.7957
2. MultiDCP: 0.7694
3. PDGrapher: 0.7689

## Requirements

### Validated (v1.0)

- ✓ All 10 models evaluated with TopKEvaluator — v1.0
- ✓ Fold 1 results for 9 cell lines consolidated — v1.0
- ✓ Metrics verified against canonical implementation — v1.0
- ✓ Model results ready for ensemble combination — v1.0

### Active (v2.0)

- [ ] Simple ensemble: averaging of all 10 model predictions
- [ ] Simple ensemble: top score selection per sample
- [ ] Simple ensemble: weighted average with learned weights
- [ ] Multi-fold: OOF predictions for 3 folds (10 models × 9 cells × 3 folds)
- [ ] Stacked meta-learner: ridge regression on OOF predictions
- [ ] Stacked meta-learner: MLP on OOF predictions
- [ ] Stacked meta-learner: embedding-based (explore)
- [ ] Comparison table: all ensemble methods vs individual models
- [ ] Beat Biolord's 0.7957 mean R² Top-20

### Out of Scope

- New model development (use existing 10 models)
- Hyperparameter optimization of base models
- Production deployment
- Full 5-fold evaluation (3 folds sufficient for OOF)

## Tech Stack

- Python 3.10+ with PyTorch 2.0+
- Conda environments: `mdcp_env`, `scgen_env`, `biolord_env`
- WandB for experiment tracking
- Data: 182K samples × 10,716 genes from PDGrapher splits
- Scikit-learn for simple meta-learners (ridge, etc.)

## Key Decisions

| Decision | Choice | Outcome |
|----------|--------|---------|
| Metric Standard | TopKEvaluator (Pearson² on DE) | ✓ Consistent across all models |
| Output Location | `data/topk_r2_results.csv` | ✓ Single source of truth |
| Naming Convention | PascalCase | ✓ Standardized |
| CheMoE Architecture | 4 experts, top-k=2, gene embeddings | ✓ 5.5M params, working |
| OOF Folds | 3 folds (not 5) | — Pending |
| Embedding-based stacking | Explore, keep if competitive | — Pending |

## Constraints

- Must use existing PDGrapher train/val/test splits
- Data shared with MultiDCP_pdg project
- TopKEvaluator in `data_loader.py` is canonical
- CheMoE_PDG needs retraining on fold 1 (currently fold 0)

## References

- Codebase map: `.planning/codebase/`
- Milestone history: `.planning/MILESTONES.md`
- v1.0 archive: `.planning/milestones/v1.0-ROADMAP.md`

---
*Last updated: 2026-01-19 after v2.0 milestone start*
