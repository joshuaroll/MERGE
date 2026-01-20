# PROJECT.md

## What This Is

A comprehensive baseline evaluation framework for differential gene expression prediction, comparing 10 perturbation models across 9 cell lines using consistent TopKEvaluator metrics.

## Core Value

Standardized, reproducible baseline comparisons for perturbation prediction research.

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

### Active (Next Milestone)

- [ ] Folds 2-5 evaluation for statistical significance
- [ ] Cross-fold variance analysis
- [ ] Ensemble architecture exploration

### Out of Scope

- New model development (use existing architectures)
- Hyperparameter optimization (use defaults)
- Production deployment

## Tech Stack

- Python 3.10+ with PyTorch 2.0+
- Conda environments: `mdcp_env`, `scgen_env`, `biolord_env`
- WandB for experiment tracking
- Data: 182K samples × 10,716 genes from PDGrapher splits

## Key Decisions

| Decision | Choice | Outcome |
|----------|--------|---------|
| Metric Standard | TopKEvaluator (Pearson² on DE) | ✓ Consistent across all models |
| Output Location | `data/topk_r2_results.csv` | ✓ Single source of truth |
| Naming Convention | PascalCase | ✓ Standardized |
| CheMoE Architecture | 4 experts, top-k=2, gene embeddings | ✓ 5.5M params, working |

## Constraints

- Must use existing PDGrapher train/val/test splits
- Data shared with MultiDCP_pdg project
- TopKEvaluator in `data_loader.py` is canonical

## References

- Codebase map: `.planning/codebase/`
- Milestone history: `.planning/MILESTONES.md`
- v1.0 archive: `.planning/milestones/v1.0-ROADMAP.md`

---
*Last updated: 2026-01-19 after v1.0 milestone*
