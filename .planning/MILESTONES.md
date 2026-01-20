# Project Milestones: PDGrapher Baseline Models

## v1.0 Fold 1 Baseline Evaluation (Shipped: 2026-01-19)

**Delivered:** Complete fold 1 evaluation of 10 perturbation prediction models across 9 cell lines with consistent TopKEvaluator metrics.

**Phases completed:** 1-5 (3 documented plans)

**Key accomplishments:**

- Evaluated all 10 models (ChemCPA, MultiDCP, MultiDCP_CheMoE, scGen, TranSiGen, TranSiGen_MoE_Sparse, TranSiGen_MoE_Balanced, Biolord, PDGrapher, CheMoE_PDG) with 9/9 cell lines each
- Created CheMoE_PDG model adapted for 10,716-gene prediction (5.5M parameters, MoE architecture)
- Consolidated 124 results to `data/topk_r2_results.csv` with standardized PascalCase naming
- Established Biolord as top performer (0.7957 mean R² Top-20)

**Stats:**

- 14 commits
- 5 phases, 3 documented plans
- 2 days (2026-01-18 → 2026-01-19)
- 124 model×cell evaluation results

**Git range:** `0b852e1` → `ea92dba`

**What's next:** M2 - Fold 2-5 Expansion for statistical significance

---
