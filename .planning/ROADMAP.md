# Milestone v2.0: Ensemble Architecture

**Status:** In Progress
**Goal:** Build ensemble models that combine all 10 baseline models to beat Biolord (0.7957 mean R² Top-20)

## Overview

Leverage existing `ensemble.py` infrastructure to create comprehensive ensemble comparisons. Primary work: OOF generation, simple baselines, and systematic evaluation.

## Phases

### Phase 1: Simple Ensemble Baselines

**Goal**: Implement trivial ensemble methods as performance floor
**Depends on**: None
**Plans**: 1 plan

**Deliverables:**
- [ ] Mean averaging across all 10 models
- [ ] Weighted average with inverse-MSE weights
- [ ] Top-score selection (pick best model per sample)
- [ ] Baseline metrics logged to results CSV

**Research needed:** No (standard patterns)

---

### Phase 2: OOF Prediction Generation

**Goal**: Generate out-of-fold predictions for proper stacking
**Depends on**: None (parallel with Phase 1)
**Plans**: 1 plan

**Deliverables:**
- [ ] Fold-specific directory structure (predictions/fold{0,1,2}/)
- [ ] OOF predictions for all 10 models × 9 cells × 3 folds
- [ ] Sample alignment verification
- [ ] Storage: ~3 GB predictions

**Research needed:** No (workflow documented in ARCHITECTURE.md)

**Compute cost:** ~270 model training runs

---

### Phase 3: Stacked Meta-Learners

**Goal**: Train ridge and MLP meta-learners on OOF predictions
**Depends on**: Phase 2
**Plans**: 1 plan

**Deliverables:**
- [ ] RidgeCV with auto-regularization (upgrade from fixed Ridge)
- [ ] Per-gene ridge stacking with proper OOF training
- [ ] Optional: MLP meta-learner if ridge doesn't beat baselines
- [ ] Evaluate for data leakage (compare train vs test performance)

**Research needed:** Maybe (if MLP tuning required)

---

### Phase 4: Comparison Table & Evaluation

**Goal**: Systematic comparison of all methods
**Depends on**: Phases 1, 2, 3
**Plans**: 1 plan

**Deliverables:**
- [ ] Comparison table: all ensemble methods vs individual models
- [ ] TopKEvaluator metrics (R² Top-20, Top-40, Top-80)
- [ ] Verify ensemble beats Biolord (0.7957)
- [ ] Update data/topk_r2_results.csv

**Success criteria:** At least one ensemble method achieves R² Top-20 > 0.7957

---

### Phase 5: Embedding Stacking (Optional)

**Goal**: Explore embedding-based stacking as alternative
**Depends on**: Phase 3 (compare results first)
**Plans**: 1 plan

**Deliverables:**
- [ ] Extract embeddings from 4+ supported models
- [ ] Train embedding-based meta-MLP
- [ ] Compare to prediction-based stacking
- [ ] Keep if competitive, drop otherwise

**Research needed:** No (implementation exists in EmbeddingStackingEnsemble)

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Meta-learner | Ridge/RidgeCV | Native multi-output, 0.15s for 10K genes |
| OOF folds | 3 (not 5) | User specified, sufficient for stacking |
| Advanced meta-learner | MLP only if ridge fails | Avoid unnecessary complexity |
| Embedding stacking | Optional | Explore but don't commit unless competitive |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Data leakage | Medium | Strict OOF protocol, compare train vs test |
| Biolord dominance | Medium | Check weight distribution, verify diversity |
| Per-gene overfitting | Medium | Compare to global weights, increase regularization |
| Compute cost (270 runs) | High | Parallelize across GPUs, cache predictions |

## Success Criteria

- [ ] All 4 simple ensemble methods evaluated
- [ ] OOF predictions generated for all 10 models × 9 cells × 3 folds
- [ ] At least one ensemble beats Biolord (R² Top-20 > 0.7957)
- [ ] Comparison table with all methods created

---

*Created: 2026-01-19 after research phase*
