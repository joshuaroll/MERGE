# Research Summary: Ensemble Learning for Gene Expression Prediction

**Domain:** High-dimensional regression ensemble (10,716 genes)
**Researched:** 2026-01-19
**Overall confidence:** HIGH

## Executive Summary

Research confirms that scikit-learn's native Ridge regression is the optimal choice for combining 10 gene expression prediction models. Key findings:

1. **No new dependencies required** - scikit-learn 1.5.2 (already installed) handles 10,716-dimensional output natively in 0.15 seconds
2. **Avoid sklearn's VotingRegressor/StackingRegressor** - These are designed for different estimator types, not precomputed predictions from external models
3. **Per-gene stacking already implemented** - The existing `ensemble.py` uses the correct approach (Ridge per-gene weights)
4. **Main gap: simple baselines** - Missing averaging, top-score selection, and systematic comparison table

The existing codebase is well-architected. The v2.0 milestone primarily needs to add simple ensemble methods and systematic evaluation, not architectural changes.

## Key Findings

**Stack:** Scikit-learn Ridge regression for meta-learners, PyTorch for advanced MLP if needed (both already available)

**Architecture:** Per-gene Ridge stacking is correct approach - learns (n_genes, n_models) weight matrix efficiently

**Critical pitfall:** sklearn's VotingRegressor/StackingRegressor expect sklearn estimators, not precomputed arrays - must use manual implementation (already done in ensemble.py)

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Simple Ensemble Baselines** - Implement trivial methods first
   - Addresses: Simple averaging, equal weights, top-score per sample
   - Avoids: Overengineering before having baselines to compare against

2. **OOF Prediction Consolidation** - Generate 3-fold OOF for all models
   - Addresses: Proper stacking without data leakage
   - Avoids: Using test data for meta-learner training

3. **Stacked Meta-Learners** - Build on existing CalibratedEnsemble
   - Addresses: Ridge (done), RidgeCV (upgrade), MLP (optional)
   - Avoids: Unnecessary complexity (XGBoost, LightGBM)

4. **Comparison Table** - Systematic evaluation
   - Addresses: v2.0 requirement to beat Biolord 0.7957
   - Avoids: Cherry-picking results

**Phase ordering rationale:**
- Simple baselines first: establish floor, quick wins
- OOF second: required foundation for proper stacking
- Meta-learners third: build on OOF infrastructure
- Comparison last: need all methods complete to compare

**Research flags for phases:**
- Phase 1-2: Standard patterns, unlikely to need research
- Phase 3 MLP: May need tuning research if Ridge doesn't improve over baselines
- Phase 4: No research needed, just execution

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Tested locally, all components work |
| Features | HIGH | Clear requirements from PROJECT.md |
| Architecture | HIGH | Existing ensemble.py is well-designed |
| Pitfalls | MEDIUM | May discover issues during OOF generation |

## Gaps to Address

- **OOF generation for precomputed models**: Need workflow for generating 3-fold predictions from models in different conda environments
- **Model alignment**: Some models may have different sample ordering - need alignment verification
- **Embedding-based stacking**: Current EmbeddingStackingEnsemble needs embeddings that may not exist for all models

## Technology Decisions (Final)

| Decision | Recommendation | Rationale |
|----------|----------------|-----------|
| Meta-learner | Ridge/RidgeCV | Native multi-output, fast, stable |
| OOF method | Manual 3-fold split | Models are precomputed, can't use cross_val_predict directly |
| Advanced meta-learner | PyTorch PerGeneLinear | Only if Ridge doesn't beat baselines |
| Experiment tracking | WandB | Already in use |

## Ready for Roadmap

Research complete. Key recommendations:

1. Use existing `ensemble.py` architecture as foundation
2. Add simple baselines (averaging, top-score) before complex methods
3. Upgrade Ridge to RidgeCV for automatic regularization
4. Skip XGBoost/LightGBM - overkill for this task
5. Focus on proper OOF generation across conda environments
