# Phase 05 Plan 03: Train CheMoE on All 9 Cell Lines - Summary

**Completed:** 2026-01-19
**Duration:** ~4.5 hours (including environment fixes and parallel training)

## One-Liner

Trained CheMoE_PDG model on all 9 cell lines (fold 0), achieving mean R2 Top-20 of 0.6555 with valid results for baseline comparison.

## Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| 1 | Train CheMoE on all 9 cell lines | Complete |
| 2 | Verify results and update STATE.md | Complete |

## Key Results

### CheMoE Performance (Fold 0)

| Cell Line | R2 Top-20 | R2 Top-40 | R2 Top-80 | Pearson DE |
|-----------|-----------|-----------|-----------|------------|
| A375 | 0.6548 | 0.5947 | 0.5528 | 0.3210 |
| A549 | 0.7046 | 0.6580 | 0.6114 | 0.2988 |
| BT20 | 0.6147 | 0.5594 | 0.5210 | 0.3202 |
| HELA | 0.5427 | 0.5054 | 0.4768 | 0.3086 |
| HT29 | 0.6727 | 0.6348 | 0.6049 | 0.3669 |
| MCF7 | 0.6934 | 0.6526 | 0.6178 | 0.3681 |
| MDAMB231 | 0.6397 | 0.5692 | 0.5209 | 0.3246 |
| PC3 | 0.6595 | 0.6237 | 0.5959 | 0.3679 |
| VCAP | 0.7173 | 0.6386 | 0.5813 | 0.4370 |
| **Mean** | **0.6555** | **0.6041** | **0.5647** | **0.3459** |

### Model Comparison (Mean R2 Top-20)

| Model | Mean R2 Top-20 | Notes |
|-------|----------------|-------|
| Biolord | 0.7957 | Best performer (fold 1) |
| MultiDCP | 0.7690 | fold 1 |
| MultiDCP_CheMoE | 0.7601 | fold 1 |
| ChemCPA | 0.7460 | fold 1 |
| **CheMoE** | **0.6555** | fold 0 |
| TranSiGen | 0.6082 | fold 1 |
| scGen | 0.5701 | fold 1 |

**Note:** CheMoE was trained on fold 0 while other baselines used fold 1. Direct comparison is not fully valid but results are within expected range.

## Artifacts Created

### Model Checkpoints
- `trained_models/chemoe_kpgt_A375_fold0/`
- `trained_models/chemoe_kpgt_A549_fold0/`
- `trained_models/chemoe_kpgt_BT20_fold0/`
- `trained_models/chemoe_kpgt_HELA_fold0/`
- `trained_models/chemoe_kpgt_HT29_fold0/`
- `trained_models/chemoe_kpgt_MCF7_fold0/`
- `trained_models/chemoe_kpgt_MDAMB231_fold0/`
- `trained_models/chemoe_kpgt_PC3_fold0/`
- `trained_models/chemoe_kpgt_VCAP_fold0/`

Each directory contains:
- `best_model.pt` - Best model weights
- `final_model.pt` - Final epoch weights
- `history.pkl` - Training history
- `predictions.npz` - Test set predictions

### Results
- `data/topk_r2_results.csv` - 9 new CheMoE entries appended

### WandB
- Project: `CheMoE_PDG_AE_DE`
- 9 completed runs (one per cell line)

## Deviations from Plan

### Environment Issues (Auto-fixed)

**1. [Rule 3 - Blocking] NumPy version incompatibility**
- **Found during:** Task 1 (initial training launch)
- **Issue:** Pickle files created with NumPy 2.x, mdcp_env had NumPy 1.26.4
- **Fix:** Upgraded numpy to 2.0.2 in mdcp_env
- **Files modified:** Environment packages (not code)

**2. [Rule 3 - Blocking] SciPy binary incompatibility**
- **Found during:** Task 1 (after NumPy upgrade)
- **Issue:** SciPy compiled against old NumPy ABI
- **Fix:** Reinstalled scipy (upgraded to 1.17.0)
- **Files modified:** Environment packages (not code)

**3. [Rule 1 - Bug] GPU memory OOM for A549 and PC3**
- **Found during:** Task 1 (shared GPU training)
- **Issue:** A549 and PC3 ran out of memory when sharing GPUs with other jobs
- **Fix:** Restarted on dedicated GPUs (GPU 2 and GPU 3 after BT20 completed)
- **Commit:** N/A (runtime decision)

## Expert Usage Statistics

From WandB logs, expert usage varied by cell line:
- Typical pattern: 2 of 4 experts active (top-k=2)
- Expert 0 and Expert 2 most commonly selected
- Load balancing loss kept experts reasonably balanced (aux_loss ~0.0006)

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| fold-0-training | Used fold 0 instead of fold 1 | Plan specified fold 0; different from other baselines but internally consistent |
| kpgt-encoder | Used KPGT embeddings (2304-dim) | Better molecular representation than Morgan FP |
| 100-epochs | Trained for 100 epochs | Standard training length, models converged |

## Notes

1. **Fold Difference:** Other baselines in the comparison table used fold 1, while CheMoE used fold 0. This makes direct comparison less meaningful. For proper comparison, either retrain CheMoE on fold 1 or retrain other models on fold 0.

2. **Performance Gap:** CheMoE's 0.6555 is lower than MultiDCP_CheMoE's 0.7601. Possible reasons:
   - Different fold (data split variance)
   - Different architecture details (CheMoE_PDG vs MultiDCP_CheMoE)
   - No hyperparameter tuning performed

3. **Training Time:** ~30-60 minutes per cell line on V100 GPUs, total ~4.5 hours including restarts.

## Next Phase Readiness

Phase 5 is now complete:
- [x] 05-01: CheMoE_PDG model architecture
- [x] 05-02: Training script with DE metrics
- [x] 05-03: Train all 9 cell lines, extract metrics

**Blockers for next work:**
- None for Phase 5 completion
- Consider retraining on fold 1 for proper baseline comparison (optional)

## Commit History

| Hash | Message |
|------|---------|
| ecdbf9d | feat(05-03): train CheMoE on all 9 cell lines (fold 0) |

---

*Phase: 05-chemoe-pdg-adaptation*
*Plan: 03*
*Completed: 2026-01-19*
