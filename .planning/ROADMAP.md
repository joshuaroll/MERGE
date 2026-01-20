# Roadmap: PDGrapher Baseline Models

## Current Milestone: M1 - Complete Fold 1 Baseline Evaluation

**Goal**: Achieve complete fold 1 evaluation results for all 10 models across all 9 cell lines with consistent top-k R² metrics.

**Success Criteria**:
- All 10 models evaluated with TopKEvaluator methodology
- Results consolidated in `data/topk_r2_results_fold1_01172026.csv`
- Model naming standardized (PascalCase)

---

## Phases

### Phase 1: TranSiGen Completion
**Goal**: Run missing TranSiGen MDAMB231 fold 1
**Scope**: Single model (TranSiGen base), single cell line (MDAMB231), fold 1
**Status**: Complete

**Deliverables**:
- [x] TranSiGen MDAMB231 fold 1 trained
- [x] Results added to CSV

---

### Phase 2: Biolord Metric Extraction
**Goal**: Extract top-k R² metrics from trained Biolord models
**Scope**: 9 cell lines, metrics extraction only (models already trained)
**Status**: Complete

**Deliverables**:
- [x] Extract metrics from Biolord logs/checkpoints for all 9 cells
- [x] Add to results CSV with standardized naming

---

### Phase 3: PDGrapher Evaluation
**Goal**: Run PDGrapher with proper DE metrics for all cell lines
**Scope**: 9 cell lines total (A549 re-evaluate, 8 others train from scratch)
**Status**: Complete

**Deliverables**:
- [x] A549, A375, BT20, HELA, HT29 forward models trained
- [x] MCF7, MDAMB231, PC3, VCAP complete
- [x] All results extracted with TopKEvaluator

**Notes**: PDGrapher requires separate forward/backward model training. Focus on forward (diseased→treated) for DE prediction.

---

### Phase 4: Results Consolidation
**Goal**: Merge all results into final CSV with standardized naming
**Scope**: Data consolidation, validation, and formatting
**Status**: Complete

**Deliverables**:
- [x] All 90 model×cell combinations in CSV (10 models × 9 cells)
- [x] Naming verified (PascalCase)
- [x] Metrics verified against TopKEvaluator standard

---

### Phase 5: CheMoE PDGrapher Adaptation
**Goal**: Adapt original CheMoE model to work with PDGrapher's 10,716-gene data
**Scope**: Create wrapper/training script, train all 9 cell lines fold 0
**Status**: Complete
**Source**: `/raid/home/joshua/projects/CheMoE`
**Plans:** 3 plans
**Completed**: 2026-01-19

Plans:
- [x] 05-01-PLAN.md — Create CheMoE_PDG model (model.py, __init__.py)
- [x] 05-02-PLAN.md — Create training script (train_chemoe_pdg.py)
- [x] 05-03-PLAN.md — Train all 9 cell lines and collect results

**Deliverables**:
- [x] CheMoE wrapper adapted for 10,716 genes (like TranSiGen adaptation)
- [x] Training script `train_chemoe_pdg.py` created
- [x] All 9 cell lines trained (fold 0)
- [x] Results extracted with TopKEvaluator

**Results**: Mean R² Top-20: 0.6555 (fold 0)

**Notes**:
- Original CheMoE uses 978 genes (L1000), needs adaptation to 10,716
- Uses MoE architecture: 4 experts, top-k=2 sparse routing
- Input: SMILES (KPGT or Morgan FP), cell embedding, basal expression → DE prediction
- No dose encoder (PDGrapher has no dose variation)
- Trained on fold 0; other baselines use fold 1 — retraining recommended for proper comparison

---

## Phase Dependencies

```
Phase 1 (TranSiGen) ──┐
                      │
Phase 2 (Biolord) ────┼──► Phase 4 (Consolidation)
                      │
Phase 3 (PDGrapher) ──┤
                      │
Phase 5 (CheMoE) ─────┘
```

Phases 1, 3, 5 can run in parallel. Phase 4 depends on all completing.

---

## Future Milestones (Backlog)

### M2: Fold 2-5 Expansion
Extend evaluation to remaining folds for statistical significance.

### M3: Ensemble Architecture
Combine model predictions using learned weights or stacking.

### M4: TranSiGen_CheMoE Integration
Integrate new CheMoE-style MoE variant into baseline comparison.

---

## Model Status Summary

| Model | Cells Complete | Status |
|-------|----------------|--------|
| ChemCPA | 9/9 | Complete |
| MultiDCP | 9/9 | Complete |
| MultiDCP_CheMoE | 9/9 | Complete |
| scGen | 9/9 | Complete |
| TranSiGen | 9/9 | Complete |
| TranSiGen_MoE_Sparse | 9/9 | Complete |
| TranSiGen_MoE_Balanced | 9/9 | Complete |
| Biolord | 9/9 | Complete |
| PDGrapher | 9/9 | Complete |
| CheMoE | 9/9 | Complete |

---

*Last Updated: 2026-01-19*
*Milestone: M1*
