# Domain Pitfalls: Ensemble/Stacking for Gene Expression Prediction

**Domain:** Ensemble methods for high-dimensional perturbation prediction
**Researched:** 2026-01-19
**Context:** 10 models, 10,716 genes, 3-fold OOF stacking, target: beat Biolord (0.7957 R2 Top-20)

---

## Critical Pitfalls

Mistakes that cause invalid results or require complete rework.

### Pitfall 1: Data Leakage Through Meta-Learner Training

**What goes wrong:** Using the same data to both generate base model predictions AND train the meta-learner. This inflates ensemble performance because the meta-learner sees the same samples used to train base models.

**Why it happens:** Intuitive approach is "train base models, get predictions, train meta-learner on predictions." But if base models were trained on fold X and you use fold X predictions for meta-learner training, you have leakage.

**Consequences:**
- Ensemble appears to beat all base models in validation
- Performance collapses on truly held-out test data
- Published results are not reproducible

**Your codebase risk (HIGH):**
```python
# From ensemble.py StackingEnsemble.train() - PROBLEMATIC:
# Split into train/val for meta-learning
val_size = min(500, n_samples // 5)
train_diseased = diseased[:-val_size]
train_treated = treated[:-val_size]
# Models trained on train_diseased, then evaluated on val_diseased
# But val_diseased may overlap with original training data!
```

**Prevention - Proper OOF (Out-of-Fold) Protocol:**
```
For 3-fold stacking:

1. Split data into 3 folds: A, B, C

2. For each base model:
   - Train on A+B, predict on C -> OOF_C
   - Train on A+C, predict on B -> OOF_B
   - Train on B+C, predict on A -> OOF_A
   - Concatenate: OOF = [OOF_A, OOF_B, OOF_C]

3. Train meta-learner on OOF predictions (never seen during base training)

4. Evaluate on held-out TEST set (never used for OOF generation)
```

**Detection:** If ensemble dramatically outperforms best base model (e.g., +20% improvement), suspect leakage. True ensemble improvements are typically 1-5%.

---

### Pitfall 2: Evaluating Ensemble on Meta-Learner Training Data

**What goes wrong:** Computing final metrics on the same data used to learn combination weights.

**Why it happens:** After training meta-learner on validation split, reporting metrics on that same validation split.

**Consequences:**
- Optimistic bias in reported metrics
- Ensemble weights overfit to validation samples
- Fails to generalize to new perturbations

**Your codebase risk (HIGH):**
```python
# From ensemble.py CalibratedEnsemble - Issue at lines 1391-1400:
# Training ensemble on scaled_train_preds, then evaluating on train_treated
ensemble_pred = np.zeros((len(train_idx), n_genes))
# ...
ensemble_results = self._evaluate_model(ensemble_pred, train_treated, train_diseased)
# This evaluates on the SAME data used for weight learning!
```

**Prevention:**
```
Correct evaluation hierarchy:

DATA SPLIT:
├── Train (60%) - for base model training
├── Validation (20%) - for meta-learner weight learning
└── Test (20%) - for FINAL metrics only

Never compute final metrics on validation data.
If using 5-fold CV internally, report on held-out test fold.
```

**Detection:** Run ensemble on completely fresh data (new fold, new cell line). If performance drops significantly from validation, weights were overfit.

---

### Pitfall 3: Using Test Fold for Hyperparameter Selection

**What goes wrong:** Choosing ensemble architecture (ridge vs MLP, regularization strength) based on test set performance.

**Why it happens:** "Let's try different meta-learners and see which works best on test set."

**Consequences:**
- Test set effectively becomes validation set
- No true held-out evaluation
- Reported performance is optimistic

**Your codebase risk (MEDIUM):**
```python
# If running: evaluate_calibrated_ensemble with different lambda_reg values
# and picking best based on test R2 Top-20, this is leakage
lambda_reg = 0.01  # How was this chosen? If based on test performance = leakage
```

**Prevention:**
1. Fix hyperparameters BEFORE seeing test data
2. Use nested cross-validation if tuning is needed
3. Pre-register analysis plan: "We will use ridge regression with lambda=0.01"

---

### Pitfall 4: Misaligned Predictions Across Models

**What goes wrong:** Model predictions are not aligned to same samples/genes, causing incorrect combinations.

**Why it happens:**
- Different models use different preprocessing
- Sample ordering changes between runs
- Gene filtering differs (some models use top 5000 HVGs)

**Consequences:**
- Predictions combined for wrong samples
- Gene-level weights applied to wrong genes
- Garbage ensemble predictions

**Your codebase evidence:**
```python
# From ensemble.py PrecomputedEnsemble.load_predictions():
# Alignment check exists but is imperfect:
majority_count = max(sample_counts.keys(), key=lambda k: len(sample_counts[k]))
# Only checks sample COUNT, not sample IDENTITY
# Two models with 1000 samples each could have different samples!
```

**Prevention:**
```python
# Always store and verify sample identifiers:
np.savez(path,
    predictions=predictions,
    sample_ids=sample_ids,  # CRITICAL: unique identifiers
    gene_names=gene_names,  # CRITICAL: column order
)

# When loading, verify alignment:
for model_name in models:
    assert np.array_equal(loaded[model_name]['sample_ids'], reference_sample_ids)
    assert np.array_equal(loaded[model_name]['gene_names'], reference_gene_names)
```

**Detection:** Check if ensemble is worse than simple averaging. Misalignment often manifests as random-level performance.

---

### Pitfall 5: Forgetting to Handle Missing Predictions

**What goes wrong:** Some models fail to produce predictions for certain samples, creating NaN/missing values in the stacking matrix.

**Why it happens:**
- Model crashes on specific inputs
- Out-of-distribution samples
- Memory errors on large batches

**Consequences:**
- NaN propagation destroys meta-learner training
- Silent failures: ensemble runs but produces garbage
- Inconsistent sample counts across models

**Your codebase evidence:**
```python
# From ensemble.py PrecomputedModel.predict():
if self._predictions.shape[0] != n_samples:
    print(f"Warning: {self._name} has {self._predictions.shape[0]} predictions "
          f"but {n_samples} samples requested. Using available predictions.")
    return self._predictions[:n_samples]  # DANGEROUS: sample mismatch!
```

**Prevention:**
```python
# Explicit handling:
predictions = []
valid_mask = np.ones(n_samples, dtype=bool)

for model in models:
    pred = model.predict(X)
    if pred is None or np.any(np.isnan(pred)):
        valid_mask &= ~np.isnan(pred).any(axis=1)
    predictions.append(pred)

# Only use samples where ALL models have valid predictions
final_predictions = [p[valid_mask] for p in predictions]
```

---

## Moderate Pitfalls

Mistakes that cause suboptimal performance or technical debt.

### Pitfall 6: Ignoring Model Correlation (Diversity Failure)

**What goes wrong:** Combining highly correlated models provides minimal ensemble benefit.

**Why it happens:**
- Models trained on same data with similar architectures
- MoE variants (TranSiGen, TranSiGen_SparseMoE, TranSiGen_BalancedMoE) may be nearly identical
- Biolord dominates, other models contribute noise

**Your context:** 10 models include:
- 3 TranSiGen variants (likely correlated)
- 2 MultiDCP variants (likely correlated)
- PerGeneLinear baseline (uncorrelated but weak)

**Consequences:**
- Ensemble ≈ best single model
- Wasted computation on redundant models
- Overfitting risk from adding parameters without information gain

**Detection - Correlation Analysis:**
```python
# Compute prediction correlation matrix
corr_matrix = np.zeros((n_models, n_models))
for i, model_i in enumerate(models):
    for j, model_j in enumerate(models):
        # Flatten predictions and compute correlation
        corr, _ = pearsonr(predictions[i].flatten(), predictions[j].flatten())
        corr_matrix[i, j] = corr

# If correlation > 0.95 between models, they're redundant
print("High correlation pairs:", np.where(corr_matrix > 0.95))
```

**Prevention:**
- Measure diversity before building ensemble
- Consider removing highly correlated models
- Use regularization that penalizes similar weights

---

### Pitfall 7: Per-Gene Overfitting with 10,716 Outputs

**What goes wrong:** Learning separate weights for each gene leads to massive overfitting when n_samples << n_genes * n_models.

**Your codebase:**
```python
# From ensemble.py StackingEnsemble - learns n_genes * n_models parameters:
self._meta_weights = np.zeros((n_models, n_genes))  # 10 * 10716 = 107,160 parameters
for g in range(n_genes):
    # Ridge regression per gene with only val_size samples
    # If val_size=500, we're fitting 10 parameters with 500 points per gene
```

**Why it's risky:**
- 10 models * 10,716 genes = 107,160 weight parameters
- Validation set of 500 samples means 500 data points per gene
- Ridge regularization helps, but still risks overfitting

**Consequences:**
- Weights fit noise in validation set
- Poor generalization to test set
- Ensemble can perform WORSE than best single model

**Prevention:**
```python
# Option 1: Shared weights across genes (10 parameters total)
weights = learn_global_weights(val_preds, val_targets)  # Single set of model weights

# Option 2: Gene-group weights (reduce dimensionality)
# Group genes by expression level or pathway
gene_groups = cluster_genes(n_groups=100)
group_weights = np.zeros((n_models, n_groups))

# Option 3: Stronger regularization
lambda_reg = 1.0  # Much higher than 0.01
```

---

### Pitfall 8: Scale Mismatch Across Models

**What goes wrong:** Models predict in different scales, causing weighted combination to be dominated by one model.

**Why it happens:**
- Different output normalization (z-score vs raw counts)
- Different magnitude of predictions (one model predicts [-10, 10], another [-1, 1])
- Neural networks with different final layer scales

**Your codebase addresses this:**
```python
# CalibratedEnsemble learns scale factors - GOOD approach
scale = np.clip(np.sum(all_pred * all_true) / np.sum(all_pred**2), 0.5, 3.0)
self._scale_factors[m] = scale
```

**But risk remains:** Scale calibration on Top-100 genes may not generalize to all genes.

**Prevention:**
```python
# Standardize each model's predictions before combination:
for m in range(n_models):
    mean_pred = np.mean(predictions[m])
    std_pred = np.std(predictions[m])
    predictions[m] = (predictions[m] - mean_pred) / std_pred

# Or use rank-based combination (immune to scale)
```

---

### Pitfall 9: Simple Averaging Beats Stacking

**What goes wrong:** Complex stacking ensemble underperforms simple average, wasting effort.

**When this happens:**
- One model (Biolord) dominates at 0.7957 vs others at ~0.76
- Meta-learner overfits to noise
- High model correlation reduces diversity benefit

**Your context risk (MEDIUM-HIGH):**
- Biolord is significantly better than #2 (MultiDCP at 0.7694)
- Gap of 0.026 R2 is large
- Stacking may just learn to weight Biolord at ~1.0

**Detection:**
```python
# Always compare ensemble to baselines:
baselines = {
    'simple_average': np.mean(all_predictions, axis=0),
    'best_model': predictions[best_model_idx],
    'trimmed_mean': scipy.stats.trim_mean(all_predictions, 0.1, axis=0),
}

for name, pred in baselines.items():
    metrics = evaluate(pred)
    print(f"{name}: R2 Top-20 = {metrics['r2_top20']:.4f}")
```

**Prevention:**
1. Start with simple averaging as baseline
2. Only use stacking if it beats averaging by >1%
3. Consider "model selection" (pick best per sample) as alternative

---

### Pitfall 10: Numerical Instability in Ridge Regression

**What goes wrong:** Ridge regression solution becomes numerically unstable with ill-conditioned matrices.

**Why it happens:**
- Highly correlated model predictions
- Small regularization parameter
- Float32 precision issues

**Your codebase:**
```python
# From ensemble.py - potential instability:
XtX = X.T @ X + lambda_reg * np.eye(n_models)
weights = np.linalg.solve(XtX, Xty)
```

**Consequences:**
- Exploding weights (some models get weight = 1000)
- NaN propagation
- Unreproducible results

**Prevention:**
```python
# Use more stable solver:
from scipy.linalg import lstsq

weights, residuals, rank, s = lstsq(X, y, cond=1e-10)

# Or explicitly check condition number:
cond = np.linalg.cond(XtX)
if cond > 1e10:
    print(f"Warning: ill-conditioned matrix, cond={cond:.2e}")
    # Fall back to stronger regularization
    lambda_reg *= 10
```

---

## Minor Pitfalls

Mistakes that cause annoyance but are fixable.

### Pitfall 11: Inconsistent Preprocessing Between Training and Inference

**What goes wrong:** Models trained with different preprocessing than used during ensemble prediction.

**Examples:**
- Training: z-score per sample, Inference: z-score per gene
- Training: top 5000 HVGs, Inference: all 10,716 genes

**Prevention:** Document preprocessing exactly and verify consistency.

---

### Pitfall 12: Not Saving OOF Predictions

**What goes wrong:** Regenerating OOF predictions on every run, wasting compute time.

**Why it happens:** "Just retrain everything" mentality.

**Your context:** 10 models * 9 cell lines * 3 folds = 270 training runs per iteration.

**Prevention:**
```python
# Save OOF predictions with full metadata
np.savez(f"oof_{model}_{cell_line}.npz",
    oof_predictions=oof_preds,
    fold_assignments=fold_ids,
    sample_ids=sample_ids,
    model_version=model.__version__,
    training_params=params,
    timestamp=datetime.now().isoformat()
)
```

---

### Pitfall 13: Wrong Fold Assignments in OOF

**What goes wrong:** Sample gets assigned to wrong fold, contaminating OOF predictions.

**Your codebase:**
```python
# From data_loader.py - fold assignment is deterministic but fragile:
rng = np.random.RandomState(42)
shuffled = rng.permutation(cell_indices)
# If cell_indices changes (new data added), all folds shift!
```

**Prevention:**
```python
# Use hash-based fold assignment (stable to data additions):
def get_fold(sample_id, n_folds=3):
    return hash(sample_id) % n_folds

# Or store explicit fold assignments in metadata
```

---

### Pitfall 14: Memory Explosion with All Predictions in RAM

**What goes wrong:** Loading all 10 models * all samples * 10,716 genes crashes system.

**Your context:**
- 10 models * 182K samples * 10,716 genes * 4 bytes = ~75 GB
- May exceed available RAM

**Prevention:**
```python
# Process in chunks or use memory-mapped arrays:
predictions = np.memmap('predictions.npy', dtype='float32', mode='w+',
                        shape=(n_models, n_samples, n_genes))

# Or stream through one model at a time, computing contributions incrementally
```

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| OOF Generation (3 folds) | Wrong fold splits | Verify sample IDs per fold before training |
| Precomputed Predictions | Sample misalignment | Store and verify sample_ids in .npz files |
| Per-Gene Ridge Regression | Overfitting 107K params | Compare to global-weight baseline |
| Stacking Meta-Learner | Leakage through val split | Strict train/val/test separation |
| Beating Biolord | Stacking just learns Biolord~1.0 | Check weight distribution, try model selection |
| Embedding Stacking | Different embedding spaces | Normalize embeddings before concatenation |
| Final Evaluation | Evaluating on weight-learning data | Reserve truly held-out test fold |

---

## When Ensembles Fail to Improve

**Red flags suggesting ensemble won't beat best model:**

1. **Dominant Model:** Biolord at 0.7957 vs #2 at 0.7694 (3.4% gap) is large. Ensemble may just learn to weight Biolord highly.

2. **High Model Correlation:** If TranSiGen variants correlate >0.95, they contribute no diversity.

3. **Error Patterns Match:** If all models fail on the same samples (e.g., rare cell types, unusual drugs), ensemble can't help.

4. **Overfitting Risk:** With 10 models and limited samples, meta-learner may fit noise.

**Diagnostic checklist:**
- [ ] Compute model correlation matrix
- [ ] Identify samples where models disagree (diversity)
- [ ] Check if errors are correlated across models
- [ ] Verify that simple averaging provides any lift
- [ ] Test on completely held-out cell line (not just fold)

---

## Sources

**Domain knowledge sources:**
- Kaggle ensemble best practices (empirical community knowledge)
- sklearn documentation on stacking: https://scikit-learn.org/stable/modules/ensemble.html#stacking
- MLxtend stacking: http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/
- Reviewed existing codebase: `ensemble.py`, `data_loader.py`

**Confidence levels:**
- Data leakage pitfalls: HIGH (well-documented in ML literature)
- High-dimensional overfitting: HIGH (fundamental statistical issue)
- Model diversity: MEDIUM (empirical, domain-specific)
- Memory/numerical issues: HIGH (practical engineering constraints)
