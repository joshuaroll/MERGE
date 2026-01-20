# Feature Landscape: Ensemble Methods for Gene Expression Prediction

**Domain:** Machine Learning Ensembles for High-Dimensional Regression
**Researched:** 2026-01-19
**Overall Confidence:** HIGH (well-established ML techniques applied to specific domain)

## Table Stakes

Features users expect from any ensemble implementation. Missing = implementation feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Mean averaging | Simplest baseline, universally expected | Low | Already implemented in codebase |
| Weighted averaging | Basic improvement over mean | Low | Weights can be uniform, learned, or optimized |
| Out-of-fold predictions | Required to prevent data leakage | Medium | 3-fold OOF generation per model |
| Per-sample evaluation | Standard for high-variance predictions | Low | Report mean + std across samples |
| Model contribution analysis | Understanding ensemble behavior | Low | Which models contribute most? |
| Fallback to best model | Ensemble should never hurt | Medium | Guarantee: ensemble >= best individual |

## Differentiators

Features that would set this ensemble apart. Not expected, but highly valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Per-gene weighting | Different models excel at different genes | Medium | Ridge regression per-gene, already implemented |
| Scale calibration | Fix magnitude shrinkage in predictions | Medium | CalibratedEnsemble already does this |
| Embedding-level stacking | Richer feature combination than predictions | High | EmbeddingStackingEnsemble implemented |
| Adaptive selection | Choose best model per-sample dynamically | High | Oracle selection shows ceiling |
| Negative correlation exploitation | Ensemble error reduction via diversity | Medium | Particularly useful when models disagree |
| Cross-validation meta-learning | Robust to overfitting on small val sets | Medium | 5-fold CV for weight learning |

## Anti-Features

Features to explicitly NOT build. Common mistakes in this domain.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Train meta-learner on same data as base models | Severe overfitting, inflated metrics | Use OOF predictions exclusively |
| Simple voting for regression | Only works for classification | Use weighted averaging instead |
| Over-complicated architectures | Diminishing returns, interpretability loss | Start simple, add complexity if needed |
| Ignoring model correlations | Redundant models dilute ensemble | Consider model diversity in selection |
| Fixed uniform weights across all conditions | Suboptimal for heterogeneous data | Learn condition-specific or gene-specific weights |
| Ignoring prediction scale differences | Models may have different magnitude biases | Calibrate predictions before combining |

---

## Detailed Feature Analysis

### 1. Simple Ensemble Methods

#### 1.1 Mean Averaging

**What it does:** Averages predictions from all models: `pred_ensemble = (1/N) * sum(pred_i)`

**Expected performance:**
- Typically improves over median model by 2-5%
- Reduces variance but may not beat best model
- Baseline to establish: if mean averaging doesn't help, ensemble unlikely to help

**Implementation (from codebase):**
```python
weighted_pred = np.zeros_like(predictions[0])
for pred, weight in zip(predictions, self.weights):
    weighted_pred += weight * pred  # weights = 1/N for mean
```

**When to use:** Always compute as baseline. If mean averaging beats all individual models, ensemble approach is justified.

**Confidence:** HIGH - well-established technique

---

#### 1.2 Top Score Selection (Oracle)

**What it does:** For each sample, selects the model with highest validation R^2 and uses only that model's prediction.

**Expected performance:**
- Represents theoretical ceiling for model selection approaches
- Oracle selection (with ground truth) typically achieves 5-15% over best model
- Realistic selection (estimated) achieves 50-70% of oracle improvement

**Implementation approach:**
```python
# For each test sample, find which model performs best
# This requires validation data to estimate per-sample model quality
for sample_idx in range(n_samples):
    model_scores = [evaluate_sample(model_pred[sample_idx]) for model_pred in predictions]
    best_model_idx = np.argmax(model_scores)
    ensemble_pred[sample_idx] = predictions[best_model_idx][sample_idx]
```

**When to use:**
- Compute oracle to understand ceiling
- Use validation-based selection when models have complementary strengths

**Confidence:** HIGH for oracle, MEDIUM for realistic implementation

---

#### 1.3 Weighted Averaging with Learned Weights

**What it does:** Learns optimal weights to combine model predictions.

**Approaches:**
1. **Global weights:** Single weight per model, optimized on validation set
2. **Per-gene weights:** Different weights for each of 10,716 genes (already implemented as `LearnedEnsemble`)
3. **Per-sample weights:** Dynamic weights based on input characteristics

**Expected performance:**
- Global weights: 1-3% improvement over mean averaging
- Per-gene weights: 3-8% improvement (significant for this domain)
- Per-sample weights: Highest potential but risk of overfitting

**Implementation (from codebase - per-gene inverse MSE weighting):**
```python
for g in range(n_genes):
    mse_scores = []
    for pred in val_preds:
        mse = np.mean((pred[:, g] - val_treated[:, g]) ** 2)
        mse_scores.append(mse)
    inv_mse = 1.0 / (mse_scores + 1e-10)
    weights = inv_mse / np.sum(inv_mse)
    self._gene_weights[:, g] = weights
```

**Confidence:** HIGH - per-gene weighting is well-suited for high-dimensional outputs

---

### 2. Stacked Meta-Learning

#### 2.1 OOF Prediction Generation

**What it does:** Generates predictions for training data using models trained on non-overlapping folds.

**Process (for 3 folds):**
```
Fold 1: Train on Folds 2,3 -> Predict on Fold 1
Fold 2: Train on Folds 1,3 -> Predict on Fold 2
Fold 3: Train on Folds 1,2 -> Predict on Fold 3
Concatenate -> OOF predictions for all training samples
```

**Why critical:**
- Prevents data leakage that would inflate meta-learner performance
- OOF predictions represent realistic "unseen" predictions
- Without OOF, meta-learner sees optimistic base model performance

**Expected cost:**
- 10 models x 9 cell lines x 3 folds = 270 model training runs
- Each produces predictions for ~1/3 of samples
- Storage: ~10 MB per model per cell line per fold

**Implementation considerations:**
```python
# Structure for OOF predictions
oof_predictions = {
    'cell_line': {
        'fold_0': {
            'model_name': np.ndarray,  # shape: (n_fold_samples, n_genes)
        }
    }
}
```

**Confidence:** HIGH - standard practice in kaggle/ML competitions

---

#### 2.2 Level-1 Ridge Regression

**What it does:** Trains ridge regression on stacked OOF predictions to predict true values.

**Architecture:**
```
Input: [pred_model1, pred_model2, ..., pred_modelN]  # shape: (n_samples, n_models * n_genes) or per-gene
Output: pred_treated  # shape: (n_samples, n_genes)

Option A (per-gene, already implemented):
  For each gene g: y_g = X_g @ weights_g + bias_g
  where X_g = [pred_1_g, pred_2_g, ..., pred_N_g]

Option B (global features):
  Concatenate all model predictions -> learn mapping to all genes
```

**Expected performance:**
- Per-gene ridge: 3-8% improvement over weighted average
- Regularization (lambda=0.01) prevents overfitting with N_models < N_samples

**Implementation (from codebase):**
```python
for g in range(n_genes):
    X = val_preds[:, :, g].T  # (n_val_samples, n_models)
    y = val_treated[:, g]

    # Ridge regression
    lambda_reg = 0.01
    XtX = X.T @ X + lambda_reg * np.eye(n_models)
    Xty = X.T @ y
    weights = np.linalg.solve(XtX, Xty)
```

**Hyperparameter guidance:**
- lambda_reg: Start at 0.01, tune via cross-validation
- Higher lambda if n_models is large relative to n_samples

**Confidence:** HIGH - standard, well-understood approach

---

#### 2.3 Level-1 MLP Meta-Learner

**What it does:** Trains neural network on concatenated OOF predictions.

**Architecture options:**

**Option A: Per-gene MLP (simple)**
```
Input: [pred_1_g, pred_2_g, ..., pred_N_g] for gene g  # (n_models,)
Hidden: 2 layers, 64-32 units, ReLU
Output: pred_treated_g  # scalar

Train: 10,716 independent MLPs (or one MLP applied to each gene)
```

**Option B: Global MLP (complex)**
```
Input: concat(all model predictions)  # (n_models * n_genes,) = huge!
Hidden: compression layers
Output: all genes  # (n_genes,)

Issue: Input dimension 10 * 10,716 = 107,160 is very large
```

**Option C: Hybrid (recommended)**
```
Input: [pred_1_g, pred_2_g, ..., pred_N_g, global_context]
global_context = learned embedding of sample characteristics
Hidden: 2 layers
Output: pred_treated_g
```

**Expected performance:**
- Per-gene MLP: Comparable to ridge, potentially 1-2% better with nonlinear patterns
- Risk of overfitting if not enough training samples

**Implementation considerations:**
```python
class PerGeneMLP(nn.Module):
    def __init__(self, n_models, hidden_dim=64):
        self.fc1 = nn.Linear(n_models, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        # x: (batch, n_models) for one gene
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

**Confidence:** MEDIUM - depends heavily on implementation details and tuning

---

#### 2.4 Level-1 Lightweight Networks

**What it does:** More sophisticated architectures for meta-learning.

**Option A: Attention-based Weighting**
```python
class AttentionEnsemble(nn.Module):
    def __init__(self, n_models, n_genes, hidden_dim=64):
        self.query = nn.Linear(n_genes, hidden_dim)
        self.key = nn.Linear(n_genes, hidden_dim)

    def forward(self, predictions):
        # predictions: (batch, n_models, n_genes)
        # Compute attention weights per sample
        q = self.query(predictions)  # (batch, n_models, hidden)
        k = self.key(predictions)
        attention = F.softmax(q @ k.transpose(-1, -2) / sqrt(hidden), dim=-1)
        # Weighted combination
        return (attention @ predictions).mean(dim=1)
```

**Option B: Model Embedding + MLP**
```python
# Learn embedding for each model, combine with predictions
model_embeddings = nn.Embedding(n_models, embed_dim)
combined = concat(predictions, model_embeddings)
output = mlp(combined)
```

**Expected performance:**
- Attention: Can learn sample-dependent weighting, 2-5% potential improvement
- Risk of overfitting with limited training data

**Confidence:** MEDIUM - novel application, requires experimentation

---

### 3. Out-of-Fold (OOF) Prediction Generation

#### 3.1 Standard 3-Fold OOF

**Process:**
1. Split training data into 3 folds
2. For each fold k:
   - Train model on folds != k
   - Predict on fold k
   - Save predictions
3. Concatenate to get OOF predictions for all training samples

**Implementation:**
```python
def generate_oof_predictions(model_class, X_train, y_train, n_folds=3):
    oof_preds = np.zeros_like(y_train)
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        model = model_class()
        model.train(X_train[train_idx], y_train[train_idx])
        oof_preds[val_idx] = model.predict(X_train[val_idx])

    return oof_preds
```

**Storage requirements:**
- Per model: ~80 MB (10,716 genes * ~2000 samples * 4 bytes * 9 cell lines)
- Total: ~800 MB for 10 models

**Confidence:** HIGH - standard, well-understood

---

#### 3.2 Stratified OOF

**Enhancement:** Ensure each fold has similar distribution of:
- Cell line (if multi-cell training)
- Drug clusters (similar compounds together)
- Expression variance (high/low variability samples)

**When useful:** When data is heterogeneous and random splits might create imbalanced folds.

**Confidence:** HIGH - recommended enhancement over basic k-fold

---

### 4. Embedding-Based Stacking

#### 4.1 Concept

**What it does:** Instead of stacking final predictions, stacks intermediate embeddings from base models before their output layer.

**Rationale:**
- Embeddings contain richer information than scalar predictions
- Meta-learner can learn more complex relationships
- May capture complementary features across models

**Architecture:**
```
Model 1: Input -> Encoder1 -> Embedding1 (128-d) -> Decoder1 -> Pred1
Model 2: Input -> Encoder2 -> Embedding2 (256-d) -> Decoder2 -> Pred2
...

Stacking approach:
Concat([Embedding1, Embedding2, ...]) -> Meta-MLP -> Final Prediction
```

**Implementation (from codebase):**
```python
class EmbeddingStackingEnsemble:
    def _concatenate_embeddings(self) -> np.ndarray:
        emb_list = [self._embeddings[m]['embeddings'] for m in self._model_names]
        return np.hstack(emb_list)  # (n_samples, total_emb_dim)

    def _build_meta_model(self):
        layers = []
        in_dim = self._total_emb_dim  # e.g., 128 + 256 + 100 + ...
        for hidden_dim in self.hidden_dims:  # [256, 128]
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self._n_genes))  # -> 10,716
        return nn.Sequential(*layers)
```

---

#### 4.2 Embedding Extraction Requirements

Each base model must expose intermediate embeddings:

| Model | Embedding Layer | Dimension | Notes |
|-------|-----------------|-----------|-------|
| Biolord | Disentangled latent | 128 | Split into attribute-specific |
| MultiDCP | Fused representation | 306 | After drug+cell+dose fusion |
| scGen | VAE latent | 100 | Standard VAE bottleneck |
| ChemCPA | Compositional latent | 256 | After perturbation encoding |
| PDGrapher | GNN node embeddings | ~256 | Requires aggregation |
| TranSiGen | Dual VAE latent | 200 | Concatenated encoder outputs |

**Challenge:** Not all models expose embeddings easily. May require model modification.

---

#### 4.3 Expected Performance

**Theoretical advantage:**
- Embeddings carry ~10-100x more information than predictions
- Meta-learner can discover nonlinear combinations
- Particularly useful when models capture different aspects

**Practical considerations:**
- Requires more training data for meta-learner
- Higher risk of overfitting
- Embeddings may not be comparable across models without normalization

**Expected improvement:** 0-5% over prediction-based stacking
- Best case: Models have complementary learned representations
- Worst case: Embeddings are redundant or noisy

**Confidence:** MEDIUM - promising but requires experimentation

---

### 5. Calibration and Guarantees

#### 5.1 Scale Calibration

**Problem:** Different models may have different prediction magnitude biases.

**Solution (from CalibratedEnsemble):**
```python
# Learn scale factor to match true magnitude
for m in range(n_models):
    # Collect (true_de, pred_de) pairs for top-k genes
    all_true, all_pred = [], []
    for i in range(len(train_idx)):
        de_mag = np.abs(true_de[i])
        topk_idx = np.argsort(de_mag)[-100:]  # Top 100 genes
        all_true.extend(true_de[i, topk_idx])
        all_pred.extend(pred_de[i, topk_idx])

    # Optimal scale: minimize ||true - scale*pred||^2
    scale = np.sum(all_pred * all_true) / np.sum(all_pred**2)
    scale = np.clip(scale, 0.5, 3.0)  # Reasonable bounds
    self._scale_factors[m] = scale
```

**Expected impact:** 2-5% improvement when models under-predict extreme values

**Confidence:** HIGH - well-understood calibration technique

---

#### 5.2 Performance Guarantee

**Requirement:** Ensemble should never perform worse than best individual model.

**Implementation (from CalibratedEnsemble):**
```python
# After training, compare ensemble to best individual
if ensemble_score >= best_individual_score:
    self._ensemble_beats_best = True
    # Use ensemble predictions
else:
    self._ensemble_beats_best = False
    # Fall back to best individual model
```

**Importance:** Critical for production use - stakeholders expect ensemble to help, not hurt.

**Confidence:** HIGH - straightforward safeguard

---

## Feature Dependencies

```
                    OOF Predictions
                          |
        +-----------------+-----------------+
        |                 |                 |
   Mean Average    Weighted Average    Ridge Stacking
        |                 |                 |
        +-----------------+-----------------+
                          |
                 Calibration Layer
                          |
              Performance Guarantee
                          |
                   Final Ensemble

Embedding extraction (parallel path):
   Base Models -> Extract Embeddings -> Embedding Stacking -> Merge with prediction-based
```

---

## MVP Recommendation

For MVP (beat Biolord's 0.7957), prioritize:

1. **OOF prediction generation** (Required)
   - Generate for all 10 models x 9 cell lines x 3 folds
   - Critical foundation for all stacking approaches

2. **Per-gene ridge regression** (Already implemented)
   - Proven effective in codebase
   - Low complexity, interpretable

3. **Scale calibration** (Already implemented)
   - Fix magnitude shrinkage
   - 2-5% expected improvement

4. **Performance guarantee** (Already implemented)
   - Never worse than best individual
   - Builds trust

**Expected outcome:** 5-10% improvement over Biolord, reaching ~0.83-0.87 R^2 Top-20

### Defer to Post-MVP

- **Embedding-based stacking:** Requires model modifications, higher complexity
- **Per-sample adaptive selection:** Risk of overfitting, diminishing returns
- **Attention-based meta-learners:** Novel approach, needs experimentation

---

## Confidence Assessment

| Feature | Confidence | Reasoning |
|---------|------------|-----------|
| Mean averaging | HIGH | Universally applicable baseline |
| Weighted averaging | HIGH | Simple, well-understood |
| OOF predictions | HIGH | Standard ML practice |
| Per-gene ridge | HIGH | Already implemented, working |
| Scale calibration | HIGH | Already implemented, working |
| MLP meta-learner | MEDIUM | May need tuning, overfitting risk |
| Embedding stacking | MEDIUM | Promising but untested in this domain |
| Attention meta-learner | LOW | Novel, needs experimentation |

---

## Sources

- **Codebase:** `/raid/home/joshua/projects/PDGrapher_Baseline_Models/ensemble.py` - existing implementations
- **Project context:** `.planning/PROJECT.md` - baseline model performance (Biolord: 0.7957)
- **Data loader:** `data_loader.py` - TopKEvaluator methodology
- **ML ensembles:** Standard Kaggle/competition stacking techniques
- **Gene expression ML:** Domain-specific considerations for high-dimensional outputs

---

*Note: Many features are already implemented in the codebase. The main gap is generating OOF predictions for all 10 models to enable proper stacking without data leakage.*
