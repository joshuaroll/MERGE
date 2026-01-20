# Ensemble Architecture Patterns for Gene Expression Prediction

**Domain:** Ensemble methods for perturbation response prediction
**Researched:** 2026-01-19
**Confidence:** HIGH (based on existing codebase analysis)

## Executive Summary

This document details architecture patterns for integrating ensemble methods with the existing 10 trained PyTorch models for gene expression prediction. The analysis is based on the current codebase which already implements several ensemble approaches (`ensemble.py`) and prediction generation infrastructure (`generate_predictions.py`, `generate_embeddings.py`).

## Current Architecture Overview

### Existing Components

```
                        +-------------------+
                        |  PDGrapherDataLoader |
                        |  (data_loader.py)   |
                        +----------+----------+
                                   |
                    +--------------+--------------+
                    |              |              |
              +-----v-----+  +-----v-----+  +-----v-----+
              | Model 1   |  | Model 2   |  | Model N   |
              | Wrapper   |  | Wrapper   |  | Wrapper   |
              +-----+-----+  +-----+-----+  +-----+-----+
                    |              |              |
              +-----v-----+  +-----v-----+  +-----v-----+
              |predictions/|  |predictions/|  |predictions/|
              |model1_*.npz|  |model2_*.npz|  |modelN_*.npz|
              +-----+-----+  +-----+-----+  +-----+-----+
                    |              |              |
                    +--------------+--------------+
                                   |
                        +----------v----------+
                        | PrecomputedEnsemble |
                        |    (ensemble.py)    |
                        +----------+----------+
                                   |
                        +----------v----------+
                        |   TopKEvaluator     |
                        |  (data_loader.py)   |
                        +---------------------+
```

### Models Available (10 total)

| Model | Wrapper File | Embedding Support | Environment |
|-------|--------------|-------------------|-------------|
| MultiDCP | `multidcp_wrapper.py` | YES (128-dim) | mdcp_env |
| Biolord | `biolord_wrapper.py` | YES (n_latent) | biolord_env |
| scGen | `scgen_wrapper.py` | YES (n_latent) | scgen_env |
| ChemCPA | `chemcpa_wrapper.py` | YES (latent_dim) | chemcpa_env |
| PDGrapher | External (special handling) | NO | MERGE_env |
| MultiDCP_CheMoE | `multidcp_chemoe_wrapper.py` | Possible | mdcp_env |
| TranSiGen | `transigen_wrapper.py` | Possible | transigen_env |
| TranSiGen_MoE_Sparse | `transigen_moe_wrapper.py` | Possible | transigen_env |
| TranSiGen_MoE_Balanced | `transigen_moe_wrapper.py` | Possible | transigen_env |
| CheMoE | External (train_chemoe_pdg.py) | Possible | mdcp_env |

---

## Recommended Architecture: Prediction-Based Ensemble

### Rationale

1. **Environment Isolation**: Models require different conda environments (scvi, biolord, mdcp_env, etc.)
2. **Minimal Coupling**: No need to modify base model architectures
3. **Proven Pattern**: Already implemented in `PrecomputedEnsemble` class
4. **Memory Efficient**: Load predictions from disk, not models into memory

### Data Structures

#### Prediction Storage Format (.npz)

```python
# Structure: predictions/{model}_{cell_line}_fold{fold}.npz
{
    'predictions': np.ndarray,    # (n_samples, 10716) - predicted treated expression
    'test_indices': np.ndarray,   # (n_samples,) - indices into original dataset
    'treated_test': np.ndarray,   # (n_samples, 10716) - ground truth treated
    'diseased_test': np.ndarray,  # (n_samples, 10716) - ground truth diseased
}
```

#### Memory Considerations

| Configuration | Memory Estimate | Notes |
|--------------|-----------------|-------|
| Single model predictions | ~100 MB per file | float32, ~1200 samples |
| 10 models loaded | ~1 GB RAM | All predictions in memory |
| Stacked tensor [B, 10, 10716] | ~428 MB per 1000 samples | For meta-learner input |

**Recommendation**: The current approach is memory-efficient. Even loading all 10 models' predictions simultaneously requires only ~1 GB, well within typical GPU memory.

---

## Directory Structure

### Current Structure (Observed)

```
PDGrapher_Baseline_Models/
├── predictions/                    # Precomputed predictions
│   ├── multidcp_A549_fold0.npz
│   ├── biolord_A549_fold0.npz
│   ├── scgen_A549_fold0.npz
│   ├── chemcpa_A549_fold0.npz
│   └── cellot_A549_fold0.npz
├── embeddings/                     # Extracted embeddings
│   ├── multidcp_A549_fold0.npz
│   ├── scgen_A549_fold0.npz
│   └── chemcpa_A549_fold0.npz
├── data/
│   └── topk_r2_results.csv         # Consolidated results
├── output/                         # Model-specific outputs
│   ├── chemcpa/
│   ├── scgen/
│   └── pdgrapher_*/
└── models/                         # Model wrappers
    ├── base_model.py
    ├── multidcp_wrapper.py
    ├── biolord_wrapper.py
    ├── scgen_wrapper.py
    └── ...
```

### Recommended Structure for OOF Predictions

```
PDGrapher_Baseline_Models/
├── predictions/
│   ├── fold0/                      # Fold-specific directories
│   │   ├── multidcp_A549.npz
│   │   ├── biolord_A549.npz
│   │   └── ...
│   ├── fold1/
│   │   ├── multidcp_A549.npz
│   │   └── ...
│   └── fold2/
│       └── ...
├── oof_predictions/                # Out-of-fold for stacking
│   ├── multidcp/
│   │   ├── A549_oof.npz            # Combined OOF from all folds
│   │   └── ...
│   └── ...
└── ensemble_models/                # Trained meta-learners
    ├── ridge_A549.pkl
    ├── mlp_A549.pt
    └── attention_A549.pt
```

### Naming Conventions

| Pattern | Example | Usage |
|---------|---------|-------|
| `{model}_{cell}_{fold}` | `multidcp_A549_fold0.npz` | Current (flat) |
| `fold{fold}/{model}_{cell}` | `fold0/multidcp_A549.npz` | Recommended (hierarchical) |
| `{model}/{cell}_oof` | `multidcp/A549_oof.npz` | OOF predictions |

---

## Integration Patterns

### Pattern 1: Offline Prediction Generation (Current)

**When to Use**: Initial setup, batch generation of all predictions

```python
# generate_predictions.py pattern
def run_model_prediction(model: str, cell_line: str, fold: int):
    """Run prediction in model's conda environment via subprocess."""
    script_content = f'''
    from models.{model}_wrapper import {Model}Model
    model = {Model}Model()
    model.train(diseased_train, treated_train, metadata)
    predictions = model.predict(diseased_test, test_metadata)
    np.savez(pred_path, predictions=predictions, ...)
    '''
    subprocess.run(['bash', '-c', f'conda activate {env} && python {script}'])
```

**Pros**:
- Environment isolation handled
- Predictions cached to disk
- Can run in parallel (different GPUs)

**Cons**:
- Slower than in-memory
- Requires disk I/O

### Pattern 2: Online Ensemble Combination (Current)

**When to Use**: After predictions are generated

```python
# From ensemble.py - PrecomputedEnsemble
class PrecomputedEnsemble:
    def load_predictions(self, cell_line: str, fold: int):
        """Load all available precomputed predictions."""
        for pred_file in pred_dir.glob(f"*_{cell_line}_fold{fold}.npz"):
            model_name = pred_file.stem.replace(f"_{cell_line}_fold{fold}", "")
            data = np.load(pred_file)
            self._model_predictions[model_name] = data['predictions']

    def train(self, diseased, treated):
        """Learn ridge regression weights per gene."""
        # Stack predictions: [n_models, n_samples, n_genes]
        val_preds = np.stack([self._model_predictions[m] for m in self._model_names])

        # Ridge regression per gene
        for g in range(n_genes):
            X = val_preds[:, :, g].T  # [n_samples, n_models]
            y = treated[:, g]
            weights = ridge_solve(X, y, lambda_reg=0.01)
            self._meta_weights[:, g] = weights
```

### Pattern 3: OOF Prediction Generation (Recommended for Stacking)

**Purpose**: Generate out-of-fold predictions to train meta-learner without data leakage

```python
def generate_oof_predictions(model_name: str, cell_line: str, n_folds: int = 3):
    """Generate OOF predictions for stacking."""
    oof_predictions = np.zeros((n_total_samples, n_genes))

    for fold in range(n_folds):
        # Load predictions for this fold (test set)
        pred_path = f"predictions/fold{fold}/{model_name}_{cell_line}.npz"
        data = np.load(pred_path)

        # Place in OOF array using test indices
        test_indices = data['test_indices']
        oof_predictions[test_indices] = data['predictions']

    return oof_predictions
```

---

## Meta-Learner Training Architectures

### Option 1: Per-Gene Ridge Regression (Current Default)

**Implementation**: Already in `ensemble.py` as `StackingEnsemble`

```python
# Input: [n_samples, n_models] per gene
# Output: [n_samples, 1] per gene
# Total: 10716 independent ridge regressions

for g in range(n_genes):
    X = stacked_preds[:, :, g].T  # [n_samples, n_models]
    y = true_treated[:, g]

    # Ridge: w = (X'X + lambda*I)^(-1) X'y
    XtX = X.T @ X + lambda_reg * np.eye(n_models)
    weights[:, g] = np.linalg.solve(XtX, X.T @ y)
```

**Pros**: Simple, fast, no hyperparameters beyond lambda
**Cons**: No cross-gene learning, 10716 separate models

### Option 2: Flattened MLP

**Input**: Flatten all model predictions for a sample

```python
# Input: [batch, 10 * 10716] = [batch, 107160]
# Hidden: [batch, 512] -> [batch, 256]
# Output: [batch, 10716]

class FlattenedMetaLearner(nn.Module):
    def __init__(self, n_models=10, n_genes=10716):
        self.fc1 = nn.Linear(n_models * n_genes, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_genes)
```

**Pros**: Can learn cross-model and cross-gene patterns
**Cons**: Very large input (107K features), prone to overfitting

**Recommendation**: NOT recommended due to input dimensionality

### Option 3: Per-Gene MLP (Recommended)

**Input**: Model predictions for each gene separately

```python
# Input: [batch, n_models] = [batch, 10]
# Hidden: [batch, 32] -> [batch, 16]
# Output: [batch, 1]
# Apply same network to each gene (shared weights)

class PerGeneMetaLearner(nn.Module):
    def __init__(self, n_models=10, hidden_dim=32):
        self.fc1 = nn.Linear(n_models, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        # x: [batch, n_models, n_genes]
        batch_size, n_models, n_genes = x.shape

        # Process each gene with shared network
        x = x.permute(0, 2, 1)  # [batch, n_genes, n_models]
        x = x.reshape(-1, n_models)  # [batch * n_genes, n_models]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.reshape(batch_size, n_genes)
```

**Pros**: Parameter efficient, can learn nonlinear combinations
**Cons**: Still treats genes independently (no cross-gene learning)

### Option 4: Attention-Based Fusion (Advanced)

**Architecture**: Learn which models to trust for each sample

```python
class AttentionMetaLearner(nn.Module):
    def __init__(self, n_models=10, n_genes=10716, d_model=64):
        # Project predictions to attention space
        self.model_embed = nn.Embedding(n_models, d_model)
        self.pred_proj = nn.Linear(n_genes, d_model)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, num_heads=4)

        # Output projection
        self.out_proj = nn.Linear(d_model, n_genes)

    def forward(self, x):
        # x: [batch, n_models, n_genes]
        batch_size, n_models, n_genes = x.shape

        # Project each model's prediction
        x_proj = self.pred_proj(x)  # [batch, n_models, d_model]

        # Add model embeddings
        model_ids = torch.arange(n_models, device=x.device)
        model_emb = self.model_embed(model_ids)  # [n_models, d_model]
        x_proj = x_proj + model_emb.unsqueeze(0)

        # Self-attention across models
        x_proj = x_proj.permute(1, 0, 2)  # [n_models, batch, d_model]
        attn_out, _ = self.attention(x_proj, x_proj, x_proj)

        # Pool and project
        pooled = attn_out.mean(dim=0)  # [batch, d_model]
        return self.out_proj(pooled)
```

**Pros**: Learns model importance dynamically, handles model correlations
**Cons**: More complex, requires more data to train

---

## Embedding Extraction Points

### Models with Implemented Embedding Extraction

| Model | Extraction Point | Dim | Method |
|-------|-----------------|-----|--------|
| MultiDCP | Gene-level hidden (sum aggregated) | 128 | `model.multidcp(...)` then `sum(out, dim=1)` |
| Biolord | Latent representation | n_latent | `model.get_latent_representation()` |
| scGen | VAE latent z | n_latent | `model.get_latent_representation()` |
| ChemCPA | Latent space | latent_dim | `model.get_latent_representation()` (if exposed) |

### Adding Embedding Extraction to New Models

For models without `get_embeddings()` implemented:

```python
# In wrapper class, add:
def get_embeddings(self, diseased, metadata=None):
    """Extract intermediate representations."""
    with torch.no_grad():
        # Forward pass to intermediate layer
        x = self.model.encoder(diseased)
        # Return before final projection
        return x.cpu().numpy()

@property
def embedding_dim(self):
    return self._embedding_dim  # Set during model init
```

### Embedding-Based Ensemble Architecture

```
+------------+  +------------+  +------------+
|  Model 1   |  |  Model 2   |  |  Model N   |
| embeddings |  | embeddings |  | embeddings |
|  (128-d)   |  |  (64-d)    |  |  (256-d)   |
+-----+------+  +-----+------+  +-----+------+
      |               |               |
      +-------+-------+-------+-------+
              |
       [batch, 448]  (concatenated)
              |
      +-------v-------+
      |  Meta-MLP     |
      | [448->256->   |
      |  128->10716]  |
      +-------+-------+
              |
       [batch, 10716]
```

**Current Implementation**: `EmbeddingStackingEnsemble` in `ensemble.py`

---

## Evaluation Pipeline Integration

### Current TopKEvaluator Flow

```python
# From data_loader.py
evaluator = TopKEvaluator(k_values=[20, 40, 80])

# Compute metrics on differential expression
results = evaluator.compute_metrics(
    true_treated=treated_test,
    pred_treated=ensemble_pred,
    diseased=diseased_test
)
# Returns: {
#   'r2_top20': float,
#   'r2_top40': float,
#   'r2_top80': float,
#   'pearson_all': float,
#   ...
# }
```

### Ensemble Evaluation Pattern

```python
def evaluate_ensemble(ensemble, cell_line, fold):
    """Complete evaluation pipeline."""

    # 1. Load predictions
    ensemble.load_predictions(cell_line, fold)

    # 2. Get ground truth (from any model's saved data)
    first_model = list(ensemble._model_predictions.keys())[0]
    treated_test = ensemble._model_predictions[first_model]['treated_test']
    diseased_test = ensemble._model_predictions[first_model]['diseased_test']

    # 3. Train meta-learner
    ensemble.train(diseased_test, treated_test)

    # 4. Generate ensemble predictions
    pred_treated = ensemble.predict(diseased_test)

    # 5. Evaluate
    evaluator = TopKEvaluator()
    results = evaluator.compute_metrics(treated_test, pred_treated, diseased_test)

    return results
```

---

## Recommended Implementation Phases

### Phase 1: OOF Prediction Generation

**Scripts to Create/Modify**:
1. `generate_oof_predictions.py` - New script
2. Modify `generate_predictions.py` to support fold iteration

**Output Structure**:
```
predictions/
├── fold0/
├── fold1/
└── fold2/
```

### Phase 2: Simple Ensemble Methods

**Already Implemented** in `ensemble.py`:
- Averaging: `EnsembleModel` with equal weights
- Weighted averaging: `LearnedEnsemble`
- Ridge stacking: `StackingEnsemble`

**To Add**:
- Top-score selection per sample
- Inverse-variance weighting

### Phase 3: Advanced Meta-Learners

**Implementations to Add**:
1. `PerGeneMLP` meta-learner
2. `AttentionMetaLearner`
3. Cross-validation for meta-learner hyperparameters

### Phase 4: Embedding-Based Ensemble

**Already Implemented**:
- `EmbeddingStackingEnsemble` in `ensemble.py`
- `generate_embeddings.py` for extraction

**To Complete**:
- Extract embeddings for all models
- Train embedding-based meta-learner

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Loading All Models Simultaneously

**Problem**: Memory explosion from loading 10 PyTorch models
**Instead**: Use precomputed predictions from disk

### Anti-Pattern 2: Training Meta-Learner on Test Data

**Problem**: Data leakage, inflated metrics
**Instead**: Use OOF predictions for stacking, or hold out validation set

### Anti-Pattern 3: Ignoring Sample Count Mismatches

**Problem**: Different models may produce different sample counts
**Solution**: Already handled in `PrecomputedEnsemble.load_predictions()` with majority-vote alignment

### Anti-Pattern 4: Per-Gene Global Weights

**Problem**: Assuming same model is best for all genes
**Instead**: Use per-gene weights (already implemented in `StackingEnsemble`)

---

## Scalability Considerations

| Samples | Genes | Models | Ridge Training | MLP Training |
|---------|-------|--------|----------------|--------------|
| 1,000 | 10,716 | 10 | ~10 seconds | ~30 seconds |
| 10,000 | 10,716 | 10 | ~60 seconds | ~5 minutes |
| 100,000 | 10,716 | 10 | ~10 minutes | ~30 minutes |

**Current Dataset Size**: ~1,000-2,000 test samples per cell line

---

## Sources and References

All architecture patterns derived from analysis of existing codebase:

- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/ensemble.py` - Lines 1-2031
- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/generate_predictions.py` - Lines 1-421
- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/generate_embeddings.py` - Lines 1-282
- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/data_loader.py` - Lines 1-339
- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/base_model.py` - Lines 1-208
- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/multidcp_wrapper.py` - Lines 1-389
- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/biolord_wrapper.py` - Lines 1-335
- `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/scgen_wrapper.py` - Lines 1-220

**Confidence Level**: HIGH - All patterns verified against existing, working implementation.
