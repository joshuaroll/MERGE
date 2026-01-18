# Architecture

**Analysis Date:** 2025-01-18

## Pattern Overview

**Overall:** Model-Wrapper Ensemble Architecture

The codebase implements a multi-model ensemble framework for predicting gene expression responses to chemical perturbations. Each prediction model is wrapped with a common interface, enabling uniform training, evaluation, and ensemble combination.

**Key Characteristics:**
- Abstract base class pattern for model wrappers (`BasePerturbationModel`)
- Unified data loading through `PDGrapherDataLoader` that converts to model-specific formats
- Training scripts per model family with consistent CLI interfaces
- Ensemble combiners that aggregate predictions from multiple models
- Differential expression (DE) evaluation as the primary metric focus

## Layers

**Data Layer:**
- Purpose: Load, preprocess, and serve gene expression data
- Location: `data_loader.py`, `utils/datareader_pdg_12122025.py`, `utils/data_utils_pdg.py`
- Contains: Data loading classes, format converters, train/test splitters
- Depends on: External pickle files from MultiDCP_pdg project
- Used by: All training scripts and model wrappers

**Model Layer:**
- Purpose: Define and wrap prediction models
- Location: `models/`
- Contains: Base classes, model implementations, wrappers for external models
- Depends on: Data layer, PyTorch, external model libraries (scGen, ChemCPA, etc.)
- Used by: Training scripts, ensemble layer

**Training Layer:**
- Purpose: Train individual models with consistent evaluation
- Location: `train_*.py` files in project root
- Contains: Training loops, metric computation, WandB logging, result saving
- Depends on: Model layer, data layer
- Used by: Shell scripts, manual execution

**Ensemble Layer:**
- Purpose: Combine predictions from multiple models
- Location: `ensemble.py`
- Contains: Ensemble strategies (weighted, learned, stacking)
- Depends on: Model layer
- Used by: Evaluation scripts

**Evaluation Layer:**
- Purpose: Compute and aggregate metrics across models/cells/folds
- Location: `run_evaluation.py`, `run_full_evaluation.py`
- Contains: Evaluation orchestration, result aggregation
- Depends on: All other layers
- Used by: End users

## Data Flow

**Training Flow:**

1. Load pickle data via `PDGrapherDataLoader` from MultiDCP_pdg project
2. Apply cell line filter and PDGrapher train/val/test splits
3. Convert expression matrices to model-specific format (AnnData for scGen, tensors for PyTorch models)
4. Train model with periodic evaluation on test set
5. Save best model checkpoint and predictions to `trained_models/` or `output/`
6. Log metrics to WandB and shared CSV results file

**Prediction Flow:**

1. Load trained model from checkpoint
2. Pass diseased expression + metadata (drug SMILES, dose) through model
3. Model outputs predicted treated expression
4. Compute differential expression: `pred_DE = pred_treated - diseased`
5. Evaluate against true DE using top-k R² and Pearson metrics

**State Management:**
- Models are stateful (trained weights stored in PyTorch checkpoints)
- Predictions cached as `.npz` files in model output directories
- Results aggregated in `data/topk_r2_results.csv`
- WandB provides experiment tracking and hyperparameter logging

## Key Abstractions

**BasePerturbationModel:**
- Purpose: Define common interface for all prediction models
- Examples: `models/base_model.py`
- Pattern: Abstract base class with `train()`, `predict()`, `get_embeddings()` methods
- All wrappers inherit from this class

**PDGrapherDataLoader:**
- Purpose: Unified data access regardless of model type
- Examples: `data_loader.py`
- Pattern: Lazy-loading with format conversion methods (`to_anndata()`, `to_perturbation_pairs()`)
- Handles train/test splits compatible with PDGrapher paper

**TopKEvaluator:**
- Purpose: Compute differential expression metrics
- Examples: `data_loader.py`
- Pattern: Stateless evaluator with `compute_metrics()` method
- Returns dict of R² top-20/40/80 and Pearson scores

**Model Wrappers:**
- Purpose: Adapt external models to common interface
- Examples: `models/scgen_wrapper.py`, `models/chemcpa_wrapper.py`, `models/transigen_wrapper.py`
- Pattern: Wrapper inherits from `BasePerturbationModel`, delegates to underlying model

**MoELayer:**
- Purpose: Provide Mixture-of-Experts routing for TranSiGen and MultiDCP variants
- Examples: `models/moe_modules.py`
- Pattern: Gating network selects top-k experts, weighted combination of outputs

## Entry Points

**Training Entry Points:**

| Script | Model(s) | Description |
|--------|----------|-------------|
| `train_transigen_pdg.py` | TranSiGen, TranSiGen_MoE | VAE with molecular embeddings |
| `train_chemcpa_pdg.py` | ChemCPA | Compositional perturbation autoencoder |
| `train_scgen_pdg.py` | scGen | VAE with latent arithmetic |
| `train_multidcp_chemoe_pdg.py` | MultiDCP_CheMoE | MoE extension of MultiDCP |
| `train_pdgrapher_pdg.py` | PDGrapher | GNN on PPI network |

**Evaluation Entry Points:**
- `run_evaluation.py`: Quick evaluation on single cell line
- `run_full_evaluation.py`: Full evaluation across all cell lines/folds

**Batch Execution:**
- `run_training.sh`: Launch multiple training jobs with nohup
- `run_transigen_batch.sh`: Batch train TranSiGen across cell lines
- `run_all_transigen.sh`: Run all TranSiGen variants

## Error Handling

**Strategy:** Fail-fast with warnings for non-critical issues

**Patterns:**
- Training scripts catch and log exceptions, continue with next cell line/fold
- Model wrappers return zero/placeholder predictions for missing data
- Import errors handled gracefully with availability flags in `ensemble.py`
- File lock used for concurrent CSV writes (`filelock.FileLock`)

```python
# Example from ensemble.py - graceful import handling
try:
    from models.multidcp_wrapper import MultiDCPModel
    MULTIDCP_AVAILABLE = True
except ImportError:
    MULTIDCP_AVAILABLE = False
```

## Cross-Cutting Concerns

**Logging:**
- WandB for experiment tracking (project per model family)
- Console logging with epoch progress
- Results saved to shared CSV for cross-model comparison

**Validation:**
- Input shapes validated in model forward passes
- Gene dimension mismatch handled by padding/truncation
- NaN/Inf values masked in metric computation

**Authentication:**
- WandB login required for experiment tracking
- No other authentication needed (local file system access)

**Configuration:**
- CLI arguments with argparse
- Default paths hardcoded at top of each script
- Shared paths reference MultiDCP_pdg project data

---

*Architecture analysis: 2025-01-18*
