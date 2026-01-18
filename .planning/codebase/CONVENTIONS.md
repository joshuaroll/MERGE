# Coding Conventions

**Analysis Date:** 2026-01-18

## Naming Patterns

**Files:**
- Snake_case for Python modules: `data_loader.py`, `train_chemcpa_pdg.py`, `multidcp_wrapper.py`
- Suffix `_pdg` indicates PDGrapher integration: `train_scgen_pdg.py`, `datareader_pdg_12122025.py`
- Suffix `_wrapper` for model adapters: `chemcpa_wrapper.py`, `biolord_wrapper.py`, `transigen_wrapper.py`
- Constants in UPPER_SNAKE_CASE at module level: `DATA_PICKLE`, `SPLITS_BASE`, `GEX_SIZE`

**Functions:**
- snake_case throughout: `load_data()`, `compute_de_metrics()`, `get_train_test_split()`
- Prefix `compute_` for calculation functions: `compute_global_metrics()`, `compute_persample_metrics()`
- Prefix `load_` for data loading: `load_and_match_diseased_data()`, `load_precomputed_predictions()`
- Prefix `evaluate_` for evaluation: `evaluate_model()`, `evaluate_predictions()`
- Prefix `train_` for training: `train_model()`, `train_scgen()`

**Variables:**
- snake_case: `train_idx`, `test_indices`, `true_treated`, `pred_de`
- Suffix `_np` for numpy arrays: `predict_np`, `lb_np`, `diseased_np`
- Suffix `_tensor` for PyTorch tensors: `x1_tensor`, `mol_tensor`, `dose_tensor`
- Suffix `_df` for DataFrames: `treated_df`, `diseased_df`
- Suffix `_list` for lists: `pearson_list`, `r2_scores`, `predict_list`
- Suffix `_mask` for boolean arrays: `train_mask`, `valid_mask`

**Classes:**
- PascalCase: `PDGrapherDataLoader`, `TopKEvaluator`, `BasePerturbationModel`
- Prefix `Simple` for simplified implementations: `SimpleChemCPA`, `SimpleChemCPADataset`
- Neural network components follow PyTorch naming: `DrugEncoder`, `GeneDecoder`, `GatingNetwork`

**Types:**
- Type hints used consistently in function signatures:
  ```python
  def train(self, diseased: np.ndarray, treated: np.ndarray,
            metadata: Optional[Dict] = None) -> None:
  ```

## Code Style

**Formatting:**
- No automatic formatter configured (no `.prettierrc`, `pyproject.toml` with black/ruff)
- Manual formatting conventions observed:
  - 4-space indentation
  - Line length approximately 100-120 characters
  - Double quotes for strings
  - Spaces around operators: `x = y + z`

**Linting:**
- No linting configuration detected
- Warnings suppressed at module level: `warnings.filterwarnings('ignore')`

**Type Hints:**
- Consistently used in class methods and public functions
- Import from typing: `from typing import Dict, List, Tuple, Optional`
- NumPy and PyTorch types specified: `np.ndarray`, `torch.Tensor`

## Import Organization

**Order:**
1. Standard library imports (os, sys, argparse, pickle, warnings)
2. Third-party scientific imports (numpy, pandas, torch, scipy)
3. Third-party ML framework imports (wandb, anndata, scanpy, scgen)
4. Local/project imports (from models.base_model, from utils)

**Example from `train_chemcpa_pdg.py`:**
```python
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import wandb
import warnings
```

**Path Manipulation:**
- sys.path.insert(0, ...) used for local imports: `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))`
- Project-relative paths defined as module constants

**Path Aliases:**
- No aliases configured (no pyproject.toml paths section)
- Explicit path construction: `os.path.join(MULTIDCP_SRC, "models")`

## Error Handling

**Patterns:**
- Try/except with pass for non-critical calculations:
  ```python
  try:
      r, _ = pearsonr(true_de[i, top_k_idx], pred_de[i, top_k_idx])
      r2 = r ** 2
      if not np.isnan(r2) and not np.isinf(r2):
          r2_scores.append(r2)
  except:
      pass
  ```

- FileNotFoundError for missing checkpoints/data:
  ```python
  if not os.path.exists(self.checkpoint_path):
      raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
  ```

- ValueError for invalid inputs:
  ```python
  if metadata is None:
      raise ValueError("MultiDCP requires metadata with 'smiles' and 'dose' keys")
  ```

- RuntimeError for state violations:
  ```python
  if not self._trained:
      raise RuntimeError("Model must be trained/loaded first")
  ```

- NaN/Inf checking for numerical stability:
  ```python
  if not np.isnan(r) and not np.isinf(r):
      pearson_scores.append(r)
  ```

## Logging

**Framework:** print statements and WandB (`wandb`)

**Patterns:**
- Print statements for progress and debugging:
  ```python
  print(f"Loading data for {cell_line} fold {fold}...")
  print(f"  Samples: {len(treated_df)}, Genes: {len(gene_cols)}")
  ```

- WandB for experiment tracking:
  ```python
  wandb.init(project="ChemCPA_AE_DE", name=f"chemcpa_{args.cell_line}_fold{args.fold}", config=vars(args))
  wandb.log({'epoch': epoch, 'train_loss': train_loss, 'test_r2_top20': test_metrics['r2_top20']})
  ```

- Conditional logging with USE_WANDB flag:
  ```python
  if USE_WANDB:
      wandb.log({'{0} Dev loss'.format(job): epoch_loss/steps_per_epoch}, step=epoch)
  ```

- Formatted output tables for diagnostics:
  ```python
  print(f"{'Metric':<40} {'Value':>12}")
  print(f"{'-'*55}")
  print(f"{'Global Pearson (DE)':<40} {results['global_pearson']:>12.4f}")
  ```

## Comments

**When to Comment:**
- Module-level docstrings required with usage examples:
  ```python
  """
  Training script for ChemCPA on PDGrapher data.

  Usage:
      python train_chemcpa_pdg.py --cell_line A549 --fold 0 --gpu 0
  """
  ```

- Section separators with `=` lines:
  ```python
  # ============================================================================
  # ChemCPA Model Components (from wrapper)
  # ============================================================================
  ```

- Brief inline comments for non-obvious operations:
  ```python
  # Compute differential expression (this is the key!)
  true_de = true_treated - diseased
  ```

**Docstrings:**
- Google-style docstrings for public functions:
  ```python
  def compute_de_metrics(true_treated, pred_treated, diseased):
      """
      Compute evaluation metrics on differential expression.

      Args:
          true_treated: True treated expression (n_samples, n_genes)
          pred_treated: Predicted treated expression (n_samples, n_genes)
          diseased: Diseased/control expression (n_samples, n_genes)

      Returns:
          Dictionary of metrics
      """
  ```

- Shape annotations in docstrings: `(n_samples, n_genes)`, `[batch, input_dim]`

## Function Design

**Size:**
- Training functions: 50-100 lines (`train_model()`, `model_training()`)
- Utility functions: 10-30 lines
- Main functions: 100+ lines with clear section separators

**Parameters:**
- Use keyword arguments with defaults for optional params:
  ```python
  def __init__(self, checkpoint_path: str = MULTIDCP_CHECKPOINT, device: str = "cuda"):
  ```

- Metadata passed as Optional[Dict]:
  ```python
  def predict(self, diseased: np.ndarray, metadata: Optional[Dict] = None) -> np.ndarray:
  ```

- argparse for CLI arguments with descriptive help text

**Return Values:**
- Single return type where possible (np.ndarray, Dict, float)
- Tuples for multiple returns: `return treated_df, diseased_df, gene_cols, train_idx, test_idx`
- Explicit None return indicated with `-> None`

## Module Design

**Exports:**
- No `__all__` lists defined (implicit exports)
- Single class per module pattern for model wrappers

**Structure:**
- Constants at top after imports
- Classes defined before functions
- `if __name__ == "__main__":` block for testing at bottom

**Model Wrapper Pattern:**
```python
class SomeModel(BasePerturbationModel):
    def __init__(self, ...):
        super().__init__()
        # Initialize attributes

    @property
    def name(self) -> str:
        return "ModelName"

    def train(self, diseased, treated, metadata=None):
        # Load pretrained weights or train
        self._trained = True

    def predict(self, diseased, metadata=None):
        if not self._trained:
            raise RuntimeError(...)
        # Return predictions
```

## Data Conventions

**Array Shapes:**
- Gene expression: `(n_samples, n_genes)` - samples in rows, genes in columns
- Embeddings: `(n_samples, embed_dim)`
- Standard n_genes = 10716 for PDGrapher dataset

**Metadata Columns:**
- Standard: `['sig_id', 'idx', 'pert_id', 'pert_type', 'cell_id', 'pert_idose', 'cell_type', 'dose', 'smiles']`
- Gene columns: all columns not in metadata list

**Cell Lines:**
- Standard list: `["A549", "A375", "BT20", "HELA", "HT29", "MCF7", "MDAMB231", "PC3", "VCAP"]`

---

*Convention analysis: 2026-01-18*
