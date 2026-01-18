# Testing Patterns

**Analysis Date:** 2026-01-18

## Test Framework

**Runner:**
- No unified test runner configured
- Tests run as standalone scripts with `if __name__ == "__main__":` blocks
- External model tests use pytest (ChemCPA, scGen, Biolord submodules)

**Assertion Libraries:**
- numpy.testing for array comparisons
- torch.testing for tensor comparisons
- pytest.mark.parametrize for external model tests

**Run Commands:**
```bash
# Run standalone test scripts
python test_baseline.py
python test_data.py
python test_new_models.py

# External model tests (require pytest)
cd models/ChemCPA && pytest tests/
cd models/scGen && pytest tests/
cd models/Biolord && pytest tests/
```

## Test File Organization

**Location:**
- Root-level test scripts: `test_baseline.py`, `test_data.py`, `test_new_models.py`
- Model submodule tests: `models/<ModelName>/tests/test_*.py`
- Inline tests at bottom of modules: `if __name__ == "__main__":` blocks

**Naming:**
- Test files: `test_<functionality>.py`
- Test functions: `test_<what_is_being_tested>()`

**Structure:**
```
/raid/home/joshua/projects/PDGrapher_Baseline_Models/
├── test_baseline.py          # No-change baseline evaluation
├── test_data.py              # Data format exploration
├── test_new_models.py        # Multi-model comparison tests
├── models/
│   ├── ChemCPA/tests/
│   │   ├── test_embedding.py
│   │   ├── test_dosers.py
│   │   └── test_dataset.py
│   ├── scGen/tests/
│   │   └── test_scgen.py
│   └── Biolord/tests/
│       └── test_basic.py
```

## Test Structure

**Standalone Test Scripts Pattern:**
```python
#!/usr/bin/env python
"""
Brief description of what is tested.
"""
import pickle
import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
data = pickle.load(open("path/to/data.pkl", "rb"))

# Compute metrics
r2_scores = []
for i in range(n_samples):
    # Calculation logic
    r2_scores.append(result)

# Print results
print(f"=== Results (n={n_samples}) ===")
print(f"R² Top-20: {np.mean(r2_scores):.4f}")
```

**Module Inline Test Pattern:**
```python
if __name__ == "__main__":
    # Test the wrapper
    print("Testing ModelName wrapper...")

    # Create model
    model = ModelClass()

    # Dummy data
    n_samples = 10
    diseased = np.random.randn(n_samples, n_genes).astype(np.float32)
    treated = np.random.randn(n_samples, n_genes).astype(np.float32)

    # Train and predict
    model.train(diseased, treated)
    predictions = model.predict(diseased)

    # Verify shapes
    print(f"Input shape: {diseased.shape}")
    print(f"Output shape: {predictions.shape}")
    print("Test passed!")
```

**External Model pytest Pattern:**
```python
import pytest
import torch.testing
import numpy.testing

@pytest.mark.parametrize("param", [value1, value2, value3])
def test_functionality(param):
    # Setup
    model = create_model(param)

    # Execute
    result = model.forward(input)

    # Assert
    torch.testing.assert_close(result, expected)
```

## Mocking

**Framework:** No dedicated mocking framework (no unittest.mock usage detected)

**Patterns:**
- Real data loading in tests (heavy integration tests)
- Random data for quick smoke tests:
  ```python
  diseased = np.random.randn(n_samples, n_genes).astype(np.float32)
  treated = np.random.randn(n_samples, n_genes).astype(np.float32)
  ```

- Synthetic data from scvi for model tests:
  ```python
  adata = scvi.data.synthetic_iid()
  ```

**What to Mock:**
- Not commonly used; tests rely on real or synthetic data

**What NOT to Mock:**
- Model forward passes (always real)
- Data loading (tests data integrity too)
- GPU operations (tested on actual hardware)

## Fixtures and Factories

**Test Data:**
- Direct pickle loading from production paths:
  ```python
  with open("/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl", "rb") as f:
      data = pickle.load(f)
  ```

- PDGrapherDataLoader class for structured access:
  ```python
  loader = PDGrapherDataLoader()
  train_idx, test_idx = loader.get_train_test_split("A549", fold=0)
  treated, diseased = loader.get_expression_arrays(test_idx)
  ```

- Limited sample sizes for quick tests:
  ```python
  n_samples = min(1000, len(data))  # Test on 1000 samples
  test_idx = test_idx[:500]  # Limit test samples
  ```

**Data Factory Pattern in `data_loader.py`:**
```python
def get_expression_arrays(self, indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Get treated and diseased expression as numpy arrays."""
    if indices is None:
        treated = self.treated_df[self.gene_cols].values.astype(np.float32)
        diseased = self.diseased_df[self.gene_cols].values.astype(np.float32)
    else:
        treated = self.treated_df.iloc[indices][self.gene_cols].values.astype(np.float32)
        diseased = self.diseased_df.iloc[indices][self.gene_cols].values.astype(np.float32)
    return treated, diseased
```

**Location:**
- No separate fixtures directory
- Data loading code embedded in test files or using `data_loader.py`

## Coverage

**Requirements:** None enforced

**View Coverage:**
- No coverage tool configured
- Manual inspection of test coverage via test file contents

## Test Types

**Unit Tests:**
- Minimal pure unit tests
- MoE module tests in `models/moe_modules.py`:
  ```python
  if __name__ == "__main__":
      # Test both MoE styles
      x = torch.randn(batch_size, input_dim)
      sparse_moe = MoELayer(input_dim, output_dim, moe_style='sparse')
      out_sparse = sparse_moe(x)
      print(f"Input: {x.shape}, Output: {out_sparse.shape}")
  ```

**Integration Tests:**
- Primary testing approach
- End-to-end model training and prediction tests:
  ```python
  # Full model test in test_new_models.py
  model.train(diseased_test, treated_test, metadata)
  predictions = model.predict(diseased_test, metadata)
  results = evaluate_predictions(predictions, treated_test, diseased_test)
  ```

**E2E Tests:**
- Training scripts serve as E2E tests
- Run via bash scripts: `run_training.sh`, `run_all_transigen.sh`

## Common Patterns

**Evaluation Consistency:**
All tests use differential expression metrics:
```python
def evaluate_predictions(predictions, treated_test, diseased_test, k_values=[20, 40, 80]):
    """Evaluate predictions using standard metrics."""
    true_de = treated_test - diseased_test
    pred_de = predictions - diseased_test

    results = {}
    for k in k_values:
        r2_scores = []
        for i in range(n_samples):
            de_mag = np.abs(true_de[i])
            top_k_idx = np.argsort(de_mag)[-k:]
            r2 = r2_score(true_de[i, top_k_idx], pred_de[i, top_k_idx])
            r2_scores.append(r2)
        results[f'r2_top{k}'] = np.mean(r2_scores)
    return results
```

**Model Import Testing:**
```python
# Check model availability
MODELS_AVAILABLE = {'transigen': False, 'multidcp_moe': False}
try:
    from models.transigen_wrapper import TranSiGenModel
    MODELS_AVAILABLE['transigen'] = True
except ImportError as e:
    print(f"TranSiGen import failed: {e}")
```

**Shape Verification:**
```python
print(f"Input shape: {diseased.shape}")
print(f"Output shape: {predictions.shape}")
assert predictions.shape == treated_test.shape
```

**Metric Printing:**
```python
print(f"\n=== {model_name} Results ===")
for k in [20, 40, 80]:
    print(f"R² Top-{k} DEGs: {results[f'r2_top{k}']:.4f}")
print(f"Pearson DE: {results['pearson_de']:.4f}")
```

**Error Recovery:**
```python
try:
    results, preds = test_model(model, diseased, treated, metadata, "ModelName")
    if results:
        all_results['model'] = results
except Exception as e:
    print(f"\nModel FAILED: {e}")
    import traceback
    traceback.print_exc()
```

## TopKEvaluator Standard Test Class

The `TopKEvaluator` class in `data_loader.py` provides standardized evaluation:

```python
class TopKEvaluator:
    """Standard evaluator for top-k DEG metrics."""

    def __init__(self, k_values: List[int] = [20, 40, 80]):
        self.k_values = k_values

    def compute_metrics(self, true_treated, pred_treated, diseased) -> Dict[str, float]:
        """Compute R² from Pearson² on differential expression."""
        # ... implementation

    def print_results(self, results: Dict[str, float], model_name: str = "Model"):
        """Print formatted results."""
        print(f"\n=== {model_name} Results ===")
        for k in self.k_values:
            print(f"R² Top-{k} DEGs: {results[f'r2_top{k}']:.4f}")
```

**Usage:**
```python
from data_loader import PDGrapherDataLoader, TopKEvaluator

loader = PDGrapherDataLoader()
evaluator = TopKEvaluator()

treated, diseased = loader.get_expression_arrays(test_idx)
pred = model.predict(diseased)

results = evaluator.compute_metrics(treated, pred, diseased)
evaluator.print_results(results, "ModelName")
```

## Test Data Paths

**Production Data (used in tests):**
- Treated: `/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl`
- Diseased: `/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl`
- Splits: `/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical`

**Predictions (test artifacts):**
- Output: `predictions/` directory with NPZ files
- Pattern: `{model_name}_{cell_line}_fold{fold}.npz`

---

*Testing analysis: 2026-01-18*
