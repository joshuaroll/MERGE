# Codebase Concerns

**Analysis Date:** 2026-01-18

## Tech Debt

**Massive ensemble.py monolith (2030 lines):**
- Issue: Single file contains 10+ model classes, multiple ensemble implementations, evaluation code, and CLI
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/ensemble.py`
- Impact: Difficult to maintain, test, and extend. Changes risk breaking unrelated functionality
- Fix approach: Refactor into separate modules: `ensemble/base.py`, `ensemble/calibrated.py`, `ensemble/embedding_stacking.py`, `ensemble/evaluation.py`

**Duplicate code across training scripts:**
- Issue: `train_transigen_pdg.py`, `train_scgen_pdg.py`, `train_chemcpa_pdg.py` have nearly identical data loading, split handling, and metrics computation
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_transigen_pdg.py` (lines 135-194), `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_scgen_pdg.py` (lines 37-95), `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_chemcpa_pdg.py`
- Impact: Bug fixes and improvements must be applied to multiple files; inconsistency risk
- Fix approach: Extract common functions to `utils/training_utils.py` or extend `data_loader.py`

**Hardcoded paths throughout codebase:**
- Issue: Absolute paths to data files, splits, and external models are scattered across files
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_transigen_pdg.py` (lines 33-38), `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_scgen_pdg.py` (lines 32-34), `/raid/home/joshua/projects/PDGrapher_Baseline_Models/ensemble.py` (lines 52, 444, 577, 780)
- Impact: Code not portable; requires manual edits to run on different systems
- Fix approach: Create `config.py` or `.env` file for all paths; use environment variables

**Silenced warnings globally:**
- Issue: `warnings.filterwarnings('ignore')` at top of multiple files
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/ensemble.py` (line 18), `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_transigen_pdg.py` (line 23), `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_scgen_pdg.py` (line 24)
- Impact: Important warnings (deprecation, numerical issues) are hidden; debugging is harder
- Fix approach: Remove global suppression; add targeted suppression only where necessary

**WandB artifacts scattered in repo:**
- Issue: 3.3GB of WandB run data stored in `wandb/` directory with code snapshots
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/wandb/` (152 subdirectories)
- Impact: Large repo size; redundant code copies; clutters project
- Fix approach: Add `wandb/` to `.gitignore` (already done); clean existing runs with `rm -rf wandb/`

**train_multidcp_chemoe_pdg.py is 1169 lines:**
- Issue: Combines model definition imports, diagnostic functions, data loading, training loop, and evaluation in single file
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_multidcp_chemoe_pdg.py`
- Impact: Hard to navigate, test, and maintain; functions have unclear boundaries
- Fix approach: Move diagnostic functions to `utils/de_diagnostics.py`, data loading to `utils/pdg_data_loader.py`

## Known Bugs

**CellOT produces broken predictions:**
- Symptoms: ICNN training is numerically unstable, predictions diverge
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/cellot_wrapper.py`, `/raid/home/joshua/projects/PDGrapher_Baseline_Models/generate_predictions.py` (line 34)
- Trigger: Training CellOT model on PDGrapher data
- Workaround: CellOT added to `DISABLED_MODELS` set (line 34 of generate_predictions.py); excluded from ensemble by default

**Sample count mismatch handling is fragile:**
- Symptoms: Shape mismatches between treated/diseased arrays silently truncated
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_multidcp_chemoe_pdg.py` (lines 559-565, 677-681, 759-765, 860-866)
- Trigger: When split indices don't perfectly align with dataframe rows
- Workaround: Truncation with warning message; should raise error or handle properly

**PrecomputedEnsemble silent failure on sample mismatch:**
- Symptoms: Models with different sample counts are silently skipped
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/ensemble.py` (lines 624-642)
- Trigger: When model predictions have different test set sizes (e.g., different folds)
- Workaround: Majority vote on sample counts; minority models skipped

## Security Considerations

**No input validation on file paths:**
- Risk: Path traversal if user-supplied paths are used without validation
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/generate_predictions.py` (subprocess calls), `/raid/home/joshua/projects/PDGrapher_Baseline_Models/data_loader.py` (pickle loading)
- Current mitigation: Scripts run locally with trusted input
- Recommendations: Validate paths are within expected directories before loading

**Pickle files loaded without verification:**
- Risk: Arbitrary code execution if malicious pickle files are loaded
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/data_loader.py` (lines 46-51), all training scripts
- Current mitigation: Only loading from known project paths
- Recommendations: Use `torch.load(..., weights_only=True)` where possible; consider safer formats like safetensors

**subprocess calls with user input:**
- Risk: Command injection if paths contain shell metacharacters
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/generate_predictions.py` (line 292+), `/raid/home/joshua/projects/PDGrapher_Baseline_Models/generate_embeddings.py` (line 161+)
- Current mitigation: Paths are constructed internally, not user-supplied
- Recommendations: Use `subprocess.run()` with `shell=False` and list arguments

## Performance Bottlenecks

**Per-sample metric computation in loops:**
- Problem: Computing Pearson/Spearman correlation per sample in Python loops
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_multidcp_chemoe_pdg.py` (lines 217-224), `/raid/home/joshua/projects/PDGrapher_Baseline_Models/data_loader.py` (lines 258-275)
- Cause: scipy.stats.pearsonr called in loop; no vectorization
- Improvement path: Use numpy correlation matrix operations or numba JIT compilation

**Loading entire pickle files into memory:**
- Problem: 10K+ gene expression data loaded fully into RAM for each training run
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/data_loader.py` (lines 43-51), all training scripts
- Cause: Pickle format requires full deserialization
- Improvement path: Convert to HDF5 or parquet format for chunked loading; use memory mapping

**Redundant data loading in ensemble evaluation:**
- Problem: Ground truth data loaded separately for each model in PrecomputedEnsemble
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/ensemble.py` (lines 610-620)
- Cause: Each .npz file contains full treated_test, diseased_test arrays
- Improvement path: Store ground truth once; only store predictions in model-specific files

## Fragile Areas

**Model import try/except blocks:**
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/ensemble.py` (lines 33-89)
- Why fragile: ImportError silently sets model availability to False; ensemble runs with whatever models import successfully
- Safe modification: Check log output for "Failed to initialize" messages before interpreting results
- Test coverage: No tests for ensemble behavior when models fail to import

**Data split handling across multiple formats:**
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_transigen_pdg.py` (lines 158-191), `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_multidcp_chemoe_pdg.py` (lines 474-506)
- Why fragile: PDGrapher splits use 1-indexed folds; code manually converts between idx column and dataframe indices
- Safe modification: Verify split indices match expected format; add explicit fold index validation
- Test coverage: No unit tests for split loading

**Conda environment switching for predictions:**
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/generate_predictions.py`
- Why fragile: Relies on subprocess calls to conda environments; assumes specific env names exist
- Safe modification: Check env exists before running; provide clear error messages
- Test coverage: None; manual verification required

## Scaling Limits

**Single-GPU training only:**
- Current capacity: All training scripts use single GPU
- Limit: Memory constraints at batch_size > 128 for full gene set (10716 genes)
- Scaling path: Add DataParallel or DistributedDataParallel support; gradient accumulation

**Ensemble meta-learner per-gene loop:**
- Current capacity: ~10K genes processed in Python loop
- Limit: Ridge regression per gene is O(n_samples * n_models) per gene
- Scaling path: Vectorize ridge regression across genes; use batch matrix operations

## Dependencies at Risk

**External model repositories as subdirectories:**
- Risk: Biolord, ChemCPA, scGen repos are copied into `models/` without version pinning
- Impact: Updates to upstream repos could break compatibility
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/Biolord/`, `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/ChemCPA/`, `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/scGen/`
- Migration plan: Pin to specific commits; document versions; consider using git submodules

**Large h5ad file in Biolord directory:**
- Risk: 4.3GB `adata_atac.h5ad` file that appears to be test/example data
- Impact: Unnecessarily large repo if committed
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/Biolord/adata_atac.h5ad`
- Migration plan: Remove or add to `.gitignore`; store externally if needed

**4GB tar.gz in project root:**
- Risk: `torch_data_chemical.tar.gz` (4GB) in root directory
- Impact: Should not be committed to git
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/torch_data_chemical.tar.gz`
- Migration plan: Already in `.gitignore`; consider moving to external storage

## Missing Critical Features

**No automated testing:**
- Problem: No pytest tests, no CI/CD pipeline
- Blocks: Confident refactoring; regression detection
- Impact: Manual testing required for all changes

**No model versioning/checkpointing strategy:**
- Problem: Best models saved as `best_model.pt` without versioning
- Blocks: Reproducibility; model comparison over time
- Impact: Easy to overwrite good models accidentally

**No unified configuration system:**
- Problem: Hyperparameters scattered across argparse defaults in each script
- Blocks: Experiment tracking; reproducibility
- Impact: Hard to compare runs with different configurations

## Test Coverage Gaps

**No unit tests for data_loader.py:**
- What's not tested: `PDGrapherDataLoader` methods, `TopKEvaluator` computation
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/data_loader.py`
- Risk: Data loading bugs could silently corrupt experiments
- Priority: High

**No integration tests for ensemble:**
- What's not tested: Full ensemble training/prediction pipeline
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/ensemble.py`
- Risk: Ensemble weights could be computed incorrectly
- Priority: High

**No tests for model wrappers:**
- What's not tested: Wrapper classes in `models/*_wrapper.py`
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/multidcp_wrapper.py`, `/raid/home/joshua/projects/PDGrapher_Baseline_Models/models/biolord_wrapper.py`, etc.
- Risk: API changes in underlying models break wrappers silently
- Priority: Medium

**No validation of metrics computation:**
- What's not tested: R2, Pearson, Spearman calculations on differential expression
- Files: `/raid/home/joshua/projects/PDGrapher_Baseline_Models/train_multidcp_chemoe_pdg.py` (diagnostic functions)
- Risk: Incorrect metric computation leads to wrong model selection
- Priority: High

---

*Concerns audit: 2026-01-18*
