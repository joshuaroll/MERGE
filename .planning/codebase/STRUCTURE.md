# Codebase Structure

**Analysis Date:** 2025-01-18

## Directory Layout

```
PDGrapher_Baseline_Models/
├── data/                       # Shared results and derived data
├── docs/                       # Documentation
├── embeddings/                 # Precomputed embeddings
├── logs/                       # Training logs (nohup output)
├── models/                     # Model implementations and wrappers
│   ├── Biolord/               # External: Biolord source
│   ├── ChemCPA/               # External: ChemCPA source
│   ├── MultiDCP/              # External: MultiDCP source
│   ├── PDGrapher/             # PDGrapher data and results
│   ├── scGen/                 # External: scGen source
│   ├── transigen/             # TranSiGen adapted for PDGrapher
│   ├── transigen_moe/         # TranSiGen with MoE routing
│   ├── multidcp_chemoe/       # MultiDCP with CheMoE architecture
│   ├── *_wrapper.py           # Model wrapper classes
│   ├── base_model.py          # Abstract base class + baselines
│   └── moe_modules.py         # Shared MoE components
├── output/                     # Model outputs organized by model
│   ├── scgen/                 # scGen checkpoints and predictions
│   ├── chemcpa/               # ChemCPA outputs
│   └── pdgrapher_A549/        # PDGrapher outputs
├── predictions/               # Saved prediction arrays
├── results/                   # Evaluation results
├── trained_models/            # Model checkpoints (primary location)
├── utils/                     # Shared utilities
├── wandb/                     # WandB run data
├── archive/                   # Archived/deprecated code (CellOT, GEARS)
├── train_*.py                 # Training scripts per model
├── run_*.py                   # Evaluation/orchestration scripts
├── run_*.sh                   # Batch execution scripts
├── data_loader.py             # Unified data loading
├── ensemble.py                # Ensemble implementations
└── README.md                  # Project documentation
```

## Directory Purposes

**`models/`:**
- Purpose: All model implementations, both custom and external
- Contains: Wrapper classes, model architectures, external library copies
- Key files:
  - `base_model.py`: Abstract base + NoChange/MeanShift/PerGeneLinear baselines
  - `moe_modules.py`: SparseGatingNetwork, MultiDCPGatingNetwork, MoELayer, FusionMoE
  - `transigen/model.py`: TranSiGen_PDG adapted for 10716 genes
  - `multidcp_chemoe/model.py`: MultiDCP with CheMoE routing

**`models/transigen/`:**
- Purpose: TranSiGen dual-VAE adapted for PDGrapher's 10716 genes
- Contains: Model definition with encoder, decoder, fusion layers
- Key files: `model.py` (TranSiGen_PDG class)

**`models/multidcp_chemoe/`:**
- Purpose: MultiDCP extended with Mixture-of-Experts architecture
- Contains: Model with 4 experts, top-k=2 gating, load balancing
- Key files: `model.py`, neural fingerprint components

**`models/Biolord/`, `models/ChemCPA/`, etc.:**
- Purpose: External model source code (vendored)
- Contains: Full source trees from original repositories
- Generated: No (copied from external repos)
- Committed: Yes

**`output/`:**
- Purpose: Training outputs organized by model type
- Contains: Model checkpoints, predictions, metrics per run
- Pattern: `output/{model}/{model}_{cell}_{fold}/`

**`trained_models/`:**
- Purpose: Primary location for model checkpoints
- Contains: Best model checkpoints, final models, training history
- Pattern: `trained_models/{model}_{cell}_fold{n}/`
- Key files: `best_model.pt`, `final_model.pt`, `predictions.npz`, `history.pkl`

**`utils/`:**
- Purpose: Shared utility functions
- Contains: Data processing, metrics, training utilities
- Key files:
  - `datareader_pdg_12122025.py`: PyTorch Dataset/DataLoader classes
  - `data_utils_pdg.py`: Data transformation functions
  - `metric.py`: Pearson, Spearman, R², RMSE, MAE
  - `multidcp_ae_pdg_utils.py`: Training loop helpers
  - `molecules.py`: Molecular feature processing
  - `molecule_utils.py`: SMILES handling utilities

**`data/`:**
- Purpose: Derived data and aggregated results
- Contains: Results CSV, intermediate files
- Key files: `topk_r2_results.csv` (shared results file with file locking)

**`archive/`:**
- Purpose: Deprecated or experimental code
- Contains: CellOT wrapper, GEARS experiments
- Generated: No
- Committed: Yes (for reference)

## Key File Locations

**Entry Points:**
- `train_transigen_pdg.py`: TranSiGen and TranSiGen_MoE training
- `train_chemcpa_pdg.py`: ChemCPA training
- `train_scgen_pdg.py`: scGen training (requires scgen_env conda)
- `train_multidcp_chemoe_pdg.py`: MultiDCP-CheMoE training
- `train_pdgrapher_pdg.py`: PDGrapher training
- `run_evaluation.py`: Quick single-cell evaluation
- `run_full_evaluation.py`: Full cross-validation evaluation

**Configuration:**
- No config files - all configuration via CLI args
- Default paths hardcoded at top of each training script
- External data: `/raid/home/joshua/projects/MultiDCP_pdg/data/`
- Splits: `/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical`

**Core Logic:**
- `data_loader.py`: PDGrapherDataLoader, TopKEvaluator
- `ensemble.py`: EnsembleModel, LearnedEnsemble, StackingEnsemble
- `models/base_model.py`: BasePerturbationModel, baseline models
- `models/moe_modules.py`: MoE components shared across models

**Testing:**
- `test_baseline.py`: Test baseline models
- `test_data.py`: Test data loading
- `test_new_models.py`: Test new model implementations

## Naming Conventions

**Files:**
- Training scripts: `train_{model}_pdg.py`
- Batch scripts: `run_{model}_batch.sh` or `run_all_{model}.sh`
- Model wrappers: `models/{model}_wrapper.py`
- Utility modules: `utils/{purpose}.py`

**Directories:**
- Model outputs: `{model}_{cell}_fold{n}/`
- External models: `models/{ModelName}/` (PascalCase)
- Custom models: `models/{model_name}/` (snake_case)

**Classes:**
- Model wrappers: `{Model}Model` (e.g., `TranSiGenModel`, `ChemCPAModel`)
- Base models: `{Name}Baseline` (e.g., `PerGeneLinearBaseline`)
- Ensembles: `{Type}Ensemble` (e.g., `StackingEnsemble`)

**Functions:**
- Metric computation: `compute_{metric}_metrics()`
- Data loading: `load_data()`, `load_{type}_embeddings()`
- Training: `train_model()`, `evaluate_model()`

## Where to Add New Code

**New Model:**
1. Create wrapper: `models/{new_model}_wrapper.py`
   - Inherit from `BasePerturbationModel`
   - Implement `train()`, `predict()`, `name` property
2. Create training script: `train_{new_model}_pdg.py`
   - Follow pattern from `train_transigen_pdg.py`
   - Add CLI args, WandB logging, result saving
3. Register in `ensemble.py` with import and availability flag
4. Add batch script: `run_{new_model}_batch.sh`

**New Feature (e.g., MoE variant):**
1. Implement in `models/moe_modules.py` or new file in `models/`
2. Integrate into existing model wrapper or training script
3. Update CLI args in training script

**Utilities:**
- Shared helpers: `utils/{purpose}.py`
- Model-specific helpers: Inside model directory (e.g., `models/transigen/`)

**New Baseline:**
1. Add class to `models/base_model.py`
2. Inherit from `BasePerturbationModel`
3. Update `run_evaluation.py` to include in evaluation

## Special Directories

**`wandb/`:**
- Purpose: WandB experiment tracking data
- Generated: Yes (by WandB during training)
- Committed: No (in .gitignore)
- Contains: Run artifacts, logs, checkpoints synced to cloud

**`.planning/`:**
- Purpose: GSD planning documents
- Generated: By mapping tools
- Committed: Typically yes
- Contains: Phase plans, codebase analysis docs

**`__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No
- Location: In each Python package directory

**External Data (not in repo):**
- `/raid/home/joshua/projects/MultiDCP_pdg/data/`: Expression data pickles
- `/raid/home/public/chemoe_collab_102025/PDGrapher/data/`: PDGrapher splits
- `/raid/home/joshua/projects/cbio_032024/zenodo_TranSiGen/`: Molecular embeddings

---

*Structure analysis: 2025-01-18*
