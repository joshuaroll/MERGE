# Technology Stack

**Analysis Date:** 2026-01-18

## Languages

**Primary:**
- Python 3.11 - All model training, data processing, and evaluation scripts

**Secondary:**
- Bash - Training orchestration scripts (`run_training.sh`, `run_transigen_batch.sh`, etc.)

## Runtime

**Environment:**
- Python 3.11 (via Miniforge/Conda)
- Multiple conda environments for different model dependencies:
  - `scgen_py311` - scGen and general training
  - `scgen_env` - Alternative scGen environment
  - `biolord_env` - Biolord model
  - `cellot_env` - CellOT and ChemCPA models
  - `mdcp_env` - MultiDCP models (shared with MultiDCP_pdg project)

**Package Manager:**
- Conda (Miniforge)
- pip (for packages not in conda)
- Environment specs in: `envs/scgen_py311.yml`

## Frameworks

**Core:**
- PyTorch 2.6.0 - Primary deep learning framework
- PyTorch Lightning 2.6.0 - Training loop abstraction

**Machine Learning:**
- scikit-learn 1.8.0 - Metrics, preprocessing, baselines
- NumPy 2.3.5 - Array operations
- SciPy 1.17.0 - Statistical functions (Pearson, Spearman correlations)

**Single-Cell Specific:**
- AnnData 0.12.7 - Single-cell data format
- Scanpy 1.11.5 - Single-cell preprocessing
- scvi-tools 1.4.1 - Variational inference for single-cell
- scGen 2.1.1 - VAE-based perturbation prediction

**Testing:**
- No formal test framework detected

**Build/Dev:**
- wandb 0.16.6+ - Experiment tracking
- tqdm 4.67.1 - Progress bars

## Key Dependencies

**Critical:**
- `torch` 2.6.0 - Core neural network operations
- `anndata` 0.12.7 - Gene expression data structure (AnnData format)
- `scgen` 2.1.1 - VAE model for perturbation prediction
- `wandb` 0.16.6+ - Experiment tracking and logging
- `filelock` - Concurrent CSV file access for results logging

**Infrastructure:**
- `pandas` 2.3.3 - Data manipulation and pickle file handling
- `pickle` (stdlib) - Serialized data loading
- `h5py` 3.15.1 - HDF5 file support for AnnData

**Model-Specific:**
- `scvi-tools` 1.4.1 - Base for scGen
- `pytorch-lightning` 2.6.0 - Training loops
- `pyro-ppl` 1.9.1 - Probabilistic programming (for VAE models)

## Configuration

**Environment:**
- GPU selection via `CUDA_VISIBLE_DEVICES` environment variable
- No `.env` files detected - paths hardcoded in scripts
- Conda environment activation required before running

**Key Hardcoded Paths:**
- Data pickles: `/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_*.pkl`
- Splits: `/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical`
- Molecule embeddings: `/raid/home/joshua/projects/cbio_032024/zenodo_TranSiGen/TranSiGen/data/LINCS2020/KPGT_emb2304.pickle`

**Build:**
- No formal build system
- Direct script execution: `python train_*.py`

## Platform Requirements

**Development:**
- Linux (tested on Linux 5.4.0)
- NVIDIA GPU with CUDA 12.4+ support
- 8+ GPUs available for parallel training
- High memory (models train on 10,716 genes)

**Production:**
- Same as development (research project)
- Models trained locally, results tracked via WandB

## GPU/CUDA Configuration

**CUDA Version:** 12.4
- `nvidia-cublas-cu12` 12.4.5.8
- `nvidia-cudnn-cu12` 9.1.0.70
- `nvidia-nvjitlink-cu12` 12.4.127

**GPU Selection Pattern:**
```python
# Before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_arg
# Then: device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```

## Data Format

**Input Data:**
- Pickle files containing pandas DataFrames
- Shape: ~182,246 samples x 10,716 genes
- Columns: gene expression values + metadata (`sig_id`, `idx`, `pert_id`, `cell_id`, `pert_idose`, `smiles`, etc.)

**Model Checkpoints:**
- PyTorch state dicts (`.pt` files)
- scGen models saved via `model.save()` method

**Results:**
- WandB logging
- CSV files: `data/topk_r2_results.csv`
- NPZ files: predictions and ground truth

---

*Stack analysis: 2026-01-18*
