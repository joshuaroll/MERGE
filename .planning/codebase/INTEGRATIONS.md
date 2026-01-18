# External Integrations

**Analysis Date:** 2026-01-18

## APIs & External Services

**Experiment Tracking:**
- Weights & Biases (WandB)
  - SDK/Client: `wandb` 0.16.6+
  - Auth: `WANDB_API_KEY` (via `wandb login`)
  - Projects:
    - `MultiDCP_CheMoE_AE_DE` - MultiDCP MoE training
    - `TranSiGen_AE_DE` - TranSiGen training
    - `TranSiGen_MoE_Sparse_AE_DE` - TranSiGen Sparse MoE
    - `TranSiGen_MoE_Balanced_AE_DE` - TranSiGen Balanced MoE
    - `ChemCPA_AE_DE` - ChemCPA training
    - `scGen_AE_DE` - scGen training
  - Usage in: `train_multidcp_chemoe_pdg.py`, `train_transigen_pdg.py`, `train_chemcpa_pdg.py`, `train_scgen_pdg.py`

**No External APIs:**
- No cloud APIs (AWS, GCP, Azure)
- No external data fetching
- All data local via pickle files

## Data Storage

**Databases:**
- None - All data in pickle files

**File Storage:**
- Local filesystem only
- Pickle files (`.pkl`) for gene expression data
- PyTorch checkpoints (`.pt`) for model weights
- NPZ files for predictions
- HDF5 via AnnData (`.h5ad`) for some model inputs

**Key Data Locations:**
```
/raid/home/joshua/projects/MultiDCP_pdg/data/
├── pdg_brddrugfiltered.pkl         # Treated expression (182K samples)
└── pdg_diseased_brddrugfiltered.pkl # Diseased/control expression

/raid/home/public/chemoe_collab_102025/PDGrapher/data/
├── processed/splits/chemical/       # Train/val/test splits per cell type
│   ├── A549/random/5fold/splits.pt
│   ├── A375/random/5fold/splits.pt
│   └── ... (9 cell types)
└── full_downloads/                  # Alternative data location

/raid/home/joshua/projects/cbio_032024/zenodo_TranSiGen/
└── TranSiGen/data/LINCS2020/KPGT_emb2304.pickle  # Molecule embeddings
```

**Caching:**
- None

## Authentication & Identity

**Auth Provider:**
- None for main application
- WandB uses API key authentication (`wandb login`)

## Monitoring & Observability

**Error Tracking:**
- None - stdout/stderr logging only

**Logs:**
- Console output with print statements
- Training logs in `logs/` directory via nohup
- WandB run logs in `wandb/` directory

**Metrics Tracked (via WandB):**
- Training loss per epoch
- Validation/test metrics: Pearson, Spearman, R², RMSE
- Top-k DEG metrics (k=20, 40, 80)
- Expert selection statistics (for MoE models)

## CI/CD & Deployment

**Hosting:**
- Local compute cluster with NVIDIA GPUs
- No cloud deployment

**CI Pipeline:**
- None - Research project without CI/CD

## Environment Configuration

**Required Environment Variables:**
- `CUDA_VISIBLE_DEVICES` - GPU selection (set programmatically)
- `WANDB_API_KEY` - WandB authentication (optional, via `wandb login`)
- `WANDB_MODE=dryrun` - Disable WandB logging (optional)

**Secrets Location:**
- No secrets management
- WandB API key stored in `~/.netrc` after `wandb login`

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- WandB automatic syncing (metrics, logs, artifacts)

## External Model Dependencies

**Pre-trained Embeddings:**
- KPGT molecular embeddings (2304-dim)
  - Path: `/raid/home/joshua/projects/cbio_032024/zenodo_TranSiGen/TranSiGen/data/LINCS2020/KPGT_emb2304.pickle`
  - Used by: TranSiGen models for drug encoding

**Pre-trained Models:**
- MultiDCP pretrained weights (optional)
  - Via `--pretrained_model` argument
  - Not commonly used (retraining preferred)

## Third-Party Model Libraries

**Integrated Models:**
| Library | Source | Integration |
|---------|--------|-------------|
| scGen | `scgen` PyPI package | Direct import, `scgen.SCGEN` class |
| scvi-tools | `scvi-tools` PyPI package | Dependency for scGen |
| Biolord | Local copy in `models/Biolord/` | Custom wrapper |
| ChemCPA | Local copy in `models/ChemCPA/` | Custom wrapper |
| CellOT | Local copy in `archive/CellOT/` | Archived, custom wrapper |
| GEARS | Local copy in `archive/GEARS/` | Archived |
| PDGrapher | Results only in `models/PDGrapher/` | No code execution |
| MultiDCP | Local copy in `models/MultiDCP/` | Custom wrapper |
| TranSiGen | Local copy in `models/transigen/` | Custom implementation |

## Shared Data with Other Projects

**MultiDCP_pdg:**
- Shares data at `/raid/home/joshua/projects/MultiDCP_pdg/data/`
- Shares conda environment `mdcp_env`

**MultiDCP_CheMoE_pdg:**
- Similar architecture, separate project
- Uses same data format and evaluation methodology

---

*Integration audit: 2026-01-18*
