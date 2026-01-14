# MERGE

**M**odel **E**nsemble for **R**esponse **G**ene **E**xpression

An ensemble framework for predicting differential gene expression in response to chemical perturbations. Combines multiple state-of-the-art models including scGen, ChemCPA, Biolord, and CellOT with simple but effective baselines.

Benchmarked against PDGrapher ([Nature Biomedical Engineering 2025](https://www.nature.com/articles/s41551-025-01481-x)).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/joshuaroll/MERGE.git
cd MERGE

# Quick evaluation on single cell line
python run_evaluation.py --mode quick --cell_line A549

# Full evaluation on all cell lines (5-fold CV)
python run_evaluation.py --mode full --n_folds 5 --output_dir results

# Train individual models
conda activate scgen_env && python models/scgen_wrapper.py
conda activate biolord_env && python models/biolord_wrapper.py
conda activate cellot_env && python models/cellot_wrapper.py
conda activate cellot_env && python models/chemcpa_wrapper.py
```

## Ensemble Components

| Model | Source | Status | Notes |
|-------|--------|--------|-------|
| PDGrapher | [valence-labs/PDGrapher](https://github.com/valence-labs/pdgrapher) | 109% of paper | GNN on PPI network |
| PerGeneLinear | MERGE | 2nd best | Per-gene ridge regression |
| scGen | [theislab/scgen](https://github.com/theislab/scgen) | ~94% of paper | VAE latent arithmetic |
| ChemCPA | [theislab/chemCPA](https://github.com/theislab/chemCPA) | ~95% of paper | Compositional perturbation autoencoder |
| MultiDCP | [ICDM-UESTC/MultiDCP](https://github.com/ICDM-UESTC/MultiDCP) | Pretrained | Neural fingerprints + multimodal fusion |
| Biolord | [nitzanlab/biolord](https://github.com/nitzanlab/biolord) | ~93% of paper | Disentangled representation learning |
| CellOT | [bunnech/cellot](https://github.com/bunnech/cellot) | Matches paper | Optimal transport with ICNN |
| NoChange | MERGE | Baseline | Predicts DE = 0 |
| MeanShift | MERGE | Baseline | Adds mean differential expression |

## Results (A549)

### MERGE Models (Trained from Scratch)

```
Model                   R² Top-20    R² Top-40    R² Top-80    Fold     Notes
MultiDCP (retrained)       0.7698       0.7593       0.7494    fold 1   <- Best (trained from scratch)
PDGrapher (retrained)      0.7666       0.7561       0.7463    fold 1
PerGeneLinear              0.7636       0.7554       0.7447    fold 0
ChemCPA                    0.7056       0.6893       0.6680    fold 0
scGen                      0.7002       0.6775       0.6582    fold 0
Biolord (tuned)            0.6769       0.6442       0.6062    fold 0
MultiDCP (pretrained)      0.6537       0.6349       0.6120    fold 0   Uses pretrained weights
CellOT_PCA100              0.4603       0.4551       0.4501    fold 0
MeanShift                  0.3116       0.3006       0.2976    fold 0
CellOT (no PCA)            0.0885         -            -       fold 0
NoChange                   0.0000       0.0000       0.0000    fold 0
```

### Published Baselines (PDGrapher paper)

```
Model                   R² Top-20    R² Top-40    R² Top-80
scGen (paper)              0.7402       0.7502       0.7547
ChemCPA (paper)            0.7442       0.7577       0.7645
Biolord (paper)            0.7248       0.7362       0.7432
PDGrapher (paper)          0.7031       0.7124       0.7184
Baseline (paper)           0.6603       0.6665       0.6745
CellOT (paper)             0.0031       0.0035       0.0036
```

### Key Findings

1. **MultiDCP (retrained)** achieves the best performance (R² Top-20 = 0.7698), 9.5% higher than paper's reported baseline
2. **PDGrapher (retrained)** achieves strong performance (R² Top-20 = 0.7666), 9% higher than paper's reported value
3. **PerGeneLinear** remains highly competitive (R² = 0.7636), nearly matching deep learning models with far simpler architecture
4. **scGen** and **ChemCPA** reach 94-95% of published performance
5. **Biolord** benefits significantly from more HVGs (5000 vs 2000: +9.8% improvement)
6. **CellOT** requires PCA dimensionality reduction for reasonable performance on this task
7. **MultiDCP (pretrained vs retrained)**: Retraining improves R² Top-20 from 0.6537 to 0.7698 (+17.8%)

## Evaluation Metrics

All metrics computed on **differential expression (DE = treated - diseased)**:

- **R² Top-k DEGs**: Pearson² on top-k genes selected by |true_DE| magnitude
- **R² All**: Pearson² across all 10,716 genes
- Methodology matches PDGrapher paper (Extended Figure 1)

## Dataset

MultiDCP/PDGrapher chemical perturbation data:
- 182,246 samples across 9 cell lines
- 10,716 genes per sample
- Cell lines: A549, A375, BT20, HELA, HT29, MCF7, MDAMB231, PC3, VCAP

## Project Structure

```
MERGE/
├── data_loader.py          # Unified data loader with TopKEvaluator
├── ensemble.py             # Ensemble implementations
├── run_evaluation.py       # Main evaluation script
├── models/
│   ├── base_model.py       # Base class + simple baselines
│   ├── cellot_wrapper.py   # CellOT wrapper
│   ├── chemcpa_wrapper.py  # ChemCPA wrapper
│   ├── scgen_wrapper.py    # scGen wrapper
│   ├── biolord_wrapper.py  # Biolord wrapper
│   └── pdgrapher_wrapper.py
├── Biolord/                # Biolord source
├── CellOT/                 # CellOT source
├── ChemCPA/                # ChemCPA source
├── GEARS/                  # GEARS source
├── PDGrapher/              # PDGrapher results
└── scGen/                  # scGen source
```

## Installation

```bash
# Create conda environments for each model
conda create -n scgen_env python=3.9
conda create -n biolord_env python=3.9
conda create -n cellot_env python=3.9

# Install dependencies (see individual model directories)
```

## Usage

```python
from data_loader import PDGrapherDataLoader, TopKEvaluator
from models.base_model import PerGeneLinearBaseline

# Load data
loader = PDGrapherDataLoader()
train_idx, test_idx = loader.get_train_test_split("A549", fold=0)
treated_train, diseased_train = loader.get_expression_arrays(train_idx)
treated_test, diseased_test = loader.get_expression_arrays(test_idx)

# Train model
model = PerGeneLinearBaseline()
model.train(diseased_train, treated_train)
predictions = model.predict(diseased_test)

# Evaluate on differential expression
evaluator = TopKEvaluator()
results = evaluator.compute_metrics(treated_test, predictions, diseased_test)
evaluator.print_results(results, "PerGeneLinear")
```

## Model Architectures

### PerGeneLinear
Per-gene ridge regression learning independent linear models for each of 10,716 genes. Simple but highly effective for capturing gene-specific treatment responses.

### scGen
VAE-based approach using latent space arithmetic. Computes perturbation vectors as the difference between treated and control latent means.

### ChemCPA
Compositional Perturbation Autoencoder with drug and dose embeddings. Learns disentangled representations of cellular states and perturbations.

### Biolord
Disentangled representation learning treating treatment condition as a categorical attribute. Best with n_hvg=5000, n_latent=128.

### CellOT
Optimal transport using Input Convex Neural Networks (ICNN). Requires PCA (100 components) for high-dimensional gene expression data.

## Citation

If you use MERGE in your research, please cite:

```bibtex
@software{merge2025,
  title={MERGE: Model Ensemble for Response Gene Expression},
  author={Roll, Joshua},
  year={2025},
  url={https://github.com/joshuaroll/MERGE}
}
```

## License

MIT License

## Acknowledgments

- PDGrapher paper and dataset: [Nature BME 2025](https://www.nature.com/articles/s41551-025-01481-x)
- Model implementations: [scGen](https://github.com/theislab/scgen), [ChemCPA](https://github.com/theislab/chemCPA), [Biolord](https://github.com/nitzanlab/biolord), [CellOT](https://github.com/bunnech/cellot)
