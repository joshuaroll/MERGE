#!/usr/bin/env python
"""
Test the "no change" baseline on MultiDCP/PDGrapher data.
This computes R² between diseased and treated expression (predicting no change).
"""
import pickle
import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("Loading MultiDCP data pickle...")
with open("/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Data type: {type(data)}")
print(f"Data shape: {data.shape}")
print(f"Columns (first 10): {list(data.columns)[:10]}")

# Load diseased data
print("\nLoading diseased data pickle...")
with open("/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl", "rb") as f:
    diseased = pickle.load(f)

print(f"Diseased shape: {diseased.shape}")

# Identify gene columns (exclude metadata columns)
metadata_cols = ['sig_id', 'idx', 'pert_id', 'pert_type', 'cell_id', 'pert_idose', 'cell_type', 'dose', 'smiles']
gene_cols = [c for c in data.columns if c not in metadata_cols]
print(f"\nNumber of gene columns: {len(gene_cols)}")

# Compute baseline: R² between diseased and treated expression
# This is the "no change" baseline - predicting treated = diseased
print("\nComputing baseline R² (no change prediction)...")

n_samples = min(1000, len(data))  # Test on 1000 samples
r2_top20 = []
r2_top40 = []
r2_top80 = []
pearson_all = []

for i in range(n_samples):
    # Get treated and diseased expression for same sample
    treated = data.iloc[i][gene_cols].values.astype(float)
    dis = diseased.iloc[i][gene_cols].values.astype(float)

    # Compute differential expression magnitude
    de = np.abs(treated - dis)

    # Get indices of top-k differentially expressed genes
    top20_idx = np.argsort(de)[-20:]
    top40_idx = np.argsort(de)[-40:]
    top80_idx = np.argsort(de)[-80:]

    # Baseline prediction: treated = diseased (no change)
    # R² = pearson²

    # Top-20 DEGs
    r = pearsonr(dis[top20_idx], treated[top20_idx])[0]
    r2_top20.append(r**2 if not np.isnan(r) else 0)

    # Top-40 DEGs
    r = pearsonr(dis[top40_idx], treated[top40_idx])[0]
    r2_top40.append(r**2 if not np.isnan(r) else 0)

    # Top-80 DEGs
    r = pearsonr(dis[top80_idx], treated[top80_idx])[0]
    r2_top80.append(r**2 if not np.isnan(r) else 0)

    # All genes
    r = pearsonr(dis, treated)[0]
    pearson_all.append(r if not np.isnan(r) else 0)

print(f"\n=== Baseline Results (n={n_samples} samples) ===")
print(f"R² Top-20 DEGs: {np.mean(r2_top20):.4f} ± {np.std(r2_top20):.4f}")
print(f"R² Top-40 DEGs: {np.mean(r2_top40):.4f} ± {np.std(r2_top40):.4f}")
print(f"R² Top-80 DEGs: {np.mean(r2_top80):.4f} ± {np.std(r2_top80):.4f}")
print(f"Pearson (all genes): {np.mean(pearson_all):.4f} ± {np.std(pearson_all):.4f}")
print(f"R² (all genes): {np.mean(pearson_all)**2:.4f}")
