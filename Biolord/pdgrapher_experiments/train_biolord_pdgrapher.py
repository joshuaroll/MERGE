# import libraries
import pandas as pd
import numpy as np
import scanpy as sc
import biolord
import warnings
from pathlib import Path

warnings.simplefilter("ignore", UserWarning)

# Set scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

# File paths
diseased_data_path = '/raid/home/public/chemoe_collab_102025/PDGrapher/data/full_downloads/chemical/real_lognorm/chemical_PDGrapher_diseased_df_10302025.pkl'
treated_data_path = "/raid/home/public/chemoe_collab_102025/PDGrapher/data/full_downloads/chemical/real_lognorm/chemical_PDGrapher_df_10302025.pkl"

print("Loading data...")

# Load both as pickle files
diseased_df = pd.read_pickle(diseased_data_path)
print(f"Diseased data shape: {diseased_df.shape}")
print(f"Diseased columns: {diseased_df.columns.tolist()[:20]}...")  # First 20 columns

treated_df = pd.read_pickle(treated_data_path)
print(f"Treated data shape: {treated_df.shape}")
print(f"Treated columns: {treated_df.columns.tolist()[:20]}...")

# Explore the data structure
print("\n--- Data Exploration ---")
print(f"Diseased dtypes:\n{diseased_df.dtypes.value_counts()}")
print(f"\nTreated dtypes:\n{treated_df.dtypes.value_counts()}")

# Identify metadata columns vs gene expression columns
# Common metadata columns in PDGrapher data
potential_metadata_cols = ['cell_id', 'pert_id', 'pert_idose', 'idx', 'sig_id', 'pert_type', 
                           'pert_iname', 'cell_type', 'dose', 'time', 'batch']

# Find which metadata columns exist in treated_df
metadata_cols = [col for col in potential_metadata_cols if col in treated_df.columns]
print(f"\nMetadata columns found: {metadata_cols}")

# Gene columns are everything else (numeric columns)
gene_cols = [col for col in treated_df.columns if col not in metadata_cols]
print(f"Number of gene columns: {len(gene_cols)}")

# Check unique values in metadata
for col in metadata_cols:
    if col in treated_df.columns:
        n_unique = treated_df[col].nunique()
        print(f"  {col}: {n_unique} unique values")
        if n_unique < 20:
            print(f"    Values: {treated_df[col].unique().tolist()}")

# Preprocess data for biolord
print("\n--- Preprocessing for Biolord ---")

# Use treated data (has perturbation responses)
# Extract gene expression matrix and metadata
X = treated_df[gene_cols].values.astype(np.float32)
obs = treated_df[metadata_cols].copy()

# Create AnnData object (cells x genes)
print(f"Creating AnnData with shape: {X.shape} (cells x genes)")
adata = sc.AnnData(X=X)
adata.var_names = gene_cols
adata.obs = obs.reset_index(drop=True)

# Add condition label to distinguish diseased vs treated if needed
adata.obs['condition'] = 'treated'

# Create train/val/test split
print("\nCreating train/val/test split...")
np.random.seed(42)
n_cells = adata.n_obs
split_indices = np.random.choice(['train', 'val', 'test'], size=n_cells, p=[0.7, 0.15, 0.15])
adata.obs['split_random'] = pd.Categorical(split_indices)

print(f"Split distribution:\n{adata.obs['split_random'].value_counts()}")

# Setup categorical attributes for biolord
# Use the metadata columns that make sense for disentanglement
categorical_attributes = []
ordered_attributes = []

# Add cell_id if present (cell type is a key attribute to disentangle)
if 'cell_id' in adata.obs.columns:
    adata.obs['cell_id'] = adata.obs['cell_id'].astype('category')
    categorical_attributes.append('cell_id')
    print(f"Added 'cell_id' as categorical attribute: {adata.obs['cell_id'].nunique()} categories")

# Add perturbation if present
if 'pert_id' in adata.obs.columns:
    adata.obs['pert_id'] = adata.obs['pert_id'].astype('category')
    categorical_attributes.append('pert_id')
    print(f"Added 'pert_id' as categorical attribute: {adata.obs['pert_id'].nunique()} categories")

# Add dose as ordered attribute if present
if 'pert_idose' in adata.obs.columns:
    # Convert dose to numeric if it's not already
    adata.obs['dose_numeric'] = pd.to_numeric(
        adata.obs['pert_idose'].astype(str).str.extract(r'([\d.]+)')[0], 
        errors='coerce'
    ).fillna(0)
    ordered_attributes.append('dose_numeric')
    print(f"Added 'dose_numeric' as ordered attribute")

print(f"\nCategorical attributes: {categorical_attributes}")
print(f"Ordered attributes: {ordered_attributes}")

# Validate we have at least one attribute
if not categorical_attributes and not ordered_attributes:
    raise ValueError("No attributes found for biolord. Check your data columns.")

# Setup biolord AnnData
print("\nSetting up biolord AnnData...")
biolord.Biolord.setup_anndata(
    adata,
    ordered_attributes_keys=ordered_attributes if ordered_attributes else None,
    categorical_attributes_keys=categorical_attributes if categorical_attributes else None,
)

# Train model
print("\nInstantiating Biolord model...")

module_params = {
    "decoder_width": 1024,
    "decoder_depth": 4,
    "attribute_nn_width": 512,
    "attribute_nn_depth": 2,
    "n_latent_attribute_categorical": 4,
    "gene_likelihood": "normal",
    "reconstruction_penalty": 1e2,
    "unknown_attribute_penalty": 1e1,
    "unknown_attribute_noise_param": 1e-1,
    "attribute_dropout_rate": 0.1,
    "use_batch_norm": False,
    "use_layer_norm": False,
    "seed": 42,
}

model = biolord.Biolord(
    adata=adata,
    n_latent=32,
    model_name="biolord_pdgrapher_model",
    module_params=module_params,
    train_classifiers=False,
    split_key="split_random",
)

print("\nTraining model...")

trainer_params = {
    "n_epochs_warmup": 0,
    "latent_lr": 1e-4,
    "latent_wd": 1e-4,
    "decoder_lr": 1e-4,
    "decoder_wd": 1e-4,
    "attribute_nn_lr": 1e-2,
    "attribute_nn_wd": 4e-8,
    "step_size_lr": 45,
    "cosine_scheduler": True,
    "scheduler_final_lr": 1e-5,
}

model.train(
    max_epochs=500,
    batch_size=512,
    plan_kwargs=trainer_params,
    early_stopping=True,
    early_stopping_patience=20,
    check_val_every_n_epoch=10,
    num_workers=4,
    enable_checkpointing=False,
)

print("\nTraining completed!")

# Evaluate model
print("\nEvaluating model...")

if hasattr(model, 'training_plan') and hasattr(model.training_plan, 'epoch_history'):
    model.epoch_history = pd.DataFrame.from_dict(model.training_plan.epoch_history)
    print("\nTraining metrics (last 5 validation epochs):")
    val_history = model.epoch_history[model.epoch_history["mode"] == "valid"]
    print(val_history.tail())

# Save model and data
print("\nSaving model...")
save_dir = Path("/raid/home/joshua/projects/biolord_comp_122025")
save_dir.mkdir(parents=True, exist_ok=True)

model_save_path = save_dir / "biolord_pdgrapher_model"
model.save(str(model_save_path), overwrite=True)
print(f"Model saved to: {model_save_path}")

adata_save_path = save_dir / "adata_pdgrapher.h5ad"
adata.write(adata_save_path)
print(f"AnnData saved to: {adata_save_path}")

print("\nPipeline completed successfully!")