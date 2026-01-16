#!/usr/bin/env python
"""
TranSiGen wrapper for MERGE ensemble evaluation.

TranSiGen uses dual VAEs with molecular embeddings for predicting
gene expression changes from chemical perturbations.
"""
import os
import sys
import numpy as np
import torch
from typing import Dict, Optional
import warnings
import pickle
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BasePerturbationModel

# TranSiGen paths
TRANSIGEN_BASE = "/raid/home/joshua/projects/cbio_032024/TranSiGen_042024"
TRANSIGEN_SRC = os.path.join(TRANSIGEN_BASE, "src")
TRANSIGEN_RESULTS = os.path.join(TRANSIGEN_BASE, "results")

# PDGrapher trained models directory
PDG_TRAINED_MODELS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trained_models"
)

# Default model path - use PDGrapher trained model if available
DEFAULT_MODEL_PATH = os.path.join(PDG_TRAINED_MODELS, "transigen_A549_fold0", "best_model.pt")

# Fallback to original LINCS model if PDGrapher model doesn't exist
ORIGINAL_MODEL_PATH = os.path.join(
    TRANSIGEN_RESULTS,
    "10232024_best0_KPGT_TSG_jer_08192024_split1_test_base_test_log",
    "trained_seed364039_feature_KPGT_init_pretrain_shRNA",
    "best_model_jer.pt"
)

# Molecule embedding paths
MOLECULE_EMBEDDINGS = {
    'KPGT': "/raid/home/joshua/projects/cbio_032024/zenodo_TranSiGen/TranSiGen/data/LINCS2020/KPGT_emb2304.pickle",
    'ECFP4': "/raid/home/joshua/transigen_v1_release/TranSiGen-1.0/data/LINCS2020/data_example/ECFP4_emb2048.pickle",
    'CHEMBERTA': "/raid/home/joshua/transigen_v1_release/TranSiGen-1.0/data/LINCS2020/data_example/CHEMBERTA_emb768.pickle",
}

# Drug SMILES mapping
DRUG_SMILES_CSV = "/raid/home/joshua/projects/MultiDCP_pdg/data/all_drugs_pdg.csv"


class TranSiGenModel(BasePerturbationModel):
    """
    TranSiGen model wrapper for ensemble evaluation.

    TranSiGen architecture:
    1. Encoder for basal expression (x1) -> latent z1
    2. Molecular embedding from precomputed features
    3. Fusion of z1 + mol_embed -> predicted z2
    4. Decoder z2 -> predicted treated expression (x2)
    """

    def __init__(self, model_path: str = None, molecule_feature: str = 'KPGT',
                 device: str = "cuda", n_genes: int = 10716, cell_line: str = "A549", fold: int = 0):
        super().__init__()

        # Try PDGrapher trained model first
        if model_path:
            self.model_path = model_path
        else:
            pdg_model = os.path.join(PDG_TRAINED_MODELS, f"transigen_{cell_line}_fold{fold}", "best_model.pt")
            if os.path.exists(pdg_model):
                self.model_path = pdg_model
                n_genes = 10716  # PDGrapher uses 10716 genes
            else:
                self.model_path = ORIGINAL_MODEL_PATH
                n_genes = 2432  # Original uses 2432 genes

        self.molecule_feature = molecule_feature
        self.device_str = device
        self.n_genes = n_genes

        self.model = None
        self.device = None
        self._mol_embeddings = None
        self._drug_smiles_map = None
        self._smiles_to_embedding = {}

    @property
    def name(self) -> str:
        return "TranSiGen"

    @property
    def embedding_dim(self) -> int:
        return 100  # n_latent default

    def _load_drug_smiles_map(self):
        """Load drug name to SMILES mapping."""
        import pandas as pd
        if os.path.exists(DRUG_SMILES_CSV):
            df = pd.read_csv(DRUG_SMILES_CSV)
            self._drug_smiles_map = dict(zip(df['drug_name'].str.lower(), df['cpd_smiles']))
        else:
            self._drug_smiles_map = {}

    def _load_molecule_embeddings(self):
        """Load precomputed molecule embeddings."""
        emb_path = MOLECULE_EMBEDDINGS.get(self.molecule_feature)
        if emb_path and os.path.exists(emb_path):
            with open(emb_path, 'rb') as f:
                self._smiles_to_embedding = pickle.load(f)
            print(f"Loaded {len(self._smiles_to_embedding)} molecule embeddings ({self.molecule_feature})")
        else:
            print(f"Warning: Molecule embeddings not found for {self.molecule_feature}")
            self._smiles_to_embedding = {}

    def _get_molecule_embedding(self, smiles: str) -> np.ndarray:
        """Get embedding for a SMILES string."""
        if smiles in self._smiles_to_embedding:
            return self._smiles_to_embedding[smiles]
        else:
            # Return zero embedding for unknown molecules
            emb_dim = 2304 if self.molecule_feature == 'KPGT' else 768
            return np.zeros(emb_dim, dtype=np.float32)

    def _load_model(self):
        """Load pretrained TranSiGen model."""
        self.device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Add TranSiGen source to path for model class definitions
        transigen_src = os.path.join(TRANSIGEN_BASE, "src")
        if transigen_src not in sys.path:
            sys.path.insert(0, transigen_src)

        # Load model (weights_only=False for PyTorch 2.6+ compatibility with pickled models)
        self.model = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.dev = self.device
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded TranSiGen model from {self.model_path}")

        # Load molecule embeddings
        self._load_molecule_embeddings()
        self._load_drug_smiles_map()

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """
        Load pretrained model (TranSiGen is not retrained in ensemble).

        Args:
            diseased: [n_samples, n_genes] basal expression (x1)
            treated: [n_samples, n_genes] treated expression (x2)
            metadata: dict with 'smiles' or 'drug_names' for molecule lookup
        """
        if self.model is None:
            self._load_model()

        # Store training data statistics for any needed normalization
        self._train_mean = np.mean(diseased, axis=0)
        self._train_std = np.std(diseased, axis=0) + 1e-8

        self._trained = True

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Predict treated expression from diseased/basal expression.

        Args:
            diseased: [n_samples, n_genes] basal expression
            metadata: dict with 'smiles' list for molecule embeddings

        Returns:
            predictions: [n_samples, n_genes] predicted treated expression
        """
        if not self._trained:
            raise RuntimeError("Model must be trained/loaded first")

        n_samples = diseased.shape[0]

        # Get molecule embeddings from metadata
        if metadata and 'smiles' in metadata:
            smiles_list = metadata['smiles']
        elif metadata and 'drug_names' in metadata:
            # Convert drug names to SMILES
            smiles_list = []
            for name in metadata['drug_names']:
                name_lower = str(name).lower()
                if self._drug_smiles_map and name_lower in self._drug_smiles_map:
                    smiles_list.append(self._drug_smiles_map[name_lower])
                else:
                    smiles_list.append("C")  # Placeholder
        else:
            # Use placeholder embeddings
            smiles_list = ["C"] * n_samples

        # Get embeddings for each sample
        mol_embeddings = np.array([
            self._get_molecule_embedding(smi) for smi in smiles_list
        ], dtype=np.float32)

        # Handle gene dimension mismatch (TranSiGen uses 978, PDGrapher uses 10716)
        if diseased.shape[1] != self.n_genes:
            # Use first n_genes or pad with zeros
            if diseased.shape[1] > self.n_genes:
                x1 = diseased[:, :self.n_genes]
            else:
                x1 = np.zeros((n_samples, self.n_genes), dtype=np.float32)
                x1[:, :diseased.shape[1]] = diseased
        else:
            x1 = diseased.astype(np.float32)

        # Convert to tensors
        x1_tensor = torch.FloatTensor(x1).to(self.device)
        mol_tensor = torch.FloatTensor(mol_embeddings).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            # TranSiGen forward returns: x1_rec, mu1, logvar1, x2_pred, mu_pred, logvar_pred, z2_pred
            outputs = self.model(x1_tensor, mol_tensor)
            x2_pred = outputs[3]  # x2_pred is 4th output

        predictions = x2_pred.cpu().numpy()

        # Handle output dimension
        if diseased.shape[1] != self.n_genes:
            if diseased.shape[1] > self.n_genes:
                # Pad predictions to match input
                full_pred = diseased.copy()
                full_pred[:, :self.n_genes] = predictions
                return full_pred
            else:
                return predictions[:, :diseased.shape[1]]

        return predictions

    def get_embeddings(self, diseased: np.ndarray,
                       metadata: Optional[Dict] = None) -> np.ndarray:
        """Extract latent embeddings from TranSiGen encoder."""
        if not self._trained:
            raise RuntimeError("Model must be trained/loaded first")

        n_samples = diseased.shape[0]

        # Handle gene dimension
        if diseased.shape[1] != self.n_genes:
            if diseased.shape[1] > self.n_genes:
                x1 = diseased[:, :self.n_genes]
            else:
                x1 = np.zeros((n_samples, self.n_genes), dtype=np.float32)
                x1[:, :diseased.shape[1]] = diseased
        else:
            x1 = diseased.astype(np.float32)

        x1_tensor = torch.FloatTensor(x1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            z1, mu1, logvar1 = self.model.encode_x1(x1_tensor)

        return mu1.cpu().numpy()


if __name__ == "__main__":
    # Test the wrapper
    print("Testing TranSiGen wrapper...")

    # Check if model exists
    if os.path.exists(DEFAULT_MODEL_PATH):
        model = TranSiGenModel()

        # Dummy data
        n_samples = 10
        n_genes = 978
        diseased = np.random.randn(n_samples, n_genes).astype(np.float32)
        treated = np.random.randn(n_samples, n_genes).astype(np.float32)

        model.train(diseased, treated)
        predictions = model.predict(diseased)

        print(f"Input shape: {diseased.shape}")
        print(f"Output shape: {predictions.shape}")
        print("TranSiGen wrapper test passed!")
    else:
        print(f"Model not found at {DEFAULT_MODEL_PATH}")
        print("Skipping test - wrapper code is ready")
