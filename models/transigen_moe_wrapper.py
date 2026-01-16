#!/usr/bin/env python
"""
TranSiGen + MoE wrapper for MERGE ensemble evaluation.

TranSiGen with Mixture-of-Experts fusion layer, supporting:
1. Sparse MoE: Top-k + LayerNorm + re-normalization
2. MultiDCP MoE: Top-k + softmax over selected + load balancing
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

# TranSiGen + MoE paths
TSG_MOE_BASE = "/raid/home/joshua/projects/cbio_032024/TSG_AI_122024"
TSG_MOE_SRC = os.path.join(TSG_MOE_BASE, "src", "models")
TSG_MOE_RESULTS = os.path.join(TSG_MOE_BASE, "src", "results")
TSG_MOE_PARENT = os.path.join(TSG_MOE_BASE, "src")

# PDGrapher trained models directory
PDG_TRAINED_MODELS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trained_models"
)

# Model paths for different MoE styles
MODEL_PATHS = {
    'sparse': {
        'KPGT': os.path.join(
            TSG_MOE_RESULTS,
            "TSG_s-2_ftsparse_moe_chemembKPGT_emb400_bat64_lr0.0001_beta0.1_betaalign0.1_do0.1_wd0.001_lamse",
            "trained_seed364039_feature_KPGT_init_pretrain_shRNA",
            "best_model_jer.pt"
        ),
    },
    'balanced': {
        'KPGT': os.path.join(
            TSG_MOE_RESULTS,
            "TSG_s-2_ftbalanced_moe_chemembKPGT_emb400_bat64_lr0.0001_beta0.1_betaalign0.1_do0.1_wd0.001_lamse",
            "trained_seed364039_feature_KPGT_init_pretrain_shRNA",
            "best_model_jer.pt"
        ),
        'UNIMOL': os.path.join(
            TSG_MOE_RESULTS,
            "TSG_s-2_ftbalanced_moe_chemembUNIMOL_emb400_bat64_lr0.0001_beta0.1_betaalign0.1_do0.1_wd0.001_lamse",
            "trained_seed364039_feature_UNIMOL_init_pretrain_shRNA",
            "best_model_jer.pt"
        ),
        'CHEMBERTA_846': os.path.join(
            TSG_MOE_RESULTS,
            "TSG_s-2_ftbalanced_moe_chemembCHEMBERTA_846_emb400_bat64_lr0.0001_beta0.1_betaalign0.1_do0.1_wd0.001_lamse",
            "trained_seed364039_feature_CHEMBERTA_846_init_pretrain_shRNA",
            "best_model_jer.pt"
        ),
    }
}

# Molecule embedding paths
MOLECULE_EMBEDDINGS = {
    'KPGT': "/raid/home/joshua/projects/cbio_032024/zenodo_TranSiGen/TranSiGen/data/LINCS2020/KPGT_emb2304.pickle",
    'ECFP4': "/raid/home/joshua/transigen_v1_release/TranSiGen-1.0/data/LINCS2020/data_example/ECFP4_emb2048.pickle",
    'CHEMBERTA_846': "/raid/home/joshua/transigen_v1_release/TranSiGen-1.0/data/LINCS2020/data_example/CHEMBERTA_emb768.pickle",
    'UNIMOL': "/raid/home/joshua/transigen_v1_release/TranSiGen-1.0/data/LINCS2020/data_example/UNIMOL_emb512.pickle",
}

# Drug SMILES mapping
DRUG_SMILES_CSV = "/raid/home/joshua/projects/MultiDCP_pdg/data/all_drugs_pdg.csv"


class TranSiGenMoEModel(BasePerturbationModel):
    """
    TranSiGen + MoE model wrapper for ensemble evaluation.

    Supports two MoE styles:
    - 'sparse': SparseMoE with top-k=1, LayerNorm, re-normalization
    - 'balanced': BalancedMoE with top-k=2, load balancing loss

    Architecture:
    1. Encoder for basal expression (x1) -> latent z1
    2. Molecular embedding from precomputed features
    3. MoE fusion: (z1, mol_embed) -> experts -> gated combination
    4. Prediction head -> predicted z2 -> decoded to x2
    """

    def __init__(self, moe_style: str = 'balanced', molecule_feature: str = 'KPGT',
                 model_path: str = None, device: str = "cuda", n_genes: int = 10716,
                 cell_line: str = "A549", fold: int = 0):
        super().__init__()

        if moe_style not in ['sparse', 'balanced']:
            raise ValueError(f"moe_style must be 'sparse' or 'balanced', got {moe_style}")

        self.moe_style = moe_style
        self.molecule_feature = molecule_feature
        self.device_str = device

        # Get model path - prefer PDGrapher trained model
        if model_path:
            self.model_path = model_path
            self.n_genes = n_genes
        else:
            # Check for PDGrapher trained model first
            pdg_model = os.path.join(PDG_TRAINED_MODELS, f"transigen_moe_{cell_line}_fold{fold}", "best_model.pt")
            if os.path.exists(pdg_model):
                self.model_path = pdg_model
                self.n_genes = 10716
            else:
                # Fallback to original LINCS model
                style_paths = MODEL_PATHS.get(moe_style, {})
                self.model_path = style_paths.get(molecule_feature)
                if not self.model_path:
                    self.model_path = MODEL_PATHS.get(moe_style, {}).get('KPGT')
                self.n_genes = 2432

        self.model = None
        self.device = None
        self._smiles_to_embedding = {}
        self._drug_smiles_map = None

    @property
    def name(self) -> str:
        return f"TranSiGen_{self.moe_style.capitalize()}MoE"

    @property
    def embedding_dim(self) -> int:
        return 100  # n_latent

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
            # Try alternative paths
            alt_paths = [
                f"/raid/home/joshua/TranSiGen/data/LINCS2020/{self.molecule_feature}_emb.pickle",
                f"/raid/home/joshua/projects/cbio_032024/TSG_AI_122024/data/{self.molecule_feature}_emb.pickle",
            ]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    with open(alt_path, 'rb') as f:
                        self._smiles_to_embedding = pickle.load(f)
                    print(f"Loaded molecule embeddings from {alt_path}")
                    return
            print(f"Warning: Molecule embeddings not found for {self.molecule_feature}")
            self._smiles_to_embedding = {}

    def _get_molecule_embedding(self, smiles: str) -> np.ndarray:
        """Get embedding for a SMILES string."""
        if smiles in self._smiles_to_embedding:
            emb = self._smiles_to_embedding[smiles]
            if isinstance(emb, np.ndarray):
                return emb
            return np.array(emb, dtype=np.float32)
        else:
            # Return zero embedding for unknown molecules
            emb_dims = {'KPGT': 2304, 'ECFP4': 2048, 'CHEMBERTA_846': 768, 'UNIMOL': 512}
            emb_dim = emb_dims.get(self.molecule_feature, 768)
            return np.zeros(emb_dim, dtype=np.float32)

    def _load_model(self):
        """Load pretrained TranSiGen + MoE model."""
        self.device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Add TranSiGen MoE source to path for model loading
        if TSG_MOE_SRC not in sys.path:
            sys.path.insert(0, TSG_MOE_SRC)
        if TSG_MOE_PARENT not in sys.path:
            sys.path.insert(0, TSG_MOE_PARENT)

        # Load model (weights_only=False for PyTorch 2.6+ compatibility with pickled models)
        self.model = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.dev = self.device
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded TranSiGen {self.moe_style} MoE model from {self.model_path}")

        # Load molecule embeddings
        self._load_molecule_embeddings()
        self._load_drug_smiles_map()

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """
        Load pretrained model (not retrained in ensemble).
        """
        if self.model is None:
            self._load_model()

        self._train_mean = np.mean(diseased, axis=0)
        self._train_std = np.std(diseased, axis=0) + 1e-8
        self._trained = True

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Predict treated expression using TranSiGen + MoE.

        Args:
            diseased: [n_samples, n_genes] basal expression
            metadata: dict with 'smiles' list for molecule embeddings

        Returns:
            predictions: [n_samples, n_genes] predicted treated expression
        """
        if not self._trained:
            raise RuntimeError("Model must be trained/loaded first")

        n_samples = diseased.shape[0]

        # Get molecule embeddings
        if metadata and 'smiles' in metadata:
            smiles_list = metadata['smiles']
        elif metadata and 'drug_names' in metadata:
            smiles_list = []
            for name in metadata['drug_names']:
                name_lower = str(name).lower()
                if self._drug_smiles_map and name_lower in self._drug_smiles_map:
                    smiles_list.append(self._drug_smiles_map[name_lower])
                else:
                    smiles_list.append("C")
        else:
            smiles_list = ["C"] * n_samples

        mol_embeddings = np.array([
            self._get_molecule_embedding(smi) for smi in smiles_list
        ], dtype=np.float32)

        # Handle gene dimension mismatch
        if diseased.shape[1] != self.n_genes:
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
            outputs = self.model(x1_tensor, mol_tensor)
            x2_pred = outputs[3]  # x2_pred

        predictions = x2_pred.cpu().numpy()

        # Handle output dimension
        if diseased.shape[1] != self.n_genes:
            if diseased.shape[1] > self.n_genes:
                full_pred = diseased.copy()
                full_pred[:, :self.n_genes] = predictions
                return full_pred
            else:
                return predictions[:, :diseased.shape[1]]

        return predictions

    def get_embeddings(self, diseased: np.ndarray,
                       metadata: Optional[Dict] = None) -> np.ndarray:
        """Extract latent embeddings."""
        if not self._trained:
            raise RuntimeError("Model must be trained/loaded first")

        n_samples = diseased.shape[0]

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


# Convenience classes for each MoE style
class TranSiGenSparseMoE(TranSiGenMoEModel):
    """TranSiGen with Sparse MoE (top-k=1)."""
    def __init__(self, molecule_feature: str = 'KPGT', **kwargs):
        super().__init__(moe_style='sparse', molecule_feature=molecule_feature, **kwargs)

    @property
    def name(self) -> str:
        return "TranSiGen_SparseMoE"


class TranSiGenBalancedMoE(TranSiGenMoEModel):
    """TranSiGen with Balanced MoE (top-k=2, load balancing)."""
    def __init__(self, molecule_feature: str = 'KPGT', **kwargs):
        super().__init__(moe_style='balanced', molecule_feature=molecule_feature, **kwargs)

    @property
    def name(self) -> str:
        return "TranSiGen_BalancedMoE"


# Alias for MultiDCP-style (balanced is closest)
TranSiGenMultiDCPMoE = TranSiGenBalancedMoE


if __name__ == "__main__":
    print("Testing TranSiGen MoE wrappers...")

    # Test balanced MoE
    balanced_path = MODEL_PATHS['balanced'].get('KPGT')
    if balanced_path and os.path.exists(balanced_path):
        print("\nTesting Balanced MoE...")
        model = TranSiGenBalancedMoE()

        n_samples, n_genes = 10, 978
        diseased = np.random.randn(n_samples, n_genes).astype(np.float32)
        treated = np.random.randn(n_samples, n_genes).astype(np.float32)

        model.train(diseased, treated)
        predictions = model.predict(diseased)
        print(f"  Input: {diseased.shape}, Output: {predictions.shape}")
        print("  Balanced MoE test passed!")
    else:
        print(f"Balanced MoE model not found at {balanced_path}")

    # Test sparse MoE
    sparse_path = MODEL_PATHS['sparse'].get('KPGT')
    if sparse_path and os.path.exists(sparse_path):
        print("\nTesting Sparse MoE...")
        model = TranSiGenSparseMoE()

        model.train(diseased, treated)
        predictions = model.predict(diseased)
        print(f"  Input: {diseased.shape}, Output: {predictions.shape}")
        print("  Sparse MoE test passed!")
    else:
        print(f"Sparse MoE model not found at {sparse_path}")

    print("\nWrapper code ready!")
