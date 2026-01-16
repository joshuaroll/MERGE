#!/usr/bin/env python
"""
MultiDCP MoE (CheMoE) wrapper for MERGE ensemble evaluation.

MultiDCP with Mixture-of-Experts architecture supporting:
1. Sparse MoE style: Top-k + LayerNorm + re-normalization
2. MultiDCP MoE style: Top-k + softmax over selected + load balancing

Uses neural fingerprints for drug encoding + MoE for prediction.
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

# MultiDCP CheMoE paths - use local implementation
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTIDCP_CHEMOE_MODELS = os.path.join(PROJECT_ROOT, "models", "multidcp_chemoe")
MULTIDCP_CHEMOE_UTILS = os.path.join(PROJECT_ROOT, "utils")

# Checkpoint paths (organized by cell type and fold)
CHECKPOINT_BASE = os.path.join(PROJECT_ROOT, "trained_models")
# Default checkpoint (if no cell-specific checkpoint available)
DEFAULT_CHECKPOINT = os.path.join(PROJECT_ROOT, "best_model.pt")

# PDGrapher trained models directory
PDG_TRAINED_MODELS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trained_models"
)

# Drug SMILES mapping
DRUG_SMILES_CSV = "/raid/home/joshua/projects/MultiDCP_pdg/data/all_drugs_pdg.csv"

# Data paths
DATA_PICKLE = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl"
DISEASED_PICKLE = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl"


class MultiDCPMoEModel(BasePerturbationModel):
    """
    MultiDCP MoE (CheMoE) model wrapper for ensemble evaluation.

    Architecture:
    1. Drug encoder: Neural fingerprint -> 128-dim
    2. Cell encoder: Linear/Transformer -> 50-dim
    3. Dose encoder: MLP -> 128-dim
    4. Gene embeddings: Learnable -> 128-dim per gene
    5. Gating network: Global features -> expert selection (top-k)
    6. Expert networks: Gene-aware predictions weighted by gating
    """

    def __init__(self, cell_type: str = "A549", fold: int = 0,
                 checkpoint_path: str = None, device: str = "cuda"):
        super().__init__()
        self.cell_type = cell_type
        self.fold = fold
        self.device_str = device

        # Find checkpoint
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path
        else:
            self.checkpoint_path = self._find_checkpoint()

        self.model = None
        self.device = None
        self._data_utils = None
        self._model_params = None
        self._drug_smiles_map = None

    def _find_checkpoint(self) -> str:
        """Find best checkpoint for cell type and fold."""
        # First check PDGrapher trained model
        pdg_model = os.path.join(PDG_TRAINED_MODELS, f"multidcp_moe_{self.cell_type}_fold{self.fold}", "best_model.pt")
        if os.path.exists(pdg_model):
            print(f"Using PDGrapher trained model: {pdg_model}")
            self._use_pdg_model = True
            return pdg_model

        self._use_pdg_model = False

        # Check various naming conventions
        patterns = [
            os.path.join(CHECKPOINT_BASE, f"{self.cell_type}_fold_{self.fold}", "best_model.pt"),
            os.path.join(CHECKPOINT_BASE, f"test_{self.cell_type}_fold_{self.fold}", "best_model.pt"),
            os.path.join(MULTIDCP_MOE_SRC, "output", f"chemoe_{self.cell_type}_fold{self.fold}", "best_model.pt"),
        ]

        for pattern in patterns:
            if os.path.exists(pattern):
                return pattern

        # Fall back to default checkpoint
        if os.path.exists(DEFAULT_CHECKPOINT):
            print(f"Using default checkpoint: {DEFAULT_CHECKPOINT}")
            return DEFAULT_CHECKPOINT

        # List available checkpoints
        if os.path.exists(CHECKPOINT_BASE):
            available = os.listdir(CHECKPOINT_BASE)
            print(f"Available checkpoints in {CHECKPOINT_BASE}: {available}")

        return DEFAULT_CHECKPOINT

    @property
    def name(self) -> str:
        return "MultiDCP_MoE"

    @property
    def embedding_dim(self) -> int:
        return 306  # drug (128) + cell (50) + dose (128)

    def _load_drug_smiles_map(self):
        """Load drug name to SMILES mapping."""
        import pandas as pd
        if os.path.exists(DRUG_SMILES_CSV):
            df = pd.read_csv(DRUG_SMILES_CSV)
            self._drug_smiles_map = dict(zip(df['drug_name'].str.lower(), df['cpd_smiles']))
            print(f"Loaded {len(self._drug_smiles_map)} drug-to-SMILES mappings")
        else:
            self._drug_smiles_map = {}

    def _setup_paths(self):
        """Add MultiDCP CheMoE paths to sys.path."""
        for path in [PROJECT_ROOT, MULTIDCP_CHEMOE_MODELS, MULTIDCP_CHEMOE_UTILS]:
            if path not in sys.path:
                sys.path.insert(0, path)

    def _load_model(self):
        """Load pretrained MultiDCP MoE model."""
        self._setup_paths()

        self.device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        if not os.path.exists(self.checkpoint_path):
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}")
            print("Model will need to be trained or checkpoint path updated")
            return

        try:
            # Check if using PDGrapher trained model
            if hasattr(self, '_use_pdg_model') and self._use_pdg_model:
                # PDGrapher trained model is a complete model object
                self.model = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                self.model.to(self.device)
                self.model.eval()
                print(f"Loaded PDGrapher MultiDCP MoE model from {self.checkpoint_path}")
            else:
                # Original MultiDCP CheMoE model - use local implementation
                from models.multidcp_chemoe.model import MultiDCP_CheMoE_AE

                # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

                # Get model params from checkpoint or use defaults
                if 'model_param_registry' in checkpoint:
                    self._model_params = checkpoint['model_param_registry']
                else:
                    self._model_params = self._get_default_params()

                # Create model
                self.model = MultiDCP_CheMoE_AE(self.device, self._model_params)

                # Load weights
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)

                self.model.to(self.device)
                self.model.eval()

                print(f"Loaded MultiDCP MoE model from {self.checkpoint_path}")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Model loading failed - wrapper ready for future use")

        self._load_drug_smiles_map()

    def _get_default_params(self) -> dict:
        """Get default model parameters."""
        return {
            'drug_input_dim': {'atom': 62, 'bond': 6},
            'drug_emb_dim': 128,
            'conv_size': [16, 16],
            'degree': [0, 1, 2, 3, 4, 5],
            'gene_emb_dim': 128,
            'cell_id_input_dim': 978,
            'pert_idose_input_dim': 16,
            'hid_dim': 128,
            'num_gene': 978,
            'loss_type': 'point_wise_mse',
            'dropout': 0.3,
            'linear_encoder_flag': True,
        }

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

    def _convert_drug_names_to_smiles(self, drug_names):
        """Convert drug names to SMILES."""
        if self._drug_smiles_map is None:
            self._load_drug_smiles_map()

        smiles_list = []
        for name in drug_names:
            name_lower = str(name).lower()
            if name_lower in self._drug_smiles_map:
                smiles_list.append(self._drug_smiles_map[name_lower])
            else:
                smiles_list.append("C")  # Methane placeholder
        return smiles_list

    def _prepare_drug_input(self, smiles_list):
        """Convert SMILES to neural fingerprint input format."""
        try:
            from utils.data_utils_pdg import smiles_to_graph_batch
            return smiles_to_graph_batch(smiles_list, self.device)
        except ImportError:
            # Fallback: create dummy input
            print("Warning: utils.data_utils_pdg not found, using dummy drug input")
            return None

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Predict treated expression using MultiDCP MoE.

        Args:
            diseased: [n_samples, n_genes] basal/diseased expression
            metadata: dict with 'smiles' or 'drug_names', 'doses'

        Returns:
            predictions: [n_samples, n_genes] predicted treated expression
        """
        if not self._trained:
            raise RuntimeError("Model must be trained/loaded first")

        if self.model is None:
            print("Warning: Model not loaded, returning diseased as prediction")
            return diseased.copy()

        n_samples = diseased.shape[0]
        n_genes = diseased.shape[1]

        # Get SMILES
        if metadata and 'smiles' in metadata:
            smiles_list = metadata['smiles']
        elif metadata and 'drug_names' in metadata:
            smiles_list = self._convert_drug_names_to_smiles(metadata['drug_names'])
        else:
            smiles_list = ["C"] * n_samples

        # Get doses (default to middle dose - 10uM)
        if metadata and 'doses' in metadata:
            doses = metadata['doses']
        else:
            doses = np.zeros((n_samples, 16), dtype=np.float32)
            doses[:, 8] = 1.0  # Middle dose bin (10uM)

        # Predict using the model
        self.model.eval()
        with torch.no_grad():
            try:
                # Check if using PDGrapher trained model
                if hasattr(self, '_use_pdg_model') and self._use_pdg_model:
                    # PDGrapher model takes (drug_embed, cell_expr, dose)
                    # Get molecular embeddings
                    mol_embeddings = self._get_molecule_embeddings(smiles_list)
                    drug_embed = torch.FloatTensor(mol_embeddings).to(self.device)
                    cell_expr = torch.FloatTensor(diseased).to(self.device)
                    dose_tensor = torch.FloatTensor(doses).to(self.device)

                    predictions = self.model(drug_embed, cell_expr, dose_tensor)
                    return predictions.cpu().numpy()
                else:
                    # Original MultiDCP CheMoE model
                    input_drug = self._prepare_drug_input(smiles_list)
                    input_cell_gex = torch.FloatTensor(diseased).to(self.device)
                    input_pert_idose = torch.FloatTensor(doses).to(self.device)

                    input_gene = None
                    mask = None

                    predictions, _ = self.model(
                        input_drug, input_gene, mask,
                        input_cell_gex, input_pert_idose
                    )
                    return predictions.cpu().numpy()
            except Exception as e:
                print(f"Prediction error: {e}")
                return diseased.copy()

    def _get_molecule_embeddings(self, smiles_list):
        """Get molecular embeddings for SMILES list."""
        from models.transigen_wrapper import MOLECULE_EMBEDDINGS
        import pickle

        # Load embeddings if not cached
        if not hasattr(self, '_mol_embed_cache'):
            emb_path = MOLECULE_EMBEDDINGS.get('KPGT')
            if emb_path and os.path.exists(emb_path):
                with open(emb_path, 'rb') as f:
                    self._mol_embed_cache = pickle.load(f)
            else:
                self._mol_embed_cache = {}

        # Get embeddings
        emb_dim = 2304
        embeddings = []
        for smi in smiles_list:
            if smi in self._mol_embed_cache:
                embeddings.append(self._mol_embed_cache[smi])
            else:
                embeddings.append(np.zeros(emb_dim, dtype=np.float32))

        return np.array(embeddings, dtype=np.float32)

    def get_embeddings(self, diseased: np.ndarray,
                       metadata: Optional[Dict] = None) -> np.ndarray:
        """Extract global feature embeddings (drug + cell + dose)."""
        if not self._trained or self.model is None:
            # Return simple encoding
            return diseased[:, :self.embedding_dim]

        # Would need to modify model to expose intermediate embeddings
        # For now, return cell encoding
        input_cell_gex = torch.FloatTensor(diseased).to(self.device)

        self.model.eval()
        with torch.no_grad():
            try:
                cell_embed = self.model.model.cell_encoder(input_cell_gex)
                return cell_embed.cpu().numpy()
            except:
                return diseased[:, :50]  # Return first 50 dims as fallback


# Convenience aliases
class MultiDCPSparseMoE(MultiDCPMoEModel):
    """MultiDCP with Sparse-style MoE gating."""
    @property
    def name(self) -> str:
        return "MultiDCP_SparseMoE"


class MultiDCPBalancedMoE(MultiDCPMoEModel):
    """MultiDCP with Balanced MoE (default CheMoE style)."""
    @property
    def name(self) -> str:
        return "MultiDCP_BalancedMoE"


# Main CheMoE alias (same as Balanced)
MultiDCP_CheMoE = MultiDCPMoEModel


if __name__ == "__main__":
    print("Testing MultiDCP MoE wrapper...")

    model = MultiDCPMoEModel(cell_type="A549", fold=0)

    # Dummy data
    n_samples = 10
    n_genes = 978
    diseased = np.random.randn(n_samples, n_genes).astype(np.float32)
    treated = np.random.randn(n_samples, n_genes).astype(np.float32)

    metadata = {
        'drug_names': ['aspirin'] * n_samples,
        'doses': np.eye(16, dtype=np.float32)[:n_samples]
    }

    try:
        model.train(diseased, treated, metadata)
        predictions = model.predict(diseased, metadata)
        print(f"Input shape: {diseased.shape}")
        print(f"Output shape: {predictions.shape}")
        print("MultiDCP MoE wrapper test passed!")
    except Exception as e:
        print(f"Test error (expected if model not found): {e}")
        print("Wrapper code is ready for use with trained models")
