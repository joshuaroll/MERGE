#!/usr/bin/env python
"""
MultiDCP wrapper for MERGE ensemble evaluation.
MultiDCP uses neural fingerprints for drug encoding and multimodal fusion
for predicting gene expression changes.
"""
import os
import sys
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_model import BasePerturbationModel

# MultiDCP paths - now using local copy
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MULTIDCP_SRC = os.path.join(_BASE_DIR, "MultiDCP", "src")
MULTIDCP_MODELS = os.path.join(MULTIDCP_SRC, "models")
MULTIDCP_UTILS = os.path.join(MULTIDCP_SRC, "utils")
# Checkpoint still from trained model location
MULTIDCP_CHECKPOINT = "/raid/home/joshua/projects/MultiDCP_pdg/src/multidcp_baseline_de"
# Drug name to SMILES mapping
DRUG_SMILES_CSV = "/raid/home/joshua/projects/MultiDCP_pdg/data/all_drugs_pdg.csv"


class MultiDCPModel(BasePerturbationModel):
    """
    MultiDCP model wrapper implementing BasePerturbationModel interface.

    MultiDCP uses:
    1. Neural fingerprints for drug encoding
    2. Linear/Transformer encoder for cell gene expression
    3. Dose embedding
    4. MLP layers for final prediction
    """

    def __init__(self, checkpoint_path: str = MULTIDCP_CHECKPOINT, device: str = "cuda"):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.device_str = device
        self.model = None
        self.device = None
        self._data_utils = None
        self._model_params = None
        self._drug_smiles_map = None

    def _load_drug_smiles_map(self):
        """Load drug name to SMILES mapping from CSV."""
        import pandas as pd
        if os.path.exists(DRUG_SMILES_CSV):
            df = pd.read_csv(DRUG_SMILES_CSV)
            self._drug_smiles_map = dict(zip(df['drug_name'].str.lower(), df['cpd_smiles']))
            print(f"Loaded {len(self._drug_smiles_map)} drug-to-SMILES mappings")
        else:
            self._drug_smiles_map = {}
            print(f"Warning: Drug SMILES mapping not found at {DRUG_SMILES_CSV}")

    def _convert_drug_names_to_smiles(self, drug_names):
        """Convert drug names to SMILES using the mapping."""
        if self._drug_smiles_map is None:
            self._load_drug_smiles_map()

        smiles_list = []
        for name in drug_names:
            name_lower = str(name).lower()
            if name_lower in self._drug_smiles_map:
                smiles_list.append(self._drug_smiles_map[name_lower])
            else:
                # Use a default SMILES for unknown drugs (simple molecule)
                smiles_list.append("C")  # Methane as placeholder
        return smiles_list

    @property
    def name(self) -> str:
        return "MultiDCP"

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension (hid_dim from model params)."""
        return 128  # hid_dim from model parameters

    def _setup_paths(self):
        """Add MultiDCP paths to sys.path."""
        for path in [MULTIDCP_SRC, MULTIDCP_MODELS, MULTIDCP_UTILS]:
            if path not in sys.path:
                sys.path.insert(0, path)

    def _load_model(self):
        """Load pretrained MultiDCP model."""
        import torch

        self._setup_paths()
        import multidcp_pdg as multidcp

        self.device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")

        # Model parameters (matching the trained model)
        self._model_params = {
            'drug_input_dim': {'atom': 62, 'bond': 6},
            'drug_emb_dim': 128,
            'conv_size': [16, 16],
            'degree': [0, 1, 2, 3, 4, 5],
            'gene_input_dim': 128,
            'gene_emb_dim': 128,
            'cell_id_input_dim': 10716,
            'cell_id_emb_dim': 50,
            'hid_dim': 128,
            'num_gene': 10716,
            'loss_type': 'point_wise_mse',
            'initializer': torch.nn.init.xavier_uniform_,
            'pert_idose_input_dim': 2,  # Matches checkpoint
            'pert_idose_emb_dim': 4,
            'dropout': 0.3,
            'linear_encoder_flag': True,
            'cell_decoder_dim': 10716
        }

        # Create model
        self.model = multidcp.MultiDCP_AE(self.device, self._model_params)

        # Load weights
        if os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            # Convert all float64 tensors to float32
            state_dict = {k: v.float() if v.dtype == torch.float64 else v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            print(f"Loaded MultiDCP checkpoint from {self.checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """
        For MultiDCP, we use pretrained weights rather than training from scratch.
        This method loads the pretrained model.

        Args:
            diseased: Diseased expression array (n_samples, n_genes) - not used
            treated: Treated expression array (n_samples, n_genes) - not used
            metadata: Optional metadata - not used
        """
        self._load_model()
        self._trained = True
        print("MultiDCP: Using pretrained model (no additional training)")

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Predict treated expression from diseased expression.

        Args:
            diseased: Diseased expression (n_samples, n_genes)
            metadata: Dict with required keys:
                - 'smiles': List of SMILES strings for each sample
                - 'dose': List of dose values (or one-hot encoded)

        Returns:
            Predicted treated expression (n_samples, n_genes)
        """
        if not self._trained:
            raise RuntimeError("Model must be trained/loaded first")

        if metadata is None:
            raise ValueError("MultiDCP requires metadata with 'smiles' and 'dose' keys")

        import torch
        self._setup_paths()
        import data_utils_pdg as data_utils

        n_samples = diseased.shape[0]
        smiles_list = metadata.get('smiles', None)
        drug_names = metadata.get('drug_names', metadata.get('pert_id', None))
        dose_list = metadata.get('dose', None)

        # Convert drug names to SMILES if needed
        if smiles_list is None and drug_names is not None:
            smiles_list = self._convert_drug_names_to_smiles(drug_names)
        elif smiles_list is not None:
            # Check if smiles_list contains drug names instead of actual SMILES
            # (SMILES typically contain 'C', 'N', 'O', '(' etc.)
            if not any(c in str(smiles_list[0]) for c in ['C', 'N', 'O', '(', ')', '=']):
                smiles_list = self._convert_drug_names_to_smiles(smiles_list)

        if smiles_list is None:
            raise ValueError("metadata must contain 'smiles', 'drug_names', or 'pert_id' key")

        # Default dose if not provided
        if dose_list is None:
            dose_list = ['x'] * n_samples

        # Process in batches
        batch_size = 32
        all_predictions = []

        # Set default dtype to float32 to avoid dtype mismatches
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_size_actual = end_idx - start_idx

                # Prepare batch inputs
                batch_diseased = diseased[start_idx:end_idx]
                batch_smiles = smiles_list[start_idx:end_idx]
                batch_dose = dose_list[start_idx:end_idx]

                # Convert to tensors
                input_cell_gex = torch.FloatTensor(batch_diseased).to(self.device)

                # Convert SMILES to drug features (already float32 after data_utils fix)
                drug_features = data_utils.convert_smile_to_feature(batch_smiles, self.device)
                mask = data_utils.create_mask_feature(drug_features, self.device)

                # Prepare dose (always use 2-dim encoding to match checkpoint)
                dose_tensor = self._encode_dose(batch_dose)

                # Gene features (placeholder - not used in current architecture)
                gene_features = None

                # Forward pass
                predictions, _ = self.model(
                    input_cell_gex=input_cell_gex,
                    input_drug=drug_features,
                    input_gene=gene_features,
                    mask=mask,
                    input_pert_idose=dose_tensor,
                    job_id='perturbed'
                )

                all_predictions.append(predictions.cpu().numpy())

        # Restore original dtype
        torch.set_default_dtype(original_dtype)

        return np.vstack(all_predictions)

    def get_embeddings(self, diseased: np.ndarray,
                       metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Extract embeddings from MultiDCP's intermediate representations.

        Uses the gene-level hidden state aggregated (summed) across genes,
        similar to how MultiDCPPretraining works.

        Args:
            diseased: Diseased expression (n_samples, n_genes)
            metadata: Dict with 'smiles'/'drug_names' and 'dose'

        Returns:
            Embeddings array (n_samples, hid_dim=128)
        """
        if not self._trained:
            raise RuntimeError("Model must be trained/loaded first")

        if metadata is None:
            raise ValueError("MultiDCP requires metadata with 'smiles' and 'dose' keys")

        import torch
        self._setup_paths()
        import data_utils_pdg as data_utils

        n_samples = diseased.shape[0]
        smiles_list = metadata.get('smiles', None)
        drug_names = metadata.get('drug_names', metadata.get('pert_id', None))
        dose_list = metadata.get('dose', None)

        # Convert drug names to SMILES if needed
        if smiles_list is None and drug_names is not None:
            smiles_list = self._convert_drug_names_to_smiles(drug_names)
        elif smiles_list is not None:
            if not any(c in str(smiles_list[0]) for c in ['C', 'N', 'O', '(', ')', '=']):
                smiles_list = self._convert_drug_names_to_smiles(smiles_list)

        if smiles_list is None:
            raise ValueError("metadata must contain 'smiles', 'drug_names', or 'pert_id' key")

        if dose_list is None:
            dose_list = ['x'] * n_samples

        batch_size = 32
        all_embeddings = []

        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)

                batch_diseased = diseased[start_idx:end_idx]
                batch_smiles = smiles_list[start_idx:end_idx]
                batch_dose = dose_list[start_idx:end_idx]

                input_cell_gex = torch.FloatTensor(batch_diseased).to(self.device)
                drug_features = data_utils.convert_smile_to_feature(batch_smiles, self.device)
                mask = data_utils.create_mask_feature(drug_features, self.device)
                dose_tensor = self._encode_dose(batch_dose)
                gene_features = None

                # Get gene-level hidden states from the internal multidcp module
                # The model.model is MultiDCP_AE which contains self.multidcp
                out, cell_hidden = self.model.multidcp(
                    input_drug=drug_features,
                    input_gene=gene_features,
                    mask=mask,
                    input_cell_gex=input_cell_gex,
                    input_pert_idose=dose_tensor,
                )
                # out = [batch * num_gene * hid_dim]

                # Aggregate across genes (sum, like MultiDCPPretraining)
                embedding = torch.sum(out, dim=1)  # [batch * hid_dim]
                all_embeddings.append(embedding.cpu().numpy())

        torch.set_default_dtype(original_dtype)

        return np.vstack(all_embeddings)

    def _encode_dose(self, dose_list):
        """Encode dose values as 2-dimensional vector (matching checkpoint)."""
        import torch

        n_samples = len(dose_list)
        # Checkpoint was trained with 2-dim dose input
        # Use default encoding: [1, 0] for all samples since data has uniform dose
        dose_encoded = np.zeros((n_samples, 2), dtype=np.float32)
        dose_encoded[:, 0] = 1.0  # Default to first dose level

        return torch.FloatTensor(dose_encoded).to(self.device)


def test_multidcp_with_dataloader(cell_line: str = "A549", fold: int = 0, n_samples: int = 1000):
    """
    Test MultiDCP using the PDGrapherDataLoader.
    """
    print(f"\n=== Testing MultiDCP on {cell_line} (fold {fold}) ===")

    # Import data loader
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_loader import PDGrapherDataLoader, TopKEvaluator

    # Load data
    loader = PDGrapherDataLoader()
    train_idx, test_idx = loader.get_train_test_split(cell_line, fold=fold)

    # Limit samples
    if len(test_idx) > n_samples:
        test_idx = test_idx[:n_samples]

    print(f"Test samples: {len(test_idx)}")

    # Get expression arrays
    treated_test, diseased_test = loader.get_expression_arrays(test_idx)

    # Get metadata (drug names and dose)
    test_df = loader.treated_df.iloc[test_idx]
    drug_names = test_df['pert_id'].tolist() if 'pert_id' in test_df.columns else None
    dose_list = test_df['pert_idose'].tolist() if 'pert_idose' in test_df.columns else None

    # Create and load model
    model = MultiDCPModel()
    model.train(diseased_test, treated_test)  # This loads pretrained weights

    # Predict
    print("Generating predictions...")
    metadata = {'drug_names': drug_names, 'dose': dose_list}
    pred_treated = model.predict(diseased_test, metadata)

    # Evaluate
    evaluator = TopKEvaluator()
    results = evaluator.compute_metrics(treated_test, pred_treated, diseased_test)
    evaluator.print_results(results, f"MultiDCP ({cell_line})")

    return results


if __name__ == "__main__":
    # Test MultiDCP
    results = test_multidcp_with_dataloader(cell_line="A549", fold=0, n_samples=500)
