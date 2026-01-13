#!/usr/bin/env python
"""
ChemCPA wrapper for PDGrapher baseline evaluation.
This creates a simplified ChemCPA training pipeline using our PDGrapher data.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ChemCPA'))

from data_loader import PDGrapherDataLoader, TopKEvaluator
from models.base_model import BasePerturbationModel


class SimpleChemCPADataset(Dataset):
    """
    Simple dataset for ChemCPA-style training.
    """
    def __init__(self, diseased: np.ndarray, treated: np.ndarray,
                 drug_indices: np.ndarray, dosages: np.ndarray,
                 cell_type_indices: np.ndarray):
        self.diseased = torch.FloatTensor(diseased)
        self.treated = torch.FloatTensor(treated)
        self.drug_indices = torch.LongTensor(drug_indices)
        self.dosages = torch.FloatTensor(dosages)
        self.cell_type_indices = torch.LongTensor(cell_type_indices)

    def __len__(self):
        return len(self.diseased)

    def __getitem__(self, idx):
        return (
            self.diseased[idx],
            self.treated[idx],
            self.drug_indices[idx],
            self.dosages[idx],
            self.cell_type_indices[idx]
        )


class DrugEncoder(nn.Module):
    """Simple drug embedding + MLP encoder."""
    def __init__(self, num_drugs: int, drug_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.drug_embedding = nn.Embedding(num_drugs, drug_dim)
        self.dose_encoder = nn.Linear(1, 32)
        self.mlp = nn.Sequential(
            nn.Linear(drug_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, drug_idx, dosage):
        drug_emb = self.drug_embedding(drug_idx)
        dose_emb = self.dose_encoder(dosage.unsqueeze(-1))
        combined = torch.cat([drug_emb, dose_emb], dim=-1)
        return self.mlp(combined)


class GeneEncoder(nn.Module):
    """Encoder for gene expression."""
    def __init__(self, num_genes: int, latent_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class GeneDecoder(nn.Module):
    """Decoder for gene expression."""
    def __init__(self, latent_dim: int, num_genes: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_genes),
        )

    def forward(self, x):
        return self.decoder(x)


class SimpleChemCPA(nn.Module):
    """
    Simplified ChemCPA model.

    Architecture:
    1. Encode diseased expression to latent space
    2. Encode drug + dose to perturbation embedding
    3. Add perturbation to latent
    4. Decode to predicted treated expression
    """
    def __init__(self, num_genes: int, num_drugs: int, num_cell_types: int,
                 latent_dim: int = 256, drug_dim: int = 128):
        super().__init__()

        self.gene_encoder = GeneEncoder(num_genes, latent_dim)
        self.drug_encoder = DrugEncoder(num_drugs, drug_dim, latent_dim)
        self.cell_type_embedding = nn.Embedding(num_cell_types, 64)

        # Combine latent + drug + cell type
        self.combiner = nn.Sequential(
            nn.Linear(latent_dim + latent_dim + 64, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self.gene_decoder = GeneDecoder(latent_dim, num_genes)

    def forward(self, diseased, drug_idx, dosage, cell_type_idx):
        # Encode diseased expression
        latent = self.gene_encoder(diseased)

        # Encode drug perturbation
        drug_latent = self.drug_encoder(drug_idx, dosage)

        # Cell type embedding
        cell_emb = self.cell_type_embedding(cell_type_idx)

        # Combine
        combined = torch.cat([latent, drug_latent, cell_emb], dim=-1)
        perturbed_latent = self.combiner(combined)

        # Decode
        pred_treated = self.gene_decoder(perturbed_latent)

        return pred_treated

    def predict(self, diseased, drug_idx, dosage, cell_type_idx):
        """Predict treated expression."""
        self.eval()
        with torch.no_grad():
            return self.forward(diseased, drug_idx, dosage, cell_type_idx)


class ChemCPAModel(BasePerturbationModel):
    """
    ChemCPA-style model implementing BasePerturbationModel interface.
    """

    def __init__(self, latent_dim: int = 256, n_epochs: int = 50, batch_size: int = 64,
                 learning_rate: float = 1e-3, device: str = 'cuda'):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.model = None
        self._drug_to_idx = None
        self._cell_type_to_idx = None
        self._num_genes = None

    @property
    def name(self) -> str:
        return "ChemCPA"

    def train(self, diseased: np.ndarray, treated: np.ndarray,
              metadata: Optional[Dict] = None) -> None:
        """
        Train ChemCPA model.

        Args:
            diseased: Diseased expression (n_samples, n_genes)
            treated: Treated expression (n_samples, n_genes)
            metadata: Dict with 'smiles', 'dose', 'cell_type' arrays
        """
        n_samples, n_genes = diseased.shape
        self._num_genes = n_genes

        # Process metadata
        if metadata is None:
            metadata = {}

        # Create drug indices
        smiles = metadata.get('smiles')
        if smiles is None:
            smiles = ['drug_0'] * n_samples
        unique_drugs = list(set(smiles))
        self._drug_to_idx = {d: i for i, d in enumerate(unique_drugs)}
        drug_indices = np.array([self._drug_to_idx[s] for s in smiles])

        # Create cell type indices
        cell_types = metadata.get('cell_type')
        if cell_types is None:
            cell_types = ['cell_0'] * n_samples
        unique_cell_types = list(set(cell_types))
        self._cell_type_to_idx = {c: i for i, c in enumerate(unique_cell_types)}
        cell_type_indices = np.array([self._cell_type_to_idx[c] for c in cell_types])

        # Doses
        doses = metadata.get('dose')
        if doses is None:
            doses = np.ones(n_samples)
        if isinstance(doses, list):
            doses = np.array(doses)

        # Create dataset and dataloader
        dataset = SimpleChemCPADataset(
            diseased, treated, drug_indices, doses.astype(np.float32), cell_type_indices
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Create model
        self.model = SimpleChemCPA(
            num_genes=n_genes,
            num_drugs=len(unique_drugs),
            num_cell_types=len(unique_cell_types),
            latent_dim=self.latent_dim,
        ).to(self.device)

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch in dataloader:
                diseased_b, treated_b, drug_idx, dosage, cell_type_idx = batch
                diseased_b = diseased_b.to(self.device)
                treated_b = treated_b.to(self.device)
                drug_idx = drug_idx.to(self.device)
                dosage = dosage.to(self.device)
                cell_type_idx = cell_type_idx.to(self.device)

                optimizer.zero_grad()
                pred = self.model(diseased_b, drug_idx, dosage, cell_type_idx)
                loss = criterion(pred, treated_b)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.n_epochs}, Loss: {total_loss/len(dataloader):.4f}")

        self._trained = True

    def predict(self, diseased: np.ndarray,
                metadata: Optional[Dict] = None) -> np.ndarray:
        """
        Predict treated expression.

        Args:
            diseased: Diseased expression (n_samples, n_genes)
            metadata: Dict with 'smiles', 'dose', 'cell_type'

        Returns:
            Predicted treated expression (n_samples, n_genes)
        """
        if not self._trained:
            raise RuntimeError("Model must be trained first")

        n_samples = len(diseased)

        if metadata is None:
            metadata = {}

        # Get indices using saved mappings
        smiles = metadata.get('smiles')
        if smiles is None:
            smiles = ['drug_0'] * n_samples
        drug_indices = np.array([
            self._drug_to_idx.get(s, 0) for s in smiles
        ])

        cell_types = metadata.get('cell_type')
        if cell_types is None:
            cell_types = ['cell_0'] * n_samples
        cell_type_indices = np.array([
            self._cell_type_to_idx.get(c, 0) for c in cell_types
        ])

        doses = metadata.get('dose')
        if doses is None:
            doses = np.ones(n_samples)
        if isinstance(doses, list):
            doses = np.array(doses)

        # Convert to tensors
        diseased_t = torch.FloatTensor(diseased).to(self.device)
        drug_idx_t = torch.LongTensor(drug_indices).to(self.device)
        dosage_t = torch.FloatTensor(doses.astype(np.float32)).to(self.device)
        cell_type_idx_t = torch.LongTensor(cell_type_indices).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            pred = self.model(diseased_t, drug_idx_t, dosage_t, cell_type_idx_t)

        return pred.cpu().numpy()


def train_and_evaluate_chemcpa(cell_line: str = "A549", fold: int = 0, n_samples: int = 5000):
    """
    Train and evaluate ChemCPA on a specific cell line.
    """
    print(f"\n=== Training ChemCPA on {cell_line} (fold {fold}) ===")

    # Load data
    loader = PDGrapherDataLoader()
    train_idx, test_idx = loader.get_train_test_split(cell_line, fold=fold)

    # Limit samples
    if len(train_idx) > n_samples:
        train_idx = train_idx[:n_samples]
    if len(test_idx) > n_samples // 4:
        test_idx = test_idx[:n_samples // 4]

    print(f"Training samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # Get data
    treated_train, diseased_train = loader.get_expression_arrays(train_idx)
    treated_test, diseased_test = loader.get_expression_arrays(test_idx)
    meta_train = loader.get_metadata(train_idx)
    meta_test = loader.get_metadata(test_idx)

    # Prepare metadata
    train_metadata = {
        'smiles': meta_train['smiles'].tolist() if 'smiles' in meta_train.columns else None,
        'dose': meta_train['dose'].values if 'dose' in meta_train.columns else None,
        'cell_type': [cell_line] * len(train_idx),
    }
    test_metadata = {
        'smiles': meta_test['smiles'].tolist() if 'smiles' in meta_test.columns else None,
        'dose': meta_test['dose'].values if 'dose' in meta_test.columns else None,
        'cell_type': [cell_line] * len(test_idx),
    }

    # Train model
    model = ChemCPAModel(latent_dim=256, n_epochs=50, batch_size=64)
    model.train(diseased_train, treated_train, train_metadata)

    # Predict
    print("Generating predictions...")
    pred_treated = model.predict(diseased_test, test_metadata)

    # Evaluate
    evaluator = TopKEvaluator()
    results = evaluator.compute_metrics(treated_test, pred_treated, diseased_test)
    evaluator.print_results(results, f"ChemCPA ({cell_line})")

    return results, model


if __name__ == "__main__":
    results, model = train_and_evaluate_chemcpa(cell_line="A549", fold=0, n_samples=5000)
