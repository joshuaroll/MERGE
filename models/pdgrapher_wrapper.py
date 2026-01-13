#!/usr/bin/env python
"""
PDGrapher wrapper for baseline evaluation.
Trains PDGrapher from scratch and evaluates on the same metrics as other baselines.
"""
import os
import sys
import numpy as np
import torch
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Add PDGrapher to path
sys.path.insert(0, '/raid/home/public/chemoe_collab_102025/PDGrapher/src')

from pdgrapher import PDGrapher, Dataset, Trainer


def train_pdgrapher(cell_line: str = "A549",
                    n_layers_gnn: int = 2,
                    n_layers_nn: int = 1,
                    n_epochs: int = 50,
                    fold: int = 1,
                    output_dir: str = None):
    """
    Train PDGrapher on a specific cell line.

    Args:
        cell_line: Cell line to train on
        n_layers_gnn: Number of GCN layers
        n_layers_nn: Number of MLP layers
        n_epochs: Number of training epochs
        fold: Which fold to train (1-5)
        output_dir: Directory to save results
    """
    print(f"\n{'='*70}")
    print(f"Training PDGrapher on {cell_line} (fold {fold})")
    print(f"Config: n_layers_gnn={n_layers_gnn}, n_layers_nn={n_layers_nn}, n_epochs={n_epochs}")
    print(f"{'='*70}")

    # Data paths - use fresh data from Zenodo download
    data_base = "/raid/home/joshua/projects/PDFGraph/torch_data_chemical/chemical/real_lognorm"
    splits_base = "/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed"
    forward_path = f"{data_base}/data_forward_{cell_line}.pt"
    backward_path = f"{data_base}/data_backward_{cell_line}.pt"
    edge_index_path = f"{data_base}/edge_index_{cell_line}.pt"
    splits_path = f"{splits_base}/splits/chemical/{cell_line}/random/5fold/splits.pt"

    # Check if files exist
    for path in [forward_path, backward_path, edge_index_path, splits_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

    # Load dataset
    print("Loading dataset...")
    dataset = Dataset(
        forward_path=forward_path,
        backward_path=backward_path,
        splits_path=splits_path,
        test_indices=True
    )

    # Load edge index (PPI network)
    edge_index = torch.load(edge_index_path)
    print(f"Loaded edge index with {edge_index.shape[1]} edges")

    # Get number of genes
    num_vars = dataset.get_num_vars()
    print(f"Number of genes: {num_vars}")

    # Initialize model
    print("Initializing PDGrapher model...")
    model = PDGrapher(
        edge_index,
        model_kwargs={
            "n_layers_nn": n_layers_nn,
            "n_layers_gnn": n_layers_gnn,
            "num_vars": num_vars,
            "positional_features_dims": 16,
            "embedding_layer_dim": 16,
            "dim_gnn": 16,
        },
        response_kwargs={'train': True},
        perturbation_kwargs={'train': True}
    )

    # Setup output directory
    if output_dir is None:
        output_dir = f"/raid/home/joshua/projects/PDGrapher_Baseline_Models/output/pdgrapher_{cell_line}"
    os.makedirs(output_dir, exist_ok=True)

    # Setup trainer
    print("Setting up trainer...")
    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda", "devices": 1},
        log=True,
        logging_dir=output_dir,
        use_forward_data=True,
        use_backward_data=True,
        use_intervention_data=True,
        use_supervision=True,
        supervision_multiplier=0.01,
        log_train=False,  # Disable train logging for speed
        log_test=True,
    )

    # Train single fold
    print(f"\nTraining fold {fold}...")
    dataset.prepare_fold(fold)

    results = trainer.train(
        model,
        dataset,
        n_epochs=n_epochs,
        early_stopping_kwargs={"patience": 15, "skip": 0}
    )

    print(f"\nTraining complete!")
    print(f"Results saved to: {output_dir}")

    return results, model, dataset


def evaluate_pdgrapher_on_test(model, dataset, trainer):
    """
    Evaluate trained PDGrapher on test set and compute metrics.
    """
    print("\nEvaluating on test set...")

    # Get test metrics from trainer
    test_results = trainer._test_one_pass(model, dataset)

    return test_results


def train_and_evaluate_pdgrapher(cell_line: str = "A549", fold: int = 1, n_epochs: int = 50):
    """
    Train PDGrapher and evaluate with same metrics as other baselines.
    """
    # Train
    results, model, dataset = train_pdgrapher(
        cell_line=cell_line,
        n_layers_gnn=2,
        n_layers_nn=1,
        n_epochs=n_epochs,
        fold=fold
    )

    # Print results
    print("\n" + "="*70)
    print(f"PDGrapher Results ({cell_line}, fold {fold})")
    print("="*70)

    if results:
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PDGrapher")
    parser.add_argument("--cell_line", default="A549", help="Cell line to train on")
    parser.add_argument("--fold", type=int, default=1, help="Fold to train (1-5)")
    parser.add_argument("--n_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--n_layers_gnn", type=int, default=2, help="Number of GCN layers")
    parser.add_argument("--n_layers_nn", type=int, default=1, help="Number of MLP layers")

    args = parser.parse_args()

    results = train_and_evaluate_pdgrapher(
        cell_line=args.cell_line,
        fold=args.fold,
        n_epochs=args.n_epochs
    )
