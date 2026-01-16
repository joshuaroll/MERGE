#!/usr/bin/env python
"""
Training script for PDGrapher on PDGrapher data.

PDGrapher uses a Graph Neural Network to model gene regulatory networks for
perturbation response prediction.

Note: This script requires the pdgrapher environment with torch-geometric:
    PYTHONPATH=/raid/home/public/chemoe_collab_102025/PDGrapher/src \
    python train_pdgrapher_pdg.py --cell_line A549 --fold 1 --gpu 0

Usage:
    python train_pdgrapher_pdg.py --cell_line A549 --fold 1 --gpu 0
"""
import os
import sys
import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import wandb
import warnings
warnings.filterwarnings('ignore')

# Add PDGrapher to path
sys.path.insert(0, '/raid/home/public/chemoe_collab_102025/PDGrapher/src')

from pdgrapher import PDGrapher, Dataset, Trainer

# Data paths
DATA_BASE = "/raid/home/joshua/projects/PDFGraph/torch_data_chemical/chemical/real_lognorm"
SPLITS_BASE = "/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed"


def compute_de_metrics(true_treated, pred_treated, diseased):
    """
    Compute evaluation metrics on differential expression.

    Args:
        true_treated: True treated expression (n_samples, n_genes)
        pred_treated: Predicted treated expression (n_samples, n_genes)
        diseased: Diseased/control expression (n_samples, n_genes)

    Returns:
        Dictionary of metrics
    """
    # Compute differential expression
    true_de = true_treated - diseased
    pred_de = pred_treated - diseased

    results = {}

    # Per-sample top-k R² scores
    for k in [20, 40, 80]:
        r2_scores = []
        for i in range(len(true_de)):
            de_mag = np.abs(true_de[i])
            top_k_idx = np.argsort(de_mag)[-k:]
            try:
                r2 = r2_score(true_de[i, top_k_idx], pred_de[i, top_k_idx])
                if not np.isnan(r2) and not np.isinf(r2):
                    r2_scores.append(r2)
            except:
                pass
        results[f'r2_top{k}'] = np.mean(r2_scores) if r2_scores else 0.0

    # Per-sample Pearson correlation on DE
    pearson_scores = []
    for i in range(len(true_de)):
        try:
            r, _ = pearsonr(true_de[i], pred_de[i])
            if not np.isnan(r):
                pearson_scores.append(r)
        except:
            pass
    results['pearson_de'] = np.mean(pearson_scores) if pearson_scores else 0.0

    # Per-sample Spearman correlation on DE
    spearman_scores = []
    for i in range(len(true_de)):
        try:
            r, _ = spearmanr(true_de[i], pred_de[i])
            if not np.isnan(r):
                spearman_scores.append(r)
        except:
            pass
    results['spearman_de'] = np.mean(spearman_scores) if spearman_scores else 0.0

    return results


class WandBCallback:
    """Simple callback for WandB logging during PDGrapher training."""

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, trainer_state):
        """Log metrics at end of epoch."""
        self.epoch += 1

        metrics = {
            'epoch': self.epoch,
        }

        # Extract available metrics from trainer state
        if hasattr(trainer_state, 'logs') and trainer_state.logs:
            for key, value in trainer_state.logs.items():
                if isinstance(value, (int, float)):
                    metrics[key] = value

        wandb.log(metrics)


def train_pdgrapher(cell_line, fold, args):
    """
    Train PDGrapher on a specific cell line and fold.

    Args:
        cell_line: Cell line to train on
        fold: Fold number (1-indexed for PDGrapher)
        args: Training arguments

    Returns:
        Trained model, dataset, trainer
    """
    print(f"Loading data for {cell_line}...")

    # Data paths
    forward_path = f"{DATA_BASE}/data_forward_{cell_line}.pt"
    backward_path = f"{DATA_BASE}/data_backward_{cell_line}.pt"
    edge_index_path = f"{DATA_BASE}/edge_index_{cell_line}.pt"
    splits_path = f"{SPLITS_BASE}/splits/chemical/{cell_line}/random/5fold/splits.pt"

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
            "n_layers_nn": args.n_layers_nn,
            "n_layers_gnn": args.n_layers_gnn,
            "num_vars": num_vars,
            "positional_features_dims": args.pos_dim,
            "embedding_layer_dim": args.embed_dim,
            "dim_gnn": args.gnn_dim,
        },
        response_kwargs={'train': True},
        perturbation_kwargs={'train': True}
    )

    param_count = sum(p.numel() for p in model.response_model.parameters())
    param_count += sum(p.numel() for p in model.perturbation_model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Setup trainer
    print("Setting up trainer...")
    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda", "devices": [args.gpu]},
        log=True,
        logging_dir=str(args.output_path),
        use_forward_data=True,
        use_backward_data=True,
        use_intervention_data=True,
        use_supervision=True,
        supervision_multiplier=args.supervision_mult,
        log_train=False,
        log_test=True,
    )

    # Train
    print(f"\nTraining fold {fold}...")
    dataset.prepare_fold(fold)

    results = trainer.train(
        model,
        dataset,
        n_epochs=args.n_epochs,
        early_stopping_kwargs={"patience": args.patience, "skip": 0}
    )

    return model, dataset, trainer, results


def evaluate_pdgrapher(model, dataset, device):
    """
    Extract predictions and compute DE metrics.

    Args:
        model: Trained PDGrapher model
        dataset: Dataset with test data
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    model.response_model.eval()
    model.perturbation_model.eval()

    all_true_treated = []
    all_pred_treated = []
    all_diseased = []

    # Get test data loader
    test_loader = dataset.backward_test_loader

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)

            # Get diseased (control) state
            diseased = batch.x.cpu().numpy()

            # Get true treated
            true_treated = batch.y.cpu().numpy()

            # Predict treated state
            # PDGrapher uses the perturbation to predict response
            pred = model.response_model(batch)
            pred_treated = pred.cpu().numpy()

            all_diseased.append(diseased)
            all_true_treated.append(true_treated)
            all_pred_treated.append(pred_treated)

    # Stack arrays
    diseased = np.vstack(all_diseased)
    true_treated = np.vstack(all_true_treated)
    pred_treated = np.vstack(all_pred_treated)

    # Compute metrics
    metrics = compute_de_metrics(true_treated, pred_treated, diseased)

    return metrics


def main(args):
    """Main training function."""
    print(f"\n{'='*60}")
    print(f"PDGrapher Training - {args.cell_line} Fold {args.fold}")
    print(f"{'='*60}\n")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    args.output_path = Path(args.output_dir) / f"pdgrapher_{args.cell_line}_fold{args.fold}"
    args.output_path.mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    wandb.init(
        project="PDGrapher_DE",
        name=f"pdgrapher_{args.cell_line}_fold{args.fold}",
        config=vars(args),
        reinit=True
    )

    # Train model
    model, dataset, trainer, results = train_pdgrapher(args.cell_line, args.fold, args)

    print(f"\nTraining complete!")

    # Log PDGrapher's native results
    if results:
        print("\n=== PDGrapher Native Metrics ===")
        for key, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
                wandb.log({f'pdg_{key}': value})

    # Try to compute DE metrics (may not work depending on model structure)
    try:
        device = torch.device(f'cuda:{args.gpu}')
        de_metrics = evaluate_pdgrapher(model, dataset, device)

        print("\n=== DE Metrics ===")
        print(f"R² Top-20:   {de_metrics['r2_top20']:.4f}")
        print(f"R² Top-40:   {de_metrics['r2_top40']:.4f}")
        print(f"R² Top-80:   {de_metrics['r2_top80']:.4f}")
        print(f"Pearson DE:  {de_metrics['pearson_de']:.4f}")
        print(f"Spearman DE: {de_metrics['spearman_de']:.4f}")

        wandb.log({
            'final_r2_top20': de_metrics['r2_top20'],
            'final_r2_top40': de_metrics['r2_top40'],
            'final_r2_top80': de_metrics['r2_top80'],
            'final_pearson_de': de_metrics['pearson_de'],
            'final_spearman_de': de_metrics['spearman_de'],
        })

        # Save metrics
        with open(args.output_path / 'de_metrics.pkl', 'wb') as f:
            pickle.dump(de_metrics, f)

    except Exception as e:
        print(f"\nWarning: Could not compute DE metrics: {e}")
        print("Using PDGrapher's native metrics only.")

    # Save native results
    if results:
        with open(args.output_path / 'pdgrapher_results.pkl', 'wb') as f:
            pickle.dump(results, f)

    print(f"\nResults saved to: {args.output_path}")

    wandb.finish()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PDGrapher on PDGrapher data')

    # Data args
    parser.add_argument('--cell_line', type=str, default='A549',
                        help='Cell line to train on')
    parser.add_argument('--fold', type=int, default=1,
                        help='Cross-validation fold (1-indexed for PDGrapher)')

    # Model args
    parser.add_argument('--n_layers_gnn', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--n_layers_nn', type=int, default=1,
                        help='Number of MLP layers')
    parser.add_argument('--pos_dim', type=int, default=16,
                        help='Positional feature dimension')
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='Embedding layer dimension')
    parser.add_argument('--gnn_dim', type=int, default=16,
                        help='GNN hidden dimension')

    # Training args
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--supervision_mult', type=float, default=0.01,
                        help='Supervision loss multiplier')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output_dir', type=str,
                        default='/raid/home/joshua/projects/PDGrapher_Baseline_Models/output/pdgrapher',
                        help='Output directory')

    args = parser.parse_args()
    main(args)
