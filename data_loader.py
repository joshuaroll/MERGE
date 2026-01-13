#!/usr/bin/env python
"""
Unified data loader for PDGrapher baseline models.
Loads data from MultiDCP pickle format and converts to formats needed by each model.
"""
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PDGrapherDataLoader:
    """
    Unified data loader that can convert PDGrapher/MultiDCP data to various formats
    needed by different baseline models.
    """

    # Cell lines available in the chemical perturbation dataset
    CELL_LINES = ["A549", "A375", "BT20", "HELA", "HT29", "MCF7", "MDAMB231", "PC3", "VCAP"]

    # Metadata columns to exclude from gene expression
    METADATA_COLS = ['sig_id', 'idx', 'pert_id', 'pert_type', 'cell_id', 'pert_idose',
                     'cell_type', 'dose', 'smiles']

    def __init__(self,
                 treated_path: str = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl",
                 diseased_path: str = "/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl"):
        """
        Initialize the data loader.

        Args:
            treated_path: Path to the treated expression pickle file
            diseased_path: Path to the diseased expression pickle file
        """
        self.treated_path = treated_path
        self.diseased_path = diseased_path
        self._treated_df = None
        self._diseased_df = None
        self._gene_cols = None

    def load(self):
        """Load the data from pickle files."""
        print("Loading treated expression data...")
        with open(self.treated_path, "rb") as f:
            self._treated_df = pickle.load(f)

        print("Loading diseased expression data...")
        with open(self.diseased_path, "rb") as f:
            self._diseased_df = pickle.load(f)

        # Identify gene columns
        self._gene_cols = [c for c in self._treated_df.columns if c not in self.METADATA_COLS]

        print(f"Loaded {len(self._treated_df)} samples with {len(self._gene_cols)} genes")
        return self

    @property
    def treated_df(self) -> pd.DataFrame:
        if self._treated_df is None:
            self.load()
        return self._treated_df

    @property
    def diseased_df(self) -> pd.DataFrame:
        if self._diseased_df is None:
            self.load()
        return self._diseased_df

    @property
    def gene_cols(self) -> List[str]:
        if self._gene_cols is None:
            self.load()
        return self._gene_cols

    @property
    def n_genes(self) -> int:
        return len(self.gene_cols)

    @property
    def n_samples(self) -> int:
        return len(self.treated_df)

    def get_cell_line_data(self, cell_line: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get data for a specific cell line."""
        mask = self.treated_df['cell_id'] == cell_line
        return self.treated_df[mask], self.diseased_df[mask]

    def get_expression_arrays(self, indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get treated and diseased expression as numpy arrays.

        Args:
            indices: Optional list of sample indices. If None, returns all samples.

        Returns:
            Tuple of (treated_expr, diseased_expr) arrays with shape (n_samples, n_genes)
        """
        if indices is None:
            treated = self.treated_df[self.gene_cols].values.astype(np.float32)
            diseased = self.diseased_df[self.gene_cols].values.astype(np.float32)
        else:
            treated = self.treated_df.iloc[indices][self.gene_cols].values.astype(np.float32)
            diseased = self.diseased_df.iloc[indices][self.gene_cols].values.astype(np.float32)
        return treated, diseased

    def get_differential_expression(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """Get differential expression (treated - diseased)."""
        treated, diseased = self.get_expression_arrays(indices)
        return treated - diseased

    def get_metadata(self, indices: Optional[List[int]] = None) -> pd.DataFrame:
        """Get metadata columns for samples."""
        meta_cols = [c for c in self.METADATA_COLS if c in self.treated_df.columns]
        if indices is None:
            return self.treated_df[meta_cols]
        return self.treated_df.iloc[indices][meta_cols]

    def to_anndata(self, indices: Optional[List[int]] = None):
        """
        Convert to AnnData format for scGen/Biolord.

        Returns:
            AnnData object with:
            - X: diseased expression (control)
            - obs['condition']: 'control' or 'treated'
            - obs['cell_type']: cell line
            - obs['perturbation']: drug SMILES
            - layers['treated']: treated expression
        """
        try:
            import anndata as ad
        except ImportError:
            raise ImportError("anndata is required. Install with: pip install anndata")

        treated, diseased = self.get_expression_arrays(indices)
        meta = self.get_metadata(indices)

        # Create AnnData with diseased (control) as X
        adata = ad.AnnData(X=diseased)
        adata.var_names = self.gene_cols

        # Add metadata to obs
        adata.obs['cell_type'] = meta['cell_id'].values
        adata.obs['perturbation'] = meta['smiles'].values if 'smiles' in meta.columns else 'unknown'
        adata.obs['dose'] = meta['dose'].values if 'dose' in meta.columns else 1.0
        adata.obs['condition'] = 'control'  # diseased state is the control

        # Store treated expression in layers
        adata.layers['treated'] = treated
        adata.layers['diseased'] = diseased

        return adata

    def to_perturbation_pairs(self, indices: Optional[List[int]] = None) -> List[Dict]:
        """
        Convert to perturbation pairs format for ChemCPA/CellOT.

        Returns:
            List of dicts with:
            - 'source': diseased expression (n_genes,)
            - 'target': treated expression (n_genes,)
            - 'smiles': drug SMILES string
            - 'cell_type': cell line
            - 'dose': dose value
        """
        treated, diseased = self.get_expression_arrays(indices)
        meta = self.get_metadata(indices)

        pairs = []
        for i in range(len(treated)):
            pairs.append({
                'source': diseased[i],
                'target': treated[i],
                'smiles': meta.iloc[i].get('smiles', 'unknown'),
                'cell_type': meta.iloc[i].get('cell_id', 'unknown'),
                'dose': meta.iloc[i].get('dose', 1.0),
            })
        return pairs

    def get_train_test_split(self, cell_line: str, fold: int = 0, n_folds: int = 5) -> Tuple[List[int], List[int]]:
        """
        Get train/test indices for a cell line using k-fold split.

        Args:
            cell_line: Cell line to split
            fold: Which fold to use as test set (0 to n_folds-1)
            n_folds: Number of folds

        Returns:
            Tuple of (train_indices, test_indices)
        """
        mask = self.treated_df['cell_id'] == cell_line
        cell_indices = np.where(mask)[0]

        # Shuffle with fixed seed for reproducibility
        rng = np.random.RandomState(42)
        shuffled = rng.permutation(cell_indices)

        # Split into folds
        fold_size = len(shuffled) // n_folds
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(shuffled)

        test_indices = shuffled[test_start:test_end].tolist()
        train_indices = np.concatenate([shuffled[:test_start], shuffled[test_end:]]).tolist()

        return train_indices, test_indices


class TopKEvaluator:
    """
    Evaluator for top-k differentially expressed genes metrics.
    Matches the evaluation approach from MultiDCP_pdg.

    Key: All metrics are computed on DIFFERENTIAL EXPRESSION (DE = treated - diseased),
    not on raw expression values. R² = Pearson² on DE values.
    """

    def __init__(self, k_values: List[int] = [20, 40, 80]):
        self.k_values = k_values

    def compute_metrics(self,
                        true_treated: np.ndarray,
                        pred_treated: np.ndarray,
                        diseased: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics on DIFFERENTIAL EXPRESSION.

        Matches MultiDCP_pdg/PDGrapher paper methodology:
        - DE = treated - diseased
        - Top-k genes selected by |true_DE|
        - R² = Pearson² computed on DE values

        Args:
            true_treated: Ground truth treated expression (n_samples, n_genes)
            pred_treated: Predicted treated expression (n_samples, n_genes)
            diseased: Diseased expression (n_samples, n_genes)

        Returns:
            Dict of metric names to values
        """
        from scipy.stats import pearsonr

        n_samples = len(true_treated)

        # Compute differential expression (this is the key!)
        true_de = true_treated - diseased
        pred_de = pred_treated - diseased

        results = {}

        # Per-sample metrics for each k (on DE values)
        for k in self.k_values:
            r2_list = []
            pearson_list = []

            for i in range(n_samples):
                # Get top-k DEGs by absolute true DE (descending order)
                de_magnitude = np.abs(true_de[i])
                top_k_idx = np.argsort(de_magnitude)[::-1][:k]

                # Compute Pearson on DE values for top-k genes
                true_de_k = true_de[i, top_k_idx]
                pred_de_k = pred_de[i, top_k_idx]

                # Skip if no variance (constant values)
                if np.std(true_de_k) < 1e-10 or np.std(pred_de_k) < 1e-10:
                    continue

                r, _ = pearsonr(true_de_k, pred_de_k)
                if not np.isnan(r):
                    r2_list.append(r ** 2)
                    pearson_list.append(r)

            results[f'r2_top{k}'] = np.mean(r2_list) if r2_list else 0.0
            results[f'r2_top{k}_std'] = np.std(r2_list) if r2_list else 0.0
            results[f'pearson_top{k}'] = np.mean(pearson_list) if pearson_list else 0.0
            results[f'pearson_top{k}_std'] = np.std(pearson_list) if pearson_list else 0.0

        # All genes metrics (on DE values)
        pearson_all = []
        for i in range(n_samples):
            if np.std(true_de[i]) < 1e-10 or np.std(pred_de[i]) < 1e-10:
                continue
            r, _ = pearsonr(true_de[i], pred_de[i])
            if not np.isnan(r):
                pearson_all.append(r)

        results['pearson_all'] = np.mean(pearson_all) if pearson_all else 0.0
        results['pearson_all_std'] = np.std(pearson_all) if pearson_all else 0.0
        results['r2_all'] = results['pearson_all'] ** 2

        return results

    def print_results(self, results: Dict[str, float], model_name: str = "Model"):
        """Print formatted results."""
        print(f"\n=== {model_name} Results ===")
        for k in self.k_values:
            print(f"R² Top-{k} DEGs: {results[f'r2_top{k}']:.4f} ± {results[f'r2_top{k}_std']:.4f}")
        print(f"Pearson DE (all genes): {results['pearson_all']:.4f} ± {results['pearson_all_std']:.4f}")
        print(f"R² DE (all genes): {results['r2_all']:.4f}")


if __name__ == "__main__":
    # Test the data loader
    loader = PDGrapherDataLoader()
    loader.load()

    print(f"\nDataset summary:")
    print(f"  Total samples: {loader.n_samples}")
    print(f"  Number of genes: {loader.n_genes}")
    print(f"  Cell lines: {loader.CELL_LINES}")

    # Test getting data for one cell line
    cell_line = "A549"
    treated, diseased = loader.get_cell_line_data(cell_line)
    print(f"\n{cell_line} samples: {len(treated)}")

    # Test train/test split
    train_idx, test_idx = loader.get_train_test_split(cell_line, fold=0)
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Test AnnData conversion
    print("\nTesting AnnData conversion...")
    adata = loader.to_anndata(indices=list(range(100)))
    print(f"  AnnData shape: {adata.shape}")
    print(f"  Layers: {list(adata.layers.keys())}")
    print(f"  Obs columns: {list(adata.obs.columns)}")

    # Test no-change baseline
    print("\nComputing no-change baseline metrics...")
    evaluator = TopKEvaluator()
    treated_arr, diseased_arr = loader.get_expression_arrays(list(range(1000)))
    # No-change baseline: predict treated = diseased
    baseline_results = evaluator.compute_metrics(treated_arr, diseased_arr, diseased_arr)
    evaluator.print_results(baseline_results, "No-Change Baseline")
