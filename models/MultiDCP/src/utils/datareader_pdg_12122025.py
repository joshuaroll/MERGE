"""
datareader_pdg_de.py

Datareader for Differential Expression metrics.
Key change: Uses aggregate=False to keep all samples for sample-by-sample matching
with diseased data (matching biolord evaluation methodology).

Drop-in replacement for datareader_pdg.py when doing DE-based evaluation.
"""

import numpy as np
import random
import torch
import data_utils_pdg as data_utils
import pandas as pd
import pdb
import warnings
from pathlib import Path
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
warnings.filterwarnings("ignore")


class AEDataDataset(Dataset):

    def __init__(self, input_file_name, label_file_name, device):
        super(AEDataDataset, self).__init__()
        self.device = device
        self.feature = torch.from_numpy(np.asarray(pd.read_csv(input_file_name, index_col=0).values, dtype=np.float32)).to(device)
        self.label = torch.from_numpy(np.asarray(pd.read_csv(label_file_name, index_col=0).values, dtype=np.float32)).to(device)
        self.cell_type_code = torch.Tensor([*range(len(self.feature))]).long()

    def __len__(self):
        return self.feature.shape[0]

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx], self.cell_type_code[idx]


class AEDataLoader(pl.LightningDataModule):

    def __init__(self, device, args):
        super(AEDataLoader, self).__init__()
        self.batch_size = args.batch_size
        self.train_data_file = args.ae_input_file + '_train.csv'
        self.dev_data_file = args.ae_input_file + '_dev.csv'
        self.test_data_file = args.ae_input_file + '_test.csv'
        self.train_label_file = args.ae_label_file + '_train.csv'
        self.dev_label_file = args.ae_label_file + '_dev.csv'
        self.test_label_file = args.ae_label_file + '_test.csv'
        self.device = device

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_data = AEDataDataset(self.train_data_file, self.train_label_file, self.device)
        self.dev_data = AEDataDataset(self.dev_data_file, self.dev_label_file, self.device)
        self.test_data = AEDataDataset(self.test_data_file, self.test_label_file, self.device)
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_data, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)


class PerturbedDataset(Dataset):

    def __init__(self, drug_file, data_file, data_filter, device, cell_ge_file_name, aggregate=False):
        """
        Args:
            drug_file: Path to drug CSV file
            data_file: DataFrame or path to data
            data_filter: Filter dict
            device: torch device
            cell_ge_file_name: Cell gene expression file
            aggregate: If True, aggregate replicates. If False (default for DE), keep all samples.
        """
        super(PerturbedDataset, self).__init__()
        self.device = device
        self.drug, self.drug_dim = data_utils.read_drug_string(drug_file)
        self.aggregate = aggregate
        
        feature, label, self.cell_type = self._load_feature_label(data_file, data_filter)
        
        self.feature, self.label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            data_utils.transform_to_tensor_per_dataset(feature, label, self.drug, self.device, cell_ge_file_name)

    def _load_feature_label(self, data_source, data_filter):
        if isinstance(data_source, pd.DataFrame):
            return data_utils.read_data_from_dataframe(data_source, data_filter, aggregate=self.aggregate)
        if data_source is None:
            raise ValueError("Data source for PerturbedDataset cannot be None.")
        data_path = Path(data_source)
        if data_path.suffix in {".pkl", ".pickle"}:
            df = pd.read_pickle(data_path)
            return data_utils.read_data_from_dataframe(df, data_filter, aggregate=self.aggregate)
        # For CSV files, fall back to original read_data (which always aggregates)
        return data_utils.read_data(str(data_source), data_filter)

    def __len__(self):
        return self.feature['drug'].shape[0]

    def __getitem__(self, idx):
        output = dict()
        output['drug'] = self.feature['drug'][idx]
        if self.use_cell_id:
            output['cell_id'] = self.feature['cell_id'][idx]
        if self.use_pert_idose:
            output['pert_idose'] = self.feature['pert_idose'][idx]
        return output, self.label[idx], self.cell_type[idx]


class PerturbedDataLoader(pl.LightningDataModule):

    def __init__(self, data_filter, device, args):
        super(PerturbedDataLoader, self).__init__()
        self.batch_size = args.batch_size
        self.train_data_file = getattr(args, 'train_file', None)
        self.dev_data_file = getattr(args, 'dev_file', None)
        self.test_data_file = getattr(args, 'test_file', None)
        self.drug_file = args.drug_file
        self.data_filter = data_filter
        self.device = device
        self.cell_ge_file_name = args.cell_ge_file
        self.gene = data_utils.read_gene(args.gene_file, self.device)
        self.data_pickle = getattr(args, "data_pickle", None)
        self.test_cell = getattr(args, "test_cell", None)
        self.dev_cell = getattr(args, "dev_cell", None)
        
        # Split file support
        self.splits_base_path = getattr(args, "splits_base_path", None)
        self.fold = getattr(args, "fold", 1)
        self.use_split_file = getattr(args, "use_split_file", False)
        
        # KEY: aggregate option - False for DE metrics to match sample-by-sample
        # Default is False for this DE-specific datareader
        self.aggregate = getattr(args, "aggregate_replicates", False)
            
    def collate_fn(self, batch):
        features = {}
        features['drug'] = data_utils.convert_smile_to_feature([output['drug'] for output, _, _ in batch], self.device)
        features['mask'] = data_utils.create_mask_feature(features['drug'], self.device)
        for key in batch[0][0].keys():
            if key == 'drug':
                continue
            features[key] = torch.stack([output[key] for output, _, _ in batch], dim=0)
        labels = torch.stack([label for _, label, _ in batch], dim=0)
        cell_types = torch.Tensor([cell_type for _, _, cell_type in batch])
        return features, labels, torch.Tensor(cell_types).to(self.device)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        print("Setting up PerturbedDataLoader...")
        print(f"  Aggregate replicates: {self.aggregate}")
        
        # Option 1: Use pre-defined split files (PDGrapher style)
        if self.use_split_file and self.splits_base_path and self.test_cell:
            print(f"Using split file for cell type: {self.test_cell}, fold: {self.fold}")
            
            if not self.data_pickle:
                raise ValueError("`data_pickle` must be provided when using split files.")
            df = pd.read_pickle(self.data_pickle)
            
            # Filter to only the test_cell's data
            df = df[df['cell_id'] == self.test_cell].copy()
            print(f"Filtered to {self.test_cell}: {len(df)} samples")
            
            # Use the idx column for matching split indices
            df['_split_idx'] = df['idx']
            
            # Construct the path to the splits file
            splits_path = Path(self.splits_base_path) / self.test_cell / "random" / "5fold" / "splits.pt"
            if not splits_path.exists():
                raise FileNotFoundError(f"Splits file not found at: {splits_path}")
            
            # Load the splits
            splits = torch.load(splits_path, weights_only=False)
            print(f"Loaded splits from: {splits_path}")
            print(f"Available folds: {list(splits.keys())}")
            
            if self.fold not in splits:
                raise ValueError(f"Fold {self.fold} not found in splits. Available: {list(splits.keys())}")
            
            fold_splits = splits[self.fold]
            
            # Get indices for train/val/test
            train_indices = fold_splits['train_index_backward']
            val_indices = fold_splits['val_index_backward']
            test_indices = fold_splits['test_index_backward']
            
            # Convert to numpy/list if they're tensors
            if isinstance(train_indices, torch.Tensor):
                train_indices = train_indices.numpy()
            if isinstance(val_indices, torch.Tensor):
                val_indices = val_indices.numpy()
            if isinstance(test_indices, torch.Tensor):
                test_indices = test_indices.numpy()
            
            # Convert to sets for faster lookup
            train_idx_set = set(train_indices.tolist() if hasattr(train_indices, 'tolist') else train_indices)
            val_idx_set = set(val_indices.tolist() if hasattr(val_indices, 'tolist') else val_indices)
            test_idx_set = set(test_indices.tolist() if hasattr(test_indices, 'tolist') else test_indices)
            
            print(f"Split sizes from file - Train: {len(train_idx_set)}, Val: {len(val_idx_set)}, Test: {len(test_idx_set)}")
            
            # Match rows by their extracted _split_idx
            train_df = df[df['_split_idx'].isin(train_idx_set)].copy().reset_index(drop=True)
            dev_df = df[df['_split_idx'].isin(val_idx_set)].copy().reset_index(drop=True)
            test_df = df[df['_split_idx'].isin(test_idx_set)].copy().reset_index(drop=True)
            
            # Drop the idx and temporary _split_idx columns
            cols_to_drop = ['_split_idx', 'idx']
            train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], inplace=True)
            dev_df.drop(columns=[c for c in cols_to_drop if c in dev_df.columns], inplace=True)
            test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], inplace=True)
            
            print(f"Matched sizes - Train: {len(train_df)}, Val: {len(dev_df)}, Test: {len(test_df)}")
            
            if train_df.empty or dev_df.empty or test_df.empty:
                raise ValueError("One of the train/dev/test splits is empty.")
            
            self.train_data = PerturbedDataset(self.drug_file, train_df,
                    self.data_filter, self.device, self.cell_ge_file_name, aggregate=self.aggregate)
            self.dev_data = PerturbedDataset(self.drug_file, dev_df,
                    self.data_filter, self.device, self.cell_ge_file_name, aggregate=self.aggregate)
            self.test_data = PerturbedDataset(self.drug_file, test_df,
                    self.data_filter, self.device, self.cell_ge_file_name, aggregate=self.aggregate)
        
        # Option 2: Use cell-type based splitting
        elif self.data_pickle:
            df = pd.read_pickle(self.data_pickle)
            if not self.test_cell or not self.dev_cell:
                raise ValueError("`test_cell` and `dev_cell` must be provided.")
            test_df = df[df["cell_id"] == self.test_cell].copy()
            dev_df = df[df["cell_id"] == self.dev_cell].copy()
            train_df = df[~df["cell_id"].isin([self.test_cell, self.dev_cell])].copy()
            
            if train_df.empty or dev_df.empty or test_df.empty:
                raise ValueError("One of the train/dev/test splits is empty.")
            
            self.train_data = PerturbedDataset(self.drug_file, train_df,
                    self.data_filter, self.device, self.cell_ge_file_name, aggregate=self.aggregate)
            self.dev_data = PerturbedDataset(self.drug_file, dev_df,
                    self.data_filter, self.device, self.cell_ge_file_name, aggregate=self.aggregate)
            self.test_data = PerturbedDataset(self.drug_file, test_df,
                    self.data_filter, self.device, self.cell_ge_file_name, aggregate=self.aggregate)
        
        # Option 3: Use separate train/dev/test files
        else:
            if not all([self.train_data_file, self.dev_data_file, self.test_data_file]):
                raise ValueError("train_file, dev_file, and test_file must be provided.")
            self.train_data = PerturbedDataset(self.drug_file, self.train_data_file,
                     self.data_filter, self.device, self.cell_ge_file_name, aggregate=self.aggregate)
            self.dev_data = PerturbedDataset(self.drug_file, self.dev_data_file,
                     self.data_filter, self.device, self.cell_ge_file_name, aggregate=self.aggregate)
            self.test_data = PerturbedDataset(self.drug_file, self.test_data_file,
                     self.data_filter, self.device, self.cell_ge_file_name, aggregate=self.aggregate)
        
        self.use_pert_type = self.train_data.use_pert_type
        self.use_cell_id = self.train_data.use_cell_id
        self.use_pert_idose = self.train_data.use_pert_idose
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev_data, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.collate_fn)


# Keep remaining classes for backward compatibility
class EhillDataset(Dataset):
    def __init__(self, drug_file, data_file, data_filter, device, cell_ge_file_name):
        super(EhillDataset, self).__init__()
        self.device = device
        self.drug, self.drug_dim = data_utils.read_drug_string(drug_file)
        feature, label, self.cell_type = data_utils.read_data(data_file, data_filter)
        self.feature, self.label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            data_utils.transform_to_tensor_per_dataset_ehill(feature, label, self.drug, self.device, cell_ge_file_name)

    def __len__(self):
        return self.feature['drug'].shape[0]

    def __getitem__(self, idx):
        output = dict()
        output['drug'] = self.feature['drug'][idx]
        if self.use_cell_id:
            output['cell_id'] = self.feature['cell_id'][idx]
        if self.use_pert_idose:
            output['pert_idose'] = self.feature['pert_idose'][idx]
        return output, self.label[idx], self.cell_type[idx]


class EhillDataLoader(pl.LightningDataModule):

    def __init__(self, data_filter, device, args):
        super(EhillDataLoader, self).__init__()
        self.batch_size = args.batch_size
        self.train_data_file = args.hill_train_file
        self.dev_data_file = args.hill_dev_file
        self.test_data_file = args.hill_test_file
        self.drug_file = args.drug_file
        self.data_filter = data_filter
        self.device = device
        self.cell_ge_file_name = args.cell_ge_file
        self.gene = data_utils.read_gene(args.gene_file, self.device)
    
    def collate_fn(self, batch):
        features = {}
        features['drug'] = data_utils.convert_smile_to_feature([output['drug'] for output, _, _ in batch], self.device)
        features['mask'] = data_utils.create_mask_feature(features['drug'], self.device)
        for key in batch[0][0].keys():
            if key == 'drug':
                continue
            features[key] = torch.stack([output[key] for output, _, _ in batch], dim=0)
        labels = torch.stack([label for _, label, _ in batch], dim=0)
        cell_types = torch.Tensor([cell_type for _, _, cell_type in batch])
        return features, labels, torch.Tensor(cell_types).to(self.device)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.test_data = EhillDataset(self.drug_file, self.test_data_file,
                self.data_filter, self.device, self.cell_ge_file_name)
        self.use_pert_type = self.test_data.use_pert_type
        self.use_cell_id = self.test_data.use_cell_id
        self.use_pert_idose = self.test_data.use_pert_idose
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dev_data, batch_size=self.batch_size, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, collate_fn=self.collate_fn)


class DataReader(object):

    def __init__(self, drug_file, gene_file, data_file_train, data_file_dev, data_file_test,
                 data_filter, device, cell_ge_file_name):
        self.device = device
        self.drug, self.drug_dim = data_utils.read_drug_string(drug_file)
        self.gene = data_utils.read_gene(gene_file, self.device)
        feature_train, label_train, self.train_cell_type = data_utils.read_data(data_file_train, data_filter)
        feature_dev, label_dev, self.dev_cell_type = data_utils.read_data(data_file_dev, data_filter)
        feature_test, label_test, self.test_cell_type = data_utils.read_data(data_file_test, data_filter)
        self.train_feature, self.dev_feature, self.test_feature, self.train_label, \
        self.dev_label, self.test_label, self.use_pert_type, self.use_cell_id, self.use_pert_idose = \
            data_utils.transfrom_to_tensor(feature_train, label_train, feature_dev, label_dev,
                                           feature_test, label_test, self.drug, self.device, cell_ge_file_name)

    def get_batch_data(self, dataset, batch_size, shuffle):
        if dataset == 'train':
            feature = self.train_feature
            label = self.train_label
            cell_type = torch.Tensor(self.train_cell_type).to(self.device)
        elif dataset == 'dev':
            feature = self.dev_feature
            label = self.dev_label
            cell_type = torch.Tensor(self.dev_cell_type).to(self.device)
        elif dataset == 'test':
            feature = self.test_feature
            label = self.test_label
            cell_type = torch.Tensor(self.test_cell_type).to(self.device)
        if shuffle:
            index = torch.randperm(len(feature['drug'])).long()
            index = index.numpy()
        for start_idx in range(0, len(feature['drug']), batch_size):
            if shuffle:
                excerpt = index[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            output = dict()
            output['drug'] = data_utils.convert_smile_to_feature(feature['drug'][excerpt], self.device)
            output['mask'] = data_utils.create_mask_feature(output['drug'], self.device)
            if self.use_pert_type:
                output['pert_type'] = feature['pert_type'][excerpt]
            if self.use_cell_id:
                output['cell_id'] = feature['cell_id'][excerpt]
            if self.use_pert_idose:
                output['pert_idose'] = feature['pert_idose'][excerpt]
            yield output, label[excerpt], cell_type[excerpt]


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    class _Args:
        data_pickle = '/raid/home/joshua/projects/MultiDCP/MultiDCP/data/pdg_brddrugfiltered.pkl'
        
        use_split_file = True
        splits_base_path = '/raid/home/public/chemoe_collab_102025/PDGrapher/data/full_downloads/splits/chemical'
        test_cell = 'A375'
        fold = 1
        dev_cell = 'MCF7'
        
        aggregate_replicates = False  # KEY: False for DE metrics
        
        batch_size = 32
        ae_input_file = '../data/gene_expression_for_ae/gene_expression_combat_norm_978_split1'
        ae_label_file = ae_input_file
        drug_file = '../data/all_drugs_pdg.csv'
        gene_file = '../data/gene_vector.csv'
        cell_ge_file = '../data/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv'

    args = _Args()

    data_filter = {'pert_type': ['trt_cp']}

    pert_loader = PerturbedDataLoader(data_filter, device, args)
    pert_loader.setup()
    print(f'Perturbed sizes: train={len(pert_loader.train_data)}, dev={len(pert_loader.dev_data)}, test={len(pert_loader.test_data)}')