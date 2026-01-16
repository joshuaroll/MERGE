import numpy as np
import random
import torch
from .molecules import Molecules
import pdb
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Define metadata columns that are NOT gene expression
METADATA_COLS = ['sig_id', 'idx', 'pert_id', 'pert_type', 'cell_id', 'pert_idose', 'pert_iname', 'time']


def read_drug_number(input_file, num_feature):
    drug = []
    drug_vec = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            assert len(line) == num_feature + 1, "Wrong format"
            bin_vec = [float(i) for i in line[1:]]
            drug.append(line[0])
            drug_vec.append(bin_vec)
    drug_vec = np.asarray(drug_vec, dtype=np.float64)
    index = []
    for i in range(np.shape(drug_vec)[1]):
        if len(set(drug_vec[:, i])) > 1:
            index.append(i)
    drug_vec = drug_vec[:, index]
    drug = dict(zip(drug, drug_vec))
    return drug, len(index)


def read_drug_string(input_file):
    with open(input_file, 'r') as f:
        drug = dict()
        for line in f:
            line = line.strip().split(',')
            assert len(line) == 2, "Wrong format"
            drug[line[0]] = line[1]
    return drug, None


def read_gene(input_file, device):
    with open(input_file, 'r') as f:
        gene = []
        for line in f:
            line = line.strip().split(',')
            assert len(line) == 129, "Wrong format"
            gene.append([float(i) for i in line[1:]])
    return torch.from_numpy(np.asarray(gene, dtype=np.float64)).to(device)


def convert_smile_to_feature(smiles, device):
    molecules = Molecules(smiles)
    node_repr = torch.FloatTensor([node.data for node in molecules.get_node_list('atom')]).to(device).double()
    edge_repr = torch.FloatTensor([node.data for node in molecules.get_node_list('bond')]).to(device).double()
    return {'molecules': molecules, 'atom': node_repr, 'bond': edge_repr}


def create_mask_feature(data, device):
    batch_idx = data['molecules'].get_neighbor_idx_by_batch('atom')
    molecule_length = [len(idx) for idx in batch_idx]
    mask = torch.zeros(len(batch_idx), max(molecule_length)).to(device).double()
    for idx, length in enumerate(molecule_length):
        mask[idx][:length] = 1
    return mask


def choose_mean_example(examples):
    num_example = len(examples)
    mean_value = (num_example - 1) / 2
    indexes = np.argsort(examples, axis=0)
    indexes = np.argsort(indexes, axis=0)
    indexes = np.mean(indexes, axis=1)
    distance = (indexes - mean_value)**2
    index = np.argmin(distance)
    return examples[index]


def split_data_by_pert_id(pert_id):
    random.shuffle(pert_id)
    num_pert_id = len(pert_id)
    fold_size = int(num_pert_id/10)
    train_pert_id = pert_id[:fold_size*6]
    dev_pert_id = pert_id[fold_size*6: fold_size*8]
    test_pert_id = pert_id[fold_size*8:]
    return train_pert_id, dev_pert_id, test_pert_id


def read_data_binary(input_file, filter):
    """
    :param input_file: including the time, pertid, perttype, cellid, dosage and the perturbed gene expression file (label)
    :param filter: help to check whether the pertid is in the research scope, cells in the research scope ...
    :return: the features, labels and cell type
    """
    feature = []
    labels = []
    
    data = dict()
    pert_id = []
    with open(input_file, 'r') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip().split(',')
           
            ft = ','.join(line[0:4])
            lb = [i for i in line[4:]]
            if ft in data.keys():
                data[ft].append(lb)
            else:
                data[ft] = [lb]
                    
    for ft, lb in sorted(data.items()):
    
        ft = ft.split(',')
        feature.append(ft)
        labels.append(lb[0])
    
    return np.asarray(feature), np.asarray(labels, dtype=np.float64)


def read_data(input_file, filter=None):
    """
    :param input_file: including the time, pertid, perttype, cellid, dosage and the perturbed gene expression file (label)
    :param filter: help to check whether the pertid is in the research scope, cells in the research scope ...
                   If no filters are provided, process all data.
    :return: the features, labels and cell type
    """
    feature = []
    label = []
    data = dict()
    pert_id = []

    if filter is None:
        filter = {}

    with open(input_file, 'r') as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip().split(',')
            
            time_cond = "time" not in filter or filter["time"] in line[0]
            pert_id_cond = "pert_id" not in filter or line[1] in filter.get('pert_id', [])
            pert_type_cond = "pert_type" not in filter or line[2] in filter.get("pert_type", [])
            cell_id_cond = "cell_id" not in filter or line[3] in filter.get('cell_id', [])
            pert_idose_cond = "pert_idose" not in filter or line[4] in filter.get("pert_idose", [])
            
            if time_cond and pert_id_cond and pert_type_cond and cell_id_cond and pert_idose_cond:
                ft = ','.join(line[1:6])
                lb = [float(i) for i in line[6:]]
                if ft in data.keys():
                    data[ft].append(lb)
                else:
                    data[ft] = [lb]

    for ft, lb in sorted(data.items()):
        ft = ft.split(',')
        feature.append(ft)
        pert_id.append(ft[0])
        if len(lb) == 1:
            label.append(lb[0])
        else:
            lb = choose_mean_example(lb)
            label.append(lb)
    _, cell_type = np.unique(np.asarray([x[2] for x in feature]), return_inverse=True)
    return np.asarray(feature), np.asarray(label, dtype=np.float64), cell_type


def get_gene_columns(frame):
    """
    Identify gene expression columns (everything that's not metadata).
    Returns list of column names that are gene expression values.
    """
    gene_cols = [col for col in frame.columns if col not in METADATA_COLS]
    return gene_cols


def read_data_from_dataframe(frame, filter=None, aggregate=True):
    """
    Wrapper around read_data for in-memory pandas DataFrames.
    
    Handles data with columns: [sig_id, idx, pert_id, pert_type, cell_id, pert_idose, <gene columns>...]
    
    Automatically identifies gene columns as any column not in METADATA_COLS.
    
    Args:
        frame: pandas DataFrame with the data
        filter: dict of filter conditions (optional)
        aggregate: if True (default), aggregate replicates by (pert_id, pert_type, cell_id, pert_idose).
                   if False, keep each row as a separate sample for sample-by-sample matching.
    """
    if filter is None:
        filter = {}

    required_cols = ["pert_id", "pert_type", "cell_id", "pert_idose"]
    missing_cols = [col for col in required_cols if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")

    # Identify gene columns dynamically
    gene_cols = get_gene_columns(frame)
    if len(gene_cols) == 0:
        raise ValueError("No gene expression columns found! Check that data has columns beyond metadata.")
    
    feature = []
    label = []
    pert_id_list = []

    if aggregate:
        # Original behavior: aggregate by (pert_id, pert_type, cell_id, pert_idose)
        data = dict()
        
        for _, row in frame.iterrows():
            time_cond = "time" not in filter or filter["time"] in str(row.get("time", ""))
            pert_id_cond = "pert_id" not in filter or row["pert_id"] in filter.get("pert_id", [])
            pert_type_cond = "pert_type" not in filter or row["pert_type"] in filter.get("pert_type", [])
            cell_id_cond = "cell_id" not in filter or row["cell_id"] in filter.get("cell_id", [])
            pert_idose_cond = "pert_idose" not in filter or row["pert_idose"] in filter.get("pert_idose", [])

            if time_cond and pert_id_cond and pert_type_cond and cell_id_cond and pert_idose_cond:
                ft_key = ",".join([str(row["pert_id"]), str(row["pert_type"]), str(row["cell_id"]), str(row["pert_idose"])])
                lb = row[gene_cols].astype(float).tolist()
                data.setdefault(ft_key, []).append(lb)

        for ft_key, lb in sorted(data.items()):
            ft = ft_key.split(',')
            feature.append(ft)
            pert_id_list.append(ft[0])
            if len(lb) == 1:
                label.append(lb[0])
            else:
                label.append(choose_mean_example(lb))
    else:
        # New behavior: keep each row as separate sample (for sample-by-sample DE matching)
        for _, row in frame.iterrows():
            time_cond = "time" not in filter or filter["time"] in str(row.get("time", ""))
            pert_id_cond = "pert_id" not in filter or row["pert_id"] in filter.get("pert_id", [])
            pert_type_cond = "pert_type" not in filter or row["pert_type"] in filter.get("pert_type", [])
            cell_id_cond = "cell_id" not in filter or row["cell_id"] in filter.get("cell_id", [])
            pert_idose_cond = "pert_idose" not in filter or row["pert_idose"] in filter.get("pert_idose", [])

            if time_cond and pert_id_cond and pert_type_cond and cell_id_cond and pert_idose_cond:
                ft = [str(row["pert_id"]), str(row["pert_type"]), str(row["cell_id"]), str(row["pert_idose"])]
                lb = row[gene_cols].astype(float).tolist()
                feature.append(ft)
                label.append(lb)
                pert_id_list.append(ft[0])
    
    if len(feature) == 0:
        raise ValueError("No samples passed the filter! Check filter conditions.")
    
    _, cell_type = np.unique(np.asarray([x[2] for x in feature]), return_inverse=True)
    return np.asarray(feature), np.asarray(label, dtype=np.float64), cell_type


def transform_to_tensor_per_dataset_binary(feature, label, drug, device, basal_expression_file):
    if not basal_expression_file.endswith('csv'):
        basal_expression_file += '.csv'
    basal_csv = pd.read_csv(basal_expression_file, index_col=0)

    pert_type_set  = sorted(list(set(feature[:, 1])))
    cell_id_set    = sorted(list(set(feature[:, 2])))
    pert_idose_set = sorted(list(set(feature[:, 3])))

    use_pert_type  = len(pert_type_set) > 1
    use_cell_id    = True
    use_pert_idose = len(pert_idose_set) > 1

    if use_pert_type:
        pert_type_dict = dict(zip(pert_type_set, range(len(pert_type_set))))
        final_pert_type_feature = []
    if use_pert_idose:
        pert_idose_dict = dict(zip(pert_idose_set, range(len(pert_idose_set))))
        final_pert_idose_feature = []

    drug_feature = []
    final_cell_id_feature = []

    print('Feature Summary (printing from data_utils):')
    print(pert_type_set); print(cell_id_set); print(pert_idose_set)

    for ft in feature:
        drug_feature.append(drug[ft[0]])

        cell_id = ft[2]
        try:
            vec = basal_csv.loc[cell_id, :].to_numpy(dtype=np.float64)
        except KeyError:
            raise KeyError(f"Cell '{cell_id}' not found in basal expression file {basal_expression_file}")
        final_cell_id_feature.append(vec)

        if use_pert_type:
            v = np.zeros(len(pert_type_set)); v[pert_type_dict[ft[1]]] = 1
            final_pert_type_feature.append(v.astype(np.float64))
        if use_pert_idose:
            v = np.zeros(len(pert_idose_set)); v[pert_idose_dict[ft[3]]] = 1
            final_pert_idose_feature.append(v.astype(np.float64))

    feature_dict = {'drug': np.asarray(drug_feature)}
    if use_pert_type:
        feature_dict['pert_type'] = torch.from_numpy(np.asarray(final_pert_type_feature, dtype=np.float64)).to(device)
    feature_dict['cell_id'] = torch.from_numpy(np.asarray(final_cell_id_feature, dtype=np.float64)).to(device)
    if use_pert_idose:
        feature_dict['pert_idose'] = torch.from_numpy(np.asarray(final_pert_idose_feature, dtype=np.float64)).to(device)

    label_binary = torch.from_numpy(label).to(device)
    return feature_dict, label_binary, use_pert_type, use_cell_id, use_pert_idose


def transform_to_tensor_per_dataset_ehill(feature, label, drug, device, basal_expression_file):
    if not basal_expression_file.endswith('csv'):
        basal_expression_file += '.csv'
    basal_csv = pd.read_csv(basal_expression_file, index_col=0)

    cell_id_set    = sorted(list(set(feature[:, 1])))
    pert_idose_set = sorted(list(set(feature[:, 2])))

    use_cell_id    = True
    use_pert_idose = len(pert_idose_set) > 1

    if use_pert_idose:
        pert_idose_dict = dict(zip(pert_idose_set, range(len(pert_idose_set))))
        final_pert_idose_feature = []

    drug_feature = []
    final_cell_id_feature = []

    for ft in feature:
        drug_feature.append(drug[ft[0]])

        cell_id = ft[1]
        try:
            vec = basal_csv.loc[cell_id, :].to_numpy(dtype=np.float64)
        except KeyError:
            raise KeyError(f"Cell '{cell_id}' not found in basal expression file {basal_expression_file}")
        final_cell_id_feature.append(vec)

        if use_pert_idose:
            v = np.zeros(len(pert_idose_set)); v[pert_idose_dict[ft[2]]] = 1
            final_pert_idose_feature.append(v.astype(np.float64))

    feature_dict = {
        'drug': np.asarray(drug_feature),
        'cell_id': torch.from_numpy(np.asarray(final_cell_id_feature, dtype=np.float64)).to(device),
    }
    if use_pert_idose:
        feature_dict['pert_idose'] = torch.from_numpy(np.asarray(final_pert_idose_feature, dtype=np.float64)).to(device)

    label_regression = torch.from_numpy(label).to(device)
    return feature_dict, label_regression, False, use_cell_id, use_pert_idose


def transform_to_tensor_per_dataset(feature, label, drug, device, basal_expression_file):
    if not basal_expression_file.endswith('csv'):
        basal_expression_file += '.csv'
    basal_csv = pd.read_csv(basal_expression_file, index_col=0)

    pert_type_set  = sorted(list(set(feature[:, 1])))
    cell_id_set    = sorted(list(set(feature[:, 2])))
    pert_idose_set = sorted(list(set(feature[:, 3])))

    use_pert_type  = len(pert_type_set) > 1
    use_cell_id    = True
    use_pert_idose = len(pert_idose_set) > 1

    if use_pert_type:
        pert_type_dict = dict(zip(pert_type_set, range(len(pert_type_set))))
        final_pert_type_feature = []
    if not use_pert_idose and len(pert_idose_set) == 1:
        pert_idose_set = pert_idose_set + ['3 uM']
        use_pert_idose = True
    if use_pert_idose:
        pert_idose_dict = dict(zip(pert_idose_set, range(len(pert_idose_set))))
        final_pert_idose_feature = []

    drug_feature = []
    final_cell_id_feature = []

    for ft in feature:
        drug_feature.append(drug[ft[0]])

        if use_pert_type:
            v = np.zeros(len(pert_type_set)); v[pert_type_dict[ft[1]]] = 1
            final_pert_type_feature.append(v.astype(np.float64))

        cell_id = ft[2]
        try:
            vec = basal_csv.loc[cell_id, :].to_numpy(dtype=np.float64)
        except KeyError:
            raise KeyError(f"Cell '{cell_id}' not found in basal expression file {basal_expression_file}")
        final_cell_id_feature.append(vec)

        if use_pert_idose:
            v = np.zeros(len(pert_idose_set)); v[pert_idose_dict[ft[3]]] = 1
            final_pert_idose_feature.append(v.astype(np.float64))

    feature_dict = {'drug': np.asarray(drug_feature)}
    if use_pert_type:
        feature_dict['pert_type'] = torch.from_numpy(np.asarray(final_pert_type_feature, dtype=np.float64)).to(device)
    feature_dict['cell_id'] = torch.from_numpy(np.asarray(final_cell_id_feature, dtype=np.float64)).to(device)
    if use_pert_idose:
        feature_dict['pert_idose'] = torch.from_numpy(np.asarray(final_pert_idose_feature, dtype=np.float64)).to(device)

    label_regression = torch.from_numpy(label).to(device)
    return feature_dict, label_regression, use_pert_type, use_cell_id, use_pert_idose


def transfrom_to_tensor(feature_train, label_train, feature_dev, label_dev, feature_test, label_test, drug,
                        device, basal_expression_file_name):
    """
    :param feature_train: features like pertid, dosage, cell id, etc. will be used to transfer to tensor over here
    :param label_train:
    :param feature_dev:
    :param label_dev:
    :param feature_test:
    :param label_test:
    :param drug: a drug dictionary mapping drug name into smile strings
    :param device: save on gpu device if necessary
    :return:
    """
    train_feature, train_label_regression, use_pert_type_train, use_cell_id_train, use_pert_idose_train = \
        transform_to_tensor_per_dataset(feature_train, label_train, drug, device, basal_expression_file_name)
    dev_feature, dev_label_regression, use_pert_type_dev, use_cell_id_dev, use_pert_idose_dev = \
        transform_to_tensor_per_dataset(feature_dev, label_dev, drug, device, basal_expression_file_name)
    test_feature, test_label_regression, use_pert_type_test, use_cell_id_test, use_pert_idose_test = \
        transform_to_tensor_per_dataset(feature_test, label_test, drug, device, basal_expression_file_name)
    assert use_pert_type_train == use_pert_type_dev and use_pert_type_train == use_pert_type_test, \
            'use pert type is not consistent'
    assert use_cell_id_train == use_cell_id_dev and use_cell_id_train == use_cell_id_test, \
            'use cell id is not consistent'
    assert use_pert_idose_train == use_pert_idose_dev and use_pert_idose_train == use_pert_idose_test, \
            'use pert idose is not consistent'
    return train_feature, dev_feature, test_feature, train_label_regression, dev_label_regression, \
           test_label_regression, use_pert_type_train, use_cell_id_train, use_pert_idose_train


if __name__ == '__main__':
    filter = {"time": "24H", "pert_id": ['BRD-U41416256', 'BRD-U60236422'], "pert_type": ["trt_cp"],
              "cell_id": ['A375', 'HA1E', 'HELA', 'HT29', 'MCF7', 'PC3', 'YAPC'],
              "pert_idose": ["0.04 um", "0.12 um", "0.37 um", "1.11 um", "3.33 um", "10.0 um"]}
    ft, lb = read_data('../data/signature_train.csv', filter)
    print(np.shape(ft))
    print(np.shape(lb))