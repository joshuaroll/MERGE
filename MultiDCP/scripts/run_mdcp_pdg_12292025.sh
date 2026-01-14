#!/usr/bin/env bash
set -euo pipefail

# Run fold 2 on GPU 6
#./run_mdcp_pdg_12292025.sh --fold 3 --gpu 5


# Parse command-line arguments
SELECTED_FOLD=""
SELECTED_GPU=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --fold|-f)
            SELECTED_FOLD="$2"
            shift 2
            ;;
        --gpu|-g)
            SELECTED_GPU="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--fold FOLD] [--gpu GPU]"
            echo "  --fold, -f: Fold number (1-5). If not specified, runs all folds."
            echo "  --gpu, -g: GPU device ID. If not specified, defaults to 4."
            exit 1
            ;;
    esac
done

PYTHON_BIN="python3"
CUDA_VISIBLE_DEVICES="${SELECTED_GPU:-4}"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

SCRIPT="${BASE_DIR}/src/multidcp_ae_de_pdg_12192025.py"

# Data files - using original locations (shared data)
DRUG_FILE="/raid/home/joshua/projects/MultiDCP_pdg/data/all_drugs_pdg.csv"
GENE_FILE="/raid/home/joshua/data/MultiDCP/data/gene_vector.csv"
DATA_PICKLE="/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl"
DISEASED_PICKLE="/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl"

PRED_RESULT="/raid/home/joshua/data/MultiDCP/data/teacher_student/teach_stu_perturbedGX_split1_03242023.csv"
HIDDEN_RESULT="/raid/home/joshua/data/MultiDCP/data/teacher_student/teach_stu_perturbedGX_hidden.csv"

CELL_GE_FILE="/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv"
ALL_CELLS="/raid/home/joshua/data/MultiDCP/data/ccle_tcga_ad_cells.p"

# New: splits base path for PDGrapher-style splits
SPLITS_BASE_PATH="/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical"

OUTDIR="${BASE_DIR}/output"
mkdir -p "${OUTDIR}"

# Test cells (now each gets its own split file)
TEST_CELLS=(HT29 PC3 HELA MCF7 A549 A375 VCAP MDAMB231 BT20)

# Folds - use provided fold or all folds
if [[ -n "${SELECTED_FOLD}" ]]; then
    FOLDS=("${SELECTED_FOLD}")
else
    FOLDS=(1 2 3 4 5)
fi

for FOLD in "${FOLDS[@]}"; do
    for TEST in "${TEST_CELLS[@]}"; do

    LOG_FILE="${OUTDIR}/cellwise_output_pdg_de_test_${TEST}_fold_${FOLD}_cuda_${CUDA_VISIBLE_DEVICES}.log"

    echo "============================================================"
    echo "Running TEST=${TEST} FOLD=${FOLD}"
    echo "Log: ${LOG_FILE}"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" "${SCRIPT}" \
      --drug_file "${DRUG_FILE}" \
      --gene_file "${GENE_FILE}" \
      --data_pickle "${DATA_PICKLE}" \
      --diseased_pickle "${DISEASED_PICKLE}" \
      --test_cell "${TEST}" \
      --use_split_file \
      --splits_base_path "${SPLITS_BASE_PATH}" \
      --fold "${FOLD}" \
      --gpu "${CUDA_VISIBLE_DEVICES}" \
      --dropout 0.3 \
      --batch_size 32 \
      --max_epoch 300 \
      --predicted_result_for_testset "${PRED_RESULT}" \
      --hidden_repr_result_for_testset "${HIDDEN_RESULT}" \
      --cell_ge_file "${CELL_GE_FILE}" \
      --all_cells "${ALL_CELLS}" \
      | tee "${LOG_FILE}"

  done
done
