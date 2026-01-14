#!/usr/bin/env bash
set -euo pipefail

# Run baseline MultiDCP for 4 specific cell types on different GPUs
# Usage: nohup bash run_baseline_4cells.sh > ../output/baseline_4cells.log 2>&1 &

PYTHON_BIN="python3"
SCRIPT="multidcp_ae_de_pdg_12192025.py"
SRC_DIR="/raid/home/joshua/projects/MultiDCP_pdg/src"

DRUG_FILE="/raid/home/joshua/projects/MultiDCP_pdg/data/all_drugs_pdg.csv"
GENE_FILE="/raid/home/joshua/data/MultiDCP/data/gene_vector.csv"
DATA_PICKLE="/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_brddrugfiltered.pkl"
DISEASED_PICKLE="/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered.pkl"
PRED_RESULT="/raid/home/joshua/data/MultiDCP/data/teacher_student/teach_stu_perturbedGX_split1_03242023.csv"
HIDDEN_RESULT="/raid/home/joshua/data/MultiDCP/data/teacher_student/teach_stu_perturbedGX_hidden.csv"
CELL_GE_FILE="/raid/home/joshua/projects/MultiDCP_pdg/data/pdg_diseased_brddrugfiltered_avg_over_celltype_10x10717.csv"
ALL_CELLS="/raid/home/joshua/data/MultiDCP/data/ccle_tcga_ad_cells.p"
SPLITS_BASE_PATH="/raid/home/public/chemoe_collab_102025/PDGrapher/data/processed/splits/chemical"

OUTDIR="/raid/home/joshua/projects/MultiDCP_pdg/output"
mkdir -p "${OUTDIR}"

FOLD=1

# Cell types to GPU mapping
declare -A CELL_GPU=(
    ["A375"]=4
    ["VCAP"]=5
    ["MDAMB231"]=6
    ["BT20"]=7
)

echo "========================================================================"
echo "Running Baseline MultiDCP for 4 cell types (Fold ${FOLD})"
echo "========================================================================"

for CELL in A375 VCAP MDAMB231 BT20; do
    GPU="${CELL_GPU[$CELL]}"
    LOG_FILE="${OUTDIR}/cellwise_output_baseline_de_test_${CELL}_fold_${FOLD}_cuda_${GPU}.log"

    echo "Starting ${CELL} on GPU ${GPU} (PID will be shown below)"

    (cd "${SRC_DIR}" && CUDA_VISIBLE_DEVICES="${GPU}" \
    "${PYTHON_BIN}" "${SCRIPT}" \
      --drug_file "${DRUG_FILE}" \
      --gene_file "${GENE_FILE}" \
      --data_pickle "${DATA_PICKLE}" \
      --diseased_pickle "${DISEASED_PICKLE}" \
      --test_cell "${CELL}" \
      --use_split_file \
      --splits_base_path "${SPLITS_BASE_PATH}" \
      --fold "${FOLD}" \
      --gpu "${GPU}" \
      --dropout 0.3 \
      --batch_size 32 \
      --max_epoch 300 \
      --predicted_result_for_testset "${PRED_RESULT}" \
      --hidden_repr_result_for_testset "${HIDDEN_RESULT}" \
      --cell_ge_file "${CELL_GE_FILE}" \
      --all_cells "${ALL_CELLS}" \
      --linear_encoder_flag \
      --model_name multidcp_baseline_de \
      > "${LOG_FILE}" 2>&1) &

    PID=$!
    echo "  → ${CELL} started on GPU ${GPU}, PID: ${PID}, Log: ${LOG_FILE}"
    sleep 2  # Brief pause between launches
done

echo ""
echo "========================================================================"
echo "All 4 cell types launched in parallel!"
echo "Monitor with: tail -f ${OUTDIR}/cellwise_output_baseline_de_test_*_fold_${FOLD}_cuda_*.log"
echo "Check GPUs: nvidia-smi"
echo "========================================================================"
