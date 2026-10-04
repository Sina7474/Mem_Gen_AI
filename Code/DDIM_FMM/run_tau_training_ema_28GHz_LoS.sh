#!/bin/bash
# ==============================================================================
# Tau-Based DDIM Training — 28 GHz LoS-only scene — ALL DATASET SIZES
# ==============================================================================
# Trains the DDIM channel model on the 28 GHz LoS-only scene for every dataset
# size N, sequentially and unattended, with W = 256 (n_feat).
#
# The configuration is IDENTICAL to the 3.5 GHz W=256 EMA study
# (logs_ema/DDIM_tau_ema_*_incremental), so the two scenes are comparable:
#
#     n_feat (W)      256           (15,230,978 parameters)
#     n_T             200
#     betas           (1e-4, 0.02)
#     lrate           1e-4 (Adam)
#     batch size      B = min(N, 500)
#     EMA decay       0.9999 with warmup
#     seed            0
#     tau grid        DENSE_TAU_GRID (≈6 points/decade)
#     max_tau         200000  for N <= 1000
#                     300000  for N = 2000
#                     400000  for N = 4000
#
# Dataset: 28GHz_Channel_UE_positions_full_LoS_iso_115_8759_*.npz  (7222 users)
#          Training subsets are nested; the test set is the last 10 % (722 users).
#
# Each N is skipped automatically if its run already finished (model_final.pth
# present), so the script is safe to re-launch after an interruption.
#
# Usage:
#     chmod +x run_tau_training_ema_28GHz_LoS.sh
#     ./run_tau_training_ema_28GHz_LoS.sh
#
# Outputs:
#     Code/DDIM_FMM/logs_ema_28GHz_LoS/DDIM_tau_ema_28GHz_LoS_<N>_bs<B>_incremental/
#     Console logs: Code/DDIM_FMM/logs_ema_28GHz_LoS/run_logs/train_N<N>.log
# ==============================================================================

set -u

CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="${CODE_DIR}/logs_ema_28GHz_LoS"
RUN_LOGS="${LOGS_DIR}/run_logs"
SCRIPT="train_DDIM_tau_ema_28GHz_LoS.py"

N_FEAT=256
SIZES=(100 200 500 1000 2000 4000)

# max_tau per N — same convention as the 3.5 GHz W=256 runs
declare -A MAX_TAU=(
    [100]=200000
    [200]=200000
    [500]=200000
    [1000]=200000
    [2000]=300000
    [4000]=400000
)

cd "${CODE_DIR}" || exit 1
mkdir -p "${RUN_LOGS}"

echo "=================================================================="
echo " TAU-BASED DDIM TRAINING — 28 GHz LoS SCENE — ALL SIZES"
echo "=================================================================="
echo " Sizes    : ${SIZES[*]}"
echo " Width W  : ${N_FEAT}"
echo " Batch    : min(N, 500)"
echo " Logs     : ${LOGS_DIR}"
echo " Started  : $(date)"
echo "=================================================================="
echo ""

FAILED=()

for N in "${SIZES[@]}"; do
    BS=$(( N < 500 ? N : 500 ))
    TAU=${MAX_TAU[$N]}
    RUN_DIR="${LOGS_DIR}/DDIM_tau_ema_28GHz_LoS_${N}_bs${BS}_incremental"
    LOG="${RUN_LOGS}/train_N${N}.log"

    if [[ -f "${RUN_DIR}/model_final.pth" ]]; then
        echo "[$(date +%H:%M:%S)] N=${N}: already complete — skipping."
        continue
    fi

    echo "------------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] TRAINING N=${N}  (W=${N_FEAT}, batch=${BS}, max_tau=${TAU})"
    echo "  log -> ${LOG}"
    echo "------------------------------------------------------------------"

    python -u "${SCRIPT}" "${N}" \
        --max_tau "${TAU}" \
        --incremental \
        --n_feat "${N_FEAT}" \
        --batch_size "${BS}" \
        2>&1 | tee "${LOG}"

    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        echo "[$(date +%H:%M:%S)] !! N=${N} FAILED — continuing with the next size."
        FAILED+=("${N}")
    else
        echo "[$(date +%H:%M:%S)] N=${N} complete."
    fi
    echo ""
done

echo "=================================================================="
echo " ALL 28 GHz LoS TRAINING RUNS FINISHED"
echo " Finished : $(date)"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo " FAILED sizes: ${FAILED[*]}"
else
    echo " All sizes completed successfully."
fi
echo " Results  : ${LOGS_DIR}/DDIM_tau_ema_28GHz_LoS_*_incremental/"
echo " CSVs     : tau_loss_curve_N_*.csv"
echo "=================================================================="
