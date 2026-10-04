#!/bin/bash
# ==============================================================================
# Run Tau-Based DDIM Training for All Dataset Sizes
# ==============================================================================
# This script trains the unconditional DDIM channel model for all 7 dataset
# sizes with tau-based checkpointing and evaluation.
#
# Tau = optimizer update steps = epochs × ceil(N / batch_size)
#
# Each size trains until MAX_TAU optimizer steps, evaluating L_train and L_test
# at the TAU_GRID points defined in train_DDIM_tau.py.
#
# Usage:
#   chmod +x run_tau_training.sh
#   ./run_tau_training.sh
#
# Or run individual sizes:
#   conda activate Mem_Gen
#   cd Code/DDIM_FMM
#   python train_DDIM_tau.py 1000 --max_tau 200000 --incremental
# ==============================================================================

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
MAX_TAU=200000
SIZES=(100 200 500 1000 2000 4000 8000)

echo "============================================================"
echo " TAU-BASED DDIM TRAINING — ALL SIZES"
echo "============================================================"
echo " MAX_TAU: $MAX_TAU"
echo " Sizes: ${SIZES[*]}"
echo " Batch size: 100 (hardcoded)"
echo "============================================================"
echo ""

cd "${SCRIPT_DIR}"

for N in "${SIZES[@]}"; do
    echo ""
    echo "============================================================"
    echo " TRAINING N=$N  (max_tau=$MAX_TAU)"
    echo "============================================================"
    
    # Compute expected epochs for reference
    STEPS_PER_EPOCH=$(python -c "import math; print(math.ceil($N / 100))")
    MAX_EPOCHS=$(python -c "import math; print(math.ceil($MAX_TAU / math.ceil($N / 100)))")
    echo " steps_per_epoch: $STEPS_PER_EPOCH"
    echo " max_epochs needed: $MAX_EPOCHS"
    echo ""
    
    python train_DDIM_tau.py $N --max_tau $MAX_TAU --incremental
    
    echo ""
    echo " ✓ N=$N training complete."
    echo ""
done

echo ""
echo "============================================================"
echo " ALL TRAINING COMPLETE"
echo "============================================================"
echo ""
echo " Results in: Code/DDIM_FMM/logs/DDIM_tau_*_incremental/"
echo " CSV files: tau_loss_curve_N_*.csv"
echo ""
echo " To generate plots, run:"
echo "   cd DDIM_Evaluation"
echo "   python plot_tau_train_test_loss.py"
echo ""
