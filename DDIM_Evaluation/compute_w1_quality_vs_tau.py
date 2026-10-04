"""
compute_w1_quality_vs_tau.py — W1(Gen, Test) Quality vs Training Time τ
=========================================================================
For each dataset size N and each tau-based checkpoint, generate 5000 samples
and compute Wasserstein-1 distances on effective-rank distributions.

This is the channel-domain analogue of the paper's FID-vs-tau quality curve,
using effective-rank Wasserstein distance instead of image FID.

Metrics computed per (N, tau):
  - W1(Gen, Test)   : main quality metric (lower is better)
  - W1(Gen, Train)  : memorization diagnostic
  - W1(Test, Train) : real-data baseline (constant over tau for fixed N)
  - Excess_Gap      : W1(Gen, Test) - W1(Test, Train)

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python compute_w1_quality_vs_tau.py

    # Debug mode (fewer samples, coarser grid):
    python compute_w1_quality_vs_tau.py --debug
"""

import sys
import os
import json
import csv
import argparse
from math import ceil
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import importlib.util

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cDDIM_cFMM_Repo', 'Effective_Rank_Github'))

from effective_rank_core import compute_effective_rank, compute_wasserstein, normalize_max_abs, to_beamspace

# Import model classes from train_DDIM_tau.py to ensure architecture match
_train_path = os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM', 'train_DDIM_tau.py')
_spec = importlib.util.spec_from_file_location("train_ddim_tau_module", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
sys.modules['train_ddim_tau_module'] = _train_mod
_spec.loader.exec_module(_train_mod)
Unet = _train_mod.Unet
DDIM = _train_mod.DDIM


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000, 8000]
LOGS_BASE = os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM', 'logs')
OUTPUT_DIR = Path("results/w1_quality_vs_tau")

# Tau grid for generation-quality evaluation
GEN_EVAL_TAU_GRID = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

# Debug mode (fewer samples, coarser grid)
GEN_EVAL_TAU_GRID_DEBUG = [1000, 10000, 50000, 200000]

# Generation settings
NUM_GENERATED = 5000
NUM_GENERATED_DEBUG = 500
BATCH_SIZE_GEN = 100
GEN_SEED = 0  # Fixed seed for reproducible generation

# Training settings (must match train_DDIM_tau.py)
BATCH_SIZE_TRAIN = 100

# UPA antenna dimensions
ANTENNA = dict(Nrx_x=2, Nrx_y=2, Ntx_x=4, Ntx_y=8)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         HELPERS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_tau_log_dir(n_train):
    """Get the tau-based training log directory for a given N."""
    return os.path.join(LOGS_BASE, f"DDIM_tau_{n_train}_incremental")


def create_model(device):
    """Create the same model architecture used in training."""
    nn_model = Unet(in_channels=2, n_feat=256)
    ddim = DDIM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=200, device=device)
    return ddim


def generate_samples(ddim, n_samples, batch_size, device, seed=0):
    """Generate samples from a loaded model with fixed seed."""
    torch.manual_seed(seed)
    ddim.eval()
    all_samples = []
    n_batches = ceil(n_samples / batch_size)
    with torch.no_grad():
        for _ in range(n_batches):
            bs = min(batch_size, n_samples - len(all_samples))
            samples = ddim.sample(bs, (2, 4, 32), device)
            all_samples.append(samples.cpu().numpy())
    return np.concatenate(all_samples, axis=0)[:n_samples]


def samples_to_erank(samples_beamspace):
    """
    Convert beamspace samples (N, 2, 4, 32) to effective rank values.
    Generated samples are already in normalized beamspace.
    """
    H = (samples_beamspace[:, 0] + 1j * samples_beamspace[:, 1]).astype(np.complex64)
    H = normalize_max_abs(H)
    er, _ = compute_effective_rank(H, use_power=True)
    return er


def load_test_erank(log_dir):
    """
    Load test data (spatial domain), convert to beamspace, compute erank.
    test.npy shape: (N, 2, 4, 32), spatial domain.
    """
    test_path = os.path.join(log_dir, 'test.npy')
    test_arr = np.load(test_path)  # (4086, 2, 4, 32) spatial domain
    H_spatial = (test_arr[:, 0] + 1j * test_arr[:, 1]).astype(np.complex64)
    H_beam = to_beamspace(H_spatial, **ANTENNA)
    H_beam = normalize_max_abs(H_beam)
    er, _ = compute_effective_rank(H_beam, use_power=True)
    return er


def load_train_erank(log_dir):
    """
    Load training data (spatial domain), convert to beamspace, compute erank.
    train.npy shape: (N, 2, 4, 32), spatial domain.
    """
    train_path = os.path.join(log_dir, 'train.npy')
    train_arr = np.load(train_path)
    H_spatial = (train_arr[:, 0] + 1j * train_arr[:, 1]).astype(np.complex64)
    H_beam = to_beamspace(H_spatial, **ANTENNA)
    H_beam = normalize_max_abs(H_beam)
    er, _ = compute_effective_rank(H_beam, use_power=True)
    return er


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Compute W1(Gen, Test) quality vs tau for all dataset sizes")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: fewer samples, coarser grid")
    parser.add_argument("--sizes", type=int, nargs='+', default=None,
                        help="Specific sizes to evaluate (default: all)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    # Select configuration
    if args.debug:
        num_gen = NUM_GENERATED_DEBUG
        tau_grid = GEN_EVAL_TAU_GRID_DEBUG
        print("*** DEBUG MODE: 500 samples, coarse grid ***")
    else:
        num_gen = NUM_GENERATED
        tau_grid = GEN_EVAL_TAU_GRID

    sizes = args.sizes if args.sizes else TRAIN_SIZES
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("W1 Sample-Quality vs Training Time τ")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Sizes: {sizes}")
    print(f"  Tau grid: {tau_grid}")
    print(f"  Generating {num_gen} samples per checkpoint")
    print(f"  Generation seed: {GEN_SEED}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # ── Pre-compute test effective rank (same test set for all sizes) ─────
    # Use test data from any tau directory (they all have the same test split)
    test_log_dir = get_tau_log_dir(sizes[0])
    print(f"\nLoading test data from: {test_log_dir}")
    er_test = load_test_erank(test_log_dir)
    print(f"  Test erank: {len(er_test)} samples, mean={np.mean(er_test):.4f}, "
          f"std={np.std(er_test):.4f}")

    # ── Prepare CSV ───────────────────────────────────────────────────────
    csv_path = output_dir / 'wasserstein_quality_vs_tau.csv'
    csv_columns = [
        'N', 'target_tau', 'actual_tau', 'epoch_float', 'steps_per_epoch',
        'checkpoint_path', 'num_generated', 'num_train', 'num_test',
        'W1_Gen_Test', 'W1_Gen_Train', 'W1_Test_Train', 'Excess_Gap',
        'mean_erank_gen', 'std_erank_gen',
        'mean_erank_train', 'std_erank_train',
        'mean_erank_test', 'std_erank_test',
    ]

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)

    # ── Main loop: for each N ─────────────────────────────────────────────
    for n_train in sizes:
        log_dir = get_tau_log_dir(n_train)
        steps_per_epoch = ceil(n_train / BATCH_SIZE_TRAIN)

        print(f"\n{'=' * 70}")
        print(f"  N = {n_train}")
        print(f"  steps_per_epoch = {steps_per_epoch}")
        print(f"  Log dir: {log_dir}")
        print(f"{'=' * 70}")

        if not os.path.exists(log_dir):
            print(f"  WARNING: Directory not found, skipping N={n_train}")
            continue

        # Load training data erank for this N
        er_train = load_train_erank(log_dir)
        print(f"  Train erank: {len(er_train)} samples, mean={np.mean(er_train):.4f}")

        # Compute real-data baseline (constant for this N)
        w1_test_train = compute_wasserstein(er_test, er_train, n_select=None)
        print(f"  W1(Test, Train) baseline = {w1_test_train:.6f}")

        # Create model once, reload checkpoints
        ddim = create_model(DEVICE)

        for target_tau in tau_grid:
            # Find checkpoint
            ckpt_path = os.path.join(log_dir, 'checkpoints',
                                     f'checkpoint_tau_{target_tau}.pth')

            if not os.path.exists(ckpt_path):
                print(f"    τ={target_tau:>7d}: SKIP (checkpoint not found)")
                continue

            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            ddim.load_state_dict(ckpt['model_state_dict'])
            actual_tau = ckpt.get('global_step', target_tau)
            epoch_float = actual_tau / steps_per_epoch

            # Generate samples
            print(f"    τ={actual_tau:>7d} (epoch≈{epoch_float:.1f}): generating {num_gen} samples...", end='')
            samples = generate_samples(ddim, num_gen, BATCH_SIZE_GEN, DEVICE, seed=GEN_SEED)

            # Compute erank of generated samples
            er_gen = samples_to_erank(samples)

            # Compute Wasserstein distances
            w1_gen_test = compute_wasserstein(er_gen, er_test, n_select=None)
            w1_gen_train = compute_wasserstein(er_gen, er_train, n_select=None)
            excess_gap = w1_gen_test - w1_test_train

            print(f" W1(Gen,Test)={w1_gen_test:.4f}, "
                  f"W1(Gen,Train)={w1_gen_train:.4f}, "
                  f"Excess={excess_gap:.4f}")

            # Save generated channels
            gen_dir = os.path.join(log_dir, 'generated_tau')
            os.makedirs(gen_dir, exist_ok=True)
            gen_path = os.path.join(gen_dir, f'generated_tau_{actual_tau}.npz')
            np.savez_compressed(gen_path, channels=samples)

            # Append row to CSV
            row = [
                n_train, target_tau, actual_tau, epoch_float, steps_per_epoch,
                ckpt_path, num_gen, len(er_train), len(er_test),
                w1_gen_test, w1_gen_train, w1_test_train, excess_gap,
                float(np.mean(er_gen)), float(np.std(er_gen)),
                float(np.mean(er_train)), float(np.std(er_train)),
                float(np.mean(er_test)), float(np.std(er_test)),
            ]
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Computation complete!")
    print(f"  CSV saved to: {csv_path}")
    print(f"  Generated channels saved to: <log_dir>/generated_tau/")
    print(f"\n  To plot, run:")
    print(f"    python plot_w1_quality_vs_tau.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
