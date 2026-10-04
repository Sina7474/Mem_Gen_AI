"""
compute_fmem_vs_tau.py — Memorization Fraction f_mem vs Training Time τ
=========================================================================
For each dataset size N and each tau checkpoint, compute the nearest-neighbor
ratio test to estimate the fraction of generated samples that are memorized.

Metric: For each generated sample x_gen, find:
  d1 = distance to nearest training sample
  d2 = distance to second-nearest training sample
  rho = d1 / d2

A sample is classified as memorized if rho < k (default k=1/3).
f_mem = fraction of generated samples with rho < k.

Distance: L2 on flattened normalized real/imag beamspace tensors (dim=256).
This matches the representation used by the DDIM model.

Also computes:
  - f_mem at k=1/4 and k=1/2 for robustness
  - Bootstrap 95% CI for f_mem at k=1/3
  - Test-to-train baseline f_mem (false positive rate)
  - Summary statistics of the ratio distribution

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python compute_fmem_vs_tau.py

    # Debug mode (only a few sizes/taus):
    python compute_fmem_vs_tau.py --debug

    # Specific sizes:
    python compute_fmem_vs_tau.py --sizes 100 200
"""

import sys
import os
import csv
import argparse
from math import ceil
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         BEAMSPACE UTILITIES                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def dft_matrix(N: int) -> np.ndarray:
    """Unitary N×N DFT matrix."""
    n = np.arange(N)
    k = n.reshape(-1, 1)
    return np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)


def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    """Kronecker UPA DFT codebook."""
    return np.kron(dft_matrix(Ny), dft_matrix(Nx))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000, 8000]
LOGS_BASE = os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM', 'logs')
OUTPUT_DIR = Path("results/fmem_vs_tau")

# Tau grid (same as W1 quality evaluation)
GEN_EVAL_TAU_GRID = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
GEN_EVAL_TAU_GRID_DEBUG = [1000, 10000, 200000]

# Thresholds for memorization classification
K_VALUES = [1/4, 1/3, 1/2]
K_MAIN = 1/3

# Bootstrap settings
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 42

# Nearest-neighbor batch size (controls GPU memory usage)
NN_BATCH_SIZE = 512

# Training batch size (must match train_DDIM_tau.py)
BATCH_SIZE_TRAIN = 100

# UPA antenna dimensions (match training script)
NRX_X, NRX_Y = 2, 2
NTX_X, NTX_Y = 8, 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         HELPERS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_tau_log_dir(n_train):
    """Get the tau-based training log directory for a given N."""
    return os.path.join(LOGS_BASE, f"DDIM_tau_{n_train}_incremental")


def spatial_to_normalized_beamspace_flat(arr):
    """
    Convert spatial-domain array (N, 2, 4, 32) to flattened normalized beamspace.

    Steps:
      1. Reconstruct complex: H = arr[:,0] + 1j*arr[:,1]  -> (N, 4, 32)
      2. Beamspace transform: Hv = Ar^H @ H @ At         -> (N, 4, 32)
      3. Per-sample normalize: Hv / max(|Hv|)
      4. Split to real/imag and flatten                    -> (N, 256)
    """
    H = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)  # (N, 4, 32)

    # Beamspace transform
    Ar = upa_dft_codebook(NRX_X, NRX_Y)  # (4, 4)
    At = upa_dft_codebook(NTX_X, NTX_Y)  # (32, 32)
    Hv = np.matmul(np.matmul(Ar.conj().T, H), At)  # (N, 4, 32)

    # Per-sample normalization (same as training pipeline)
    mag = np.sqrt(np.real(Hv)**2 + np.imag(Hv)**2)  # (N, 4, 32)
    max_mag = mag.reshape(len(Hv), -1).max(axis=1, keepdims=True)[:, :, np.newaxis]  # (N,1,1)
    max_mag = np.where(max_mag > 1e-12, max_mag, 1.0)
    Hv = Hv / max_mag

    # Stack real/imag and flatten
    flat = np.concatenate([np.real(Hv), np.imag(Hv)], axis=1).reshape(len(Hv), -1)
    return flat.astype(np.float32)


def generated_to_flat(arr):
    """
    Convert generated beamspace array (N, 2, 4, 32) to flat (N, 256).
    Generated samples are already in normalized beamspace.
    """
    return arr.reshape(len(arr), -1).astype(np.float32)


def nearest_ratio_l2(gen, train, batch_size=512, device=None, eps=1e-12):
    """
    Compute nearest-neighbor ratio rho = d1/d2 for each generated sample.

    Parameters
    ----------
    gen   : torch.Tensor, shape (num_gen, dim)
    train : torch.Tensor, shape (num_train, dim)
    batch_size : chunk size to avoid OOM
    device : torch device

    Returns
    -------
    dict with keys: ratios, d1, d2, nearest_index, second_nearest_index
    """
    if device is None:
        device = torch.device('cpu')

    train = train.to(device)
    ratios = []
    d1_all = []
    d2_all = []
    nearest_indices = []
    second_indices = []

    for start in range(0, len(gen), batch_size):
        g = gen[start:start + batch_size].to(device)
        D = torch.cdist(g, train, p=2)  # (batch, num_train)

        top2 = torch.topk(D, k=2, largest=False, dim=1)
        d1 = top2.values[:, 0]
        d2 = top2.values[:, 1]
        idx1 = top2.indices[:, 0]
        idx2 = top2.indices[:, 1]

        rho = d1 / (d2 + eps)
        ratios.append(rho.cpu())
        d1_all.append(d1.cpu())
        d2_all.append(d2.cpu())
        nearest_indices.append(idx1.cpu())
        second_indices.append(idx2.cpu())

    return {
        "ratios": torch.cat(ratios).numpy(),
        "d1": torch.cat(d1_all).numpy(),
        "d2": torch.cat(d2_all).numpy(),
        "nearest_index": torch.cat(nearest_indices).numpy(),
        "second_nearest_index": torch.cat(second_indices).numpy(),
    }


def compute_fmem_bootstrap(is_mem, B=1000, seed=42):
    """Compute f_mem with bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    n = len(is_mem)
    f_mem = is_mem.mean()

    boot_fmem = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_fmem[b] = is_mem[idx].mean()

    ci_low = np.percentile(boot_fmem, 2.5)
    ci_high = np.percentile(boot_fmem, 97.5)
    return f_mem, ci_low, ci_high


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Compute memorization fraction f_mem vs tau for all dataset sizes")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: fewer sizes, coarser grid")
    parser.add_argument("--sizes", type=int, nargs='+', default=None,
                        help="Specific sizes to evaluate (default: all)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--batch_size", type=int, default=NN_BATCH_SIZE,
                        help="Batch size for nearest-neighbor computation")
    args = parser.parse_args()

    # Select configuration
    if args.debug:
        tau_grid = GEN_EVAL_TAU_GRID_DEBUG
        print("*** DEBUG MODE: coarse grid ***")
    else:
        tau_grid = GEN_EVAL_TAU_GRID

    sizes = args.sizes if args.sizes else TRAIN_SIZES
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios_dir = output_dir / "nearest_neighbor_ratios"
    ratios_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Memorization Fraction f_mem vs Training Time τ")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Sizes: {sizes}")
    print(f"  Tau grid: {tau_grid}")
    print(f"  Thresholds k: {K_VALUES}")
    print(f"  Bootstrap: B={BOOTSTRAP_B}, seed={BOOTSTRAP_SEED}")
    print(f"  NN batch size: {args.batch_size}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # ── Prepare CSV ───────────────────────────────────────────────────────
    csv_path = output_dir / 'fmem_vs_tau.csv'
    csv_columns = [
        'N', 'target_tau', 'actual_tau', 'epoch_float', 'steps_per_epoch',
        'checkpoint_path', 'generated_file', 'num_generated', 'num_train',
        'distance_type', 'k_main',
        'f_mem_k_1_3', 'f_mem_k_1_3_ci_low', 'f_mem_k_1_3_ci_high',
        'f_mem_k_1_4', 'f_mem_k_1_2',
        'mean_ratio', 'median_ratio', 'p05_ratio', 'p95_ratio',
        'mean_d1', 'mean_d2', 'median_d1', 'median_d2',
        'f_mem_test_baseline_k_1_3',
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

        # ── Load and preprocess training data ─────────────────────────────
        train_path = os.path.join(log_dir, 'train.npy')
        if not os.path.exists(train_path):
            print(f"  WARNING: train.npy not found, skipping N={n_train}")
            continue

        train_spatial = np.load(train_path)  # (N, 2, 4, 32) spatial domain
        train_flat = spatial_to_normalized_beamspace_flat(train_spatial)  # (N, 256)
        train_tensor = torch.from_numpy(train_flat)
        print(f"  Training data: {train_flat.shape}")

        # ── Compute test baseline f_mem ───────────────────────────────────
        test_path = os.path.join(log_dir, 'test.npy')
        test_baseline = np.nan
        if os.path.exists(test_path):
            test_spatial = np.load(test_path)  # (4086, 2, 4, 32) spatial domain
            test_flat = spatial_to_normalized_beamspace_flat(test_spatial)  # (4086, 256)
            test_tensor = torch.from_numpy(test_flat)

            print(f"  Computing test baseline (test vs train)...")
            test_result = nearest_ratio_l2(
                test_tensor, train_tensor,
                batch_size=args.batch_size, device=DEVICE
            )
            test_is_mem = test_result['ratios'] < K_MAIN
            test_baseline = float(test_is_mem.mean())
            print(f"  Test baseline f_mem(k=1/3): {100*test_baseline:.2f}%")
        else:
            print(f"  WARNING: test.npy not found, skipping baseline")

        # ── For each tau checkpoint ───────────────────────────────────────
        for target_tau in tau_grid:
            # Load generated samples (already created by compute_w1_quality_vs_tau.py)
            gen_path = os.path.join(log_dir, 'generated_tau',
                                    f'generated_tau_{target_tau}.npz')
            ckpt_path = os.path.join(log_dir, 'checkpoints',
                                     f'checkpoint_tau_{target_tau}.pth')

            if not os.path.exists(gen_path):
                print(f"    τ={target_tau:>7d}: SKIP (generated file not found)")
                continue

            # Load generated channels
            gen_data = np.load(gen_path)
            gen_arr = gen_data['channels']  # (5000, 2, 4, 32) normalized beamspace
            gen_flat = generated_to_flat(gen_arr)  # (5000, 256)
            gen_tensor = torch.from_numpy(gen_flat)

            # Get actual tau from checkpoint if available
            actual_tau = target_tau
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                actual_tau = ckpt.get('global_step', target_tau)
                del ckpt

            epoch_float = actual_tau / steps_per_epoch

            # Compute nearest-neighbor ratios
            print(f"    τ={actual_tau:>7d} (epoch≈{epoch_float:.1f}): "
                  f"{len(gen_flat)} gen × {len(train_flat)} train...", end='')

            result = nearest_ratio_l2(
                gen_tensor, train_tensor,
                batch_size=args.batch_size, device=DEVICE
            )
            ratios = result['ratios']

            # Compute f_mem for all thresholds
            fmem_dict = {}
            for k in K_VALUES:
                is_mem = ratios < k
                fmem_dict[k] = float(is_mem.mean())

            # Bootstrap CI for k=1/3
            is_mem_main = ratios < K_MAIN
            f_mem_main, ci_low, ci_high = compute_fmem_bootstrap(
                is_mem_main, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED
            )

            # Summary statistics
            mean_ratio = float(ratios.mean())
            median_ratio = float(np.median(ratios))
            p05_ratio = float(np.percentile(ratios, 5))
            p95_ratio = float(np.percentile(ratios, 95))
            mean_d1 = float(result['d1'].mean())
            mean_d2 = float(result['d2'].mean())
            median_d1 = float(np.median(result['d1']))
            median_d2 = float(np.median(result['d2']))

            print(f" f_mem(k=1/3)={100*f_mem_main:.2f}% "
                  f"[{100*ci_low:.2f}%, {100*ci_high:.2f}%]")

            # Save per-(N,tau) ratio details
            ratio_file = ratios_dir / f"ratios_N_{n_train}_tau_{actual_tau}.npz"
            np.savez_compressed(
                ratio_file,
                ratios=ratios,
                d1=result['d1'],
                d2=result['d2'],
                nearest_index=result['nearest_index'],
                second_nearest_index=result['second_nearest_index'],
                is_mem_k_1_3=is_mem_main,
            )

            # Append row to CSV
            row = [
                n_train, target_tau, int(actual_tau), epoch_float, steps_per_epoch,
                ckpt_path, gen_path, len(gen_flat), len(train_flat),
                'L2_normalized_beamspace', K_MAIN,
                f_mem_main, ci_low, ci_high,
                fmem_dict[1/4], fmem_dict[1/2],
                mean_ratio, median_ratio, p05_ratio, p95_ratio,
                mean_d1, mean_d2, median_d1, median_d2,
                test_baseline,
            ]

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    print(f"\n{'=' * 70}")
    print(f"DONE. Results saved to: {csv_path}")
    print(f"Ratio details saved to: {ratios_dir}/")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
