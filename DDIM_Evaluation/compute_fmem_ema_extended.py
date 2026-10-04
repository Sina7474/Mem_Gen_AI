"""
compute_fmem_ema_extended.py — Compute f_mem for extended EMA tau points
=========================================================================
Appends fmem rows to the existing fmem_vs_tau.csv for N=4000 at tau points
that only exist in the EMA training (250k, 300k, 350k, 400k).

Uses the same nearest-neighbor ratio metric as compute_fmem_vs_tau.py, but
reads generated samples from the EMA cache (generated_ema/) instead.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python compute_fmem_ema_extended.py
"""

import os
import csv
from math import ceil
from pathlib import Path
import numpy as np
import torch

# ─── Configuration ────────────────────────────────────────────────────────────
N_TRAIN = 4000
BATCH_SIZE_TRAIN = 100
STEPS_PER_EPOCH = ceil(N_TRAIN / BATCH_SIZE_TRAIN)  # 40

EMA_LOG_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM', 'logs_ema',
    'DDIM_tau_ema_4000_incremental'
)
GENERATED_DIR = os.path.join(EMA_LOG_DIR, 'generated_ema')

# Tau points to compute (the extended ones not in the original fmem CSV)
TAU_POINTS = [250000, 300000, 350000, 400000]

# Memorization thresholds
K_VALUES = [1/4, 1/3, 1/2]
K_MAIN = 1/3

# Bootstrap
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 42

# NN batch
NN_BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output
OUTPUT_CSV = Path("results/fmem_vs_tau/fmem_vs_tau.csv")

# ─── Beamspace utilities ──────────────────────────────────────────────────────
def dft_matrix(N):
    n = np.arange(N)
    k = n.reshape(-1, 1)
    return np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)

def upa_dft_codebook(Nx, Ny):
    return np.kron(dft_matrix(Ny), dft_matrix(Nx))

NRX_X, NRX_Y = 2, 2
NTX_X, NTX_Y = 8, 4
_AR = upa_dft_codebook(NRX_X, NRX_Y)
_AT = upa_dft_codebook(NTX_X, NTX_Y)

def spatial_to_normalized_beamspace_flat(arr):
    H = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
    Hv = np.matmul(np.matmul(_AR.conj().T, H), _AT)
    mag = np.sqrt(np.real(Hv)**2 + np.imag(Hv)**2)
    max_mag = mag.reshape(len(Hv), -1).max(axis=1, keepdims=True)[:, :, np.newaxis]
    max_mag = np.where(max_mag > 1e-12, max_mag, 1.0)
    Hv = Hv / max_mag
    flat = np.concatenate([np.real(Hv), np.imag(Hv)], axis=1).reshape(len(Hv), -1)
    return flat.astype(np.float32)

def generated_to_flat(arr):
    return arr.reshape(len(arr), -1).astype(np.float32)

# ─── NN ratio ─────────────────────────────────────────────────────────────────
def nearest_ratio_l2(gen, train, batch_size=512, device=None, eps=1e-12):
    if device is None:
        device = torch.device('cpu')
    train = train.to(device)
    ratios, d1_all, d2_all = [], [], []
    nearest_indices, second_indices = [], []
    for start in range(0, len(gen), batch_size):
        g = gen[start:start + batch_size].to(device)
        D = torch.cdist(g, train, p=2)
        top2 = torch.topk(D, k=2, largest=False, dim=1)
        d1 = top2.values[:, 0]
        d2 = top2.values[:, 1]
        ratios.append((d1 / (d2 + eps)).cpu())
        d1_all.append(d1.cpu())
        d2_all.append(d2.cpu())
        nearest_indices.append(top2.indices[:, 0].cpu())
        second_indices.append(top2.indices[:, 1].cpu())
    return {
        "ratios": torch.cat(ratios).numpy(),
        "d1": torch.cat(d1_all).numpy(),
        "d2": torch.cat(d2_all).numpy(),
        "nearest_index": torch.cat(nearest_indices).numpy(),
        "second_nearest_index": torch.cat(second_indices).numpy(),
    }

def compute_fmem_bootstrap(is_mem, B=1000, seed=42):
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

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Compute f_mem for EMA-extended tau points (N=4000)")
    print("=" * 70)
    print(f"  EMA log dir: {EMA_LOG_DIR}")
    print(f"  Tau points: {TAU_POINTS}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # Load training data
    train_path = os.path.join(EMA_LOG_DIR, 'train.npy')
    train_spatial = np.load(train_path)
    train_flat = spatial_to_normalized_beamspace_flat(train_spatial)
    train_tensor = torch.from_numpy(train_flat)
    print(f"  Training data: {train_flat.shape}")

    # Compute test baseline
    test_path = os.path.join(EMA_LOG_DIR, 'test.npy')
    test_baseline = np.nan
    if os.path.exists(test_path):
        test_spatial = np.load(test_path)
        test_flat = spatial_to_normalized_beamspace_flat(test_spatial)
        test_tensor = torch.from_numpy(test_flat)
        test_result = nearest_ratio_l2(test_tensor, train_tensor,
                                       batch_size=NN_BATCH_SIZE, device=DEVICE)
        test_is_mem = test_result['ratios'] < K_MAIN
        test_baseline = float(test_is_mem.mean())
        print(f"  Test baseline f_mem(k=1/3): {100*test_baseline:.2f}%")

    # Process each tau point
    for tau in TAU_POINTS:
        gen_path = os.path.join(GENERATED_DIR, f'gen_ema_tau{tau}_seed0.npz')
        ckpt_path = os.path.join(EMA_LOG_DIR, 'checkpoints',
                                 f'checkpoint_tau_{tau}.pth')

        if not os.path.exists(gen_path):
            print(f"  τ={tau}: SKIP (no generated samples)")
            continue

        gen_arr = np.load(gen_path)['channels']
        gen_flat = generated_to_flat(gen_arr)
        gen_tensor = torch.from_numpy(gen_flat)

        epoch_float = tau / STEPS_PER_EPOCH

        print(f"  τ={tau:>7d} (epoch≈{epoch_float:.1f}): "
              f"{len(gen_flat)} gen × {len(train_flat)} train...", end='')

        result = nearest_ratio_l2(gen_tensor, train_tensor,
                                  batch_size=NN_BATCH_SIZE, device=DEVICE)
        ratios = result['ratios']

        # f_mem for all thresholds
        fmem_dict = {}
        for k in K_VALUES:
            fmem_dict[k] = float((ratios < k).mean())

        # Bootstrap CI for k=1/3
        is_mem_main = ratios < K_MAIN
        f_mem_main, ci_low, ci_high = compute_fmem_bootstrap(
            is_mem_main, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)

        # Stats
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

        # Append row to existing CSV
        row = [
            N_TRAIN, tau, int(tau), epoch_float, STEPS_PER_EPOCH,
            ckpt_path, gen_path, len(gen_flat), len(train_flat),
            'L2_normalized_beamspace', K_MAIN,
            f_mem_main, ci_low, ci_high,
            fmem_dict[1/4], fmem_dict[1/2],
            mean_ratio, median_ratio, p05_ratio, p95_ratio,
            mean_d1, mean_d2, median_d1, median_d2,
            test_baseline,
        ]

        with open(OUTPUT_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

    print(f"\n{'=' * 70}")
    print(f"DONE. Appended to: {OUTPUT_CSV}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
