"""
compute_w1_quality_errorbars.py — W1 quality error bars via 5 reference subsets
=================================================================================
Generates 5,000 samples per checkpoint and computes W1(Gen, Ref) against each
of 5 disjoint reference subsets of 5,000 real channels drawn from the
never-used portion of the training pool (shuffled positions 4000–29000).

Error bar = 2 × std over the 5 W1 values (mean ± 2σ), matching the paper's
(arXiv:2505.17638v2) "5 different test sets" methodology for quality metrics.

Previous results and code are NOT touched. All output goes to new files
under DDIM_Evaluation/results/w1_quality_errorbars/.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python compute_w1_quality_errorbars.py

    # Specific model widths / dataset sizes:
    python compute_w1_quality_errorbars.py --n_feats 64 256 --sizes 100 1000

    # Quick smoke-test:
    python compute_w1_quality_errorbars.py --debug
"""

import sys
import os
import csv
import math
import argparse
from pathlib import Path
from math import ceil

import numpy as np
import torch
import importlib.util
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths & imports
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DDIM_DIR   = os.path.join(_SCRIPT_DIR, '..', 'Code', 'DDIM_FMM')
_ERANK_DIR  = os.path.join(_SCRIPT_DIR, '..', 'cDDIM_cFMM_Repo',
                            'Effective_Rank_Github')

sys.path.insert(0, _DDIM_DIR)
sys.path.insert(0, _ERANK_DIR)

from effective_rank_core import (compute_effective_rank, compute_wasserstein,
                                  normalize_max_abs, to_beamspace)

# Import Unet + DDIM from train_DDIM_tau.py (guarantees architecture match)
_train_path = os.path.join(_DDIM_DIR, 'train_DDIM_tau.py')
_spec = importlib.util.spec_from_file_location("_train_mod", _train_path)
_train_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
Unet = _train_mod.Unet
DDIM = _train_mod.DDIM


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_FEAT_GRID  = [64, 128, 256, 512]
TRAIN_SIZES  = [100, 200, 500, 1000, 2000, 4000]
LOGS_BASE    = os.path.join(_DDIM_DIR, 'logs')

# Shared indices may be in shared_indices/ or fall through to any incremental dir
def _find_shared_idx_dir() -> str:
    """Return the directory that holds indices_los.npy / indices_nlos.npy / indices.npy."""
    candidates = [
        os.path.join(LOGS_BASE, 'shared_indices'),
        # Fallback: any incremental training directory (all use same indices)
        os.path.join(LOGS_BASE, 'DDIM_tau_1000_incremental'),
        os.path.join(LOGS_BASE, 'DDIM_tau_4000_incremental'),
        os.path.join(LOGS_BASE, 'DDIM_tau_100_incremental'),
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, 'indices_los.npy')):
            return d
    raise FileNotFoundError(
        "Cannot find indices_los.npy in any expected directory. "
        "Run train_DDIM_tau.py first to generate the shared shuffled indices.")

SHARED_IDX = _find_shared_idx_dir()
OUTPUT_DIR   = Path("results/w1_quality_errorbars")

# Dataset files
DATA_PATH_LOS  = os.path.join(_SCRIPT_DIR, '..', 'dataset',
                               'Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz')
DATA_PATH_NLOS = os.path.join(_SCRIPT_DIR, '..', 'dataset',
                               'Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz')
SUBCARRIER_IDX = 128

# Reference pool: positions [4000 : 29000] of the combined shuffled order
# (all positions >= 4000 so they were never used in any training run with N≤4000)
N_TRAIN_MAX   = 4000   # largest training set used — defines the reference pool start
N_REF_EACH    = 5000   # samples per reference subset
N_REF_SETS    = 5      # number of disjoint reference subsets

# Generated samples per checkpoint
N_GEN         = 5000
BATCH_SIZE_GEN = 100
GEN_SEED      = 0      # fixed seed for reproducibility

# Tau grid for generation-quality evaluation
GEN_EVAL_TAU_GRID = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
GEN_EVAL_TAU_GRID_DEBUG = [1000, 50000, 200000]

# UPA antenna dimensions for reference pool conversion
# (matches the convention in compute_w1_quality_vs_tau.py and effective_rank_core)
ANTENNA = dict(Nrx_x=2, Nrx_y=2, Ntx_x=4, Ntx_y=8)

BETAS  = (1e-4, 0.02)
N_T    = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Reference pool construction
# ─────────────────────────────────────────────────────────────────────────────

def build_reference_pool() -> np.ndarray:
    """
    Reconstruct the combined shuffled dataset using the shared indices from
    training, then extract positions [N_TRAIN_MAX : N_TRAIN_MAX + N_REF_SETS*N_REF_EACH]
    as the reference pool for W1 quality evaluation.

    Returns
    -------
    pool_H : (N_REF_SETS * N_REF_EACH, 4, 32) complex64
        Complex channel matrices in antenna domain, ready for `to_beamspace`.
    """
    print("\nBuilding reference pool from raw dataset ...")

    # Load raw arrays
    print(f"  Loading LoS  : {os.path.basename(DATA_PATH_LOS)}")
    raw_los  = np.load(DATA_PATH_LOS)['combined_array']
    print(f"  Loading NLoS : {os.path.basename(DATA_PATH_NLOS)}")
    raw_nlos = np.load(DATA_PATH_NLOS)['combined_array']

    # Same indexing as in train_DDIM_tau.py's CustomSionnaDataset.load_dataset()
    data_los  = raw_los [:, :, 0, :, 0, SUBCARRIER_IDX, :]   # (N_LoS, 4, 32, ?)
    data_nlos = raw_nlos[:, :, 0, :, 0, SUBCARRIER_IDX, :]   # (N_NLoS, 4, 32, ?)
    print(f"  LoS  shape after indexing: {data_los.shape}")
    print(f"  NLoS shape after indexing: {data_nlos.shape}")

    # Apply the same shuffled indices used during training
    idx_los  = np.load(os.path.join(SHARED_IDX, 'indices_los.npy'))
    idx_nlos = np.load(os.path.join(SHARED_IDX, 'indices_nlos.npy'))
    idx_comb = np.load(os.path.join(SHARED_IDX, 'indices.npy'))

    data_los  = data_los [idx_los]
    data_nlos = data_nlos[idx_nlos]
    data_all  = np.concatenate([data_los, data_nlos], axis=0)
    data_all  = data_all[idx_comb]   # combined shuffle

    total = len(data_all)
    pool_end = N_TRAIN_MAX + N_REF_SETS * N_REF_EACH
    if pool_end > total:
        raise RuntimeError(
            f"Not enough data: need {pool_end} samples but only {total} available.")

    print(f"  Total samples (shuffled): {total}")
    print(f"  Reference pool           : [{N_TRAIN_MAX} : {pool_end}]"
          f" = {pool_end - N_TRAIN_MAX} samples")

    # Extract complex channel matrices from the reference pool
    pool_raw = data_all[N_TRAIN_MAX:pool_end]   # (..., 4, 32, ?)
    pool_H   = pool_raw[:, :, :, 0].astype(np.complex64)   # (pool_size, 4, 32)
    return pool_H


def precompute_ref_eranks(pool_H: np.ndarray):
    """
    Convert each of the 5 reference subsets to normalized beamspace and
    compute their per-sample effective rank distributions.

    Returns
    -------
    er_refs : list of 5 arrays, each of shape (N_REF_EACH,)
    """
    print("\nPrecomputing effective rank for 5 reference subsets ...")
    er_refs = []
    for i in range(N_REF_SETS):
        H_sub  = pool_H[i * N_REF_EACH : (i + 1) * N_REF_EACH]   # (5000, 4, 32)
        H_beam = to_beamspace(H_sub, **ANTENNA)
        H_norm = normalize_max_abs(H_beam)
        er, _  = compute_effective_rank(H_norm, use_power=True)
        er_refs.append(er)
        print(f"  Subset {i+1}: mean_erank={np.mean(er):.4f}")
    return er_refs


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_log_dir(n_train: int, n_feat: int) -> str:
    nfeat_suffix = f'_nfeat{n_feat}' if n_feat != 256 else ''
    return os.path.join(LOGS_BASE, f"DDIM_tau_{n_train}{nfeat_suffix}_incremental")


def create_model(n_feat: int) -> 'DDIM':
    nn_model = Unet(in_channels=2, n_feat=n_feat)
    ddim = DDIM(nn_model=nn_model, betas=BETAS, n_T=N_T, device=DEVICE)
    ddim.to(DEVICE)
    return ddim


def generate_samples(ddim, n_samples: int, seed: int = GEN_SEED) -> np.ndarray:
    """Generate n_samples from model, return (n_samples, 2, 4, 32) numpy."""
    torch.manual_seed(seed)
    ddim.eval()
    all_samples = []
    with torch.no_grad():
        while sum(s.shape[0] for s in all_samples) < n_samples:
            bs = min(BATCH_SIZE_GEN, n_samples - sum(s.shape[0] for s in all_samples))
            s  = ddim.sample(bs, (2, 4, 32), DEVICE)
            all_samples.append(s.cpu().numpy())
    return np.concatenate(all_samples, axis=0)[:n_samples]


def samples_to_erank(samples: np.ndarray) -> np.ndarray:
    """
    Generated samples (N, 2, 4, 32) are already in normalized beamspace.
    Just re-complexify, re-normalize, and compute effective rank.
    """
    H     = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64)
    H     = normalize_max_abs(H)
    er, _ = compute_effective_rank(H, use_power=True)
    return er


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="W1 quality error bars via 5 disjoint reference subsets")
    parser.add_argument("--n_feats", type=int, nargs='+', default=N_FEAT_GRID)
    parser.add_argument("--sizes",   type=int, nargs='+', default=TRAIN_SIZES)
    parser.add_argument("--debug",   action="store_true",
                        help="Quick smoke-test: 1 size, 3 taus")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    tau_grid = GEN_EVAL_TAU_GRID_DEBUG if args.debug else GEN_EVAL_TAU_GRID
    n_feats  = args.n_feats
    sizes    = [args.sizes[0]] if args.debug else args.sizes
    out_dir  = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("W1 quality error bars (5 disjoint reference subsets of 5000 each)")
    print("=" * 70)
    print(f"  Device      : {DEVICE}")
    print(f"  n_feats     : {n_feats}")
    print(f"  sizes       : {sizes}")
    print(f"  tau grid    : {tau_grid}")
    print(f"  N_GEN       : {N_GEN}")
    print(f"  N_REF_EACH  : {N_REF_EACH}  (× {N_REF_SETS} subsets)")
    print(f"  ref pool    : [{N_TRAIN_MAX} : {N_TRAIN_MAX + N_REF_SETS*N_REF_EACH}]")
    print(f"  output      : {out_dir}")
    print("=" * 70)

    # ── Build reference pool once (shared across all models) ─────────────────
    pool_H  = build_reference_pool()
    er_refs = precompute_ref_eranks(pool_H)

    # ── Combined CSV ─────────────────────────────────────────────────────────
    all_csv_path = out_dir / 'w1_quality_errorbars_all.csv'
    all_cols = [
        'N', 'n_feat', 'target_tau', 'actual_tau', 'epoch_float', 'steps_per_epoch',
        'num_generated', 'num_ref_each',
        'W1_mean', 'W1_2std',
        'W1_s1', 'W1_s2', 'W1_s3', 'W1_s4', 'W1_s5',
        'mean_erank_gen', 'std_erank_gen',
    ]
    with open(all_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(all_cols)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for n_feat in n_feats:
        print(f"\n{'='*70}")
        print(f"  n_feat = {n_feat}")
        print(f"{'='*70}")
        ddim = create_model(n_feat)

        per_csv_path = out_dir / f'w1_quality_errorbars_nfeat{n_feat}.csv'
        with open(per_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['N'] + all_cols[2:])

        for n_train in sizes:
            log_dir = get_log_dir(n_train, n_feat)
            if not os.path.exists(log_dir):
                print(f"  N={n_train}: log dir missing, skip")
                continue

            steps_per_epoch = ceil(n_train / 100)  # BATCH_SIZE = 100
            print(f"\n  N = {n_train}  (steps/epoch = {steps_per_epoch})")

            for target_tau in tqdm(tau_grid, desc=f"    N={n_train}  taus"):
                ckpt_path = os.path.join(log_dir, 'checkpoints',
                                         f'checkpoint_tau_{target_tau}.pth')
                if not os.path.exists(ckpt_path):
                    continue

                # Load checkpoint
                ckpt = torch.load(ckpt_path, map_location=DEVICE,
                                  weights_only=False)
                ddim.load_state_dict(ckpt['model_state_dict'])
                actual_tau  = int(ckpt.get('global_step', target_tau))
                epoch_float = actual_tau / steps_per_epoch

                # Generate 5000 samples
                samples = generate_samples(ddim, N_GEN, seed=GEN_SEED)
                er_gen  = samples_to_erank(samples)   # (N_GEN,)

                # W1 vs each reference subset
                w1_vals = [float(compute_wasserstein(er_gen, er_ref, n_select=None))
                           for er_ref in er_refs]

                w1_mean  = float(np.mean(w1_vals))
                w1_2std  = float(2 * np.std(w1_vals, ddof=1))

                row_common = [
                    target_tau, actual_tau, epoch_float, steps_per_epoch,
                    N_GEN, N_REF_EACH,
                    w1_mean, w1_2std,
                    *w1_vals,
                    float(np.mean(er_gen)), float(np.std(er_gen)),
                ]

                with open(per_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([n_train] + row_common)
                with open(all_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([n_train, n_feat] + row_common)

    print(f"\n{'='*70}")
    print("Done!")
    print(f"  Combined CSV : {all_csv_path}")
    print(f"  Per-n_feat   : {out_dir}/w1_quality_errorbars_nfeat{{W}}.csv")
    print(f"\nTo plot, run:")
    print(f"  python plot_errorbars_combined.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
