"""
compute_loss_errorbars.py — L_train / L_test with 5-noise-seed error bars
=========================================================================
Offline evaluation that loads existing tau checkpoints and evaluates the
denoising loss at fixed diffusion time t_eval with 5 independent noise
realizations. Reports mean ± 2×std across the 5 seeds, matching the
paper's (arXiv:2505.17638v2) error bar methodology for losses.

Previous results and code are NOT touched. All output goes to new files
under DDIM_Evaluation/results/loss_errorbars/.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python compute_loss_errorbars.py

    # Specific model widths / dataset sizes:
    python compute_loss_errorbars.py --n_feats 64 256 --sizes 100 1000

    # Quick smoke-test (2 taus, one size):
    python compute_loss_errorbars.py --debug
"""

import sys
import os
import csv
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import importlib.util
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths & imports
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DDIM_DIR   = os.path.join(_SCRIPT_DIR, '..', 'Code', 'DDIM_FMM')

# Import Unet + DDIM from train_DDIM_tau.py via importlib so the architecture
# is guaranteed to be identical to the one used during training.
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
OUTPUT_DIR   = Path("results/loss_errorbars")

# All tau checkpoints saved during training (tau=0 is the initial model)
TAU_GRID = [0, 100, 200, 500, 1000, 2000, 5000, 10000,
            20000, 50000, 100000, 200000]

TAU_GRID_DEBUG = [100, 10000, 200000]

# 5 noise realizations — seeds fixed for reproducibility across all runs
NOISE_SEEDS = [12345, 12346, 12347, 12348, 12349]

# Fixed evaluation timestep (corresponds to alpha_bar ≈ 0.9802 for n_T=200,
# betas=(1e-4, 0.02)). Pre-computed; matches every training run.
T_EVAL    = 19
N_T       = 200
BETAS     = (1e-4, 0.02)
EVAL_BATCH = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Beamspace utilities  (copied from train_DDIM_tau.py — TRAINING convention)
# Ntx_x=8, Ntx_y=4  (same order used during model training)
# ─────────────────────────────────────────────────────────────────────────────

def dft_matrix(N: int) -> np.ndarray:
    n = np.arange(N)
    k = n.reshape(-1, 1)
    return np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)


def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    return np.kron(dft_matrix(Ny), dft_matrix(Nx))


def spatial_to_model_input(arr: np.ndarray) -> np.ndarray:
    """
    Convert spatial-domain data saved by train_DDIM_tau.py to the
    normalized beamspace representation seen by the model.

    Parameters
    ----------
    arr : (N, 2, 4, 32)  float32, real/imag stacked, spatial domain

    Returns
    -------
    (N, 2, 4, 32)  float32, normalized beamspace (model input format)
    """
    H = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)   # (N, 4, 32)

    # Training beamspace transform  (Nrx_x=2, Nrx_y=2, Ntx_x=8, Ntx_y=4)
    Ar = upa_dft_codebook(2, 2)   # (4, 4)
    At = upa_dft_codebook(8, 4)   # (32, 32)
    Hv = np.matmul(np.matmul(Ar.conj().T, H), At)             # (N, 4, 32)

    # Per-sample max-magnitude normalization
    mag     = np.sqrt(np.real(Hv)**2 + np.imag(Hv)**2)
    max_mag = mag.reshape(len(Hv), -1).max(axis=1)[:, None, None]
    max_mag = np.where(max_mag > 1e-12, max_mag, 1.0)
    Hv      = Hv / max_mag

    out = np.stack([np.real(Hv), np.imag(Hv)], axis=1)
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_log_dir(n_train: int, n_feat: int) -> str:
    nfeat_suffix = f'_nfeat{n_feat}' if n_feat != 256 else ''
    return os.path.join(LOGS_BASE, f"DDIM_tau_{n_train}{nfeat_suffix}_incremental")


def create_model(n_feat: int) -> 'DDIM':
    nn_model = Unet(in_channels=2, n_feat=n_feat)
    ddim = DDIM(nn_model=nn_model, betas=BETAS, n_T=N_T, device=DEVICE)
    ddim.to(DEVICE)
    return ddim


def evaluate_loss_batched(ddim, data_tensor: torch.Tensor,
                          eval_noise: torch.Tensor,
                          t_eval: int) -> float:
    """
    Evaluate denoising MSE over all samples at fixed t_eval with given noise.
    Runs in batches to respect GPU memory.
    """
    ddim.eval()
    total = 0.0
    n = data_tensor.shape[0]
    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH):
            end  = min(start + EVAL_BATCH, n)
            x    = data_tensor[start:end].to(DEVICE)
            eps  = eval_noise[start:end].to(DEVICE)
            loss = ddim.evaluate_denoising_loss(x, t_eval, eps)
            total += loss.item() * (end - start)
    return total / n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="L_train/L_test error bars via 5 noise realizations")
    parser.add_argument("--n_feats", type=int, nargs='+', default=N_FEAT_GRID)
    parser.add_argument("--sizes",   type=int, nargs='+', default=TRAIN_SIZES)
    parser.add_argument("--debug",   action="store_true",
                        help="Quick smoke-test: 1 size, 3 taus")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    tau_grid  = TAU_GRID_DEBUG if args.debug else TAU_GRID
    n_feats   = args.n_feats
    sizes     = [args.sizes[0]] if args.debug else args.sizes
    out_dir   = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("L_train / L_test error bars (5 noise realizations)")
    print("=" * 70)
    print(f"  Device  : {DEVICE}")
    print(f"  n_feats : {n_feats}")
    print(f"  sizes   : {sizes}")
    print(f"  tau grid: {tau_grid}")
    print(f"  seeds   : {NOISE_SEEDS}")
    print(f"  t_eval  : {T_EVAL}  (alpha_bar ≈ 0.9802)")
    print(f"  output  : {out_dir}")
    print("=" * 70)

    # ── Combined CSV ─────────────────────────────────────────────────────────
    all_csv_path = out_dir / 'loss_errorbars_all.csv'
    all_cols = [
        'N', 'n_feat', 'target_tau', 'actual_tau', 'epoch_float', 'steps_per_epoch',
        'num_train', 'num_test',
        'L_train_mean', 'L_train_2std',
        'L_test_mean',  'L_test_2std',
        'L_gap_mean',   'L_gap_2std',
        'L_train_s1', 'L_train_s2', 'L_train_s3', 'L_train_s4', 'L_train_s5',
        'L_test_s1',  'L_test_s2',  'L_test_s3',  'L_test_s4',  'L_test_s5',
    ]
    with open(all_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(all_cols)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for n_feat in n_feats:
        print(f"\n{'='*70}")
        print(f"  n_feat = {n_feat}")
        print(f"{'='*70}")
        ddim = create_model(n_feat)

        # Per-(N, n_feat) CSV
        per_csv_path = out_dir / f'loss_errorbars_nfeat{n_feat}.csv'
        per_cols = ['N'] + all_cols[2:]  # skip N, n_feat (already in filename)
        with open(per_csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['N'] + all_cols[2:])

        for n_train in sizes:
            log_dir = get_log_dir(n_train, n_feat)
            if not os.path.exists(log_dir):
                print(f"  N={n_train}: log dir missing, skip")
                continue

            train_npy = os.path.join(log_dir, 'train.npy')
            test_npy  = os.path.join(log_dir, 'test.npy')
            if not os.path.exists(train_npy) or not os.path.exists(test_npy):
                print(f"  N={n_train}: train.npy/test.npy missing, skip")
                continue

            # Load spatial data and convert to model input
            print(f"\n  N = {n_train}")
            train_spatial = np.load(train_npy)   # (N, 2, 4, 32) spatial
            test_spatial  = np.load(test_npy)    # (4086, 2, 4, 32) spatial
            train_model = spatial_to_model_input(train_spatial)
            test_model  = spatial_to_model_input(test_spatial)
            train_eval  = torch.from_numpy(train_model)   # (N, 2, 4, 32) float32
            test_eval   = torch.from_numpy(test_model)    # (4086, 2, 4, 32) float32
            steps_per_epoch = math.ceil(n_train / 100)    # BATCH_SIZE = 100

            print(f"    train shape: {train_eval.shape}, "
                  f"test shape: {test_eval.shape}")

            for target_tau in tqdm(tau_grid, desc=f"    N={n_train}  taus"):
                ckpt_path = os.path.join(log_dir, 'checkpoints',
                                         f'checkpoint_tau_{target_tau}.pth')
                if not os.path.exists(ckpt_path):
                    continue

                ckpt = torch.load(ckpt_path, map_location=DEVICE,
                                  weights_only=False)
                ddim.load_state_dict(ckpt['model_state_dict'])
                actual_tau  = int(ckpt.get('global_step', target_tau))
                epoch_float = actual_tau / steps_per_epoch

                L_trains, L_tests = [], []
                for seed in NOISE_SEEDS:
                    g = torch.Generator()
                    g.manual_seed(seed)
                    train_noise = torch.randn(train_eval.shape, generator=g)
                    test_noise  = torch.randn(test_eval.shape,  generator=g)

                    L_tr = evaluate_loss_batched(ddim, train_eval,
                                                 train_noise, T_EVAL)
                    L_te = evaluate_loss_batched(ddim, test_eval,
                                                 test_noise,  T_EVAL)
                    L_trains.append(L_tr)
                    L_tests.append(L_te)

                L_trains = np.array(L_trains)
                L_tests  = np.array(L_tests)
                L_gaps   = L_tests - L_trains

                L_train_mean = float(L_trains.mean())
                L_train_2std = float(2 * L_trains.std(ddof=1))
                L_test_mean  = float(L_tests.mean())
                L_test_2std  = float(2 * L_tests.std(ddof=1))
                L_gap_mean   = float(L_gaps.mean())
                L_gap_2std   = float(2 * L_gaps.std(ddof=1))

                row_common = [
                    target_tau, actual_tau, epoch_float, steps_per_epoch,
                    len(train_eval), len(test_eval),
                    L_train_mean, L_train_2std,
                    L_test_mean,  L_test_2std,
                    L_gap_mean,   L_gap_2std,
                    *L_trains.tolist(),
                    *L_tests.tolist(),
                ]

                # Write to per-(n_feat) CSV
                with open(per_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([n_train] + row_common)

                # Write to combined CSV
                with open(all_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([n_train, n_feat] + row_common)

    print(f"\n{'='*70}")
    print("Done!")
    print(f"  Combined CSV : {all_csv_path}")
    print(f"  Per-n_feat   : {out_dir}/loss_errorbars_nfeat{{W}}.csv")
    print(f"\nTo plot, run:")
    print(f"  python plot_errorbars_combined.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
