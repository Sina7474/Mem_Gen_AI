"""
compute_w1_quality_vs_tau_ckptavg.py
    — W1(Gen, Test) Quality vs τ with POST-HOC CHECKPOINT AVERAGING (approx-EMA)
================================================================================
This is the "approximation method" (Option 1B from
Teach_Docs/sample_quality_smoothness_diagnosis.md): the last training run did
NOT save EMA weights, and we do NOT want to re-train. Instead of sampling from a
single raw checkpoint at each τ (which makes the quality curve jump up and
down), we approximate a weight-EMA by averaging the state_dicts of a trailing
window of already-saved checkpoints around each τ.

    W_smoothed(τ_k) = mean( state_dict(τ_{k-w+1}), ..., state_dict(τ_k) )

Only floating-point tensors are averaged (weights, biases, BatchNorm running
mean/var, and the — identical — DDIM schedule buffers). Integer buffers such as
BatchNorm `num_batches_tracked` are copied from the newest checkpoint in the
window (averaging them is meaningless).

This mirrors compute_w1_quality_vs_tau.py exactly EXCEPT for how the weights are
obtained. Fast metric first: effective-rank Wasserstein.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation

    # Default: window = 3 trailing checkpoints, all sizes
    python compute_w1_quality_vs_tau_ckptavg.py

    # Try a wider / narrower smoothing window
    python compute_w1_quality_vs_tau_ckptavg.py --window 4
    python compute_w1_quality_vs_tau_ckptavg.py --window 2

    # window = 1 reproduces the ORIGINAL (no averaging) — useful as a sanity check
    python compute_w1_quality_vs_tau_ckptavg.py --window 1

    # Only a couple of sizes, debug (fewer samples / coarse grid)
    python compute_w1_quality_vs_tau_ckptavg.py --sizes 1000 2000 --debug
"""

import sys
import os
import re
import glob
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
OUTPUT_DIR = Path("results/w1_quality_vs_tau_ckptavg")

# Tau grid for generation-quality evaluation
GEN_EVAL_TAU_GRID = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
GEN_EVAL_TAU_GRID_DEBUG = [1000, 10000, 50000, 200000]

# Generation settings
NUM_GENERATED = 5000
NUM_GENERATED_DEBUG = 500
BATCH_SIZE_GEN = 100
GEN_SEED = 0  # Fixed seed for reproducible generation (common latents across τ)

# Default checkpoint-averaging window (trailing count, incl. the target τ)
DEFAULT_WINDOW = 3

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


def list_available_checkpoints(log_dir):
    """
    Return a sorted list of (tau, path) for every saved checkpoint in log_dir.
    """
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    items = []
    for f in glob.glob(os.path.join(ckpt_dir, 'checkpoint_tau_*.pth')):
        m = re.search(r'checkpoint_tau_(\d+)\.pth$', os.path.basename(f))
        if m:
            items.append((int(m.group(1)), f))
    items.sort(key=lambda x: x[0])
    return items


def select_window_paths(available, target_tau, window):
    """
    Trailing window of `window` checkpoints with tau <= target_tau, ending at the
    checkpoint whose tau == target_tau. Returns (list_of_paths, list_of_taus).

    If the exact target_tau checkpoint is missing, returns ([], []).
    If fewer than `window` earlier checkpoints exist, returns whatever is there.
    """
    taus = [t for t, _ in available]
    if target_tau not in taus:
        return [], []
    end_idx = taus.index(target_tau)
    start_idx = max(0, end_idx - window + 1)
    chosen = available[start_idx:end_idx + 1]
    return [p for _, p in chosen], [t for t, _ in chosen]


def average_state_dicts(paths, device):
    """
    Average the 'model_state_dict' of the given checkpoints.

    Floating-point tensors are averaged in float64 for numerical safety and cast
    back to their original dtype. Non-float tensors (e.g. BatchNorm
    `num_batches_tracked`) are taken verbatim from the newest checkpoint.
    """
    state_dicts = []
    for p in paths:
        ck = torch.load(p, map_location=device, weights_only=False)
        state_dicts.append(ck['model_state_dict'])

    newest = state_dicts[-1]  # window is ordered oldest -> newest
    out = {}
    n = len(state_dicts)
    for k, v in newest.items():
        if torch.is_floating_point(v):
            acc = torch.zeros_like(v, dtype=torch.float64)
            for sd in state_dicts:
                acc += sd[k].to(torch.float64)
            out[k] = (acc / n).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


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
    """Convert beamspace samples (N, 2, 4, 32) to effective-rank values."""
    H = (samples_beamspace[:, 0] + 1j * samples_beamspace[:, 1]).astype(np.complex64)
    H = normalize_max_abs(H)
    er, _ = compute_effective_rank(H, use_power=True)
    return er


def load_test_erank(log_dir):
    """Load test data (spatial), convert to beamspace, compute erank."""
    test_arr = np.load(os.path.join(log_dir, 'test.npy'))
    H_spatial = (test_arr[:, 0] + 1j * test_arr[:, 1]).astype(np.complex64)
    H_beam = to_beamspace(H_spatial, **ANTENNA)
    H_beam = normalize_max_abs(H_beam)
    er, _ = compute_effective_rank(H_beam, use_power=True)
    return er


def load_train_erank(log_dir):
    """Load training data (spatial), convert to beamspace, compute erank."""
    train_arr = np.load(os.path.join(log_dir, 'train.npy'))
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
        description="W1(Gen, Test) quality vs τ with post-hoc checkpoint averaging (approx-EMA)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: fewer samples, coarser grid")
    parser.add_argument("--sizes", type=int, nargs='+', default=None,
                        help="Specific sizes to evaluate (default: all)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help="Trailing checkpoint-averaging window (>=1). "
                             "window=1 reproduces the original single-checkpoint result.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    assert args.window >= 1, "--window must be >= 1"

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
    print("W1 Sample-Quality vs τ  —  POST-HOC CHECKPOINT AVERAGING (approx-EMA)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Sizes: {sizes}")
    print(f"  Tau grid: {tau_grid}")
    print(f"  Averaging window (trailing): {args.window}")
    print(f"  Generating {num_gen} samples per τ")
    print(f"  Generation seed: {GEN_SEED}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # ── Pre-compute test effective rank (same test set for all sizes) ─────
    test_log_dir = get_tau_log_dir(sizes[0])
    print(f"\nLoading test data from: {test_log_dir}")
    er_test = load_test_erank(test_log_dir)
    print(f"  Test erank: {len(er_test)} samples, mean={np.mean(er_test):.4f}, "
          f"std={np.std(er_test):.4f}")

    # ── Prepare CSV ───────────────────────────────────────────────────────
    csv_path = output_dir / f'wasserstein_quality_vs_tau_win{args.window}.csv'
    csv_columns = [
        'N', 'target_tau', 'actual_tau', 'epoch_float', 'steps_per_epoch',
        'window', 'window_taus', 'num_generated', 'num_train', 'num_test',
        'W1_Gen_Test', 'W1_Gen_Train', 'W1_Test_Train', 'Excess_Gap',
        'mean_erank_gen', 'std_erank_gen',
        'mean_erank_train', 'std_erank_train',
        'mean_erank_test', 'std_erank_test',
    ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(csv_columns)

    # ── Main loop: for each N ─────────────────────────────────────────────
    for n_train in sizes:
        log_dir = get_tau_log_dir(n_train)
        steps_per_epoch = ceil(n_train / BATCH_SIZE_TRAIN)

        print(f"\n{'=' * 70}")
        print(f"  N = {n_train}   steps_per_epoch = {steps_per_epoch}")
        print(f"  Log dir: {log_dir}")
        print(f"{'=' * 70}")

        if not os.path.exists(log_dir):
            print(f"  WARNING: Directory not found, skipping N={n_train}")
            continue

        available = list_available_checkpoints(log_dir)
        if not available:
            print(f"  WARNING: No checkpoints found, skipping N={n_train}")
            continue

        er_train = load_train_erank(log_dir)
        print(f"  Train erank: {len(er_train)} samples, mean={np.mean(er_train):.4f}")

        w1_test_train = compute_wasserstein(er_test, er_train, n_select=None)
        print(f"  W1(Test, Train) baseline = {w1_test_train:.6f}")

        ddim = create_model(DEVICE)

        for target_tau in tau_grid:
            paths, win_taus = select_window_paths(available, target_tau, args.window)
            if not paths:
                print(f"    τ={target_tau:>7d}: SKIP (checkpoint not found)")
                continue

            # Average the trailing window of checkpoints (approx-EMA)
            avg_sd = average_state_dicts(paths, DEVICE)
            ddim.load_state_dict(avg_sd)

            actual_tau = target_tau
            epoch_float = actual_tau / steps_per_epoch
            win_str = "|".join(str(t) for t in win_taus)

            print(f"    τ={actual_tau:>7d} (epoch≈{epoch_float:6.1f}) "
                  f"avg[{win_str}]: generating {num_gen}...", end='', flush=True)
            samples = generate_samples(ddim, num_gen, BATCH_SIZE_GEN, DEVICE, seed=GEN_SEED)

            er_gen = samples_to_erank(samples)
            w1_gen_test = compute_wasserstein(er_gen, er_test, n_select=None)
            w1_gen_train = compute_wasserstein(er_gen, er_train, n_select=None)
            excess_gap = w1_gen_test - w1_test_train

            print(f" W1(Gen,Test)={w1_gen_test:.4f}, "
                  f"W1(Gen,Train)={w1_gen_train:.4f}, Excess={excess_gap:.4f}")

            # Save generated channels (separate dir so raw results are untouched)
            gen_dir = os.path.join(log_dir, 'generated_tau_ckptavg')
            os.makedirs(gen_dir, exist_ok=True)
            gen_path = os.path.join(gen_dir, f'generated_tau_{actual_tau}_win{args.window}.npz')
            np.savez_compressed(gen_path, channels=samples)

            row = [
                n_train, target_tau, actual_tau, epoch_float, steps_per_epoch,
                args.window, win_str, num_gen, len(er_train), len(er_test),
                w1_gen_test, w1_gen_train, w1_test_train, excess_gap,
                float(np.mean(er_gen)), float(np.std(er_gen)),
                float(np.mean(er_train)), float(np.std(er_train)),
                float(np.mean(er_test)), float(np.std(er_test)),
            ]
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)

    print(f"\n{'=' * 70}")
    print("Computation complete!")
    print(f"  CSV saved to: {csv_path}")
    print(f"  Generated channels: <log_dir>/generated_tau_ckptavg/")
    print(f"\n  Compare against the raw (non-averaged) curve:")
    print(f"    results/w1_quality_vs_tau/wasserstein_quality_vs_tau.csv")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
