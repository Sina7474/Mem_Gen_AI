"""
compute_model_size_metrics.py — Evaluate W1 and f_mem for all (n_feat, N, tau)
===============================================================================
Task 11: Model size/capacity study.

For each combination of model width (n_feat), dataset size (N), and training
time (tau), loads the checkpoint, generates samples, and computes:
  - W1(Gen, Test) on effective rank
  - W1(Gen, Train) on effective rank
  - f_mem with bootstrap 95% CI

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python compute_model_size_metrics.py

    # Specific widths/sizes:
    python compute_model_size_metrics.py --n_feats 64 128 --sizes 200 1000
    
    # Debug (fewer samples):
    python compute_model_size_metrics.py --debug
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

# Import model classes
_train_path = os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM', 'train_DDIM_tau.py')
_spec = importlib.util.spec_from_file_location("train_ddim_tau_module", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
sys.modules['train_ddim_tau_module'] = _train_mod
_spec.loader.exec_module(_train_mod)
Unet = _train_mod.Unet
DDIM = _train_mod.DDIM

# Import f_mem functions
_fmem_path = os.path.join(os.path.dirname(__file__), 'compute_fmem_vs_tau.py')
_spec2 = importlib.util.spec_from_file_location("compute_fmem_module", _fmem_path)
_fmem_mod = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_fmem_mod)
nearest_ratio_l2 = _fmem_mod.nearest_ratio_l2
compute_fmem_bootstrap = _fmem_mod.compute_fmem_bootstrap
spatial_to_normalized_beamspace_flat = _fmem_mod.spatial_to_normalized_beamspace_flat
generated_to_flat = _fmem_mod.generated_to_flat


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

N_FEAT_GRID = [64, 128, 256, 512]
TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000]
EVAL_TAU_GRID = [1000, 5000, 10000, 20000, 50000, 100000, 200000]
EVAL_TAU_GRID_DEBUG = [5000, 50000, 200000]

LOGS_BASE = os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM', 'logs')
OUTPUT_DIR = Path("results/model_size")

NUM_GENERATED = 5000
NUM_GENERATED_DEBUG = 500
BATCH_SIZE_GEN = 200
GEN_SEED = 0
FMEM_K = 1.0 / 3.0

ANTENNA = dict(Nrx_x=2, Nrx_y=2, Ntx_x=4, Ntx_y=8)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         HELPERS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_log_dir(n_train, n_feat):
    """Get the log directory for a given (N, n_feat) combination."""
    nfeat_suffix = f'_nfeat{n_feat}' if n_feat != 256 else ''
    return os.path.join(LOGS_BASE, f"DDIM_tau_{n_train}{nfeat_suffix}_incremental")


def create_model(n_feat, device):
    """Create model with specified width."""
    nn_model = Unet(in_channels=2, n_feat=n_feat)
    ddim = DDIM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=200, device=device)
    return ddim


def load_checkpoint(ddim, log_dir, tau, device):
    """Load a specific tau checkpoint. Returns actual tau or None if not found."""
    ckpt_path = os.path.join(log_dir, 'checkpoints', f'checkpoint_tau_{tau}.pth')

    if not os.path.exists(ckpt_path):
        # Find closest checkpoint >= tau
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            return None
        available = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_tau_')]
        available_taus = sorted([int(f.split('_')[-1].replace('.pth', '')) for f in available])
        candidates = [t for t in available_taus if t >= tau]
        if not candidates:
            return None
        closest = candidates[0]
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_tau_{closest}.pth')
        tau = closest

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in ckpt:
        ddim.load_state_dict(ckpt["model_state_dict"])
    else:
        ddim.load_state_dict(ckpt)
    return tau


def generate_samples(ddim, n_samples, batch_size, device, seed=0):
    """Generate samples from loaded model."""
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
    """(N, 2, 4, 32) normalized beamspace -> (N,) effective rank."""
    H = (samples_beamspace[:, 0] + 1j * samples_beamspace[:, 1]).astype(np.complex64)
    H = normalize_max_abs(H)
    er, _ = compute_effective_rank(H, use_power=True)
    return er


def load_test_erank(log_dir):
    """Load test.npy, convert to beamspace, return erank."""
    test_path = os.path.join(log_dir, 'test.npy')
    test_arr = np.load(test_path)
    H_spatial = (test_arr[:, 0] + 1j * test_arr[:, 1]).astype(np.complex64)
    H_beam = to_beamspace(H_spatial, **ANTENNA)
    H_beam = normalize_max_abs(H_beam)
    er, _ = compute_effective_rank(H_beam, use_power=True)
    return er


def load_train_erank(log_dir):
    """Load train.npy, convert to beamspace, return erank."""
    train_path = os.path.join(log_dir, 'train.npy')
    train_arr = np.load(train_path)
    H_spatial = (train_arr[:, 0] + 1j * train_arr[:, 1]).astype(np.complex64)
    H_beam = to_beamspace(H_spatial, **ANTENNA)
    H_beam = normalize_max_abs(H_beam)
    er, _ = compute_effective_rank(H_beam, use_power=True)
    return er


def compute_fmem_for_samples(gen_samples, log_dir, device):
    """
    Compute f_mem for generated samples against training data.
    gen_samples: (N, 2, 4, 32) already in normalized beamspace.
    """
    # Load training data (spatial domain)
    train_path = os.path.join(log_dir, 'train.npy')
    train_arr = np.load(train_path)  # (N_train, 2, 4, 32) spatial

    # Convert train to normalized beamspace flat
    train_flat = spatial_to_normalized_beamspace_flat(train_arr)  # (N_train, 256)

    # Generated is already normalized beamspace
    gen_flat = generated_to_flat(gen_samples)  # (N_gen, 256)

    # Compute nearest-neighbor ratios
    result = nearest_ratio_l2(
        torch.from_numpy(gen_flat),
        torch.from_numpy(train_flat),
        batch_size=512,
        device=device,
    )

    # Classify memorized (k = 1/3)
    is_mem = result["ratios"] < FMEM_K
    f_mem, ci_low, ci_high = compute_fmem_bootstrap(is_mem, B=1000, seed=42)
    return f_mem, ci_low, ci_high


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Compute W1 and f_mem metrics for model-size study (Task 11)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: fewer samples, coarser grid")
    parser.add_argument("--n_feats", type=int, nargs='+', default=None,
                        help=f"Specific widths to evaluate (default: {N_FEAT_GRID})")
    parser.add_argument("--sizes", type=int, nargs='+', default=None,
                        help=f"Specific sizes to evaluate (default: {TRAIN_SIZES})")
    parser.add_argument("--taus", type=int, nargs='+', default=None,
                        help="Specific tau values (default: EVAL_TAU_GRID)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--skip_fmem", action="store_true",
                        help="Skip f_mem computation (faster)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV instead of overwriting it")
    args = parser.parse_args()

    # Config
    n_feats = args.n_feats if args.n_feats else N_FEAT_GRID
    sizes = args.sizes if args.sizes else TRAIN_SIZES
    tau_grid = args.taus if args.taus else (EVAL_TAU_GRID_DEBUG if args.debug else EVAL_TAU_GRID)
    num_gen = NUM_GENERATED_DEBUG if args.debug else NUM_GENERATED
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 11: Model Size Metrics — W1 and f_mem")
    print("=" * 70)
    print(f"  n_feat grid: {n_feats}")
    print(f"  Train sizes: {sizes}")
    print(f"  Tau grid:    {tau_grid}")
    print(f"  Num generated: {num_gen}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # CSV output
    csv_path = output_dir / "model_size_metrics.csv"
    csv_fields = [
        'n_feat', 'p', 'N', 'tau', 'W1_Gen_Test', 'W1_Gen_Train',
        'W1_Test_Train', 'f_mem', 'f_mem_ci_low', 'f_mem_ci_high',
    ]

    csv_mode = 'a' if (args.append and csv_path.exists()) else 'w'
    with open(csv_path, csv_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if csv_mode == 'w':
            writer.writeheader()

    total_combos = len(n_feats) * len(sizes) * len(tau_grid)
    combo_idx = 0

    for n_feat in n_feats:
        # Create model once per width
        ddim = create_model(n_feat, DEVICE)
        ddim.to(DEVICE)
        p = sum(pp.numel() for pp in ddim.parameters())
        print(f"\n{'─' * 70}")
        print(f"n_feat={n_feat}, p={p:,} ({p/1e6:.2f}M)")
        print(f"{'─' * 70}")

        for N in sizes:
            log_dir = get_log_dir(N, n_feat)

            if not os.path.exists(log_dir):
                print(f"\n  [n_feat={n_feat}, N={N}] SKIP — dir not found: {log_dir}")
                combo_idx += len(tau_grid)
                continue

            print(f"\n  [n_feat={n_feat}, N={N}] Loading reference data...")

            # Load reference erank distributions (once per N)
            try:
                er_test = load_test_erank(log_dir)
                er_train = load_train_erank(log_dir)
                w1_test_train = compute_wasserstein(er_test, er_train)
                print(f"    Test erank: {len(er_test)} samples, "
                      f"Train erank: {len(er_train)} samples, "
                      f"W1(Test,Train)={w1_test_train:.4f}")
            except Exception as e:
                print(f"    ERROR loading reference data: {e}. Skipping.")
                combo_idx += len(tau_grid)
                continue

            for tau in tau_grid:
                combo_idx += 1
                print(f"\n    [{combo_idx}/{total_combos}] "
                      f"n_feat={n_feat}, N={N}, tau={tau}...")

                # Load checkpoint
                actual_tau = load_checkpoint(ddim, log_dir, tau, DEVICE)
                if actual_tau is None:
                    print(f"      No checkpoint found for tau>={tau}. Skipping.")
                    continue

                if actual_tau != tau:
                    print(f"      Using closest checkpoint: tau={actual_tau}")

                # Generate samples
                gen_samples = generate_samples(ddim, num_gen, BATCH_SIZE_GEN, DEVICE, GEN_SEED)

                # W1 metrics
                er_gen = samples_to_erank(gen_samples)
                w1_gen_test = compute_wasserstein(er_gen, er_test)
                w1_gen_train = compute_wasserstein(er_gen, er_train)
                print(f"      W1(Gen,Test)={w1_gen_test:.4f}, "
                      f"W1(Gen,Train)={w1_gen_train:.4f}")

                # f_mem
                f_mem, ci_low, ci_high = 0.0, 0.0, 0.0
                if not args.skip_fmem:
                    try:
                        f_mem, ci_low, ci_high = compute_fmem_for_samples(
                            gen_samples, log_dir, DEVICE)
                        print(f"      f_mem={100*f_mem:.2f}% "
                              f"[{100*ci_low:.2f}%, {100*ci_high:.2f}%]")
                    except Exception as e:
                        print(f"      f_mem ERROR: {e}")

                # Write row
                row = {
                    'n_feat': n_feat,
                    'p': p,
                    'N': N,
                    'tau': actual_tau,
                    'W1_Gen_Test': w1_gen_test,
                    'W1_Gen_Train': w1_gen_train,
                    'W1_Test_Train': w1_test_train,
                    'f_mem': f_mem,
                    'f_mem_ci_low': ci_low,
                    'f_mem_ci_high': ci_high,
                }
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fields)
                    writer.writerow(row)

    print(f"\n{'=' * 70}")
    print(f"Done! Results saved to: {csv_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
