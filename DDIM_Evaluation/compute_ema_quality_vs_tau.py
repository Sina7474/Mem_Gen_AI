"""
compute_ema_quality_vs_tau.py
    — Quality vs τ from PROPER EMA weights (width 256), both metrics in one pass
================================================================================
This reads the EMA training runs produced by
    Code/DDIM_FMM/train_DDIM_tau_ema.py
which store the true weight-EMA in every checkpoint under the key
    'ema_model_state_dict'
on a DENSE τ grid.  For each available checkpoint we generate ONE batch of
samples from the EMA weights and compute BOTH quality metrics:

  (1) Effective-rank Wasserstein   W1(Gen, Test)   — the paper's FID analogue
  (2) BS-Beamspace Fréchet Distance BS-BFD(Gen,Test) — the channel "FID"

For each we also compute:
  • the Gen–Train curve  (memorization probe: collapses toward 0 when the model
    starts reproducing training points), and
  • the Train–Test benchmark (real–real floor). This flat line is what lets you
    *see the generalization window*: the window is the τ-range where Gen–Test
    sits on the Train–Test floor while Gen–Train has NOT yet collapsed.

Everything is read-only w.r.t. training; results go to a dedicated folder.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation

    # All finished EMA sizes (200, 1000, 4000), width 256, full dense grid
    python compute_ema_quality_vs_tau.py

    # A subset of sizes
    python compute_ema_quality_vs_tau.py --sizes 1000 4000

    # Quick smoke-test (few samples, coarse grid)
    python compute_ema_quality_vs_tau.py --sizes 200 --debug

    # Use RAW weights instead of EMA (sanity comparison)
    python compute_ema_quality_vs_tau.py --weights raw
"""

import sys
import os
import re
import glob
import csv
import argparse
from math import ceil
from pathlib import Path
import numpy as np
import torch
import importlib.util
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths & imports
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DDIM_DIR   = os.path.join(_SCRIPT_DIR, '..', 'Code', 'DDIM_FMM')
_ERANK_DIR  = os.path.join(_SCRIPT_DIR, '..', 'cDDIM_cFMM_Repo', 'Effective_Rank_Github')

sys.path.insert(0, _DDIM_DIR)
sys.path.insert(0, _ERANK_DIR)

# Effective-rank utilities (metric 1)
from effective_rank_core import (
    compute_effective_rank, compute_wasserstein, normalize_max_abs, to_beamspace)

# BS-BFD utilities (metric 2) — reuse the exact validated implementation
_bfd_path = os.path.join(_SCRIPT_DIR, 'compute_bs_bfd_vs_tau.py')
_bfd_spec = importlib.util.spec_from_file_location("_bfd_mod", _bfd_path)
_bfd_mod = importlib.util.module_from_spec(_bfd_spec)
_bfd_spec.loader.exec_module(_bfd_mod)
upa_dft_codebook              = _bfd_mod.upa_dft_codebook
channels_to_bs_power_features = _bfd_mod.channels_to_bs_power_features
bs_beamspace_frechet_distance = _bfd_mod.bs_beamspace_frechet_distance

# Model classes
_train_path = os.path.join(_DDIM_DIR, 'train_DDIM_tau.py')
_spec = importlib.util.spec_from_file_location("_train_mod_ema_eval", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
Unet = _train_mod.Unet
DDIM = _train_mod.DDIM


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_FEAT       = 256                       # width trained with EMA
TRAIN_SIZES  = [200, 1000, 4000]         # finished EMA runs
LOGS_EMA     = os.path.join(_DDIM_DIR, 'logs_ema')

# Generation
N_GEN        = 5000
N_GEN_DEBUG  = 500
BATCH_GEN    = 100
GEN_SEED     = 0                          # common latents across τ → comparable curves

# Restrict the dense grid in debug mode
DEBUG_TAUS   = [1000, 10000, 50000, 200000]

# Effective-rank beamspace convention (metric 1)
ERANK_ANTENNA = dict(Nrx_x=2, Nrx_y=2, Ntx_x=4, Ntx_y=8)

# BS-BFD beamspace convention (metric 2)
NRX_X, NRX_Y = 2, 2
NTX_X, NTX_Y = 8, 4

BETAS  = (1e-4, 0.02)
N_T    = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("results/ema_quality_vs_tau")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_ema_log_dir(n_train):
    return os.path.join(LOGS_EMA, f"DDIM_tau_ema_{n_train}_incremental")


def create_model():
    nn_model = Unet(in_channels=2, n_feat=N_FEAT)
    ddim = DDIM(nn_model=nn_model, betas=BETAS, n_T=N_T, device=DEVICE)
    ddim.to(DEVICE)
    return ddim


def list_checkpoints(log_dir):
    """Sorted list of (tau, path) for every checkpoint_tau_*.pth in log_dir."""
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    items = []
    for f in glob.glob(os.path.join(ckpt_dir, 'checkpoint_tau_*.pth')):
        m = re.search(r'checkpoint_tau_(\d+)\.pth$', os.path.basename(f))
        if m:
            items.append((int(m.group(1)), f))
    items.sort(key=lambda x: x[0])
    return items


def generate_samples(ddim, n_samples, seed=GEN_SEED):
    """Generate (n_samples, 2, 4, 32) numpy array from the loaded model."""
    torch.manual_seed(seed)
    ddim.eval()
    out = []
    got = 0
    with torch.no_grad():
        while got < n_samples:
            bs = min(BATCH_GEN, n_samples - got)
            s = ddim.sample(bs, (2, 4, 32), DEVICE)
            out.append(s.cpu().numpy())
            got += bs
    return np.concatenate(out, axis=0)[:n_samples]


# ── Metric-1 (effective-rank) data prep ──────────────────────────────────────
def erank_from_generated(samples):
    """Generated samples are already in beamspace → no transform."""
    H = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64)
    H = normalize_max_abs(H)
    er, _ = compute_effective_rank(H, use_power=True)
    return er


def erank_from_spatial_npy(arr):
    """Spatial real/imag (N,2,4,32) → beamspace → effective rank."""
    H_spatial = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
    H_beam = to_beamspace(H_spatial, **ERANK_ANTENNA)
    H_beam = normalize_max_abs(H_beam)
    er, _ = compute_effective_rank(H_beam, use_power=True)
    return er


# ── Metric-2 (BS-BFD) data prep ──────────────────────────────────────────────
_AR = upa_dft_codebook(NRX_X, NRX_Y)
_AT = upa_dft_codebook(NTX_X, NTX_Y)


def bfd_features_from_generated(samples):
    """Generated samples already in beamspace → power features directly."""
    return channels_to_bs_power_features(samples)


def bfd_features_from_spatial_npy(arr):
    """Spatial real/imag (N,2,4,32) → beamspace (UPA-DFT) → power features."""
    H = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)     # (N,4,32)
    Hv = np.matmul(np.matmul(_AR.conj().T, H), _AT)
    mag = np.abs(Hv)
    mx = mag.reshape(len(Hv), -1).max(axis=1)[:, None, None]
    mx = np.where(mx > 1e-12, mx, 1.0)
    Hv = Hv / mx
    ri = np.stack([np.real(Hv), np.imag(Hv)], axis=1).astype(np.float32)
    return channels_to_bs_power_features(ri)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap error bars (finite-sample variance of the metric estimate)
# ─────────────────────────────────────────────────────────────────────────────
# Each point on the curve is estimated from finite samples; the visible point-to-
# point jitter is largely this estimator variance. We quantify it by resampling
# BOTH sides (generated and reference) with replacement K times and reporting the
# std of the resulting metric. This reuses the SAME generated samples → no extra
# GPU cost.

def bootstrap_w1(a, b, K, rng):
    """Mean/std of W1(a, b) over K joint bootstrap resamples of both sets."""
    if K <= 0:
        return float('nan')
    na, nb = len(a), len(b)
    vals = np.empty(K)
    for i in range(K):
        ai = rng.integers(0, na, na)
        bi = rng.integers(0, nb, nb)
        vals[i] = compute_wasserstein(a[ai], b[bi], n_select=None)
    return float(np.std(vals))


def bootstrap_bfd(fa, fb, K, rng):
    """Std of BS-BFD(fa, fb) over K joint bootstrap resamples of both sets."""
    if K <= 0:
        return float('nan')
    na, nb = len(fa), len(fb)
    vals = np.empty(K)
    for i in range(K):
        ai = rng.integers(0, na, na)
        bi = rng.integers(0, nb, nb)
        vals[i] = bs_beamspace_frechet_distance(fa[ai], fb[bi])['distance']
    return float(np.std(vals))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="EMA-based quality (W1 + BS-BFD) vs τ, with Train-Test benchmark")
    ap.add_argument("--sizes", type=int, nargs='+', default=TRAIN_SIZES)
    ap.add_argument("--weights", choices=['ema', 'raw'], default='ema',
                    help="Which weights to sample from (default: ema)")
    ap.add_argument("--debug", action="store_true",
                    help="Few samples + coarse grid")
    ap.add_argument("--bootstrap", type=int, default=100,
                    help="Bootstrap resamples for error bars (0 disables). "
                         "Reuses the same generated samples → near-free.")
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    weight_key = 'ema_model_state_dict' if args.weights == 'ema' else 'model_state_dict'
    num_gen = N_GEN_DEBUG if args.debug else N_GEN
    K_boot = args.bootstrap
    rng = np.random.default_rng(12345)
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EMA Quality vs τ   (Effective-rank W1  +  BS-BFD)  — width", N_FEAT)
    print("=" * 72)
    print(f"  Device      : {DEVICE}")
    print(f"  Sizes       : {args.sizes}")
    print(f"  Weights     : {args.weights}  (key='{weight_key}')")
    print(f"  N_GEN       : {num_gen}")
    print(f"  Gen seed    : {GEN_SEED}")
    print(f"  Bootstrap   : {K_boot} resamples")
    print(f"  Output      : {out_dir}")
    print("=" * 72)

    csv_path = out_dir / f'ema_quality_vs_tau_{args.weights}.csv'
    cols = [
        'N', 'n_feat', 'tau', 'epoch_float', 'weights', 'num_generated',
        'num_train', 'num_test',
        # effective-rank Wasserstein (+ bootstrap std)
        'W1_Gen_Test', 'W1_Gen_Test_std',
        'W1_Gen_Train', 'W1_Gen_Train_std',
        'W1_Train_Test', 'W1_Train_Test_std',
        'mean_erank_gen', 'mean_erank_train', 'mean_erank_test',
        # BS-BFD (channel FID) (+ bootstrap std)
        'BFD_Gen_Test', 'BFD_Gen_Test_std',
        'BFD_Gen_Train', 'BFD_Gen_Train_std',
        'BFD_Train_Test', 'BFD_Train_Test_std',
    ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(cols)

    for n_train in args.sizes:
        log_dir = get_ema_log_dir(n_train)
        print(f"\n{'='*72}\n  N = {n_train}\n  {log_dir}\n{'='*72}")
        if not os.path.isdir(log_dir):
            print("  WARNING: EMA log dir missing, skip.")
            continue

        ckpts = list_checkpoints(log_dir)
        if not ckpts:
            print("  WARNING: no checkpoints, skip.")
            continue
        if args.debug:
            ckpts = [(t, p) for (t, p) in ckpts if t in DEBUG_TAUS]

        steps_per_epoch = ceil(n_train / 100)

        # ── Reference data (spatial → each metric's beamspace) ──────────────
        train_arr = np.load(os.path.join(log_dir, 'train.npy'))
        test_arr  = np.load(os.path.join(log_dir, 'test.npy'))

        er_train = erank_from_spatial_npy(train_arr)
        er_test  = erank_from_spatial_npy(test_arr)
        feat_train = bfd_features_from_spatial_npy(train_arr)
        feat_test  = bfd_features_from_spatial_npy(test_arr)

        # ── Benchmarks (real–real floors; constant in τ) ────────────────────
        w1_train_test  = compute_wasserstein(er_train, er_test, n_select=None)
        bfd_train_test = bs_beamspace_frechet_distance(feat_train, feat_test)['distance']
        w1_tt_std  = bootstrap_w1(er_train, er_test, K_boot, rng)
        bfd_tt_std = bootstrap_bfd(feat_train, feat_test, K_boot, rng)
        print(f"  Benchmark  W1(Train,Test) = {w1_train_test:.6f} ± {w1_tt_std:.6f}")
        print(f"  Benchmark BFD(Train,Test) = {bfd_train_test:.6f} ± {bfd_tt_std:.6f}")

        ddim = create_model()

        for tau, path in tqdm(ckpts, desc=f"  N={n_train} τ"):
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            if weight_key not in ck:
                print(f"    τ={tau}: '{weight_key}' missing, skip.")
                continue
            ddim.load_state_dict(ck[weight_key])

            samples = generate_samples(ddim, num_gen, seed=GEN_SEED)

            # metric 1: effective-rank W1
            er_gen = erank_from_generated(samples)
            w1_gen_test  = compute_wasserstein(er_gen, er_test,  n_select=None)
            w1_gen_train = compute_wasserstein(er_gen, er_train, n_select=None)
            w1_gt_std  = bootstrap_w1(er_gen, er_test,  K_boot, rng)
            w1_gtr_std = bootstrap_w1(er_gen, er_train, K_boot, rng)

            # metric 2: BS-BFD
            feat_gen = bfd_features_from_generated(samples)
            bfd_gen_test  = bs_beamspace_frechet_distance(feat_gen, feat_test)['distance']
            bfd_gen_train = bs_beamspace_frechet_distance(feat_gen, feat_train)['distance']
            bfd_gt_std  = bootstrap_bfd(feat_gen, feat_test,  K_boot, rng)
            bfd_gtr_std = bootstrap_bfd(feat_gen, feat_train, K_boot, rng)

            row = [
                n_train, N_FEAT, tau, tau / steps_per_epoch, args.weights, num_gen,
                len(er_train), len(er_test),
                w1_gen_test, w1_gt_std, w1_gen_train, w1_gtr_std,
                w1_train_test, w1_tt_std,
                float(np.mean(er_gen)), float(np.mean(er_train)), float(np.mean(er_test)),
                bfd_gen_test, bfd_gt_std, bfd_gen_train, bfd_gtr_std,
                bfd_train_test, bfd_tt_std,
            ]
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)

    print(f"\n{'='*72}")
    print("Done.")
    print(f"  CSV: {csv_path}")
    print(f"  Next: python plot_ema_quality_vs_tau.py --weights {args.weights}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
