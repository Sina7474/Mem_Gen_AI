"""
compute_fcd_vs_tau.py
    — Fréchet Channel Distance (FCD, "Option A") vs τ from EMA weights (width 256)
================================================================================
This is the *faithful channel analogue of the paper's FID* using the simplest
feature space (Option A in Teach_Docs/FID_Original_code.md):

    φ(H) = flatten the (2,4,32) beamspace channel  →  R^256      (raw features)

The metric is EXACTLY the Fréchet/2-Wasserstein distance between two Gaussians,
the same formula the pytorch-fid package uses for images:

    FCD = ||μ_g - μ_r||² + Tr(Σ_g + Σ_r - 2 (Σ_g Σ_r)^{1/2})

Everything else copies the paper's protocol:
  • reference = held-out TEST channels (disjoint from the N training channels),
    split into 5 disjoint folds → error bars = 2·std over folds (like `istat`);
  • FCD(Gen, Test)  = quality curve      (want it LOW, saturating);
  • FCD(Train, Test) = the real–real floor (irreducible finite-sample distance).

Memorization is NOT tracked here — you already have a dedicated f_mem metric
(compute_fmem_vs_tau.py), exactly like the paper's Figure 2 (left) which shows
FID (solid) + f_mem (dashed). The plot script can overlay that f_mem.

Generated samples are CACHED to <log_dir>/generated_ema/ on first use, so a
re-run — or any other metric — reuses them instead of regenerating on the GPU
(generation is the only expensive step; the Fréchet math is milliseconds).

The generalization window is the τ-band where FCD(Gen,Test) has descended onto
the real–real floor while f_mem is still ~0.

Read-only w.r.t. training; results go to results/fcd_vs_tau/.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation

    python compute_fcd_vs_tau.py                 # ema, sizes 200 1000 4000
    python compute_fcd_vs_tau.py --sizes 1000    # one size
    python compute_fcd_vs_tau.py --debug         # quick smoke test (few samples)
    python compute_fcd_vs_tau.py --weights raw   # sanity vs non-EMA weights
    python compute_fcd_vs_tau.py --standardize   # z-score features with train stats
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
from scipy import linalg
import torch
import importlib.util
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths & imports  (reuse the SAME model + beamspace pipeline as the EMA script)
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DDIM_DIR   = os.path.join(_SCRIPT_DIR, '..', 'Code', 'DDIM_FMM')
sys.path.insert(0, _DDIM_DIR)

# UPA-DFT beamspace codebook (identical convention to BS-BFD / training)
_bfd_path = os.path.join(_SCRIPT_DIR, 'compute_bs_bfd_vs_tau.py')
_bfd_spec = importlib.util.spec_from_file_location("_bfd_mod_fcd", _bfd_path)
_bfd_mod = importlib.util.module_from_spec(_bfd_spec)
_bfd_spec.loader.exec_module(_bfd_mod)
upa_dft_codebook = _bfd_mod.upa_dft_codebook

# Model classes
_train_path = os.path.join(_DDIM_DIR, 'train_DDIM_tau.py')
_spec = importlib.util.spec_from_file_location("_train_mod_fcd", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
Unet = _train_mod.Unet
DDIM = _train_mod.DDIM


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
N_FEAT       = 256                       # model width trained with EMA
TRAIN_SIZES  = [200, 1000, 4000]
LOGS_EMA     = os.path.join(_DDIM_DIR, 'logs_ema')

N_GEN        = 5000
N_GEN_DEBUG  = 500
BATCH_GEN    = 100
GEN_SEED     = 0
N_FOLDS      = 5                          # disjoint TEST folds → 2σ error bars
FOLD_SEED    = 0

DEBUG_TAUS   = [1000, 10000, 50000, 200000]

# Beamspace convention (matches training + BS-BFD)
NRX_X, NRX_Y = 2, 2
NTX_X, NTX_Y = 8, 4

BETAS  = (1e-4, 0.02)
N_T    = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("results/fcd_vs_tau")

_AR = upa_dft_codebook(NRX_X, NRX_Y)
_AT = upa_dft_codebook(NTX_X, NTX_Y)


# ─────────────────────────────────────────────────────────────────────────────
# Fréchet distance core  (identical math to pytorch-fid)
# ─────────────────────────────────────────────────────────────────────────────
def fit_gaussian(features):
    """features: (M, D) real → (mu (D,), sigma (D,D))."""
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """FID/FCD formula: ||mu1-mu2||^2 + Tr(S1 + S2 - 2 sqrt(S1 S2))."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor φ  (Option A: raw beamspace flatten → R^256)
# ─────────────────────────────────────────────────────────────────────────────
def _norm_per_sample(ri):
    """Per-sample max-magnitude normalization, matching the training pipeline.
    ri: (N,2,4,32) real/imag → same shape, each sample scaled so max|H|=1."""
    mag = np.sqrt(ri[:, 0] ** 2 + ri[:, 1] ** 2)               # (N,4,32)
    mx = mag.reshape(len(ri), -1).max(axis=1)[:, None, None]
    mx = np.where(mx > 1e-12, mx, 1.0)
    out = ri.copy()
    out[:, 0] /= mx
    out[:, 1] /= mx
    return out


def features_from_generated(samples):
    """Generated samples are ALREADY in normalized beamspace → just flatten."""
    return samples.reshape(len(samples), -1).astype(np.float64)


def features_from_spatial_npy(arr):
    """Spatial real/imag (N,2,4,32) → UPA-DFT beamspace → per-sample norm → flatten."""
    H = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)      # (N,4,32)
    Hv = np.matmul(np.matmul(_AR.conj().T, H), _AT)            # beamspace
    ri = np.stack([np.real(Hv), np.imag(Hv)], axis=1).astype(np.float32)
    ri = _norm_per_sample(ri)
    return ri.reshape(len(ri), -1).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Model / checkpoint helpers  (same as the EMA script)
# ─────────────────────────────────────────────────────────────────────────────
def get_ema_log_dir(n_train):
    return os.path.join(LOGS_EMA, f"DDIM_tau_ema_{n_train}_incremental")


def create_model():
    ddim = DDIM(nn_model=Unet(in_channels=2, n_feat=N_FEAT),
                betas=BETAS, n_T=N_T, device=DEVICE)
    ddim.to(DEVICE)
    return ddim


def list_checkpoints(log_dir):
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    items = []
    for f in glob.glob(os.path.join(ckpt_dir, 'checkpoint_tau_*.pth')):
        m = re.search(r'checkpoint_tau_(\d+)\.pth$', os.path.basename(f))
        if m:
            items.append((int(m.group(1)), f))
    items.sort(key=lambda x: x[0])
    return items


def generate_samples(ddim, n_samples, seed=GEN_SEED):
    torch.manual_seed(seed)
    ddim.eval()
    out, got = [], 0
    with torch.no_grad():
        while got < n_samples:
            bs = min(BATCH_GEN, n_samples - got)
            s = ddim.sample(bs, (2, 4, 32), DEVICE)
            out.append(s.cpu().numpy())
            got += bs
    return np.concatenate(out, axis=0)[:n_samples]


def get_or_generate_samples(ddim, log_dir, tau, n_samples, weights, seed=GEN_SEED):
    """Load cached generated samples if present (enough of them), else generate
    them ONCE and save to <log_dir>/generated_ema/ so every metric can reuse them.

    Cache key encodes weights + tau + seed so runs never collide. Returns
    (samples (n_samples,2,4,32), cache_hit: bool).
    """
    cache_dir = os.path.join(log_dir, 'generated_ema')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'gen_{weights}_tau{tau}_seed{seed}.npz')

    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path)['channels']
            if arr.shape[0] >= n_samples:
                return arr[:n_samples], True
        except Exception:
            pass  # corrupt/partial cache → regenerate below

    samples = generate_samples(ddim, n_samples, seed=seed)
    np.savez_compressed(cache_path, channels=samples)
    return samples, False


# ─────────────────────────────────────────────────────────────────────────────
# FCD against a set of reference Gaussians (5 test folds) → mean ± 2·std
# ─────────────────────────────────────────────────────────────────────────────
def fcd_vs_reference_folds(mu_g, sig_g, ref_stats):
    vals = [frechet_distance(mu_g, sig_g, mu_r, sig_r) for (mu_r, sig_r) in ref_stats]
    vals = np.asarray(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std())


def build_fold_stats(features, n_folds=N_FOLDS, seed=FOLD_SEED):
    """Split feature matrix into n_folds disjoint subsets → list of (mu, sigma)."""
    idx = np.random.default_rng(seed).permutation(len(features))
    return [fit_gaussian(features[f]) for f in np.array_split(idx, n_folds)]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Fréchet Channel Distance (Option A) vs τ, paper-style protocol")
    ap.add_argument("--sizes", type=int, nargs='+', default=TRAIN_SIZES)
    ap.add_argument("--weights", choices=['ema', 'raw'], default='ema')
    ap.add_argument("--debug", action="store_true", help="Few samples + coarse grid")
    ap.add_argument("--standardize", action="store_true",
                    help="z-score features using TRAIN mean/std before fitting Gaussians")
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    weight_key = 'ema_model_state_dict' if args.weights == 'ema' else 'model_state_dict'
    num_gen = N_GEN_DEBUG if args.debug else N_GEN
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Fréchet Channel Distance (FCD, Option A: raw beamspace)  vs τ  — width", N_FEAT)
    print("=" * 72)
    print(f"  Device      : {DEVICE}")
    print(f"  Sizes       : {args.sizes}")
    print(f"  Weights     : {args.weights}  (key='{weight_key}')")
    print(f"  N_GEN       : {num_gen}")
    print(f"  Gen seed    : {GEN_SEED}")
    print(f"  Test folds  : {N_FOLDS}  (→ 2σ error bars)")
    print(f"  Standardize : {args.standardize}")
    print(f"  Output      : {out_dir}")
    print("=" * 72)

    csv_path = out_dir / f'fcd_vs_tau_{args.weights}.csv'
    cols = [
        'N', 'n_feat', 'tau', 'epoch_float', 'weights', 'num_generated',
        'num_train', 'num_test', 'feature_dim', 'n_folds',
        'FCD_Gen_Test', 'FCD_Gen_Test_std',
        'FCD_Train_Test', 'FCD_Train_Test_std',
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

        # ── Reference features (spatial → beamspace → norm → flatten) ───────
        train_arr = np.load(os.path.join(log_dir, 'train.npy'))
        test_arr  = np.load(os.path.join(log_dir, 'test.npy'))
        feat_train = features_from_spatial_npy(train_arr)
        feat_test  = features_from_spatial_npy(test_arr)

        # Optional standardization (fit on TRAIN, apply to everything)
        if args.standardize:
            mu_s = feat_train.mean(axis=0)
            sd_s = feat_train.std(axis=0)
            sd_s = np.where(sd_s > 1e-12, sd_s, 1.0)
            _std = lambda F: (F - mu_s) / sd_s
            feat_train = _std(feat_train)
            feat_test  = _std(feat_test)
        else:
            _std = lambda F: F

        # ── Reference Gaussians: 5 disjoint TEST folds + real–real floor ────
        ref_stats = build_fold_stats(feat_test)
        mu_train, sig_train = fit_gaussian(feat_train)
        fcd_tt_mean, fcd_tt_std = fcd_vs_reference_folds(mu_train, sig_train, ref_stats)
        print(f"  Feature dim : {feat_train.shape[1]}")
        print(f"  Benchmark FCD(Train,Test) = {fcd_tt_mean:.6f} ± {fcd_tt_std:.6f}")

        ddim = create_model()

        n_hits = 0
        for tau, path in tqdm(ckpts, desc=f"  N={n_train} τ"):
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            if weight_key not in ck:
                print(f"    τ={tau}: '{weight_key}' missing, skip.")
                continue
            ddim.load_state_dict(ck[weight_key])

            # Reuse cached samples if available, else generate once and save.
            samples, hit = get_or_generate_samples(
                ddim, log_dir, tau, num_gen, args.weights, seed=GEN_SEED)
            n_hits += int(hit)

            feat_gen = _std(features_from_generated(samples))
            mu_g, sig_g = fit_gaussian(feat_gen)
            fcd_gt_mean, fcd_gt_std = fcd_vs_reference_folds(mu_g, sig_g, ref_stats)

            row = [
                n_train, N_FEAT, tau, tau / steps_per_epoch, args.weights, num_gen,
                len(feat_train), len(feat_test), feat_train.shape[1], N_FOLDS,
                fcd_gt_mean, fcd_gt_std,
                fcd_tt_mean, fcd_tt_std,
            ]
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
        print(f"  Cache hits: {n_hits}/{len(ckpts)}  "
              f"(cached samples in {os.path.join(log_dir, 'generated_ema')})")

    print(f"\n{'='*72}")
    print("Done.")
    print(f"  CSV: {csv_path}")
    print(f"  Next: python plot_fcd_vs_tau.py --weights {args.weights}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
