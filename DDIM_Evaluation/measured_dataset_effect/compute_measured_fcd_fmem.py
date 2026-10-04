"""
compute_measured_fcd_fmem.py
    — MEASURED dataset: FCD(Gen,Test) + f_mem vs τ  (single UE ant, shape (2,4,8))
================================================================================
Measured-data analogue of dataset_size_effect/compute_dsize_fcd_fmem.py. For each
dataset size N and every τ checkpoint produced by train_DDIM_tau_ema_measured.py,
this computes from the SAME set of EMA-generated samples:

  • FCD(Gen, Test)   — Fréchet Channel Distance (quality, ↓) with 5 test folds
  • FCD(Train, Test) — the real–real floor (per-N)
  • f_mem(τ)         — memorization fraction (nearest-neighbour ratio, k=1/3)
                       with a 95% bootstrap CI

Representation (measured): per-sample-normalized BS-ONLY beamspace of the
physically-ordered 4x8 channel, flattened →  R^64  (2 * 4 * 8).
  - generated samples are already normalized beamspace (2,4,8) → just flatten
  - reference train/test are spatial (2,4,8) → F4^H @ H @ F8 → per-sample norm → flatten

Reads   : ../../Code/DDIM_FMM/logs_ema/DDIM_tau_ema_measured_1p272GHz_N{N}_incremental/
Caches  : <log_dir>/generated_ema/gen_ema_tau{τ}_seed0.npz   (reused on re-run)
Writes  : results/measured_fcd_fmem.csv

Usage
-----
    conda activate Mem_Gen
    cd DDIM_Evaluation/measured_dataset_effect
    python compute_measured_fcd_fmem.py                  # sizes 200 1000
    python compute_measured_fcd_fmem.py --sizes 1000     # a single size
    python compute_measured_fcd_fmem.py --debug          # quick smoke test
"""

import os
import re
import sys
import csv
import glob
import argparse
import importlib.util
from math import ceil
from pathlib import Path

import numpy as np
from scipy import linalg
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths & measured model import (Unet/DDIM for (2,4,8))
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DDIM_DIR = os.path.join(_HERE, '..', '..', 'Code', 'DDIM_FMM')
_LOGS_EMA = os.path.join(_DDIM_DIR, 'logs_ema')

_meas_path = os.path.join(_DDIM_DIR, 'train_DDIM_measured.py')
_spec = importlib.util.spec_from_file_location("_meas_mod_fcd", _meas_path)
_meas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_meas)

Unet = _meas.Unet
DDIM = _meas.DDIM
dft_matrix = _meas.dft_matrix
SPATIAL_SHAPE = _meas.SPATIAL_SHAPE      # (4, 8)
SAMPLE_SIZE = _meas.SAMPLE_SIZE          # (2, 4, 8)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SIZES = [200, 1000]
N_FEAT = 256
BATCH_SIZE_TRAIN = 100                   # trainer default (→ no _bs suffix in dir name)
N_GEN = 5000
N_GEN_DEBUG = 500
BATCH_GEN = 100
GEN_SEED = 0
N_FOLDS = 5
FOLD_SEED = 0
BETAS = (1e-4, 0.02)
N_T = 200
WEIGHT_KEY = 'ema_model_state_dict'
WEIGHTS = 'ema'

# f_mem
K_MAIN = 1/3
NN_BATCH_SIZE = 512
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 42

DEBUG_TAUS = [1000, 10000, 50000, 200000]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("results")

# BS-only beamspace matrices (F4 elevation, F8 azimuth)
_F4 = dft_matrix(SPATIAL_SHAPE[0])       # (4,4)
_F8 = dft_matrix(SPATIAL_SHAPE[1])       # (8,8)


def log_dir_for(n_train):
    return os.path.join(_LOGS_EMA,
                        f"DDIM_tau_ema_measured_1p272GHz_N{n_train}_incremental")


# ─────────────────────────────────────────────────────────────────────────────
# Fréchet distance core (identical math to pytorch-fid)
# ─────────────────────────────────────────────────────────────────────────────
def fit_gaussian(features):
    mu = features.mean(axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def build_fold_stats(features, n_folds=N_FOLDS, seed=FOLD_SEED):
    idx = np.random.default_rng(seed).permutation(len(features))
    return [fit_gaussian(features[f]) for f in np.array_split(idx, n_folds)]


def fcd_vs_reference_folds(mu_g, sig_g, ref_stats):
    vals = np.asarray([frechet_distance(mu_g, sig_g, mu_r, sig_r)
                       for (mu_r, sig_r) in ref_stats], dtype=np.float64)
    return float(vals.mean()), float(vals.std())


# ─────────────────────────────────────────────────────────────────────────────
# Feature extractor φ  (measured: BS-only DFT, flatten → R^64)
# ─────────────────────────────────────────────────────────────────────────────
def features_from_spatial_npy(arr):
    """Spatial real/imag (N,2,4,8) physically-ordered → BS-only beamspace →
    per-sample max-magnitude norm → flatten → (N,64)."""
    H = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)         # (N,4,8)
    Hv = np.matmul(np.matmul(_F4.conj().T, H), _F8)               # (N,4,8) beamspace
    mag = np.sqrt(np.real(Hv) ** 2 + np.imag(Hv) ** 2)            # (N,4,8)
    mx = mag.reshape(len(Hv), -1).max(axis=1)[:, None, None]
    mx = np.where(mx > 1e-12, mx, 1.0)
    Hv = Hv / mx
    flat = np.concatenate([np.real(Hv), np.imag(Hv)], axis=1).reshape(len(Hv), -1)
    return flat.astype(np.float64)


def features_from_generated(samples):
    """Generated samples already in normalized beamspace (N,2,4,8) → flatten."""
    return samples.reshape(len(samples), -1).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Model / checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
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
            s = ddim.sample(bs, SAMPLE_SIZE, DEVICE)     # (bs,2,4,8)
            out.append(s.cpu().numpy())
            got += bs
    return np.concatenate(out, axis=0)[:n_samples]


def get_or_generate_samples(ddim, log_dir, tau, n_samples, weights, seed=GEN_SEED):
    cache_dir = os.path.join(log_dir, 'generated_ema')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'gen_{weights}_tau{tau}_seed{seed}.npz')
    if os.path.exists(cache_path):
        try:
            arr = np.load(cache_path)['channels']
            if arr.shape[0] >= n_samples:
                return arr[:n_samples], True
        except Exception:
            pass
    samples = generate_samples(ddim, n_samples, seed=seed)
    np.savez_compressed(cache_path, channels=samples)
    return samples, False


# ─────────────────────────────────────────────────────────────────────────────
# f_mem: nearest-neighbour ratio + bootstrap CI
# ─────────────────────────────────────────────────────────────────────────────
def nearest_ratio_l2(gen, train, batch_size=512, device=None, eps=1e-12):
    if device is None:
        device = torch.device('cpu')
    train = train.to(device)
    ratios = []
    for start in range(0, len(gen), batch_size):
        g = gen[start:start + batch_size].to(device)
        D = torch.cdist(g, train, p=2)
        top2 = torch.topk(D, k=2, largest=False, dim=1)
        d1 = top2.values[:, 0]
        d2 = top2.values[:, 1]
        ratios.append((d1 / (d2 + eps)).cpu())
    return {"ratios": torch.cat(ratios).numpy()}


def compute_fmem_bootstrap(is_mem, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(is_mem)
    f_mem = is_mem.mean()
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = is_mem[idx].mean()
    return f_mem, np.percentile(boot, 2.5), np.percentile(boot, 97.5)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="MEASURED: FCD(Gen,Test) + f_mem vs τ (shape (2,4,8))")
    ap.add_argument("--sizes", type=int, nargs='+', default=DEFAULT_SIZES)
    ap.add_argument("--debug", action="store_true", help="Few samples + a few τ points")
    args = ap.parse_args()

    num_gen = N_GEN_DEBUG if args.debug else N_GEN
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "measured_fcd_fmem.csv"

    print("=" * 78)
    print("MEASURED DATASET  —  FCD(Gen,Test) + f_mem vs τ   (shape (2,4,8), W=256)")
    print("=" * 78)
    print(f"  Sizes N     : {args.sizes}")
    print(f"  Weights     : {WEIGHTS}  (key='{WEIGHT_KEY}')")
    print(f"  N_GEN       : {num_gen}   Gen seed: {GEN_SEED}")
    print(f"  Test folds  : {N_FOLDS}   f_mem k: {K_MAIN:.4f}")
    print(f"  Feature dim : {2 * SPATIAL_SHAPE[0] * SPATIAL_SHAPE[1]} (BS-only beamspace)")
    print(f"  Output CSV  : {csv_path}")
    print("=" * 78)

    cols = [
        'N', 'batch_size', 'n_feat', 'tau', 'epoch_float', 'weights',
        'num_generated', 'num_train', 'num_test', 'feature_dim', 'n_folds',
        'FCD_Gen_Test', 'FCD_Gen_Test_std',
        'FCD_Train_Test', 'FCD_Train_Test_std',
        'f_mem', 'f_mem_ci_low', 'f_mem_ci_high', 'k',
    ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(cols)

    for n_train in args.sizes:
        log_dir = log_dir_for(n_train)
        print(f"\n{'='*78}\n  N = {n_train}\n  {log_dir}\n{'='*78}")
        if not os.path.isdir(log_dir):
            print("  WARNING: log dir missing — did you train this size yet? Skipping.")
            continue

        ckpts = list_checkpoints(log_dir)
        if not ckpts:
            print("  WARNING: no checkpoints found, skipping.")
            continue
        if args.debug:
            ckpts = [(t, p) for (t, p) in ckpts if t in DEBUG_TAUS] or ckpts[:2]

        steps_per_epoch = ceil(n_train / BATCH_SIZE_TRAIN)

        train_arr = np.load(os.path.join(log_dir, 'train.npy'))    # (N,2,4,8)
        test_arr = np.load(os.path.join(log_dir, 'test.npy'))      # (M,2,4,8)
        feat_train = features_from_spatial_npy(train_arr)          # (N,64)
        feat_test = features_from_spatial_npy(test_arr)            # (M,64)

        ref_stats = build_fold_stats(feat_test)
        mu_train, sig_train = fit_gaussian(feat_train)
        fcd_tt_mean, fcd_tt_std = fcd_vs_reference_folds(mu_train, sig_train, ref_stats)
        train_t = torch.from_numpy(feat_train.astype(np.float32))

        print(f"  Feature dim : {feat_train.shape[1]}")
        print(f"  steps/epoch : {steps_per_epoch}")
        print(f"  Floor FCD(Train,Test) = {fcd_tt_mean:.6f} ± {fcd_tt_std:.6f}")

        ddim = create_model()
        n_hits = 0
        for tau, path in tqdm(ckpts, desc=f"  N={n_train} τ"):
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            if WEIGHT_KEY not in ck:
                print(f"    τ={tau}: '{WEIGHT_KEY}' missing, skip.")
                continue
            ddim.load_state_dict(ck[WEIGHT_KEY])

            samples, hit = get_or_generate_samples(
                ddim, log_dir, tau, num_gen, WEIGHTS, seed=GEN_SEED)
            n_hits += int(hit)

            feat_gen = features_from_generated(samples)            # (num_gen,64)
            mu_g, sig_g = fit_gaussian(feat_gen)
            fcd_gt_mean, fcd_gt_std = fcd_vs_reference_folds(mu_g, sig_g, ref_stats)

            gen_t = torch.from_numpy(feat_gen.astype(np.float32))
            nn = nearest_ratio_l2(gen_t, train_t,
                                  batch_size=NN_BATCH_SIZE, device=DEVICE)
            is_mem = (nn["ratios"] < K_MAIN)
            f_mem, fmem_lo, fmem_hi = compute_fmem_bootstrap(
                is_mem, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    n_train, BATCH_SIZE_TRAIN, N_FEAT, tau, tau / steps_per_epoch, WEIGHTS,
                    num_gen, len(feat_train), len(feat_test),
                    feat_train.shape[1], N_FOLDS,
                    fcd_gt_mean, fcd_gt_std, fcd_tt_mean, fcd_tt_std,
                    float(f_mem), float(fmem_lo), float(fmem_hi), K_MAIN,
                ])

        print(f"  Cache hits: {n_hits}/{len(ckpts)}  "
              f"(EMA samples in {os.path.join(log_dir, 'generated_ema')})")

    print(f"\n{'='*78}")
    print("Done.")
    print(f"  CSV : {csv_path}")
    print(f"  Next: python plot_measured_fcd_fmem.py")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
