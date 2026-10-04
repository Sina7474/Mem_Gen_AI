"""
compute_bs_fcd_fmem.py
    — Batch-size sweep: FCD(Gen,Test) + f_mem vs τ for a fixed N (default N=1000)
================================================================================
PURPOSE
-------
We keep the dataset size N, the model width (n_feat=256) and the τ grid FIXED and
vary ONLY the training mini-batch size (100, 512, 1024). Because τ is measured in
optimizer steps, and the dense τ grid is defined in step units, the τ axis stays
directly comparable across batch sizes — the *only* thing that changes is how many
samples the model saw per gradient update.

For every batch size and every τ checkpoint this script computes, from the SAME
set of EMA-generated samples, two paper-style curves:

  • FCD(Gen, Test)   — Fréchet Channel Distance (quality, ↓, want it to saturate)
  • FCD(Train, Test) — the real–real floor (irreducible finite-sample distance)
  • f_mem(τ)         — memorization fraction (nearest-neighbour ratio test, k=1/3)

Both metrics use the IDENTICAL 256-D representation (per-sample-normalized UPA-DFT
beamspace, flattened) and the IDENTICAL generated samples, so the FCD and f_mem
curves for a given batch size are mutually self-consistent, and the three batch
sizes are mutually comparable.

DATA REUSE (no wasted GPU work)
-------------------------------
  • batch size 100  → already trained: logs_ema/DDIM_tau_ema_1000_incremental
                      (its EMA samples are already cached in generated_ema/ and are
                      REUSED verbatim — nothing is regenerated).
  • batch size 512  → logs_ema/DDIM_tau_ema_1000_bs512_incremental   (you train this)
  • batch size 1024 → logs_ema/DDIM_tau_ema_1000_bs1024_incremental  (you train this)

All heavy lifting (feature extraction, Fréchet math, generation + caching, and the
nearest-neighbour ratio) is imported verbatim from the existing, already-validated
scripts so this file adds NO new metric logic:
    ../compute_fcd_vs_tau.py   (FCD + generation/caching)
    ../compute_fmem_vs_tau.py  (nearest-neighbour ratio → f_mem)

OUTPUT
------
    results/bs_fcd_fmem_N{N}.csv   (one row per batch size × τ checkpoint)

USAGE
-----
    conda activate Mem_Gen
    cd DDIM_Evaluation/batch_size_effect
    python compute_bs_fcd_fmem.py                       # N=1000, bs 100 512 1024
    python compute_bs_fcd_fmem.py --batch_sizes 512 1024
    python compute_bs_fcd_fmem.py --debug               # quick smoke test
"""

import os
import sys
import csv
import argparse
import importlib.util
from math import ceil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Locate & import the two validated metric modules (no logic is re-implemented)
# ─────────────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR  = os.path.join(_HERE, '..')                 # DDIM_Evaluation/
_DDIM_DIR  = os.path.join(_EVAL_DIR, '..', 'Code', 'DDIM_FMM')
_LOGS_EMA  = os.path.join(_DDIM_DIR, 'logs_ema')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fcd  = _load_module('_fcd_mod',  os.path.join(_EVAL_DIR, 'compute_fcd_vs_tau.py'))
_fmem = _load_module('_fmem_mod', os.path.join(_EVAL_DIR, 'compute_fmem_vs_tau.py'))

# FCD side (identical protocol as compute_fcd_vs_tau.py) ----------------------
features_from_spatial_npy = _fcd.features_from_spatial_npy
features_from_generated   = _fcd.features_from_generated
fit_gaussian              = _fcd.fit_gaussian
build_fold_stats          = _fcd.build_fold_stats
fcd_vs_reference_folds    = _fcd.fcd_vs_reference_folds
create_model              = _fcd.create_model
list_checkpoints          = _fcd.list_checkpoints
get_or_generate_samples   = _fcd.get_or_generate_samples
N_FEAT   = _fcd.N_FEAT
N_GEN    = _fcd.N_GEN
N_GEN_DEBUG = _fcd.N_GEN_DEBUG
N_FOLDS  = _fcd.N_FOLDS
GEN_SEED = _fcd.GEN_SEED
DEVICE   = _fcd.DEVICE
DEBUG_TAUS = _fcd.DEBUG_TAUS

# f_mem side (identical nearest-neighbour ratio as compute_fmem_vs_tau.py) ----
nearest_ratio_l2 = _fmem.nearest_ratio_l2
K_MAIN           = _fmem.K_MAIN            # 1/3
NN_BATCH_SIZE    = _fmem.NN_BATCH_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_N            = 1000
DEFAULT_BATCH_SIZES  = [100, 512, 1024]
BASELINE_BATCH_SIZE  = 100                 # the run that already exists
WEIGHTS              = 'ema'               # sample from the EMA weights
WEIGHT_KEY           = 'ema_model_state_dict'
OUTPUT_DIR           = Path("results")


def log_dir_for(n_train, batch_size):
    """Log directory produced by train_DDIM_tau_ema.py for a given (N, batch_size).

    batch_size == 100 is the trainer default → no suffix (the original run);
    other batch sizes get a `_bs<SIZE>` suffix (kept in a separate directory).
    """
    bs_suffix = '' if batch_size == BASELINE_BATCH_SIZE else f'_bs{batch_size}'
    return os.path.join(_LOGS_EMA, f"DDIM_tau_ema_{n_train}{bs_suffix}_incremental")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Batch-size sweep: FCD(Gen,Test) + f_mem vs τ for fixed N")
    ap.add_argument("--N", type=int, default=DEFAULT_N,
                    help=f"Training set size (default: {DEFAULT_N})")
    ap.add_argument("--batch_sizes", type=int, nargs='+', default=DEFAULT_BATCH_SIZES,
                    help=f"Batch sizes to include (default: {DEFAULT_BATCH_SIZES})")
    ap.add_argument("--debug", action="store_true",
                    help="Few samples + a handful of τ points (quick smoke test)")
    args = ap.parse_args()

    num_gen = N_GEN_DEBUG if args.debug else N_GEN
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"bs_fcd_fmem_N{args.N}.csv"

    print("=" * 78)
    print("BATCH-SIZE SWEEP  —  FCD(Gen,Test) + f_mem vs τ")
    print("=" * 78)
    print(f"  N            : {args.N}")
    print(f"  Batch sizes  : {args.batch_sizes}")
    print(f"  Model width  : {N_FEAT}")
    print(f"  Weights      : {WEIGHTS}  (key='{WEIGHT_KEY}')")
    print(f"  N_GEN        : {num_gen}   Gen seed: {GEN_SEED}")
    print(f"  Test folds   : {N_FOLDS}  (→ 2σ error bars on FCD)")
    print(f"  f_mem k      : {K_MAIN:.4f}  (nearest-neighbour ratio test)")
    print(f"  Output CSV   : {csv_path}")
    print("=" * 78)

    cols = [
        'N', 'batch_size', 'n_feat', 'tau', 'epoch_float', 'weights',
        'num_generated', 'num_train', 'num_test', 'feature_dim', 'n_folds',
        'FCD_Gen_Test', 'FCD_Gen_Test_std',
        'FCD_Train_Test', 'FCD_Train_Test_std',
        'f_mem', 'k',
    ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(cols)

    for bs in args.batch_sizes:
        log_dir = log_dir_for(args.N, bs)
        print(f"\n{'='*78}\n  batch_size = {bs}\n  {log_dir}\n{'='*78}")
        if not os.path.isdir(log_dir):
            print("  WARNING: log dir missing — did you train this batch size yet? Skipping.")
            continue

        ckpts = list_checkpoints(log_dir)
        if not ckpts:
            print("  WARNING: no checkpoints found, skipping.")
            continue
        if args.debug:
            ckpts = [(t, p) for (t, p) in ckpts if t in DEBUG_TAUS] or ckpts[:2]

        steps_per_epoch = ceil(args.N / bs)

        # ── Reference features: spatial → beamspace → per-sample norm → flat ──
        train_arr = np.load(os.path.join(log_dir, 'train.npy'))
        test_arr  = np.load(os.path.join(log_dir, 'test.npy'))
        feat_train = features_from_spatial_npy(train_arr)          # (N,256)
        feat_test  = features_from_spatial_npy(test_arr)           # (M,256)

        # 5 disjoint TEST folds → FCD error bars + the real–real floor
        ref_stats = build_fold_stats(feat_test)
        mu_train, sig_train = fit_gaussian(feat_train)
        fcd_tt_mean, fcd_tt_std = fcd_vs_reference_folds(mu_train, sig_train, ref_stats)

        # f_mem needs the train reference as a torch tensor in the same 256-D space
        train_t = torch.from_numpy(feat_train.astype(np.float32))

        print(f"  Feature dim   : {feat_train.shape[1]}")
        print(f"  steps/epoch   : {steps_per_epoch}  (= ceil(N/bs))")
        print(f"  Floor FCD(Train,Test) = {fcd_tt_mean:.6f} ± {fcd_tt_std:.6f}")

        ddim = create_model()
        n_hits = 0
        for tau, path in tqdm(ckpts, desc=f"  bs={bs} τ"):
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            if WEIGHT_KEY not in ck:
                print(f"    τ={tau}: '{WEIGHT_KEY}' missing, skip.")
                continue
            ddim.load_state_dict(ck[WEIGHT_KEY])

            # Reuse cached EMA samples if present, else generate once & cache.
            samples, hit = get_or_generate_samples(
                ddim, log_dir, tau, num_gen, WEIGHTS, seed=GEN_SEED)
            n_hits += int(hit)

            # ── FCD(Gen,Test) on the 5 test folds ─────────────────────────────
            feat_gen = features_from_generated(samples)            # (num_gen,256)
            mu_g, sig_g = fit_gaussian(feat_gen)
            fcd_gt_mean, fcd_gt_std = fcd_vs_reference_folds(mu_g, sig_g, ref_stats)

            # ── f_mem on the SAME generated samples (nearest-neighbour ratio) ─
            gen_t = torch.from_numpy(feat_gen.astype(np.float32))
            nn = nearest_ratio_l2(gen_t, train_t,
                                  batch_size=NN_BATCH_SIZE, device=DEVICE)
            f_mem = float((nn["ratios"] < K_MAIN).mean())

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    args.N, bs, N_FEAT, tau, tau / steps_per_epoch, WEIGHTS,
                    num_gen, len(feat_train), len(feat_test),
                    feat_train.shape[1], N_FOLDS,
                    fcd_gt_mean, fcd_gt_std, fcd_tt_mean, fcd_tt_std,
                    f_mem, K_MAIN,
                ])

        print(f"  Cache hits: {n_hits}/{len(ckpts)}  "
              f"(EMA samples in {os.path.join(log_dir, 'generated_ema')})")

    print(f"\n{'='*78}")
    print("Done.")
    print(f"  CSV : {csv_path}")
    print(f"  Next: python plot_bs_fcd_fmem.py --N {args.N}")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
