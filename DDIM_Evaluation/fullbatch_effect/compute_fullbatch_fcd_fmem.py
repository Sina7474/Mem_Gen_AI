"""
compute_fullbatch_fcd_fmem.py
    — Full-batch sweep: FCD(Gen,Test) + f_mem vs τ, with batch size = dataset size
================================================================================
PURPOSE
-------
Here we vary the dataset size N ∈ {200, 1000, 4000} at a FIXED model width
(n_feat = 256), and for EACH N we set the training mini-batch size **equal to N**:

      N = 200   →  batch size = 200   (full-batch gradient descent)
      N = 1000  →  batch size = 1000  (full-batch gradient descent)
      N = 4000  →  batch size = 4000  (full-batch gradient descent)

Because batch = N, every optimizer step is a **full pass over the whole training
set** (steps_per_epoch = ⌈N/N⌉ = 1), so here τ (optimizer steps) equals the number
of epochs / data repetitions. This is the "pure" regime where every gradient is the
exact full-dataset gradient (no mini-batch stochasticity), letting us study
memorization vs. dataset size decoupled from gradient noise.

For every N (and its matched batch size) and every τ checkpoint this script computes,
from the SAME set of EMA-generated samples, three paper-style curves:

  • FCD(Gen, Test)   — Fréchet Channel Distance (quality, ↓, want it to saturate)
  • FCD(Train, Test) — the real–real floor (irreducible finite-sample distance);
                       this differs per N because each N has a different train set.
  • f_mem(τ)         — memorization fraction (nearest-neighbour ratio test, k=1/3)

Both metrics use the IDENTICAL 256-D representation (per-sample-normalized UPA-DFT
beamspace, flattened) and the IDENTICAL generated samples, so the FCD and f_mem
curves for a given N are mutually self-consistent, and the three N values are
mutually comparable on the shared τ axis.

DATA REUSE (no wasted GPU work)
-------------------------------
Each (N, batch=N) run lives in its own log directory produced by the trainer:
    N = 200   → logs_ema/DDIM_tau_ema_200_bs200_incremental
    N = 1000  → logs_ema/DDIM_tau_ema_1000_bs1000_incremental
    N = 4000  → logs_ema/DDIM_tau_ema_4000_bs4000_incremental
Generated EMA samples are cached in each run's generated_ema/ folder on first use
and reused verbatim on any re-run.

No new metric logic is implemented here — all heavy lifting is imported verbatim
from the already-validated parent scripts:
    ../compute_fcd_vs_tau.py   (FCD + generation/caching)
    ../compute_fmem_vs_tau.py  (nearest-neighbour ratio → f_mem)

OUTPUT
------
    results/fullbatch_fcd_fmem.csv   (one row per N × τ checkpoint)

USAGE
-----
    conda activate Mem_Gen
    cd DDIM_Evaluation/fullbatch_effect
    python compute_fullbatch_fcd_fmem.py                 # N = 200 1000 4000
    python compute_fullbatch_fcd_fmem.py --sizes 1000    # a single size
    python compute_fullbatch_fcd_fmem.py --debug         # quick smoke test
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


_fcd  = _load_module('_fcd_mod_fb',  os.path.join(_EVAL_DIR, 'compute_fcd_vs_tau.py'))
_fmem = _load_module('_fmem_mod_fb', os.path.join(_EVAL_DIR, 'compute_fmem_vs_tau.py'))

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
DEFAULT_SIZES = [200, 1000, 4000]          # for each N, batch size = N
WEIGHTS       = 'ema'                       # sample from the EMA weights
WEIGHT_KEY    = 'ema_model_state_dict'
OUTPUT_DIR    = Path("results")


def log_dir_for(n_train):
    """Log directory produced by train_DDIM_tau_ema.py for the full-batch run of N.

    Full-batch means batch size == N, so the trainer writes a `_bs<N>` suffix.
    """
    return os.path.join(_LOGS_EMA, f"DDIM_tau_ema_{n_train}_bs{n_train}_incremental")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Full-batch sweep (batch = N): FCD(Gen,Test) + f_mem vs τ")
    ap.add_argument("--sizes", type=int, nargs='+', default=DEFAULT_SIZES,
                    help=f"Dataset sizes; batch size is set equal to each "
                         f"(default: {DEFAULT_SIZES})")
    ap.add_argument("--debug", action="store_true",
                    help="Few samples + a handful of τ points (quick smoke test)")
    args = ap.parse_args()

    num_gen = N_GEN_DEBUG if args.debug else N_GEN
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "fullbatch_fcd_fmem.csv"

    print("=" * 78)
    print("FULL-BATCH SWEEP (batch size = dataset size)  —  FCD(Gen,Test) + f_mem vs τ")
    print("=" * 78)
    print(f"  Sizes (=batch): {args.sizes}")
    print(f"  Model width   : {N_FEAT}")
    print(f"  Weights       : {WEIGHTS}  (key='{WEIGHT_KEY}')")
    print(f"  N_GEN         : {num_gen}   Gen seed: {GEN_SEED}")
    print(f"  Test folds    : {N_FOLDS}  (→ 2σ error bars on FCD)")
    print(f"  f_mem k       : {K_MAIN:.4f}  (nearest-neighbour ratio test)")
    print(f"  Output CSV    : {csv_path}")
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

    for n_train in args.sizes:
        bs = n_train                                   # full-batch: batch size = N
        log_dir = log_dir_for(n_train)
        print(f"\n{'='*78}\n  N = {n_train}   (batch size = {bs}, full-batch)\n"
              f"  {log_dir}\n{'='*78}")
        if not os.path.isdir(log_dir):
            print("  WARNING: log dir missing — did you train this size yet? Skipping.")
            continue

        ckpts = list_checkpoints(log_dir)
        if not ckpts:
            print("  WARNING: no checkpoints found, skipping.")
            continue
        if args.debug:
            ckpts = [(t, p) for (t, p) in ckpts if t in DEBUG_TAUS] or ckpts[:2]

        steps_per_epoch = ceil(n_train / bs)           # == 1 for full-batch

        # ── Reference features: spatial → beamspace → per-sample norm → flat ──
        train_arr = np.load(os.path.join(log_dir, 'train.npy'))
        test_arr  = np.load(os.path.join(log_dir, 'test.npy'))
        feat_train = features_from_spatial_npy(train_arr)          # (N,256)
        feat_test  = features_from_spatial_npy(test_arr)           # (M,256)

        # 5 disjoint TEST folds → FCD error bars + the real–real floor (per N)
        ref_stats = build_fold_stats(feat_test)
        mu_train, sig_train = fit_gaussian(feat_train)
        fcd_tt_mean, fcd_tt_std = fcd_vs_reference_folds(mu_train, sig_train, ref_stats)

        # f_mem needs the train reference as a torch tensor in the same 256-D space
        train_t = torch.from_numpy(feat_train.astype(np.float32))

        print(f"  Feature dim   : {feat_train.shape[1]}")
        print(f"  steps/epoch   : {steps_per_epoch}  (= ceil(N/bs) = 1 for full-batch)")
        print(f"  Floor FCD(Train,Test) = {fcd_tt_mean:.6f} ± {fcd_tt_std:.6f}")

        ddim = create_model()
        n_hits = 0
        for tau, path in tqdm(ckpts, desc=f"  N={n_train} τ"):
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
                    n_train, bs, N_FEAT, tau, tau / steps_per_epoch, WEIGHTS,
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
    print(f"  Next: python plot_fullbatch_fcd_fmem.py")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
