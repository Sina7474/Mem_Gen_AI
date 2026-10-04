"""
compute_dsize_fcd_fmem.py
    — Dataset-size sweep: FCD(Gen,Test) + f_mem vs τ, at fixed width W = 256
      with the training batch size capped by the rule  B = min(N, 500).
================================================================================
PURPOSE
-------
This is the *canonical scaling experiment*: we hold the model width fixed
(n_feat = 256) and sweep the dataset size

        N ∈ {200, 500, 1000, 2000, 4000}

For every N the training mini-batch size follows the fixed protocol

        B = min(N, 500)

i.e.

        N = 200   →  B = 200
        N = 500   →  B = 500
        N = 1000  →  B = 500
        N = 2000  →  B = 500
        N = 4000  →  B = 500

Rationale for the rule.  For the small sets (N < 500) a full-batch update is the
natural, lowest-noise choice, so B = N. Once N ≥ 500 we FREEZE the batch at 500 so
that every larger set is optimised with the *same* per-step gradient-noise scale.
This isolates the effect of the DATASET SIZE itself (the paper's t_mem ∝ N claim)
from the confound of a changing gradient-noise level, which a growing batch would
otherwise introduce.

For every N (and its matched batch size) and every τ checkpoint this script
computes, from the SAME set of EMA-generated samples, three paper-style curves:

  • FCD(Gen, Test)   — Fréchet Channel Distance (quality, ↓, want it to saturate)
  • FCD(Train, Test) — the real–real floor (irreducible finite-sample distance);
                       this differs per N because each N has a different train set.
  • f_mem(τ)         — memorization fraction (nearest-neighbour ratio test, k=1/3)

Both metrics use the IDENTICAL 256-D representation (per-sample-normalized UPA-DFT
beamspace, flattened) and the IDENTICAL generated samples, so the FCD and f_mem
curves for a given N are mutually self-consistent, and the five N values are
mutually comparable on the shared τ axis.

DATA REUSE (no wasted GPU work)
-------------------------------
Each (N, B=min(N,500)) run lives in its own log directory produced by the trainer:
    N = 200   (B=200) → logs_ema/DDIM_tau_ema_200_bs200_incremental   ← already
                        exists (identical to the N=200 full-batch run — REUSED)
    N = 500   (B=500) → logs_ema/DDIM_tau_ema_500_bs500_incremental
    N = 1000  (B=500) → logs_ema/DDIM_tau_ema_1000_bs500_incremental
    N = 2000  (B=500) → logs_ema/DDIM_tau_ema_2000_bs500_incremental
    N = 4000  (B=500) → logs_ema/DDIM_tau_ema_4000_bs500_incremental
Generated EMA samples are cached in each run's generated_ema/ folder on first use
and reused verbatim on any re-run.

No new metric logic is implemented here — all heavy lifting is imported verbatim
from the already-validated parent scripts:
    ../compute_fcd_vs_tau.py   (FCD + generation/caching)
    ../compute_fmem_vs_tau.py  (nearest-neighbour ratio → f_mem)

OUTPUT
------
    results/dsize_fcd_fmem.csv   (one row per N × τ checkpoint)

USAGE
-----
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect
    python compute_dsize_fcd_fmem.py                     # N = 200 500 1000 2000 4000
    python compute_dsize_fcd_fmem.py --sizes 1000        # a single size
    python compute_dsize_fcd_fmem.py --debug             # quick smoke test
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


_fcd  = _load_module('_fcd_mod_ds',  os.path.join(_EVAL_DIR, 'compute_fcd_vs_tau.py'))
_fmem = _load_module('_fmem_mod_ds', os.path.join(_EVAL_DIR, 'compute_fmem_vs_tau.py'))

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
compute_fmem_bootstrap = _fmem.compute_fmem_bootstrap   # 95% bootstrap CI for f_mem
BOOTSTRAP_B      = _fmem.BOOTSTRAP_B       # 1000 resamples
BOOTSTRAP_SEED   = _fmem.BOOTSTRAP_SEED    # 42


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SIZES   = [200, 500, 1000, 2000, 4000]   # swept dataset sizes
BATCH_CAP       = 500                             # B = min(N, BATCH_CAP)
TRAINER_DEFAULT_BS = 100                          # trainer's default (→ no _bs suffix)
WEIGHTS         = 'ema'                            # sample from the EMA weights
WEIGHT_KEY      = 'ema_model_state_dict'
OUTPUT_DIR      = Path("results")


def batch_for(n_train):
    """The fixed batch-size protocol for this experiment:  B = min(N, 500)."""
    return min(n_train, BATCH_CAP)


def log_dir_for(n_train):
    """Log directory produced by train_DDIM_tau_ema.py for this experiment's run.

    Width is fixed at 256 (the trainer default → no `_nfeat` suffix). The batch
    size is B = min(N, 500); since that is never the trainer default (100), every
    run carries a `_bs<B>` suffix.
    """
    bs = batch_for(n_train)
    bs_suffix = '' if bs == TRAINER_DEFAULT_BS else f'_bs{bs}'
    return os.path.join(_LOGS_EMA, f"DDIM_tau_ema_{n_train}{bs_suffix}_incremental")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Dataset-size sweep (W=256, B=min(N,500)): "
                    "FCD(Gen,Test) + f_mem vs τ")
    ap.add_argument("--sizes", type=int, nargs='+', default=DEFAULT_SIZES,
                    help=f"Dataset sizes to include (default: {DEFAULT_SIZES}). "
                         f"Batch size for each is min(N, {BATCH_CAP}).")
    ap.add_argument("--debug", action="store_true",
                    help="Few samples + a handful of τ points (quick smoke test)")
    ap.add_argument("--append", action="store_true",
                    help="Append to the existing dsize_fcd_fmem.csv instead of "
                         "overwriting it. Any rows for the N values being "
                         "(re)computed are dropped first, then re-added — so the "
                         "other sizes already in the CSV are preserved. Use this "
                         "when adding a new size (e.g. --sizes 100 --append).")
    args = ap.parse_args()

    num_gen = N_GEN_DEBUG if args.debug else N_GEN
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "dsize_fcd_fmem.csv"

    print("=" * 78)
    print("DATASET-SIZE SWEEP  (W=256, B=min(N,500))  —  FCD(Gen,Test) + f_mem vs τ")
    print("=" * 78)
    print(f"  Sizes N       : {args.sizes}")
    print(f"  Batch rule    : B = min(N, {BATCH_CAP})  → "
          f"{ {n: batch_for(n) for n in args.sizes} }")
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
        'f_mem', 'f_mem_ci_low', 'f_mem_ci_high', 'k',
    ]
    if args.append and csv_path.exists():
        # Preserve rows for every OTHER size; drop rows for the sizes we are about
        # to (re)compute so re-runs are idempotent. Header is taken from the file.
        import pandas as _pd
        _old = _pd.read_csv(csv_path)
        _kept = _old[~_old['N'].isin(args.sizes)]
        _kept.to_csv(csv_path, index=False)
        print(f"  Append mode: kept {len(_kept)} existing rows for sizes "
              f"{sorted(_kept['N'].unique())}; (re)computing {args.sizes}.")
    else:
        if args.append:
            print("  Append requested but no CSV exists yet → creating a new one.")
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(cols)

    for n_train in args.sizes:
        bs = batch_for(n_train)
        log_dir = log_dir_for(n_train)
        print(f"\n{'='*78}\n  N = {n_train}   (batch size = {bs} = min(N,{BATCH_CAP}))\n"
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

        steps_per_epoch = ceil(n_train / bs)

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
        print(f"  steps/epoch   : {steps_per_epoch}  (= ceil(N/bs))")
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
            is_mem = (nn["ratios"] < K_MAIN)
            # f_mem point estimate + 95% bootstrap CI over the per-sample flags
            # (same procedure as ../compute_fmem_vs_tau.py). Samples are cached,
            # so this adds only the cheap resampling pass — no regeneration.
            f_mem, fmem_ci_low, fmem_ci_high = compute_fmem_bootstrap(
                is_mem, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
            f_mem = float(f_mem)

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    n_train, bs, N_FEAT, tau, tau / steps_per_epoch, WEIGHTS,
                    num_gen, len(feat_train), len(feat_test),
                    feat_train.shape[1], N_FOLDS,
                    fcd_gt_mean, fcd_gt_std, fcd_tt_mean, fcd_tt_std,
                    f_mem, float(fmem_ci_low), float(fmem_ci_high), K_MAIN,
                ])

        print(f"  Cache hits: {n_hits}/{len(ckpts)}  "
              f"(EMA samples in {os.path.join(log_dir, 'generated_ema')})")

    print(f"\n{'='*78}")
    print("Done.")
    print(f"  CSV : {csv_path}")
    print(f"  Next: python plot_dsize_fcd_fmem.py")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
