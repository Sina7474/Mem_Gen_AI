"""
compute_wsize_fcd_fmem.py
    — Model-size sweep: FCD(Gen,Test) + f_mem vs τ, at a LIMITED set of dataset
      sizes, while sweeping the U-Net width W (= n_feat).
================================================================================
PURPOSE
-------
The companion to ../dataset_size_effect/. There we held the width fixed (W = 256)
and swept the dataset size N. Here we do the ORTHOGONAL experiment: we hold the
dataset size at a small, fixed set and sweep the MODEL CAPACITY (width W), to see
how capacity alone moves sample quality (FCD) and memorization (f_mem).

        N ∈ {200, 1000}          (limited: two representative dataset sizes)
        W ∈ {64, 128, 256}       (three U-Net widths / capacities)

The training mini-batch size follows the SAME rule as the dataset-size sweep,

        B = min(N, 500)          →   N = 200  → B = 200
                                     N = 1000 → B = 500

so the two experiments are mutually consistent and the W = 256 runs are literally
the SAME checkpoints already used in ../dataset_size_effect/ (nothing is retrained).

TRAINING BUDGET (τ_max) PER CONFIG
----------------------------------
The original width sweep (N ∈ {200, 1000}) plus the PHASE-DIAGRAM completion runs
(N ∈ {500, 2000, 4000} for the narrow widths W ∈ {64, 128}). τ_max grows with N so
every curve reaches/traverses the memorization regime, and each N uses the SAME
τ_max as its W = 256 counterpart in ../dataset_size_effect/ so the τ grids are
per-N identical and directly comparable:

        (N=50  , W=64 )  → τ_max = 200000     ← phase-diagram run
        (N=50  , W=128)  → τ_max = 200000     ← phase-diagram run
        (N=50  , W=256)  → τ_max = 200000     ← phase-diagram run
        (N=200 , W=64 )  → τ_max = 200000
        (N=200 , W=128)  → τ_max = 200000
        (N=200 , W=256)  → τ_max = 200000     (reused from dataset_size_effect)
        (N=500 , W=64 )  → τ_max = 200000     ← phase-diagram run
        (N=500 , W=128)  → τ_max = 200000     ← phase-diagram run
        (N=500 , W=256)  → τ_max = 200000     (reused from dataset_size_effect)
        (N=1000, W=64 )  → τ_max = 400000     ← extended
        (N=1000, W=128)  → τ_max = 400000     ← extended
        (N=1000, W=256)  → τ_max = 200000     (reused from dataset_size_effect)
        (N=2000, W=64 )  → τ_max = 300000     ← phase-diagram run
        (N=2000, W=128)  → τ_max = 300000     ← phase-diagram run
        (N=2000, W=256)  → τ_max = 300000     (reused from dataset_size_effect)
        (N=4000, W=64 )  → τ_max = 400000     ← phase-diagram run
        (N=4000, W=128)  → τ_max = 400000     ← phase-diagram run
        (N=4000, W=256)  → τ_max = 400000     (reused from dataset_size_effect)

The τ_max only bounds how far each curve extends; this script simply reads every
checkpoint that exists in each run's log directory, and silently skips any config
whose log directory has not been trained yet (WARNING + continue).

For every (N, W) config and every τ checkpoint this script computes, from the SAME
set of EMA-generated samples, three paper-style curves:

  • FCD(Gen, Test)   — Fréchet Channel Distance (quality, ↓, want it to saturate)
  • FCD(Train, Test) — the real–real floor (irreducible finite-sample distance);
                       this depends on N only (same train set for all W at a given N)
  • f_mem(τ)         — memorization fraction (nearest-neighbour ratio test, k=1/3)
                       with a 95% bootstrap CI (columns f_mem_ci_low / f_mem_ci_high)

Both metrics use the IDENTICAL 256-D representation (per-sample-normalized UPA-DFT
beamspace, flattened) and the IDENTICAL generated samples, so the FCD and f_mem
curves for a given (N, W) are mutually self-consistent, and all configs are
comparable on the shared τ axis.

CRITICAL — WIDTH-AWARE MODEL BUILD
----------------------------------
Unlike ../dataset_size_effect/ (which could reuse the parent's fixed-width
`create_model`), here the network width changes per config, so this script builds
the DDIM/U-Net with the CORRECT `n_feat = W` before loading each checkpoint. A
width mismatch would fail to load the state dict, so this is checked implicitly by
a successful `load_state_dict`.

DATA REUSE (no wasted GPU work)
-------------------------------
Each (N, W, B=min(N,500)) run lives in its own trainer log directory:
    (50  , 64 ) → logs_ema/DDIM_tau_ema_50_nfeat64_bs50_incremental
    (50  , 128) → logs_ema/DDIM_tau_ema_50_nfeat128_bs50_incremental
    (50  , 256) → logs_ema/DDIM_tau_ema_50_bs50_incremental
    (200 , 64 ) → logs_ema/DDIM_tau_ema_200_nfeat64_bs200_incremental
    (200 , 128) → logs_ema/DDIM_tau_ema_200_nfeat128_bs200_incremental
    (200 , 256) → logs_ema/DDIM_tau_ema_200_bs200_incremental          (REUSED)
    (500 , 64 ) → logs_ema/DDIM_tau_ema_500_nfeat64_bs500_incremental
    (500 , 128) → logs_ema/DDIM_tau_ema_500_nfeat128_bs500_incremental
    (500 , 256) → logs_ema/DDIM_tau_ema_500_bs500_incremental          (REUSED)
    (1000, 64 ) → logs_ema/DDIM_tau_ema_1000_nfeat64_bs500_incremental
    (1000, 128) → logs_ema/DDIM_tau_ema_1000_nfeat128_bs500_incremental
    (1000, 256) → logs_ema/DDIM_tau_ema_1000_bs500_incremental         (REUSED)
    (2000, 64 ) → logs_ema/DDIM_tau_ema_2000_nfeat64_bs500_incremental
    (2000, 128) → logs_ema/DDIM_tau_ema_2000_nfeat128_bs500_incremental
    (2000, 256) → logs_ema/DDIM_tau_ema_2000_bs500_incremental         (REUSED)
    (4000, 64 ) → logs_ema/DDIM_tau_ema_4000_nfeat64_bs500_incremental
    (4000, 128) → logs_ema/DDIM_tau_ema_4000_nfeat128_bs500_incremental
    (4000, 256) → logs_ema/DDIM_tau_ema_4000_bs500_incremental         (REUSED)
Note: width W = 256 is the trainer default, so those dirs carry NO `_nfeat` suffix.
Generated EMA samples are cached in each run's generated_ema/ folder on first use
and reused verbatim on any re-run.

No new metric logic is implemented here — all heavy lifting is imported verbatim
from the already-validated parent scripts:
    ../compute_fcd_vs_tau.py   (FCD + generation/caching + beamspace features)
    ../compute_fmem_vs_tau.py  (nearest-neighbour ratio → f_mem + bootstrap CI)

OUTPUT
------
    results/wsize_fcd_fmem.csv   (one row per (N, W) × τ checkpoint)

USAGE
-----
    conda activate Mem_Gen
    cd DDIM_Evaluation/model_size_effect
    python compute_wsize_fcd_fmem.py                       # all 6 (N, W) configs
    python compute_wsize_fcd_fmem.py --configs 1000:64     # a single config
    python compute_wsize_fcd_fmem.py --configs 200:64 200:128 200:256
    python compute_wsize_fcd_fmem.py --debug               # quick smoke test
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


_fcd  = _load_module('_fcd_mod_ws',  os.path.join(_EVAL_DIR, 'compute_fcd_vs_tau.py'))
_fmem = _load_module('_fmem_mod_ws', os.path.join(_EVAL_DIR, 'compute_fmem_vs_tau.py'))

# FCD side (identical protocol as compute_fcd_vs_tau.py) ----------------------
features_from_spatial_npy = _fcd.features_from_spatial_npy
features_from_generated   = _fcd.features_from_generated
fit_gaussian              = _fcd.fit_gaussian
build_fold_stats          = _fcd.build_fold_stats
fcd_vs_reference_folds    = _fcd.fcd_vs_reference_folds
list_checkpoints          = _fcd.list_checkpoints
get_or_generate_samples   = _fcd.get_or_generate_samples
N_GEN    = _fcd.N_GEN
N_GEN_DEBUG = _fcd.N_GEN_DEBUG
N_FOLDS  = _fcd.N_FOLDS
GEN_SEED = _fcd.GEN_SEED
DEVICE   = _fcd.DEVICE
DEBUG_TAUS = _fcd.DEBUG_TAUS

# Width-aware model factory (the parent create_model() hardwires N_FEAT=256, which
# is wrong here because the width VARIES per config). Build the U-Net with n_feat=W.
Unet  = _fcd.Unet
DDIM  = _fcd.DDIM
BETAS = _fcd.BETAS
N_T   = _fcd.N_T


def create_model(n_feat):
    """DDIM/U-Net built with the requested base width (n_feat = W)."""
    ddim = DDIM(nn_model=Unet(in_channels=2, n_feat=n_feat),
                betas=BETAS, n_T=N_T, device=DEVICE)
    ddim.to(DEVICE)
    return ddim


def load_checkpoint_model(path, width):
    """Build an isolated model and load one EMA checkpoint.

    Checkpoints also contain the raw model and Adam optimizer state.  Loading the
    whole checkpoint directly onto the GPU needlessly keeps all of those tensors
    resident during sampling.  Load on CPU, copy only the EMA state into a fresh
    width-aware model, and release the checkpoint before generation.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if WEIGHT_KEY not in ck:
        return None

    state = ck[WEIGHT_KEY]
    width_key = "nn_model.init_conv.conv1.0.weight"
    if width_key not in state:
        raise KeyError(
            f"Cannot verify model width: '{width_key}' is missing from {path}")
    checkpoint_width = int(state[width_key].shape[0])
    if checkpoint_width != width:
        raise ValueError(
            f"Checkpoint/model width mismatch for {path}: "
            f"checkpoint W={checkpoint_width}, requested W={width}")

    ddim = create_model(width)
    ddim.load_state_dict(state, strict=True)
    del state, ck
    return ddim


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
DEFAULT_N      = [50, 200, 500, 1000, 2000, 4000]   # full phase-diagram grid
DEFAULT_W      = [64, 128, 256]           # swept model widths
BATCH_CAP      = 500                      # B = min(N, BATCH_CAP)
TRAINER_DEFAULT_W  = 256                  # trainer default width (→ no _nfeat suffix)
WEIGHTS        = 'ema'                    # sample from the EMA weights
WEIGHT_KEY     = 'ema_model_state_dict'
OUTPUT_DIR     = Path("results")


def default_configs():
    """All (N, W) pairs of the sweep, as a flat ordered list of tuples."""
    return [(n, w) for n in DEFAULT_N for w in DEFAULT_W]


def batch_for(n_train):
    """The fixed batch-size protocol, shared with dataset_size_effect: B=min(N,500)."""
    return min(n_train, BATCH_CAP)


def log_dir_for(n_train, width):
    """Log directory produced by train_DDIM_tau_ema.py for this (N, W) run.

    Width W = 256 is the trainer default → NO `_nfeat` suffix. Other widths carry
    `_nfeat<W>`. The batch size is B = min(N, 500), which is never the trainer
    default (100), so every run carries a `_bs<B>` suffix.
    """
    nfeat_suffix = '' if width == TRAINER_DEFAULT_W else f'_nfeat{width}'
    bs = batch_for(n_train)
    bs_suffix = f'_bs{bs}'
    return os.path.join(_LOGS_EMA,
                        f"DDIM_tau_ema_{n_train}{nfeat_suffix}{bs_suffix}_incremental")


def parse_configs(items):
    """Turn ['200:64', '1000:128'] into [(200, 64), (1000, 128)]."""
    out = []
    for it in items:
        if ':' not in it:
            raise ValueError(f"--configs entry '{it}' must be 'N:W' (e.g. 1000:64)")
        n_str, w_str = it.split(':', 1)
        out.append((int(n_str), int(w_str)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Model-size sweep (N in {50,200,500,1000,2000,4000}, "
                    "W in {64,128,256}, B=min(N,500)): FCD(Gen,Test) + f_mem vs τ")
    ap.add_argument("--configs", type=str, nargs='+', default=None,
                    help="(N, W) configs as 'N:W' strings (default: all 18 = "
                         "{50,200,500,1000,2000,4000} × {64,128,256}). "
                         "Batch is min(N, 500).")
    ap.add_argument("--out", type=str, default="wsize_fcd_fmem.csv",
                    help="Output CSV file name inside results/ "
                         "(default: %(default)s).")
    ap.add_argument("--append", action="store_true",
                    help="Append to the output CSV instead of overwriting it. "
                         "Use this to resume an interrupted sweep or to add "
                         "extra --configs without losing earlier rows. The "
                         "header is written only when the file does not exist.")
    ap.add_argument("--debug", action="store_true",
                    help="Few samples + a handful of τ points (quick smoke test)")
    ap.add_argument("--generation-batch-size", type=int, default=_fcd.BATCH_GEN,
                    help="DDIM sampling batch size (default: %(default)s). Lower "
                         "this if GPU memory is tight; keep it fixed across runs "
                         "for exactly reproducible seeded samples.")
    args = ap.parse_args()
    if args.generation_batch_size < 1:
        ap.error("--generation-batch-size must be at least 1")

    configs = parse_configs(args.configs) if args.configs else default_configs()
    num_gen = N_GEN_DEBUG if args.debug else N_GEN
    # generate_samples() resolves BATCH_GEN from its defining (_fcd) module.
    _fcd.BATCH_GEN = args.generation_batch_size
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / args.out

    print("=" * 82)
    print("MODEL-SIZE SWEEP  (N∈{50,200,500,1000,2000,4000}, W∈{64,128,256}, "
          "B=min(N,500))")
    print("                  —  FCD(Gen,Test) + f_mem vs τ")
    print("=" * 82)
    print(f"  Configs (N,W) : {configs}")
    print(f"  Batch rule    : B = min(N, {BATCH_CAP})")
    print(f"  Weights       : {WEIGHTS}  (key='{WEIGHT_KEY}')")
    print(f"  N_GEN         : {num_gen}   Gen seed: {GEN_SEED}")
    print(f"  Generation B  : {args.generation_batch_size}")
    print(f"  Test folds    : {N_FOLDS}  (→ 2σ error bars on FCD)")
    print(f"  f_mem k       : {K_MAIN:.4f}  (nearest-neighbour ratio test)")
    print(f"  f_mem CI      : {BOOTSTRAP_B} bootstraps, seed {BOOTSTRAP_SEED} "
          f"(95% CI)")
    print(f"  Output CSV    : {csv_path}"
          f"{'  (APPEND)' if args.append else '  (overwrite)'}")
    print("=" * 82)

    cols = [
        'N', 'W', 'batch_size', 'n_feat', 'tau', 'epoch_float', 'weights',
        'num_generated', 'num_train', 'num_test', 'feature_dim', 'n_folds',
        'FCD_Gen_Test', 'FCD_Gen_Test_std',
        'FCD_Train_Test', 'FCD_Train_Test_std',
        'f_mem', 'f_mem_ci_low', 'f_mem_ci_high', 'k',
    ]
    # In --append mode an existing CSV is kept as-is (header written only when
    # the file is new), so an interrupted sweep can be resumed with --configs
    # without losing the rows already computed.
    if args.append and csv_path.exists():
        print(f"  Appending to existing CSV ({csv_path.stat().st_size} bytes).")
    else:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(cols)

    for n_train, width in configs:
        bs = batch_for(n_train)
        log_dir = log_dir_for(n_train, width)
        print(f"\n{'='*82}\n  N = {n_train}   W = {width}   "
              f"(batch size = {bs} = min(N,{BATCH_CAP}))\n"
              f"  {log_dir}\n{'='*82}")
        if not os.path.isdir(log_dir):
            print("  WARNING: log dir missing — did you train this (N, W) yet? "
                  "Skipping.")
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

        n_hits = 0
        for tau, path in tqdm(ckpts, desc=f"  N={n_train} W={width} τ"):
            # Use a fresh model per checkpoint. This prevents metric-side GPU
            # work from carrying any model/module state into the next τ.
            ddim = load_checkpoint_model(path, width)
            if ddim is None:
                print(f"    τ={tau}: '{WEIGHT_KEY}' missing, skip.")
                continue

            # Reuse cached EMA samples if present, else generate once & cache.
            samples, hit = get_or_generate_samples(
                ddim, log_dir, tau, num_gen, WEIGHTS, seed=GEN_SEED)
            n_hits += int(hit)

            # f_mem uses the GPU too. Release the sampler first so the U-Net and
            # nearest-neighbour distance matrix never compete for GPU memory.
            del ddim
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            # ── FCD(Gen,Test) on the 5 test folds ─────────────────────────────
            feat_gen = features_from_generated(samples)            # (num_gen,256)
            mu_g, sig_g = fit_gaussian(feat_gen)
            fcd_gt_mean, fcd_gt_std = fcd_vs_reference_folds(mu_g, sig_g, ref_stats)

            # ── f_mem on the SAME generated samples (nearest-neighbour ratio) ─
            gen_t = torch.from_numpy(feat_gen.astype(np.float32))
            nn_result = nearest_ratio_l2(
                gen_t, train_t, batch_size=NN_BATCH_SIZE, device=DEVICE)
            is_mem = (nn_result["ratios"] < K_MAIN)
            # f_mem point estimate + 95% bootstrap CI over the per-sample flags
            # (same procedure as ../compute_fmem_vs_tau.py). Samples are cached,
            # so this adds only the cheap resampling pass — no regeneration.
            f_mem, fmem_ci_low, fmem_ci_high = compute_fmem_bootstrap(
                is_mem, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
            f_mem = float(f_mem)

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    n_train, width, bs, width, tau, tau / steps_per_epoch, WEIGHTS,
                    num_gen, len(feat_train), len(feat_test),
                    feat_train.shape[1], N_FOLDS,
                    fcd_gt_mean, fcd_gt_std, fcd_tt_mean, fcd_tt_std,
                    f_mem, float(fmem_ci_low), float(fmem_ci_high), K_MAIN,
                ])

        print(f"  Cache hits: {n_hits}/{len(ckpts)}  "
              f"(EMA samples in {os.path.join(log_dir, 'generated_ema')})")

    print(f"\n{'='*82}")
    print("Done.")
    print(f"  CSV : {csv_path}")
    print(f"  Next: python plot_wsize_fcd_fmem.py")
    print(f"{'='*82}")


if __name__ == "__main__":
    main()
