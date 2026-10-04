"""
compute_dsize_fmem_28GHz_LoS.py
    — 28 GHz LoS scene: f_mem vs τ for the dataset-size sweep at fixed width
      W = 256 with the batch rule  B = min(N, 500).
================================================================================
PURPOSE
-------
This is the 28 GHz LoS counterpart of

    ../dataset_size_effect/compute_dsize_fcd_fmem.py     (3.5 GHz LoS+NLoS)

and is written so that the two are directly comparable: identical model width,
identical batch protocol, identical τ grid, identical 256-D feature space and
identical memorization test. The ONLY differences are the propagation scene
(28 GHz, LoS-only, ~7.2k users) and therefore the log tree that is read:

    3.5 GHz : Code/DDIM_FMM/logs_ema/DDIM_tau_ema_{N}_bs{B}_incremental
    28 GHz  : Code/DDIM_FMM/logs_ema_28GHz_LoS/DDIM_tau_ema_28GHz_LoS_{N}_bs{B}_incremental

Dataset sizes swept (N = 100 is intentionally excluded):

        N ∈ {200, 500, 1000, 2000, 4000}   with   B = min(N, 500)

METRIC (verbatim from the validated 3.5 GHz code — nothing is re-implemented)
----------------------------------------------------------------------------
For every generated sample we compute the L2 distance to its nearest (d1) and
second-nearest (d2) TRAINING sample in the per-sample-normalized UPA-DFT
beamspace (flattened to R^256), and declare the sample memorized when

        rho = d1 / d2  <  k = 1/3.

f_mem(τ) is the fraction of memorized samples, reported with a 95 % bootstrap
confidence interval (B = 1000 resamples of the per-sample binary flags).

SAMPLE GENERATION / CACHING
---------------------------
5000 samples are drawn from the EMA weights of every τ checkpoint with a fixed
seed and cached in

        <run_dir>/generated_ema/gen_ema_tau{τ}_seed0.npz

exactly as in the 3.5 GHz pipeline. The cache is checked first, so re-runs (and
resumption after an interruption) cost almost nothing — only the cheap
nearest-neighbour pass is repeated. This cache is separate from the
`generated_downstream/` pools used by the beam-alignment task.

OUTPUT
------
    results/dsize_fmem_28GHz_LoS.csv     (one row per N × τ checkpoint)

USAGE
-----
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect_28GHz_LoS
    python compute_dsize_fmem_28GHz_LoS.py                  # N = 200 500 1000 2000 4000
    python compute_dsize_fmem_28GHz_LoS.py --sizes 1000     # a single size
    python compute_dsize_fmem_28GHz_LoS.py --debug          # quick smoke test
"""

import os
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
_HERE     = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.join(_HERE, '..')                       # DDIM_Evaluation/
_DDIM_DIR = os.path.join(_EVAL_DIR, '..', 'Code', 'DDIM_FMM')
_LOGS_28G = os.path.join(_DDIM_DIR, 'logs_ema_28GHz_LoS')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fcd  = _load_module('_fcd_mod_28g',  os.path.join(_EVAL_DIR, 'compute_fcd_vs_tau.py'))
_fmem = _load_module('_fmem_mod_28g', os.path.join(_EVAL_DIR, 'compute_fmem_vs_tau.py'))

# Feature / model / sampling side (identical protocol to the 3.5 GHz sweep) ----
features_from_spatial_npy = _fcd.features_from_spatial_npy
features_from_generated   = _fcd.features_from_generated
create_model              = _fcd.create_model
list_checkpoints          = _fcd.list_checkpoints
get_or_generate_samples   = _fcd.get_or_generate_samples
N_FEAT      = _fcd.N_FEAT          # 256
N_GEN       = _fcd.N_GEN           # 5000
N_GEN_DEBUG = _fcd.N_GEN_DEBUG     # 500
GEN_SEED    = _fcd.GEN_SEED        # 0
DEVICE      = _fcd.DEVICE
DEBUG_TAUS  = _fcd.DEBUG_TAUS

# f_mem side (identical nearest-neighbour ratio test) -------------------------
nearest_ratio_l2       = _fmem.nearest_ratio_l2
K_MAIN                 = _fmem.K_MAIN          # 1/3
NN_BATCH_SIZE          = _fmem.NN_BATCH_SIZE
compute_fmem_bootstrap = _fmem.compute_fmem_bootstrap
BOOTSTRAP_B            = _fmem.BOOTSTRAP_B     # 1000
BOOTSTRAP_SEED         = _fmem.BOOTSTRAP_SEED  # 42


# ─────────────────────────────────────────────────────────────────────────────
# Configuration  (28 GHz LoS scene)
# ─────────────────────────────────────────────────────────────────────────────
SCENE         = '28GHz_LoS'
DEFAULT_SIZES = [200, 500, 1000, 2000, 4000]   # N = 100 deliberately excluded
BATCH_CAP     = 500                            # B = min(N, BATCH_CAP)
WEIGHTS       = 'ema'
WEIGHT_KEY    = 'ema_model_state_dict'
OUTPUT_DIR    = Path(_HERE) / "results"
OUTPUT_CSV    = OUTPUT_DIR / "dsize_fmem_28GHz_LoS.csv"


def batch_for(n_train):
    """The fixed batch-size protocol for this experiment:  B = min(N, 500)."""
    return min(n_train, BATCH_CAP)


def log_dir_for(n_train):
    """Run directory produced by train_DDIM_tau_ema_28GHz_LoS.py.

    Width is fixed at 256 (trainer default → no `_nfeat` suffix); the batch size
    B = min(N, 500) is always written explicitly as a `_bs<B>` suffix.
    """
    return os.path.join(
        _LOGS_28G, f"DDIM_tau_ema_{SCENE}_{n_train}_bs{batch_for(n_train)}_incremental")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="28 GHz LoS dataset-size sweep (W=256, B=min(N,500)): f_mem vs τ")
    ap.add_argument("--sizes", type=int, nargs='+', default=DEFAULT_SIZES,
                    help=f"Dataset sizes to include (default: {DEFAULT_SIZES}). "
                         f"Batch size for each is min(N, {BATCH_CAP}).")
    ap.add_argument("--min_tau", type=int, default=0,
                    help="Skip τ checkpoints below this value (default: 0, i.e. keep "
                         "the τ=0 random-init reference point).")
    ap.add_argument("--gen_batch", type=int, default=None,
                    help="Sampling batch size for DDIM generation (default: the "
                         "3.5 GHz pipeline value). Larger is faster on a big GPU.")
    ap.add_argument("--debug", action="store_true",
                    help="Few samples + a handful of τ points (quick smoke test)")
    args = ap.parse_args()

    if args.gen_batch is not None:
        _fcd.BATCH_GEN = args.gen_batch

    num_gen = N_GEN_DEBUG if args.debug else N_GEN
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("28 GHz LoS — DATASET-SIZE SWEEP (W=256, B=min(N,500)) — f_mem vs τ")
    print("=" * 78)
    print(f"  Scene         : {SCENE}")
    print(f"  Log tree      : {os.path.normpath(_LOGS_28G)}")
    print(f"  Sizes N       : {args.sizes}")
    print(f"  Batch rule    : B = min(N, {BATCH_CAP})  → "
          f"{ {n: batch_for(n) for n in args.sizes} }")
    print(f"  Model width   : {N_FEAT}")
    print(f"  Weights       : {WEIGHTS}  (key='{WEIGHT_KEY}')")
    print(f"  N_GEN         : {num_gen}   Gen seed: {GEN_SEED}   "
          f"Gen batch: {_fcd.BATCH_GEN}")
    print(f"  f_mem k       : {K_MAIN:.4f}  (nearest-neighbour ratio test)")
    print(f"  Output CSV    : {OUTPUT_CSV}")
    print("=" * 78)

    cols = [
        'scene', 'N', 'batch_size', 'n_feat', 'tau', 'epoch_float', 'weights',
        'num_generated', 'num_train', 'num_test', 'feature_dim',
        'f_mem', 'f_mem_ci_low', 'f_mem_ci_high', 'k',
        'mean_ratio', 'median_ratio', 'mean_d1', 'mean_d2',
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(cols)

    for n_train in args.sizes:
        bs = batch_for(n_train)
        log_dir = log_dir_for(n_train)
        print(f"\n{'='*78}\n  N = {n_train}   (batch size = {bs} = min(N,{BATCH_CAP}))\n"
              f"  {os.path.normpath(log_dir)}\n{'='*78}")
        if not os.path.isdir(log_dir):
            print("  WARNING: run dir missing — is this size trained yet? Skipping.")
            continue

        ckpts = [(t, p) for (t, p) in list_checkpoints(log_dir) if t >= args.min_tau]
        if not ckpts:
            print("  WARNING: no checkpoints found, skipping.")
            continue
        if args.debug:
            ckpts = [(t, p) for (t, p) in ckpts if t in DEBUG_TAUS] or ckpts[:2]

        steps_per_epoch = ceil(n_train / bs)

        # ── Reference: spatial real/imag → UPA-DFT beamspace → per-sample norm ──
        train_arr = np.load(os.path.join(log_dir, 'train.npy'))
        test_arr  = np.load(os.path.join(log_dir, 'test.npy'))
        feat_train = features_from_spatial_npy(train_arr)          # (N,256)
        train_t = torch.from_numpy(feat_train.astype(np.float32))

        print(f"  Feature dim   : {feat_train.shape[1]}")
        print(f"  steps/epoch   : {steps_per_epoch}  (= ceil(N/bs))")
        print(f"  train/test    : {len(train_arr)} / {len(test_arr)}")

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

            # ── f_mem via the nearest-neighbour ratio test ────────────────────
            feat_gen = features_from_generated(samples)            # (num_gen,256)
            gen_t = torch.from_numpy(feat_gen.astype(np.float32))
            nn = nearest_ratio_l2(gen_t, train_t,
                                  batch_size=NN_BATCH_SIZE, device=DEVICE)
            is_mem = (nn["ratios"] < K_MAIN)
            f_mem, ci_low, ci_high = compute_fmem_bootstrap(
                is_mem, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)

            with open(OUTPUT_CSV, 'a', newline='') as f:
                csv.writer(f).writerow([
                    SCENE, n_train, bs, N_FEAT, tau, tau / steps_per_epoch, WEIGHTS,
                    num_gen, len(feat_train), len(test_arr), feat_train.shape[1],
                    float(f_mem), float(ci_low), float(ci_high), K_MAIN,
                    float(np.mean(nn["ratios"])), float(np.median(nn["ratios"])),
                    float(np.mean(nn["d1"])), float(np.mean(nn["d2"])),
                ])

        print(f"  Cache hits: {n_hits}/{len(ckpts)}  "
              f"(EMA samples in {os.path.join(log_dir, 'generated_ema')})")

    print(f"\n{'='*78}")
    print("Done.")
    print(f"  CSV : {OUTPUT_CSV}")
    print(f"  Next: python plot_dsize_fmem_28GHz_LoS.py")
    print(f"        python plot_fmem_28GHz_vs_3p5GHz.py")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
