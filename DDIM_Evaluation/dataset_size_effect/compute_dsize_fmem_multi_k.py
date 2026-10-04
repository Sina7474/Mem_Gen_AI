"""
compute_dsize_fmem_multi_k.py
    — Robustness of the memorization scaling to the ratio-test threshold k.
================================================================================
PURPOSE
-------
The memorization fraction f_mem is defined via the nearest-neighbour RATIO test:
for every generated sample we form

        rho = d1 / d2      (d1 = nearest-train dist, d2 = 2nd-nearest-train dist)

and flag the sample as *memorized* iff  rho < k.  The main paper (and our
canonical experiment in compute_dsize_fcd_fmem.py) works with k = 1/3.

This script re-uses the SAME cached EMA-generated samples and the SAME 256-D
beamspace representation, computes the rho values ONCE per checkpoint, and then
thresholds them at

        k ∈ {1/4, 1/3, 1/2}

so that we can check the claim (paper): "we choose to work with k = 1/3, but we
checked that varying k to 1/2 or 1/4 does not impact the claims about the
scaling."  Because rho is independent of k, the three curves come for FREE from a
single nearest-neighbour pass — no regeneration, no extra GPU work beyond the
cheap thresholding + bootstrap.

Only N ∈ {200, 1000} are evaluated (to keep the comparison figure uncluttered),
using the same batch-size rule B = min(N, 500) and width W = 256 as the parent
experiment.

OUTPUT
------
    results/dsize_fmem_multi_k.csv   (one row per N × τ × k)

USAGE
-----
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect
    python compute_dsize_fmem_multi_k.py                 # N = 200 1000, k = 1/4 1/3 1/2
    python compute_dsize_fmem_multi_k.py --sizes 200     # a single size
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
_HERE      = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR  = os.path.join(_HERE, '..')                 # DDIM_Evaluation/
_DDIM_DIR  = os.path.join(_EVAL_DIR, '..', 'Code', 'DDIM_FMM')
_LOGS_EMA  = os.path.join(_DDIM_DIR, 'logs_ema')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fcd  = _load_module('_fcd_mod_mk',  os.path.join(_EVAL_DIR, 'compute_fcd_vs_tau.py'))
_fmem = _load_module('_fmem_mod_mk', os.path.join(_EVAL_DIR, 'compute_fmem_vs_tau.py'))

features_from_spatial_npy = _fcd.features_from_spatial_npy
features_from_generated   = _fcd.features_from_generated
create_model              = _fcd.create_model
list_checkpoints          = _fcd.list_checkpoints
get_or_generate_samples   = _fcd.get_or_generate_samples
N_FEAT   = _fcd.N_FEAT
N_GEN    = _fcd.N_GEN
GEN_SEED = _fcd.GEN_SEED
DEVICE   = _fcd.DEVICE

nearest_ratio_l2       = _fmem.nearest_ratio_l2
NN_BATCH_SIZE          = _fmem.NN_BATCH_SIZE
compute_fmem_bootstrap = _fmem.compute_fmem_bootstrap
BOOTSTRAP_B            = _fmem.BOOTSTRAP_B
BOOTSTRAP_SEED         = _fmem.BOOTSTRAP_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SIZES   = [200, 1000]            # keep the robustness figure uncluttered
K_VALUES        = [1/4, 1/3, 1/2]        # thresholds to compare
BATCH_CAP       = 500                    # B = min(N, BATCH_CAP)
TRAINER_DEFAULT_BS = 100
WEIGHTS         = 'ema'
WEIGHT_KEY      = 'ema_model_state_dict'
OUTPUT_DIR      = Path("results")


def batch_for(n_train):
    return min(n_train, BATCH_CAP)


def log_dir_for(n_train):
    bs = batch_for(n_train)
    bs_suffix = '' if bs == TRAINER_DEFAULT_BS else f'_bs{bs}'
    return os.path.join(_LOGS_EMA, f"DDIM_tau_ema_{n_train}{bs_suffix}_incremental")


def main():
    ap = argparse.ArgumentParser(
        description="f_mem vs τ at k ∈ {1/4,1/3,1/2} for N ∈ {200,1000} (3.5 GHz)")
    ap.add_argument("--sizes", type=int, nargs='+', default=DEFAULT_SIZES)
    ap.add_argument("--k_values", type=float, nargs='+', default=K_VALUES,
                    help="Ratio-test thresholds to compute (default: 1/4 1/3 1/2).")
    ap.add_argument("--min_tau", type=int, default=0,
                    help="Skip checkpoints below this optimizer step (default: 0).")
    ap.add_argument("--output_csv", type=Path, default=None,
                    help="Write to this CSV instead of replacing the default "
                         "results/dsize_fmem_multi_k.csv.")
    args = ap.parse_args()

    k_values = sorted(set(float(k) for k in args.k_values))
    if not k_values or any(not 0.0 < k < 1.0 for k in k_values):
        ap.error("Every --k_values entry must lie strictly between 0 and 1.")
    if args.min_tau < 0:
        ap.error("--min_tau must be non-negative.")

    num_gen = N_GEN
    csv_path = args.output_csv or (OUTPUT_DIR / "dsize_fmem_multi_k.csv")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("MULTI-k ROBUSTNESS  —  f_mem vs τ  at  k ∈ {1/4, 1/3, 1/2}")
    print("=" * 78)
    print(f"  Sizes N       : {args.sizes}")
    print(f"  k values      : {[f'{k:.4f}' for k in k_values]}")
    print(f"  Minimum tau   : {args.min_tau}")
    print(f"  Model width   : {N_FEAT}")
    print(f"  N_GEN         : {num_gen}   Gen seed: {GEN_SEED}")
    print(f"  Output CSV    : {csv_path}")
    print("=" * 78)

    cols = [
        'N', 'batch_size', 'n_feat', 'tau', 'epoch_float', 'weights',
        'num_generated', 'num_train', 'feature_dim',
        'k', 'f_mem', 'f_mem_ci_low', 'f_mem_ci_high',
    ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(cols)

    for n_train in args.sizes:
        bs = batch_for(n_train)
        log_dir = log_dir_for(n_train)
        print(f"\n{'='*78}\n  N = {n_train}   (batch size = {bs})\n  {log_dir}\n{'='*78}")
        if not os.path.isdir(log_dir):
            print("  WARNING: log dir missing — skipping.")
            continue

        ckpts = list_checkpoints(log_dir)
        ckpts = [(tau, path) for tau, path in ckpts if tau >= args.min_tau]
        if not ckpts:
            print("  WARNING: no checkpoints found, skipping.")
            continue

        steps_per_epoch = ceil(n_train / bs)

        train_arr = np.load(os.path.join(log_dir, 'train.npy'))
        feat_train = features_from_spatial_npy(train_arr)          # (N,256)
        train_t = torch.from_numpy(feat_train.astype(np.float32))

        ddim = None
        n_hits = 0
        for tau, path in tqdm(ckpts, desc=f"  N={n_train} τ"):
            cache_path = os.path.join(
                log_dir, 'generated_ema',
                f'gen_{WEIGHTS}_tau{tau}_seed{GEN_SEED}.npz')
            samples = None
            if os.path.exists(cache_path):
                try:
                    cached = np.load(cache_path)['channels']
                    if cached.shape[0] >= num_gen:
                        samples = cached[:num_gen]
                except Exception:
                    samples = None

            if samples is not None:
                hit = True
            else:
                if ddim is None:
                    ddim = create_model()
                ck = torch.load(path, map_location=DEVICE, weights_only=False)
                if WEIGHT_KEY not in ck:
                    continue
                ddim.load_state_dict(ck[WEIGHT_KEY])
                samples, hit = get_or_generate_samples(
                    ddim, log_dir, tau, num_gen, WEIGHTS, seed=GEN_SEED)
            n_hits += int(hit)

            feat_gen = features_from_generated(samples)            # (num_gen,256)
            gen_t = torch.from_numpy(feat_gen.astype(np.float32))

            # ── nearest-neighbour ratios computed ONCE, shared by all k ───────
            nn = nearest_ratio_l2(gen_t, train_t,
                                  batch_size=NN_BATCH_SIZE, device=DEVICE)
            ratios = nn["ratios"]

            for k in k_values:
                is_mem = (ratios < k)
                f_mem, ci_lo, ci_hi = compute_fmem_bootstrap(
                    is_mem, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
                with open(csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow([
                        n_train, bs, N_FEAT, tau, tau / steps_per_epoch, WEIGHTS,
                        num_gen, len(feat_train), feat_train.shape[1],
                        float(k), float(f_mem), float(ci_lo), float(ci_hi),
                    ])

        print(f"  Cache hits: {n_hits}/{len(ckpts)}")

    print(f"\n{'='*78}\nDone.\n  CSV : {csv_path}\n"
          f"  Next: python plot_dsize_fmem_multi_k.py\n{'='*78}")


if __name__ == "__main__":
    main()
