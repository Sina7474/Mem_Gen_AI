"""
compute_fmem_kappa_N500.py — f_mem(tau) at kappa in {1/4, 1/3} for N = 500
==========================================================================
The existing robustness CSV
    DDIM_Evaluation/dataset_size_effect/results/dsize_fmem_multi_k.csv
only contains N in {200, 1000}.  The shared-reference CSI experiment now also
uses N = 500, so to draw the generalization window at the alternative ratio-test
threshold kappa = 1/4 we need f_mem(tau) at kappa = 1/4 for N = 500 as well.

This script reproduces the *exact* memorization pipeline of
    DDIM_Evaluation/dataset_size_effect/compute_dsize_fmem_multi_k.py
(same 256-D beamspace features, same nearest-neighbour ratio test, same cached
EMA samples, same bootstrap) but restricted to N = 500 and writes to a NEW file

    results/shared_reference/fmem_kappa_N500.csv

so no existing DDIM_Evaluation result is modified.  Because the ratios rho are
independent of kappa, both thresholds come from a single nearest-neighbour pass
over the already-cached generated samples — no GPU regeneration is needed.

Usage:
    conda activate Mem_Gen
    cd Downstream_Tasks/CSI_Compression
    python shared_reference/compute_fmem_kappa_N500.py
"""

from __future__ import annotations

import csv
import importlib.util
import os
from math import ceil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent.parent
EVAL_DIR = PROJECT / "DDIM_Evaluation"
DSIZE_DIR = EVAL_DIR / "dataset_size_effect"
LOGS_EMA = PROJECT / "Code" / "DDIM_FMM" / "logs_ema"

OUT_CSV = HERE / "results" / "fmem_kappa_N500.csv"

# N = 500 EMA generator (batch-size-500 run; note the non-uniform directory name)
LOG_DIR = LOGS_EMA / "DDIM_tau_ema_500_bs500_incremental"

N_TRAIN = 500
BATCH = 500
K_VALUES = [1.0 / 4.0, 1.0 / 3.0]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    fcd = _load_module("_fcd_srk", EVAL_DIR / "compute_fcd_vs_tau.py")
    fmem = _load_module("_fmem_srk", EVAL_DIR / "compute_fmem_vs_tau.py")

    features_from_spatial_npy = fcd.features_from_spatial_npy
    features_from_generated = fcd.features_from_generated
    create_model = fcd.create_model
    list_checkpoints = fcd.list_checkpoints
    get_or_generate_samples = fcd.get_or_generate_samples
    N_FEAT = fcd.N_FEAT
    N_GEN = fcd.N_GEN
    GEN_SEED = fcd.GEN_SEED
    DEVICE = fcd.DEVICE
    WEIGHT_KEY = "ema_model_state_dict"
    WEIGHTS = "ema"

    nearest_ratio_l2 = fmem.nearest_ratio_l2
    NN_BATCH_SIZE = fmem.NN_BATCH_SIZE
    compute_fmem_bootstrap = fmem.compute_fmem_bootstrap
    BOOTSTRAP_B = fmem.BOOTSTRAP_B
    BOOTSTRAP_SEED = fmem.BOOTSTRAP_SEED

    if not LOG_DIR.is_dir():
        raise FileNotFoundError(f"N=500 EMA log dir not found: {LOG_DIR}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("f_mem vs tau  at  kappa in {1/4, 1/3}   for  N = 500")
    print("=" * 78)
    print(f"  Log dir   : {LOG_DIR}")
    print(f"  Output    : {OUT_CSV}")
    print(f"  N_GEN     : {N_GEN}   Gen seed: {GEN_SEED}   W: {N_FEAT}")
    print("=" * 78)

    cols = [
        "N", "batch_size", "n_feat", "tau", "epoch_float", "weights",
        "num_generated", "num_train", "feature_dim",
        "k", "f_mem", "f_mem_ci_low", "f_mem_ci_high",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(cols)

    ckpts = list_checkpoints(str(LOG_DIR))
    if not ckpts:
        raise RuntimeError(f"No checkpoints found in {LOG_DIR}")

    steps_per_epoch = ceil(N_TRAIN / BATCH)
    train_arr = np.load(LOG_DIR / "train.npy")
    feat_train = features_from_spatial_npy(train_arr)
    train_t = torch.from_numpy(feat_train.astype(np.float32))

    ddim = create_model()
    n_hits = 0
    for tau, path in tqdm(ckpts, desc="  N=500 tau"):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        if WEIGHT_KEY not in ck:
            continue
        ddim.load_state_dict(ck[WEIGHT_KEY])

        samples, hit = get_or_generate_samples(
            ddim, str(LOG_DIR), tau, N_GEN, WEIGHTS, seed=GEN_SEED)
        n_hits += int(hit)

        feat_gen = features_from_generated(samples)
        gen_t = torch.from_numpy(feat_gen.astype(np.float32))

        nn = nearest_ratio_l2(gen_t, train_t,
                              batch_size=NN_BATCH_SIZE, device=DEVICE)
        ratios = nn["ratios"]

        for k in K_VALUES:
            is_mem = (ratios < k)
            f_mem, ci_lo, ci_hi = compute_fmem_bootstrap(
                is_mem, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
            with open(OUT_CSV, "a", newline="") as f:
                csv.writer(f).writerow([
                    N_TRAIN, BATCH, N_FEAT, tau, tau / steps_per_epoch, WEIGHTS,
                    N_GEN, len(feat_train), feat_train.shape[1],
                    float(k), float(f_mem), float(ci_lo), float(ci_hi),
                ])

    print(f"\n  Cache hits: {n_hits}/{len(ckpts)}   (0 regenerations expected)")
    print(f"  Saved: {OUT_CSV}")


if __name__ == "__main__":
    main()
