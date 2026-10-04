"""
compute_N100_28GHz_LoS.py
    — Compute FCD + f_mem for N=100 at 28 GHz LoS and APPEND to
      results/dsize_fcd_fmem_28GHz_LoS.csv (same format as existing rows).
================================================================================
Run:
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect_28GHz_LoS
    python compute_N100_28GHz_LoS.py
"""

import os
import csv
import importlib.util
from math import ceil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR = os.path.join(_HERE, '..')
_DDIM_DIR = os.path.join(_EVAL_DIR, '..', 'Code', 'DDIM_FMM')
_LOGS_28G = os.path.join(_DDIM_DIR, 'logs_ema_28GHz_LoS')


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_fcd  = _load_module('_fcd_mod_n100', os.path.join(_EVAL_DIR, 'compute_fcd_vs_tau.py'))
_fmem = _load_module('_fmem_mod_n100', os.path.join(_EVAL_DIR, 'compute_fmem_vs_tau.py'))

# FCD
features_from_spatial_npy = _fcd.features_from_spatial_npy
features_from_generated   = _fcd.features_from_generated
fit_gaussian              = _fcd.fit_gaussian
build_fold_stats          = _fcd.build_fold_stats
fcd_vs_reference_folds    = _fcd.fcd_vs_reference_folds
create_model              = _fcd.create_model
list_checkpoints          = _fcd.list_checkpoints
get_or_generate_samples   = _fcd.get_or_generate_samples
N_FEAT      = _fcd.N_FEAT
N_GEN       = _fcd.N_GEN
N_FOLDS     = _fcd.N_FOLDS
GEN_SEED    = _fcd.GEN_SEED
DEVICE      = _fcd.DEVICE

# f_mem
nearest_ratio_l2       = _fmem.nearest_ratio_l2
K_MAIN                 = _fmem.K_MAIN
NN_BATCH_SIZE          = _fmem.NN_BATCH_SIZE
compute_fmem_bootstrap = _fmem.compute_fmem_bootstrap
BOOTSTRAP_B            = _fmem.BOOTSTRAP_B
BOOTSTRAP_SEED         = _fmem.BOOTSTRAP_SEED

# ─────────────────────────────────────────────────────────────────────────────
SCENE     = '28GHz_LoS'
N_TRAIN   = 100
BATCH_SIZE = 100  # min(100, 500) = 100
WEIGHTS   = 'ema'
WEIGHT_KEY = 'ema_model_state_dict'
OUTPUT_CSV = Path(_HERE) / 'results' / 'dsize_fcd_fmem_28GHz_LoS.csv'

LOG_DIR = os.path.join(
    _LOGS_28G, f"DDIM_tau_ema_{SCENE}_{N_TRAIN}_bs{BATCH_SIZE}_incremental")


def main():
    print("=" * 78)
    print(f"28 GHz LoS — FCD + f_mem for N = {N_TRAIN}")
    print("=" * 78)
    print(f"  Log dir  : {LOG_DIR}")
    print(f"  Output   : {OUTPUT_CSV}")

    if not os.path.isdir(LOG_DIR):
        raise FileNotFoundError(f"Run dir missing: {LOG_DIR}")

    ckpts = list_checkpoints(LOG_DIR)
    print(f"  Checkpoints: {len(ckpts)}")

    steps_per_epoch = ceil(N_TRAIN / BATCH_SIZE)

    # Reference features
    train_arr = np.load(os.path.join(LOG_DIR, 'train.npy'))
    test_arr  = np.load(os.path.join(LOG_DIR, 'test.npy'))
    feat_train = features_from_spatial_npy(train_arr)
    feat_test  = features_from_spatial_npy(test_arr)
    train_t = torch.from_numpy(feat_train.astype(np.float32))

    # FCD reference: fold stats of test set
    fold_stats = build_fold_stats(feat_test, N_FOLDS)
    # FCD(Train, Test) — fixed reference-reference distance
    mu_train, sig_train = fit_gaussian(feat_train)
    fcd_train_test_mean, fcd_train_test_std = fcd_vs_reference_folds(
        mu_train, sig_train, fold_stats)

    print(f"  FCD(Train,Test): {fcd_train_test_mean:.4f} ± {fcd_train_test_std:.4f}")
    print(f"  Feature dim   : {feat_train.shape[1]}")
    print(f"  train/test    : {len(train_arr)} / {len(test_arr)}")

    ddim = create_model()

    # Check if we need to write a header (if file doesn't exist or is empty)
    cols = [
        'scene', 'N', 'batch_size', 'n_feat', 'tau', 'epoch_float', 'weights',
        'num_generated', 'num_train', 'num_test', 'feature_dim', 'n_folds',
        'FCD_Gen_Test', 'FCD_Gen_Test_std', 'FCD_Train_Test', 'FCD_Train_Test_std',
        'f_mem', 'f_mem_ci_low', 'f_mem_ci_high', 'k',
        'mean_ratio', 'median_ratio', 'mean_d1', 'mean_d2',
    ]

    # We'll append to the existing CSV
    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(cols)

    for tau, path in tqdm(ckpts, desc=f"  N={N_TRAIN} τ"):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        if WEIGHT_KEY not in ck:
            print(f"    τ={tau}: '{WEIGHT_KEY}' missing, skip.")
            continue
        ddim.load_state_dict(ck[WEIGHT_KEY])

        samples, hit = get_or_generate_samples(
            ddim, LOG_DIR, tau, N_GEN, WEIGHTS, seed=GEN_SEED)

        # FCD(Gen, Test)
        feat_gen = features_from_generated(samples)
        mu_g, sig_g = fit_gaussian(feat_gen)
        fcd_gen_test_mean, fcd_gen_test_std = fcd_vs_reference_folds(
            mu_g, sig_g, fold_stats)

        # f_mem
        gen_t = torch.from_numpy(feat_gen.astype(np.float32))
        nn = nearest_ratio_l2(gen_t, train_t,
                              batch_size=NN_BATCH_SIZE, device=DEVICE)
        is_mem = (nn["ratios"] < K_MAIN)
        f_mem, ci_low, ci_high = compute_fmem_bootstrap(
            is_mem, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)

        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([
                SCENE, N_TRAIN, BATCH_SIZE, N_FEAT, tau, tau / steps_per_epoch,
                WEIGHTS, N_GEN, len(feat_train), len(test_arr),
                feat_train.shape[1], N_FOLDS,
                fcd_gen_test_mean, fcd_gen_test_std,
                fcd_train_test_mean, fcd_train_test_std,
                float(f_mem), float(ci_low), float(ci_high), K_MAIN,
                float(np.mean(nn["ratios"])), float(np.median(nn["ratios"])),
                float(np.mean(nn["d1"])), float(np.mean(nn["d2"])),
            ])

    print(f"\nDone. N=100 results appended to:\n  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
