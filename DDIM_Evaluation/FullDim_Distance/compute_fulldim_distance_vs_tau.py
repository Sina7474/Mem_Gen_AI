"""
compute_fulldim_distance_vs_tau.py
    — Full 256-D distribution distance (Sinkhorn-OT + Sliced-Wasserstein) vs τ
================================================================================
GOAL (Task 15).  Both effective-rank (1 scalar / sample → 1-D Wasserstein) and
FCD (mean + covariance → Gaussian assumption) COLLAPSE the distribution and lose
information.  Here we compare the DDIM-generated and the held-out TEST channel
distributions DIRECTLY in the FULL 256-D beamspace, without ever reducing a
sample to a scalar:

    each sample  H ∈ (2,4,32)  →  flatten  →  x ∈ R^256

and we measure the distance between the two 256-D empirical distributions with
two complementary multivariate optimal-transport metrics:

    • Sinkhorn-W2  : entropic-regularized 2-Wasserstein (true multivariate OT,
                     GPU log-domain solver, tractable for n≈5000);
    • SWD-W2       : Sliced-Wasserstein-2 — the exact multivariate generalization
                     of the 1-D Wasserstein used on effective-rank: it averages
                     the 1-D W2 over many random 256-D projections.

────────────────────────────────────────────────────────────────────────────────
DATA REUSE — identical inputs to FCD so the curves are directly comparable
────────────────────────────────────────────────────────────────────────────────
This script IMPORTS compute_fcd_vs_tau.py and reuses ITS functions:
    • features_from_generated   (generated beamspace samples → flatten 256)
    • features_from_spatial_npy (test/train spatial → beamspace → norm → flatten)
    • get_or_generate_samples   (loads the SAME cached generated_ema/*.npz)
    • list_checkpoints / get_ema_log_dir / create_model
So the generated samples and the TEST/TRAIN reference are EXACTLY the ones FCD
used (same cache, same seed, same folds).

Protocol (paper-style, same as FCD):
    reference = held-out TEST channels split into 5 disjoint folds → 2σ error bars
    Sinkhorn/SWD(Gen , Test)  = quality curve            (want LOW, saturating)
    Sinkhorn/SWD(Train, Test) = real–real floor          (irreducible finite-N)

Read-only w.r.t. training.  Results → FullDim_Distance/results/.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/FullDim_Distance

    python compute_fulldim_distance_vs_tau.py                 # ema, N=200 1000 4000
    python compute_fulldim_distance_vs_tau.py --sizes 1000    # one size
    python compute_fulldim_distance_vs_tau.py --debug         # quick smoke test
"""

import os
import sys
import csv
import argparse
from math import ceil
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Reuse the FCD pipeline (feature extraction + generated-sample cache) verbatim
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR   = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))   # DDIM_Evaluation
sys.path.insert(0, _EVAL_DIR)

import importlib.util
_fcd_path = os.path.join(_EVAL_DIR, 'compute_fcd_vs_tau.py')
_fcd_spec = importlib.util.spec_from_file_location("_fcd_mod_fulldim", _fcd_path)
_fcd = importlib.util.module_from_spec(_fcd_spec)
_fcd_spec.loader.exec_module(_fcd)

# Constants shared with FCD (guarantees identical generated data)
N_FEAT      = _fcd.N_FEAT
TRAIN_SIZES = _fcd.TRAIN_SIZES
N_GEN       = _fcd.N_GEN
N_GEN_DEBUG = _fcd.N_GEN_DEBUG
GEN_SEED    = _fcd.GEN_SEED
N_FOLDS     = 5              # 5 disjoint test folds, each with 5000 samples
N_TEST_PER_FOLD = 5000       # MUST match N_GEN to avoid sample-size bias in OT
FOLD_SEED   = _fcd.FOLD_SEED
DEBUG_TAUS  = _fcd.DEBUG_TAUS
DEVICE      = _fcd.DEVICE

OUTPUT_DIR = Path(_SCRIPT_DIR) / "results"

# Dataset paths (to load the full test pool, not just the small test.npy)
_DATASET_DIR = os.path.join(_SCRIPT_DIR, '..', '..', 'dataset')
DATA_PATH_LOS  = os.path.join(_DATASET_DIR, 'Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz')
DATA_PATH_NLOS = os.path.join(_DATASET_DIR, 'Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz')
SUBCARRIER_IDX = 128

# ─────────────────────────────────────────────────────────────────────────────
# Metric hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
SINKHORN_REG_FRAC = 0.05     # entropic reg = frac × median(cost)  (auto-scaled)
SINKHORN_ITERS    = 300
SINKHORN_TOL      = 1e-6
SWD_N_PROJ        = 512       # random 256-D directions
SWD_N_QUANTILES   = 256       # quantile levels for unequal-size 1-D W2
SWD_SEED          = 42


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Sinkhorn (entropic) 2-Wasserstein, stabilized log-domain, on GPU
# ─────────────────────────────────────────────────────────────────────────────
def sinkhorn_w2(X, Y, reg_frac=SINKHORN_REG_FRAC,
                n_iter=SINKHORN_ITERS, tol=SINKHORN_TOL, device=DEVICE):
    """Entropic-regularized W2 between empirical measures on rows of X, Y.

    X: (n, d), Y: (m, d) numpy/torch.  Uniform weights.  Cost = squared Euclid.
    Returns the W2 distance = sqrt(<P, C>)  (transport cost of the plan).
    """
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    n, m = Xt.shape[0], Yt.shape[0]

    C = torch.cdist(Xt, Yt, p=2) ** 2                      # (n, m) squared Euclid
    reg = reg_frac * torch.median(C).clamp_min(1e-12)      # auto-scaled epsilon

    log_a = torch.full((n,), -np.log(n), device=device)
    log_b = torch.full((m,), -np.log(m), device=device)
    f = torch.zeros(n, device=device)
    g = torch.zeros(m, device=device)

    for _ in range(n_iter):
        f_prev = f
        # f_i = reg*(log_a_i - logsumexp_j[(-C_ij + g_j)/reg])
        f = reg * (log_a - torch.logsumexp((-C + g[None, :]) / reg, dim=1))
        # g_j = reg*(log_b_j - logsumexp_i[(-C_ij + f_i)/reg])
        g = reg * (log_b - torch.logsumexp((-C + f[:, None]) / reg, dim=0))
        if torch.max(torch.abs(f - f_prev)) < tol:
            break

    logP = (-C + f[:, None] + g[None, :]) / reg
    P = torch.exp(logP)
    cost = torch.sum(P * C)                                # ≈ W2^2
    return float(torch.sqrt(cost.clamp_min(0.0)).item())


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — Sliced-Wasserstein-2 in full 256-D (many random 1-D projections)
# ─────────────────────────────────────────────────────────────────────────────
def sliced_wasserstein_w2(X, Y, n_proj=SWD_N_PROJ, n_quantiles=SWD_N_QUANTILES,
                          seed=SWD_SEED, device=DEVICE):
    """Multivariate generalization of the 1-D Wasserstein: average 1-D W2 over
    `n_proj` random unit directions in R^d.  Uses quantile matching so the two
    sample sizes need NOT be equal.

    Returns SWD-W2 = sqrt(mean_over_projections mean_over_quantiles (q_x-q_y)^2).
    """
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    d = Xt.shape[1]

    gen = torch.Generator(device=device).manual_seed(seed)
    dirs = torch.randn(d, n_proj, generator=gen, device=device)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)           # (d, L) unit columns

    xp = Xt @ dirs                                         # (n, L)
    yp = Yt @ dirs                                         # (m, L)

    q = torch.linspace(0.0, 1.0, n_quantiles, device=device)
    xq = torch.quantile(xp, q, dim=0)                      # (Q, L)
    yq = torch.quantile(yp, q, dim=0)                      # (Q, L)

    w2sq_per_proj = ((xq - yq) ** 2).mean(dim=0)           # (L,)
    return float(torch.sqrt(w2sq_per_proj.mean().clamp_min(0.0)).item())


# ─────────────────────────────────────────────────────────────────────────────
# Large test pool — load 5×5000 = 25000 non-training samples from the dataset
# ─────────────────────────────────────────────────────────────────────────────
def load_test_folds(log_dir, n_train, n_folds=N_FOLDS,
                    n_per_fold=N_TEST_PER_FOLD, seed=FOLD_SEED):
    """Load n_folds × n_per_fold non-training samples from the full dataset.

    Uses the SAME shuffled indices as training (so we never overlap with train),
    taking samples from position n_train onward in the shuffled combined pool.
    Converts to beamspace, normalizes, and flattens to 256-D features.

    Returns list of 5 arrays, each (n_per_fold, 256).
    """
    indices_path = os.path.join(log_dir, 'indices.npy')
    indices_los_path = os.path.join(log_dir, 'indices_los.npy')
    indices_nlos_path = os.path.join(log_dir, 'indices_nlos.npy')

    # Load the shuffled indices used during training
    indices = np.load(indices_path)
    indices_los = np.load(indices_los_path)
    indices_nlos = np.load(indices_nlos_path)

    # Load raw dataset (only the subcarrier slice we need)
    los_raw = np.load(DATA_PATH_LOS)['combined_array'][:, :, 0, :, 0, SUBCARRIER_IDX, :]
    nlos_raw = np.load(DATA_PATH_NLOS)['combined_array'][:, :, 0, :, 0, SUBCARRIER_IDX, :]

    # Apply the same per-dataset shuffle as training did
    los_raw = los_raw[indices_los]
    nlos_raw = nlos_raw[indices_nlos]

    # Concatenate and apply the combined shuffle
    combined = np.concatenate((los_raw, nlos_raw), axis=0)  # (40851, 4, 32, 4)
    combined = combined[indices]

    # Take the non-training samples (from position n_train onward)
    n_needed = n_folds * n_per_fold
    test_pool = combined[n_train : n_train + n_needed]
    if len(test_pool) < n_needed:
        raise ValueError(
            f"Not enough non-training samples: need {n_needed}, got {len(test_pool)}. "
            f"Total={len(combined)}, n_train={n_train}")

    # Convert each sample: extract H, split real/imag, beamspace, normalize, flatten
    # H_set shape from raw: (N, 4, 32, 4) → H = data[:,:,:,0] → (N, 4, 32) complex
    H_set = test_pool[:, :, :, 0]  # (N, 4, 32) complex
    ri = np.stack([np.real(H_set), np.imag(H_set)], axis=1).astype(np.float32)  # (N, 2, 4, 32)

    # Apply beamspace transform + normalization (same as FCD pipeline)
    feat_all = _fcd.features_from_spatial_npy(ri)  # (N, 256) float64

    # Split into n_folds disjoint folds of n_per_fold each
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_needed)
    folds = []
    for i in range(n_folds):
        fold_idx = perm[i * n_per_fold : (i + 1) * n_per_fold]
        folds.append(feat_all[fold_idx])
    return folds


def dist_vs_folds(metric_fn, feat_query, test_folds):
    """metric_fn(query, ref_fold) over the test folds → (mean, std)."""
    vals = np.array([metric_fn(feat_query, fold) for fold in test_folds],
                    dtype=np.float64)
    return float(vals.mean()), float(vals.std())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Full 256-D distribution distance (Sinkhorn-OT + SWD) vs τ")
    ap.add_argument("--sizes", type=int, nargs='+', default=TRAIN_SIZES)
    ap.add_argument("--weights", choices=['ema', 'raw'], default='ema')
    ap.add_argument("--debug", action="store_true", help="Few samples + coarse grid")
    ap.add_argument("--sinkhorn_reg_frac", type=float, default=SINKHORN_REG_FRAC)
    ap.add_argument("--sinkhorn_iters", type=int, default=SINKHORN_ITERS)
    ap.add_argument("--n_proj", type=int, default=SWD_N_PROJ)
    ap.add_argument("--n_quantiles", type=int, default=SWD_N_QUANTILES)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    weight_key = 'ema_model_state_dict' if args.weights == 'ema' else 'model_state_dict'
    num_gen = N_GEN_DEBUG if args.debug else N_GEN
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    def _sink(q, r):
        return sinkhorn_w2(q, r, reg_frac=args.sinkhorn_reg_frac,
                           n_iter=args.sinkhorn_iters)

    def _swd(q, r):
        return sliced_wasserstein_w2(q, r, n_proj=args.n_proj,
                                     n_quantiles=args.n_quantiles)

    print("=" * 76)
    print("Full 256-D distribution distance (Sinkhorn-OT + Sliced-Wasserstein) vs τ")
    print("=" * 76)
    print(f"  Device        : {DEVICE}")
    print(f"  Sizes         : {args.sizes}")
    print(f"  Weights       : {args.weights}  (key='{weight_key}')")
    print(f"  N_GEN         : {num_gen}")
    print(f"  Test folds    : {N_FOLDS} × {N_TEST_PER_FOLD} samples each  (→ 2σ error bars)")
    print(f"  Sinkhorn      : reg_frac={args.sinkhorn_reg_frac}  iters={args.sinkhorn_iters}")
    print(f"  SWD           : n_proj={args.n_proj}  n_quantiles={args.n_quantiles}")
    print(f"  Output        : {out_dir}")
    print("  NOTE: Gen and each test fold both have 5000 samples → no sample-size OT bias")
    print("=" * 76)

    csv_path = out_dir / f'fulldim_distance_vs_tau_{args.weights}.csv'
    cols = [
        'N', 'n_feat', 'tau', 'epoch_float', 'weights', 'num_generated',
        'num_train', 'num_test', 'feature_dim', 'n_folds',
        'Sinkhorn_Gen_Test', 'Sinkhorn_Gen_Test_std',
        'Sinkhorn_Train_Test', 'Sinkhorn_Train_Test_std',
        'SWD_Gen_Test', 'SWD_Gen_Test_std',
        'SWD_Train_Test', 'SWD_Train_Test_std',
    ]
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(cols)

    for n_train in args.sizes:
        log_dir = _fcd.get_ema_log_dir(n_train)
        print(f"\n{'='*76}\n  N = {n_train}\n  {log_dir}\n{'='*76}")
        if not os.path.isdir(log_dir):
            print("  WARNING: EMA log dir missing, skip.")
            continue

        ckpts = _fcd.list_checkpoints(log_dir)
        if not ckpts:
            print("  WARNING: no checkpoints, skip.")
            continue
        if args.debug:
            ckpts = [(t, p) for (t, p) in ckpts if t in DEBUG_TAUS]

        steps_per_epoch = ceil(n_train / 100)

        # ── Load 5 test folds of 5000 each from the FULL dataset pool ────────
        print(f"  Loading {N_FOLDS}×{N_TEST_PER_FOLD} test samples from dataset...")
        test_folds = load_test_folds(log_dir, n_train)
        print(f"  Each test fold: {test_folds[0].shape}")

        # ── Training features for Train-Test floor ────────────────────────────
        train_arr = np.load(os.path.join(log_dir, 'train.npy'))
        feat_train = _fcd.features_from_spatial_npy(train_arr)      # (N_train, 256)

        # Real–real floor: Train(N) vs each Test fold(5000).
        # NOTE: Train has N samples (200/1000/4000).  For a FAIR comparison,
        # Gen-Test will also use exactly N generated samples (subsampled),
        # so both sides have the same sample count → no OT sample-size bias.
        sink_tt_m, sink_tt_s = dist_vs_folds(_sink, feat_train, test_folds)
        swd_tt_m,  swd_tt_s  = dist_vs_folds(_swd,  feat_train, test_folds)
        print(f"  Feature dim   : {feat_train.shape[1]}")
        print(f"  Train samples : {feat_train.shape[0]}")
        print(f"  Gen subsample : {n_train}  (matched to train size for fair OT)")
        print(f"  Floor Sinkhorn(Train,Test) = {sink_tt_m:.6f} ± {sink_tt_s:.6f}")
        print(f"  Floor SWD     (Train,Test) = {swd_tt_m:.6f} ± {swd_tt_s:.6f}")

        ddim = _fcd.create_model()

        n_hits = 0
        for tau, path in tqdm(ckpts, desc=f"  N={n_train} τ"):
            ck = torch.load(path, map_location=DEVICE, weights_only=False)
            if weight_key not in ck:
                print(f"    τ={tau}: '{weight_key}' missing, skip.")
                continue
            ddim.load_state_dict(ck[weight_key])

            # Reuse the SAME cached generated samples FCD produced (or make once).
            samples, hit = _fcd.get_or_generate_samples(
                ddim, log_dir, tau, num_gen, args.weights, seed=GEN_SEED)
            n_hits += int(hit)

            feat_gen = _fcd.features_from_generated(samples)        # (5000, 256)

            # Subsample Gen to exactly N_train for fair comparison with Train-Test floor.
            # Both Train(N) and Gen(N) have the same sample count against Test(5000).
            n_sub = min(n_train, len(feat_gen))
            rng_sub = np.random.default_rng(GEN_SEED + tau)  # tau-dependent but reproducible
            sub_idx = rng_sub.choice(len(feat_gen), size=n_sub, replace=False)
            feat_gen_sub = feat_gen[sub_idx]                         # (N_train, 256)

            sink_gt_m, sink_gt_s = dist_vs_folds(_sink, feat_gen_sub, test_folds)
            swd_gt_m,  swd_gt_s  = dist_vs_folds(_swd,  feat_gen_sub, test_folds)

            row = [
                n_train, N_FEAT, tau, tau / steps_per_epoch, args.weights, num_gen,
                len(feat_train), N_TEST_PER_FOLD, feat_train.shape[1], N_FOLDS,
                sink_gt_m, sink_gt_s, sink_tt_m, sink_tt_s,
                swd_gt_m,  swd_gt_s,  swd_tt_m,  swd_tt_s,
            ]
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)
        print(f"  Cache hits: {n_hits}/{len(ckpts)}  "
              f"(generated samples from {os.path.join(log_dir, 'generated_ema')})")

    print(f"\n{'='*76}")
    print("Done.")
    print(f"  CSV: {csv_path}")
    print(f"  Next: python plot_fulldim_distance_vs_tau.py --weights {args.weights} \\")
    print(f"          --fmem_csv ../results/fmem_vs_tau/fmem_vs_tau.csv")
    print(f"{'='*76}")


if __name__ == "__main__":
    main()
