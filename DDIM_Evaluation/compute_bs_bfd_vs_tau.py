"""
compute_bs_bfd_vs_tau.py — BS-Beamspace Fréchet Distance vs τ
==============================================================
Task 14: FID-style quality metric using Gaussian mean+covariance comparison
on 32-dimensional normalized BS beam-power profiles.

For each (n_feat, N, tau) checkpoint:
  1. Generate 5000 samples
  2. Extract 32-dimensional normalized BS beam-power profiles
  3. Compute BS-BFD against 5 disjoint reference subsets of 5000 real channels
  4. Report mean ± 2σ error bars, plus mean/covariance decomposition

Also computes:
  - Covariance-regularization sensitivity check (once)
  - BS-BFD(Gen, Train) memorization diagnostic
  - BS-BFD(RealA, RealB) real-real baseline

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python compute_bs_bfd_vs_tau.py

    # Specific model widths / dataset sizes:
    python compute_bs_bfd_vs_tau.py --n_feats 16 64 --sizes 100 1000

    # Quick smoke-test:
    python compute_bs_bfd_vs_tau.py --debug
"""

import sys
import os
import csv
import argparse
from math import ceil
from pathlib import Path
import numpy as np
import torch
import importlib.util
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Paths & imports
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DDIM_DIR   = os.path.join(_SCRIPT_DIR, '..', 'Code', 'DDIM_FMM')

sys.path.insert(0, _DDIM_DIR)

# Import Unet + DDIM from train_DDIM_tau.py
_train_path = os.path.join(_DDIM_DIR, 'train_DDIM_tau.py')
_spec = importlib.util.spec_from_file_location("_train_mod_bfd", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train_mod)
Unet = _train_mod.Unet
DDIM_Model = _train_mod.DDIM


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_FEAT_GRID  = [16, 64, 256]
TRAIN_SIZES  = [100, 200, 500, 1000, 2000, 4000]
LOGS_BASE    = os.path.join(_DDIM_DIR, 'logs')

# Dataset files
DATA_PATH_LOS  = os.path.join(_SCRIPT_DIR, '..', 'dataset',
                               'Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz')
DATA_PATH_NLOS = os.path.join(_SCRIPT_DIR, '..', 'dataset',
                               'Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz')
SUBCARRIER_IDX = 128

# Tau grid
GEN_EVAL_TAU_GRID       = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
GEN_EVAL_TAU_GRID_DEBUG = [1000, 50000, 200000]

# Reference pool
N_TRAIN_MAX   = 4000
N_REF_EACH    = 5000
N_REF_SETS    = 5

# Generation settings
N_GEN          = 5000
BATCH_SIZE_GEN = 100
GEN_SEED       = 0

# Covariance regularization
COV_EPS        = 1e-8
COV_EPS_VALUES = [1e-10, 1e-8, 1e-6]  # sensitivity check

# UPA antenna dimensions
NRX_X, NRX_Y = 2, 2
NTX_X, NTX_Y = 8, 4

BETAS  = (1e-4, 0.02)
N_T    = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("results/bs_bfd_vs_tau")


# ─────────────────────────────────────────────────────────────────────────────
# Beamspace utilities
# ─────────────────────────────────────────────────────────────────────────────

def dft_matrix(N: int) -> np.ndarray:
    n = np.arange(N)
    k = n.reshape(-1, 1)
    return np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)


def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    return np.kron(dft_matrix(Ny), dft_matrix(Nx))


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction: 32-dim normalized BS beam-power profile
# ─────────────────────────────────────────────────────────────────────────────

def channels_to_bs_power_features(channels_ri: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert normalized beamspace real/imag channels to 32-dim BS beam-power features.

    Input:  (N, 2, 4, 32)
    Output: (N, 32), nonneg rows summing to ~1
    """
    if channels_ri.ndim != 4 or channels_ri.shape[1:] != (2, 4, 32):
        raise ValueError(f"Expected shape (N, 2, 4, 32), got {channels_ri.shape}")

    real = channels_ri[:, 0, :, :]       # (N, 4, 32)
    imag = channels_ri[:, 1, :, :]       # (N, 4, 32)

    power = real**2 + imag**2            # (N, 4, 32)
    bs_power = power.sum(axis=1)         # (N, 32)

    total = bs_power.sum(axis=1, keepdims=True)
    features = bs_power / (total + eps)
    return features.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# BS-Beamspace Fréchet Distance
# ─────────────────────────────────────────────────────────────────────────────

def symmetric_psd_sqrt(matrix: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """Symmetric PSD matrix square root via eigendecomposition."""
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, a_min=eps, a_max=None)
    return (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T


def bs_beamspace_frechet_distance(x: np.ndarray, y: np.ndarray,
                                   cov_eps: float = COV_EPS) -> dict:
    """
    Compute BS-Beamspace Fréchet Distance.

    x: (Nx, 32)
    y: (Ny, 32)

    Returns dict with distance, mean_term, covariance_term.
    """
    d = x.shape[1]
    eye = np.eye(d, dtype=np.float64)

    mu_x = np.mean(x, axis=0)
    mu_y = np.mean(y, axis=0)

    cov_x = np.cov(x, rowvar=False, ddof=1) + cov_eps * eye
    cov_y = np.cov(y, rowvar=False, ddof=1) + cov_eps * eye

    mean_term = float(np.sum((mu_x - mu_y) ** 2))

    sqrt_cov_x = symmetric_psd_sqrt(cov_x)
    middle = sqrt_cov_x @ cov_y @ sqrt_cov_x
    sqrt_middle = symmetric_psd_sqrt(middle)

    covariance_term = float(np.trace(cov_x + cov_y - 2.0 * sqrt_middle))

    total = mean_term + covariance_term
    if total < 0 and abs(total) < 1e-10:
        total = 0.0

    return {
        "distance": float(total),
        "mean_term": mean_term,
        "covariance_term": covariance_term,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared index & reference pool
# ─────────────────────────────────────────────────────────────────────────────

def _find_shared_idx_dir() -> str:
    candidates = [
        os.path.join(LOGS_BASE, 'shared_indices'),
        os.path.join(LOGS_BASE, 'DDIM_tau_1000_incremental'),
        os.path.join(LOGS_BASE, 'DDIM_tau_4000_incremental'),
        os.path.join(LOGS_BASE, 'DDIM_tau_100_incremental'),
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, 'indices_los.npy')):
            return d
    raise FileNotFoundError("Cannot find indices_los.npy in any expected directory.")


def build_reference_pool_beamspace() -> np.ndarray:
    """
    Load combined dataset, apply shared shuffled indices, extract reference pool,
    convert to normalized beamspace (N, 2, 4, 32).
    """
    print("\nBuilding reference pool (beamspace) ...")
    shared_idx_dir = _find_shared_idx_dir()

    raw_los  = np.load(DATA_PATH_LOS)['combined_array']
    raw_nlos = np.load(DATA_PATH_NLOS)['combined_array']

    data_los  = raw_los [:, :, 0, :, 0, SUBCARRIER_IDX, :]
    data_nlos = raw_nlos[:, :, 0, :, 0, SUBCARRIER_IDX, :]

    idx_los  = np.load(os.path.join(shared_idx_dir, 'indices_los.npy'))
    idx_nlos = np.load(os.path.join(shared_idx_dir, 'indices_nlos.npy'))
    idx_comb = np.load(os.path.join(shared_idx_dir, 'indices.npy'))

    data_los  = data_los[idx_los]
    data_nlos = data_nlos[idx_nlos]
    data_all  = np.concatenate([data_los, data_nlos], axis=0)
    data_all  = data_all[idx_comb]

    total = len(data_all)
    pool_end = N_TRAIN_MAX + N_REF_SETS * N_REF_EACH
    if pool_end > total:
        raise RuntimeError(
            f"Not enough data: need {pool_end} but only {total} available.")

    print(f"  Total samples: {total}")
    print(f"  Reference pool: [{N_TRAIN_MAX} : {pool_end}] = {pool_end - N_TRAIN_MAX}")

    pool_raw = data_all[N_TRAIN_MAX:pool_end]
    pool_H   = pool_raw[:, :, :, 0].astype(np.complex64)   # (pool_size, 4, 32)

    # Beamspace transform + per-sample normalization
    Ar = upa_dft_codebook(NRX_X, NRX_Y)
    At = upa_dft_codebook(NTX_X, NTX_Y)
    Hv = np.matmul(np.matmul(Ar.conj().T, pool_H), At)

    mag = np.abs(Hv)
    max_mag = mag.reshape(len(Hv), -1).max(axis=1)[:, None, None]
    max_mag = np.where(max_mag > 1e-12, max_mag, 1.0)
    Hv = Hv / max_mag

    pool_ri = np.stack([np.real(Hv), np.imag(Hv)], axis=1).astype(np.float32)
    print(f"  pool_ri shape: {pool_ri.shape}")
    return pool_ri


def precompute_ref_features(pool_ri: np.ndarray) -> list:
    """Convert each of 5 reference subsets to 32-dim features."""
    print("\nPrecomputing BS beam-power features for 5 reference subsets ...")
    feat_refs = []
    for i in range(N_REF_SETS):
        sub = pool_ri[i * N_REF_EACH : (i + 1) * N_REF_EACH]
        feat = channels_to_bs_power_features(sub)
        feat_refs.append(feat)
        print(f"  Subset {i+1}: shape={feat.shape}, "
              f"mean_max_beam={feat.max(axis=1).mean():.4f}")
    return feat_refs


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_log_dir(n_train: int, n_feat: int) -> str:
    nfeat_suffix = f'_nfeat{n_feat}' if n_feat != 256 else ''
    return os.path.join(LOGS_BASE, f"DDIM_tau_{n_train}{nfeat_suffix}_incremental")


def create_model(n_feat: int):
    nn_model = Unet(in_channels=2, n_feat=n_feat)
    ddim = DDIM_Model(nn_model=nn_model, betas=BETAS, n_T=N_T, device=DEVICE)
    ddim.to(DEVICE)
    return ddim


def generate_samples(ddim, n_samples: int, seed: int = GEN_SEED) -> np.ndarray:
    """Generate n_samples, return (n_samples, 2, 4, 32) numpy."""
    torch.manual_seed(seed)
    ddim.eval()
    all_samples = []
    with torch.no_grad():
        while sum(s.shape[0] for s in all_samples) < n_samples:
            bs = min(BATCH_SIZE_GEN, n_samples - sum(s.shape[0] for s in all_samples))
            s = ddim.sample(bs, (2, 4, 32), DEVICE)
            all_samples.append(s.cpu().numpy())
    return np.concatenate(all_samples, axis=0)[:n_samples]


# ─────────────────────────────────────────────────────────────────────────────
# Covariance sensitivity check
# ─────────────────────────────────────────────────────────────────────────────

def run_cov_eps_sensitivity(feat_gen: np.ndarray, feat_ref: np.ndarray,
                            output_dir: Path):
    """Evaluate BS-BFD at different cov_eps values."""
    print("\n  Covariance-eps sensitivity check ...")
    csv_path = output_dir / 'bs_bfd_cov_eps_sensitivity.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['cov_eps', 'BS_BFD', 'mean_term', 'covariance_term'])

    for eps_val in COV_EPS_VALUES:
        res = bs_beamspace_frechet_distance(feat_gen, feat_ref, cov_eps=eps_val)
        print(f"    eps={eps_val:.1e} : BS-BFD={res['distance']:.6f} "
              f"(mean={res['mean_term']:.6f}, cov={res['covariance_term']:.6f})")
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([eps_val, res['distance'],
                                    res['mean_term'], res['covariance_term']])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute BS-BFD vs tau for all model sizes")
    parser.add_argument("--n_feats", type=int, nargs='+', default=N_FEAT_GRID)
    parser.add_argument("--sizes", type=int, nargs='+', default=TRAIN_SIZES)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_sensitivity", action="store_true")
    args = parser.parse_args()

    tau_grid = GEN_EVAL_TAU_GRID_DEBUG if args.debug else GEN_EVAL_TAU_GRID
    n_feats  = args.n_feats
    sizes    = args.sizes
    out_dir  = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BS-Beamspace Fréchet Distance vs τ")
    print("=" * 70)
    print(f"  Device      : {DEVICE}")
    print(f"  n_feats     : {n_feats}")
    print(f"  sizes       : {sizes}")
    print(f"  tau grid    : {tau_grid}")
    print(f"  N_GEN       : {N_GEN}")
    print(f"  N_REF_EACH  : {N_REF_EACH} (× {N_REF_SETS} subsets)")
    print(f"  COV_EPS     : {COV_EPS}")
    print(f"  output      : {out_dir}")
    print("=" * 70)

    # ── Build reference pool ─────────────────────────────────────────────────
    pool_ri = build_reference_pool_beamspace()
    feat_refs = precompute_ref_features(pool_ri)

    # Save reference info
    np.savez(out_dir / 'reference_subset_indices.npz',
             n_train_max=N_TRAIN_MAX,
             n_ref_each=N_REF_EACH,
             n_ref_sets=N_REF_SETS,
             pool_start=N_TRAIN_MAX,
             pool_end=N_TRAIN_MAX + N_REF_SETS * N_REF_EACH)

    # ── Real-real baseline ───────────────────────────────────────────────────
    print("\nComputing real-real baseline: BS-BFD(Ref_1, Ref_2) ...")
    res_rr = bs_beamspace_frechet_distance(feat_refs[0], feat_refs[1])
    bfd_real_real = res_rr['distance']
    print(f"  BS-BFD(Ref_1, Ref_2) = {bfd_real_real:.6f}")
    print(f"    mean_term = {res_rr['mean_term']:.6f}")
    print(f"    cov_term  = {res_rr['covariance_term']:.6f}")

    # ── Sanity: BS-BFD(X, X) ≈ 0 ────────────────────────────────────────────
    res_self = bs_beamspace_frechet_distance(feat_refs[0], feat_refs[0])
    print(f"  BS-BFD(X, X) = {res_self['distance']:.8f}  (should be ≈ 0)")

    # ── Prepare main CSV ─────────────────────────────────────────────────────
    all_csv_path = out_dir / 'bs_bfd_vs_tau.csv'
    all_cols = [
        'N', 'n_feat', 'target_tau', 'actual_tau', 'epoch_float',
        'steps_per_epoch',
        'num_generated_total', 'num_generated_per_reference',
        'num_test_total', 'num_test_per_reference',
        'feature_dimension', 'cov_eps',
        'BS_BFD_Gen_Test_mean', 'BS_BFD_Gen_Test_error_2std', 'BS_BFD_Gen_Test_std',
        'BS_BFD_ref1', 'BS_BFD_ref2', 'BS_BFD_ref3', 'BS_BFD_ref4', 'BS_BFD_ref5',
        'BS_BFD_mean_term_mean', 'BS_BFD_covariance_term_mean',
        'BS_BFD_Gen_Train', 'BS_BFD_Train_Test',
        'BS_BFD_real_real_baseline',
    ]
    with open(all_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(all_cols)

    # Reference subset values CSV
    ref_csv_path = out_dir / 'bs_bfd_reference_subset_values.csv'
    with open(ref_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'N', 'n_feat', 'target_tau',
            'BS_BFD_ref1', 'BS_BFD_ref2', 'BS_BFD_ref3',
            'BS_BFD_ref4', 'BS_BFD_ref5',
            'mean_term_ref1', 'mean_term_ref2', 'mean_term_ref3',
            'mean_term_ref4', 'mean_term_ref5',
            'cov_term_ref1', 'cov_term_ref2', 'cov_term_ref3',
            'cov_term_ref4', 'cov_term_ref5',
        ])

    sensitivity_done = False

    # ── Main loop ────────────────────────────────────────────────────────────
    for n_feat in n_feats:
        print(f"\n{'='*70}")
        print(f"  n_feat = {n_feat}")
        print(f"{'='*70}")
        ddim = create_model(n_feat)

        for n_train in sizes:
            log_dir = get_log_dir(n_train, n_feat)
            if not os.path.exists(log_dir):
                print(f"  N={n_train}: log dir missing, skip")
                continue

            steps_per_epoch = ceil(n_train / 100)
            print(f"\n  N = {n_train}  (steps/epoch = {steps_per_epoch})")

            # Load train data for BS-BFD(Gen, Train) diagnostic
            train_path = os.path.join(log_dir, 'train.npy')
            feat_train = None
            if os.path.exists(train_path):
                train_arr = np.load(train_path)
                H_train = (train_arr[:, 0] + 1j * train_arr[:, 1]).astype(np.complex64)
                Ar = upa_dft_codebook(NRX_X, NRX_Y)
                At = upa_dft_codebook(NTX_X, NTX_Y)
                Hv_train = np.matmul(np.matmul(Ar.conj().T, H_train), At)
                mag_t = np.abs(Hv_train)
                max_t = mag_t.reshape(len(Hv_train), -1).max(axis=1)[:, None, None]
                max_t = np.where(max_t > 1e-12, max_t, 1.0)
                Hv_train = Hv_train / max_t
                train_ri = np.stack([np.real(Hv_train), np.imag(Hv_train)],
                                    axis=1).astype(np.float32)
                feat_train = channels_to_bs_power_features(train_ri)

            # BS-BFD(Train, Test) baseline
            bfd_train_test = None
            if feat_train is not None and len(feat_train) >= 50:
                res_tt = bs_beamspace_frechet_distance(feat_train, feat_refs[0])
                bfd_train_test = res_tt['distance']
                print(f"    BS-BFD(Train, Test_ref1) = {bfd_train_test:.6f}")

            for target_tau in tqdm(tau_grid, desc=f"    N={n_train} taus"):
                ckpt_path = os.path.join(log_dir, 'checkpoints',
                                         f'checkpoint_tau_{target_tau}.pth')
                if not os.path.exists(ckpt_path):
                    continue

                ckpt = torch.load(ckpt_path, map_location=DEVICE,
                                  weights_only=False)
                ddim.load_state_dict(ckpt['model_state_dict'])
                actual_tau = int(ckpt.get('global_step', target_tau))
                epoch_float = actual_tau / steps_per_epoch

                # Generate samples
                samples = generate_samples(ddim, N_GEN, seed=GEN_SEED)
                feat_gen = channels_to_bs_power_features(samples)

                # Sensitivity check (run once on first valid checkpoint)
                if not sensitivity_done and not args.skip_sensitivity:
                    run_cov_eps_sensitivity(feat_gen, feat_refs[0], out_dir)
                    sensitivity_done = True

                # BS-BFD vs each of 5 reference subsets
                bfd_vals = []
                mean_terms = []
                cov_terms = []
                for feat_ref in feat_refs:
                    res = bs_beamspace_frechet_distance(feat_gen, feat_ref)
                    bfd_vals.append(res['distance'])
                    mean_terms.append(res['mean_term'])
                    cov_terms.append(res['covariance_term'])

                bfd_mean = float(np.mean(bfd_vals))
                bfd_std  = float(np.std(bfd_vals, ddof=1))
                bfd_2std = 2 * bfd_std
                mean_term_mean = float(np.mean(mean_terms))
                cov_term_mean  = float(np.mean(cov_terms))

                # BS-BFD(Gen, Train) diagnostic
                bfd_gen_train = None
                if feat_train is not None and len(feat_train) >= 50:
                    res_gt = bs_beamspace_frechet_distance(feat_gen, feat_train)
                    bfd_gen_train = res_gt['distance']

                # Write to main CSV
                row = [
                    n_train, n_feat, target_tau, actual_tau, epoch_float,
                    steps_per_epoch,
                    N_GEN, N_GEN,
                    N_REF_EACH * N_REF_SETS, N_REF_EACH,
                    32, COV_EPS,
                    bfd_mean, bfd_2std, bfd_std,
                    *bfd_vals,
                    mean_term_mean, cov_term_mean,
                    bfd_gen_train if bfd_gen_train is not None else '',
                    bfd_train_test if bfd_train_test is not None else '',
                    bfd_real_real,
                ]
                with open(all_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow(row)

                # Reference subset values CSV
                with open(ref_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow(
                        [n_train, n_feat, target_tau]
                        + bfd_vals + mean_terms + cov_terms)

    # Save generated subset indices
    np.savez(out_dir / 'generated_subset_indices.npz',
             n_gen=N_GEN, seed=GEN_SEED, note='All 5000 generated used per subset')

    print(f"\n{'='*70}")
    print("Done!")
    print(f"  Main CSV       : {all_csv_path}")
    print(f"  Ref values     : {ref_csv_path}")
    print(f"  Sensitivity    : {out_dir / 'bs_bfd_cov_eps_sensitivity.csv'}")
    print(f"\nTo plot, run:")
    print(f"  python plot_bs_bfd_vs_tau.py")
    print(f"  python plot_quality_and_fmem_bs_bfd_vs_tau.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
