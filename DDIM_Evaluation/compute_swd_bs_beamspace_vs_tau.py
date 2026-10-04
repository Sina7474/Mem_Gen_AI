"""
compute_swd_bs_beamspace_vs_tau.py — Sliced Wasserstein-1 on BS Beamspace vs τ
================================================================================
Task 13: Multidimensional BS-Beamspace quality metric.

For each (n_feat, N, tau) checkpoint:
  1. Generate 5000 samples
  2. Extract 32-dimensional normalized BS beam-power profiles
  3. Compute SW1 against 5 disjoint reference subsets of 5000 real channels each
  4. Report mean ± 2σ error bars

Also computes:
  - Projection convergence check (saved once)
  - SW1(Gen, Train) memorization diagnostic
  - SW1(RealA, RealB) real-real baseline

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python compute_swd_bs_beamspace_vs_tau.py

    # Specific model widths / dataset sizes:
    python compute_swd_bs_beamspace_vs_tau.py --n_feats 64 256 --sizes 100 1000

    # Quick smoke-test:
    python compute_swd_bs_beamspace_vs_tau.py --debug
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
from scipy.stats import wasserstein_distance

# ─────────────────────────────────────────────────────────────────────────────
# Paths & imports
# ─────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DDIM_DIR   = os.path.join(_SCRIPT_DIR, '..', 'Code', 'DDIM_FMM')

sys.path.insert(0, _DDIM_DIR)

# Import Unet + DDIM from train_DDIM_tau.py (guarantees architecture match)
_train_path = os.path.join(_DDIM_DIR, 'train_DDIM_tau.py')
_spec = importlib.util.spec_from_file_location("_train_mod", _train_path)
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

# Reference pool: positions [4000 : 29000] of the combined shuffled order
N_TRAIN_MAX   = 4000    # largest training set — defines reference pool start
N_REF_EACH    = 5000    # samples per reference subset
N_REF_SETS    = 5       # number of disjoint reference subsets

# Generation settings
N_GEN          = 5000
BATCH_SIZE_GEN = 100
GEN_SEED       = 0

# Projection settings
NUM_PROJECTIONS   = 1024
PROJECTION_SEED   = 42
FEATURE_DIMENSION = 32

# Projection convergence check
PROJECTION_COUNTS = [128, 256, 512, 1024, 2048]

# UPA antenna dimensions
NRX_X, NRX_Y = 2, 2
NTX_X, NTX_Y = 8, 4

BETAS  = (1e-4, 0.02)
N_T    = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = Path("results/swd_bs_beamspace_vs_tau")


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
    Convert normalized beamspace real/imag channels into 32-dimensional
    normalized BS beam-power features.

    Parameters
    ----------
    channels_ri : ndarray, shape (N, 2, 4, 32)
        Dimension 1: real and imaginary parts.

    Returns
    -------
    features : ndarray, shape (N, 32)
        Each row is nonnegative and sums approximately to 1.
    """
    if channels_ri.ndim != 4 or channels_ri.shape[1:] != (2, 4, 32):
        raise ValueError(f"Expected shape (N, 2, 4, 32), got {channels_ri.shape}")

    real = channels_ri[:, 0, :, :]       # (N, 4, 32)
    imag = channels_ri[:, 1, :, :]       # (N, 4, 32)

    power = real**2 + imag**2            # (N, 4, 32)
    bs_power = power.sum(axis=1)         # (N, 32)  — sum over UE/Rx dimension

    total = bs_power.sum(axis=1, keepdims=True)   # (N, 1)
    features = bs_power / (total + eps)

    return features.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Projection directions
# ─────────────────────────────────────────────────────────────────────────────

def make_projection_directions(dimension: int = FEATURE_DIMENSION,
                                num_projections: int = NUM_PROJECTIONS,
                                seed: int = PROJECTION_SEED) -> np.ndarray:
    """
    Generate fixed random unit-norm projection directions.

    Returns
    -------
    directions : (dimension, num_projections)
    """
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(dimension, num_projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    return directions.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Sliced Wasserstein-1 Distance
# ─────────────────────────────────────────────────────────────────────────────

def sliced_wasserstein_w1(x: np.ndarray, y: np.ndarray,
                          directions: np.ndarray) -> tuple:
    """
    Compute Sliced Wasserstein-1 Distance.

    Parameters
    ----------
    x : (Nx, 32)
    y : (Ny, 32)
    directions : (32, L)

    Returns
    -------
    sw1 : float — mean sliced W1
    per_projection : ndarray of shape (L,) — W1 per projection
    """
    x_proj = x @ directions   # (Nx, L)
    y_proj = y @ directions   # (Ny, L)

    L = directions.shape[1]
    per_projection = np.empty(L, dtype=np.float64)

    for i in range(L):
        per_projection[i] = wasserstein_distance(x_proj[:, i], y_proj[:, i])

    return float(per_projection.mean()), per_projection


# ─────────────────────────────────────────────────────────────────────────────
# Shared index & reference pool construction
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
    raise FileNotFoundError(
        "Cannot find indices_los.npy in any expected directory.")


def build_reference_pool_beamspace() -> np.ndarray:
    """
    Load the combined dataset, apply shared shuffled indices, extract
    positions [N_TRAIN_MAX : N_TRAIN_MAX + N_REF_SETS*N_REF_EACH],
    convert to normalized beamspace real/imag (N, 2, 4, 32).

    Returns
    -------
    pool_ri : (N_REF_SETS * N_REF_EACH, 2, 4, 32) float32
        Channels in DDIM-consistent normalized beamspace representation.
    """
    print("\nBuilding reference pool (beamspace) ...")
    shared_idx_dir = _find_shared_idx_dir()

    # Load raw arrays
    raw_los  = np.load(DATA_PATH_LOS)['combined_array']
    raw_nlos = np.load(DATA_PATH_NLOS)['combined_array']

    data_los  = raw_los [:, :, 0, :, 0, SUBCARRIER_IDX, :]
    data_nlos = raw_nlos[:, :, 0, :, 0, SUBCARRIER_IDX, :]

    # Apply shared shuffled indices (same as training)
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
            f"Not enough data: need {pool_end} samples but only {total} available.")

    print(f"  Total samples: {total}")
    print(f"  Reference pool: [{N_TRAIN_MAX} : {pool_end}] = {pool_end - N_TRAIN_MAX} samples")

    pool_raw = data_all[N_TRAIN_MAX:pool_end]   # (..., 4, 32, ?)
    pool_H   = pool_raw[:, :, :, 0].astype(np.complex64)   # (pool_size, 4, 32)

    # Convert to beamspace and normalize (same pipeline as training)
    Ar = upa_dft_codebook(NRX_X, NRX_Y)  # (4, 4)
    At = upa_dft_codebook(NTX_X, NTX_Y)  # (32, 32)
    Hv = np.matmul(np.matmul(Ar.conj().T, pool_H), At)  # (pool_size, 4, 32)

    # Per-sample normalization (matches training: divide by max |.|)
    mag = np.abs(Hv)
    max_mag = mag.reshape(len(Hv), -1).max(axis=1)[:, None, None]
    max_mag = np.where(max_mag > 1e-12, max_mag, 1.0)
    Hv = Hv / max_mag

    # Stack real/imag → (pool_size, 2, 4, 32)
    pool_ri = np.stack([np.real(Hv), np.imag(Hv)], axis=1).astype(np.float32)

    print(f"  pool_ri shape: {pool_ri.shape}")
    return pool_ri


def precompute_ref_features(pool_ri: np.ndarray) -> list:
    """
    Convert each of 5 reference subsets to 32-dim BS beam-power features.

    Returns
    -------
    feat_refs : list of 5 arrays, each of shape (N_REF_EACH, 32)
    """
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
# Projection convergence check
# ─────────────────────────────────────────────────────────────────────────────

def run_projection_convergence(feat_gen: np.ndarray, feat_ref: np.ndarray,
                                output_dir: Path):
    """
    Evaluate SW1 at different projection counts (nested) to verify convergence.
    Uses the first reference subset as the test target.
    """
    print("\n  Projection convergence check ...")

    # Generate max projections (nested — first 128 are reused)
    max_proj = max(PROJECTION_COUNTS)
    directions_full = make_projection_directions(
        FEATURE_DIMENSION, max_proj, PROJECTION_SEED)

    conv_csv = output_dir / 'swd_projection_convergence.csv'
    with open(conv_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_projections', 'SW1'])

    results = []
    for n_proj in PROJECTION_COUNTS:
        dirs_sub = directions_full[:, :n_proj]
        sw1, _ = sliced_wasserstein_w1(feat_gen, feat_ref, dirs_sub)
        results.append((n_proj, sw1))
        print(f"    L={n_proj:>5d} : SW1 = {sw1:.6f}")

        with open(conv_csv, 'a', newline='') as f:
            csv.writer(f).writerow([n_proj, sw1])

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute SW1 on BS beamspace features vs tau")
    parser.add_argument("--n_feats", type=int, nargs='+', default=N_FEAT_GRID)
    parser.add_argument("--sizes", type=int, nargs='+', default=TRAIN_SIZES)
    parser.add_argument("--debug", action="store_true",
                        help="Quick smoke-test: fewer taus")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_convergence", action="store_true",
                        help="Skip projection convergence check")
    args = parser.parse_args()

    tau_grid = GEN_EVAL_TAU_GRID_DEBUG if args.debug else GEN_EVAL_TAU_GRID
    n_feats  = args.n_feats
    sizes    = args.sizes
    out_dir  = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Sliced Wasserstein-1 on BS Beamspace Features vs τ")
    print("=" * 70)
    print(f"  Device         : {DEVICE}")
    print(f"  n_feats        : {n_feats}")
    print(f"  sizes          : {sizes}")
    print(f"  tau grid       : {tau_grid}")
    print(f"  N_GEN          : {N_GEN}")
    print(f"  N_REF_EACH     : {N_REF_EACH} (× {N_REF_SETS} subsets)")
    print(f"  NUM_PROJECTIONS: {NUM_PROJECTIONS}")
    print(f"  output         : {out_dir}")
    print("=" * 70)

    # ── Generate and save projection directions ──────────────────────────────
    directions = make_projection_directions()
    dir_path = out_dir / 'projection_directions.npy'
    np.save(dir_path, directions)
    print(f"\n  Saved projection directions: {dir_path}")

    # ── Build reference pool ─────────────────────────────────────────────────
    pool_ri = build_reference_pool_beamspace()
    feat_refs = precompute_ref_features(pool_ri)

    # Save reference subset indices info
    np.savez(out_dir / 'reference_subset_indices.npz',
             n_train_max=N_TRAIN_MAX,
             n_ref_each=N_REF_EACH,
             n_ref_sets=N_REF_SETS,
             pool_start=N_TRAIN_MAX,
             pool_end=N_TRAIN_MAX + N_REF_SETS * N_REF_EACH)

    # ── Real-real baseline: SW1(Ref_1, Ref_2) ────────────────────────────────
    print("\nComputing real-real baseline: SW1(Ref_1, Ref_2) ...")
    sw1_real_real, _ = sliced_wasserstein_w1(feat_refs[0], feat_refs[1], directions)
    print(f"  SW1(Ref_1, Ref_2) = {sw1_real_real:.6f}")

    # ── Sanity check: SW1(X, X) ≈ 0 ─────────────────────────────────────────
    sw1_self, _ = sliced_wasserstein_w1(feat_refs[0], feat_refs[0], directions)
    print(f"  SW1(X, X) = {sw1_self:.8f}  (should be ≈ 0)")

    # ── Prepare main CSV ─────────────────────────────────────────────────────
    all_csv_path = out_dir / 'swd_bs_beamspace_vs_tau.csv'
    all_cols = [
        'N', 'n_feat', 'target_tau', 'actual_tau', 'epoch_float',
        'steps_per_epoch', 'num_generated', 'num_test_total',
        'num_reference_subsets', 'reference_subset_size',
        'feature_dimension', 'num_projections', 'projection_seed',
        'SW1_Gen_Test_mean', 'SW1_Gen_Test_error_2std', 'SW1_Gen_Test_std',
        'SW1_Gen_Test_ref1', 'SW1_Gen_Test_ref2', 'SW1_Gen_Test_ref3',
        'SW1_Gen_Test_ref4', 'SW1_Gen_Test_ref5',
        'SW1_Gen_Train', 'SW1_Train_Test',
        'SW1_real_real_baseline',
    ]
    with open(all_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(all_cols)

    # Per-subset values CSV
    ref_csv_path = out_dir / 'swd_reference_subset_values.csv'
    with open(ref_csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'N', 'n_feat', 'target_tau',
            'SW1_ref1', 'SW1_ref2', 'SW1_ref3', 'SW1_ref4', 'SW1_ref5'])

    convergence_done = False

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

            # Load train data features for SW1(Gen, Train) diagnostic
            train_path = os.path.join(log_dir, 'train.npy')
            if os.path.exists(train_path):
                train_arr = np.load(train_path)  # (N, 2, 4, 32) spatial domain
                # Convert spatial → beamspace → normalized → features
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
            else:
                feat_train = None

            # SW1(Train, Test) baseline for this N
            sw1_train_test = None
            if feat_train is not None:
                sw1_train_test, _ = sliced_wasserstein_w1(
                    feat_train, feat_refs[0], directions)
                print(f"    SW1(Train, Test_ref1) baseline = {sw1_train_test:.6f}")

            for target_tau in tqdm(tau_grid, desc=f"    N={n_train} taus"):
                ckpt_path = os.path.join(log_dir, 'checkpoints',
                                         f'checkpoint_tau_{target_tau}.pth')
                if not os.path.exists(ckpt_path):
                    continue

                # Load checkpoint
                ckpt = torch.load(ckpt_path, map_location=DEVICE,
                                  weights_only=False)
                ddim.load_state_dict(ckpt['model_state_dict'])
                actual_tau = int(ckpt.get('global_step', target_tau))
                epoch_float = actual_tau / steps_per_epoch

                # Generate samples
                samples = generate_samples(ddim, N_GEN, seed=GEN_SEED)
                feat_gen = channels_to_bs_power_features(samples)

                # Projection convergence (run once on first valid checkpoint)
                if not convergence_done and not args.skip_convergence:
                    run_projection_convergence(feat_gen, feat_refs[0], out_dir)
                    convergence_done = True

                # SW1 vs each of 5 reference subsets
                sw1_vals = []
                for feat_ref in feat_refs:
                    sw1_i, _ = sliced_wasserstein_w1(feat_gen, feat_ref, directions)
                    sw1_vals.append(sw1_i)

                sw1_mean = float(np.mean(sw1_vals))
                sw1_std  = float(np.std(sw1_vals, ddof=1))
                sw1_2std = 2 * sw1_std

                # SW1(Gen, Train) diagnostic
                sw1_gen_train = None
                if feat_train is not None:
                    sw1_gen_train, _ = sliced_wasserstein_w1(
                        feat_gen, feat_train, directions)

                # Write to CSV
                row = [
                    n_train, n_feat, target_tau, actual_tau, epoch_float,
                    steps_per_epoch, N_GEN, N_REF_EACH * N_REF_SETS,
                    N_REF_SETS, N_REF_EACH,
                    FEATURE_DIMENSION, NUM_PROJECTIONS, PROJECTION_SEED,
                    sw1_mean, sw1_2std, sw1_std,
                    *sw1_vals,
                    sw1_gen_train if sw1_gen_train is not None else '',
                    sw1_train_test if sw1_train_test is not None else '',
                    sw1_real_real,
                ]
                with open(all_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow(row)

                with open(ref_csv_path, 'a', newline='') as f:
                    csv.writer(f).writerow(
                        [n_train, n_feat, target_tau] + sw1_vals)

    print(f"\n{'='*70}")
    print("Done!")
    print(f"  Main CSV         : {all_csv_path}")
    print(f"  Reference values : {ref_csv_path}")
    print(f"  Projections      : {dir_path}")
    print(f"  Convergence      : {out_dir / 'swd_projection_convergence.csv'}")
    print(f"\nTo plot, run:")
    print(f"  python plot_swd_bs_beamspace_vs_tau.py")
    print(f"  python plot_quality_and_fmem_swd_bs_vs_tau.py")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
