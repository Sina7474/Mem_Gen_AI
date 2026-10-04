"""
data_utils.py — Data loading and leakage-safe splitting for Task 10 (Beam Alignment)
====================================================================================
Self-contained (no external project dependencies). Handles:

    - Loading raw Sionna RT LoS/NLoS channels + UE coordinates (subcarrier 128,
      matching the DDIM training preprocessing exactly). Channels are returned in
      the ANTENNA domain as (N, Nr, Nt) complex64 — the format the DL-GF
      beam-alignment model expects.
    - Loading DDIM-generated synthetic channels (normalized beamspace) and
      converting them back to the antenna domain with the SAME UPA DFT codebook
      convention used to create the DDIM training data (Rx 2x2, Tx 4x8). The
      reference DLGF `int(sqrt(Nt))` inversion is INCORRECT for Nt=32 (not a
      perfect square); this module uses the correct kron(F_Nx, F_Ny) convention.
    - Loading the DDIM generator's own training coordinates so the downstream
      test set can be built from channels the DDIM never saw (no leakage).
    - Building a fixed, reproducible master split (test held out from all).
    - Coordinate-based overlap accounting for the data-overlap report.

Coordinate matching:
    Each UE channel has a unique (x, y, z) position stored in the raw dataset.
    The DDIM training script saved these as train_coords.npy. We match channels
    across datasets by rounding coordinates to a fixed precision and hashing.
"""

import os
import numpy as np

# ── Constants matching the DDIM training preprocessing ───────────────────────
SUBCARRIER_INDEX = 128           # DDIM used subcarrier 128 (== middle of 256)
NRX_X, NRX_Y = 2, 2              # UE UPA (Nr = 4)
NTX_X, NTX_Y = 4, 8             # BS UPA (Nt = 32) — matches DDIM/CRNet convention
COORD_DECIMALS = 3               # rounding precision for coordinate matching


# ─────────────────────────────────────────────────────────────────────────────
# UPA / BEAMSPACE HELPERS (kron(F_Nx, F_Ny), unitary — matches DDIM convention)
# ─────────────────────────────────────────────────────────────────────────────
def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    """Unitary 2-D DFT codebook for a UPA with Nx*Ny elements.
    Returns (Nx*Ny, Nx*Ny) = kron(F_Nx, F_Ny), where F_N = DFT_N / sqrt(N).
    """
    Fx = np.fft.fft(np.eye(Nx, dtype=np.complex64), axis=0) / np.sqrt(Nx)
    Fy = np.fft.fft(np.eye(Ny, dtype=np.complex64), axis=0) / np.sqrt(Ny)
    return np.kron(Fx, Fy).astype(np.complex64)


def beamspace_to_antenna(Hv: np.ndarray) -> np.ndarray:
    """Inverse UPA beamspace transform: H_antenna = Ar @ Hv @ At^H.
    Hv: (..., Nr, Nt) complex → same shape in antenna domain.
    """
    Ar = upa_dft_codebook(NRX_X, NRX_Y)
    At = upa_dft_codebook(NTX_X, NTX_Y)
    return (Ar @ Hv) @ At.conj().T


# ─────────────────────────────────────────────────────────────────────────────
# RAW SIONNA LOADING  →  antenna-domain (N, Nr, Nt) + coords
# ─────────────────────────────────────────────────────────────────────────────
def load_sionna_raw(path: str):
    """Load raw Sionna RT channels + coordinates, matching DDIM preprocessing.

    Returns
    -------
    H     : (N, Nr, Nt) complex64 — antenna-domain channel at subcarrier 128
    coords: (N, 3) float32        — UE (x, y, z) positions
    """
    d = np.load(path, allow_pickle=True)
    arr = d["combined_array"]                          # (N, Nr, 1, Nt, 1, Nsc, 4)
    sel = arr[:, :, 0, :, 0, SUBCARRIER_INDEX, :]      # (N, Nr, Nt, 4)
    H = sel[:, :, :, 0].astype(np.complex64)           # (N, Nr, Nt) channel
    coords = sel[:, 0, 0, 1:].astype(np.complex64)     # (N, 3) position (x, y, z)
    coords = np.real(coords).astype(np.float32)        # imaginary part is 0
    return H, coords


def coord_keys(coords: np.ndarray) -> np.ndarray:
    """Return hashable string keys for (x, y, z) coordinate matching."""
    rounded = np.round(coords.astype(np.float64), COORD_DECIMALS)
    return np.array([f"{r[0]:.3f}_{r[1]:.3f}_{r[2]:.3f}" for r in rounded])


# ─────────────────────────────────────────────────────────────────────────────
# DDIM GENERATOR TRAINING COORDINATES (for leakage accounting)
# ─────────────────────────────────────────────────────────────────────────────
def load_ddim_train_coords(logs_base: str, ddim_n: int) -> np.ndarray:
    """Load the (x, y, z) coordinates of the channels the DDIM_N generator saw.
    Returns (M, 3) float32 array, or an empty array if not found.
    """
    p = os.path.join(logs_base, f"DDIM_tau_{ddim_n}_incremental", "train_coords.npy")
    if not os.path.exists(p):
        return np.zeros((0, 3), dtype=np.float32)
    c = np.load(p)
    return np.real(c).astype(np.float32)


def load_ddim_train_coords_ema(logs_ema_base: str, ddim_n: int,
                               n_feat: int = 256) -> np.ndarray:
    """Same as load_ddim_train_coords but for the EMA runs (logs_ema/…)."""
    p = os.path.join(ema_run_dir(logs_ema_base, ddim_n, n_feat), "train_coords.npy")
    if not os.path.exists(p):
        return np.zeros((0, 3), dtype=np.float32)
    return np.real(np.load(p)).astype(np.float32)


def load_ddim_train_channels_ema(logs_ema_base: str, ddim_n: int,
                                 H_los: np.ndarray, H_nlos: np.ndarray,
                                 n_feat: int = 256) -> np.ndarray:
    """Load the exact raw antenna-domain channels the DDIM_N EMA generator was
    trained on, using the stored LoS/NLoS index arrays.

    The DDIM trainer draws N/2 from LoS and N/2 from NLoS (balanced).  The
    index arrays ``indices_los.npy`` and ``indices_nlos.npy`` are shared
    across all N values (same master permutation); the first N/2 entries of
    each array form the training set for generator size N.

    Returns (N, Nr, Nt) complex64 — antenna-domain channels.
    """
    run = ema_run_dir(logs_ema_base, ddim_n, n_feat)
    idx_los = np.load(os.path.join(run, "indices_los.npy"))
    idx_nlos = np.load(os.path.join(run, "indices_nlos.npy"))
    n_half = ddim_n // 2
    return np.concatenate([H_los[idx_los[:n_half]],
                           H_nlos[idx_nlos[:n_half]]], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC (DDIM-GENERATED) LOADING  →  antenna domain (M, Nr, Nt)
# ─────────────────────────────────────────────────────────────────────────────
def synthetic_path(logs_base: str, ddim_n: int, tau: int, suffix: str = "") -> str:
    return os.path.join(logs_base, f"DDIM_tau_{ddim_n}_incremental",
                        "generated_tau", f"generated_tau_{tau}{suffix}.npz")


# ── EMA runs (logs_ema/…) — these are the generators the memorization metrics
#    (f_mem, FCD in DDIM_Evaluation/dataset_size_effect) were computed from. ──
def ema_run_dir(logs_ema_base: str, ddim_n: int, n_feat: int = 256) -> str:
    """Directory of the 3.5 GHz EMA run for generator size N.

    Reproduces the naming rule of train_DDIM_tau_ema.py:
    batch size B = min(N, 500), width suffix only when W != 256.
    """
    bs = min(ddim_n, 500)
    nfeat_suffix = f"_nfeat{n_feat}" if n_feat != 256 else ""
    return os.path.join(logs_ema_base,
                        f"DDIM_tau_ema_{ddim_n}{nfeat_suffix}_bs{bs}_incremental")


def ema_synthetic_path(logs_ema_base: str, ddim_n: int, tau: int,
                       seed: int = 0, n_feat: int = 256) -> str:
    """Path of the cached EMA sample pool used by the f_mem / FCD pipeline."""
    return os.path.join(ema_run_dir(logs_ema_base, ddim_n, n_feat),
                        "generated_ema", f"gen_ema_tau{tau}_seed{seed}.npz")


def _beamspace_npz_to_antenna(path: str) -> np.ndarray:
    """Read a generated .npz (key 'channels', (M,2,Nr,Nt) normalized beamspace)
    and return antenna-domain (M, Nr, Nt) complex64.

    A sanity guard rejects pools that are not per-sample max-magnitude
    normalized.  The DDIM was trained on beamspace channels scaled so that
    max_i |Hv_i| == 1, therefore every valid generated pool must have a
    per-sample peak magnitude close to 1.  A pool whose peak magnitude is far
    from 1 means the sampler ran with weights that were never loaded (or a
    mismatched checkpoint) and would silently poison every downstream result.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Generated file not found: {path}")
    ch = np.load(path)["channels"]                         # (M, 2, Nr, Nt)
    mag = np.sqrt(ch[:, 0] ** 2 + ch[:, 1] ** 2)
    peak = np.median(mag.reshape(len(ch), -1).max(axis=1))
    if not (0.5 <= peak <= 2.0):
        raise ValueError(
            f"Generated pool is not in normalized beamspace: {path}\n"
            f"  median per-sample peak magnitude = {peak:.4f} (expected ~1.0).\n"
            f"  This pool was produced by a sampler whose checkpoint weights "
            f"were not loaded — regenerate it before using it downstream.")
    Hv = (ch[:, 0] + 1j * ch[:, 1]).astype(np.complex64)   # beamspace
    return beamspace_to_antenna(Hv).astype(np.complex64)   # antenna domain


def load_synthetic_pool(logs_base: str, ddim_n: int, tau: int, suffix: str = "") -> np.ndarray:
    """Load DDIM-generated channels for (ddim_n, tau) → antenna-domain (M, Nr, Nt).

    The generated .npz stores key 'channels' with shape (M, 2, Nr, Nt) in
    NORMALIZED BEAMSPACE. We convert to the antenna domain with the codebook
    convention that matches DDIM training, so real/synthetic data share the same
    antenna-domain representation.

    suffix : file name suffix, e.g. "_10k" for the 10 000-sample files.
    """
    return _beamspace_npz_to_antenna(
        synthetic_path(logs_base, ddim_n, tau, suffix=suffix))


def load_synthetic_pool_ema(logs_ema_base: str, ddim_n: int, tau: int,
                            seed: int = 0, n_feat: int = 256) -> np.ndarray:
    """Load the EMA sample pool for (ddim_n, tau) → antenna-domain (M, Nr, Nt).

    These are the *same* files that DDIM_Evaluation/dataset_size_effect uses to
    compute f_mem and FCD, so a downstream curve at horizon tau corresponds
    exactly to the memorization value reported at that tau.
    """
    return _beamspace_npz_to_antenna(
        ema_synthetic_path(logs_ema_base, ddim_n, tau, seed=seed, n_feat=n_feat))


# ─────────────────────────────────────────────────────────────────────────────
# LEAKAGE-SAFE MASTER SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def build_master_split(H_los, coords_los, H_nlos, coords_nlos,
                       ddim_seen_keys, n_test_per_side=500, seed=42):
    """Build a fixed, reproducible, leakage-safe split.

    The TEST set (n_test_per_side LoS + n_test_per_side NLoS) is drawn ONLY from
    channels whose coordinates were never seen by any DDIM generator, and is
    fully disjoint from the training pool. The training pool is everything else.
    """
    rng = np.random.default_rng(seed)

    keys_los = coord_keys(coords_los)
    keys_nlos = coord_keys(coords_nlos)

    unseen_los_idx = np.where(np.array([k not in ddim_seen_keys for k in keys_los]))[0]
    unseen_nlos_idx = np.where(np.array([k not in ddim_seen_keys for k in keys_nlos]))[0]

    if len(unseen_los_idx) < n_test_per_side or len(unseen_nlos_idx) < n_test_per_side:
        raise ValueError(
            f"Not enough DDIM-unseen channels for the test set: "
            f"LoS unseen={len(unseen_los_idx)}, NLoS unseen={len(unseen_nlos_idx)}, "
            f"need {n_test_per_side} each.")

    test_los_idx = rng.permutation(unseen_los_idx)[:n_test_per_side]
    test_nlos_idx = rng.permutation(unseen_nlos_idx)[:n_test_per_side]

    pool_los_idx = np.setdiff1d(np.arange(len(H_los)), test_los_idx)
    pool_nlos_idx = np.setdiff1d(np.arange(len(H_nlos)), test_nlos_idx)

    return {
        "H_test": np.concatenate([H_los[test_los_idx], H_nlos[test_nlos_idx]], axis=0),
        "test_los_idx": test_los_idx,
        "test_nlos_idx": test_nlos_idx,
        "pool_los_idx": pool_los_idx,
        "pool_nlos_idx": pool_nlos_idx,
        "keys_los": keys_los,
        "keys_nlos": keys_nlos,
    }


def draw_real_subset(H_los, H_nlos, pool_los_idx, pool_nlos_idx, n_real, seed):
    """Draw a balanced real subset (n_real/2 LoS + n_real/2 NLoS) from the pool.

    Returns (H_subset antenna-domain, los_idx_used, nlos_idx_used).
    Fixed seed → reproducible and nested-consistent across experiments.
    """
    rng = np.random.default_rng(seed)
    n_half = n_real // 2
    los_sel = rng.permutation(pool_los_idx)[:n_half]
    nlos_sel = rng.permutation(pool_nlos_idx)[:n_half]
    H = np.concatenate([H_los[los_sel], H_nlos[nlos_sel]], axis=0)
    return H, los_sel, nlos_sel
