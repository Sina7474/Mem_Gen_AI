"""
data_utils.py — Data loading and leakage-safe splitting for Task 09
====================================================================
Handles:
    - Loading raw Sionna RT LoS/NLoS channels + UE coordinates (subcarrier 128,
      matching the DDIM training preprocessing exactly).
    - Loading DDIM-generated synthetic channels (normalized beamspace) and
      converting them back to the antenna domain (consistent codebook).
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

from crnet_core import beamspace_to_antenna

# ── Constants matching the DDIM training preprocessing ───────────────────────
SUBCARRIER_INDEX = 128           # DDIM used subcarrier 128 (== middle of 256)
NRX_X, NRX_Y = 2, 2              # UE UPA (Nr = 4)
NTX_X, NTX_Y = 4, 8              # BS UPA (Nt = 32); CRNet convention matches DDIM
COORD_DECIMALS = 3               # rounding precision for coordinate matching


# ─────────────────────────────────────────────────────────────────────────────
# RAW SIONNA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_sionna_raw(path: str):
    """Load raw Sionna RT channels + coordinates, matching DDIM preprocessing.

    Returns
    -------
    H     : (N, Nr, Nt) complex64 — antenna-domain channel at subcarrier 128
    coords: (N, 3) float32        — UE (x, y, z) positions
    """
    d = np.load(path, allow_pickle=True)
    arr = d["combined_array"]   # (N, Nr, 1, Nt, 1, Nsc, 4) complex64
    # Match DDIM: combined_array[:, :, 0, :, 0, SUBCARRIER_INDEX, :]
    sel = arr[:, :, 0, :, 0, SUBCARRIER_INDEX, :]   # (N, Nr, Nt, 4)
    H = sel[:, :, :, 0].astype(np.complex64)        # (N, Nr, Nt) channel
    coords = sel[:, 0, 0, 1:].astype(np.complex64)  # (N, 3) position (x, y, z)
    coords = np.real(coords).astype(np.float32)     # imaginary part is 0
    return H, coords


def coord_keys(coords: np.ndarray) -> np.ndarray:
    """Return an array of hashable string keys for (x, y, z) coordinate matching."""
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


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC (DDIM-GENERATED) LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_synthetic_pool(logs_base: str, ddim_n: int, tau: int) -> np.ndarray:
    """Load DDIM-generated channels for (ddim_n, tau) → antenna-domain (M, Nr, Nt).

    The generated .npz stores key 'channels' with shape (M, 2, Nr, Nt) in
    NORMALIZED BEAMSPACE. We convert to the antenna domain with the codebook
    convention that matches DDIM training, so real/synthetic data share the
    same representation after to_tensor().
    """
    p = os.path.join(logs_base, f"DDIM_tau_{ddim_n}_incremental",
                     "generated_tau", f"generated_tau_{tau}.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Generated file not found: {p}")
    ch = np.load(p)["channels"]                                  # (M, 2, Nr, Nt)
    Hv = (ch[:, 0] + 1j * ch[:, 1]).astype(np.complex64)         # beamspace
    H = beamspace_to_antenna(Hv, NRX_X, NRX_Y, NTX_X, NTX_Y)     # antenna domain
    return H.astype(np.complex64)


# ─────────────────────────────────────────────────────────────────────────────
# LEAKAGE-SAFE MASTER SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def build_master_split(H_los, coords_los, H_nlos, coords_nlos,
                       ddim_seen_keys, n_test_per_side=500, seed=42):
    """Build a fixed, reproducible, leakage-safe split.

    The TEST set (n_test_per_side LoS + n_test_per_side NLoS) is drawn ONLY from
    channels whose coordinates were never seen by any DDIM generator, and is
    fully disjoint from the training pool. The training pool is everything else.

    Returns a dict with test arrays and per-side pool index arrays.
    """
    rng = np.random.default_rng(seed)

    keys_los = coord_keys(coords_los)
    keys_nlos = coord_keys(coords_nlos)

    unseen_los = np.array([k not in ddim_seen_keys for k in keys_los])
    unseen_nlos = np.array([k not in ddim_seen_keys for k in keys_nlos])

    unseen_los_idx = np.where(unseen_los)[0]
    unseen_nlos_idx = np.where(unseen_nlos)[0]

    if len(unseen_los_idx) < n_test_per_side or len(unseen_nlos_idx) < n_test_per_side:
        raise ValueError(
            f"Not enough DDIM-unseen channels for the test set: "
            f"LoS unseen={len(unseen_los_idx)}, NLoS unseen={len(unseen_nlos_idx)}, "
            f"need {n_test_per_side} each.")

    # Test indices (from unseen channels only)
    test_los_idx = rng.permutation(unseen_los_idx)[:n_test_per_side]
    test_nlos_idx = rng.permutation(unseen_nlos_idx)[:n_test_per_side]

    # Training pool = everything not in the test set (per side)
    all_los = np.arange(len(H_los))
    all_nlos = np.arange(len(H_nlos))
    pool_los_idx = np.setdiff1d(all_los, test_los_idx)
    pool_nlos_idx = np.setdiff1d(all_nlos, test_nlos_idx)

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
    Uses a fixed seed so subsets are reproducible and nested-consistent.
    """
    rng = np.random.default_rng(seed)
    n_half = n_real // 2
    los_sel = rng.permutation(pool_los_idx)[:n_half]
    nlos_sel = rng.permutation(pool_nlos_idx)[:n_half]
    H = np.concatenate([H_los[los_sel], H_nlos[nlos_sel]], axis=0)
    return H, los_sel, nlos_sel
