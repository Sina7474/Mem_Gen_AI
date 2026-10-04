"""
data_utils_28GHz_LoS.py — Data loading / leakage-safe splitting for the 28 GHz task
====================================================================================
28 GHz analogue of `data_utils.py` (3.5 GHz). Same responsibilities, adapted to
the second study scene:

    - Loads the raw 28 GHz Sionna RT channels + UE coordinates, using the SAME
      preprocessing as the 28 GHz DDIM trainer
      (`Code/DDIM_FMM/train_DDIM_tau_ema_28GHz_LoS.py`). Channels are returned in
      the ANTENNA domain as (N, Nr, Nt) complex64, which is what the DL-GF
      beam-alignment model consumes.
    - Loads DDIM-generated synthetic channels (normalized beamspace) and inverts
      the UPA DFT transform back to the antenna domain with the codebook
      convention used at training time (Rx 2x2, Tx 4x8).
    - Loads each DDIM generator's own training coordinates so the downstream test
      set can be built exclusively from channels no generator ever saw.
    - Builds a fixed, reproducible master split.

Differences vs. the 3.5 GHz module
-----------------------------------
    subcarrier index : 0   (the 28 GHz file stores ONE subcarrier;
                            the 3.5 GHz file stored 256 and used index 128)
    propagation      : LoS only — a single .npz, no LoS/NLoS concatenation and
                       no LoS/NLoS balancing anywhere in the split logic
    generator logs   : Code/DDIM_FMM/logs_ema_28GHz_LoS/
                       DDIM_tau_ema_28GHz_LoS_<N>_bs<B>_incremental/
    scene size       : 7 222 users total (vs. ~40 k + ~40 k at 3.5 GHz), so the
                       test / training-pool sizes are correspondingly smaller

Unchanged from 3.5 GHz
-----------------------
    array geometry   : UE UPA 2x2 (Nr = 4), BS UPA 4x8 (Nt = 32)
    beamspace basis  : unitary kron DFT codebooks, identical matrices
    coordinate match : (x, y, z) rounded to 3 decimals and hashed
"""

import os
import numpy as np

# ── Constants matching the 28 GHz DDIM training preprocessing ───────────────
SUBCARRIER_INDEX = 0             # the 28 GHz file carries a single subcarrier
NRX_X, NRX_Y = 2, 2              # UE UPA (Nr = 4)
NTX_X, NTX_Y = 4, 8              # BS UPA (Nt = 32)
COORD_DECIMALS = 3               # rounding precision for coordinate matching

# Generated-sample directory written by generate_downstream_28GHz_LoS.py
GEN_SUBDIR = "generated_downstream"


# ─────────────────────────────────────────────────────────────────────────────
# UPA / BEAMSPACE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    """Unitary 2-D DFT codebook for a UPA with Nx*Ny elements.

    Returns kron(F_Nx, F_Ny) with F_N = DFT_N / sqrt(N). This is numerically the
    same matrix the DDIM trainer builds (it calls kron(F_Ny, F_Nx) with the two
    array dimensions swapped), so the forward and inverse transforms match.
    """
    Fx = np.fft.fft(np.eye(Nx, dtype=np.complex64), axis=0) / np.sqrt(Nx)
    Fy = np.fft.fft(np.eye(Ny, dtype=np.complex64), axis=0) / np.sqrt(Ny)
    return np.kron(Fx, Fy).astype(np.complex64)


def beamspace_to_antenna(Hv: np.ndarray) -> np.ndarray:
    """Inverse UPA beamspace transform: H_antenna = Ar @ Hv @ At^H."""
    Ar = upa_dft_codebook(NRX_X, NRX_Y)
    At = upa_dft_codebook(NTX_X, NTX_Y)
    return (Ar @ Hv) @ At.conj().T


# ─────────────────────────────────────────────────────────────────────────────
# RAW 28 GHz LOADING  →  antenna-domain (N, Nr, Nt) + coords
# ─────────────────────────────────────────────────────────────────────────────
def load_sionna_raw_28ghz(path: str):
    """Load the raw 28 GHz LoS channels + UE coordinates.

    The stored array has shape (U, Nr, 1, Nt, 1, Nsc, 4) where the last axis is
    [channel, x, y, z]. Indexing mirrors the 28 GHz DDIM trainer exactly.

    Returns
    -------
    H      : (U, Nr, Nt) complex64 — antenna-domain channel
    coords : (U, 3)      float32   — UE (x, y, z) position
    """
    d = np.load(path, allow_pickle=True)
    arr = d["combined_array"]
    sel = arr[:, :, 0, :, 0, SUBCARRIER_INDEX, :]      # (U, Nr, Nt, 4)
    H = sel[:, :, :, 0].astype(np.complex64)           # (U, Nr, Nt)
    coords = np.real(sel[:, 0, 0, 1:]).astype(np.float32)   # (U, 3)
    return H, coords


def coord_keys(coords: np.ndarray) -> np.ndarray:
    """Hashable string keys for (x, y, z) coordinate matching."""
    rounded = np.round(coords.astype(np.float64), COORD_DECIMALS)
    return np.array([f"{r[0]:.3f}_{r[1]:.3f}_{r[2]:.3f}" for r in rounded])


# ─────────────────────────────────────────────────────────────────────────────
# 28 GHz DDIM RUN LOCATIONS
# ─────────────────────────────────────────────────────────────────────────────
def ddim_run_dir(logs_base: str, ddim_n: int, n_feat: int = 256) -> str:
    """Directory of the 28 GHz DDIM run for generator size N.

    Reproduces the naming rule of train_DDIM_tau_ema_28GHz_LoS.py:
    batch size B = min(N, 500), width suffix only when W != 256.
    """
    bs = min(ddim_n, 500)
    nfeat_suffix = f"_nfeat{n_feat}" if n_feat != 256 else ""
    return os.path.join(
        logs_base, f"DDIM_tau_ema_28GHz_LoS_{ddim_n}{nfeat_suffix}_bs{bs}_incremental")


def load_ddim_train_coords(logs_base: str, ddim_n: int, n_feat: int = 256) -> np.ndarray:
    """(x, y, z) coordinates of the users the DDIM_N generator was trained on."""
    p = os.path.join(ddim_run_dir(logs_base, ddim_n, n_feat), "train_coords.npy")
    if not os.path.exists(p):
        return np.zeros((0, 3), dtype=np.float32)
    return np.real(np.load(p)).astype(np.float32)


def load_ddim_train_channels(logs_base: str, ddim_n: int,
                              H_all: np.ndarray,
                              n_feat: int = 256) -> np.ndarray:
    """Load the exact raw antenna-domain channels the DDIM_N generator was
    trained on, using the stored index array.

    The index array ``indices_28ghz_los.npy`` is a master permutation of all
    7222 users; the first N entries form the training set for generator size N.

    Returns (N, Nr, Nt) complex64 — antenna-domain channels.
    """
    run = ddim_run_dir(logs_base, ddim_n, n_feat)
    indices = np.load(os.path.join(run, "indices_28ghz_los.npy"))
    train_idx = indices[:ddim_n]
    return H_all[train_idx]


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC (DDIM-GENERATED) LOADING  →  antenna domain (M, Nr, Nt)
# ─────────────────────────────────────────────────────────────────────────────
def synthetic_path(logs_base: str, ddim_n: int, tau: int, n_generate: int,
                   n_feat: int = 256) -> str:
    return os.path.join(ddim_run_dir(logs_base, ddim_n, n_feat), GEN_SUBDIR,
                        f"generated_tau_{tau}_{n_generate}.npz")


def load_synthetic_pool(logs_base: str, ddim_n: int, tau: int, n_generate: int,
                        n_feat: int = 256) -> np.ndarray:
    """Load DDIM-generated 28 GHz channels for (N, tau) → antenna-domain (M, Nr, Nt).

    The .npz stores key 'channels' of shape (M, 2, Nr, Nt) in NORMALIZED
    BEAMSPACE. Converting back to the antenna domain with the training codebook
    puts real and synthetic channels in the same representation.
    """
    p = synthetic_path(logs_base, ddim_n, tau, n_generate, n_feat)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Generated file not found: {p}\n"
            f"Run:  cd Code/DDIM_FMM && python generate_downstream_28GHz_LoS.py")
    ch = np.load(p)["channels"]                            # (M, 2, Nr, Nt)
    Hv = (ch[:, 0] + 1j * ch[:, 1]).astype(np.complex64)   # beamspace
    return beamspace_to_antenna(Hv).astype(np.complex64)   # antenna domain


# ─────────────────────────────────────────────────────────────────────────────
# LEAKAGE-SAFE MASTER SPLIT (LoS only — no LoS/NLoS balancing)
# ─────────────────────────────────────────────────────────────────────────────
def build_master_split(H, coords, ddim_seen_keys, n_test=2000, seed=42):
    """Fixed, reproducible, leakage-safe split of the 28 GHz scene.

    The TEST set is drawn ONLY from users whose coordinates were never seen by
    ANY DDIM generator, and is disjoint from the training pool. The training
    pool is every remaining user.

    Returns a dict with the test channels, the test indices, the pool indices
    and the coordinate keys (for the overlap report).
    """
    rng = np.random.default_rng(seed)
    keys = coord_keys(coords)

    unseen_idx = np.where(np.array([k not in ddim_seen_keys for k in keys]))[0]
    if len(unseen_idx) < n_test:
        raise ValueError(
            f"Not enough DDIM-unseen users for the test set: "
            f"{len(unseen_idx)} available, {n_test} requested.")

    test_idx = np.sort(rng.permutation(unseen_idx)[:n_test])
    pool_idx = np.setdiff1d(np.arange(len(H)), test_idx)

    return {
        "H_test": H[test_idx],
        "test_idx": test_idx,
        "pool_idx": pool_idx,
        "keys": keys,
    }


def draw_real_subset(H, pool_idx, n_real, seed):
    """Draw a real training subset of size n_real from the (test-free) pool.

    Fixed seed → reproducible across experiments and re-runs.
    Returns (H_subset antenna-domain, indices used).
    """
    if n_real > len(pool_idx):
        raise ValueError(
            f"Requested {n_real} real training users but the pool holds only "
            f"{len(pool_idx)}.")
    rng = np.random.default_rng(seed)
    sel = rng.permutation(pool_idx)[:n_real]
    return H[sel], sel
