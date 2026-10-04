"""
data_utils_28GHz_LoS_SceneV1.py — Data loading for the NEW 28 GHz SceneV1 beam alignment
=========================================================================================
Analogue of `data_utils_28GHz_LoS.py` but for the SceneV1 dataset.

Dataset: Final_Single_Scene_Channel_Sionna_V1_28GHz_LoS_UPA.npz
    12 940 LoS users, same antenna config (Rx 2x2, Tx 8x4 = 4x32).

Logs tree: Code/DDIM_FMM/logs_ema_28GHz_LoS_SceneV1/
    DDIM_tau_ema_28GHz_LoS_SceneV1_<N>_bs<B>_incremental/
"""

import os
import numpy as np

# ── Constants matching the DDIM training preprocessing ──────────────────────
SUBCARRIER_INDEX = 0
NRX_X, NRX_Y = 2, 2              # UE UPA (Nr = 4)
NTX_X, NTX_Y = 4, 8              # BS UPA (Nt = 32)
COORD_DECIMALS = 3

GEN_SUBDIR = "generated_downstream"
INDICES_FILE = "indices_28ghz_los_scenev1.npy"


# ─────────────────────────────────────────────────────────────────────────────
# UPA / BEAMSPACE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    Fx = np.fft.fft(np.eye(Nx, dtype=np.complex64), axis=0) / np.sqrt(Nx)
    Fy = np.fft.fft(np.eye(Ny, dtype=np.complex64), axis=0) / np.sqrt(Ny)
    return np.kron(Fx, Fy).astype(np.complex64)


def beamspace_to_antenna(Hv: np.ndarray) -> np.ndarray:
    """Inverse UPA beamspace transform: H_antenna = Ar @ Hv @ At^H."""
    Ar = upa_dft_codebook(NRX_X, NRX_Y)
    At = upa_dft_codebook(NTX_X, NTX_Y)
    return (Ar @ Hv) @ At.conj().T


# ─────────────────────────────────────────────────────────────────────────────
# RAW LOADING  →  antenna-domain (N, Nr, Nt) + coords
# ─────────────────────────────────────────────────────────────────────────────
def load_sionna_raw_28ghz_scenev1(path: str):
    """Load the raw 28 GHz LoS SceneV1 channels + UE coordinates.

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
    rounded = np.round(coords.astype(np.float64), COORD_DECIMALS)
    return np.array([f"{r[0]:.3f}_{r[1]:.3f}_{r[2]:.3f}" for r in rounded])


# ─────────────────────────────────────────────────────────────────────────────
# DDIM RUN LOCATIONS
# ─────────────────────────────────────────────────────────────────────────────
def ddim_run_dir(logs_base: str, ddim_n: int, n_feat: int = 256) -> str:
    bs = min(ddim_n, 500)
    nfeat_suffix = f"_nfeat{n_feat}" if n_feat != 256 else ""
    return os.path.join(
        logs_base,
        f"DDIM_tau_ema_28GHz_LoS_SceneV1_{ddim_n}{nfeat_suffix}_bs{bs}_incremental")


def load_ddim_train_coords(logs_base: str, ddim_n: int, n_feat: int = 256) -> np.ndarray:
    p = os.path.join(ddim_run_dir(logs_base, ddim_n, n_feat), "train_coords.npy")
    if not os.path.exists(p):
        return np.zeros((0, 3), dtype=np.float32)
    return np.real(np.load(p)).astype(np.float32)


def load_ddim_train_channels(logs_base: str, ddim_n: int,
                              H_all: np.ndarray,
                              n_feat: int = 256) -> np.ndarray:
    """Load the exact raw antenna-domain channels the DDIM_N generator was
    trained on, using the stored index array."""
    run = ddim_run_dir(logs_base, ddim_n, n_feat)
    indices = np.load(os.path.join(run, INDICES_FILE))
    train_idx = indices[:ddim_n]
    return H_all[train_idx]


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC LOADING  →  antenna domain (M, Nr, Nt)
# ─────────────────────────────────────────────────────────────────────────────
def synthetic_path(logs_base: str, ddim_n: int, tau: int, n_generate: int,
                   n_feat: int = 256) -> str:
    return os.path.join(ddim_run_dir(logs_base, ddim_n, n_feat), GEN_SUBDIR,
                        f"generated_tau_{tau}_{n_generate}.npz")


def load_synthetic_pool(logs_base: str, ddim_n: int, tau: int, n_generate: int,
                        n_feat: int = 256) -> np.ndarray:
    """Load DDIM-generated SceneV1 channels → antenna-domain (M, Nr, Nt)."""
    p = synthetic_path(logs_base, ddim_n, tau, n_generate, n_feat)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Generated file not found: {p}\n"
            f"Run:  cd Code/DDIM_FMM && python generate_downstream_28GHz_LoS_SceneV1.py")
    ch = np.load(p)["channels"]                            # (M, 2, Nr, Nt)
    Hv = (ch[:, 0] + 1j * ch[:, 1]).astype(np.complex64)   # beamspace
    return beamspace_to_antenna(Hv).astype(np.complex64)   # antenna domain


# ─────────────────────────────────────────────────────────────────────────────
# LEAKAGE-SAFE MASTER SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def build_master_split(H, coords, ddim_seen_keys, n_test=5000, seed=42):
    """Fixed, reproducible, leakage-safe split of the SceneV1 scene."""
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
