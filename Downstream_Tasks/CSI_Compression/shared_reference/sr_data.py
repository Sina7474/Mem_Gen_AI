"""
sr_data.py — Shared-Reference data layer for the CRNet CSI-compression study
============================================================================
Implements the *shared-reference* design of

    reports/CRNet_Shared_Reference_Experiment_Implementation_Guide.md

The single, non-negotiable invariant is:

        N_csi == N        and       I_DDIM == I_CRNet ,

i.e. the exact channels CRNet uses as its real reference D_N are byte-for-byte
the same channels the DDIM(W=256, N) generator was trained on.  This is
guaranteed here by rebuilding D_N from the *same* raw dataset and the *same*
saved shuffle indices the DDIM training used, and asserting equality against the
generator's own ``train.npy``.

Everything downstream (real reference, synthetic pool, validation and test) is
represented in the *same normalised beamspace* the DDIM operated in:

    real   :  antenna  --antenna_to_beamspace(2,2,8,4)-->  beamspace --to_tensor-->
    synth  :  beamspace (generator output)                            --to_tensor-->

Because that DFT is exactly ``upa_to_beamspace(2,2,8,4)`` used by the DDIM data
pipeline, real and synthetic channels share one basis, so the CRNet input space
and the NMSE metric are consistent across the two sources.

This module never trains anything and never touches the legacy Task-09 files.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the vetted transforms/metric from the legacy core (imported, not edited).
HERE = Path(__file__).resolve().parent
SRC_DIR = HERE.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from crnet_core import to_tensor, beamspace_to_antenna  # noqa: E402
from data_utils import load_sionna_raw, coord_keys       # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT = HERE.parent.parent.parent
DATASET_DIR = PROJECT / "dataset"
LOS_NPZ = DATASET_DIR / "Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"
NLOS_NPZ = DATASET_DIR / "Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz"

LOGS_EMA = PROJECT / "Code" / "DDIM_FMM" / "logs_ema"
WINDOW_CSV = (PROJECT / "DDIM_Evaluation" / "dataset_size_effect" /
              "results" / "dsize_fcd_fmem.csv")

RESULTS_DIR = HERE / "results"
CACHE_DIR = RESULTS_DIR / "cache"

# ─────────────────────────────────────────────────────────────────────────────
# Operating domain (IDENTICAL to the legacy Task-09 pipeline that reached -6 dB)
#
# CRNet operates in the ANTENNA domain, exactly like src/data_utils.py:
#   * real channels are used as-is (antenna domain) and passed to to_tensor;
#   * DDIM beamspace samples are mapped BACK to the antenna domain with
#     beamspace_to_antenna(Nrx_x=2, Nrx_y=2, Ntx_x=4, Ntx_y=8) — the very same
#     codebook convention the legacy load_synthetic_pool used.
# This keeps real and synthetic in one shared representation AND reproduces the
# legacy reference NMSE (~ -6 dB), which a beamspace-domain CRNet would not.
# ─────────────────────────────────────────────────────────────────────────────
NRX_X, NRX_Y = 2, 2
NTX_X, NTX_Y = 4, 8


def ema_dir(n: int) -> Path:
    """Directory of the W=256 EMA DDIM generator trained on N channels.

    Naming is not uniform across N: N=200 and N=1000 have no batch-size suffix,
    whereas N=500 was trained with an explicit bs500 tag. The map below records
    the exact W=256 (n_feat=256) directory for each supported N; extend it if you
    add more sizes.
    """
    override = {
        200: "DDIM_tau_ema_200_incremental",
        500: "DDIM_tau_ema_500_bs500_incremental",
        1000: "DDIM_tau_ema_1000_incremental",
    }
    name = override.get(n, f"DDIM_tau_ema_{n}_incremental")
    return LOGS_EMA / name


# ─────────────────────────────────────────────────────────────────────────────
# Raw data (cached)
# ─────────────────────────────────────────────────────────────────────────────
def load_or_cache_raw():
    """Load raw Sionna LoS/NLoS antenna channels + coords, caching to .npy.

    Caching is best-effort: if the cache cannot be written (e.g. a full disk),
    the raw arrays are still returned from memory.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = {
        "H_los": CACHE_DIR / "sr_H_los.npy", "c_los": CACHE_DIR / "sr_c_los.npy",
        "H_nlos": CACHE_DIR / "sr_H_nlos.npy", "c_nlos": CACHE_DIR / "sr_c_nlos.npy",
    }
    if all(p.exists() for p in f.values()):
        try:
            return (np.load(f["H_los"]), np.load(f["c_los"]),
                    np.load(f["H_nlos"]), np.load(f["c_nlos"]))
        except (OSError, ValueError):
            pass  # corrupt/partial cache → reload from source
    H_los, c_los = load_sionna_raw(str(LOS_NPZ))
    H_nlos, c_nlos = load_sionna_raw(str(NLOS_NPZ))
    try:
        np.save(f["H_los"], H_los);  np.save(f["c_los"], c_los)
        np.save(f["H_nlos"], H_nlos); np.save(f["c_nlos"], c_nlos)
    except OSError as e:
        print(f"  [warn] could not cache raw arrays ({e}); using in-memory copy.")
        for p in f.values():
            try:
                if p.exists():
                    p.unlink()  # drop any partial file
            except OSError:
                pass
    return H_los, c_los, H_nlos, c_nlos


def _shared_indices(ref_n: int = 1000):
    """Per-side shuffle indices used by every EMA generator (shared split)."""
    d = ema_dir(ref_n)
    il = np.load(d / "indices_los.npy")
    inl = np.load(d / "indices_nlos.npy")
    return il, inl


# ─────────────────────────────────────────────────────────────────────────────
# Antenna-domain tensor helpers  (match legacy exactly)
# ─────────────────────────────────────────────────────────────────────────────
def real_to_tensor(H_ant: np.ndarray):
    """Antenna-domain real channels (M,Nr,Nt) → CRNet tensor.

    Identical to the legacy real path: to_tensor is applied directly to the
    antenna-domain channel (per-sample max-amplitude normalisation, real/imag
    stack, mapped to [0,1]).
    """
    return to_tensor(H_ant)


def synth_to_tensor(ch: np.ndarray):
    """DDIM beamspace samples (M,2,Nr,Nt) → antenna-domain CRNet tensor.

    Identical to the legacy load_synthetic_pool: the generator's beamspace
    output is mapped back to the antenna domain with beamspace_to_antenna
    (2,2,4,8) before to_tensor, so real and synthetic share one representation.
    """
    Hv = (ch[:, 0] + 1j * ch[:, 1]).astype(np.complex64)
    H = beamspace_to_antenna(Hv, NRX_X, NRX_Y, NTX_X, NTX_Y)
    return to_tensor(H)


# ─────────────────────────────────────────────────────────────────────────────
# Generalization window  (τ_gen, τ_mem) per (N, W=256)
# ─────────────────────────────────────────────────────────────────────────────
def _log_interp_crossing(x, y, y_target, descending):
    x = np.asarray(x, float); y = np.asarray(y, float); lx = np.log(x)
    for i in range(1, len(y)):
        if descending and y[i - 1] > y_target and y[i] <= y_target:
            f = (y[i - 1] - y_target) / (y[i - 1] - y[i])
            return float(np.exp(lx[i - 1] + f * (lx[i] - lx[i - 1])))
        if (not descending) and y[i - 1] < y_target and y[i] >= y_target:
            f = (y_target - y[i - 1]) / (y[i] - y[i - 1])
            return float(np.exp(lx[i - 1] + f * (lx[i] - lx[i - 1])))
    return float("nan")


def load_window(n: int, fmem_thresh: float = 0.1, rel_tol: float = 0.5,
                min_tau: int = 1):
    """Return (tau_gen, tau_mem) for dataset size N at W=256.

    Uses the same numerical definition as
    DDIM_Evaluation/dataset_size_effect/analyze_generalization_window.py:
      tau_gen  = first τ where FCD(Gen,Test) ≤ floor_N·(1+rel_tol)
      tau_mem  = first τ where f_mem crosses fmem_thresh
    """
    df = pd.read_csv(WINDOW_CSV)
    df = df[(df["N"] == n) & (df["tau"] >= min_tau)].sort_values("tau")
    if df.empty:
        raise ValueError(f"No window data for N={n} in {WINDOW_CSV}")
    tau = df["tau"].to_numpy()
    fcd = df["FCD_Gen_Test"].to_numpy()
    fmem = df["f_mem"].to_numpy()
    floor = float(df["FCD_Train_Test"].iloc[0])
    tau_gen = _log_interp_crossing(tau, fcd, floor * (1 + rel_tol), descending=True)
    tau_mem = _log_interp_crossing(tau, fmem, fmem_thresh, descending=False)
    return tau_gen, tau_mem


# ─────────────────────────────────────────────────────────────────────────────
# Window at an alternative ratio-test threshold kappa (memorized iff rho < kappa)
#
# tau_gen depends only on FCD(Gen,Test) and is kappa-independent; only tau_mem
# moves with kappa because f_mem is the fraction of samples with rho < kappa.
#
# f_mem(tau; kappa) source, at the SAME 256-D beamspace + nearest-neighbour test:
#   * kappa = 1/3  -> the canonical WINDOW_CSV (all N)
#   * kappa = 1/4  -> DDIM multi-k CSV for N in {200,1000}, and a locally
#                     computed CSV (compute_fmem_kappa_N500.py) for N = 500.
# ─────────────────────────────────────────────────────────────────────────────
MULTI_K_CSV = (PROJECT / "DDIM_Evaluation" / "dataset_size_effect" /
               "results" / "dsize_fmem_multi_k.csv")
EXTRA_KAPPA_CSV = RESULTS_DIR / "fmem_kappa_N500.csv"

KAPPA_MAIN = 1.0 / 3.0


def _fmem_series_kappa(n: int, kappa: float, min_tau: int = 1):
    """Return (tau, f_mem) arrays for dataset size N at ratio threshold kappa."""
    if abs(kappa - KAPPA_MAIN) < 1e-6:
        df = pd.read_csv(WINDOW_CSV)
        df = df[(df["N"] == n) & (df["tau"] >= min_tau)].sort_values("tau")
        return df["tau"].to_numpy(), df["f_mem"].to_numpy()

    frames = []
    if MULTI_K_CSV.exists():
        frames.append(pd.read_csv(MULTI_K_CSV))
    if EXTRA_KAPPA_CSV.exists():
        frames.append(pd.read_csv(EXTRA_KAPPA_CSV))
    if not frames:
        raise FileNotFoundError(
            f"No multi-kappa f_mem source found for kappa={kappa:.4f} "
            f"(looked for {MULTI_K_CSV} and {EXTRA_KAPPA_CSV}).")
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["N"] == n) & (df["tau"] >= min_tau) &
            (np.isclose(df["k"], kappa, atol=1e-6))].sort_values("tau")
    if df.empty:
        raise ValueError(
            f"No f_mem rows for N={n}, kappa={kappa:.4f}. For N=500 run "
            f"shared_reference/compute_fmem_kappa_N500.py first.")
    return df["tau"].to_numpy(), df["f_mem"].to_numpy()


def load_window_kappa(n: int, fmem_thresh: float = 0.1, kappa: float = KAPPA_MAIN,
                      rel_tol: float = 0.5, min_tau: int = 1):
    """(tau_gen, tau_mem) at ratio-test threshold kappa; tau_gen is kappa-free."""
    df = pd.read_csv(WINDOW_CSV)
    df = df[(df["N"] == n) & (df["tau"] >= min_tau)].sort_values("tau")
    if df.empty:
        raise ValueError(f"No window data for N={n} in {WINDOW_CSV}")
    tau_f = df["tau"].to_numpy()
    fcd = df["FCD_Gen_Test"].to_numpy()
    floor = float(df["FCD_Train_Test"].iloc[0])
    tau_gen = _log_interp_crossing(tau_f, fcd, floor * (1 + rel_tol), descending=True)

    tau_m, fmem = _fmem_series_kappa(n, kappa, min_tau=min_tau)
    tau_mem = _log_interp_crossing(tau_m, fmem, fmem_thresh, descending=False)
    return tau_gen, tau_mem


def load_checkpoint_metrics(n: int, tau: int):
    """FCD(Gen,Test) and f_mem for a single (N, tau) checkpoint."""
    df = pd.read_csv(WINDOW_CSV)
    row = df[(df["N"] == n) & (df["tau"] == tau)]
    if row.empty:
        return float("nan"), float("nan")
    return float(row["FCD_Gen_Test"].iloc[0]), float(row["f_mem"].iloc[0])


def classify_regime(tau: float, tau_gen: float, tau_mem: float) -> str:
    if not np.isfinite(tau_gen) or not np.isfinite(tau_mem):
        return "unknown"
    if tau < tau_gen:
        return "pre-window"
    if tau < tau_mem:
        return "in-window"
    return "post-window"


def available_generated_taus(n: int):
    """τ values with a pre-generated EMA channel file for this N."""
    g = ema_dir(n) / "generated_ema"
    taus = []
    for p in g.glob("gen_ema_tau*_seed0.npz"):
        s = p.name[len("gen_ema_tau"):-len("_seed0.npz")]
        if s.isdigit():
            taus.append(int(s))
    return sorted(taus)


def select_checkpoints(n: int, tau_gen: float, tau_mem: float,
                       n_pre: int = 2, n_in: int = 3, n_post: int = 3):
    """Pick τ checkpoints spanning pre-, in- and post-window regimes.

    Only τ with an existing generated_ema file (and τ>0) are eligible. Picks are
    spread ~log-uniformly inside each regime bucket.
    """
    taus = [t for t in available_generated_taus(n) if t > 0]
    pre = [t for t in taus if t < tau_gen]
    inw = [t for t in taus if tau_gen <= t < tau_mem]
    post = [t for t in taus if t >= tau_mem]

    def spread(bucket, k):
        if len(bucket) <= k:
            return bucket
        idx = np.unique(np.round(np.linspace(0, len(bucket) - 1, k)).astype(int))
        return [bucket[i] for i in idx]

    chosen = spread(pre, n_pre) + spread(inw, n_in) + spread(post, n_post)
    return sorted(set(chosen))


# ─────────────────────────────────────────────────────────────────────────────
# Reference set D_N  (identity-locked to the DDIM generator)
# ─────────────────────────────────────────────────────────────────────────────
def build_reference(n: int, H_los, H_nlos, il, inl, verify: bool = True):
    """Return (H_ref antenna (N,Nr,Nt), los_sel, nlos_sel).

    D_N = first N/2 shuffled LoS + first N/2 shuffled NLoS, which reproduces the
    generator's train.npy exactly. Asserts equality when ``verify``.
    """
    nh = n // 2
    los_sel = il[:nh]
    nlos_sel = inl[:nh]
    H_ref = np.concatenate([H_los[los_sel], H_nlos[nlos_sel]], axis=0)

    if verify:
        train_npy = ema_dir(n) / "train.npy"
        if not train_npy.exists():
            raise FileNotFoundError(
                f"DDIM train.npy missing for N={n}: {train_npy}. Cannot verify "
                f"shared-reference identity I_DDIM == I_CRNet.")
        tr = np.load(train_npy)
        H_ddim = (tr[:, 0] + 1j * tr[:, 1]).astype(np.complex64)
        if H_ddim.shape != H_ref.shape or np.abs(H_ddim - H_ref).max() > 1e-6:
            raise RuntimeError(
                f"SHARED-REFERENCE VIOLATION for N={n}: reconstructed D_N does "
                f"not match the DDIM generator's train.npy. Aborting.")
    return H_ref, los_sel, nlos_sel


# ─────────────────────────────────────────────────────────────────────────────
# Full split (reference pools + shared leakage-safe val/test)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SharedSplit:
    H_los: np.ndarray
    H_nlos: np.ndarray
    c_los: np.ndarray
    c_nlos: np.ndarray
    il: np.ndarray
    inl: np.ndarray
    ref_keys_union: set
    val_idx_los: np.ndarray
    val_idx_nlos: np.ndarray
    test_idx_los: np.ndarray
    test_idx_nlos: np.ndarray
    meta: dict = field(default_factory=dict)

    def reference(self, n: int, verify: bool = True):
        """Antenna-domain D_N (N,Nr,Nt) with identity verification."""
        H_ref, los_sel, nlos_sel = build_reference(
            n, self.H_los, self.H_nlos, self.il, self.inl, verify=verify)
        return H_ref, los_sel, nlos_sel

    @property
    def H_val(self):
        return np.concatenate(
            [self.H_los[self.val_idx_los], self.H_nlos[self.val_idx_nlos]], axis=0)

    @property
    def H_test(self):
        return np.concatenate(
            [self.H_los[self.test_idx_los], self.H_nlos[self.test_idx_nlos]], axis=0)


def build_shared_split(reference_sizes=(200, 1000), benchmark_n=5000,
                       n_val_per_side=500, n_test_per_side=1000,
                       eval_tail_per_side=1500, seed=42, ref_n_indices=1000):
    """Build the reference pools and one shared, leakage-safe val/test split.

    The val/test channels are taken from the *tail* of the shared per-side
    shuffle (indices far beyond the largest training prefix, N/2 ≤ 2500 for
    D_5000) and then any channel whose (x,y,z) coincides with a reference-set
    coordinate is dropped, guaranteeing disjointness at both the channel and the
    coordinate level. The same val/test sets are reused for every configuration.
    """
    H_los, c_los, H_nlos, c_nlos = load_or_cache_raw()
    il, inl = _shared_indices(ref_n_indices)

    keys_los = coord_keys(c_los)
    keys_nlos = coord_keys(c_nlos)

    # Union of all reference coordinates (incl. the 5000-real benchmark).
    ref_keys = set()
    for n in tuple(reference_sizes) + (benchmark_n,):
        nh = n // 2
        ref_keys |= set(keys_los[il[:nh]]) | set(keys_nlos[inl[:nh]])

    # Candidate eval channels: tail of each side, minus reference-coord collisions.
    tail_los = il[-eval_tail_per_side:]
    tail_nlos = inl[-eval_tail_per_side:]
    tail_los = np.array([i for i in tail_los if keys_los[i] not in ref_keys])
    tail_nlos = np.array([i for i in tail_nlos if keys_nlos[i] not in ref_keys])

    need = n_val_per_side + n_test_per_side
    if len(tail_los) < need or len(tail_nlos) < need:
        raise ValueError(
            f"Not enough leakage-safe eval channels: LoS {len(tail_los)}, "
            f"NLoS {len(tail_nlos)}, need {need} each. Increase eval_tail_per_side.")

    rng = np.random.default_rng(seed)
    tl = rng.permutation(tail_los)
    tn = rng.permutation(tail_nlos)
    val_idx_los, test_idx_los = tl[:n_val_per_side], tl[n_val_per_side:need]
    val_idx_nlos, test_idx_nlos = tn[:n_val_per_side], tn[n_val_per_side:need]

    split = SharedSplit(
        H_los=H_los, H_nlos=H_nlos, c_los=c_los, c_nlos=c_nlos, il=il, inl=inl,
        ref_keys_union=ref_keys,
        val_idx_los=val_idx_los, val_idx_nlos=val_idx_nlos,
        test_idx_los=test_idx_los, test_idx_nlos=test_idx_nlos,
        meta={"seed": seed, "n_val": 2 * n_val_per_side,
              "n_test": 2 * n_test_per_side,
              "reference_sizes": list(reference_sizes),
              "benchmark_n": benchmark_n},
    )

    # Leakage sanity: eval coords must be disjoint from every reference coord.
    val_keys = set(keys_los[val_idx_los]) | set(keys_nlos[val_idx_nlos])
    test_keys = set(keys_los[test_idx_los]) | set(keys_nlos[test_idx_nlos])
    assert not (val_keys & ref_keys), "Validation set leaks reference coordinates!"
    assert not (test_keys & ref_keys), "Test set leaks reference coordinates!"
    assert not (val_keys & test_keys), "Validation/test overlap!"
    return split


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic pool  (pre-generated EMA channels)
# ─────────────────────────────────────────────────────────────────────────────
def load_synthetic(n: int, tau: int, count: int, gen_seed: int = 0):
    """Return `count` synthetic channels (antenna-domain-equivalent tensors are
    built later). Here we return the raw beamspace real/imag array (count,2,Nr,Nt)
    subsampled deterministically from the pre-generated EMA pool.
    """
    p = ema_dir(n) / "generated_ema" / f"gen_ema_tau{tau}_seed{gen_seed}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Generated EMA file not found: {p}")
    ch = np.load(p)["channels"]                    # (M, 2, Nr, Nt) beamspace
    if len(ch) < count:
        raise ValueError(f"Need {count} synthetic channels but only {len(ch)} "
                         f"available in {p.name}")
    rng = np.random.default_rng(10_000 + n + tau)  # deterministic per (N,tau)
    idx = rng.permutation(len(ch))[:count]
    return ch[idx], str(p)
