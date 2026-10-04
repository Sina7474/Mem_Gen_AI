"""
evaluate_all_sizes.py — Effective-rank evaluation for ALL training set sizes
=============================================================================
For each training size N, this script computes:

  1. Effective rank of the EXACT N training channels (spatial → beamspace)
  2. Effective rank of the 5000 generated channels
  3. W1(training data, generated data) — measures reproduction fidelity

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python evaluate_all_sizes.py
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add the Effective_Rank_Github path to import the core library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cDDIM_cFMM_Repo', 'Effective_Rank_Github'))
from effective_rank_core import (
    compute_effective_rank,
    summarize_effective_rank,
    compute_wasserstein,
    cdf_xy,
    normalize_max_abs,
    to_beamspace,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000, 8000]
LOGS_BASE = "../Code/DDIM_FMM/logs"
USE_POWER_SV = True

# ── Mode selection ────────────────────────────────────────────────────────────
# Set INCREMENTAL = True  for the incremental (nested subset) experiment
# Set INCREMENTAL = False for the independent (separate shuffle) experiment
INCREMENTAL = True

if INCREMENTAL:
    OUTPUT_DIR = Path("results/all_sizes_incremental")
else:
    OUTPUT_DIR = Path("results/all_sizes_comparison")

# UPA antenna dimensions (must match training)
ANTENNA = dict(Nrx_x=2, Nrx_y=2, Ntx_x=4, Ntx_y=8)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         HELPERS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_log_dir(n_train):
    suffix = "_incremental" if INCREMENTAL else ""
    return os.path.join(LOGS_BASE, f"DDIM_unconditional_3.5GHz_LoS+NLoS_0.0_{n_train}_nT200{suffix}")


def load_generated_beamspace(path):
    """Load generated channels from .npz: (N, 2, Nr, Nt) → complex (N, Nr, Nt)."""
    arr = np.load(path)['channels']
    return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)


def load_training_data_as_beamspace(log_dir):
    """
    Load train.npy (spatial domain) → convert to beamspace → complex (N, Nr, Nt).
    train.npy shape: (N, 2, 4, 32) where axis 1 = [real, imag], spatial domain.
    """
    train_path = os.path.join(log_dir, 'train.npy')
    arr = np.load(train_path)  # (N, 2, 4, 32), spatial domain
    H_spatial = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)  # (N, 4, 32)
    H_beam = to_beamspace(H_spatial, **ANTENNA)  # (N, 4, 32) in beamspace
    return H_beam


def _safe_key(name):
    return name.replace(" ", "_").replace("/", "_").replace(".", "p")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Effective Rank Evaluation — All Training Set Sizes")
    print("  W1(training data, generated data)")
    print("=" * 70)

    # ── 1. Check availability ────────────────────────────────────────────────
    available_sizes = []
    for size in TRAIN_SIZES:
        gen_path = os.path.join(get_log_dir(size), "generated_channels", "generated_beamspace.npz")
        train_path = os.path.join(get_log_dir(size), "train.npy")
        if os.path.exists(gen_path) and os.path.exists(train_path):
            available_sizes.append(size)
            print(f"  [✓] {size} samples")
        else:
            print(f"  [✗] {size} samples — missing files")

    if not available_sizes:
        print("\nERROR: No data found. Run training and inference first.")
        sys.exit(1)

    print(f"\nAvailable sizes: {available_sizes}")

    # ── 2. For each size: compute erank for training data & generated data ───
    print("\n" + "=" * 70)
    print("Computing effective rank per training size")
    print("=" * 70)

    erank_all = {}
    sv_norm_all = {}
    w1_distances = {}
    summary_rows = []

    for size in available_sizes:
        log_dir = get_log_dir(size)

        # --- Load and process TRAINING data (spatial → beamspace) ---
        train_name = f"Train {size}"
        print(f"\n  [{train_name}]")
        H_train = load_training_data_as_beamspace(log_dir)
        H_train = normalize_max_abs(H_train)
        print(f"    Training data shape: {H_train.shape}")

        er_train, sv_train = compute_effective_rank(H_train, use_power=USE_POWER_SV)
        erank_all[train_name] = er_train
        sv_norm_all[train_name] = sv_train
        stats_t = summarize_effective_rank(er_train)
        summary_rows.append({"dataset": train_name, **stats_t})
        print(f"    erank: mean={stats_t['mean']:.4f}  median={stats_t['median']:.4f}  std={stats_t['std']:.4f}")

        # --- Load GENERATED data (already beamspace) ---
        gen_name = f"DDIM {size} gen"
        gen_path = os.path.join(log_dir, "generated_channels", "generated_beamspace.npz")
        print(f"  [{gen_name}]")
        H_gen = load_generated_beamspace(gen_path)
        H_gen = normalize_max_abs(H_gen)
        print(f"    Generated shape: {H_gen.shape}")

        er_gen, sv_gen = compute_effective_rank(H_gen, use_power=USE_POWER_SV)
        erank_all[gen_name] = er_gen
        sv_norm_all[gen_name] = sv_gen
        stats_g = summarize_effective_rank(er_gen)
        summary_rows.append({"dataset": gen_name, **stats_g})
        print(f"    erank: mean={stats_g['mean']:.4f}  median={stats_g['median']:.4f}  std={stats_g['std']:.4f}")

        # --- W1(training data, generated) ---
        w1 = compute_wasserstein(er_train, er_gen, n_select=None)
        w1_distances[size] = w1
        print(f"    W1(Train {size}, Generated) = {w1:.6f}")

    # ── 3. Save effective-rank values ────────────────────────────────────────
    er_path = OUTPUT_DIR / "effective_rank_values.npz"
    np.savez_compressed(er_path, **{_safe_key(k): v for k, v in erank_all.items()})
    print(f"\nSaved: {er_path}")

    # ── 4. Summary CSV ───────────────────────────────────────────────────────
    cols = ["dataset", "n", "mean", "std", "min", "p05", "p25",
            "median", "p75", "p95", "max"]
    df = pd.DataFrame(summary_rows)[cols]
    csv_path = OUTPUT_DIR / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    print("\n" + df.to_string(index=False))

    # ── 5. Save Wasserstein distances ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Wasserstein Distance Summary — W1(Train Data, Generated)")
    print("=" * 70)

    w1_rows = []
    for size in available_sizes:
        print(f"  {size:5d} samples:  W1 = {w1_distances[size]:.6f}")
        w1_rows.append({
            "training_size": size,
            "W1_train_vs_gen": w1_distances[size],
        })

    w1_df = pd.DataFrame(w1_rows)
    w1_csv = OUTPUT_DIR / "wasserstein_distances.csv"
    w1_df.to_csv(w1_csv, index=False)
    print(f"\nSaved: {w1_csv}")

    w1_npz = {_safe_key(f"DDIM_{s}_gen"): w1_distances[s] for s in available_sizes}
    np.savez_compressed(OUTPUT_DIR / "wasserstein_data.npz", **w1_npz)

    # ── 6. CDF data ──────────────────────────────────────────────────────────
    cdf_dict = {}
    for name, er in erank_all.items():
        x, y = cdf_xy(er)
        cdf_dict[f"{_safe_key(name)}_x"] = x
        cdf_dict[f"{_safe_key(name)}_y"] = y
    np.savez_compressed(OUTPUT_DIR / "cdf_data.npz", **cdf_dict)
    print(f"Saved: {OUTPUT_DIR / 'cdf_data.npz'}")

    # ── 7. SV spectrum ───────────────────────────────────────────────────────
    np.savez_compressed(OUTPUT_DIR / "sv_spectrum.npz", **{
        _safe_key(k): np.mean(v, axis=0) for k, v in sv_norm_all.items()
    })
    print(f"Saved: {OUTPUT_DIR / 'sv_spectrum.npz'}")

    print("\n** Evaluation complete for all available sizes! **")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()