"""
run_task09.py — Downstream CSI Compression with DDIM-Generated Augmentation
============================================================================
Task 09 orchestrator. Trains fresh CRNet CSI-compression models under a matrix
of conditions to test whether DDIM synthetic channels from the generalization
window help downstream compression more than those from the memorization window.

Experiment groups (all evaluated on the same held-out 1000-channel test set):
  A. Benchmark      : 5000 real (2500 LoS + 2500 NLoS)
  B. Real Only      : N_real in {200, 500, 1000, 5000}
  C. Aug DDIM_N=X   : N_real in {200, 500, 1000} + synthetic to fill 5000,
                      for X in {100, 200, 1000} and tau in {1000, 10000, 100000}

Leakage safety:
  - The test set is drawn ONLY from channels never seen by ANY DDIM generator
    (coordinate matching against DDIM train_coords). It is disjoint from all
    training pools. Overlaps are quantified in data_overlap_report.csv.

Usage:
    conda activate Mem_Gen
    cd Downstream_Tasks/CSI_Compression
    python run_task09.py                 # full run (cached results are skipped)
    python run_task09.py --debug         # quick smoke test (few epochs, subset)
    python run_task09.py --epochs 300    # override epochs
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from crnet_core import to_tensor, train_crnet
from data_utils import (
    load_sionna_raw, coord_keys, load_ddim_train_coords, load_synthetic_pool,
    build_master_split, draw_real_subset,
)

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DATASET_DIR = PROJECT / "dataset"
LOS_NPZ = DATASET_DIR / "Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"
NLOS_NPZ = DATASET_DIR / "Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz"
LOGS_BASE = PROJECT / "Code" / "DDIM_FMM" / "logs"
W1_CSV = PROJECT / "DDIM_Evaluation" / "results" / "w1_quality_vs_tau" / "wasserstein_quality_vs_tau.csv"
FMEM_CSV = PROJECT / "DDIM_Evaluation" / "results" / "fmem_vs_tau" / "fmem_vs_tau.csv"

RESULTS_DIR = HERE / "results"
CACHE_DIR = RESULTS_DIR / "cache"
LOGS_DIR = RESULTS_DIR / "logs"
CKPT_DIR = LOGS_DIR / "checkpoints"

# ── Experiment configuration ─────────────────────────────────────────────────
DDIM_SIZES = [100, 200, 1000]
TAUS = [1000, 10000, 100000]
DOWNSTREAM_REAL_SIZES = [200, 500, 1000, 5000]
AUG_REAL_SIZES = [200, 500, 1000]      # sizes where synthetic is actually added
TOTAL_TRAIN = 5000
BENCH_REAL = 5000
N_TEST_PER_SIDE = 500                   # 500 LoS + 500 NLoS = 1000 test
VAL_FRAC = 0.10
SEED = 42

# CRNet hyper-parameters
NR, NT = 4, 32
REDUCTION = 4
EPOCHS = 500
BATCH_SIZE = 512
LR = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# DATA CACHING
# ─────────────────────────────────────────────────────────────────────────────
def load_or_cache_raw():
    """Load raw Sionna LoS/NLoS (antenna domain + coords), caching to .npy."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "H_los": CACHE_DIR / "H_los.npy", "coords_los": CACHE_DIR / "coords_los.npy",
        "H_nlos": CACHE_DIR / "H_nlos.npy", "coords_nlos": CACHE_DIR / "coords_nlos.npy",
    }
    if all(f.exists() for f in files.values()):
        print("Loading cached raw channel arrays ...")
        return (np.load(files["H_los"]), np.load(files["coords_los"]),
                np.load(files["H_nlos"]), np.load(files["coords_nlos"]))

    print("Loading raw Sionna LoS data (this may take a moment) ...")
    H_los, coords_los = load_sionna_raw(str(LOS_NPZ))
    print(f"  LoS: {H_los.shape[0]} channels")
    print("Loading raw Sionna NLoS data ...")
    H_nlos, coords_nlos = load_sionna_raw(str(NLOS_NPZ))
    print(f"  NLoS: {H_nlos.shape[0]} channels")

    np.save(files["H_los"], H_los)
    np.save(files["coords_los"], coords_los)
    np.save(files["H_nlos"], H_nlos)
    np.save(files["coords_nlos"], coords_nlos)
    return H_los, coords_los, H_nlos, coords_nlos


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE-WINDOW LABELS (from Task 07 / Task 08 CSVs)
# ─────────────────────────────────────────────────────────────────────────────
def load_source_metrics():
    """Return dict {(ddim_n, tau): {'W1': ..., 'fmem': ..., 'label': ...}}."""
    import pandas as pd
    metrics = {}
    w1 = pd.read_csv(W1_CSV) if W1_CSV.exists() else None
    fm = pd.read_csv(FMEM_CSV) if FMEM_CSV.exists() else None

    for ddim_n in DDIM_SIZES:
        for tau in TAUS:
            w1_val, fmem_val = np.nan, np.nan
            if w1 is not None:
                row = w1[(w1["N"] == ddim_n) & (w1["target_tau"] == tau)]
                if not row.empty:
                    w1_val = float(row["W1_Gen_Test"].iloc[0])
            if fm is not None:
                row = fm[(fm["N"] == ddim_n) & (fm["target_tau"] == tau)]
                if not row.empty:
                    fmem_val = float(row["f_mem_k_1_3"].iloc[0])
            metrics[(ddim_n, tau)] = {
                "W1": w1_val, "fmem": fmem_val,
                "label": classify_window(w1_val, fmem_val, w1),
            }
    return metrics


def classify_window(w1_val, fmem_val, w1_df):
    """Heuristic labeling: early/low-quality, generalization, or memorization."""
    if np.isnan(w1_val):
        return "unknown"
    # High W1 relative to the observed range → early / low-quality
    if w1_df is not None:
        w1_median = float(w1_df["W1_Gen_Test"].median())
        if w1_val > 1.5 * w1_median:
            return "early/low-quality"
    if not np.isnan(fmem_val):
        if fmem_val >= 0.5:
            return "memorization-window"
        return "generalization-window"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAP REPORT
# ─────────────────────────────────────────────────────────────────────────────
def write_overlap_report(split, H_los, H_nlos, ddim_coords_by_n):
    """Quantify coordinate overlap between downstream splits and DDIM training."""
    keys_los = split["keys_los"]
    keys_nlos = split["keys_nlos"]

    test_keys = set(keys_los[split["test_los_idx"]]) | set(keys_nlos[split["test_nlos_idx"]])

    rows = []
    for ddim_n, dcoords in ddim_coords_by_n.items():
        ddim_keys = set(coord_keys(dcoords))
        for n_real in DOWNSTREAM_REAL_SIZES:
            # Reconstruct the exact real subset used for this n_real
            _, los_sel, nlos_sel = draw_real_subset(
                H_los, H_nlos, split["pool_los_idx"], split["pool_nlos_idx"],
                n_real, seed=1000 + n_real)
            train_keys = set(keys_los[los_sel]) | set(keys_nlos[nlos_sel])

            rows.append({
                "experiment_name": f"aug_DDIMN{ddim_n}_Nreal{n_real}",
                "N_downstream_real": n_real,
                "DDIM_train_N": ddim_n,
                "tau": "all",
                "num_overlap_downstream_train_with_DDIM_train": len(train_keys & ddim_keys),
                "num_overlap_downstream_test_with_DDIM_train": len(test_keys & ddim_keys),
                "num_overlap_downstream_test_with_downstream_train": len(test_keys & train_keys),
            })

    out = RESULTS_DIR / "data_overlap_report.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nOverlap report written: {out}")
    # Sanity: test overlaps should all be zero
    bad = [r for r in rows if r["num_overlap_downstream_test_with_DDIM_train"] > 0
           or r["num_overlap_downstream_test_with_downstream_train"] > 0]
    if bad:
        print("  WARNING: non-zero test overlap detected!")
    else:
        print("  OK: test set is disjoint from all DDIM-train and downstream-train sets.")
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def split_train_val(H_all, seed):
    """Shuffle a combined training array and split into train/val tensors."""
    rng = np.random.default_rng(seed)
    H_all = H_all[rng.permutation(len(H_all))]
    n_val = max(1, int(len(H_all) * VAL_FRAC))
    return to_tensor(H_all[n_val:]), to_tensor(H_all[:n_val])


def append_result(csv_path, row, columns):
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Task 09 downstream CSI compression")
    parser.add_argument("--debug", action="store_true",
                        help="Quick smoke test: few epochs, subset of experiments")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--no_skip", dest="skip_done", action="store_false",
                        help="Retrain even if cached logs exist")
    parser.set_defaults(skip_done=True)
    args = parser.parse_args()

    epochs = args.epochs if args.epochs else (20 if args.debug else EPOCHS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for d in (RESULTS_DIR, LOGS_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Task 09 — Downstream CSI Compression with DDIM Augmentation")
    print("=" * 70)
    print(f"  Device        : {device}")
    print(f"  Epochs        : {epochs}")
    print(f"  Total train   : {TOTAL_TRAIN}")
    print(f"  Test set      : {2*N_TEST_PER_SIDE} (leakage-safe, DDIM-unseen)")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    H_los, coords_los, H_nlos, coords_nlos = load_or_cache_raw()

    # DDIM-seen coordinate keys (union across all DDIM generators)
    ddim_coords_by_n = {n: load_ddim_train_coords(str(LOGS_BASE), n) for n in DDIM_SIZES}
    ddim_seen_keys = set()
    for n, c in ddim_coords_by_n.items():
        ddim_seen_keys |= set(coord_keys(c))
    print(f"\nDDIM-seen unique coordinates (union): {len(ddim_seen_keys)}")

    # ── Master leakage-safe split ─────────────────────────────────────────
    split = build_master_split(H_los, coords_los, H_nlos, coords_nlos,
                               ddim_seen_keys, n_test_per_side=N_TEST_PER_SIDE, seed=SEED)
    X_test = to_tensor(split["H_test"])
    print(f"Test set built: {len(X_test)} channels (DDIM-unseen)")

    # ── Overlap report ────────────────────────────────────────────────────
    write_overlap_report(split, H_los, H_nlos, ddim_coords_by_n)

    # ── Source-window metrics ─────────────────────────────────────────────
    src_metrics = load_source_metrics()

    # ── Results CSV setup ─────────────────────────────────────────────────
    results_csv = RESULTS_DIR / "downstream_csi_compression_results.csv"
    if not args.skip_done and results_csv.exists():
        results_csv.unlink()
    columns = [
        "experiment_name", "group", "N_downstream_real", "DDIM_train_N", "tau",
        "num_real_train", "num_synthetic_train", "total_train", "augmentation_ratio",
        "synthetic_file", "model_checkpoint", "best_epoch",
        "test_NMSE_dB", "validation_NMSE_dB",
        "W1_Gen_Test_source", "f_mem_source", "source_window_label", "notes",
    ]
    # Track which experiments are already in the CSV
    done_experiments = set()
    if results_csv.exists() and args.skip_done:
        with open(results_csv) as f:
            for r in csv.DictReader(f):
                done_experiments.add(r["experiment_name"])

    def run_experiment(name, group, n_real, ddim_n, tau, H_train_all, n_syn, syn_file):
        if name in done_experiments:
            print(f"  [SKIP-CSV] {name}")
            return
        X_tr, X_vl = split_train_val(H_train_all, seed=SEED)
        log_path = LOGS_DIR / f"{name}.json"
        ckpt_path = CKPT_DIR / f"{name}.pt"
        print(f"\n{'-'*66}\n  {name}  (real={n_real}, synth={n_syn}, total={len(H_train_all)})")
        meta = train_crnet(
            X_tr, X_vl, X_test, tag=name, log_path=log_path, ckpt_path=ckpt_path,
            nr=NR, nt=NT, reduction=REDUCTION, epochs=epochs, batch_size=BATCH_SIZE,
            lr=LR, seed=SEED, device=device, skip_done=args.skip_done,
            extra_meta={"group": group, "n_real": n_real},
        )
        sm = src_metrics.get((ddim_n, tau), {}) if ddim_n else {}
        row = {
            "experiment_name": name, "group": group, "N_downstream_real": n_real,
            "DDIM_train_N": ddim_n if ddim_n else "none",
            "tau": tau if tau else "none",
            "num_real_train": n_real, "num_synthetic_train": n_syn,
            "total_train": len(H_train_all),
            "augmentation_ratio": round(n_syn / len(H_train_all), 4) if len(H_train_all) else 0,
            "synthetic_file": syn_file if syn_file else "none",
            "model_checkpoint": str(ckpt_path),
            "best_epoch": meta["best_epoch"],
            "test_NMSE_dB": round(meta["test_nmse_db"], 4),
            "validation_NMSE_dB": round(meta["val_nmse_db"], 4),
            "W1_Gen_Test_source": sm.get("W1", "none"),
            "f_mem_source": sm.get("fmem", "none"),
            "source_window_label": sm.get("label", "none"),
            "notes": "",
        }
        append_result(results_csv, row, columns)
        done_experiments.add(name)

    # ── Group A — Benchmark (real 5000) ───────────────────────────────────
    H_bench, _, _ = draw_real_subset(H_los, H_nlos, split["pool_los_idx"],
                                     split["pool_nlos_idx"], BENCH_REAL, seed=1000 + BENCH_REAL)
    run_experiment("benchmark_real5000", "Benchmark", BENCH_REAL, None, None,
                   H_bench, 0, None)

    # ── Group B — Real Only ───────────────────────────────────────────────
    real_sizes = DOWNSTREAM_REAL_SIZES if not args.debug else [200, 1000]
    for n_real in real_sizes:
        H_real, _, _ = draw_real_subset(H_los, H_nlos, split["pool_los_idx"],
                                        split["pool_nlos_idx"], n_real, seed=1000 + n_real)
        run_experiment(f"realonly_Nreal{n_real}", "Real Only", n_real, None, None,
                       H_real, 0, None)

    # ── Groups C & D — Augmentation ───────────────────────────────────────
    ddim_sizes = DDIM_SIZES if not args.debug else [200]
    taus = TAUS if not args.debug else [100000]
    aug_sizes = AUG_REAL_SIZES if not args.debug else [200]
    for ddim_n in ddim_sizes:
        for tau in taus:
            syn_pool = load_synthetic_pool(str(LOGS_BASE), ddim_n, tau)  # antenna domain
            syn_file = str(LOGS_BASE / f"DDIM_tau_{ddim_n}_incremental" /
                           "generated_tau" / f"generated_tau_{tau}.npz")
            group = f"Aug DDIM_N={ddim_n}"
            for n_real in aug_sizes:
                n_syn = TOTAL_TRAIN - n_real
                # Same real subset as real-only (fair comparison)
                H_real, _, _ = draw_real_subset(
                    H_los, H_nlos, split["pool_los_idx"], split["pool_nlos_idx"],
                    n_real, seed=1000 + n_real)
                # Draw synthetic (mixed LoS+NLoS pool)
                rng = np.random.default_rng(7000 + ddim_n + tau + n_real)
                syn_idx = rng.permutation(len(syn_pool))[:n_syn]
                H_syn = syn_pool[syn_idx]
                H_all = np.concatenate([H_real, H_syn], axis=0)
                name = f"aug_DDIMN{ddim_n}_tau{tau}_Nreal{n_real}"
                run_experiment(name, group, n_real, ddim_n, tau, H_all, n_syn, syn_file)

    print(f"\n{'='*70}\nDONE. Results CSV: {results_csv}")
    print(f"Run  python plot_task09.py  to generate figures.")
    print("=" * 70)


if __name__ == "__main__":
    main()
