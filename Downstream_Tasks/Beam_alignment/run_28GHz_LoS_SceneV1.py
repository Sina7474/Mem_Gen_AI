"""
run_28GHz_LoS_SceneV1.py — Beam alignment on the NEW 28 GHz SceneV1
====================================================================
Tests whether memorized DDIM generators hurt downstream beam alignment
on a MORE COMPLEX 28 GHz scene (SceneV1), where we hypothesize that
the memorization effect is more visible than in the original simpler scene.

Dataset: Final_Single_Scene_Channel_Sionna_V1_28GHz_LoS_UPA.npz (12 940 users)
DDIM logs: Code/DDIM_FMM/logs_ema_28GHz_LoS_SceneV1/

Curves plotted:
    1. MRT + MRC (upper bound)
    2. Genie-aided DFT beam pair
    3. Real Only (N)
    4. Real N + Gen 5000, tau=10000  (DDIM generalizing)
    5. Real N + Gen 5000, tau=100000 (DDIM memorized)

Usage:
    conda activate Mem_Gen
    cd Downstream_Tasks/Beam_alignment

    python run_28GHz_LoS_SceneV1.py --ddim_n 100 --force
    python run_28GHz_LoS_SceneV1.py --ddim_n 200 --force
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_utils_28GHz_LoS_SceneV1 import (
    load_sionna_raw_28ghz_scenev1, coord_keys, load_ddim_train_coords,
    load_synthetic_pool, synthetic_path, build_master_split,
    load_ddim_train_channels,
)
from beam_align_core import (
    make_system, prepare_training_channels, train_one, eval_snr, compute_baselines,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DATASET_DIR = PROJECT / "dataset"
DATASET_NPZ = DATASET_DIR / "Final_Single_Scene_Channel_Sionna_V1_28GHz_LoS_UPA.npz"
LOGS_BASE = PROJECT / "Code" / "DDIM_FMM" / "logs_ema_28GHz_LoS_SceneV1"

RESULTS_DIR = HERE / "28GHz_LoS_SceneV1" / "results"
CACHE_DIR = HERE / "28GHz_LoS_SceneV1" / "cache"
CKPT_DIR = RESULTS_DIR / "checkpoints"

# ── Configuration ─────────────────────────────────────────────────────────────
DDIM_N = 100
TAUS = [1000, 10000, 100000, 200000]
PLOT_TAUS = [10000, 100000]
GEN_SIZE = 5000
N_TEST = 5000
NUM_PROBING_BEAM_PAIRS = [2, 4, 8, 12, 16]
SEED = 42
GEN_FILE_SIZE = 5000

# DL-GF hyper-parameters
NEPOCH = 1000
BATCH_SIZE = 256
LR = 1e-3

SYS = make_system(tx_power_dBm=20, BW_MHz=100.0, noise_PSD_dB=-161.0,
                  measurement_gain=16.0)

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TAU_STYLE = {
    1000:   dict(color="#1f77b4", ls="--", lw=2.0, marker="D", ms=6),
    10000:  dict(color="#2ca02c", ls="-.", lw=2.0, marker="s", ms=6),
    100000: dict(color="#9467bd", ls="-",  lw=2.0, marker="v", ms=6),
    200000: dict(color="#d62728", ls="-",  lw=2.0, marker="^", ms=6),
}


def load_or_cache_raw():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f_H = CACHE_DIR / "H_28ghz_los_scenev1.npy"
    f_c = CACHE_DIR / "coords_28ghz_los_scenev1.npy"
    if f_H.exists() and f_c.exists():
        return np.load(f_H), np.load(f_c)
    H, coords = load_sionna_raw_28ghz_scenev1(str(DATASET_NPZ))
    np.save(f_H, H)
    np.save(f_c, coords)
    return H, coords


def run_one(name, h_train, frob_norm, h_test_raw, npb_list, args, device):
    """Train + evaluate one experiment across all npb values."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    h_prepped, frob_used = prepare_training_channels(h_train, seed=SEED,
                                                     frob_norm=frob_norm)
    out = {}
    for npb in npb_list:
        ckpt = CKPT_DIR / f"{name}_npb{npb}.pt"
        metric_file = CKPT_DIR / f"{name}_npb{npb}.json"
        if ckpt.exists() and metric_file.exists() and not args.force:
            m = json.loads(metric_file.read_text())
            print(f"  [cached] {name} npb={npb}: SNR={m['snr']:.2f} dB")
            out[npb] = m["snr"]
            continue
        print(f"  training {name} npb={npb} "
              f"(n_train={len(h_prepped)}, frob={frob_used}) ...")
        state, _ = train_one(h_prepped, npb, SYS, nepoch=NEPOCH,
                             batch_size=BATCH_SIZE, lr=LR, seed=SEED,
                             device=device)
        torch.save(state, ckpt)
        snr_db, rate = eval_snr(state, npb, h_test_raw, SYS,
                                frob_norm=frob_norm, eval_seed=0)
        metric_file.write_text(json.dumps({"snr": snr_db, "rate": rate}))
        print(f"    -> SNR={snr_db:.2f} dB, rate={rate:.3f} bps/Hz")
        out[npb] = snr_db
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=NEPOCH)
    ap.add_argument("--ddim_n", type=int, default=DDIM_N,
                    help="DDIM training-set size (= N_REAL). Default: 100")
    args = ap.parse_args()

    ddim_n = args.ddim_n
    n_real = ddim_n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"BEAM ALIGNMENT — 28 GHz LoS SceneV1 (NEW SCENE)")
    print(f"  Real {n_real} + Gen {GEN_SIZE}, DDIM N={ddim_n}")
    print("=" * 70)
    print(f"Device: {device}  |  epochs={args.epochs}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load raw data ─────────────────────────────────────────────────────
    H, coords = load_or_cache_raw()
    print(f"Scene: {H.shape[0]} real 28 GHz LoS SceneV1 users")

    # ── Leakage-safe split ────────────────────────────────────────────────
    ddim_coords = load_ddim_train_coords(str(LOGS_BASE), ddim_n)
    ddim_seen_keys = set(coord_keys(ddim_coords)) if len(ddim_coords) else set()
    print(f"DDIM N={ddim_n} seen channels excluded: {len(ddim_seen_keys)}")

    split = build_master_split(H, coords, ddim_seen_keys,
                               n_test=N_TEST, seed=SEED)
    H_test = split["H_test"]
    print(f"Test set: {H_test.shape[0]} channels (DDIM-unseen)")

    # ── Real training data = exact DDIM training channels ───────────────
    H_real = load_ddim_train_channels(str(LOGS_BASE), ddim_n, H)
    print(f"Real subset: {H_real.shape[0]} channels (= DDIM training set)")

    # ── Baselines ─────────────────────────────────────────────────────────
    baselines = compute_baselines(H_test, SYS)
    bl_mrt = baselines["MRT_MRC"]
    bl_genie = baselines["genie_DFT"]
    print(f"  MRT+MRC   = {bl_mrt:.2f} dB  (over {H_test.shape[0]} test channels)")
    print(f"  Genie DFT = {bl_genie:.2f} dB")

    # ── Real Only (Frobenius-normalized — same preprocessing as augmented) ─
    print(f"\n=== Real Only (N={n_real}) ===")
    snr_real_frob = run_one(f"realonly_frob_N{n_real}", H_real, frob_norm=True,
                            h_test_raw=H_test, npb_list=NUM_PROBING_BEAM_PAIRS,
                            args=args, device=device)

    # ── Real N + Gen 5000 @ each tau ──────────────────────────────────────
    snr_aug = {}
    for tau in TAUS:
        name = f"aug_real{n_real}_gen{GEN_SIZE}_tau{tau}"
        print(f"\n=== {name} ===")
        H_syn = load_synthetic_pool(str(LOGS_BASE), ddim_n, tau,
                                    n_generate=GEN_FILE_SIZE)
        if len(H_syn) > GEN_SIZE:
            rng = np.random.default_rng(SEED)
            H_syn = H_syn[rng.permutation(len(H_syn))[:GEN_SIZE]]
        H_combined = np.concatenate([H_real, H_syn], axis=0)
        print(f"  Combined: {H_real.shape[0]} real + {H_syn.shape[0]} gen "
              f"= {H_combined.shape[0]} total")
        snr_aug[tau] = run_one(name, H_combined, frob_norm=True,
                               h_test_raw=H_test, npb_list=NUM_PROBING_BEAM_PAIRS,
                               args=args, device=device)

    # ── Plot ──────────────────────────────────────────────────────────────
    npb = np.array(NUM_PROBING_BEAM_PAIRS)
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.axhline(bl_mrt, color="black", ls="-", lw=2.0, zorder=4,
               label="MRT + MRC (upper bound)")
    ax.axhline(bl_genie, color="black", ls="--", lw=2.0, zorder=4,
               label="Genie-aided DFT beam pair")

    real_frob_snrs = [snr_real_frob[k] for k in NUM_PROBING_BEAM_PAIRS]
    ax.plot(npb, real_frob_snrs, color="#8c564b", ls=":", lw=2.0, marker="o",
            ms=6, mfc="none", zorder=3,
            label=f"Real Only (N={n_real})")

    for tau in PLOT_TAUS:
        aug_snrs = [snr_aug[tau][k] for k in NUM_PROBING_BEAM_PAIRS]
        ax.plot(npb, aug_snrs, zorder=3, mfc="none",
                label=rf"Real {n_real} + Gen {GEN_SIZE}, $\tau$={tau}",
                **TAU_STYLE[tau])

    ax.set_xlabel(r"Number of probing beam pairs  $N_{\mathrm{probe}}$", fontsize=20)
    ax.set_ylabel("Average SNR (dB)", fontsize=20)
    ax.set_xticks(npb)
    ax.set_xticklabels([str(v) for v in npb], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(True, alpha=0.30, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", fontsize=14, frameon=True,
              edgecolor="grey", framealpha=0.92)

    fig.tight_layout()
    fig_dir = HERE / "28GHz_LoS_SceneV1" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = fig_dir / f"beam_alignment_N{ddim_n}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
