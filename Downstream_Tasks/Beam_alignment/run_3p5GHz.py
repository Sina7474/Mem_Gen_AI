"""
run_augmented_3p5GHz.py — Augmentation experiment: Real N + Gen 5000 @ 3.5 GHz
=================================================================================
Tests whether memorized DDIM generators hurt downstream beam alignment when
used as DATA AUGMENTATION on top of a fixed real dataset.

The 3.5 GHz scene contains BOTH LoS and NLoS users, making the channel much
higher-dimensional (effective rank ~5-10 for NLoS).  Memorized generators should
produce less diverse channels, clearly degrading downstream performance.

GENERATOR PROVENANCE (important)
--------------------------------
The synthetic pools are read from
    Code/DDIM_FMM/logs_ema/DDIM_tau_ema_<N>_bs<B>_incremental/generated_ema/
        gen_ema_tau<tau>_seed0.npz
which are the *exact* EMA sample pools that DDIM_Evaluation/dataset_size_effect
uses to compute f_mem and FCD.  A downstream curve at horizon tau therefore
corresponds one-to-one to the memorization value reported at that tau.

The earlier `logs/DDIM_tau_<N>_incremental/generated_tau/*_10k.npz` pools must
NOT be used: generate_10k.py loaded the checkpoints with strict=False against a
dict whose keys are ('model_state_dict', 'optimizer_state_dict', ...), so no
weights were ever restored and those files are the output of a randomly
initialized U-Net.

Scale handling (mirrors DLGF-main/DLGF_train_Sina.py)
-----------------------------------------------------
The DDIM emits per-sample max-magnitude-normalized BEAMSPACE channels; they are
mapped back to the antenna domain with the same UPA Kronecker DFT codebook used
to build the training data.  Real Sionna channels are raw (path-loss included,
||H||_F ~ 7e-4) while synthetic ones land at ||H||_F ~ 1e-1, so whenever the two
are mixed every training channel gets per-sample Frobenius normalization.  The
beamforming gain at evaluation is always computed on the RAW test channels.
Because that normalization also changes the effective measurement SNR, the
real-only reference is reported in BOTH modes so the augmentation gain is not
confounded with the change of input scale.

Curves plotted:
    1. MRT + MRC (upper bound, on the same test set as every other curve)
    2. Real Only (N real, physical scale)
    3. Real Only (N real, Frobenius-normalized — matched to the augmented runs)
    4. Real N + Gen 5000, per tau

Usage:
    conda activate Mem_Gen
    cd Downstream_Tasks/Beam_alignment
    python run_augmented_3p5GHz.py
"""

import argparse
import csv
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

from data_utils import (
    load_sionna_raw, coord_keys, load_ddim_train_coords,
    load_ddim_train_coords_ema, load_synthetic_pool_ema, ema_synthetic_path,
    load_ddim_train_channels_ema,
    build_master_split, draw_real_subset,
)
from beam_align_core import (
    make_system, prepare_training_channels, train_one, eval_snr, compute_baselines,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
DATASET_DIR = PROJECT / "dataset"
# EMA runs — the generators the f_mem / FCD curves were measured on.
LOGS_EMA = PROJECT / "Code" / "DDIM_FMM" / "logs_ema"
# Legacy non-EMA runs — only used to recover training coordinates for sizes that
# have no EMA counterpart (the sample indices are shared between the two trees).
LOGS_LEGACY = PROJECT / "Code" / "DDIM_FMM" / "logs"

RESULTS_DIR = HERE / "3p5GHz" / "results"
CACHE_DIR = HERE / "3p5GHz" / "cache"
CKPT_DIR = RESULTS_DIR / "checkpoints"

# ── Configuration ─────────────────────────────────────────────────────────────
DDIM_N = 200                                     # DDIM generator training size
TAUS = [1000, 10000, 100000, 200000]             # DDIM sampling horizons (all run)
PLOT_TAUS = [10000, 100000]                      # horizons shown in the figure
N_REAL = 200                                     # real channels in augmentation
GEN_SIZE = 5000                                  # generated channels to add
N_TEST_PER_SIDE = 5000                           # 5000 LoS + 5000 NLoS
NUM_PROBING_BEAM_PAIRS = [2, 4, 8, 12]           # up to 12 only
SEED = 42
GEN_SEED = 0                                     # seed tag of the EMA pools

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
    """Load 3.5 GHz channels using cached arrays if available (LoS + NLoS separate)."""
    f_Hlos = CACHE_DIR / "H_los.npy"
    f_clos = CACHE_DIR / "coords_los.npy"
    f_Hnlos = CACHE_DIR / "H_nlos.npy"
    f_cnlos = CACHE_DIR / "coords_nlos.npy"
    if f_Hlos.exists() and f_Hnlos.exists():
        return (np.load(f_Hlos), np.load(f_clos),
                np.load(f_Hnlos), np.load(f_cnlos))
    LOS_NPZ = DATASET_DIR / "Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"
    NLOS_NPZ = DATASET_DIR / "Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz"
    H_los, coords_los = load_sionna_raw(str(LOS_NPZ))
    H_nlos, coords_nlos = load_sionna_raw(str(NLOS_NPZ))
    return H_los, coords_los, H_nlos, coords_nlos


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
                    help="DDIM training-set size (= N_REAL). Default: 200")
    args = ap.parse_args()

    # Allow overriding N from the command line
    ddim_n = args.ddim_n
    n_real = ddim_n  # real channels = DDIM training size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"AUGMENTATION EXPERIMENT — Real {n_real} + Gen {GEN_SIZE} @ 3.5 GHz "
          f"(LoS+NLoS), DDIM N={ddim_n}, EMA weights")
    print("=" * 70)
    print(f"Device: {device}  |  epochs={args.epochs}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load raw data ─────────────────────────────────────────────────────────
    H_los, coords_los, H_nlos, coords_nlos = load_or_cache_raw()
    print(f"Scene: {H_los.shape[0]} LoS + {H_nlos.shape[0]} NLoS "
          f"= {H_los.shape[0] + H_nlos.shape[0]} total 3.5 GHz users")

    # ── Leakage-safe split ────────────────────────────────────────────────────
    # Only exclude the channels the specific DDIM_N generator was trained on.
    ddim_coords = load_ddim_train_coords_ema(str(LOGS_EMA), ddim_n)
    ddim_seen_keys = set(coord_keys(ddim_coords)) if len(ddim_coords) else set()
    print(f"DDIM N={ddim_n} seen channels excluded: {len(ddim_seen_keys)}")

    split = build_master_split(H_los, coords_los, H_nlos, coords_nlos,
                               ddim_seen_keys,
                               n_test_per_side=N_TEST_PER_SIDE, seed=SEED)
    H_test = split["H_test"]
    print(f"Test set: {H_test.shape[0]} channels (DDIM-unseen)")

    # ── Real training data = exact DDIM training channels ──────────────────
    H_real = load_ddim_train_channels_ema(str(LOGS_EMA), ddim_n, H_los, H_nlos)
    print(f"Real subset: {H_real.shape[0]} channels (= DDIM training set)")

    # ── Baselines — computed on the SAME test set the DL curves are scored on ──
    baselines = compute_baselines(H_test, SYS)
    bl_mrt = baselines["MRT_MRC"]
    bl_genie = baselines["genie_DFT"]
    print(f"  MRT+MRC   = {bl_mrt:.2f} dB  (over {H_test.shape[0]} test channels)")
    print(f"  Genie DFT = {bl_genie:.2f} dB")

    # ── Real Only — both scale modes ──────────────────────────────────────────
    # The augmented runs are trained and evaluated under per-sample Frobenius
    # normalization.  Reporting the real-only reference in that same mode is the
    # only way to read the augmentation gain without the scale change mixed in.
    print(f"\n=== Real Only (N={n_real}, physical scale) ===")
    snr_real = run_one(f"realonly_N{n_real}", H_real, frob_norm=False,
                       h_test_raw=H_test, npb_list=NUM_PROBING_BEAM_PAIRS,
                       args=args, device=device)

    print(f"\n=== Real Only (N={n_real}, Frobenius-normalized) ===")
    snr_real_frob = run_one(f"realonly_frob_N{n_real}", H_real, frob_norm=True,
                            h_test_raw=H_test, npb_list=NUM_PROBING_BEAM_PAIRS,
                            args=args, device=device)

    # ── Real N + Gen 5000 @ each tau ──────────────────────────────────────────
    snr_aug = {}
    for tau in TAUS:
        name = f"aug_real{n_real}_gen{GEN_SIZE}_tau{tau}"
        print(f"\n=== {name} ===")
        print(f"  pool: {ema_synthetic_path(str(LOGS_EMA), ddim_n, tau, seed=GEN_SEED)}")
        H_syn = load_synthetic_pool_ema(str(LOGS_EMA), ddim_n, tau, seed=GEN_SEED)
        if len(H_syn) > GEN_SIZE:
            rng = np.random.default_rng(SEED)
            H_syn = H_syn[rng.permutation(len(H_syn))[:GEN_SIZE]]
        # Concatenate real + synthetic
        H_combined = np.concatenate([H_real, H_syn], axis=0)
        print(f"  Combined: {H_real.shape[0]} real + {H_syn.shape[0]} gen "
              f"= {H_combined.shape[0]} total")
        snr_aug[tau] = run_one(name, H_combined, frob_norm=True,
                               h_test_raw=H_test, npb_list=NUM_PROBING_BEAM_PAIRS,
                               args=args, device=device)

    # ── Plot ──────────────────────────────────────────────────────────────────
    npb = np.array(NUM_PROBING_BEAM_PAIRS)
    fig, ax = plt.subplots(figsize=(10, 7))

    # Genie-aided bounds
    ax.axhline(bl_mrt, color="black", ls="-", lw=2.0, zorder=4,
               label="MRT + MRC (upper bound)")
    ax.axhline(bl_genie, color="black", ls="--", lw=2.0, zorder=4,
               label="Genie-aided DFT beam pair")

    # Real-only reference, Frobenius-normalized to match the augmented runs
    real_frob_snrs = [snr_real_frob[k] for k in NUM_PROBING_BEAM_PAIRS]
    ax.plot(npb, real_frob_snrs, color="#8c564b", ls=":", lw=2.0, marker="o",
            ms=6, mfc="none", zorder=3,
            label=f"Real Only (N={n_real})")

    # Augmented curves
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
    fig_dir = HERE / "3p5GHz" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = fig_dir / f"beam_alignment_N{ddim_n}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)

    # ── Numerical results ────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "beam_alignment_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["curve", "ddim_n", "tau", "n_real", "n_gen", "frob_norm",
                    "num_probing_beam_pairs", "snr_dB"])
        for k in NUM_PROBING_BEAM_PAIRS:
            w.writerow(["MRT_MRC", "", "", "", "", "", k, bl_mrt])
            w.writerow(["genie_DFT", "", "", "", "", "", k, bl_genie])
            w.writerow(["real_only", "", "", n_real, 0, False, k, snr_real[k]])
            w.writerow(["real_only_frob", "", "", n_real, 0, True, k,
                        snr_real_frob[k]])
            for tau in TAUS:
                w.writerow(["augmented", ddim_n, tau, n_real, GEN_SIZE, True,
                            k, snr_aug[tau][k]])
    print(f"Saved: {csv_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
