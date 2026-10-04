"""
plot_task09.py — Plot downstream CSI compression results (Task 09)
===================================================================
Reads downstream_csi_compression_results.csv and produces:
  1. downstream_csi_compression_augmentation.pdf/png
       Combined figure: Real Only, Benchmark, and all augmentation curves.
  2. downstream_csi_compression_augmentation_panels.pdf/png
       Multi-panel version: one panel per DDIM generator size.
  3. downstream_csi_compression_augmentation_DDIM_N<size>.pdf/png
       One SEPARATE figure per DDIM generator size (100, 200, 1000).
       Each shows Real Only + Benchmark + the three tau curves for that size.

x-axis: number of real downstream training channels (200, 500, 1000, 5000)
y-axis: Test NMSE (dB) — lower is better

Usage:
    conda activate Mem_Gen
    cd Downstream_Tasks/CSI_Compression
    python plot_task09.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results" / "downstream_csi_compression_results.csv"
FIG_DIR = HERE / "figures"

REAL_SIZES = [200, 500, 1000, 5000]
TAUS = [1000, 10000, 100000]
DDIM_SIZES = [100, 200, 1000]

# tau → color; DDIM_N → marker + line style (for the combined figure)
TAU_COLORS = {1000: "#1b9e77", 10000: "#d95f02", 100000: "#7570b3"}
TAU_LABEL = {1000: r"$\tau$=1k", 10000: r"$\tau$=10k", 100000: r"$\tau$=100k"}
DDIM_MARKER = {100: "^", 200: "o", 1000: "s"}
DDIM_LINESTYLE = {100: ":", 200: "--", 1000: "-."}


def load_results():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results CSV not found: {RESULTS_CSV}. Run run_task09.py first.")
    return pd.read_csv(RESULTS_CSV)


def available_ddim_sizes(df):
    """Return the DDIM sizes actually present in the results CSV."""
    present = set(df["DDIM_train_N"].astype(str).unique())
    return [n for n in DDIM_SIZES if str(n) in present]


def get_real_only_curve(df):
    """Return (sizes, nmse) for the real-only baseline, using benchmark for 5000."""
    xs, ys = [], []
    for n in REAL_SIZES:
        if n == 5000:
            row = df[df["group"] == "Benchmark"]
        else:
            row = df[(df["group"] == "Real Only") & (df["N_downstream_real"] == n)]
        if not row.empty:
            xs.append(n)
            ys.append(float(row["test_NMSE_dB"].iloc[0]))
    return xs, ys


def get_aug_curve(df, ddim_n, tau, real_only_5000):
    """Return (sizes, nmse) for an augmentation curve; 5000 point = real-only 5000."""
    xs, ys = [], []
    for n in REAL_SIZES:
        if n == 5000:
            if real_only_5000 is not None:
                xs.append(n)
                ys.append(real_only_5000)
            continue
        row = df[(df["DDIM_train_N"].astype(str) == str(ddim_n)) &
                 (df["tau"].astype(str) == str(tau)) &
                 (df["N_downstream_real"] == n)]
        if not row.empty:
            xs.append(n)
            ys.append(float(row["test_NMSE_dB"].iloc[0]))
    return xs, ys


def _style_axis(ax, xlabel=True):
    ax.set_xscale("log")
    ax.set_xticks(REAL_SIZES)
    ax.set_xticklabels([str(n) for n in REAL_SIZES])
    if xlabel:
        ax.set_xlabel("$N_{\mathrm{ref}}$", fontsize=20)
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# 1. COMBINED FIGURE (all sizes together)
# ─────────────────────────────────────────────────────────────────────────────
def plot_combined(df):
    xs_ro, ys_ro = get_real_only_curve(df)
    bench = df[df["group"] == "Benchmark"]
    bench_val = float(bench["test_NMSE_dB"].iloc[0]) if not bench.empty else None
    sizes = available_ddim_sizes(df)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(xs_ro, ys_ro, color="black", marker="D", linewidth=2.0,
            linestyle="-", label="Real Only")
    if bench_val is not None:
        ax.axhline(bench_val, color="gray", linestyle=":", linewidth=1.5,
                   label="Full Real 5000 (benchmark)")

    for ddim_n in sizes:
        for tau in TAUS:
            xs, ys = get_aug_curve(df, ddim_n, tau, bench_val)
            if not xs:
                continue
            ax.plot(xs, ys, color=TAU_COLORS[tau], marker=DDIM_MARKER[ddim_n],
                    linewidth=1.5, linestyle=DDIM_LINESTYLE[ddim_n],
                    label=f"Aug DDIM_N={ddim_n}, {TAU_LABEL[tau]}")

    _style_axis(ax)
    ax.set_ylabel("Test NMSE (dB)", fontsize=20)
    #ax.set_title("Downstream CSI Compression: DDIM Synthetic Augmentation", fontsize=20)
    ax.legend(fontsize=16, loc='upper right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"downstream_csi_compression_augmentation.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: downstream_csi_compression_augmentation.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MULTI-PANEL FIGURE (one panel per DDIM size)
# ─────────────────────────────────────────────────────────────────────────────
def plot_panels(df):
    xs_ro, ys_ro = get_real_only_curve(df)
    bench = df[df["group"] == "Benchmark"]
    bench_val = float(bench["test_NMSE_dB"].iloc[0]) if not bench.empty else None
    sizes = available_ddim_sizes(df)
    if not sizes:
        print("  (no DDIM sizes found; skipping panels)")
        return

    ncols = len(sizes)
    fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols, 6), sharey=True)
    if ncols == 1:
        axes = [axes]

    panel_letters = ["A", "B", "C", "D", "E"]
    for ax, ddim_n, letter in zip(axes, sizes, panel_letters):
        ax.plot(xs_ro, ys_ro, color="black", marker="D", linewidth=2.0,
                linestyle="-", label="Real Only")
        if bench_val is not None:
            ax.axhline(bench_val, color="gray", linestyle=":", linewidth=1.5,
                       label="Full Real 5000 (benchmark)")
        for tau in TAUS:
            xs, ys = get_aug_curve(df, ddim_n, tau, bench_val)
            if not xs:
                continue
            ax.plot(xs, ys, color=TAU_COLORS[tau], marker="o", linewidth=1.5,
                    linestyle="--", label=f"Aug {TAU_LABEL[tau]}")
        _style_axis(ax)
        #ax.set_title(f"Panel {letter}: DDIM_N={ddim_n}", fontsize=20)
        ax.legend(fontsize=16, loc="best")

    axes[0].set_ylabel("Test NMSE (dB)", fontsize=20)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(FIG_DIR / f"downstream_csi_compression_augmentation_panels.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: downstream_csi_compression_augmentation_panels.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEPARATE FIGURE PER DDIM SIZE
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_size(df):
    xs_ro, ys_ro = get_real_only_curve(df)
    bench = df[df["group"] == "Benchmark"]
    bench_val = float(bench["test_NMSE_dB"].iloc[0]) if not bench.empty else None

    for ddim_n in available_ddim_sizes(df):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(xs_ro, ys_ro, color="black", marker="D", linewidth=2.0,
            linestyle="-", label=r"$N_{\mathrm{csi}}$ only")
        if bench_val is not None:
            ax.axhline(bench_val, color="gray", linestyle=":", linewidth=1.5,
                   label=r"$N_{\mathrm{csi}}=5000$")

        for tau in TAUS:
            xs, ys = get_aug_curve(df, ddim_n, tau, bench_val)
            if not xs:
                continue
            ax.plot(xs, ys, color=TAU_COLORS[tau], marker="o", linewidth=1.8,
                    linestyle="--", label=f"Aug {TAU_LABEL[tau]}")

        _style_axis(ax)
        ax.set_ylabel("Test NMSE (dB)", fontsize=20)
        #ax.set_title(f"Downstream CSI Compression — DDIM_N={ddim_n}", fontsize=20)
        ax.legend(fontsize=16, loc='upper right', frameon=True, framealpha=0.9)

        plt.tight_layout()
        for ext in ["pdf", "png"]:
            fig.savefig(
                FIG_DIR / f"downstream_csi_compression_augmentation_DDIM_N{ddim_n}.{ext}",
                dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: downstream_csi_compression_augmentation_DDIM_N{ddim_n}.pdf/png")


def main():
    parser = argparse.ArgumentParser(description="Plot Task 09 downstream results")
    parser.add_argument("--csv", type=str, default=None, help="Override results CSV path")
    args = parser.parse_args()

    global RESULTS_CSV
    if args.csv:
        RESULTS_CSV = Path(args.csv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_results()
    print(f"Loaded {len(df)} experiment rows.")
    print(f"DDIM sizes present: {available_ddim_sizes(df)}")
    plot_combined(df)
    plot_panels(df)
    plot_per_size(df)
    print("\nAll plots done.")


if __name__ == "__main__":
    main()
