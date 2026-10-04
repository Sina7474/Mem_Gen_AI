"""
plot_all_sizes_comparison.py — Comparison plots across all training set sizes
=============================================================================
Creates:
  1. Overlaid CDF: training data (dashed) + generated data (solid) per size
  2. Wasserstein bar chart: W1(train data, generated) per size
  3. Mean SV spectrum comparison
  4. Mean effective rank vs training size

Run after evaluate_all_sizes.py has completed.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_all_sizes_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# ── Mode selection ────────────────────────────────────────────────────────────
# Set INCREMENTAL = True  for the incremental (nested subset) experiment
# Set INCREMENTAL = False for the independent (separate shuffle) experiment
INCREMENTAL = True

if INCREMENTAL:
    RESULTS_DIR = Path("results/all_sizes_incremental")
else:
    RESULTS_DIR = Path("results/all_sizes_comparison")

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000, 8000]

# ── Figure style (matching original Effective_Rank_Github) ────────────────────
FIG_W, FIG_H = 10, 6
FONT_LABEL = 20
FONT_TICK = 20
FONT_LEGEND = 16
FONT_ANNOT = 14

# Y-axis limits for bar chart: set to None for automatic, or e.g. (0, 1.4)
BAR_YLIM = (0, 1.45)


def _setup_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset":  "stix",
        "axes.labelsize":    FONT_LABEL,
        "xtick.labelsize":   FONT_TICK,
        "ytick.labelsize":   FONT_TICK,
        "legend.fontsize":   FONT_LEGEND,
        "legend.frameon":    True,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "0.8",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.6,
        "grid.alpha":        0.6,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


# ── Colour palette for 7 training sizes ──────────────────────────────────────
COLORS = {
    100:  "#d32f2f",
    200:  "#f57c00",
    500:  "#fbc02d",
    1000: "#388e3c",
    2000: "#1976d2",
    4000: "#7b1fa2",
    8000: "#455a64",
}


def plot_cdf_all_sizes():
    """Overlaid CDF: dashed = training data, solid = generated."""
    _setup_style()
    cdf_data = np.load(RESULTS_DIR / "cdf_data.npz")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for size in TRAIN_SIZES:
        train_key = f"Train_{size}"
        gen_key = f"DDIM_{size}_gen"
        c = COLORS[size]
        if f"{train_key}_x" in cdf_data.files:
            ax.plot(cdf_data[f"{train_key}_x"], cdf_data[f"{train_key}_y"],
                    linestyle="--", color=c, linewidth=2.0, alpha=0.7)
        if f"{gen_key}_x" in cdf_data.files:
            ax.plot(cdf_data[f"{gen_key}_x"], cdf_data[f"{gen_key}_y"],
                    linestyle="-", color=c, linewidth=2.0,
                    label=f"N={size}")

    ax.set_xlabel("Effective Rank")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right', fontsize=FONT_LEGEND, frameon=True, framealpha=0.9)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'effective_rank_cdf_all_sizes.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'effective_rank_cdf_all_sizes.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'effective_rank_cdf_all_sizes.png'}")


def plot_wasserstein_bar():
    """Bar chart: W1(train data, generated) per training size."""
    _setup_style()
    w1_df = pd.read_csv(RESULTS_DIR / "wasserstein_distances.csv")

    sizes = w1_df['training_size'].values
    w1_vals = w1_df['W1_train_vs_gen'].values

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(sizes))
    bar_width = 0.5

    bars = ax.bar(x, w1_vals, width=bar_width,
                  facecolor="#1565c0", hatch="///", edgecolor="#0d3b75",
                  linewidth=0.6, zorder=3)

    for bar, val in zip(bars, w1_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=FONT_ANNOT, fontfamily="serif", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Wasserstein Distance")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)
    if BAR_YLIM is not None:
        ax.set_ylim(BAR_YLIM)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / 'wasserstein_bar.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'wasserstein_bar.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'wasserstein_bar.png'}")


def plot_wasserstein_line():
    """Line chart: W1 vs training size (log-x)."""
    _setup_style()
    w1_df = pd.read_csv(RESULTS_DIR / "wasserstein_distances.csv")

    sizes = w1_df['training_size'].values
    w1_vals = w1_df['W1_train_vs_gen'].values

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(sizes, w1_vals, 'o-', color="#1565c0", linewidth=2, markersize=7)
    ax.set_xscale('log')
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Wasserstein Distance")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_axisbelow(True)
    fig.tight_layout()

    fig.savefig(FIGURES_DIR / 'wasserstein_vs_training_size.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'wasserstein_vs_training_size.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'wasserstein_vs_training_size.png'}")


def plot_sv_spectrum_all_sizes():
    """Mean normalized SV spectrum: training data (dashed) + generated (solid)."""
    _setup_style()
    sv_data = np.load(RESULTS_DIR / "sv_spectrum.npz")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for size in TRAIN_SIZES:
        c = COLORS[size]
        train_key = f"Train_{size}"
        gen_key = f"DDIM_{size}_gen"
        if train_key in sv_data.files:
            sv = sv_data[train_key]
            ax.plot(range(1, len(sv)+1), sv, '--', color=c,
                    linewidth=2.0, alpha=0.7, markersize=5)
        if gen_key in sv_data.files:
            sv = sv_data[gen_key]
            ax.plot(range(1, len(sv)+1), sv, '-o', color=c,
                    label=f"N={size}", markersize=5, linewidth=2.0)

    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Normalized Singular Value")
    ax.legend(loc='upper right', fontsize=FONT_LEGEND, frameon=True, framealpha=0.9)
    ax.set_yscale('log')
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'sv_spectrum_all_sizes.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'sv_spectrum_all_sizes.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'sv_spectrum_all_sizes.png'}")


def plot_mean_erank_vs_size():
    """Mean effective rank vs training size: training data and generated."""
    _setup_style()
    summary = pd.read_csv(RESULTS_DIR / "summary.csv")

    sizes_train, means_train, stds_train = [], [], []
    sizes_gen, means_gen, stds_gen = [], [], []

    for size in TRAIN_SIZES:
        row_t = summary[summary['dataset'] == f'Train {size}']
        row_g = summary[summary['dataset'] == f'DDIM {size} gen']
        if len(row_t) > 0:
            sizes_train.append(size)
            means_train.append(row_t['mean'].values[0])
            stds_train.append(row_t['std'].values[0])
        if len(row_g) > 0:
            sizes_gen.append(size)
            means_gen.append(row_g['mean'].values[0])
            stds_gen.append(row_g['std'].values[0])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    ax.errorbar(sizes_train, means_train, yerr=stds_train, fmt='s--',
                color="#2e7d32", linewidth=2.0, markersize=5, capsize=4,
                label='Training Data')
    ax.errorbar(sizes_gen, means_gen, yerr=stds_gen, fmt='o-',
                color="#1565c0", linewidth=2.0, markersize=5, capsize=5,
                label='DDIM Generated')

    ax.set_xscale('log')
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Mean Effective Rank")
    ax.legend(loc='best', fontsize=FONT_LEGEND, frameon=True, framealpha=0.9)
    ax.set_xticks(TRAIN_SIZES)
    ax.set_xticklabels([str(s) for s in TRAIN_SIZES])
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'mean_erank_vs_training_size.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'mean_erank_vs_training_size.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {FIGURES_DIR / 'mean_erank_vs_training_size.png'}")


if __name__ == "__main__":
    print("=" * 70)
    print("Plotting Comparison Across All Training Set Sizes")
    print("=" * 70)
    plot_cdf_all_sizes()
    plot_wasserstein_bar()
    plot_wasserstein_line()
    plot_sv_spectrum_all_sizes()
    plot_mean_erank_vs_size()
    print("\n** All comparison plots saved! **")
    print(f"   Location: {FIGURES_DIR}")