"""
plot_fmem_vs_tau.py — Plot Memorization Fraction f_mem vs Training Time τ
==========================================================================
Reads fmem_vs_tau.csv (from compute_fmem_vs_tau.py) and optionally the
W1 quality CSV to produce:
  1. fmem_vs_tau.pdf/png             — f_mem(%) vs τ (one curve per N)
  2. quality_and_fmem_vs_tau.pdf/png — dual y-axis: W1 (solid, left) + f_mem (dashed, right)

Plotting style matches plot_tau_train_test_loss.py (same colors, markers, fonts).

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_fmem_vs_tau.py

    # Custom paths:
    python plot_fmem_vs_tau.py --fmem_csv results/fmem_vs_tau/fmem_vs_tau.csv \
        --w1_csv results/w1_quality_vs_tau/wasserstein_quality_vs_tau.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000, 8000]

COLORS = {
    100:  '#e41a1c',
    200:  '#ff7f00',
    500:  '#984ea3',
    1000: '#4daf4a',
    2000: '#377eb8',
    4000: '#d62728',
    8000: '#a65628',
}

MARKERS = {
    100:  'o',
    200:  's',
    500:  '^',
    1000: 'D',
    2000: 'v',
    4000: 'P',
    8000: 'X',
}

DEFAULT_FMEM_CSV = "results/fmem_vs_tau/fmem_vs_tau.csv"
DEFAULT_W1_CSV = "results/w1_quality_vs_tau/wasserstein_quality_vs_tau.csv"
DEFAULT_OUTPUT_DIR = "results/fmem_vs_tau/figures"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         PLOT 1: f_mem only                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_fmem_only(df, output_dir):
    """Plot f_mem(%) vs tau with bootstrap CI shading."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for N in TRAIN_SIZES:
        sub = df[df['N'] == N].sort_values('actual_tau')
        if sub.empty:
            continue

        tau = sub['actual_tau'].values
        fmem = 100.0 * sub['f_mem_k_1_3'].values
        ci_low = 100.0 * sub['f_mem_k_1_3_ci_low'].values
        ci_high = 100.0 * sub['f_mem_k_1_3_ci_high'].values

        ax.plot(tau, fmem, color=COLORS[N], marker=MARKERS[N],
                markersize=7, linewidth=1.5, label=f'N={N}')
        ax.fill_between(tau, ci_low, ci_high, color=COLORS[N], alpha=0.15)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax.set_ylabel(r'$f_{\mathrm{mem}}$ (%)', fontsize=20)
    #ax.set_title(r'Memorization Fraction $f_{\mathrm{mem}}$ vs $\tau$ (k=1/3)', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fmem_vs_tau.{ext}'), dpi=300,
                    bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fmem_vs_tau.pdf/png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                PLOT 2: Combined W1 + f_mem (dual y-axis)                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_combined(df_fmem, df_w1, output_dir):
    """Dual y-axis plot: W1(Gen,Test) solid left, f_mem dashed right."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for N in TRAIN_SIZES:
        color = COLORS[N]
        marker = MARKERS[N]

        # W1 on left axis (solid)
        sub_w1 = df_w1[df_w1['N'] == N].sort_values('actual_tau')
        if not sub_w1.empty:
            ax1.plot(sub_w1['actual_tau'].values, sub_w1['W1_Gen_Test'].values,
                     color=color, marker=marker, markersize=7, linewidth=1.5,
                     linestyle='-', label=f'N={N}')

        # f_mem on right axis (dashed)
        sub_fm = df_fmem[df_fmem['N'] == N].sort_values('actual_tau')
        if not sub_fm.empty:
            ax2.plot(sub_fm['actual_tau'].values,
                     100.0 * sub_fm['f_mem_k_1_3'].values,
                     color=color, marker=marker, markersize=6, linewidth=1.5,
                     linestyle='--')

    ax1.set_xscale('log')
    ax1.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax1.set_ylabel(r'$W_1$(Gen, Test) on Effective Rank', fontsize=20, color='black')
    ax2.set_ylabel(r'$f_{\mathrm{mem}}$ (%)', fontsize=20, color='black')
    ax1.tick_params(axis='both', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)

    # Legend: one entry per N + line style explanation
    handles_N = [Line2D([0], [0], color=COLORS[N], marker=MARKERS[N],
                        markersize=7, linewidth=1.5, label=f'N={N}')
                 for N in TRAIN_SIZES]
    handles_style = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=1.5,
               label=r'$W_1$(Gen, Test) — solid'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5,
               label=r'$f_{\mathrm{mem}}$ — dashed'),
    ]
    ax1.legend(handles=handles_N + handles_style, fontsize=16,
               loc='upper right', ncol=2, framealpha=0.9)

    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'quality_and_fmem_vs_tau.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: quality_and_fmem_vs_tau.pdf/png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║              PLOT 3: Normalized f_mem vs tau/N (optional)                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_normalized_fmem(df, output_dir):
    """Plot f_mem(tau)/max(f_mem) vs tau/N for scale comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for N in TRAIN_SIZES:
        sub = df[df['N'] == N].sort_values('actual_tau')
        if sub.empty:
            continue

        fmem = sub['f_mem_k_1_3'].values
        max_fmem = fmem.max()
        if max_fmem < 1e-10:
            continue

        tau_over_N = sub['actual_tau'].values / N
        f_norm = fmem / max_fmem

        ax.plot(tau_over_N, f_norm, color=COLORS[N], marker=MARKERS[N],
                markersize=7, linewidth=1.5, label=f'N={N}')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau / N$', fontsize=20)
    ax.set_ylabel(r'$f_{\mathrm{mem}}(\tau) / f_{\mathrm{mem}}^{\max}$', fontsize=20)
    #ax.set_title(r'Normalized Memorization vs $\tau/N$', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'normalized_fmem_vs_tau_over_N.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: normalized_fmem_vs_tau_over_N.pdf/png")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="Plot memorization fraction f_mem vs tau")
    parser.add_argument("--fmem_csv", type=str, default=DEFAULT_FMEM_CSV,
                        help="Path to fmem_vs_tau.csv")
    parser.add_argument("--w1_csv", type=str, default=DEFAULT_W1_CSV,
                        help="Path to wasserstein_quality_vs_tau.csv")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load f_mem CSV
    print(f"Loading f_mem CSV: {args.fmem_csv}")
    df_fmem = pd.read_csv(args.fmem_csv)
    print(f"  {len(df_fmem)} rows, sizes: {sorted(df_fmem['N'].unique())}")

    # Plot 1: f_mem only
    plot_fmem_only(df_fmem, args.output_dir)

    # Plot 3: normalized f_mem vs tau/N
    plot_normalized_fmem(df_fmem, args.output_dir)

    # Plot 2: combined (needs W1 CSV)
    if os.path.exists(args.w1_csv):
        print(f"Loading W1 CSV: {args.w1_csv}")
        df_w1 = pd.read_csv(args.w1_csv)
        plot_combined(df_fmem, df_w1, args.output_dir)
    else:
        print(f"  W1 CSV not found at {args.w1_csv}, skipping combined plot.")

    print("\nAll plots done.")


if __name__ == '__main__':
    main()
