"""
plot_w1_quality_vs_tau_ckptavg.py
    — Plot W1 Sample-Quality vs τ for the CHECKPOINT-AVERAGED (approx-EMA) results
================================================================================
Reads the CSV produced by compute_w1_quality_vs_tau_ckptavg.py and creates the
same set of figures as plot_w1_quality_vs_tau.py, but for the checkpoint-averaged
(poor-man's EMA) effective-rank Wasserstein curves.

To keep these results from mixing with the raw / previous ones, the plots are
saved under a DEDICATED folder:
    results/w1_quality_vs_tau_ckptavg/figures/

Figures produced:
  1. wasserstein_quality_vs_tau_win{W}.pdf/png       — W1(Gen, Test) vs τ
  2. excess_gap_vs_tau_win{W}.pdf/png                — Excess Gap vs τ
  3. wasserstein_quality_vs_tau_over_N_win{W}.pdf/png— W1(Gen, Test) vs τ/N
  4. w1_all_metrics_vs_tau_win{W}.pdf/png            — All three W1 metrics

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation

    # Plot the window=3 result (default)
    python plot_w1_quality_vs_tau_ckptavg.py --window 3

    # Plot another window's CSV
    python plot_w1_quality_vs_tau_ckptavg.py --window 2

    # Fully custom paths
    python plot_w1_quality_vs_tau_ckptavg.py \
        --csv_path results/w1_quality_vs_tau_ckptavg/wasserstein_quality_vs_tau_win3.csv \
        --output_dir results/w1_quality_vs_tau_ckptavg/figures
"""

import os
import sys
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

# Color scheme for each N (matches plot_w1_quality_vs_tau.py)
COLORS = {
    100:  '#e41a1c',
    200:  '#ff7f00',
    500:  '#984ea3',
    1000: '#4daf4a',
    2000: '#377eb8',
    4000: '#d62728',
    8000: '#a65628',
}

# Marker for each N (matches plot_w1_quality_vs_tau.py)
MARKERS = {
    100:  'o',
    200:  's',
    500:  '^',
    1000: 'D',
    2000: 'v',
    4000: 'P',
    8000: 'X',
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         PLOTTING FUNCTIONS                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_w1_gen_test_vs_tau(df, output_dir, window):
    """Main plot: W1(Gen, Test) vs tau, one curve per N."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for N in sorted(df['N'].unique()):
        sub = df[df['N'] == N].sort_values('actual_tau')
        tau = sub['actual_tau'].values
        w1 = sub['W1_Gen_Test'].values
        color = COLORS.get(N, '#333333')
        marker = MARKERS.get(N, 'o')
        ax.plot(tau, w1, '-', color=color, linewidth=2.0,
                marker=marker, markersize=5)

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax.set_ylabel(r'$W_1$(Gen, Test) on Effective Rank', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=20)

    size_handles = []
    for N in sorted(df['N'].unique()):
        size_handles.append(
            Line2D([0], [0], color=COLORS.get(N, '#333333'),
                   marker=MARKERS.get(N, 'o'), linewidth=2.0,
                   linestyle='-', markersize=7, label=f'N={N}')
        )
    ax.legend(handles=size_handles, fontsize=16, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'wasserstein_quality_vs_tau_win{window}.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_excess_gap_vs_tau(df, output_dir, window):
    """Excess gap: W1(Gen, Test) - W1(Test, Train) vs tau."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for N in sorted(df['N'].unique()):
        sub = df[df['N'] == N].sort_values('actual_tau')
        ax.plot(sub['actual_tau'].values, sub['Excess_Gap'].values, '-',
                color=COLORS.get(N, '#333333'), linewidth=2.0,
                marker=MARKERS.get(N, 'o'), markersize=5, label=f'N={N}')

    ax.set_xscale('log')
    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax.set_ylabel(r'Excess Gap: $W_1$(Gen,Test) $-$ $W_1$(Test,Train)', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'excess_gap_vs_tau_win{window}.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_w1_vs_tau_over_N(df, output_dir, window):
    """W1(Gen, Test) vs tau/N."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for N in sorted(df['N'].unique()):
        sub = df[df['N'] == N].sort_values('actual_tau')
        ax.plot(sub['actual_tau'].values / N, sub['W1_Gen_Test'].values, '-',
                color=COLORS.get(N, '#333333'), linewidth=2.0,
                marker=MARKERS.get(N, 'o'), markersize=5, label=f'N={N}')

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau / N$ (optimizer updates per sample)', fontsize=20)
    ax.set_ylabel(r'$W_1$(Gen, Test) on Effective Rank', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'wasserstein_quality_vs_tau_over_N_win{window}.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_all_w1_metrics(df, output_dir, window):
    """Three-panel plot: W1(Gen,Test), W1(Gen,Train), W1(Test,Train) vs tau."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    for N in sorted(df['N'].unique()):
        sub = df[df['N'] == N].sort_values('actual_tau')
        tau = sub['actual_tau'].values
        color = COLORS.get(N, '#333333')
        marker = MARKERS.get(N, 'o')
        ax1.plot(tau, sub['W1_Gen_Test'].values, '-', color=color,
                 linewidth=2.0, marker=marker, markersize=4, label=f'N={N}')
        ax2.plot(tau, sub['W1_Gen_Train'].values, '-', color=color,
                 linewidth=2.0, marker=marker, markersize=4, label=f'N={N}')
        ax3.plot(tau, sub['W1_Test_Train'].values, '-', color=color,
                 linewidth=2.0, marker=marker, markersize=4, label=f'N={N}')

    for ax, ylabel in [
        (ax1, r'$W_1$(Gen, Test)'),
        (ax2, r'$W_1$(Gen, Train)'),
        (ax3, r'$W_1$(Test, Train)'),
    ]:
        ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=20)
        ax.legend(loc='upper right', fontsize=16, framealpha=0.9)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'w1_all_metrics_vs_tau_win{window}.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot checkpoint-averaged (approx-EMA) W1 sample-quality vs tau")
    parser.add_argument("--window", type=int, default=3,
                        help="Averaging window whose CSV to plot (must match a produced CSV)")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Explicit CSV path (overrides --window default)")
    parser.add_argument("--output_dir", type=str,
                        default="results/w1_quality_vs_tau_ckptavg/figures",
                        help="Dedicated output directory for these plots")
    args = parser.parse_args()

    csv_path = args.csv_path or (
        f"results/w1_quality_vs_tau_ckptavg/wasserstein_quality_vs_tau_win{args.window}.csv")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PLOTTING: W1 Sample-Quality vs Tau  (CHECKPOINT-AVERAGED / approx-EMA)")
    print("=" * 60)
    print(f"  Window: {args.window}")
    print(f"  CSV: {csv_path}")
    print(f"  Output: {args.output_dir}")

    if not os.path.exists(csv_path):
        print(f"\nERROR: CSV not found at {csv_path}")
        print("Run compute_w1_quality_vs_tau_ckptavg.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"\n  Loaded {len(df)} rows for N = {sorted(df['N'].unique().tolist())}")
    print()

    print("Generating plots...")
    plot_w1_gen_test_vs_tau(df, args.output_dir, args.window)
    plot_excess_gap_vs_tau(df, args.output_dir, args.window)
    plot_w1_vs_tau_over_N(df, args.output_dir, args.window)
    plot_all_w1_metrics(df, args.output_dir, args.window)

    print(f"\nDone! All plots saved to: {args.output_dir}")
