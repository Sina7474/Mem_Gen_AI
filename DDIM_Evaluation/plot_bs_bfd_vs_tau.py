"""
plot_bs_bfd_vs_tau.py — Plot BS-Beamspace Fréchet Distance vs τ
=================================================================
Task 14 plotting script.

Creates:
  1. bs_bfd_vs_tau.pdf/png                 — BS-BFD vs τ per N (single n_feat)
  2. bs_bfd_vs_tau_all_nfeat.pdf/png       — multi-panel, one per n_feat
  3. bs_bfd_decomposition_vs_tau.pdf/png   — mean vs covariance term decomposition

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_bs_bfd_vs_tau.py

    # Specific n_feat:
    python plot_bs_bfd_vs_tau.py --n_feat 256
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from pathlib import Path

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results/bs_bfd_vs_tau")
OUTPUT_DIR  = Path("results/bs_bfd_vs_tau/figures")

SIZE_COLORS = {
    100:  '#e41a1c',
    200:  '#ff7f00',
    500:  '#984ea3',
    1000: '#4daf4a',
    2000: '#377eb8',
    4000: '#d62728',
    8000: '#a65628',
}

SIZE_MARKERS = {
    100:  'o',
    200:  's',
    500:  '^',
    1000: 'D',
    2000: 'v',
    4000: 'P',
    8000: 'X',
}


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: BS-BFD vs tau (single n_feat)
# ─────────────────────────────────────────────────────────────────────────────

def plot_bfd_vs_tau(df: pd.DataFrame, n_feat: int, output_dir: Path):
    """BS-BFD mean ± 2σ vs τ, one curve per N."""
    sub = df[df['n_feat'] == n_feat].copy()
    if sub.empty:
        print(f"  No data for n_feat={n_feat}, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for N in sorted(sub['N'].unique()):
        d = sub[sub['N'] == N].sort_values('actual_tau')
        tau = d['actual_tau'].values
        mean = d['BS_BFD_Gen_Test_mean'].values
        err  = d['BS_BFD_Gen_Test_error_2std'].values
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')

        ax.plot(tau, mean, '-', color=color, linewidth=2.0,
                marker=marker, markersize=5)
        ax.fill_between(tau, np.maximum(mean - err, 0), mean + err,
                        color=color, alpha=0.15)

    # Real-real baseline
    if 'BS_BFD_real_real_baseline' in sub.columns:
        bl = sub['BS_BFD_real_real_baseline'].dropna()
        if not bl.empty:
            ax.axhline(y=bl.iloc[0], color='gray', linestyle='--', linewidth=1.5,
                       alpha=0.7)

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax.set_ylabel('BS-Beamspace Fréchet Distance', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=16)

    handles = []
    for N in sorted(sub['N'].unique()):
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')
        handles.append(
            Line2D([0], [0], color=color, marker=marker, linewidth=2.0,
                   linestyle='-', markersize=7, label=f'N={N}'))
    handles.append(
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5,
               label='Real-Real'))
    ax.legend(handles=handles, fontsize=14, loc='upper right', framealpha=0.9)

    suffix = f'_nfeat{n_feat}'
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = output_dir / f'bs_bfd_vs_tau{suffix}.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Multi-panel for all n_feats
# ─────────────────────────────────────────────────────────────────────────────

def plot_bfd_all_nfeat(df: pd.DataFrame, output_dir: Path):
    """One subplot per n_feat."""
    n_feats = sorted(df['n_feat'].unique())
    n_panels = len(n_feats)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for idx, n_feat in enumerate(n_feats):
        ax = axes[idx]
        sub = df[df['n_feat'] == n_feat]

        for N in sorted(sub['N'].unique()):
            d = sub[sub['N'] == N].sort_values('actual_tau')
            tau = d['actual_tau'].values
            mean = d['BS_BFD_Gen_Test_mean'].values
            err  = d['BS_BFD_Gen_Test_error_2std'].values
            color = SIZE_COLORS.get(N, '#333333')
            marker = SIZE_MARKERS.get(N, 'o')

            ax.plot(tau, mean, '-', color=color, linewidth=1.5,
                    marker=marker, markersize=4)
            ax.fill_between(tau, np.maximum(mean - err, 0), mean + err,
                            color=color, alpha=0.12)

        if 'BS_BFD_real_real_baseline' in sub.columns:
            bl = sub['BS_BFD_real_real_baseline'].dropna()
            if not bl.empty:
                ax.axhline(y=bl.iloc[0], color='gray', linestyle='--',
                           linewidth=1.0, alpha=0.6)

        ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$\tau$', fontsize=14)
        ax.set_title(f'n_feat={n_feat}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)

    axes[0].set_ylabel('BS-BFD', fontsize=14)

    handles = []
    for N in sorted(df['N'].unique()):
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')
        handles.append(
            Line2D([0], [0], color=color, marker=marker, linewidth=1.5,
                   linestyle='-', markersize=6, label=f'N={N}'))
    fig.legend(handles=handles, fontsize=11, loc='upper center',
               ncol=min(len(df['N'].unique()), 6), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = output_dir / f'bs_bfd_vs_tau_all_nfeat.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Decomposition (mean term vs covariance term)
# ─────────────────────────────────────────────────────────────────────────────

def plot_decomposition(df: pd.DataFrame, n_feat: int, output_dir: Path):
    """Stacked or side-by-side: mean_term and covariance_term vs τ."""
    sub = df[df['n_feat'] == n_feat].copy()
    if sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for N in sorted(sub['N'].unique()):
        d = sub[sub['N'] == N].sort_values('actual_tau')
        tau = d['actual_tau'].values
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')

        ax1.plot(tau, d['BS_BFD_mean_term_mean'].values, '-', color=color,
                 linewidth=2.0, marker=marker, markersize=4)
        ax2.plot(tau, d['BS_BFD_covariance_term_mean'].values, '-', color=color,
                 linewidth=2.0, marker=marker, markersize=4)

    ax1.set_xscale('log')
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel('Mean Term', fontsize=16)
    ax1.set_title('BS-BFD Decomposition: Mean Term vs τ', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=14)

    ax2.set_xscale('log')
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=18)
    ax2.set_ylabel('Covariance Term', fontsize=16)
    ax2.set_title('BS-BFD Decomposition: Covariance Term vs τ', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=14)

    handles = []
    for N in sorted(sub['N'].unique()):
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')
        handles.append(
            Line2D([0], [0], color=color, marker=marker, linewidth=2.0,
                   linestyle='-', markersize=6, label=f'N={N}'))
    ax1.legend(handles=handles, fontsize=12, loc='upper right', framealpha=0.9)

    suffix = f'_nfeat{n_feat}'
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = output_dir / f'bs_bfd_decomposition_vs_tau{suffix}.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot BS-BFD vs tau")
    parser.add_argument("--n_feat", type=int, default=256)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    output_dir  = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / 'bs_bfd_vs_tau.csv'
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run compute_bs_bfd_vs_tau.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  n_feats: {sorted(df['n_feat'].unique())}")
    print(f"  sizes:   {sorted(df['N'].unique())}")

    # Plot 1: Single n_feat
    print(f"\nPlotting BS-BFD vs tau for n_feat={args.n_feat} ...")
    plot_bfd_vs_tau(df, args.n_feat, output_dir)

    # Plot 2: All n_feats
    print("\nPlotting BS-BFD vs tau for all n_feats ...")
    plot_bfd_all_nfeat(df, output_dir)

    # Plot 3: Decomposition
    print(f"\nPlotting decomposition for n_feat={args.n_feat} ...")
    plot_decomposition(df, args.n_feat, output_dir)

    # Also for other n_feats
    for nf in sorted(df['n_feat'].unique()):
        if nf != args.n_feat:
            print(f"\nPlotting BS-BFD + decomposition for n_feat={nf} ...")
            plot_bfd_vs_tau(df, nf, output_dir)
            plot_decomposition(df, nf, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
