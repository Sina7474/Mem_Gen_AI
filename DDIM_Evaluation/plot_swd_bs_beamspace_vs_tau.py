"""
plot_swd_bs_beamspace_vs_tau.py — Plot SW1 BS Beamspace Quality vs τ
=====================================================================
Task 13: Sliced Wasserstein-1 on 32-dim BS beam-power profiles.

Reads the CSV from compute_swd_bs_beamspace_vs_tau.py and creates:
  1. swd_bs_beamspace_vs_tau.pdf/png          — SW1 vs τ per N (one n_feat)
  2. swd_bs_beamspace_vs_tau_all_nfeat.pdf/png — multi-panel, one per n_feat
  3. swd_projection_convergence.pdf/png        — convergence check plot

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_swd_bs_beamspace_vs_tau.py

    # Specific n_feat:
    python plot_swd_bs_beamspace_vs_tau.py --n_feat 256
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

RESULTS_DIR = Path("results/swd_bs_beamspace_vs_tau")
OUTPUT_DIR  = Path("tau_plots")

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
# Plot 1: SW1 vs tau for a specific n_feat, one curve per N
# ─────────────────────────────────────────────────────────────────────────────

def plot_sw1_vs_tau(df: pd.DataFrame, n_feat: int, output_dir: Path):
    """SW1(Gen, Test) mean ± 2σ vs τ, one curve per N."""
    sub = df[df['n_feat'] == n_feat].copy()
    if sub.empty:
        print(f"  No data for n_feat={n_feat}, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for N in sorted(sub['N'].unique()):
        d = sub[sub['N'] == N].sort_values('actual_tau')
        tau = d['actual_tau'].values
        mean = d['SW1_Gen_Test_mean'].values
        err  = d['SW1_Gen_Test_error_2std'].values
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')

        ax.plot(tau, mean, '-', color=color, linewidth=2.0,
                marker=marker, markersize=5)
        ax.fill_between(tau, mean - err, mean + err,
                        color=color, alpha=0.15)

    # Real-real baseline (horizontal dashed line)
    if 'SW1_real_real_baseline' in sub.columns:
        baseline = sub['SW1_real_real_baseline'].dropna().iloc[0] \
                   if not sub['SW1_real_real_baseline'].dropna().empty else None
        if baseline is not None:
            ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5,
                       alpha=0.7, label='Real-Real baseline')

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax.set_ylabel(r'SW$_1$ on BS Beam-Power Profile', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=16)

    # Legend
    handles = []
    for N in sorted(sub['N'].unique()):
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')
        handles.append(
            Line2D([0], [0], color=color, marker=marker, linewidth=2.0,
                   linestyle='-', markersize=7, label=f'N={N}'))
    if baseline is not None:
        handles.append(
            Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5,
                   label='Real-Real'))
    ax.legend(handles=handles, fontsize=14, loc='upper right', framealpha=0.9)

    suffix = f'_nfeat{n_feat}'
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = output_dir / f'swd_bs_beamspace_vs_tau{suffix}.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Multi-panel for all n_feats
# ─────────────────────────────────────────────────────────────────────────────

def plot_sw1_all_nfeat(df: pd.DataFrame, output_dir: Path):
    """One subplot per n_feat, all sharing the same y-axis."""
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
            mean = d['SW1_Gen_Test_mean'].values
            err  = d['SW1_Gen_Test_error_2std'].values
            color = SIZE_COLORS.get(N, '#333333')
            marker = SIZE_MARKERS.get(N, 'o')

            ax.plot(tau, mean, '-', color=color, linewidth=1.5,
                    marker=marker, markersize=4)
            ax.fill_between(tau, mean - err, mean + err,
                            color=color, alpha=0.12)

        # Baseline
        if 'SW1_real_real_baseline' in sub.columns:
            bl = sub['SW1_real_real_baseline'].dropna()
            if not bl.empty:
                ax.axhline(y=bl.iloc[0], color='gray', linestyle='--',
                           linewidth=1.0, alpha=0.6)

        ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$\tau$', fontsize=14)
        ax.set_title(f'n_feat={n_feat}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)

    axes[0].set_ylabel(r'SW$_1$ on BS Beam-Power', fontsize=14)

    # Shared legend
    handles = []
    all_N = sorted(df['N'].unique())
    for N in all_N:
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')
        handles.append(
            Line2D([0], [0], color=color, marker=marker, linewidth=1.5,
                   linestyle='-', markersize=6, label=f'N={N}'))
    fig.legend(handles=handles, fontsize=11, loc='upper center',
               ncol=min(len(all_N), 6), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = output_dir / f'swd_bs_beamspace_vs_tau_all_nfeat.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Projection convergence
# ─────────────────────────────────────────────────────────────────────────────

def plot_projection_convergence(results_dir: Path, output_dir: Path):
    """Plot SW1 vs number of projections."""
    conv_csv = results_dir / 'swd_projection_convergence.csv'
    if not conv_csv.exists():
        print("  Projection convergence CSV not found, skipping.")
        return

    df = pd.read_csv(conv_csv)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(df['num_projections'], df['SW1'], 'b-o', linewidth=2, markersize=7)
    ax.set_xlabel('Number of Projections (L)', fontsize=16)
    ax.set_ylabel(r'SW$_1$', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    ax.axvline(x=1024, color='red', linestyle=':', linewidth=1.0, alpha=0.7,
               label='L=1024 (used)')
    ax.legend(fontsize=13)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = output_dir / f'swd_projection_convergence.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot SW1 BS beamspace quality vs tau")
    parser.add_argument("--n_feat", type=int, default=256,
                        help="n_feat for single-panel plot (default 256)")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    output_dir  = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / 'swd_bs_beamspace_vs_tau.csv'
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run compute_swd_bs_beamspace_vs_tau.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  n_feats: {sorted(df['n_feat'].unique())}")
    print(f"  sizes:   {sorted(df['N'].unique())}")

    # Plot 1: Individual plot for each n_feat
    for nf in sorted(df['n_feat'].unique()):
        print(f"\nPlotting SW1 vs tau for n_feat={nf} ...")
        plot_sw1_vs_tau(df, nf, output_dir)

    # Plot 2: All n_feats multi-panel
    print("\nPlotting SW1 vs tau for all n_feats ...")
    plot_sw1_all_nfeat(df, output_dir)

    # Plot 3: Convergence
    print("\nPlotting projection convergence ...")
    plot_projection_convergence(results_dir, output_dir)

    # Also save figures in results dir
    fig_dir = results_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    print(f"\nCopying figures to {fig_dir} ...")
    for nf in sorted(df['n_feat'].unique()):
        plot_sw1_vs_tau(df, nf, fig_dir)
    plot_projection_convergence(results_dir, fig_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
