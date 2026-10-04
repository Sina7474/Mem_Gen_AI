"""
plot_quality_and_fmem_swd_bs_vs_tau.py — Combined SW1 + f_mem vs τ
====================================================================
Task 13: Dual-axis plot combining:
  - Left y-axis  (solid lines) : SW1 on BS beam-power profiles
  - Right y-axis (dashed lines): f_mem (%)
  - x-axis : tau (optimizer updates)
  - Same color for the same N

This is the multidimensional channel analogue of the paper's FID + f_mem plot.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_quality_and_fmem_swd_bs_vs_tau.py

    # Specific n_feat:
    python plot_quality_and_fmem_swd_bs_vs_tau.py --n_feat 256
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

SWD_CSV  = Path("results/swd_bs_beamspace_vs_tau/swd_bs_beamspace_vs_tau.csv")
FMEM_CSV = Path("results/fmem_vs_tau/fmem_vs_tau.csv")
OUTPUT_DIR = Path("tau_plots")

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
# Main plotting function
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined(df_swd: pd.DataFrame, df_fmem: pd.DataFrame,
                  n_feat: int, output_dir: Path):
    """
    Dual-axis plot: SW1 (left, solid) and f_mem (right, dashed) vs τ.
    """
    swd_sub = df_swd[df_swd['n_feat'] == n_feat].copy()
    if swd_sub.empty:
        print(f"  No SWD data for n_feat={n_feat}, skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    all_N = sorted(set(swd_sub['N'].unique()) |
                   set(df_fmem['N'].unique() if not df_fmem.empty else []))

    for N in all_N:
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')

        # SW1 (left axis, solid)
        d_swd = swd_sub[swd_sub['N'] == N].sort_values('actual_tau')
        if not d_swd.empty:
            tau_swd = d_swd['actual_tau'].values
            mean_swd = d_swd['SW1_Gen_Test_mean'].values
            err_swd  = d_swd['SW1_Gen_Test_error_2std'].values
            ax1.plot(tau_swd, mean_swd, '-', color=color, linewidth=2.0,
                     marker=marker, markersize=5)
            ax1.fill_between(tau_swd, mean_swd - err_swd, mean_swd + err_swd,
                             color=color, alpha=0.1)

        # f_mem (right axis, dashed)
        d_fmem = df_fmem[df_fmem['N'] == N].sort_values('actual_tau')
        if not d_fmem.empty:
            tau_fmem = d_fmem['actual_tau'].values
            fmem_val = d_fmem['f_mem_k_1_3'].values * 100  # Convert to %
            ax2.plot(tau_fmem, fmem_val, '--', color=color, linewidth=1.8,
                     marker=marker, markersize=4, alpha=0.8)

    ax1.set_xscale('log')
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax1.set_ylabel(r'SW$_1$ on BS Beam-Power Profile', fontsize=20, color='black')
    ax1.tick_params(labelsize=16)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylim(bottom=0)
    ax2.set_ylabel(r'$f_{\mathrm{mem}}$ (%)', fontsize=20, color='gray')
    ax2.tick_params(labelsize=16, colors='gray')
    ax2.spines['right'].set_color('gray')

    # Combined legend
    handles = []
    for N in sorted(all_N):
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')
        handles.append(
            Line2D([0], [0], color=color, marker=marker, linewidth=2.0,
                   linestyle='-', markersize=7, label=f'N={N}'))
    # Style legend entries
    handles.append(
        Line2D([0], [0], color='black', linestyle='-', linewidth=2.0,
               label=r'SW$_1$ (left)'))
    handles.append(
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.8,
               label=r'$f_{\mathrm{mem}}$ (right)'))

    ax1.legend(handles=handles, fontsize=13, loc='upper left', framealpha=0.9)

    suffix = f'_nfeat{n_feat}'
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = output_dir / f'quality_and_fmem_swd_bs_vs_tau{suffix}.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined SW1 + f_mem vs tau plot")
    parser.add_argument("--n_feat", type=int, default=256,
                        help="n_feat for the plot (default 256)")
    parser.add_argument("--swd_csv", type=str, default=None)
    parser.add_argument("--fmem_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    swd_path  = Path(args.swd_csv) if args.swd_csv else SWD_CSV
    fmem_path = Path(args.fmem_csv) if args.fmem_csv else FMEM_CSV
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not swd_path.exists():
        print(f"ERROR: {swd_path} not found. "
              f"Run compute_swd_bs_beamspace_vs_tau.py first.")
        return

    df_swd = pd.read_csv(swd_path)
    print(f"Loaded SWD: {len(df_swd)} rows")

    df_fmem = pd.DataFrame()
    if fmem_path.exists():
        df_fmem = pd.read_csv(fmem_path)
        print(f"Loaded f_mem: {len(df_fmem)} rows")
    else:
        print(f"WARNING: {fmem_path} not found. Plotting SW1 only.")

    print(f"\nPlotting combined SW1 + f_mem for n_feat={args.n_feat} ...")
    plot_combined(df_swd, df_fmem, args.n_feat, output_dir)

    # Also plot for other n_feats if data available
    other_nfeats = [nf for nf in df_swd['n_feat'].unique() if nf != args.n_feat]
    for nf in sorted(other_nfeats):
        print(f"\nPlotting combined for n_feat={nf} ...")
        plot_combined(df_swd, df_fmem, nf, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
