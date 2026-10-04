"""
plot_quality_and_fmem_bs_bfd_vs_tau.py — Combined BS-BFD + f_mem vs τ
======================================================================
Task 14: Dual-axis plot combining:
  - Left y-axis  (solid lines) : BS-Beamspace Fréchet Distance
  - Right y-axis (dashed lines): f_mem (%)
  - x-axis : tau (optimizer updates)
  - Same color for the same N

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_quality_and_fmem_bs_bfd_vs_tau.py

    # Specific n_feat:
    python plot_quality_and_fmem_bs_bfd_vs_tau.py --n_feat 256
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

BFD_CSV    = Path("results/bs_bfd_vs_tau/bs_bfd_vs_tau.csv")
FMEM_CSV   = Path("results/fmem_vs_tau/fmem_vs_tau.csv")
OUTPUT_DIR = Path("results/bs_bfd_vs_tau/figures")

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
# Combined plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined(df_bfd: pd.DataFrame, df_fmem: pd.DataFrame,
                  n_feat: int, output_dir: Path):
    """Dual-axis: BS-BFD (left, solid) and f_mem (right, dashed) vs τ."""
    bfd_sub = df_bfd[df_bfd['n_feat'] == n_feat].copy()
    if bfd_sub.empty:
        print(f"  No BFD data for n_feat={n_feat}, skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    all_N = sorted(set(bfd_sub['N'].unique()) |
                   set(df_fmem['N'].unique() if not df_fmem.empty else []))

    for N in all_N:
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')

        # BS-BFD (left axis, solid)
        d_bfd = bfd_sub[bfd_sub['N'] == N].sort_values('actual_tau')
        if not d_bfd.empty:
            tau = d_bfd['actual_tau'].values
            mean = d_bfd['BS_BFD_Gen_Test_mean'].values
            err  = d_bfd['BS_BFD_Gen_Test_error_2std'].values
            ax1.plot(tau, mean, '-', color=color, linewidth=2.0,
                     marker=marker, markersize=5)
            ax1.fill_between(tau, np.maximum(mean - err, 0), mean + err,
                             color=color, alpha=0.1)

        # f_mem (right axis, dashed)
        d_fmem = df_fmem[df_fmem['N'] == N].sort_values('actual_tau')
        if not d_fmem.empty:
            tau_fmem = d_fmem['actual_tau'].values
            fmem_val = d_fmem['f_mem_k_1_3'].values * 100
            ax2.plot(tau_fmem, fmem_val, '--', color=color, linewidth=1.8,
                     marker=marker, markersize=4, alpha=0.8)

    ax1.set_xscale('log')
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax1.set_ylabel('BS-Beamspace Fréchet Distance', fontsize=20, color='black')
    ax1.tick_params(labelsize=16)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylim(bottom=0)
    ax2.set_ylabel(r'$f_{\mathrm{mem}}$ (%)', fontsize=20, color='gray')
    ax2.tick_params(labelsize=16, colors='gray')
    ax2.spines['right'].set_color('gray')

    handles = []
    for N in sorted(all_N):
        color = SIZE_COLORS.get(N, '#333333')
        marker = SIZE_MARKERS.get(N, 'o')
        handles.append(
            Line2D([0], [0], color=color, marker=marker, linewidth=2.0,
                   linestyle='-', markersize=7, label=f'N={N}'))
    handles.append(
        Line2D([0], [0], color='black', linestyle='-', linewidth=2.0,
               label='BS-BFD (left)'))
    handles.append(
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.8,
               label=r'$f_{\mathrm{mem}}$ (right)'))
    ax1.legend(handles=handles, fontsize=13, loc='upper left', framealpha=0.9)

    suffix = f'_nfeat{n_feat}'
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        path = output_dir / f'quality_and_fmem_bs_bfd_vs_tau{suffix}.{ext}'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined BS-BFD + f_mem vs tau plot")
    parser.add_argument("--n_feat", type=int, default=256)
    parser.add_argument("--bfd_csv", type=str, default=None)
    parser.add_argument("--fmem_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    bfd_path  = Path(args.bfd_csv) if args.bfd_csv else BFD_CSV
    fmem_path = Path(args.fmem_csv) if args.fmem_csv else FMEM_CSV
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bfd_path.exists():
        print(f"ERROR: {bfd_path} not found. Run compute_bs_bfd_vs_tau.py first.")
        return

    df_bfd = pd.read_csv(bfd_path)
    print(f"Loaded BFD: {len(df_bfd)} rows")

    df_fmem = pd.DataFrame()
    if fmem_path.exists():
        df_fmem = pd.read_csv(fmem_path)
        print(f"Loaded f_mem: {len(df_fmem)} rows")
    else:
        print(f"WARNING: {fmem_path} not found. Plotting BS-BFD only.")

    print(f"\nPlotting combined for n_feat={args.n_feat} ...")
    plot_combined(df_bfd, df_fmem, args.n_feat, output_dir)

    for nf in sorted(df_bfd['n_feat'].unique()):
        if nf != args.n_feat:
            print(f"\nPlotting combined for n_feat={nf} ...")
            plot_combined(df_bfd, df_fmem, nf, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
