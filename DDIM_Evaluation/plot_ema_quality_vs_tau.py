"""
plot_ema_quality_vs_tau.py
    — Plot EMA quality vs τ with the Train–Test benchmark (window becomes visible)
================================================================================
Reads the CSV from compute_ema_quality_vs_tau.py and produces, for BOTH quality
metrics, per-N figures where you can *see the generalization window*:

  • solid  = W1/BFD (Gen, Test)     → quality  (want it LOW)
  • dashed = W1/BFD (Gen, Train)    → memorization probe (drops toward 0 = bad)
  • dotted horizontal = (Train, Test) benchmark → the real–real floor

The generalization window is the τ-band where the Gen–Test curve has reached the
benchmark floor while the Gen–Train curve has NOT yet collapsed below it.

Figures:
  1. w1_quality_vs_tau_<w>.png/pdf        — effective-rank W1, all N overlaid
  2. bfd_quality_vs_tau_<w>.png/pdf       — BS-BFD, all N overlaid
  3. window_panels_<w>.png/pdf            — one panel per N, both metrics + window

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_ema_quality_vs_tau.py                 # weights=ema (default)
    python plot_ema_quality_vs_tau.py --weights raw
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

COLORS = {200: '#ff7f00', 1000: '#4daf4a', 4000: '#d62728',
          100: '#e41a1c', 500: '#984ea3', 2000: '#377eb8', 8000: '#a65628'}
MARKERS = {200: 's', 1000: 'D', 4000: 'P',
           100: 'o', 500: '^', 2000: 'v', 8000: 'X'}


def _c(N):
    return COLORS.get(N, '#333333')


def _m(N):
    return MARKERS.get(N, 'o')


def _std_col(df, base):
    """Return the std column for `base` if present, else zeros."""
    col = base + '_std'
    if col in df.columns:
        return df[col].values
    return np.zeros(len(df))


def plot_metric_overlay(df, gt_col, gtr_col, tt_col, ylabel, fname, output_dir,
                        weights, min_tau=0):
    """One figure, all N overlaid: Gen-Test (solid), Gen-Train (dashed), benchmark (dotted)."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for N in sorted(df['N'].unique()):
        sub = df[df['N'] == N].sort_values('tau')
        sub = sub[sub['tau'] >= min_tau]
        tau = sub['tau'].values
        gt = sub[gt_col].values
        gt_sd = _std_col(sub, gt_col)
        ax.plot(tau, gt, '-', color=_c(N), lw=2.2, marker=_m(N), ms=5)
        if np.any(gt_sd > 0):
            ax.fill_between(tau, gt - 2 * gt_sd, gt + 2 * gt_sd,
                            color=_c(N), alpha=0.15, linewidth=0)
        ax.plot(tau, sub[gtr_col].values, '--', color=_c(N), lw=1.6, alpha=0.8)
        bench = sub[tt_col].values
        bench_sd = _std_col(sub, tt_col)
        if len(bench):
            ax.axhline(bench[0], color=_c(N), ls=':', lw=1.4, alpha=0.9)
            if bench_sd[0] > 0:
                ax.axhspan(bench[0] - 2 * bench_sd[0], bench[0] + 2 * bench_sd[0],
                           color=_c(N), alpha=0.06, linewidth=0)

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=16)

    size_handles = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2,
                           ls='-', ms=7, label=f'N={N}')
                    for N in sorted(df['N'].unique())]
    style_handles = [
        Line2D([0], [0], color='k', ls='-',  lw=2.2, label='Gen–Test (quality)'),
        Line2D([0], [0], color='k', ls='--', lw=1.6, label='Gen–Train (memorization)'),
        Line2D([0], [0], color='k', ls=':',  lw=1.4, label='Train–Test (benchmark)'),
    ]
    leg1 = ax.legend(handles=size_handles, fontsize=13, loc='upper right',
                     framealpha=0.9, title='dataset size')
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, fontsize=12, loc='lower left', framealpha=0.9)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'{fname}_{weights}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


def plot_window_panels(df, output_dir, weights, min_tau=0):
    """One row of panels per N: left = W1, right = BS-BFD, with shaded window."""
    Ns = sorted(df['N'].unique())
    fig, axes = plt.subplots(len(Ns), 2, figsize=(15, 4.4 * len(Ns)),
                             squeeze=False)

    for r, N in enumerate(Ns):
        sub = df[df['N'] == N].sort_values('tau')
        sub = sub[sub['tau'] >= min_tau]
        tau = sub['tau'].values

        for c, (gt, gtr, tt, ylabel, title) in enumerate([
            ('W1_Gen_Test', 'W1_Gen_Train', 'W1_Train_Test',
             r'$W_1$ effective rank', 'Effective-rank Wasserstein'),
            ('BFD_Gen_Test', 'BFD_Gen_Train', 'BFD_Train_Test',
             'BS-BFD', 'BS-Beamspace Fréchet (channel FID)'),
        ]):
            ax = axes[r][c]
            gen_test  = sub[gt].values
            gen_train = sub[gtr].values
            gt_sd = _std_col(sub, gt)
            bench = sub[tt].values[0] if len(sub) else np.nan
            bench_sd = _std_col(sub, tt)[0] if len(sub) else 0.0

            ax.plot(tau, gen_test,  '-',  color=_c(N), lw=2.4,
                    marker=_m(N), ms=5, label='Gen–Test (quality)')
            if np.any(gt_sd > 0):
                ax.fill_between(tau, gen_test - 2 * gt_sd, gen_test + 2 * gt_sd,
                                color=_c(N), alpha=0.15, linewidth=0)
            ax.plot(tau, gen_train, '--', color='#555555', lw=1.8,
                    label='Gen–Train (memorization)')
            ax.axhline(bench, color='black', ls=':', lw=1.6,
                       label='Train–Test benchmark')
            if bench_sd > 0:
                ax.axhspan(bench - 2 * bench_sd, bench + 2 * bench_sd,
                           color='black', alpha=0.06, linewidth=0)

            # ── Shade the estimated generalization window ────────────────
            #   quality reached (Gen-Test within 15% above floor) AND
            #   not yet memorizing (Gen-Train still >= 0.5 * floor)
            tol = 1.15 * bench
            reached = gen_test <= tol
            not_mem = gen_train >= 0.5 * bench
            good = reached & not_mem
            if good.any():
                gtau = tau[good]
                ax.axvspan(gtau.min(), gtau.max(), color='green', alpha=0.10,
                           label='approx. window')

            ax.set_xscale('log')
            ax.set_ylim(bottom=0)
            ax.set_xlabel(r'$\tau$ (updates)', fontsize=15)
            ax.set_ylabel(ylabel, fontsize=15)
            ax.set_title(f'N={N} — {title}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=13)
            ax.legend(fontsize=11, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'window_panels_{weights}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Plot EMA quality vs τ with Train-Test benchmark")
    ap.add_argument("--weights", choices=['ema', 'raw'], default='ema')
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--min_tau", type=int, default=0,
                    help="Drop τ below this (hide the warm-up regime, e.g. 1000)")
    ap.add_argument("--output_dir", type=str,
                    default="results/ema_quality_vs_tau/figures")
    args = ap.parse_args()

    csv_path = args.csv_path or (
        f"results/ema_quality_vs_tau/ema_quality_vs_tau_{args.weights}.csv")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 64)
    print("PLOT EMA quality vs τ  (with Train–Test benchmark)")
    print("=" * 64)
    print(f"  weights: {args.weights}")
    print(f"  CSV    : {csv_path}")
    print(f"  min_tau: {args.min_tau}")
    print(f"  output : {args.output_dir}")

    if not os.path.exists(csv_path):
        print(f"\nERROR: CSV not found: {csv_path}")
        print("Run compute_ema_quality_vs_tau.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    print(f"\n  Loaded {len(df)} rows, N = {sorted(df['N'].unique().tolist())}\n")

    plot_metric_overlay(
        df, 'W1_Gen_Test', 'W1_Gen_Train', 'W1_Train_Test',
        r'$W_1$(Gen, Test) on Effective Rank',
        'w1_quality_vs_tau', args.output_dir, args.weights, args.min_tau)
    plot_metric_overlay(
        df, 'BFD_Gen_Test', 'BFD_Gen_Train', 'BFD_Train_Test',
        'BS-BFD (channel FID)',
        'bfd_quality_vs_tau', args.output_dir, args.weights, args.min_tau)
    plot_window_panels(df, args.output_dir, args.weights, args.min_tau)

    print(f"\nDone! Figures in: {args.output_dir}")
