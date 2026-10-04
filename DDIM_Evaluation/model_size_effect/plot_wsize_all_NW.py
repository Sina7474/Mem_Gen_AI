"""
plot_wsize_all_NW.py
    — Model-size sweep: combined figures over ALL (N, W) on a single axis
================================================================================
Reads results/wsize_fcd_fmem.csv (produced by compute_wsize_fcd_fmem.py) and draws
two overview figures in which every (dataset size N, model width W) run is a single
curve on ONE axis, so the whole sweep can be compared at a glance:

  1) wsize_all_fcd_vs_tau
       y = FCD(Gen, Test)     (log scale, ↓ = better sample quality)
       x = τ  (optimizer updates, log scale)
       + the real–real floor FCD(Train, Test) for each N (dotted, one per N).

  2) wsize_all_fmem_norm_vs_tau
       y = f_mem(τ) / f_mem(τ_max)   — each run's memorization fraction rescaled by
           its OWN final value, so every curve ends at 1.0 and the *timing / shape*
           of the memorization onset is isolated from the absolute level reached.
       x = τ  (optimizer updates, log scale)

  3) wsize_all_fcd_vs_tauW
       Same FCD(Gen, Test) curves as (1), but the x-axis is rescaled to τ·W
       (optimizer updates × model width). This tests whether the compute-like
       product τ·W — how much total width-weighted optimization each run has seen —
       aligns the sample-quality dynamics across different widths.

  4) wsize_all_fmem_norm_vs_tauW_over_N
       Same normalized memorization curves as (2), but the x-axis is rescaled to
       τ·W / N. Combining the width factor W with the per-sample clock 1/N tests
       whether memorization onset collapses onto the single combined variable
       τ·W/N (width-weighted optimizer steps per training sample).

Encoding (all figures):
  • color      = model width W   {64, 128, 256}   (same palette as plot_wsize_fcd_fmem.py)
  • line style = dataset size N   (solid = N=200, dashed = N=1000)
  • marker     = dataset size N   (o = N=200, s = N=1000)

Two legends: one maps color → W, the other maps line-style/marker → N.

Styling is sized for the three-column conference-paper layout (axis labels 30,
ticks 26, legend 24, default font, pdf.fonttype 42).

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/model_size_effect
    python plot_wsize_all_NW.py
    python plot_wsize_all_NW.py --widths 64 128
    python plot_wsize_all_NW.py --sizes 200 1000 --min_tau 100
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Figure 3 places three plots across a two-column page.  These source-font
# sizes are intentionally large so that labels remain legible after each PDF
# is reduced to roughly one third of the text width.
AXIS_LABEL_FONTSIZE = 30
TICK_LABEL_FONTSIZE = 26
LEGEND_FONTSIZE = 24

# Distinct color per model width W (same palette as plot_wsize_fcd_fmem.py).
W_COLORS = {
    64:  '#377eb8',   # blue
    128: '#ff7f00',   # orange
    256: '#4daf4a',   # green
}
# Line style + marker per dataset size N.
N_STYLE  = {200: '-', 1000: '--', 500: '-.', 2000: (0, (3, 1, 1, 1)), 4000: ':'}
N_MARKER = {200: 'o', 1000: 's', 500: 'D', 2000: '^', 4000: 'P'}
# Neutral color for the per-N real–real floor lines (keyed by N so they differ).
FLOOR_LS = {200: (0, (1, 1)), 500: (0, (1, 2)), 1000: (0, (1, 3))}


def _c(W):
    return W_COLORS.get(W, '#333333')


def _ls(N):
    return N_STYLE.get(N, '-')


def _mk(N):
    return N_MARKER.get(N, 'o')


def _legends(ax, present_W, present_N, loc='upper left'):
    """Two legends: color → W, line-style/marker → N (stacked at *loc*)."""
    anchor_x = 0.01 if 'left' in loc else 0.99
    ha = 'left' if 'left' in loc else 'right'
    w_handles = [Line2D([0], [0], color=_c(W), lw=2.8, ls='-', label=f'W = {W}')
                 for W in present_W]
    n_handles = [Line2D([0], [0], color='k', lw=2.6, ls=_ls(N), marker=_mk(N),
                        ms=9, label=f'N = {N}')
                 for N in present_N]
    legend_kwargs = dict(
        fontsize=LEGEND_FONTSIZE,
        loc=loc,
        framealpha=0.9,
        borderpad=0.35,
        labelspacing=0.25,
        handlelength=1.7,
        handletextpad=0.45,
    )
    leg1 = ax.legend(handles=w_handles, bbox_to_anchor=(anchor_x, 0.99),
                     **legend_kwargs)
    ax.add_artist(leg1)
    ax.figure.canvas.draw()
    bb = leg1.get_window_extent().transformed(ax.transAxes.inverted())
    ax.legend(handles=n_handles, bbox_to_anchor=(anchor_x, bb.y0 - 0.02),
              **legend_kwargs)


def plot_all_fcd(df, sizes, widths, out_dir, fname_suffix):
    """FCD(Gen,Test) vs τ for every (N, W) on one log–log axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    floors = {}          # N -> (floor, floor_std)
    present_W, present_N = set(), set()

    for N in sizes:
        for W in widths:
            sub = df[(df['N'] == N) & (df['W'] == W)].sort_values('tau')
            if sub.empty:
                continue
            present_W.add(W); present_N.add(N)
            tau   = sub['tau'].values.astype(float)
            gt    = sub['FCD_Gen_Test'].values.astype(float)
            gt_sd = sub['FCD_Gen_Test_std'].values.astype(float)
            ax.plot(tau, gt, ls=_ls(N), color=_c(W), lw=2.2, marker=_mk(N),
                    ms=5, alpha=0.95)
            if np.any(gt_sd > 0):
                ax.fill_between(tau, gt - 2 * gt_sd, gt + 2 * gt_sd,
                                color=_c(W), alpha=0.10, linewidth=0)
            f = sub['FCD_Train_Test'].values
            fsd = sub['FCD_Train_Test_std'].values
            if len(f):
                floors[N] = (float(f[0]), float(fsd[0]))

    # One real–real floor per N (dotted, neutral color, distinct dash pattern).
    for N, (f0, f0sd) in floors.items():
        ax.axhline(f0, color='0.4', ls=FLOOR_LS.get(N, ':'), lw=1.5, alpha=0.9)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau$', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('FCD', fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, which='both', alpha=0.3)

    _legends(ax, sorted(present_W), sorted(present_N), loc='upper right')
    plt.tight_layout()
    fname = f'wsize_all_fcd_vs_tau{fname_suffix}'
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


def plot_all_fmem_norm(df, sizes, widths, out_dir, fname_suffix):
    """f_mem(τ)/f_mem(τ_max) vs τ for every (N, W) on one semilog-x axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    present_W, present_N = set(), set()

    print("\n  N     W    τ_max      f_mem(τ_max)")
    print("  ----  ---  ---------  ------------")
    for N in sizes:
        for W in widths:
            sub = df[(df['N'] == N) & (df['W'] == W)].sort_values('tau')
            if sub.empty:
                continue
            tau  = sub['tau'].values.astype(float)
            fmem = sub['f_mem'].values.astype(float)
            fmem_tau_max = fmem[-1]
            print(f"  {N:<4}  {W:<3}  {tau[-1]:>9.0f}  {fmem_tau_max:>10.4f}")
            if fmem_tau_max <= 0:
                print(f"    WARNING: f_mem(τ_max)=0 for N={N},W={W}; skipping.")
                continue
            present_W.add(W); present_N.add(N)
            y = fmem / fmem_tau_max
            ax.plot(tau, y, ls=_ls(N), color=_c(W), lw=2.2, marker=_mk(N),
                    ms=5, alpha=0.95)
            if 'f_mem_ci_low' in sub and 'f_mem_ci_high' in sub:
                y_lo = sub['f_mem_ci_low'].values.astype(float) / fmem_tau_max
                y_hi = sub['f_mem_ci_high'].values.astype(float) / fmem_tau_max
                if np.any(y_hi - y_lo > 0):
                    ax.fill_between(tau, y_lo, y_hi, color=_c(W), alpha=0.10,
                                    linewidth=0)

    ax.axhline(1.0, color='0.6', ls=':', lw=1.0, alpha=0.8)
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau$', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r'$f_{\mathrm{mem}}(\tau)\,/\,f_{\mathrm{mem}}(\tau_{\max})$',
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)

    _legends(ax, sorted(present_W), sorted(present_N))
    plt.tight_layout()
    fname = f'wsize_all_fmem_norm_vs_tau{fname_suffix}'
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


def plot_all_fcd_vs_tauW(df, sizes, widths, out_dir, fname_suffix):
    """FCD(Gen,Test) vs τ·W for every (N, W) on one log–log axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    floors = {}          # N -> (floor, floor_std)
    present_W, present_N = set(), set()

    for N in sizes:
        for W in widths:
            sub = df[(df['N'] == N) & (df['W'] == W)].sort_values('tau')
            if sub.empty:
                continue
            present_W.add(W); present_N.add(N)
            tau   = sub['tau'].values.astype(float)
            x     = tau * float(W)                 # τ·W
            gt    = sub['FCD_Gen_Test'].values.astype(float)
            gt_sd = sub['FCD_Gen_Test_std'].values.astype(float)
            ax.plot(x, gt, ls=_ls(N), color=_c(W), lw=2.2, marker=_mk(N),
                    ms=5, alpha=0.95)
            if np.any(gt_sd > 0):
                ax.fill_between(x, gt - 2 * gt_sd, gt + 2 * gt_sd,
                                color=_c(W), alpha=0.10, linewidth=0)
            f = sub['FCD_Train_Test'].values
            fsd = sub['FCD_Train_Test_std'].values
            if len(f):
                floors[N] = (float(f[0]), float(fsd[0]))

    for N, (f0, f0sd) in floors.items():
        ax.axhline(f0, color='0.4', ls=FLOOR_LS.get(N, ':'), lw=1.5, alpha=0.9)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau\,W$', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('FCD', fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, which='both', alpha=0.3)

    # Figure 3 uses a shared visual encoding across panels.  Its legends are
    # shown only in panel (a), leaving this panel uncluttered.
    plt.tight_layout()
    fname = f'wsize_all_fcd_vs_tauW{fname_suffix}'
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


def plot_all_fmem_norm_vs_tauW_over_N(df, sizes, widths, out_dir, fname_suffix):
    """f_mem(τ)/f_mem(τ_max) vs τ·W/N for every (N, W) on one semilog-x axis."""
    fig, ax = plt.subplots(figsize=(10, 6))
    present_W, present_N = set(), set()

    for N in sizes:
        for W in widths:
            sub = df[(df['N'] == N) & (df['W'] == W)].sort_values('tau')
            if sub.empty:
                continue
            tau  = sub['tau'].values.astype(float)
            fmem = sub['f_mem'].values.astype(float)
            fmem_tau_max = fmem[-1]
            if fmem_tau_max <= 0:
                continue
            present_W.add(W); present_N.add(N)
            x = tau * float(W) / float(N)          # τ·W / N
            y = fmem / fmem_tau_max
            ax.plot(x, y, ls=_ls(N), color=_c(W), lw=2.2, marker=_mk(N),
                    ms=5, alpha=0.95)
            if 'f_mem_ci_low' in sub and 'f_mem_ci_high' in sub:
                y_lo = sub['f_mem_ci_low'].values.astype(float) / fmem_tau_max
                y_hi = sub['f_mem_ci_high'].values.astype(float) / fmem_tau_max
                if np.any(y_hi - y_lo > 0):
                    ax.fill_between(x, y_lo, y_hi, color=_c(W), alpha=0.10,
                                    linewidth=0)

    ax.axhline(1.0, color='0.6', ls=':', lw=1.0, alpha=0.8)
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau\,W / N$',
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r'$f_{\mathrm{mem}}(\tau)\,/\,f_{\mathrm{mem}}(\tau_{\max})$',
                  fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)

    _legends(ax, sorted(present_W), sorted(present_N))
    plt.tight_layout()
    fname = f'wsize_all_fmem_norm_vs_tauW_over_N{fname_suffix}'
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Combined model-size-sweep figures over ALL (N, W): "
                    "FCD vs τ and normalized f_mem vs τ")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Drop τ below this (removes the τ=0 random-init spike)")
    ap.add_argument("--sizes", type=int, nargs='+', default=None,
                    help="Subset of dataset sizes N (default: all in CSV)")
    ap.add_argument("--widths", type=int, nargs='+', default=None,
                    help="Subset of widths W (default: all in CSV)")
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument(
        "--plot_group", choices=("all", "fmem"), default="all",
        help="Generate all four figures (default) or only the two normalized "
             "f_mem figures. The latter permits an alternate f_mem table to "
             "replace only memorization plots without touching FCD plots.")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'wsize_fcd_fmem.csv')
    out_dir  = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\nRun: python compute_wsize_fcd_fmem.py")

    df_all = pd.read_csv(csv_path)
    df = df_all[df_all['tau'] >= args.min_tau].sort_values(['N', 'W', 'tau'])

    all_sizes  = sorted(df_all['N'].unique())
    all_widths = sorted(df_all['W'].unique())
    sizes  = args.sizes  if args.sizes  is not None else all_sizes
    widths = args.widths if args.widths is not None else all_widths
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Dataset sizes: {sizes}")
    print(f"  Widths       : {widths}")

    suffix = ''
    is_subset = (args.sizes is not None and sorted(sizes) != all_sizes) or \
                (args.widths is not None and sorted(widths) != all_widths)
    if is_subset:
        suffix = '_N' + '_'.join(str(n) for n in sizes) + '_W' + '_'.join(str(w) for w in widths)

    if args.plot_group == "all":
        print("\n[1/4] FCD(Gen,Test) vs τ  (all N, W)")
        plot_all_fcd(df, sizes, widths, out_dir, suffix)
    print("\n[2/4] f_mem(τ)/f_mem(τ_max) vs τ  (all N, W)")
    plot_all_fmem_norm(df, sizes, widths, out_dir, suffix)
    if args.plot_group == "all":
        print("\n[3/4] FCD(Gen,Test) vs τ·W  (all N, W)")
        plot_all_fcd_vs_tauW(df, sizes, widths, out_dir, suffix)
    print("\n[4/4] f_mem(τ)/f_mem(τ_max) vs τ·W/N  (all N, W)")
    plot_all_fmem_norm_vs_tauW_over_N(df, sizes, widths, out_dir, suffix)
    print("Done.")


if __name__ == "__main__":
    main()
