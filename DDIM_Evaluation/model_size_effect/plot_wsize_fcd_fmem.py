"""
plot_wsize_fcd_fmem.py
    — Model-size sweep: FCD(Gen,Test) + f_mem vs τ  (one figure per dataset size N)
================================================================================
Reads results/wsize_fcd_fmem.csv (produced by compute_wsize_fcd_fmem.py) and draws,
for a FIXED dataset size N, one curve per model width W (= n_feat):

  • solid  line (LEFT axis)  = FCD(Gen, Test)     — sample quality (↓, want it low)
                               with a shaded 2σ band over the 5 test folds;
  • dotted line (LEFT axis)  = FCD(Train, Test)   — the real–real floor for THAT N
                               (shared by all W at a given N: same training set);
  • dashed line (RIGHT axis) = f_mem(τ)  in %      — memorization fraction, with a
                               95% bootstrap-CI band.

Color encodes the model width W; line style encodes the quantity — the same visual
language as ../dataset_size_effect/plot_dsize_fcd_fmem.py (there color = N; here
color = W). One separate figure is written PER dataset size N, because the real–real
floor and the meaningful τ range differ between N.

Styling matches ../tau_plots/ (axis labels 20, ticks 20, legend 16, default font,
pdf.fonttype 42).

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/model_size_effect
    python plot_wsize_fcd_fmem.py                    # one figure per N in the CSV
    python plot_wsize_fcd_fmem.py --sizes 1000       # only N = 1000
    python plot_wsize_fcd_fmem.py --widths 64 128    # only a subset of widths
    python plot_wsize_fcd_fmem.py --min_tau 100
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

# Distinct color per model width W.
W_COLORS = {
    64:  '#377eb8',   # blue
    128: '#ff7f00',   # orange
    256: '#4daf4a',   # green
}
W_MARKERS = {64: 'o', 128: 's', 256: 'D'}


def _c(W):
    return W_COLORS.get(W, '#333333')


def _m(W):
    return W_MARKERS.get(W, 'o')


def plot_one_N(df_N, N, widths, out_dir, min_tau):
    """Draw and save the FCD+f_mem figure for a single dataset size N."""
    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()

    floor_val = None
    for W in widths:
        sub = df_N[df_N['W'] == W].sort_values('tau')
        if sub.empty:
            continue
        tau      = sub['tau'].values
        gt       = sub['FCD_Gen_Test'].values
        gt_sd    = sub['FCD_Gen_Test_std'].values
        fmem     = sub['f_mem'].values * 100.0   # fraction → percentage (%)
        if 'f_mem_ci_low' in sub and 'f_mem_ci_high' in sub:
            fmem_lo = sub['f_mem_ci_low'].values * 100.0
            fmem_hi = sub['f_mem_ci_high'].values * 100.0
        else:
            fmem_lo = fmem_hi = None
        floor    = sub['FCD_Train_Test'].values
        floor_sd = sub['FCD_Train_Test_std'].values

        # FCD(Gen,Test): solid + 2σ band
        axL.plot(tau, gt, '-', color=_c(W), lw=2.2, marker=_m(W), ms=5, alpha=0.95)
        if np.any(gt_sd > 0):
            axL.fill_between(tau, gt - 2 * gt_sd, gt + 2 * gt_sd,
                             color=_c(W), alpha=0.12, linewidth=0)

        # Real–real floor for THIS N (dotted). It is shared across W (same train
        # set), so draw it once in a neutral color.
        if len(floor):
            floor_val = (floor[0], floor_sd[0])

        # f_mem: dashed on the right axis, with a 95% bootstrap CI band
        axR.plot(tau, fmem, '--', color=_c(W), lw=1.8, alpha=0.85)
        if fmem_lo is not None and np.any(fmem_hi - fmem_lo > 0):
            axR.fill_between(tau, fmem_lo, fmem_hi,
                             color=_c(W), alpha=0.12, linewidth=0)

    # Single shared real–real floor for this N (dotted black + faint band).
    if floor_val is not None:
        f0, f0sd = floor_val
        axL.axhline(f0, color='k', ls=':', lw=1.6, alpha=0.9)
        if f0sd > 0:
            axL.axhspan(f0 - 2 * f0sd, f0 + 2 * f0sd, color='k', alpha=0.06,
                        linewidth=0)

    axL.set_xscale('log')
    axL.set_ylim(bottom=0)
    axR.set_ylim(bottom=0)
    axL.set_xlabel(r'$\tau$', fontsize=20)
    axL.set_ylabel('FCD', fontsize=20)
    axR.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$  (%)', fontsize=20)
    axL.tick_params(labelsize=20)
    axR.tick_params(labelsize=20)
    axL.grid(True, alpha=0.3)

    # Legend 1: model width (color).  Legend 2: quantity (line style).
    present = [W for W in widths if not df_N[df_N['W'] == W].empty]
    size_handles = [Line2D([0], [0], color=_c(W), marker=_m(W), lw=2.2, ls='-',
                           ms=7, label=f'W = {W}')
                    for W in present]
    style_handles = [
        Line2D([0], [0], color='k', ls='-',  lw=2.2, label='Gen–Test'),
        Line2D([0], [0], color='k', ls=':',  lw=1.6, label='Train–Test'),
        Line2D([0], [0], color='k', ls='--', lw=1.8, label=r'$f_{\mathrm{mem}}$'),
    ]
    leg1 = axL.legend(handles=size_handles, fontsize=16, loc='upper left',
                      framealpha=0.9, bbox_to_anchor=(0.01, 0.99))
    axL.add_artist(leg1)

    # Second legend directly below the first, tight gap, using leg1's real height.
    fig.canvas.draw()
    bb = leg1.get_window_extent().transformed(axL.transAxes.inverted())
    gap = 0.02
    axL.legend(handles=style_handles, fontsize=16, loc='upper left',
               framealpha=0.9, bbox_to_anchor=(0.01, bb.y0 - gap))

    plt.tight_layout()
    fname = f'wsize_fcd_fmem_N{N}'
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(
        description="Plot FCD(Gen,Test)+f_mem vs τ for the model-size sweep "
                    "(one figure per N; color = width W)")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Drop τ below this (removes the τ=0 random-init spike)")
    ap.add_argument("--sizes", type=int, nargs='+', default=None,
                    help="Subset of dataset sizes N to plot (default: all in CSV)")
    ap.add_argument("--widths", type=int, nargs='+', default=None,
                    help="Subset of widths W to include (default: all in CSV)")
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'wsize_fcd_fmem.csv')
    out_dir  = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Run: python compute_wsize_fcd_fmem.py")

    df = pd.read_csv(csv_path)
    df = df[df['tau'] >= args.min_tau].sort_values(['N', 'W', 'tau'])

    sizes  = args.sizes  if args.sizes  is not None else sorted(df['N'].unique())
    widths = args.widths if args.widths is not None else sorted(df['W'].unique())
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Dataset sizes: {sizes}")
    print(f"  Widths       : {widths}")

    for N in sizes:
        df_N = df[df['N'] == N]
        if df_N.empty:
            print(f"  WARNING: no rows for N={N}, skipping.")
            continue
        print(f"\n  Figure for N = {N}")
        plot_one_N(df_N, N, widths, out_dir, args.min_tau)

    print("Done.")


if __name__ == "__main__":
    main()
