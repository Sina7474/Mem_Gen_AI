"""
plot_dsize_fcd_fmem_28GHz_only.py
    — 28 GHz LoS only: FCD(Gen,Test) + f_mem vs τ (dual-axis figure)
================================================================================
Same visual language as ../dataset_size_effect/plot_dsize_fcd_fmem.py but only
for the 28 GHz LoS scene. Draws N = 100, 200, 500, 1000, 2000 by default.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect_28GHz_LoS
    python plot_dsize_fcd_fmem_28GHz_only.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

N_COLORS = {
    100:  '#e41a1c',   # red
    200:  '#377eb8',   # blue
    500:  '#ff7f00',   # orange
    1000: '#4daf4a',   # green
    2000: '#984ea3',   # purple
}
N_MARKERS = {100: 'v', 200: 'o', 500: 's', 1000: 'D', 2000: '^'}


def _c(N):
    return N_COLORS.get(N, '#333333')


def _m(N):
    return N_MARKERS.get(N, 'o')


def main():
    ap = argparse.ArgumentParser(
        description="28 GHz LoS only: FCD + f_mem vs τ for the dataset-size sweep")
    ap.add_argument("--min_tau", type=int, default=100,
                    help="Drop τ below this (default: 100, removes τ=0 spike)")
    ap.add_argument("--sizes", type=int, nargs='+', default=[100, 200, 500, 1000, 2000],
                    help="Dataset sizes N to plot")
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'dsize_fcd_fmem_28GHz_LoS.csv')
    out_dir  = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Run: python compute_dsize_fmem_28GHz_LoS.py --sizes 100 200 500 1000 2000")

    df = pd.read_csv(csv_path)
    df = df[df['tau'] >= args.min_tau].sort_values(['N', 'tau'])
    df = df[df['N'].isin(args.sizes)]
    sizes = sorted(df['N'].unique())
    missing = sorted(set(args.sizes) - set(sizes))
    if missing:
        print(f"  WARNING: sizes not in CSV, ignored: {missing}")
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Dataset sizes: {sizes}")

    fname = 'dsize_fcd_fmem_28GHz_LoS_N' + '_'.join(str(n) for n in sizes)

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()

    for N in sizes:
        sub = df[df['N'] == N].sort_values('tau')
        tau      = sub['tau'].values
        gt       = sub['FCD_Gen_Test'].values
        gt_sd    = sub['FCD_Gen_Test_std'].values
        fmem     = sub['f_mem'].values * 100.0
        if 'f_mem_ci_low' in sub.columns and 'f_mem_ci_high' in sub.columns:
            fmem_lo = sub['f_mem_ci_low'].values * 100.0
            fmem_hi = sub['f_mem_ci_high'].values * 100.0
        else:
            fmem_lo = fmem_hi = None
        floor    = sub['FCD_Train_Test'].values
        floor_sd = sub['FCD_Train_Test_std'].values

        # FCD(Gen,Test): solid + 2σ band
        axL.plot(tau, gt, '-', color=_c(N), lw=2.2, marker=_m(N), ms=5, alpha=0.95)
        if np.any(gt_sd > 0):
            axL.fill_between(tau, gt - 2 * gt_sd, gt + 2 * gt_sd,
                             color=_c(N), alpha=0.12, linewidth=0)

        # Real–real floor (dotted, same color)
        if len(floor):
            axL.axhline(floor[0], color=_c(N), ls=':', lw=1.4, alpha=0.9)
            if floor_sd[0] > 0:
                axL.axhspan(floor[0] - 2 * floor_sd[0], floor[0] + 2 * floor_sd[0],
                            color=_c(N), alpha=0.05, linewidth=0)

        # f_mem: dashed on the right axis
        axR.plot(tau, fmem, '--', color=_c(N), lw=1.8, marker=_m(N), ms=5, alpha=0.85)
        if fmem_lo is not None and np.any(fmem_hi - fmem_lo > 0):
            axR.fill_between(tau, fmem_lo, fmem_hi,
                             color=_c(N), alpha=0.12, linewidth=0)

    axL.set_xscale('log')
    axL.set_ylim(bottom=0)
    axR.set_ylim(0, 100)
    axR.yaxis.set_major_locator(MultipleLocator(10))
    axL.set_xlabel(r'$\tau$', fontsize=20)
    axL.set_ylabel('FCD', fontsize=20)
    axR.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$  (%)', fontsize=20)
    axL.tick_params(labelsize=20)
    axR.tick_params(labelsize=20)
    axL.grid(True, alpha=0.3)

    # Legend 1: dataset size (color)
    size_handles = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2, ls='-',
                           ms=7, label=f'N = {N}')
                    for N in sizes]
    # Legend 2: quantity (line style)
    style_handles = [
        Line2D([0], [0], color='k', ls='-',  lw=2.2, label='FCD (Gen–Test)'),
        Line2D([0], [0], color='k', ls=':',  lw=1.4, label='FCD (Train–Test)'),
        Line2D([0], [0], color='k', ls='--', lw=1.8, label=r'$f_{\mathrm{mem}}$'),
    ]
    leg1 = axL.legend(handles=size_handles, fontsize=16, loc='upper center',
                      framealpha=0.9, bbox_to_anchor=(0.35, 0.99))
    axL.add_artist(leg1)

    fig.canvas.draw()
    bb = leg1.get_window_extent().transformed(axL.transAxes.inverted())
    gap = 0.02
    leg2_top = bb.y0 - gap
    axL.legend(handles=style_handles, fontsize=16, loc='upper center',
               framealpha=0.9, bbox_to_anchor=(0.35, leg2_top))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
