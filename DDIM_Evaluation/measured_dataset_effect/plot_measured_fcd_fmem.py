"""
plot_measured_fcd_fmem.py
    — MEASURED dataset: FCD(Gen,Test) + f_mem vs τ   (Figure-2-left visual language)
================================================================================
Reads results/measured_fcd_fmem.csv (from compute_measured_fcd_fmem.py) and draws,
per dataset size N:

  • solid  line (LEFT axis)  = FCD(Gen, Test)     — sample quality (↓) + 2σ band
  • dotted line (LEFT axis)  = FCD(Train, Test)   — the real–real floor for THAT N
  • dashed line (RIGHT axis) = f_mem(τ)  (%)      — memorization fraction + 95% CI

x-axis = τ (optimizer updates, log scale). Color encodes N; line style encodes
the quantity. Identical look to dataset_size_effect/plot_dsize_fcd_fmem.py.

Usage
-----
    conda activate Mem_Gen
    cd DDIM_Evaluation/measured_dataset_effect
    python plot_measured_fcd_fmem.py
    python plot_measured_fcd_fmem.py --min_tau 100
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
    200:  '#377eb8',   # blue
    500:  '#ff7f00',   # orange
    1000: '#4daf4a',   # green
    2000: '#984ea3',   # purple
    4000: '#d62728',   # red
}
N_MARKERS = {200: 'o', 500: 's', 1000: 'D', 2000: '^', 4000: 'P'}


def _c(N):
    return N_COLORS.get(N, '#333333')


def _m(N):
    return N_MARKERS.get(N, 'o')


def main():
    ap = argparse.ArgumentParser(
        description="Plot FCD(Gen,Test)+f_mem vs τ for the MEASURED dataset")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Drop τ below this (removes τ=0 random-init spike)")
    ap.add_argument("--sizes", type=int, nargs='+', default=None,
                    help="Subset of N to plot (default: all in the CSV).")
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'measured_fcd_fmem.csv')
    out_dir = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\nRun: python compute_measured_fcd_fmem.py")

    df = pd.read_csv(csv_path)
    df = df[df['tau'] >= args.min_tau].sort_values(['N', 'tau'])
    if args.sizes is not None:
        df = df[df['N'].isin(args.sizes)]
    sizes = sorted(df['N'].unique())
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Dataset sizes: {sizes}")

    all_sizes = sorted(pd.read_csv(csv_path)['N'].unique())
    is_subset = args.sizes is not None and sizes != all_sizes
    fname = ('measured_fcd_fmem' if not is_subset
             else 'measured_fcd_fmem_N' + '_'.join(str(n) for n in sizes))

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()

    for N in sizes:
        sub = df[df['N'] == N].sort_values('tau')
        tau = sub['tau'].values
        gt = sub['FCD_Gen_Test'].values
        gt_sd = sub['FCD_Gen_Test_std'].values
        fmem = sub['f_mem'].values * 100.0
        if 'f_mem_ci_low' in sub and 'f_mem_ci_high' in sub:
            fmem_lo = sub['f_mem_ci_low'].values * 100.0
            fmem_hi = sub['f_mem_ci_high'].values * 100.0
        else:
            fmem_lo = fmem_hi = None
        floor = sub['FCD_Train_Test'].values
        floor_sd = sub['FCD_Train_Test_std'].values

        axL.plot(tau, gt, '-', color=_c(N), lw=2.2, marker=_m(N), ms=5, alpha=0.95)
        if np.any(gt_sd > 0):
            axL.fill_between(tau, gt - 2 * gt_sd, gt + 2 * gt_sd,
                             color=_c(N), alpha=0.12, linewidth=0)

        if len(floor):
            axL.axhline(floor[0], color=_c(N), ls=':', lw=1.4, alpha=0.9)
            if floor_sd[0] > 0:
                axL.axhspan(floor[0] - 2 * floor_sd[0], floor[0] + 2 * floor_sd[0],
                            color=_c(N), alpha=0.05, linewidth=0)

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

    size_handles = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2, ls='-',
                           ms=7, label=f'N = {N}') for N in sizes]
    style_handles = [
        Line2D([0], [0], color='k', ls='-',  lw=2.2, label='Gen–Test'),
        Line2D([0], [0], color='k', ls=':',  lw=1.4, label='Train–Test'),
        Line2D([0], [0], color='k', ls='--', lw=1.8, label=r'$f_{\mathrm{mem}}$'),
    ]
    leg1 = axL.legend(handles=size_handles, fontsize=16, loc='upper left',
                      framealpha=0.9, bbox_to_anchor=(0.25, 0.99))
    axL.add_artist(leg1)

    fig.canvas.draw()
    bb = leg1.get_window_extent().transformed(axL.transAxes.inverted())
    gap = 0.02
    leg2_top = bb.y0 - gap
    axL.legend(handles=style_handles, fontsize=16, loc='upper left',
               framealpha=0.9, bbox_to_anchor=(0.25, leg2_top))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
