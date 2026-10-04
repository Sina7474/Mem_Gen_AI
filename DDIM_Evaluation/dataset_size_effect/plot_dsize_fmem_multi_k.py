"""
plot_dsize_fmem_multi_k.py
    — Robustness of f_mem(τ) to the ratio-test threshold k ∈ {1/4, 1/3, 1/2}
      for N ∈ {200, 1000}  (3.5 GHz dataset-size experiment)
================================================================================
Color encodes the dataset size N; line style encodes the threshold k.
The main-paper choice k = 1/3 is the solid line; k = 1/4 (dashed) and k = 1/2
(dotted) bracket it. If the three k-curves for a given N stay close and preserve
the ordering of the two N values, the memorization scaling is robust to k.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect
    python plot_dsize_fmem_multi_k.py
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
    1000: '#4daf4a',   # green
}
N_MARKERS = {200: 'o', 1000: 'D'}

# line style per threshold k  (main choice k=1/3 is solid)
K_STYLES = {
    0.25:            (':',  r'$\kappa=1/4$'),
    1.0 / 3.0:       ('-',  r'$\kappa=1/3$'),
    0.5:             ('--', r'$\kappa=1/2$'),
}
K_ORDER = [0.25, 1.0 / 3.0, 0.5]


def _k_key(kval):
    """Map a stored float k to the nearest canonical key."""
    return min(K_STYLES.keys(), key=lambda kk: abs(kk - kval))


def _c(N):
    return N_COLORS.get(N, '#333333')


def _m(N):
    return N_MARKERS.get(N, 'o')


def main():
    ap = argparse.ArgumentParser(
        description="f_mem(τ) at k ∈ {1/4,1/3,1/2} for N ∈ {200,1000}")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Drop τ below this (removes τ=0 random-init spike)")
    ap.add_argument("--sizes", type=int, nargs='+', default=[200, 1000])
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'dsize_fmem_multi_k.csv')
    out_dir  = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\nRun: python compute_dsize_fmem_multi_k.py")

    df = pd.read_csv(csv_path)
    df = df[df['tau'] >= args.min_tau]
    df = df[df['N'].isin(args.sizes)]
    sizes = sorted(df['N'].unique())
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Dataset sizes: {sizes}")

    fig, ax = plt.subplots(figsize=(10, 6))

    for N in sizes:
        for kval in K_ORDER:
            sub = df[(df['N'] == N) & (np.isclose(df['k'], kval))].sort_values('tau')
            if sub.empty:
                continue
            tau  = sub['tau'].values.astype(float)
            fmem = sub['f_mem'].values.astype(float) * 100.0
            ls, _ = K_STYLES[_k_key(kval)]
            mk = _m(N)
            ax.plot(tau, fmem, ls, color=_c(N), lw=2.0, marker=mk, ms=5,
                    alpha=0.95)
            if {'f_mem_ci_low', 'f_mem_ci_high'}.issubset(sub.columns):
                lo = sub['f_mem_ci_low'].values.astype(float) * 100.0
                hi = sub['f_mem_ci_high'].values.astype(float) * 100.0
                if np.any(hi - lo > 0):
                    ax.fill_between(tau, lo, hi, color=_c(N), alpha=0.08, linewidth=0)

    ax.set_xscale('log')
    # Show data from τ=300 onward; on the log scale, the first labeled
    # major tick is therefore τ=1000 (with 300 retained as the left limit).
    ax.set_xlim(left=300)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel(r'$\tau$', fontsize=20)
    ax.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$  (%)', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.grid(True, alpha=0.3)

    # two legends: N (color) and k (line style)
    size_handles = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2, ls='-',
                           ms=7, label=f'N = {N}') for N in sizes]
    style_handles = [Line2D([0], [0], color='k', ls=K_STYLES[k][0], lw=2.0,
                            label=K_STYLES[k][1]) for k in K_ORDER]

    leg1 = ax.legend(handles=size_handles, fontsize=16, loc='upper left',
                     framealpha=0.9, bbox_to_anchor=(0.01, 0.99))
    leg1.get_title().set_fontsize(16)
    ax.add_artist(leg1)

    fig.canvas.draw()
    bb = leg1.get_window_extent().transformed(ax.transAxes.inverted())
    leg2 = ax.legend(handles=style_handles, fontsize=16, loc='upper left',
                     framealpha=0.9, bbox_to_anchor=(0.01, bb.y0 - 0.03))
    leg2.get_title().set_fontsize(16)

    plt.tight_layout()
    fname = 'dsize_fmem_multi_k_N' + '_'.join(str(n) for n in sizes)
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == '__main__':
    main()
