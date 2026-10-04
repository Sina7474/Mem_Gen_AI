"""
plot_dsize_fmem_28GHz_LoS.py
    — 28 GHz LoS scene: f_mem vs τ for the dataset-size sweep (W = 256, B = min(N,500))
================================================================================
Reads results/dsize_fmem_28GHz_LoS.csv (produced by compute_dsize_fmem_28GHz_LoS.py)
and draws one f_mem(τ) curve per dataset size N, with the 95 % bootstrap CI as a
shaded band.

Colour encodes the dataset size N and is taken from a perceptually-ordered,
colour-vision-deficiency-safe viridis ramp (dark → light with increasing N), with
a distinct marker per N as a redundant cue.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect_28GHz_LoS
    python plot_dsize_fmem_28GHz_LoS.py
    python plot_dsize_fmem_28GHz_LoS.py --sizes 200 1000 4000
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

# Colour-vision-deficiency-safe, luminance-ordered palette (viridis 0 → 0.85).
N_COLORS = {
    200:  '#440154',
    500:  '#414487',
    1000: '#2a788e',
    2000: '#22a884',
    4000: '#7ad151',
}
N_MARKERS = {200: 'o', 500: 's', 1000: 'D', 2000: '^', 4000: 'P'}


def _c(N):
    return N_COLORS.get(N, '#333333')


def _m(N):
    return N_MARKERS.get(N, 'o')


def main():
    ap = argparse.ArgumentParser(
        description="Plot f_mem vs τ for the 28 GHz LoS dataset-size sweep")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Drop τ below this (removes the τ=0 random-init point)")
    ap.add_argument("--sizes", type=int, nargs='+', default=None,
                    help="Subset of dataset sizes N to plot (default: all in the CSV). "
                         "A subset writes a separate figure so the full one is kept.")
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'dsize_fmem_28GHz_LoS.csv')
    out_dir  = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Run: python compute_dsize_fmem_28GHz_LoS.py")

    df_all = pd.read_csv(csv_path)
    df = df_all[df_all['tau'] >= args.min_tau].sort_values(['N', 'tau'])
    if args.sizes is not None:
        df = df[df['N'].isin(args.sizes)]
        missing = sorted(set(args.sizes) - set(df['N'].unique()))
        if missing:
            print(f"  WARNING: requested sizes not in CSV, ignored: {missing}")
    sizes = sorted(df['N'].unique())
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Dataset sizes: {sizes}")

    all_sizes = sorted(df_all['N'].unique())
    is_subset = args.sizes is not None and sizes != all_sizes
    fname = ('dsize_fmem_28GHz_LoS' if not is_subset
             else 'dsize_fmem_28GHz_LoS_N' + '_'.join(str(n) for n in sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    for N in sizes:
        sub = df[df['N'] == N].sort_values('tau')
        tau  = sub['tau'].values
        fmem = sub['f_mem'].values * 100.0
        lo   = sub['f_mem_ci_low'].values * 100.0
        hi   = sub['f_mem_ci_high'].values * 100.0

        ax.plot(tau, fmem, '-', color=_c(N), lw=2.2, marker=_m(N), ms=5, alpha=0.95)
        if np.any(hi - lo > 0):
            ax.fill_between(tau, lo, hi, color=_c(N), alpha=0.15, linewidth=0)

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$  (%)', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.grid(True, alpha=0.3)

    handles = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2, ls='-',
                      ms=7, label=f'N = {N}') for N in sizes]
    ax.legend(handles=handles, fontsize=16, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
