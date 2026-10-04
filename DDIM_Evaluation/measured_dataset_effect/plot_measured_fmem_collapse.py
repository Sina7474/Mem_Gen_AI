"""
plot_measured_fmem_collapse.py
    — Measured 1.272 GHz: normalized memorization collapse
      y = f_mem(τ) / f_mem(τ_max)   vs   x = τ / N
================================================================================
Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/measured_dataset_effect
    python plot_measured_fmem_collapse.py
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

N_COLORS = {
    200:  '#377eb8',   # blue
    500:  '#ff7f00',   # orange
    1000: '#4daf4a',   # green
    2000: '#984ea3',   # purple
}
N_MARKERS = {200: 'o', 500: 's', 1000: 'D', 2000: '^'}


def _c(N):
    return N_COLORS.get(N, '#333333')


def _m(N):
    return N_MARKERS.get(N, 'o')


def main():
    ap = argparse.ArgumentParser(
        description="Measured dataset: normalized memorization collapse "
                    "f_mem(τ)/f_mem(τ_max) vs τ/N")
    ap.add_argument("--min_tau", type=int, default=100,
                    help="Drop τ below this (default: 100)")
    ap.add_argument("--sizes", type=int, nargs='+', default=[200, 500, 1000, 2000])
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'measured_fcd_fmem.csv')
    out_dir  = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df['tau'] >= args.min_tau].sort_values(['N', 'tau'])
    df = df[df['N'].isin(args.sizes)]
    sizes = sorted(df['N'].unique())
    missing = sorted(set(args.sizes) - set(sizes))
    if missing:
        print(f"  WARNING: sizes not in CSV, ignored: {missing}")
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Dataset sizes: {sizes}")

    fname = 'measured_fmem_collapse_N' + '_'.join(str(n) for n in sizes)

    fig, ax = plt.subplots(figsize=(10, 6))

    print("\n  N     τ_max      f_mem_max")
    print("  ----  ---------  ------------")
    for N in sizes:
        sub = df[df['N'] == N].sort_values('tau')
        tau  = sub['tau'].values.astype(float)
        fmem = sub['f_mem'].values.astype(float)

        tau_max      = tau[-1]
        fmem_max     = fmem.max()
        print(f"  {N:<4}  {tau_max:>9.0f}  {fmem_max:>10.4f}")

        if fmem_max <= 0:
            print(f"    WARNING: f_mem_max=0 for N={N}; skipping.")
            continue

        y = fmem / fmem_max
        x = tau / float(N)

        ax.plot(x, y, '-', color=_c(N), lw=2.2, marker=_m(N), ms=5, alpha=0.95,
                label=f'N = {N}')

        if 'f_mem_ci_low' in sub.columns and 'f_mem_ci_high' in sub.columns:
            y_lo = sub['f_mem_ci_low'].values.astype(float) / fmem_max
            y_hi = sub['f_mem_ci_high'].values.astype(float) / fmem_max
            if np.any(y_hi - y_lo > 0):
                ax.fill_between(x, y_lo, y_hi, color=_c(N), alpha=0.12, linewidth=0)

    ax.set_xscale('log')
    # Show data from τ/N = 5 onward; on the log scale, the first labeled
    # major tick is therefore 10 (with 5 retained as the left limit).
    ax.set_xlim(left=5)
    ax.set_ylim(bottom=0, top=1.05)
    ax.set_xlabel(r'$\tau\,/\,N$', fontsize=20)
    ax.set_ylabel(r'$f_{\mathrm{mem}}(\tau)\;/\;f_{\mathrm{mem}}(\tau_{\max})$', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color='0.4', ls=':', lw=1.2, alpha=0.7)

    ax.legend(fontsize=16, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == '__main__':
    main()
