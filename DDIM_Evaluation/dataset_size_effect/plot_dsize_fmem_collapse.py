"""
plot_dsize_fmem_collapse.py
    — Dataset-size sweep: normalized memorization collapse
      y = f_mem(τ) / f_mem(τ_max)   vs   x = τ / N
================================================================================
Reads results/dsize_fcd_fmem.csv (produced by compute_dsize_fcd_fmem.py) and draws,
for each dataset size N (fixed width W = 256, batch B = min(N, 500)), a single
normalized memorization curve:

  • x-axis = τ / N   — training time measured in "optimizer steps per training
                       sample" (a per-example exposure clock). Dividing τ by N puts
                       datasets of different size on a common, size-independent time
                       axis.
  • y-axis = f_mem(τ) / f_mem(τ_max)
                     — the memorization fraction of each run rescaled by its OWN
                       final value f_mem(τ_max). Every curve therefore ends at 1.0,
                       isolating the *shape / timing* of the memorization onset from
                       the absolute memorization level each N happens to reach inside
                       the training budget.

If, after this double rescaling, the curves fall on top of one another ("collapse"),
the memorization dynamics are governed by the single combined variable τ/N — i.e.
memorization is a function of how many times, on average, each unique sample has been
revisited, not of N and τ separately. Any residual spread reveals a genuine
dependence on N beyond the τ/N clock (e.g. the full-batch vs capped-batch confound).

Color encodes the dataset size N (same palette as plot_dsize_fcd_fmem.py).
x-axis is log-scaled (τ/N spans several decades).

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect
    python plot_dsize_fmem_collapse.py
    python plot_dsize_fmem_collapse.py --sizes 200 1000 4000
    python plot_dsize_fmem_collapse.py --min_tau 100
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

# Distinct color per dataset size (same palette family as plot_dsize_fcd_fmem.py).
N_COLORS = {
    100:  '#a65628',   # brown
    200:  '#377eb8',   # blue
    500:  '#ff7f00',   # orange
    1000: '#4daf4a',   # green
    2000: '#984ea3',   # purple
    4000: '#d62728',   # red
}
N_MARKERS = {100: 'v', 200: 'o', 500: 's', 1000: 'D', 2000: '^', 4000: 'P'}


def _c(N):
    return N_COLORS.get(N, '#333333')


def _m(N):
    return N_MARKERS.get(N, 'o')


def main():
    ap = argparse.ArgumentParser(
        description="Plot normalized memorization collapse "
                    "f_mem(τ)/f_mem(τ_max) vs τ/N for the dataset-size sweep "
                    "(W=256, B=min(N,500))")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Drop τ below this (removes the τ=0 random-init point)")
    ap.add_argument("--sizes", type=int, nargs='+', default=None,
                    help="Subset of dataset sizes N to plot (default: all in the "
                         "CSV). A subset writes a separate figure named "
                         "dsize_fmem_collapse_N<...>.png so the full figure is kept.")
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'dsize_fcd_fmem.csv')
    out_dir  = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Run: python compute_dsize_fcd_fmem.py")

    df = pd.read_csv(csv_path)
    df = df[df['tau'] >= args.min_tau].sort_values(['N', 'tau'])
    if args.sizes is not None:
        df = df[df['N'].isin(args.sizes)]
        missing = sorted(set(args.sizes) - set(df['N'].unique()))
        if missing:
            print(f"  WARNING: requested sizes not in CSV, ignored: {missing}")
    sizes = sorted(df['N'].unique())
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Dataset sizes: {sizes}")

    # Full sweep → base filename; a subset → suffix with the selected N values so
    # the original all-N figure is never overwritten.
    all_sizes = sorted(pd.read_csv(csv_path)['N'].unique())
    is_subset = args.sizes is not None and sizes != all_sizes
    fname = ('dsize_fmem_collapse' if not is_subset
             else 'dsize_fmem_collapse_N' + '_'.join(str(n) for n in sizes))

    fig, ax = plt.subplots(figsize=(10, 6))

    print("\n  N     τ_max      f_mem(τ_max)   used for normalization")
    print("  ----  ---------  ------------")
    for N in sizes:
        sub = df[df['N'] == N].sort_values('tau')
        tau  = sub['tau'].values.astype(float)
        fmem = sub['f_mem'].values.astype(float)

        tau_max      = tau[-1]
        fmem_tau_max = fmem[-1]
        print(f"  {N:<4}  {tau_max:>9.0f}  {fmem_tau_max:>10.4f}")

        # y = f_mem(τ) / f_mem(τ_max). Guard against a degenerate final value.
        if fmem_tau_max <= 0:
            print(f"    WARNING: f_mem(τ_max)=0 for N={N}; skipping (cannot normalize).")
            continue
        y = fmem / fmem_tau_max

        # x = τ / N  (optimizer steps per training sample)
        x = tau / float(N)

        ax.plot(x, y, '-', color=_c(N), lw=2.2, marker=_m(N), ms=5, alpha=0.95,
                label=f'N = {N}')

        # 95% bootstrap CI on f_mem, carried through the SAME normalizer
        # f_mem(τ_max) so the band lives on the rescaled y-axis.
        if 'f_mem_ci_low' in sub and 'f_mem_ci_high' in sub:
            y_lo = sub['f_mem_ci_low'].values.astype(float) / fmem_tau_max
            y_hi = sub['f_mem_ci_high'].values.astype(float) / fmem_tau_max
            if np.any(y_hi - y_lo > 0):
                ax.fill_between(x, y_lo, y_hi, color=_c(N), alpha=0.12, linewidth=0)

    # Reference: the f_mem = 0.1 memorization-onset threshold, expressed on the
    # normalized axis is threshold-dependent per curve, so instead we mark y = 1
    # (each curve's own τ_max endpoint) with a light guide line.
    ax.axhline(1.0, color='0.6', ls=':', lw=1.0, alpha=0.8)

    ax.set_xscale('log')
    ax.set_xlim(left=1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau / N$', fontsize=20)
    ax.set_ylabel(r'$f_{\mathrm{mem}}(\tau)\,/\,f_{\mathrm{mem}}(\tau_{\max})$',
                  fontsize=20)
    ax.tick_params(labelsize=20)
    ax.grid(True, alpha=0.3)

    leg = ax.legend(fontsize=16, loc='upper left', framealpha=0.9,
                    bbox_to_anchor=(0.01, 0.99))
    ax.add_artist(leg)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
