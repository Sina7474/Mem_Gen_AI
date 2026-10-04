"""
plot_fullbatch_fmem_collapse.py
    — Normalized full-batch memorization collapse vs τ/N
================================================================================
Draws one figure for the true full-batch runs N=batch∈{200,500,1000}:

    y = f_mem(τ) / f_mem(τ_max)
    x = τ/N  (optimizer updates per training sample)

N=200 and N=1000 come from results/fullbatch_fcd_fmem.csv. The compatible
N=500, batch=500 run is imported from the dataset-size experiment at
../dataset_size_effect/results/dsize_fcd_fmem.csv.

A dotted horizontal line marks a normalized memorization-onset threshold. The
interpolated crossing for each N is printed to stdout and marked on the curve.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/fullbatch_effect
    python plot_fullbatch_fmem_collapse.py
    python plot_fullbatch_fmem_collapse.py --thresh 0.05
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

# Same palette as the other full-batch plot.
N_COLORS  = {200: '#ff7f00', 500: '#377eb8', 1000: '#4daf4a', 4000: '#d62728'}
N_MARKERS = {200: 's', 500: 'o', 1000: 'D', 4000: 'P'}


def _c(N):
    return N_COLORS.get(N, '#333333')


def _m(N):
    return N_MARKERS.get(N, 'o')


def _tau_mem(tau, fmem, thresh):
    """First τ at which f_mem crosses `thresh` (linear interpolation between the
    bracketing points). Returns np.nan if it never crosses."""
    tau = np.asarray(tau, float)
    fmem = np.asarray(fmem, float)
    for i in range(1, len(fmem)):
        if fmem[i - 1] < thresh <= fmem[i]:
            # linear interpolation in (τ, f_mem)
            t0, t1 = tau[i - 1], tau[i]
            f0, f1 = fmem[i - 1], fmem[i]
            if f1 == f0:
                return t1
            return t0 + (thresh - f0) * (t1 - t0) / (f1 - f0)
    return np.nan


def main():
    ap = argparse.ArgumentParser(
        description="Full-batch: normalized f_mem vs τ/N (collapse)")
    ap.add_argument("--thresh", type=float, default=0.10,
                    help="Normalized f_mem threshold used to define onset "
                         "(default: 0.10)")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Drop τ below this (removes the τ=0 random-init spike)")
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--n500_csv_path", type=str, default=None,
                    help="Dataset-size CSV containing the N=batch=500 run")
    ap.add_argument("--n500_fmem_csv", type=str, default=None,
                    help="Optional alternate f_mem table for N=500. Values are "
                         "merged into --n500_csv_path while retaining its metadata.")
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()
    if not 0 < args.thresh < 1:
        ap.error("--thresh must be strictly between 0 and 1")

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'fullbatch_fcd_fmem.csv')
    n500_csv_path = args.n500_csv_path or os.path.join(
        base, '..', 'dataset_size_effect', 'results', 'dsize_fcd_fmem.csv')
    out_dir  = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\nRun: python compute_fullbatch_fcd_fmem.py")
    if not os.path.exists(n500_csv_path):
        raise FileNotFoundError(
            f"N=500 dataset-size CSV not found: {n500_csv_path}")
    if args.n500_fmem_csv and not os.path.exists(args.n500_fmem_csv):
        raise FileNotFoundError(
            f"N=500 alternate f_mem CSV not found: {args.n500_fmem_csv}")

    df_fullbatch = pd.read_csv(csv_path)
    df_dsize = pd.read_csv(n500_csv_path)
    df_n500 = df_dsize[
        (df_dsize['N'] == 500) & (df_dsize['batch_size'] == 500)
    ].copy()
    if df_n500.empty:
        raise ValueError(
            f"No N=500, batch_size=500 rows found in {n500_csv_path}")
    if args.n500_fmem_csv:
        # Alternate k tables intentionally omit the tau=0 random-init point,
        # which this plot drops via --min_tau anyway.
        df_n500 = df_n500[df_n500['tau'] >= args.min_tau].copy()
        alternate = pd.read_csv(args.n500_fmem_csv)
        required = {'N', 'tau', 'f_mem', 'k'}
        missing = sorted(required - set(alternate.columns))
        if missing:
            raise ValueError(
                f"Alternate N=500 f_mem CSV lacks columns: {missing}")
        alternate = alternate[alternate['N'] == 500][
            ['N', 'tau', 'f_mem', 'k']].copy()
        if alternate.duplicated(['N', 'tau']).any():
            raise ValueError("Alternate N=500 f_mem CSV has duplicate tau rows")
        df_n500 = df_n500.drop(columns=['f_mem', 'k']).merge(
            alternate, on=['N', 'tau'], how='left', validate='one_to_one')
        if df_n500[['f_mem', 'k']].isna().any().any():
            raise ValueError("Alternate N=500 f_mem CSV does not cover every tau")

    df = pd.concat([df_fullbatch, df_n500], ignore_index=True, sort=False)
    if df.duplicated(['N', 'tau']).any():
        duplicates = df.loc[df.duplicated(['N', 'tau'], keep=False), ['N', 'tau']]
        raise ValueError(f"Duplicate (N, tau) rows after merging:\n{duplicates}")
    df = df[df['tau'] >= args.min_tau].sort_values(['N', 'tau'])
    sizes = sorted(df['N'].unique())
    print(f"Loaded {len(df_fullbatch)} full-batch rows from {csv_path}")
    print(f"Added {len(df_n500)} N=batch=500 rows from {n500_csv_path}")
    if args.n500_fmem_csv:
        print(f"  Alternate N=500 f_mem: {args.n500_fmem_csv}")
    print(f"  Dataset sizes (=batch): {sizes}")

    fig, ax = plt.subplots(figsize=(10, 6))

    print(f"\n  Normalized memorization onset (ratio = {args.thresh}):")
    print(f"  {'N':>6} | {'f_mem(tau_max)':>14} | {'tau_mem':>10} | {'(tau/N)_mem':>12}")
    print("  " + "-" * 55)
    for N in sizes:
        sub  = df[df['N'] == N].sort_values('tau')
        tau  = sub['tau'].values.astype(float)
        fmem = sub['f_mem'].values.astype(float)
        fmem_tau_max = float(fmem[-1])
        if fmem_tau_max <= 0:
            print(f"  WARNING: f_mem(tau_max)=0 for N={N}; skipping.")
            continue

        fmem_norm = fmem / fmem_tau_max
        ax.plot(tau / N, fmem_norm, '-', color=_c(N), lw=2.2,
                marker=_m(N), ms=6, alpha=0.95, label=f'N = B = {N}')

        tmem = _tau_mem(tau, fmem_norm, args.thresh)
        if np.isfinite(tmem):
            ax.plot([tmem / N], [args.thresh], marker='o', ms=11, mfc='none',
                    mec=_c(N), mew=2.2)
            print(f"  {N:>6} | {fmem_tau_max:>14.4f} | {tmem:>10.0f} | "
                  f"{tmem / N:>12.2f}")
        else:
            print(f"  {N:>6} | {fmem_tau_max:>14.4f} | {'never':>10} | {'-':>12}")

    ax.axhline(args.thresh, color='k', ls=':', lw=1.2, alpha=0.7)
    ax.axhline(1.0, color='0.6', ls=':', lw=1.0, alpha=0.7)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r'$\tau/N$', fontsize=20)
    ax.set_ylabel(
        r'$f_{\mathrm{mem}}(\tau)\,/\,f_{\mathrm{mem}}(\tau_{\max})$',
        fontsize=20)
    ax.tick_params(labelsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=16, loc='upper left', framealpha=0.9)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'fullbatch_fmem_collapse.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"\n  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
