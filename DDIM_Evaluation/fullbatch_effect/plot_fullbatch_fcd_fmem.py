"""
plot_fullbatch_fcd_fmem.py
    — Full-batch sweep: FCD(Gen,Test) + f_mem vs τ  (Figure-2-left visual language)
================================================================================
Reads results/fullbatch_fcd_fmem.csv (produced by compute_fullbatch_fcd_fmem.py)
and imports the compatible N=500, batch=500 run from
../dataset_size_effect/results/dsize_fcd_fmem.csv. It then draws, for each dataset
size N (with its matched full-batch size = N), one curve per N:

  • solid  line (LEFT axis)  = FCD(Gen, Test)     — sample quality (↓, want it low)
                               with a shaded 2σ band over the 5 test folds;
  • dotted line (LEFT axis)  = FCD(Train, Test)   — the real–real floor for THAT N
                               (each N has its own floor: different training set);
  • dashed line (RIGHT axis) = f_mem(τ)           — memorization fraction for THAT N.

Color encodes the dataset size N; line style encodes the quantity, exactly matching
the look of ../plot_fcd_vs_tau.py. x-axis = τ (optimizer updates, log scale). Since
batch = N, one optimizer step = one epoch, so τ here also equals the number of
data repetitions per sample.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/fullbatch_effect
    python plot_fullbatch_fcd_fmem.py
    python plot_fullbatch_fcd_fmem.py --min_tau 100
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

# Distinct color per dataset size (same palette as ../plot_fcd_vs_tau.py).
N_COLORS  = {200: '#ff7f00', 500: '#377eb8', 1000: '#4daf4a', 4000: '#d62728'}
N_MARKERS = {200: 's', 500: 'o', 1000: 'D', 4000: 'P'}


def _c(N):
    return N_COLORS.get(N, '#333333')


def _m(N):
    return N_MARKERS.get(N, 'o')


def main():
    ap = argparse.ArgumentParser(
        description="Plot FCD(Gen,Test)+f_mem vs τ for the full-batch (batch=N) sweep")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Drop τ below this (removes the τ=0 random-init spike)")
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--n500_csv_path", type=str, default=None,
                    help="Dataset-size CSV containing the N=batch=500 run")
    ap.add_argument("--n500_fmem_csv", type=str, default=None,
                    help="Optional alternate f_mem table for N=500. Values are "
                         "merged into --n500_csv_path while retaining its FCD data.")
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'fullbatch_fcd_fmem.csv')
    n500_csv_path = args.n500_csv_path or os.path.join(
        base, '..', 'dataset_size_effect', 'results', 'dsize_fcd_fmem.csv')
    out_dir  = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Run: python compute_fullbatch_fcd_fmem.py")
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

    # N=500 uses B=min(N,500)=500 in the dataset-size experiment, so it is a
    # genuine full-batch run and is directly compatible with this figure.
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

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()

    for N in sizes:
        sub = df[df['N'] == N].sort_values('tau')
        tau     = sub['tau'].values
        gt      = sub['FCD_Gen_Test'].values
        gt_sd   = sub['FCD_Gen_Test_std'].values
        fmem    = sub['f_mem'].values
        floor   = sub['FCD_Train_Test'].values
        floor_sd = sub['FCD_Train_Test_std'].values

        # FCD(Gen,Test): solid + 2σ band
        axL.plot(tau, gt, '-', color=_c(N), lw=2.2, marker=_m(N), ms=5, alpha=0.95)
        if np.any(gt_sd > 0):
            axL.fill_between(tau, gt - 2 * gt_sd, gt + 2 * gt_sd,
                             color=_c(N), alpha=0.12, linewidth=0)

        # Real–real floor for THIS N (dotted, same color)
        if len(floor):
            axL.axhline(floor[0], color=_c(N), ls=':', lw=1.4, alpha=0.9)
            if floor_sd[0] > 0:
                axL.axhspan(floor[0] - 2 * floor_sd[0], floor[0] + 2 * floor_sd[0],
                            color=_c(N), alpha=0.05, linewidth=0)

        # f_mem: dashed on the right axis
        axR.plot(tau, fmem, '--', color=_c(N), lw=1.8, marker=_m(N), ms=5, alpha=0.85)

    axL.set_xscale('log')
    axL.set_ylim(bottom=0)
    axR.set_ylim(bottom=0, top=1.0)
    axR.set_yticks(np.arange(0, 1.01, 0.1))
    axR.set_yticklabels([f'{int(v*100)}' for v in np.arange(0, 1.01, 0.1)])
    axL.set_xlabel(r'$\tau$', fontsize=20)
    axL.set_ylabel('FCD', fontsize=20)
    axR.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$ (%)', fontsize=20, labelpad=15)
    axL.tick_params(labelsize=20)
    axR.tick_params(labelsize=20)
    axL.grid(True, alpha=0.3)

    # Legend 1: dataset size (color).  Legend 2: quantity (line style).
    size_handles = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2, ls='-',
                           ms=7, label=f'N = B = {N}') for N in sizes]
    style_handles = [
        Line2D([0], [0], color='k', ls='-',  lw=2.2, label='Gen–Test'),
        Line2D([0], [0], color='k', ls=':',  lw=1.4, label='Train–Test'),
        Line2D([0], [0], color='k', ls='--', lw=1.8, label=r'$f_{\mathrm{mem}}$'),
    ]
    # Keep the two legend boxes side by side and centered as a pair: dataset
    # encoding on the left, quantity/line-style encoding on the right.
    leg1 = axL.legend(handles=size_handles, fontsize=16, loc='upper right',
                      framealpha=0.9, bbox_to_anchor=(0.49, 0.99))
    axL.add_artist(leg1)
    axL.legend(handles=style_handles, fontsize=16, loc='upper left',
               framealpha=0.9, bbox_to_anchor=(0.51, 0.99))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'fullbatch_fcd_fmem.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
