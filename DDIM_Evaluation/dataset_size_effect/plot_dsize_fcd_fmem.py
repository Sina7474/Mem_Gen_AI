"""
plot_dsize_fcd_fmem.py
    — Dataset-size sweep: FCD(Gen,Test) + f_mem vs τ  (Figure-2-left visual language)
================================================================================
Reads results/dsize_fcd_fmem.csv (produced by compute_dsize_fcd_fmem.py) and draws,
for each dataset size N (fixed width W = 256, batch B = min(N, 500)), one curve
per N:

  • solid  line (LEFT axis)  = FCD(Gen, Test)     — sample quality (↓, want it low)
                               with a shaded 2σ band over the 5 test folds;
  • dotted line (LEFT axis)  = FCD(Train, Test)   — the real–real floor for THAT N
                               (each N has its own floor: different training set);
  • dashed line (RIGHT axis) = f_mem(τ)           — memorization fraction for THAT N.

Color encodes the dataset size N; line style encodes the quantity, exactly matching
the look of ../plot_fcd_vs_tau.py. x-axis = τ (optimizer updates, log scale).

Note on the τ axis. Unlike the full-batch experiment, here τ (optimizer steps) is
NOT identical to epochs for N ≥ 1000, because B is capped at 500 → steps/epoch =
ceil(N/500) > 1. τ is still the fair comparison axis (same gradient-noise scale for
all N ≥ 500), and `epoch_float = τ / steps_per_epoch` is available in the CSV if an
epoch axis is ever wanted.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect
    python plot_dsize_fcd_fmem.py
    python plot_dsize_fcd_fmem.py --min_tau 100
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

# Distinct color per dataset size (same palette family as ../plot_fcd_vs_tau.py).
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
        description="Plot FCD(Gen,Test)+f_mem vs τ for the dataset-size sweep "
                    "(W=256, B=min(N,500))")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Drop τ below this (removes the τ=0 random-init spike)")
    ap.add_argument("--xmin", type=float, default=None,
                    help="Left limit of the τ axis (axes-only). Independent of "
                         "--min_tau: use --min_tau to drop early DATA points while "
                         "--xmin controls where the AXIS starts, e.g. curves from "
                         "τ=3000 shown on an axis that begins at τ=1000.")
    ap.add_argument("--xmax", type=float, default=None,
                    help="Right limit of the τ axis (axes-only).")
    ap.add_argument("--legend_loc", type=str, default="upper left",
                    choices=["upper left", "upper center"],
                    help="Anchor corner for the two stacked legends "
                         "(default: 'upper left').")
    ap.add_argument("--sizes", type=int, nargs='+', default=None,
                    help="Subset of dataset sizes N to plot (default: all in the "
                         "CSV). Selecting a subset writes a separate figure named "
                         "dsize_fcd_fmem_N<...>.png so the full figure is preserved.")
    ap.add_argument("--legend_shift", type=float, default=0.12,
                    help="Horizontal shift of the side-by-side legends from centre. "
                         "Positive = left, negative = right (default: 0.12).")
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--fmem_csv", type=str, default=None,
                    help="Optional alternate f_mem table to merge by (N,tau), "
                         "leaving all FCD values from --csv_path unchanged.")
    ap.add_argument("--fmem_k", type=float, default=None,
                    help="Select this k from --fmem_csv when it contains multiple "
                         "ratio-test thresholds.")
    ap.add_argument("--output_suffix", type=str, default="",
                    help="Suffix added to new figure filenames, e.g. kappa1_4. "
                         "Required with --fmem_csv to protect canonical plots.")
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

    source_df = pd.read_csv(csv_path)
    df = source_df.copy()
    df = df[df['tau'] >= args.min_tau].sort_values(['N', 'tau'])
    if args.sizes is not None:
        df = df[df['N'].isin(args.sizes)]
        missing = sorted(set(args.sizes) - set(df['N'].unique()))
        if missing:
            print(f"  WARNING: requested sizes not in CSV, ignored: {missing}")
    sizes = sorted(df['N'].unique())

    if args.fmem_csv:
        if not args.output_suffix:
            raise ValueError(
                "--output_suffix is required with --fmem_csv so canonical "
                "figures cannot be overwritten.")
        alternate = pd.read_csv(args.fmem_csv)
        required = {'N', 'tau', 'f_mem', 'f_mem_ci_low', 'f_mem_ci_high'}
        missing_columns = sorted(required - set(alternate.columns))
        if missing_columns:
            raise ValueError(
                f"Alternate f_mem CSV lacks columns {missing_columns}: "
                f"{args.fmem_csv}")
        if args.fmem_k is not None:
            if 'k' not in alternate.columns:
                raise ValueError("--fmem_k requires a 'k' column in --fmem_csv.")
            alternate = alternate[
                np.isclose(alternate['k'].astype(float), args.fmem_k)
            ]
        elif 'k' in alternate.columns:
            unique_k = sorted(alternate['k'].dropna().astype(float).unique())
            if len(unique_k) != 1:
                raise ValueError(
                    f"Alternate f_mem CSV contains k={unique_k}; select one "
                    "with --fmem_k.")
        alternate = alternate[
            alternate['N'].isin(sizes) & (alternate['tau'] >= args.min_tau)
        ].copy()
        if alternate.duplicated(['N', 'tau']).any():
            duplicates = alternate.loc[
                alternate.duplicated(['N', 'tau'], keep=False), ['N', 'tau']
            ].drop_duplicates().values.tolist()
            raise ValueError(f"Duplicate alternate f_mem rows for {duplicates}")
        alternate = alternate[
            ['N', 'tau', 'f_mem', 'f_mem_ci_low', 'f_mem_ci_high']
        ].rename(columns={
            'f_mem': 'f_mem_alternate',
            'f_mem_ci_low': 'f_mem_ci_low_alternate',
            'f_mem_ci_high': 'f_mem_ci_high_alternate',
        })
        df = df.merge(alternate, on=['N', 'tau'], how='left', validate='one_to_one')
        missing_rows = df[df['f_mem_alternate'].isna()][['N', 'tau']]
        if not missing_rows.empty:
            raise ValueError(
                "Alternate f_mem CSV is incomplete for: "
                f"{missing_rows.values.tolist()}")
        df['f_mem'] = df.pop('f_mem_alternate')
        df['f_mem_ci_low'] = df.pop('f_mem_ci_low_alternate')
        df['f_mem_ci_high'] = df.pop('f_mem_ci_high_alternate')
        if args.fmem_k is not None:
            df['k'] = args.fmem_k

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Dataset sizes: {sizes}")
    if args.fmem_csv:
        print(f"  Alternate f_mem: {args.fmem_csv} (k={args.fmem_k})")

    # Full sweep → base filename; a subset → suffix with the selected N values so
    # the original all-N figure is never overwritten.
    all_sizes = sorted(source_df['N'].unique())
    is_subset = args.sizes is not None and sizes != all_sizes
    fname = ('dsize_fcd_fmem' if not is_subset
             else 'dsize_fcd_fmem_N' + '_'.join(str(n) for n in sizes))
    if args.min_tau > 1:
        fname += f'_fromtau{args.min_tau}'
    if args.output_suffix:
        suffix = args.output_suffix
        fname += suffix if suffix.startswith('_') else f'_{suffix}'

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()

    for N in sizes:
        sub = df[df['N'] == N].sort_values('tau')
        tau      = sub['tau'].values
        gt       = sub['FCD_Gen_Test'].values
        gt_sd    = sub['FCD_Gen_Test_std'].values
        fmem     = sub['f_mem'].values * 100.0   # fraction → percentage (%)
        # f_mem 95% bootstrap CI (columns present only in the recomputed CSV)
        if 'f_mem_ci_low' in sub and 'f_mem_ci_high' in sub:
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

        # Real–real floor for THIS N (dotted, same color)
        if len(floor):
            axL.axhline(floor[0], color=_c(N), ls=':', lw=1.4, alpha=0.9)
            if floor_sd[0] > 0:
                axL.axhspan(floor[0] - 2 * floor_sd[0], floor[0] + 2 * floor_sd[0],
                            color=_c(N), alpha=0.05, linewidth=0)

        # f_mem: dashed on the right axis, with a 95% bootstrap CI band
        axR.plot(tau, fmem, '--', color=_c(N), lw=1.8, marker=_m(N), ms=5, alpha=0.85)
        if fmem_lo is not None and np.any(fmem_hi - fmem_lo > 0):
            axR.fill_between(tau, fmem_lo, fmem_hi,
                             color=_c(N), alpha=0.12, linewidth=0)

    axL.set_xscale('log')
    axL.set_ylim(bottom=0)
    axR.set_ylim(bottom=0, top=100)
    axR.set_yticks(np.arange(0, 101, 10))
    if args.xmin is not None or args.xmax is not None:
        cur = axL.get_xlim()
        axL.set_xlim(left=args.xmin if args.xmin is not None else cur[0],
                     right=args.xmax if args.xmax is not None else cur[1])
    axL.set_xlabel(r'$\tau$', fontsize=20)
    axL.set_ylabel('FCD', fontsize=20)
    axR.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$  (%)', fontsize=20, labelpad=15)
    axL.tick_params(labelsize=20)
    axR.tick_params(labelsize=20)
    axL.grid(True, alpha=0.3)

    # Legend 1: dataset size (color).  Legend 2: quantity (line style).
    size_handles = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2, ls='-',
                           ms=7, label=f'N = {N}')
                    for N in sizes]
    style_handles = [
        Line2D([0], [0], color='k', ls='-',  lw=2.2, label='Gen–Test'),
        Line2D([0], [0], color='k', ls=':',  lw=1.4, label='Train–Test'),
        Line2D([0], [0], color='k', ls='--', lw=1.8, label=r'$f_{\mathrm{mem}}$'),
    ]
    # --- Legends: side by side, shifted left of centre ---
    # 1. Draw temporary legends to measure their widths in axes-fraction coords.
    tmp1 = axL.legend(handles=size_handles, fontsize=16, loc='upper left',
                      framealpha=0.9, bbox_to_anchor=(0, 0.99))
    fig.canvas.draw()
    w1 = tmp1.get_window_extent().transformed(axL.transAxes.inverted()).width
    tmp1.remove()

    tmp2 = axL.legend(handles=style_handles, fontsize=16, loc='upper left',
                      framealpha=0.9, bbox_to_anchor=(0, 0.99))
    fig.canvas.draw()
    w2 = tmp2.get_window_extent().transformed(axL.transAxes.inverted()).width
    tmp2.remove()

    # 2. Compute x positions.  Increase left_shift to move further left.
    gap = 0.02
    total = w1 + gap + w2
    x0 = (1 - total) / 2 - args.legend_shift

    # 3. Place both legends at their final positions.
    leg1 = axL.legend(handles=size_handles, fontsize=16, loc='upper left',
                      framealpha=0.9, bbox_to_anchor=(x0, 0.99))
    axL.add_artist(leg1)
    axL.legend(handles=style_handles, fontsize=16, loc='upper left',
               framealpha=0.9, bbox_to_anchor=(x0 + w1 + gap, 0.99))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{fname}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
