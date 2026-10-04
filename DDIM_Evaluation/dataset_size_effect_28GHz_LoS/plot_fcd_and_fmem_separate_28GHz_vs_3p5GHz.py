"""
plot_fcd_and_fmem_separate_28GHz_vs_3p5GHz.py
    — Cross-band comparison as TWO standalone paper figures
================================================================================
Produces two separate PNG/PDF files (one metric each) so they can be placed
independently in the paper:

    fcd_28GHz_vs_3p5GHz_N200_1000.{png,pdf}    FCD(Gen, Test) vs τ
    fmem_28GHz_vs_3p5GHz_N200_1000.{png,pdf}   f_mem(τ)       vs τ

Data
----
    3.5 GHz (LoS + NLoS)   ../dataset_size_effect/results/dsize_fcd_fmem.csv
    28  GHz (LoS only)     results/dsize_fcd_fmem_28GHz_LoS.csv

Encoding (identical in both figures)
------------------------------------
    colour        → dataset size N (blue / orange)
    line style    → scene: solid = 3.5 GHz, dashed = 28 GHz
    marker fill   → scene: filled = 3.5 GHz, open (larger) = 28 GHz
    dotted line   → FCD(Train, Test) floor (FCD figure only)

Marker SHAPE is deliberately identical for both scenes — the two carrier
frequencies are separated by line style (solid vs. dashed) reinforced by the
marker fill (filled vs. open).

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect_28GHz_LoS
    python plot_fcd_and_fmem_separate_28GHz_vs_3p5GHz.py
    python plot_fcd_and_fmem_separate_28GHz_vs_3p5GHz.py --sizes 200 1000 4000
"""

import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Colour encodes the dataset size N (blue / orange are easy to tell apart).
N_COLORS = {200: '#1f77b4', 1000: '#ff7f0e', 500: '#2ca02c',
            2000: '#9467bd', 4000: '#8c564b'}

MARKER_35 = 'o'       # 3.5 GHz: filled circle
MARKER_28 = 'D'       # 28 GHz: filled diamond
MS_35 = 7             # 3.5 GHz marker size
MS_28 = 7             # 28 GHz marker size
LS_35 = '-'           # 3.5 GHz line style
LS_28 = '--'          # 28 GHz line style


def _c(N):
    return N_COLORS.get(N, '#333333')


def _legend_handles(sizes, with_floor):
    handles = [
        Line2D([0], [0], color=_c(N), marker=MARKER_35, ms=7, ls='-', lw=2.2,
               markerfacecolor=_c(N), label=f'N = {N}') for N in sizes
    ] + [
        Line2D([0], [0], color='k', marker=MARKER_35, ms=MS_35, ls=LS_35, lw=2.2,
               markerfacecolor='k', label='3.5 GHz (LoS+NLoS)'),
        Line2D([0], [0], color='k', marker=MARKER_28, ms=MS_28, ls=LS_28, lw=2.2,
               markerfacecolor='k', label='28 GHz (LoS)'),
    ]
    if with_floor:
        handles.append(
            Line2D([0], [0], color='grey', ls=':', lw=1.2, label='FCD floor'))
    return handles


def _save(fig, out_dir, stem):
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{stem}.{ext}')
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close(fig)


def plot_fcd(d35, d28, sizes, min_tau, out_dir, stem):
    fig, ax = plt.subplots(figsize=(10, 6))
    for N in sizes:
        c = _c(N)

        # ── 3.5 GHz: solid line + filled circle ──────────────────────────────
        s35 = d35[(d35.N == N) & (d35.tau >= min_tau)].sort_values('tau')
        ax.plot(s35['tau'], s35['FCD_Gen_Test'], ls=LS_35, color=c, lw=2.2,
                marker=MARKER_35, ms=MS_35,
                markerfacecolor=c, markeredgecolor=c, alpha=0.95)
        ax.axhline(s35['FCD_Train_Test'].iloc[0], color=c, ls=':', lw=1.2, alpha=0.5)

        # ── 28 GHz: dashed line + filled diamond ─────────────────────────────
        s28 = d28[(d28.N == N) & (d28.tau >= min_tau)].sort_values('tau')
        ax.plot(s28['tau'], s28['FCD_Gen_Test'], ls=LS_28, color=c, lw=2.2,
                marker=MARKER_28, ms=MS_28,
                markerfacecolor=c, markeredgecolor=c, alpha=0.95)
        ax.axhline(s28['FCD_Train_Test'].iloc[0], color=c, ls=':', lw=1.2, alpha=0.3)

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax.set_ylabel('FCD (Gen, Test)  ↓', fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=_legend_handles(sizes, with_floor=True), fontsize=14,
              loc='upper right', framealpha=0.92, ncol=1)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def plot_fmem(d35, d28, sizes, min_tau, out_dir, stem):
    fig, ax = plt.subplots(figsize=(10, 6))
    for N in sizes:
        c = _c(N)

        # ── 3.5 GHz: solid line + filled circle ──────────────────────────────
        s35 = d35[(d35.N == N) & (d35.tau >= min_tau)].sort_values('tau')
        ax.plot(s35['tau'], s35['f_mem'] * 100.0, ls=LS_35, color=c, lw=2.2,
                marker=MARKER_35, ms=MS_35,
                markerfacecolor=c, markeredgecolor=c, alpha=0.95)

        # ── 28 GHz: dashed line + filled diamond ─────────────────────────────
        s28 = d28[(d28.N == N) & (d28.tau >= min_tau)].sort_values('tau')
        ax.plot(s28['tau'], s28['f_mem'] * 100.0, ls=LS_28, color=c, lw=2.2,
                marker=MARKER_28, ms=MS_28,
                markerfacecolor=c, markeredgecolor=c, alpha=0.95)

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$  (%)', fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=_legend_handles(sizes, with_floor=False), fontsize=14,
              loc='upper left', framealpha=0.92, ncol=1)
    fig.tight_layout()
    _save(fig, out_dir, stem)


def main():
    ap = argparse.ArgumentParser(
        description="Two standalone figures: FCD vs τ and f_mem vs τ, "
                    "28 GHz LoS against 3.5 GHz, for a few N")
    ap.add_argument("--sizes", type=int, nargs='+', default=[200, 1000],
                    help="Dataset sizes to draw (default: 200 1000)")
    ap.add_argument("--min_tau", type=int, default=100,
                    help="Drop τ below this (removes the τ=0 random-init spike)")
    ap.add_argument("--csv_28ghz", type=str, default=None)
    ap.add_argument("--csv_35ghz", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv28 = args.csv_28ghz or os.path.join(
        base, 'results', 'dsize_fcd_fmem_28GHz_LoS.csv')
    csv35 = args.csv_35ghz or os.path.join(
        base, '..', 'dataset_size_effect', 'results', 'dsize_fcd_fmem.csv')
    out_dir = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    for p, hint in ((csv28, "python compute_dsize_fmem_28GHz_LoS.py"),
                    (csv35, "python ../dataset_size_effect/compute_dsize_fcd_fmem.py")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV not found: {os.path.normpath(p)}\nRun: {hint}")

    d28 = pd.read_csv(csv28)
    d35 = pd.read_csv(csv35)

    common = set(d28['N'].unique()) & set(d35['N'].unique())
    sizes = [n for n in args.sizes if n in common]
    missing = sorted(set(args.sizes) - common)
    if missing:
        print(f"  WARNING: sizes absent from one of the CSVs, ignored: {missing}")
    if not sizes:
        raise SystemExit("No requested size is present in both CSVs.")
    print(f"  Sizes drawn: {sizes}")

    suffix = 'N' + '_'.join(str(n) for n in sizes)
    plot_fcd(d35, d28, sizes, args.min_tau, out_dir,
             f'fcd_28GHz_vs_3p5GHz_{suffix}')
    plot_fmem(d35, d28, sizes, args.min_tau, out_dir,
              f'fmem_28GHz_vs_3p5GHz_{suffix}')
    print("Done.")


if __name__ == "__main__":
    main()
