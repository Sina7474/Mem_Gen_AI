"""
plot_fcd_fmem_28GHz_vs_3p5GHz.py
    — Cross-band comparison of FCD(Gen,Test) and f_mem(τ) on a single dual-axis figure
================================================================================
Overlays, for a small set of dataset sizes (default N = 200 and 1000), the two
paper metrics measured in the two propagation scenes:

    3.5 GHz (LoS + NLoS)   ../dataset_size_effect/results/dsize_fcd_fmem.csv
    28  GHz (LoS only)     results/dsize_fcd_fmem_28GHz_LoS.csv

Only a couple of sizes are drawn on purpose — plotting all five makes the figure
unreadable.

Encoding
--------
    colour        → dataset size N          (blue / orange)
    marker fill   → scene: filled = 3.5 GHz, open (larger) = 28 GHz
    solid  line   → FCD(Gen, Test)          (left axis)
    dashed line   → f_mem(τ)                (right axis)
    dotted line   → FCD(Train, Test) floor  (left axis, one per scene & N)

Marker SHAPE is deliberately identical for both scenes so the only shape cue is
filled vs. open — that is what separates the two carrier frequencies.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect_28GHz_LoS
    python plot_fcd_fmem_28GHz_vs_3p5GHz.py
    python plot_fcd_fmem_28GHz_vs_3p5GHz.py --sizes 200 1000 4000
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

MARKER = 'o'          # same shape for BOTH scenes — fill is the only shape cue
MS_FILLED = 6         # 3.5 GHz marker size
MS_OPEN = 9           # 28 GHz marker size (larger so the ring reads clearly)
MEW_OPEN = 1.8        # edge width of the open markers


def _c(N):
    return N_COLORS.get(N, '#333333')


def main():
    ap = argparse.ArgumentParser(
        description="FCD + f_mem vs τ, 28 GHz LoS against 3.5 GHz, for a few N")
    ap.add_argument("--sizes", type=int, nargs='+', default=[200, 1000],
                    help="Dataset sizes to draw (default: 200 1000). Keep this "
                         "list short — more than 2-3 sizes makes the figure "
                         "unreadable.")
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

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()

    for N in sizes:
        c = _c(N)

        # ── 3.5 GHz: FILLED markers ──────────────────────────────────────────
        s35 = d35[(d35.N == N) & (d35.tau >= args.min_tau)].sort_values('tau')
        axL.plot(s35['tau'], s35['FCD_Gen_Test'], '-', color=c, lw=2.2,
                 marker=MARKER, ms=MS_FILLED,
                 markerfacecolor=c, markeredgecolor=c, alpha=0.95)
        axL.axhline(s35['FCD_Train_Test'].iloc[0], color=c, ls=':', lw=1.2, alpha=0.5)
        axR.plot(s35['tau'], s35['f_mem'] * 100.0, '--', color=c, lw=2.0,
                 marker=MARKER, ms=MS_FILLED - 1,
                 markerfacecolor=c, markeredgecolor=c, alpha=0.85)

        # ── 28 GHz: OPEN markers (same shape, larger) ────────────────────────
        s28 = d28[(d28.N == N) & (d28.tau >= args.min_tau)].sort_values('tau')
        axL.plot(s28['tau'], s28['FCD_Gen_Test'], '-', color=c, lw=2.2,
                 marker=MARKER, ms=MS_OPEN,
                 markerfacecolor='none', markeredgecolor=c,
                 markeredgewidth=MEW_OPEN, alpha=0.95)
        axL.axhline(s28['FCD_Train_Test'].iloc[0], color=c, ls=':', lw=1.2, alpha=0.3)
        axR.plot(s28['tau'], s28['f_mem'] * 100.0, '--', color=c, lw=2.0,
                 marker=MARKER, ms=MS_OPEN - 1,
                 markerfacecolor='none', markeredgecolor=c,
                 markeredgewidth=MEW_OPEN, alpha=0.85)

    axL.set_xscale('log')
    axL.set_ylim(bottom=0)
    axR.set_ylim(bottom=0)
    axL.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    axL.set_ylabel('FCD (Gen, Test)  ↓', fontsize=20)
    axR.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$  (%)', fontsize=20)
    axL.tick_params(labelsize=18)
    axR.tick_params(labelsize=18)
    axL.grid(True, alpha=0.3)

    handles = [
        Line2D([0], [0], color=_c(N), marker=MARKER, ms=7, ls='-', lw=2.2,
               markerfacecolor=_c(N), label=f'N = {N}') for N in sizes
    ] + [
        Line2D([0], [0], color='k', marker=MARKER, ms=MS_FILLED, ls='none',
               markerfacecolor='k', label='3.5 GHz (LoS+NLoS)'),
        Line2D([0], [0], color='k', marker=MARKER, ms=MS_OPEN, ls='none',
               markerfacecolor='none', markeredgecolor='k',
               markeredgewidth=MEW_OPEN, label='28 GHz (LoS)'),
        Line2D([0], [0], color='grey', ls='-',  lw=2.2, label='FCD'),
        Line2D([0], [0], color='grey', ls='--', lw=2.0, label=r'$f_{\mathrm{mem}}$'),
        Line2D([0], [0], color='grey', ls=':',  lw=1.2, label='FCD floor'),
    ]
    axL.legend(handles=handles, fontsize=14, loc='upper center',
               framealpha=0.92, ncol=2, bbox_to_anchor=(0.42, 1.0))

    plt.tight_layout()
    stem = 'fcd_fmem_28GHz_vs_3p5GHz_N' + '_'.join(str(n) for n in sizes)
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'{stem}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
