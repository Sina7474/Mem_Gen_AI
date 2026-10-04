"""
plot_bs_fcd_fmem.py
    — Batch-size sweep: FCD(Gen,Test) + f_mem vs τ  (Figure-2-left visual language)
================================================================================
Reads results/bs_fcd_fmem_N{N}.csv (produced by compute_bs_fcd_fmem.py) and draws,
for a FIXED dataset size N (default 1000), one curve per training batch size:

  • solid  line (LEFT axis)  = FCD(Gen, Test)     — sample quality (↓, want it low)
                               with a shaded 2σ band over the 5 test folds;
  • dotted line (LEFT axis)  = FCD(Train, Test)   — the real–real floor
                               (identical for every batch size: same train/test split);
  • dashed line (RIGHT axis) = f_mem(τ)           — memorization fraction.

Color encodes the batch size; line style encodes the quantity, exactly matching the
look of ../plot_fcd_vs_tau.py. x-axis = τ (optimizer updates, log scale), so the three
batch sizes are directly comparable at equal numbers of gradient steps.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/batch_size_effect
    python plot_bs_fcd_fmem.py                 # N=1000
    python plot_bs_fcd_fmem.py --N 1000 --min_tau 1000
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

# Distinct color per batch size (line style is reserved for the quantity).
BS_COLORS  = {100: '#377eb8', 512: '#ff7f00', 1024: '#d62728'}
BS_MARKERS = {100: 'o', 512: 's', 1024: 'D'}


def _c(bs):
    return BS_COLORS.get(bs, '#333333')


def _m(bs):
    return BS_MARKERS.get(bs, 'o')


def main():
    ap = argparse.ArgumentParser(
        description="Plot FCD(Gen,Test)+f_mem vs τ across training batch sizes")
    ap.add_argument("--N", type=int, default=1000, help="Dataset size (default 1000)")
    ap.add_argument("--min_tau", type=int, default=1000,
                    help="Drop τ below this (removes the τ=0 random-init spike)")
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', f'bs_fcd_fmem_N{args.N}.csv')
    out_dir  = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Run: python compute_bs_fcd_fmem.py --N {args.N}")

    df = pd.read_csv(csv_path)
    df = df[(df['N'] == args.N) & (df['tau'] >= args.min_tau)].sort_values(['batch_size', 'tau'])
    batch_sizes = sorted(df['batch_size'].unique())
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Batch sizes: {batch_sizes}")

    fig, axL = plt.subplots(figsize=(10, 6))
    axR = axL.twinx()

    floor_val = None
    for bs in batch_sizes:
        sub = df[df['batch_size'] == bs].sort_values('tau')
        tau   = sub['tau'].values
        gt    = sub['FCD_Gen_Test'].values
        gt_sd = sub['FCD_Gen_Test_std'].values
        fmem  = sub['f_mem'].values

        # FCD(Gen,Test): solid + 2σ band
        axL.plot(tau, gt, '-', color=_c(bs), lw=2.2, marker=_m(bs), ms=5, alpha=0.95)
        if np.any(gt_sd > 0):
            axL.fill_between(tau, gt - 2 * gt_sd, gt + 2 * gt_sd,
                             color=_c(bs), alpha=0.12, linewidth=0)

        # f_mem: dashed on the right axis
        axR.plot(tau, fmem, '--', color=_c(bs), lw=1.8, alpha=0.85)

        # Real–real floor is identical across batch sizes (same split); keep one.
        if floor_val is None and len(sub):
            floor_val = float(sub['FCD_Train_Test'].values[0])
            floor_sd  = float(sub['FCD_Train_Test_std'].values[0])

    # Single shared real–real floor (dotted, black).
    if floor_val is not None:
        axL.axhline(floor_val, color='k', ls=':', lw=1.6, alpha=0.9)
        if floor_sd > 0:
            axL.axhspan(floor_val - 2 * floor_sd, floor_val + 2 * floor_sd,
                        color='k', alpha=0.06, linewidth=0)

    axL.set_xscale('log')
    axL.set_ylim(bottom=0)
    axR.set_ylim(bottom=0)
    axL.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=18)
    axL.set_ylabel('FCD  (Fréchet Channel Distance)  ↓', fontsize=18)
    axR.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$', fontsize=18)
    axL.tick_params(labelsize=16)
    axR.tick_params(labelsize=16)
    axL.grid(True, alpha=0.3)

    # Legend 1: batch size (color).  Legend 2: quantity (line style).
    bs_handles = [Line2D([0], [0], color=_c(bs), marker=_m(bs), lw=2.2, ls='-',
                         ms=7, label=f'batch size = {bs}') for bs in batch_sizes]
    style_handles = [
        Line2D([0], [0], color='k', ls='-',  lw=2.2, label='Gen–Test (quality)'),
        Line2D([0], [0], color='k', ls=':',  lw=1.6, label='Train–Test (real–real floor)'),
        Line2D([0], [0], color='k', ls='--', lw=1.8, label=r'$f_{\mathrm{mem}}$ (memorization)'),
    ]
    leg1 = axL.legend(handles=bs_handles, fontsize=13, loc='upper center',
                      framealpha=0.9, title=f'N = {args.N}',
                      bbox_to_anchor=(0.5, 0.99))
    axL.add_artist(leg1)
    axL.legend(handles=style_handles, fontsize=12, loc='upper center',
               framealpha=0.9, bbox_to_anchor=(0.5, 0.72))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'bs_fcd_fmem_N{args.N}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
