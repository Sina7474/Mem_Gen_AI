"""
plot_fmem_28GHz_vs_3p5GHz.py
    — Cross-band comparison of the memorization curve f_mem(τ)
================================================================================
Overlays the memorization fraction f_mem(τ) measured in the two propagation
scenes for the SAME dataset-size sweep and the SAME training protocol:

    3.5 GHz (LoS + NLoS)   ../dataset_size_effect/results/dsize_fcd_fmem.csv
    28  GHz (LoS only)     results/dsize_fmem_28GHz_LoS.csv

Both were produced with width W = 256, batch rule B = min(N, 500), the same dense
τ grid, the same 256-D per-sample-normalized UPA-DFT beamspace representation,
5000 EMA-generated samples per checkpoint, and the same nearest-neighbour ratio
memorization test (ρ = d1/d2 < k = 1/3). The curves are therefore directly
comparable and any difference is attributable to the scene / carrier frequency.

Encoding
--------
    colour     → dataset size N   (colour-vision-deficiency-safe viridis ramp)
    solid  + filled marker → 3.5 GHz
    dashed + open  marker → 28 GHz
    shaded band → 95 % bootstrap CI

Outputs
-------
    results/figures/fmem_28GHz_vs_3p5GHz.{png,pdf}
    results/fmem_band_comparison.csv   (peak f_mem, τ at peak, 10 % onset, final)

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect_28GHz_LoS
    python plot_fmem_28GHz_vs_3p5GHz.py
    python plot_fmem_28GHz_vs_3p5GHz.py --sizes 200 1000 4000
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

# Per-band line style: (linestyle, marker fill, line width, label)
BANDS = {
    '3.5 GHz': dict(ls='-',  fill=True,  lw=2.2),
    '28 GHz':  dict(ls='--', fill=False, lw=2.2),
}

ONSET_LEVEL = 0.10   # f_mem threshold used to define the memorization onset τ


def _c(N):
    return N_COLORS.get(N, '#333333')


def _m(N):
    return N_MARKERS.get(N, 'o')


def _onset_tau(tau, fmem, level=ONSET_LEVEL):
    """First τ at which f_mem rises above `level`, log-linearly interpolated.

    Returns NaN when the curve never reaches the level within the measured range.
    """
    tau = np.asarray(tau, dtype=float)
    fmem = np.asarray(fmem, dtype=float)
    above = np.nonzero(fmem >= level)[0]
    if len(above) == 0:
        return float('nan')
    i = above[0]
    if i == 0:
        return float(tau[0])
    f0, f1 = fmem[i - 1], fmem[i]
    if f1 == f0:
        return float(tau[i])
    w = (level - f0) / (f1 - f0)
    return float(np.exp(np.log(tau[i - 1]) + w * (np.log(tau[i]) - np.log(tau[i - 1]))))


def main():
    ap = argparse.ArgumentParser(
        description="Compare f_mem(τ) between the 28 GHz LoS and 3.5 GHz scenes")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Drop τ below this (removes the τ=0 random-init point)")
    ap.add_argument("--sizes", type=int, nargs='+', default=None,
                    help="Subset of dataset sizes N to plot (default: the sizes "
                         "present in BOTH CSVs)")
    ap.add_argument("--csv_28ghz", type=str, default=None)
    ap.add_argument("--csv_35ghz", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv28 = args.csv_28ghz or os.path.join(
        base, 'results', 'dsize_fmem_28GHz_LoS.csv')
    csv35 = args.csv_35ghz or os.path.join(
        base, '..', 'dataset_size_effect', 'results', 'dsize_fcd_fmem.csv')
    out_dir = args.output_dir or os.path.join(base, 'results', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    for p, hint in ((csv28, "python compute_dsize_fmem_28GHz_LoS.py"),
                    (csv35, "python ../dataset_size_effect/compute_dsize_fcd_fmem.py")):
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV not found: {os.path.normpath(p)}\nRun: {hint}")

    cols = ['N', 'tau', 'f_mem', 'f_mem_ci_low', 'f_mem_ci_high']
    d28 = pd.read_csv(csv28)[cols]
    d35 = pd.read_csv(csv35)[cols]
    data = {'28 GHz': d28, '3.5 GHz': d35}

    common = sorted(set(d28['N'].unique()) & set(d35['N'].unique()))
    if args.sizes is not None:
        missing = sorted(set(args.sizes) - set(common))
        if missing:
            print(f"  WARNING: sizes absent from one of the CSVs, ignored: {missing}")
        sizes = [n for n in args.sizes if n in common]
    else:
        sizes = common
    if not sizes:
        raise SystemExit("No dataset size is present in both CSVs — nothing to compare.")

    print(f"  28 GHz  CSV : {os.path.normpath(csv28)}  ({len(d28)} rows)")
    print(f"  3.5 GHz CSV : {os.path.normpath(csv35)}  ({len(d35)} rows)")
    print(f"  Common sizes: {sizes}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = []

    for band, style in BANDS.items():
        df = data[band]
        df = df[(df['tau'] >= args.min_tau) & (df['N'].isin(sizes))]
        for N in sizes:
            sub = df[df['N'] == N].sort_values('tau')
            if sub.empty:
                continue
            tau  = sub['tau'].values
            fmem = sub['f_mem'].values
            lo   = sub['f_mem_ci_low'].values * 100.0
            hi   = sub['f_mem_ci_high'].values * 100.0

            ax.plot(tau, fmem * 100.0, style['ls'], color=_c(N), lw=style['lw'],
                    marker=_m(N), ms=5.5, alpha=0.95,
                    markerfacecolor=(_c(N) if style['fill'] else 'none'),
                    markeredgecolor=_c(N), markeredgewidth=1.4)
            if np.any(hi - lo > 0):
                ax.fill_between(tau, lo, hi, color=_c(N), alpha=0.12, linewidth=0)

            i_pk = int(np.argmax(fmem))
            summary.append(dict(
                band=band, N=N,
                f_mem_peak=float(fmem[i_pk]),
                tau_at_peak=int(tau[i_pk]),
                tau_onset_10pct=_onset_tau(tau, fmem, ONSET_LEVEL),
                f_mem_final=float(fmem[-1]),
                tau_final=int(tau[-1]),
            ))

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$  (%)', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.grid(True, alpha=0.3)

    size_handles = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2, ls='-',
                           ms=7, label=f'N = {N}') for N in sizes]
    band_handles = [
        Line2D([0], [0], color='k', ls=BANDS['3.5 GHz']['ls'], lw=2.2, marker='o',
               ms=6, markerfacecolor='k', label='3.5 GHz (LoS+NLoS)'),
        Line2D([0], [0], color='k', ls=BANDS['28 GHz']['ls'], lw=2.2, marker='o',
               ms=6, markerfacecolor='none', label='28 GHz (LoS)'),
    ]
    leg1 = ax.legend(handles=size_handles, fontsize=16, loc='upper left',
                     framealpha=0.9, bbox_to_anchor=(0.01, 0.99))
    ax.add_artist(leg1)

    # Second legend placed just below the first, using its rendered height.
    fig.canvas.draw()
    bb = leg1.get_window_extent().transformed(ax.transAxes.inverted())
    ax.legend(handles=band_handles, fontsize=16, loc='upper left',
              framealpha=0.9, bbox_to_anchor=(0.01, bb.y0 - 0.02))

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'fmem_28GHz_vs_3p5GHz.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()

    # ── Summary table ────────────────────────────────────────────────────────
    sdf = pd.DataFrame(summary).sort_values(['N', 'band'])
    sum_path = os.path.join(base, 'results', 'fmem_band_comparison.csv')
    sdf.to_csv(sum_path, index=False)
    print(f"  Saved: {sum_path}\n")
    with pd.option_context('display.width', 160, 'display.max_columns', None):
        print(sdf.to_string(index=False,
                            float_format=lambda v: f"{v:.4g}"))
    print("\nDone.")


if __name__ == "__main__":
    main()
