"""
plot_fulldim_distance_vs_tau.py
    — Figure-2-style plots for the full 256-D distribution distances vs τ
================================================================================
Reads results/fulldim_distance_vs_tau_<weights>.csv (from
compute_fulldim_distance_vs_tau.py) and draws the same visual language as the
FCD plot, but for the two full-dimensional metrics:

    • Sinkhorn-W2 (entropic OT, 256-D)
    • SWD-W2      (Sliced-Wasserstein, 256-D)

For each metric:
  • solid lines  = Metric(Gen, Test)   one per dataset size N   (quality ↓, LEFT axis)
                   with a shaded 2σ band over the 5 test folds;
  • dotted lines = Metric(Train, Test) the real–real floor per N;
  • dashed lines = f_mem(τ)            memorization fraction (OPTIONAL, RIGHT axis).

x-axis = τ (optimizer updates) on a log scale.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/FullDim_Distance
    python plot_fulldim_distance_vs_tau.py
    python plot_fulldim_distance_vs_tau.py --fmem_csv ../results/fmem_vs_tau/fmem_vs_tau.csv
    python plot_fulldim_distance_vs_tau.py --weights raw --min_tau 1000
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

COLORS = {200: '#ff7f00', 1000: '#4daf4a', 4000: '#d62728',
          100: '#e41a1c', 500: '#984ea3', 2000: '#377eb8', 8000: '#a65628'}
MARKERS = {200: 's', 1000: 'D', 4000: 'P',
           100: 'o', 500: '^', 2000: 'v', 8000: 'X'}

# (csv column prefix, y-axis label) for each metric panel
METRICS = [
    ('Sinkhorn', 'Sinkhorn-W2  (entropic OT, 256-D)  ↓'),
    ('SWD',      'SWD-W2  (Sliced-Wasserstein, 256-D)  ↓'),
]


def _c(N):
    return COLORS.get(N, '#333333')


def _m(N):
    return MARKERS.get(N, 'o')


def _steps_per_epoch(N):
    return int(np.ceil(N / 100))


def _load_fmem(fmem_csv):
    """Return {N: DataFrame(tau, fmem)} or None. Tolerant to column naming."""
    if not fmem_csv or not os.path.exists(fmem_csv):
        return None
    fm = pd.read_csv(fmem_csv)
    cols = {c.lower(): c for c in fm.columns}
    tau_c = next((cols[k] for k in ('actual_tau', 'tau', 'target_tau') if k in cols), None)
    N_c = next((cols[k] for k in ('n', 'n_train', 'num_train') if k in cols), None)
    fmem_c = next((cols[k] for k in ('f_mem', 'fmem', 'fraction', 'f_mem_mean',
                                     'f_mem_k_1_3') if k in cols), None)
    if tau_c is None or N_c is None or fmem_c is None:
        print(f"  [fmem] could not identify columns in {fmem_csv} "
              f"(have {list(fm.columns)}) — skipping overlay.")
        return None
    out = {}
    for N in sorted(fm[N_c].unique()):
        sub = fm[fm[N_c] == N].sort_values(tau_c)
        out[int(N)] = sub.rename(columns={tau_c: 'tau', fmem_c: 'fmem'})[['tau', 'fmem']]
    return out


def _plot_one_metric(ax, df, prefix, ylabel, min_tau, fmem):
    """Draw one metric panel (quality + floor + optional f_mem twin axis)."""
    gt_col, gt_sd = f'{prefix}_Gen_Test', f'{prefix}_Gen_Test_std'
    tt_col, tt_sd = f'{prefix}_Train_Test', f'{prefix}_Train_Test_std'
    axR = ax.twinx() if fmem else None

    for N in sorted(df['N'].unique()):
        sub = df[df['N'] == N].sort_values('tau')
        sub = sub[(sub['tau'] >= min_tau) & (sub['tau'] > 0)]
        if sub.empty:
            continue
        tau = sub['tau'].values
        gt = sub[gt_col].values
        gt_s = sub[gt_sd].values
        bench = sub[tt_col].values
        bench_s = sub[tt_sd].values

        ax.plot(tau, gt, '-', color=_c(N), lw=2.2, marker=_m(N), ms=5)
        if np.any(gt_s > 0):
            ax.fill_between(tau, gt - 2 * gt_s, gt + 2 * gt_s,
                            color=_c(N), alpha=0.15, linewidth=0)
        if len(bench):
            ax.axhline(bench[0], color=_c(N), ls=':', lw=1.4, alpha=0.9)
            if bench_s[0] > 0:
                ax.axhspan(bench[0] - 2 * bench_s[0], bench[0] + 2 * bench_s[0],
                           color=_c(N), alpha=0.06, linewidth=0)

        if axR is not None and N in fmem:
            fm = fmem[N]
            fm = fm[(fm['tau'] >= min_tau) & (fm['tau'] > 0)]
            axR.plot(fm['tau'].values, fm['fmem'].values, '--',
                     color=_c(N), lw=1.8, alpha=0.85)

    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=14)
    if axR is not None:
        axR.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$', fontsize=15)
        axR.set_ylim(bottom=0)
        axR.tick_params(labelsize=14)


def main():
    ap = argparse.ArgumentParser(
        description="Plot full 256-D distribution distances (Sinkhorn + SWD) vs τ")
    ap.add_argument("--weights", choices=['ema', 'raw'], default='ema')
    ap.add_argument("--csv", type=str, default=None)
    ap.add_argument("--fmem_csv", type=str, default=None)
    ap.add_argument("--min_tau", type=float, default=0)
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    csv_path = args.csv or os.path.join(
        results_dir, f'fulldim_distance_vs_tau_{args.weights}.csv')
    out_dir = args.output_dir or os.path.join(results_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV not found: {csv_path}\nRun compute_fulldim_distance_vs_tau.py first.")

    df = pd.read_csv(csv_path)
    fmem = _load_fmem(args.fmem_csv)

    # Combined figure: one panel per metric
    fig, axes = plt.subplots(1, len(METRICS), figsize=(8.0 * len(METRICS), 6.0),
                             squeeze=False)
    for ax, (prefix, ylabel) in zip(axes[0], METRICS):
        _plot_one_metric(ax, df, prefix, ylabel, args.min_tau, fmem)

    # Shared legends
    size_handles = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2,
                           ls='-', ms=7, label=f'N={N}')
                    for N in sorted(df['N'].unique())]
    style_handles = [
        Line2D([0], [0], color='k', ls='-', lw=2.2, label='Gen–Test (quality)'),
        Line2D([0], [0], color='k', ls=':', lw=1.4, label='Train–Test (real–real floor)'),
    ]
    if fmem is not None:
        style_handles.append(
            Line2D([0], [0], color='k', ls='--', lw=1.8, label=r'$f_{\mathrm{mem}}$ (memorization)'))
    leg1 = axes[0][0].legend(handles=size_handles, fontsize=12, loc='upper right',
                             framealpha=0.9, title='dataset size')
    axes[0][0].add_artist(leg1)
    axes[0][-1].legend(handles=style_handles, fontsize=11, loc='upper right',
                       framealpha=0.9)

    fig.suptitle('Full 256-D distribution distance:  DDIM-generated  vs  held-out TEST',
                 fontsize=16)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'fulldim_distance_vs_tau_{args.weights}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()

    # Also save each metric as its own single-panel figure
    for prefix, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(11, 6.5))
        _plot_one_metric(ax, df, prefix, ylabel, args.min_tau, fmem)
        h_size = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2, ls='-',
                         ms=7, label=f'N={N}') for N in sorted(df['N'].unique())]
        leg = ax.legend(handles=h_size, fontsize=13, loc='upper right',
                        framealpha=0.9, title='dataset size')
        ax.add_artist(leg)
        ax.legend(handles=style_handles, fontsize=12, loc='upper left', framealpha=0.9)
        plt.tight_layout()
        for ext in ('png', 'pdf'):
            p = os.path.join(out_dir, f'{prefix.lower()}_vs_tau_{args.weights}.{ext}')
            plt.savefig(p, dpi=300, bbox_inches='tight')
            print(f"  Saved: {p}")
        plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
