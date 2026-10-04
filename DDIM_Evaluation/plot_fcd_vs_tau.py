"""
plot_fcd_vs_tau.py
    — Reproduce Figure 2 (left) of 2505.17638v2 for wireless channels
================================================================================
Reads results/fcd_vs_tau/fcd_vs_tau_<weights>.csv (from compute_fcd_vs_tau.py)
and draws the paper's Figure-2-left visual language:

  • solid lines  = FCD(Gen, Test)     one per dataset size N   (quality ↓, LEFT axis)
                   with a shaded 2σ band over the 5 test folds;
  • dotted lines = FCD(Train, Test)   the real–real floor per N;
  • dashed lines = f_mem(τ)           memorization fraction (OPTIONAL, RIGHT axis),
                   taken from your dedicated f_mem metric — the exact analogue of
                   the paper's dashed curves.

x-axis = τ (optimizer updates) on a log scale, exactly like the paper.

The good-quality / generalization window for each N is the τ-band where its
solid curve has flattened onto its dotted floor while f_mem is still ~0.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_fcd_vs_tau.py                                  # quality only
    python plot_fcd_vs_tau.py --fmem_csv results/fmem_vs_tau/fmem_vs_tau.csv
    python plot_fcd_vs_tau.py --weights raw --min_tau 1000
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


def _c(N):
    return COLORS.get(N, '#333333')


def _m(N):
    return MARKERS.get(N, 'o')


# Optional per-size minimum-epoch rule.  DISABLED by default (empty map) so the
# plots show every checkpoint.  It can still be enabled explicitly via
# --min_epochs (e.g. "200:150,1000:1000,4000:750") if ever wanted.
DEFAULT_MIN_EPOCHS = {}


def _steps_per_epoch(N):
    return int(np.ceil(N / 100))


def _epochs(sub, N):
    """Epoch count per row: prefer the CSV's epoch_float, else derive from τ."""
    if 'epoch_float' in sub.columns:
        return sub['epoch_float'].values
    return sub['tau'].values / _steps_per_epoch(N)


def parse_min_epochs(spec):
    """Parse --min_epochs into {N: min_epochs}.

    Accepts:  None                → DEFAULT_MIN_EPOCHS
              "750"               → same threshold for every N
              "200:150,1000:1000" → explicit per-size mapping
    """
    if spec is None:
        return dict(DEFAULT_MIN_EPOCHS)
    spec = str(spec).strip()
    if ':' not in spec:                       # single scalar for all sizes
        val = float(spec)
        return {'__all__': val}
    out = {}
    for part in spec.split(','):
        k, v = part.split(':')
        out[int(k)] = float(v)
    return out


def _min_ep_for(N, min_ep_map):
    if '__all__' in min_ep_map:
        return min_ep_map['__all__']
    return min_ep_map.get(int(N), 0.0)


def _apply_epoch_cut(sub, N, min_ep_map):
    """Keep only rows at/above the size's minimum-epoch threshold."""
    thr = _min_ep_for(N, min_ep_map)
    if thr <= 0:
        return sub
    return sub[_epochs(sub, N) >= thr]


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


def _threshold_note(df, min_ep_map):
    """Human-readable disclosure string of the per-size epoch cut."""
    parts = []
    for N in sorted(df['N'].unique()):
        thr = _min_ep_for(N, min_ep_map)
        if thr > 0:
            parts.append(f"N={N}: ≥{int(thr)} ep")
    if not parts:
        return None
    return "min-epoch cut  (" + ",  ".join(parts) + ")"


def plot_fig2_left(df, output_dir, weights, min_tau=0, fmem=None, min_ep_map=None):
    """All N overlaid, paper Figure-2-left style (quality + optional f_mem)."""
    min_ep_map = min_ep_map or {}
    fig, axL = plt.subplots(figsize=(11, 6.5))
    axR = axL.twinx() if fmem else None

    for N in sorted(df['N'].unique()):
        sub = df[df['N'] == N].sort_values('tau')
        # tau=0 is the random-init model (FCD is a huge outlier and cannot be
        # shown on a log-x axis); require strictly positive tau so it does not
        # blow up the y-axis and crush the meaningful points against zero.
        sub = sub[(sub['tau'] >= min_tau) & (sub['tau'] > 0)]
        # Per-size epoch cut: remove each size's pre-convergence noisy transient.
        sub = _apply_epoch_cut(sub, N, min_ep_map)
        if sub.empty:
            continue
        tau = sub['tau'].values
        gt = sub['FCD_Gen_Test'].values
        gt_sd = sub['FCD_Gen_Test_std'].values
        bench = sub['FCD_Train_Test'].values
        bench_sd = sub['FCD_Train_Test_std'].values

        axL.plot(tau, gt, '-', color=_c(N), lw=2.2, marker=_m(N), ms=5)
        if np.any(gt_sd > 0):
            axL.fill_between(tau, gt - 2 * gt_sd, gt + 2 * gt_sd,
                             color=_c(N), alpha=0.15, linewidth=0)
        if len(bench):
            axL.axhline(bench[0], color=_c(N), ls=':', lw=1.4, alpha=0.9)
            if bench_sd[0] > 0:
                axL.axhspan(bench[0] - 2 * bench_sd[0], bench[0] + 2 * bench_sd[0],
                            color=_c(N), alpha=0.06, linewidth=0)

        if axR is not None and N in fmem:
            fm = fmem[N]
            thr = _min_ep_for(N, min_ep_map)
            fm = fm[(fm['tau'] >= min_tau) & (fm['tau'] > 0)
                    & (fm['tau'] / _steps_per_epoch(N) >= thr)]
            axR.plot(fm['tau'].values, fm['fmem'].values, '--',
                     color=_c(N), lw=1.8, alpha=0.85)

    axL.set_xscale('log')
    axL.set_ylim(bottom=0)
    axL.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=18)
    axL.set_ylabel('FCD  (Fréchet Channel Distance)  ↓', fontsize=18)
    axL.grid(True, alpha=0.3)
    axL.tick_params(labelsize=16)
    if axR is not None:
        axR.set_ylabel(r'$f_{\mathrm{mem}}(\tau)$', fontsize=18)
        axR.set_ylim(bottom=0)
        axR.tick_params(labelsize=16)

    size_handles = [Line2D([0], [0], color=_c(N), marker=_m(N), lw=2.2,
                           ls='-', ms=7, label=f'N={N}')
                    for N in sorted(df['N'].unique())]
    style_handles = [
        Line2D([0], [0], color='k', ls='-',  lw=2.2, label='Gen–Test (quality)'),
        Line2D([0], [0], color='k', ls=':',  lw=1.4, label='Train–Test (real–real floor)'),
    ]
    if axR is not None:
        style_handles.append(
            Line2D([0], [0], color='k', ls='--', lw=1.8, label=r'$f_{\mathrm{mem}}$ (memorization)'))
    leg1 = axL.legend(handles=size_handles, fontsize=13, loc='upper right',
                      framealpha=0.9, title='dataset size')
    axL.add_artist(leg1)
    axL.legend(handles=style_handles, fontsize=12, loc='upper left', framealpha=0.9)

    note = _threshold_note(df, min_ep_map)
    if note:
        axL.text(0.5, -0.14, note, transform=axL.transAxes, ha='center',
                 va='top', fontsize=11, style='italic', color='#444444')

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'fcd_vs_tau_{weights}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


def plot_per_n_panels(df, output_dir, weights, min_tau=0, fmem=None, min_ep_map=None):
    """One panel per N (clearer view of each window)."""
    min_ep_map = min_ep_map or {}
    Ns = sorted(df['N'].unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(6.0 * len(Ns), 5.2), squeeze=False)
    for j, N in enumerate(Ns):
        ax = axes[0][j]
        sub = df[df['N'] == N].sort_values('tau')
        # drop tau=0 random-init spike (see note in plot_fig2_left)
        sub = sub[(sub['tau'] >= min_tau) & (sub['tau'] > 0)]
        # per-size epoch cut (remove pre-convergence transient)
        sub = _apply_epoch_cut(sub, N, min_ep_map)
        tau = sub['tau'].values
        gt = sub['FCD_Gen_Test'].values
        gt_sd = sub['FCD_Gen_Test_std'].values
        ax.plot(tau, gt, '-', color=_c(N), lw=2.2, marker=_m(N), ms=5,
                label='Gen–Test')
        if np.any(gt_sd > 0):
            ax.fill_between(tau, gt - 2 * gt_sd, gt + 2 * gt_sd,
                            color=_c(N), alpha=0.15, linewidth=0)
        bench = sub['FCD_Train_Test'].values
        if len(bench):
            ax.axhline(bench[0], color='k', ls=':', lw=1.4, alpha=0.8,
                       label='Train–Test floor')
        ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        thr = _min_ep_for(N, min_ep_map)
        ttl = f'N = {N}' + (f'   (≥{int(thr)} ep)' if thr > 0 else '')
        ax.set_title(ttl, fontsize=16)
        ax.set_xlabel(r'$\tau$', fontsize=15)
        if j == 0:
            ax.set_ylabel('FCD ↓', fontsize=15)
        ax.grid(True, alpha=0.3)

        if fmem is not None and N in fmem:
            axr = ax.twinx()
            fm = fmem[N]
            fm = fm[(fm['tau'] >= min_tau) & (fm['tau'] > 0)
                    & (fm['tau'] / _steps_per_epoch(N) >= thr)]
            axr.plot(fm['tau'].values, fm['fmem'].values, '--', color=_c(N),
                     lw=1.8, alpha=0.85, label=r'$f_{\mathrm{mem}}$')
            axr.set_ylim(bottom=0)
            if j == len(Ns) - 1:
                axr.set_ylabel(r'$f_{\mathrm{mem}}$', fontsize=15)
        ax.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(output_dir, f'fcd_panels_{weights}.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot FCD vs τ (Figure 2 left analogue)")
    ap.add_argument("--weights", choices=['ema', 'raw'], default='ema')
    ap.add_argument("--min_tau", type=int, default=0)
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--fmem_csv", type=str, default=None,
                    help="Optional f_mem CSV to overlay (dashed, right axis)")
    ap.add_argument("--min_epochs", type=str, default=None,
                    help="Per-size minimum-epoch cut to remove each size's noisy "
                         "pre-convergence transient. Either a single number "
                         "(same for all N) or a per-size map like "
                         "'200:150,1000:1000,4000:750'. Omit to use smart defaults; "
                         "pass '0' to disable.")
    ap.add_argument("--output_dir", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(base, 'results', 'fcd_vs_tau')
    csv_path = args.csv_path or os.path.join(default_dir, f'fcd_vs_tau_{args.weights}.csv')
    out_dir = args.output_dir or os.path.join(default_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}\n"
                                f"Run: python compute_fcd_vs_tau.py --weights {args.weights}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  N values : {sorted(df['N'].unique())}")

    fmem = _load_fmem(args.fmem_csv)
    if fmem is not None:
        print(f"  f_mem overlay: {sorted(fmem.keys())}")

    min_ep_map = parse_min_epochs(args.min_epochs)
    note = _threshold_note(df, min_ep_map)
    print(f"  {note if note else 'min-epoch cut: disabled'}")

    plot_fig2_left(df, out_dir, args.weights, min_tau=args.min_tau, fmem=fmem,
                   min_ep_map=min_ep_map)
    plot_per_n_panels(df, out_dir, args.weights, min_tau=args.min_tau, fmem=fmem,
                      min_ep_map=min_ep_map)
    print("Done.")


if __name__ == "__main__":
    main()
