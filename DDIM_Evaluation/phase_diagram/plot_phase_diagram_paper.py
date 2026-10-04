"""
plot_phase_diagram_paper.py
    — Paper-style generalization–memorization phase diagram (Task 16)
================================================================================
Reproduces the visual grammar of the reference figure in
"Why Diffusion Models Don't Memorize" (2505.17638v2):

    x-axis : p   = number of trainable parameters  (shown as ×10^7)
    y-axis : n   = training-set size N
    red region (bottom) : memorizing   —  f_mem(τ) > threshold
    green region (top)  : generalizing —  f_mem(τ) ≤ threshold
    boundary curves n*(p; τ) : minimum N above which the model does NOT memorize,
                               one rising curve per training budget τ, with a
                               black dot at each measured p (= each width W).

--------------------------------------------------------------------------------
Why iso-τ (fixed training budget) instead of k·τ_gen
--------------------------------------------------------------------------------
The reference paper measures memorization at multiples of the per-model
generalization time τ_gen (τ_gen, 3τ_gen, 8τ_gen). That works there because all
their models generalize at a comparable τ_gen. In OUR setup τ_gen scales strongly
with N (a tiny N generalizes in a few hundred steps, a large N needs tens of
thousands), so k·τ_gen mixes very different absolute times and INVERTS the
n-dependence. Measured at a FIXED absolute training budget τ, our data recovers
exactly the paper's structure:

    • f_mem decreases with N  (more data ⇒ slower/less memorization)
    • f_mem increases with p  (bigger model ⇒ memorizes sooner)
    ⇒ the non-memorization boundary n*(p) RISES with p, and grows with τ.

Each curve is therefore an "iso-training-time" boundary: for a model of size p
trained for τ optimizer steps, n*(p; τ) is the least data needed to stay out of
the memorizing (red) regime. Longer τ ⇒ higher curve (needs more data), mirroring
the paper's τ_gen < 3τ_gen < 8τ_gen < n*(p) ordering.

--------------------------------------------------------------------------------
Boundary estimation (honest about the sparse grid)
--------------------------------------------------------------------------------
For a fixed width W (parameter count p) and budget τ:
  1. interpolate f_mem at τ (log-linear in τ) for every available N;
  2. f_mem(N) is monotone-decreasing, so find the crossing f_mem = threshold by
     linear interpolation in log10(N);
  3. if every N already memorizes  → boundary is ABOVE the sampled range (capped
     at the top, drawn as an open marker + dashed = extrapolated);
     if every N is already safe     → boundary is BELOW the sampled range.
W=64 and W=128 have only two N values, so their boundaries are weakly constrained
and any capped/extrapolated point is flagged.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/phase_diagram
    python plot_phase_diagram_paper.py
    python plot_phase_diagram_paper.py --taus 20000 35000 50000 100000
    python plot_phase_diagram_paper.py --fmem_threshold 0.01 --ymax 4300
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

PARAM_COUNTS = {64: 956930, 128: 3814402, 256: 15230978}

# Line style per training-budget curve (lowest τ = solid, like the paper's τ_gen).
CURVE_STYLES = ['-', '-.', ':', '--', (0, (5, 1))]


# ------------------------------------------------------------- data loading ----
def load_long_table(dsize_csv, wsize_csv):
    frames = []
    if os.path.exists(dsize_csv):
        d = pd.read_csv(dsize_csv)
        d['W'] = d['n_feat'].astype(int)
        frames.append(d[['W', 'N', 'tau', 'f_mem']])
    if os.path.exists(wsize_csv):
        w = pd.read_csv(wsize_csv)
        w['W'] = w['W'].astype(int)
        w = w[w['W'] != 256]                      # W=256 comes from the dsize sweep
        frames.append(w[['W', 'N', 'tau', 'f_mem']])
    if not frames:
        raise FileNotFoundError("No source CSVs found.")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(['W', 'N', 'tau'])
    return df.sort_values(['W', 'N', 'tau']).reset_index(drop=True)


def fmem_at_tau(g, tau_c):
    """Log-linear interpolation of f_mem at absolute budget τ_c (NaN if out of range)."""
    g = g.sort_values('tau')
    tau = g['tau'].values.astype(float)
    f   = g['f_mem'].values.astype(float)
    m = tau > 0
    tau, f = tau[m], f[m]
    if len(tau) == 0 or tau_c < tau[0] or tau_c > tau[-1]:
        return np.nan
    return float(np.interp(np.log10(tau_c), np.log10(tau), f))


def boundary_N(df_W, tau_c, threshold):
    """Least N with f_mem(τ_c) ≤ threshold for a fixed width.
    Returns (N_star, status) with status in {'ok', 'above_range', 'below_range'}."""
    Ns = sorted(df_W['N'].unique())
    fs = []
    for N in Ns:
        fs.append(fmem_at_tau(df_W[df_W['N'] == N], tau_c))
    Ns = np.array(Ns, float)
    fs = np.array(fs, float)
    ok = np.isfinite(fs)
    Ns, fs = Ns[ok], fs[ok]
    if len(Ns) == 0:
        return np.nan, 'above_range'
    # f decreases with N; find first N (ascending) that is already safe.
    safe = fs <= threshold
    if safe.all():
        return float(Ns.min()), 'below_range'      # even smallest N is safe
    if not safe.any():
        return float(Ns.max()), 'above_range'       # even largest N memorizes
    # crossing between the last memorizing and the first safe N
    i = np.argmax(safe)                              # first True
    N0, N1 = Ns[i - 1], Ns[i]
    f0, f1 = fs[i - 1], fs[i]
    t = (f0 - threshold) / (f0 - f1) if f0 != f1 else 0.5
    logN = np.log10(N0) + t * (np.log10(N1) - np.log10(N0))
    return float(10 ** logN), 'ok'


# ------------------------------------------------------------------- plotting --
def main():
    ap = argparse.ArgumentParser(description="Paper-style (p, n) memorization "
                                             "phase diagram with iso-τ boundaries.")
    ap.add_argument('--dsize_csv', type=str, default=None)
    ap.add_argument('--wsize_csv', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None)
    ap.add_argument('--fmem_threshold', type=float, default=0.01)
    ap.add_argument('--taus', type=int, nargs='+',
                    default=[20000, 35000, 50000, 100000],
                    help="Absolute training budgets τ for the boundary curves "
                         "(ascending; lowest is the red/green split).")
    ap.add_argument('--ymax', type=float, default=4300.0)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    dsize_csv = args.dsize_csv or os.path.join(
        base, '..', 'dataset_size_effect', 'results', 'dsize_fcd_fmem.csv')
    wsize_csv = args.wsize_csv or os.path.join(
        base, '..', 'model_size_effect', 'results', 'wsize_fcd_fmem.csv')
    out_dir = args.output_dir or os.path.join(base, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    df = load_long_table(dsize_csv, wsize_csv)
    widths = sorted(df['W'].unique())
    ps = np.array([PARAM_COUNTS[w] for w in widths], float)
    taus = sorted(args.taus)
    thr = args.fmem_threshold

    # Compute every boundary curve n*(p; τ).
    print(f"Boundary n*(p; τ)  (threshold f_mem = {thr}):")
    curves = {}
    for tau_c in taus:
        pts, stats = [], []
        for W in widths:
            Nstar, st = boundary_N(df[df['W'] == W], tau_c, thr)
            pts.append(Nstar); stats.append(st)
        curves[tau_c] = (np.array(pts, float), stats)
        pretty = "  ".join(f"W{W}:{n:6.0f}({s})"
                           for W, n, s in zip(widths, pts, stats))
        print(f"  τ={tau_c:>7}:  {pretty}")

    # ---- figure -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    x_dense = np.linspace(ps.min() * 0.0, ps.max() * 1.12, 400)

    # Master red/green split = the LOWEST-τ boundary (the earliest budget at which
    # memorization is assessed), interpolated across p.
    lo_tau = taus[0]
    lo_N, _ = curves[lo_tau]
    y_split = np.interp(x_dense, ps, lo_N)
    ax.fill_between(x_dense, 0, y_split, color='#f4a6a6', alpha=0.85, zorder=0)
    ax.fill_between(x_dense, y_split, args.ymax, color='#b6e2b6', alpha=0.85,
                    zorder=0)

    # Boundary curves.
    curve_handles = []
    for idx, tau_c in enumerate(taus):
        Nstar, stats = curves[tau_c]
        ls = CURVE_STYLES[idx % len(CURVE_STYLES)]
        # Split markers: filled = interpolated 'ok', open = capped/extrapolated.
        ax.plot(ps, Nstar, ls=ls, color='k', lw=2.0, zorder=3)
        for p, n, st in zip(ps, Nstar, stats):
            face = 'k' if st == 'ok' else 'white'
            ax.plot(p, n, marker='o', ms=7, mfc=face, mec='k', mew=1.4, zorder=4)
        # Label box near the right end of each curve (paper style).
        is_top = (idx == len(taus) - 1)
        lbl = (r'$n=n^{\star}(p)$' if is_top
               else rf'$\tau={tau_c//1000}\mathrm{{k}}$')
        ax.annotate(lbl, (ps[-1], Nstar[-1]),
                    xytext=(6, 6 if not is_top else 8), textcoords='offset points',
                    fontsize=13, bbox=dict(boxstyle='round,pad=0.25', fc='white',
                                           ec='0.4', alpha=0.95), zorder=5)
        curve_handles.append(Line2D([0], [0], color='k', ls=ls, lw=2.0,
                                    label=(r'$n^{\star}(p)$ at $\tau_{\max}$-budget'
                                           if is_top
                                           else rf'$\tau={tau_c//1000}$k steps')))

    # Measured configurations as small ticks on the right edge (like the paper's
    # right-edge markers), plus the actual (p, N) grid points faintly.
    for W in widths:
        for N in sorted(df[df['W'] == W]['N'].unique()):
            ax.plot(PARAM_COUNTS[W], N, marker='+', ms=8, color='0.35',
                    mew=1.2, zorder=2)

    ax.set_xlim(0, ps.max() * 1.16)
    ax.set_ylim(0, args.ymax)
    ax.set_xlabel(r'$p$   (trainable parameters)', fontsize=18)
    ax.set_ylabel(r'$n$   (training-set size)', fontsize=18)
    ax.tick_params(labelsize=15)
    # ×10^7 style x-axis like the paper.
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((7, 7))
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.get_offset_text().set_fontsize(13)

    region_handles = [
        Patch(facecolor='#f4a6a6', edgecolor='none',
              label=r'$f_{\mathrm{mem}}(\tau) > $ thr  (memorizing)'),
        Patch(facecolor='#b6e2b6', edgecolor='none',
              label=r'$f_{\mathrm{mem}}(\tau) \leq $ thr  (generalizing)'),
    ]
    leg1 = ax.legend(handles=region_handles, fontsize=12, loc='upper left',
                     framealpha=0.95)
    ax.add_artist(leg1)
    ax.legend(handles=[Line2D([0], [0], marker='o', mfc='k', mec='k', ls='none',
                              ms=7, label='measured $p$ (interp. boundary)'),
                       Line2D([0], [0], marker='o', mfc='white', mec='k', ls='none',
                              ms=7, label='boundary outside $N$ range (capped)'),
                       Line2D([0], [0], marker='+', color='0.35', ls='none',
                              ms=8, label='trained $(p, N)$ config')],
              fontsize=10.5, loc='lower right', framealpha=0.95)

    ax.set_title('Generalization–memorization phase diagram '
                 r'(iso-$\tau$ boundaries $n^{\star}(p;\tau)$)', fontsize=13)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'phase_diagram_paper_style.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()
    print("Done.")


if __name__ == '__main__':
    main()
