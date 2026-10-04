"""
plot_phase_diagram.py
    — Generalization–memorization phase diagram for DDIM channel models (Task 16)
================================================================================
Reads results/phase_diagram_summary.csv (produced by build_phase_diagram_table.py)
and draws three complementary views of the SPARSE (W, N) grid. Because the grid is
sparse (W=64,128 have only N=200,1000; W=256 has N=200..4000), we never draw dense
continuous boundaries; interpolated segments are dashed and clearly flagged.

Figures written to figures/ (both .png and .pdf):

  1) phase_diagram_safe_points     — discrete scatter: x = model width W (or the
        parameter count p), y = training-set size N. Each measured (W,N) point is a
        marker colored by its phase label, annotated with the window ratio
        r = τ_mem / τ_gen.
  2) phase_diagram_boundaries      — the same measured points plus approximate
        generalization–memorization boundaries N_gm(W) at τ_gen, 3τ_gen, 8τ_gen,
        drawn only where measured points support them (dashed = inferred).
  3) phase_diagram_heatmap_table   — an honest N×W table (rows = N, cols = W); each
        cell is colored by phase (green = safe, olive = near-floor-safe, red =
        unsafe, gray = missing) and annotated with τ_gen / τ_mem (or f_mem(τ_gen)).

Phase labels (from the summary CSV):
    safe               green   — strict τ_gen reached and f_mem(τ_gen) < threshold
    near_floor_safe    olive   — τ_gen just missed the 10% floor tolerance, but
                                 f_mem at the best-quality τ is still < threshold
    unsafe             red     — high quality reached only after memorization began
    (missing)          gray    — no trained model at that (W,N)

Styling matches ../tau_plots/ (labels 20, ticks 18–20, legend 14–16, pdf.fonttype 42).

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/phase_diagram
    python plot_phase_diagram.py
    python plot_phase_diagram.py --xaxis p     # use parameter count on the x-axis
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
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Phase → color / marker.
PHASE_COLOR = {
    'safe':              '#2ca02c',   # green
    'near_floor_safe':   '#bcbd22',   # olive/yellow-green
    'unsafe':            '#d62728',   # red
    'near_floor_unsafe': '#ff7f0e',   # orange
    'missing':           '#cfcfcf',   # gray
}
PHASE_MARKER = {
    'safe':              'o',
    'near_floor_safe':   '^',
    'unsafe':            's',
    'near_floor_unsafe': 'v',
}
PHASE_LABEL_TEXT = {
    'safe':              'safe (quality before memorization)',
    'near_floor_safe':   'near-floor safe (quality plateaus $\\gtrsim$ floor, $f_{\\mathrm{mem}}\\approx 0$)',
    'unsafe':            'unsafe (memorization before quality)',
    'near_floor_unsafe': 'near-floor unsafe',
}


def _xval(row, xaxis):
    return row['parameter_count_p'] if xaxis == 'p' else row['W']


def _xlabel(xaxis):
    return ('model size  $p$  (trainable parameters)' if xaxis == 'p'
            else 'model width  $W$  (U-Net base channels)')


# ----------------------------------------------------------- 1) scatter points --
def plot_safe_points(df, out_dir, xaxis):
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, r in df.iterrows():
        ph = r['phase_label']
        x, y = _xval(r, xaxis), r['N']
        ax.scatter(x, y, s=320, marker=PHASE_MARKER.get(ph, 'o'),
                   color=PHASE_COLOR.get(ph, '#333333'),
                   edgecolors='k', linewidths=1.4, zorder=3)
        # Annotate with the window ratio r = τ_mem / τ_gen when available.
        wr = r['window_ratio_tau_mem_over_tau_gen']
        if np.isfinite(wr):
            txt = f"r={wr:.1f}"
        elif np.isfinite(r['tau_mem']):
            txt = "near-floor"
        else:
            txt = ""
        if txt:
            ax.annotate(txt, (x, y), textcoords='offset points', xytext=(10, 8),
                        fontsize=13, zorder=4)

    ax.set_xscale('log' if xaxis == 'p' else 'linear')
    ax.set_yscale('log')
    if xaxis == 'W':
        ax.set_xticks(sorted(df['W'].unique()))
        ax.set_xticklabels([str(int(w)) for w in sorted(df['W'].unique())])
        ax.set_xlim(min(df['W']) * 0.6, max(df['W']) * 1.5)
    ax.set_yticks(sorted(df['N'].unique()))
    ax.set_yticklabels([str(int(n)) for n in sorted(df['N'].unique())])
    ax.set_ylim(min(df['N']) * 0.6, max(df['N']) * 1.6)
    ax.set_xlabel(_xlabel(xaxis), fontsize=20)
    ax.set_ylabel('training-set size  $N$', fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, which='both', alpha=0.3)

    present = [p for p in ['safe', 'near_floor_safe', 'unsafe', 'near_floor_unsafe']
               if p in set(df['phase_label'])]
    handles = [Line2D([0], [0], marker=PHASE_MARKER[p], color='w',
                      markerfacecolor=PHASE_COLOR[p], markeredgecolor='k',
                      markersize=15, label=PHASE_LABEL_TEXT[p]) for p in present]
    ax.legend(handles=handles, fontsize=13, loc='upper left', framealpha=0.92)
    ax.set_title('Generalization–memorization phase diagram  '
                 '(annotation: $r=\\tau_{\\mathrm{mem}}/\\tau_{\\mathrm{gen}}$)',
                 fontsize=15)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'phase_diagram_safe_points.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


# ------------------------------------------------------------- 2) boundaries ----
def _boundary_N(df_W, safe_col):
    """Smallest N flagged safe by `safe_col` for a fixed W, and the largest unsafe
    N below it. Returns (N_boundary_lo, N_boundary_hi) bracketing the transition,
    or (None, None) if all-safe / all-unsafe / insufficient points."""
    df_W = df_W.sort_values('N')
    Ns   = df_W['N'].values
    safe = df_W[safe_col].values.astype(bool)
    if safe.all() or (~safe).all():
        return None, None
    # First index where it flips from unsafe→safe as N grows.
    for i in range(1, len(Ns)):
        if (not safe[i - 1]) and safe[i]:
            return float(Ns[i - 1]), float(Ns[i])
    return None, None


def plot_boundaries(df, out_dir, xaxis):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Measured points (same coloring as figure 1, smaller markers).
    for _, r in df.iterrows():
        ph = r['phase_label']
        ax.scatter(_xval(r, xaxis), r['N'], s=200,
                   marker=PHASE_MARKER.get(ph, 'o'),
                   color=PHASE_COLOR.get(ph, '#333333'),
                   edgecolors='k', linewidths=1.2, zorder=3)

    # Approximate generalization–memorization boundaries N_gm(W) at τ_gen, 3τ_gen,
    # 8τ_gen. For each W we can only bracket the transition between two measured N;
    # we plot the geometric-mean N of the bracket and connect across W (dashed =
    # inferred, sparse evidence).
    styles = [('is_safe_at_tau_gen', '-',  'boundary at $\\tau_{\\mathrm{gen}}$'),
              ('is_safe_at_3tau_gen', '-.', 'boundary at $3\\,\\tau_{\\mathrm{gen}}$'),
              ('is_safe_at_8tau_gen', ':',  'boundary at $8\\,\\tau_{\\mathrm{gen}}$')]
    Ws = sorted(df['W'].unique())
    boundary_handles = []
    for safe_col, ls, lbl in styles:
        xs, ys = [], []
        for W in Ws:
            lo, hi = _boundary_N(df[df['W'] == W], safe_col)
            if lo is not None:
                xW = (df[df['W'] == W]['parameter_count_p'].iloc[0]
                      if xaxis == 'p' else W)
                xs.append(xW); ys.append(np.sqrt(lo * hi))
        if xs:
            line, = ax.plot(xs, ys, ls=ls, color='k', lw=2.0, alpha=0.8,
                            marker='x', ms=9, zorder=2)
            boundary_handles.append(Line2D([0], [0], color='k', ls=ls, lw=2.0,
                                           label=lbl))
        else:
            # No measured transition for this training-time boundary → note it in
            # the legend so we don't imply evidence we don't have.
            boundary_handles.append(Line2D([0], [0], color='0.6', ls=ls, lw=2.0,
                                           label=lbl + ' (no transition in grid)'))

    ax.set_xscale('log' if xaxis == 'p' else 'linear')
    ax.set_yscale('log')
    if xaxis == 'W':
        ax.set_xticks(Ws)
        ax.set_xticklabels([str(int(w)) for w in Ws])
        ax.set_xlim(min(Ws) * 0.6, max(Ws) * 1.5)
    ax.set_yticks(sorted(df['N'].unique()))
    ax.set_yticklabels([str(int(n)) for n in sorted(df['N'].unique())])
    ax.set_ylim(min(df['N']) * 0.6, max(df['N']) * 1.6)
    ax.set_xlabel(_xlabel(xaxis), fontsize=20)
    ax.set_ylabel('training-set size  $N$', fontsize=20)
    ax.tick_params(labelsize=18)
    ax.grid(True, which='both', alpha=0.3)

    present = [p for p in ['safe', 'near_floor_safe', 'unsafe', 'near_floor_unsafe']
               if p in set(df['phase_label'])]
    phase_handles = [Line2D([0], [0], marker=PHASE_MARKER[p], color='w',
                            markerfacecolor=PHASE_COLOR[p], markeredgecolor='k',
                            markersize=13, label=PHASE_LABEL_TEXT[p]) for p in present]
    leg1 = ax.legend(handles=phase_handles, fontsize=12, loc='upper right',
                     framealpha=0.92)
    ax.add_artist(leg1)
    ax.legend(handles=boundary_handles, fontsize=12, loc='lower left',
              framealpha=0.92)
    ax.set_title('Approximate generalization–memorization boundaries '
                 '(sparse grid — inferred segments dashed)', fontsize=14)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'phase_diagram_boundaries.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


# --------------------------------------------------------------- 3) heatmap -----
def plot_heatmap_table(df, out_dir):
    Ws = sorted(df['W'].unique())
    Ns = sorted(df['N'].unique())
    fig, ax = plt.subplots(figsize=(1.9 * len(Ws) + 3, 1.25 * len(Ns) + 2))

    for i, N in enumerate(Ns):            # rows (bottom = small N)
        for j, W in enumerate(Ws):        # cols
            sub = df[(df['W'] == W) & (df['N'] == N)]
            if sub.empty:
                color, label = PHASE_COLOR['missing'], 'no model'
            else:
                r = sub.iloc[0]
                ph = r['phase_label']
                color = PHASE_COLOR.get(ph, '#333333')
                tg = r['tau_gen']; tm = r['tau_mem']
                if np.isfinite(tg):
                    line1 = f"$\\tau_g$={int(tg):,}"
                else:
                    line1 = "$\\tau_g$: n/r"
                line2 = f"$\\tau_m$={int(tm):,}" if np.isfinite(tm) else "$\\tau_m$: n/r"
                fm = r['f_mem_at_tau_gen'] if np.isfinite(r['f_mem_at_tau_gen']) \
                    else r['f_mem_at_best_quality']
                line3 = f"$f_m(\\tau_g)$={fm*100:.1f}%"
                label = line1 + "\n" + line2 + "\n" + line3
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color,
                                       edgecolor='k', lw=1.5, alpha=0.85))
            ax.text(j + 0.5, i + 0.5, label, ha='center', va='center',
                    fontsize=11, zorder=3)

    ax.set_xlim(0, len(Ws)); ax.set_ylim(0, len(Ns))
    ax.set_xticks(np.arange(len(Ws)) + 0.5)
    ax.set_xticklabels([f"W={w}\n(p={PARAM_M(w)})" for w in Ws], fontsize=13)
    ax.set_yticks(np.arange(len(Ns)) + 0.5)
    ax.set_yticklabels([f"N={n}" for n in Ns], fontsize=13)
    ax.set_xlabel('model width  $W$', fontsize=18)
    ax.set_ylabel('training-set size  $N$', fontsize=18)
    ax.set_title('Phase-diagram table  (cell: $\\tau_{\\mathrm{gen}}$ / '
                 '$\\tau_{\\mathrm{mem}}$ / $f_{\\mathrm{mem}}(\\tau_{\\mathrm{gen}})$)',
                 fontsize=14)
    ax.tick_params(length=0)

    legend_handles = [
        Patch(facecolor=PHASE_COLOR['safe'], edgecolor='k', label='safe'),
        Patch(facecolor=PHASE_COLOR['near_floor_safe'], edgecolor='k',
              label='near-floor safe'),
        Patch(facecolor=PHASE_COLOR['unsafe'], edgecolor='k', label='unsafe'),
        Patch(facecolor=PHASE_COLOR['missing'], edgecolor='k', label='no model'),
    ]
    ax.legend(handles=legend_handles, fontsize=12, loc='upper left',
              bbox_to_anchor=(1.01, 1.0), framealpha=0.95)

    plt.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, f'phase_diagram_heatmap_table.{ext}')
        plt.savefig(p, dpi=300, bbox_inches='tight')
        print(f"  Saved: {p}")
    plt.close()


_PARAM_M = {64: '0.96M', 128: '3.81M', 256: '15.23M'}
def PARAM_M(w):
    return _PARAM_M.get(int(w), '?')


# ---------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description="Plot the generalization–memorization "
                                             "phase diagram (three views).")
    ap.add_argument('--summary_csv', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None)
    ap.add_argument('--xaxis', choices=['W', 'p'], default='W',
                    help="x-axis for the scatter/boundary figures: model width W "
                         "(default) or parameter count p")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    summary_csv = args.summary_csv or os.path.join(
        base, 'results', 'phase_diagram_summary.csv')
    out_dir = args.output_dir or os.path.join(base, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(summary_csv):
        raise FileNotFoundError(
            f"Summary CSV not found: {summary_csv}\n"
            f"Run: python build_phase_diagram_table.py")

    df = pd.read_csv(summary_csv)
    print(f"Loaded {len(df)} (W,N) configurations from {summary_csv}")
    print(f"  Phase counts: "
          f"{df['phase_label'].value_counts().to_dict()}")

    print("\n[1/3] discrete phase-diagram scatter")
    plot_safe_points(df, out_dir, args.xaxis)
    print("\n[2/3] boundary-style phase diagram")
    plot_boundaries(df, out_dir, args.xaxis)
    print("\n[3/3] heatmap-style table")
    plot_heatmap_table(df, out_dir)
    print("Done.")


if __name__ == '__main__':
    main()
