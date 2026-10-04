"""
plot_errorbars_combined.py — 3-panel error bar summary figure
==============================================================
Combines all three metrics with their error bars into one publication-ready
figure per (N, n_feat):

  Panel 1 (top)    : L_train and L_test vs τ  with ±2σ shaded bands
  Panel 2 (middle) : W1(Gen, Test) quality vs τ  with ±2σ shaded bands
  Panel 3 (bottom) : f_mem(%) vs τ  with asymmetric 95% bootstrap CI bands

Reads from (all pre-computed by other scripts, NONE are modified):
  - results/loss_errorbars/loss_errorbars_all.csv
  - results/w1_quality_errorbars/w1_quality_errorbars_all.csv
  - results/fmem_vs_tau/fmem_vs_tau.csv  (f_mem only, n_feat=256 default)

Saves to:
  - results/errorbars_combined/errorbars_N{N}_nfeat{W}.pdf

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_errorbars_combined.py

    # Specific subsets:
    python plot_errorbars_combined.py --n_feats 256 --sizes 100 1000
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_FEAT_GRID  = [64, 128, 256, 512]
TRAIN_SIZES  = [100, 200, 500, 1000, 2000, 4000]

# Color per dataset size  (same palette as other plot scripts)
SIZE_COLORS = {
    100:  '#e41a1c',
    200:  '#ff7f00',
    500:  '#984ea3',
    1000: '#4daf4a',
    2000: '#377eb8',
    4000: '#d62728',
    8000: '#a65628',
}

LOSS_CSV   = "results/loss_errorbars/loss_errorbars_all.csv"
W1_CSV     = "results/w1_quality_errorbars/w1_quality_errorbars_all.csv"
FMEM_CSV   = "results/fmem_vs_tau/fmem_vs_tau.csv"
OUTPUT_DIR = "results/errorbars_combined"


# ─────────────────────────────────────────────────────────────────────────────
# Single (N, n_feat) combined figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined(N: int, n_feat: int,
                  df_loss: pd.DataFrame,
                  df_w1:   pd.DataFrame,
                  df_fmem: pd.DataFrame,
                  output_dir: str):
    """
    3-panel figure for a specific (N, n_feat) combination.
    f_mem panel always uses n_feat=256 (the only model for which fmem is
    computed by compute_fmem_vs_tau.py).
    """
    # ── Filter relevant rows ─────────────────────────────────────────────────
    loss_sub = df_loss[(df_loss['N'] == N) & (df_loss['n_feat'] == n_feat)
                       ].sort_values('actual_tau')
    w1_sub   = df_w1  [(df_w1  ['N'] == N) & (df_w1  ['n_feat'] == n_feat)
                       ].sort_values('actual_tau')
    fmem_sub = df_fmem[ df_fmem['N'] == N].sort_values('actual_tau')

    if loss_sub.empty and w1_sub.empty and fmem_sub.empty:
        print(f"  No data for N={N}, n_feat={n_feat} — skipping")
        return

    color = SIZE_COLORS.get(N, '#333333')
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=False)
    fig.suptitle(f'Error bars — N={N}, n_feat={n_feat}', fontsize=14, y=0.99)

    # ── Panel 1: Loss ────────────────────────────────────────────────────────
    ax = axes[0]
    if not loss_sub.empty:
        tau = loss_sub['actual_tau'].values

        # L_train
        lt_m = loss_sub['L_train_mean'].values
        lt_e = loss_sub['L_train_2std'].values
        ax.plot(tau, lt_m, color='#e41a1c', lw=2, marker='o', ms=6,
                label=r'$\mathcal{L}_{\mathrm{train}}$')
        ax.fill_between(tau, lt_m - lt_e/2, lt_m + lt_e/2,
                         color='#e41a1c', alpha=0.20)

        # L_test
        le_m = loss_sub['L_test_mean'].values
        le_e = loss_sub['L_test_2std'].values
        ax.plot(tau, le_m, color='#377eb8', lw=2, marker='s', ms=6,
                label=r'$\mathcal{L}_{\mathrm{test}}$')
        ax.fill_between(tau, le_m - le_e/2, le_m + le_e/2,
                         color='#377eb8', alpha=0.20)

        ax.set_xscale('log')
        ax.set_ylabel('Denoising loss', fontsize=13)
        ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.tick_params(labelsize=12)
        _add_shaded_note(ax, r'±$\sigma$ band = ±1$\sigma$ of 2$\sigma$ error bar'
                         r' (5 noise seeds)')
    else:
        ax.text(0.5, 0.5, 'No loss data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    # ── Panel 2: W1 quality ──────────────────────────────────────────────────
    ax = axes[1]
    if not w1_sub.empty:
        tau  = w1_sub['actual_tau'].values
        w1_m = w1_sub['W1_mean'].values
        w1_e = w1_sub['W1_2std'].values

        ax.plot(tau, w1_m, color=color, lw=2, marker='D', ms=6,
                label=r'$W_1$(Gen, Ref)')
        ax.fill_between(tau, w1_m - w1_e/2, w1_m + w1_e/2,
                         color=color, alpha=0.22)

        ax.set_xscale('log')
        ax.set_ylabel(r'$W_1$ quality', fontsize=13)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        _add_shaded_note(ax, r'±band = ±$\sigma$ of 2$\sigma$ (5 ref subsets × 5000)')
    else:
        ax.text(0.5, 0.5, 'No W1 data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    # ── Panel 3: f_mem ───────────────────────────────────────────────────────
    ax = axes[2]
    if not fmem_sub.empty:
        tau    = fmem_sub['actual_tau'].values
        fmem   = 100.0 * fmem_sub['f_mem_k_1_3'].values
        ci_low = 100.0 * fmem_sub['f_mem_k_1_3_ci_low'].values
        ci_hi  = 100.0 * fmem_sub['f_mem_k_1_3_ci_high'].values

        ax.fill_between(tau, ci_low, ci_hi,
                         color=color, alpha=0.22,
                         label='95% bootstrap CI')
        ax.plot(tau, fmem, color=color, lw=2, marker='^', ms=6,
                label=rf'$f_{{\mathrm{{mem}}}}$, N={N} (k=1/3)')
        ax.plot(tau, ci_low, color=color, lw=0.8, ls='--', alpha=0.6)
        ax.plot(tau, ci_hi,  color=color, lw=0.8, ls='--', alpha=0.6)

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=13)
        ax.set_ylabel(r'$f_{\mathrm{mem}}$ (%)', fontsize=13)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        fmem_note = ('n_feat=256 (all f_mem runs)' if n_feat != 256
                     else '1000-sample bootstrap 95% CI')
        _add_shaded_note(ax, fmem_note)
    else:
        ax.text(0.5, 0.5, 'No f_mem data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f'errorbars_N{N}_nfeat{n_feat}')
    for ext in ['pdf', 'png']:
        fig.savefig(f'{fname}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: errorbars_N{N}_nfeat{n_feat}.pdf/png")


def _add_shaded_note(ax, text: str):
    """Small annotation in the lower-right corner of a panel."""
    ax.text(0.99, 0.03, text, ha='right', va='bottom',
            transform=ax.transAxes, fontsize=8,
            color='gray', style='italic')


# ─────────────────────────────────────────────────────────────────────────────
# All-sizes overview for a fixed n_feat (one page per n_feat)
# ─────────────────────────────────────────────────────────────────────────────

def plot_overview_nfeat(n_feat: int,
                        df_loss:  pd.DataFrame,
                        df_w1:    pd.DataFrame,
                        df_fmem:  pd.DataFrame,
                        output_dir: str):
    """
    3-panel figure for ALL N values at a fixed n_feat.
    Useful for a compact overview.
    """
    sizes_present = sorted(df_loss[df_loss['n_feat'] == n_feat]['N'].unique())
    if len(sizes_present) == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=False)
    fig.suptitle(f'Error bars overview — n_feat={n_feat}, all N', fontsize=14)

    for N in sizes_present:
        color = SIZE_COLORS.get(N, '#333333')
        label = f'N={N}'
        ls    = df_loss[(df_loss['N'] == N) & (df_loss['n_feat'] == n_feat)
                        ].sort_values('actual_tau')
        ws    = df_w1  [(df_w1  ['N'] == N) & (df_w1  ['n_feat'] == n_feat)
                        ].sort_values('actual_tau')
        fs    = df_fmem[ df_fmem['N'] == N].sort_values('actual_tau')

        # — Panel 1: generalization gap (L_test - L_train)  ─────────────────
        ax = axes[0]
        if not ls.empty:
            tau   = ls['actual_tau'].values
            gm    = ls['L_gap_mean'].values
            ge    = ls['L_gap_2std'].values
            ax.plot(tau, gm, color=color, lw=1.8, marker='o', ms=5, label=label)
            ax.fill_between(tau,
                             np.maximum(gm - ge/2, 0),
                             gm + ge/2,
                             color=color, alpha=0.15)

        # — Panel 2: W1 quality ──────────────────────────────────────────────
        ax = axes[1]
        if not ws.empty:
            tau  = ws['actual_tau'].values
            w1_m = ws['W1_mean'].values
            w1_e = ws['W1_2std'].values
            ax.plot(tau, w1_m, color=color, lw=1.8, marker='s', ms=5, label=label)
            ax.fill_between(tau, w1_m - w1_e/2, w1_m + w1_e/2,
                             color=color, alpha=0.15)

        # — Panel 3: f_mem ───────────────────────────────────────────────────
        ax = axes[2]
        if not fs.empty:
            tau    = fs['actual_tau'].values
            fmem   = 100.0 * fs['f_mem_k_1_3'].values
            ci_low = 100.0 * fs['f_mem_k_1_3_ci_low'].values
            ci_hi  = 100.0 * fs['f_mem_k_1_3_ci_high'].values
            ax.plot(tau, fmem, color=color, lw=1.8, marker='^', ms=5, label=label)
            ax.fill_between(tau, ci_low, ci_hi, color=color, alpha=0.15)

    # Axes labels
    axes[0].set_xscale('log')
    axes[0].set_ylabel(r'$\mathcal{L}_{\mathrm{gen}}$ (gap)', fontsize=13)
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=11, loc='best', framealpha=0.9, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xscale('log')
    axes[1].set_ylabel(r'$W_1$(Gen, Ref)', fontsize=13)
    axes[1].legend(fontsize=11, loc='best', framealpha=0.9, ncol=2)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xscale('log')
    axes[2].set_xlabel(r'$\tau$ (optimizer updates)', fontsize=13)
    axes[2].set_ylabel(r'$f_{\mathrm{mem}}$ (%)', fontsize=13)
    axes[2].legend(fontsize=11, loc='best', framealpha=0.9, ncol=2)
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(labelsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fname = os.path.join(output_dir, f'errorbars_overview_nfeat{n_feat}')
    for ext in ['pdf', 'png']:
        fig.savefig(f'{fname}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: errorbars_overview_nfeat{n_feat}.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="3-panel combined error bar figure per (N, n_feat)")
    parser.add_argument("--n_feats",  type=int, nargs='+', default=N_FEAT_GRID)
    parser.add_argument("--sizes",    type=int, nargs='+', default=TRAIN_SIZES)
    parser.add_argument("--loss_csv",  default=LOSS_CSV)
    parser.add_argument("--w1_csv",    default=W1_CSV)
    parser.add_argument("--fmem_csv",  default=FMEM_CSV)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--no_overview", action="store_true",
                        help="Skip the per-n_feat overview figures")
    args = parser.parse_args()

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    def _load(path, name):
        if not os.path.exists(path):
            print(f"  WARNING: {name} CSV not found: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        print(f"  Loaded {name}: {len(df)} rows  [{path}]")
        return df

    print("Loading CSVs ...")
    df_loss  = _load(args.loss_csv,  "loss_errorbars")
    df_w1    = _load(args.w1_csv,    "w1_quality_errorbars")
    df_fmem  = _load(args.fmem_csv,  "fmem_vs_tau")

    if df_loss.empty and df_w1.empty and df_fmem.empty:
        print("No data found. Run compute scripts first.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput dir: {args.output_dir}\n")

    # ── Per-(N, n_feat) combined figure ───────────────────────────────────────
    for n_feat in args.n_feats:
        for N in args.sizes:
            plot_combined(N, n_feat, df_loss, df_w1, df_fmem, args.output_dir)

    # ── Per-n_feat overview figure ────────────────────────────────────────────
    if not args.no_overview:
        print("\nGenerating per-n_feat overview figures ...")
        for n_feat in args.n_feats:
            plot_overview_nfeat(n_feat, df_loss, df_w1, df_fmem, args.output_dir)

    print(f"\nDone! All plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
