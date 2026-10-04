"""
plot_fmem_errorbars.py — Per-N f_mem plots with prominent asymmetric 95% CI
============================================================================
Reads the existing fmem_vs_tau.csv (from compute_fmem_vs_tau.py, which is
NOT modified) and saves one plot per N showing f_mem(%) vs τ with:
  - Solid line: point estimate
  - Shaded band: asymmetric 95% bootstrap CI  [ci_low, ci_high]

Saved to: results/fmem_vs_tau/figures/fmem_with_CI_N{N}.pdf

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_fmem_errorbars.py

    # Custom CSV path:
    python plot_fmem_errorbars.py --fmem_csv results/fmem_vs_tau/fmem_vs_tau.csv
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

TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000, 8000]

COLORS = {
    100:  '#e41a1c',
    200:  '#ff7f00',
    500:  '#984ea3',
    1000: '#4daf4a',
    2000: '#377eb8',
    4000: '#d62728',
    8000: '#a65628',
}
MARKERS = {
    100:  'o',
    200:  's',
    500:  '^',
    1000: 'D',
    2000: 'v',
    4000: 'P',
    8000: 'X',
}

DEFAULT_FMEM_CSV = "results/fmem_vs_tau/fmem_vs_tau.csv"
DEFAULT_OUT_DIR  = "results/fmem_vs_tau/figures"


# ─────────────────────────────────────────────────────────────────────────────
# Plot: all N on one figure (CI bands)
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_with_CI(df: pd.DataFrame, output_dir: str):
    """All-N overview with filled 95% CI bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for N in TRAIN_SIZES:
        sub = df[df['N'] == N].sort_values('actual_tau')
        if sub.empty:
            continue
        tau      = sub['actual_tau'].values
        fmem     = 100.0 * sub['f_mem_k_1_3'].values
        ci_low   = 100.0 * sub['f_mem_k_1_3_ci_low'].values
        ci_high  = 100.0 * sub['f_mem_k_1_3_ci_high'].values

        ax.plot(tau, fmem, color=COLORS[N], marker=MARKERS[N],
                markersize=7, linewidth=1.8, label=f'N={N}', zorder=3)
        ax.fill_between(tau, ci_low, ci_high,
                         color=COLORS[N], alpha=0.20, zorder=2)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
    ax.set_ylabel(r'$f_{\mathrm{mem}}$ (%)', fontsize=20)
    ax.tick_params(axis='both', labelsize=18)
    ax.legend(fontsize=16, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(output_dir, f'fmem_with_CI_all.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fmem_with_CI_all.pdf/png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot: one figure per N (prominent CI)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_N(df: pd.DataFrame, output_dir: str):
    """One plot per N with prominent CI shading."""
    for N in TRAIN_SIZES:
        sub = df[df['N'] == N].sort_values('actual_tau')
        if sub.empty:
            continue

        tau     = sub['actual_tau'].values
        fmem    = 100.0 * sub['f_mem_k_1_3'].values
        ci_low  = 100.0 * sub['f_mem_k_1_3_ci_low'].values
        ci_high = 100.0 * sub['f_mem_k_1_3_ci_high'].values
        ci_half = (ci_high - ci_low) / 2.0   # half-width for annotation

        fig, ax = plt.subplots(figsize=(8, 5))
        color = COLORS[N]

        ax.fill_between(tau, ci_low, ci_high,
                         color=color, alpha=0.25, label='95% CI (bootstrap)')
        ax.plot(tau, fmem, color=color, marker=MARKERS[N],
                markersize=8, linewidth=2.0, label=f'$f_{{\\mathrm{{mem}}}}$, N={N}')
        ax.plot(tau, ci_low,  color=color, linewidth=0.8,
                linestyle='--', alpha=0.6)
        ax.plot(tau, ci_high, color=color, linewidth=0.8,
                linestyle='--', alpha=0.6)

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=18)
        ax.set_ylabel(r'$f_{\mathrm{mem}}$ (%)', fontsize=18)
        ax.set_title(rf'Memorization fraction $f_{{\mathrm{{mem}}}}$ — N={N},'
                     r' $k=\frac{1}{3}$, 95% CI', fontsize=14)
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(fontsize=14, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ['pdf', 'png']:
            fig.savefig(os.path.join(output_dir, f'fmem_with_CI_N{N}.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: fmem_with_CI_N{N}.pdf/png  "
              f"(max CI half-width = {ci_half.max():.2f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-N f_mem plots with asymmetric 95% CI bands")
    parser.add_argument("--fmem_csv",  default=DEFAULT_FMEM_CSV)
    parser.add_argument("--output_dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not os.path.exists(args.fmem_csv):
        raise FileNotFoundError(
            f"fmem CSV not found: {args.fmem_csv}\n"
            "Run compute_fmem_vs_tau.py first.")

    df = pd.read_csv(args.fmem_csv)
    print(f"Loaded {len(df)} rows from {args.fmem_csv}")
    print(f"Columns: {list(df.columns)}")

    # Verify required columns exist
    required = {'N', 'actual_tau', 'f_mem_k_1_3',
                 'f_mem_k_1_3_ci_low', 'f_mem_k_1_3_ci_high'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing columns in CSV: {missing}\n"
            "Ensure compute_fmem_vs_tau.py was run with bootstrap enabled.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")
    print()

    plot_all_with_CI(df, args.output_dir)
    plot_per_N(df, args.output_dir)

    print(f"\nDone! Plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
