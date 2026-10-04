"""
plot_w1_vs_tau.py — Plot W1 vs Training Time τ (optimizer updates)
==================================================================
Creates plots similar to Figure 2 (left) from the memorization paper,
but using W1(effective rank) instead of FID.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_w1_vs_tau.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

RESULTS_DIR = Path("results/w1_vs_tau")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plot style
COLORS = {
    100: '#e41a1c',
    200: '#377eb8',
    500: '#4daf4a',
    1000: '#984ea3',
    2000: '#ff7f00',
    4000: '#a65628',
    8000: '#f781bf',
}


def main():
    # Load data
    csv_path = RESULTS_DIR / 'w1_vs_tau.csv'
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run compute_w1_vs_tau.py first.")
        return

    df = pd.read_csv(csv_path)
    sizes = sorted(df['n_train'].unique())

    print("=" * 70)
    print("Plotting W1 vs τ")
    print("=" * 70)

    # ─── Plot 1: W1(Gen, Test) vs τ — main quality plot (like FID in paper) ───
    fig, ax = plt.subplots(figsize=(10, 6))
    for n in sizes:
        subset = df[df['n_train'] == n].sort_values('tau')
        ax.plot(subset['tau'], subset['w1_gen_test'],
                '-o', color=COLORS[n], label=f'n = {n}', markersize=5, linewidth=2.0)

    ax.set_xlabel('Training time τ (optimizer updates)', fontsize=20)
    ax.set_ylabel('W1(Generated, Test)', fontsize=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    path = FIGURES_DIR / 'w1_gen_test_vs_tau.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'w1_gen_test_vs_tau.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ─── Plot 2: W1(Gen, Train) vs τ ───
    fig, ax = plt.subplots(figsize=(10, 6))
    for n in sizes:
        subset = df[df['n_train'] == n].sort_values('tau')
        ax.plot(subset['tau'], subset['w1_gen_train'],
                '-o', color=COLORS[n], label=f'n = {n}', markersize=5, linewidth=2.0)

    ax.set_xlabel('Training time τ (optimizer updates)', fontsize=20)
    ax.set_ylabel('W1(Generated, Train)', fontsize=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    path = FIGURES_DIR / 'w1_gen_train_vs_tau.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'w1_gen_train_vs_tau.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ─── Plot 3: W1(Gen, Test) vs τ/n (rescaled time, like paper inset) ───
    fig, ax = plt.subplots(figsize=(10, 6))
    for n in sizes:
        subset = df[df['n_train'] == n].sort_values('tau')
        tau_over_n = subset['tau'].values / n
        ax.plot(tau_over_n, subset['w1_gen_test'],
                '-o', color=COLORS[n], label=f'n = {n}', markersize=5, linewidth=2.0)

    ax.set_xlabel('Rescaled training time τ/n', fontsize=20)
    ax.set_ylabel('W1(Generated, Test)', fontsize=20)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    path = FIGURES_DIR / 'w1_gen_test_vs_tau_over_n.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / 'w1_gen_test_vs_tau_over_n.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # ─── Plot 4: Both W1(Gen,Test) and W1(Gen,Train) on same plot for one size ───
    fig, ax = plt.subplots(figsize=(10, 6))
    n_example = 1000
    subset = df[df['n_train'] == n_example].sort_values('tau')
    ax.plot(subset['tau'], subset['w1_gen_test'], '-o', color='#D65F5F',
            label='W1(Gen, Test)', markersize=5, linewidth=2.0)
    ax.plot(subset['tau'], subset['w1_gen_train'], '-s', color='#4878CF',
            label='W1(Gen, Train)', markersize=5, linewidth=2.0)
    ax.axhline(y=df[df['n_train'] == n_example]['w1_gen_test'].iloc[-1], 
               color='gray', linestyle='--', alpha=0.5, label='Final W1(Gen,Test)')

    ax.set_xlabel('Training time τ (optimizer updates)', fontsize=20)
    ax.set_ylabel('Wasserstein-1 Distance', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=20)
    ax.legend(fontsize=16, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    path = FIGURES_DIR / f'w1_both_vs_tau_N{n_example}.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f'w1_both_vs_tau_N{n_example}.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    print("\n** All plots saved! **")
    print(f"   Location: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
