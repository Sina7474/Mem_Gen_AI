'''
Plot Train/Test Loss vs Tau for DDIM Channel Generation
=========================================================
Reads CSV files produced by train_DDIM_tau.py and creates:
  1. train_test_loss_vs_tau.pdf/png — main figure (L_train solid, L_test dashed)
  2. train_test_loss_vs_tau_over_N.pdf/png — x-axis rescaled by tau/N
  3. generalization_gap_vs_tau.pdf/png — gap = L_test - L_train vs tau

Usage:
    python plot_tau_train_test_loss.py [--log_dir <path>] [--output_dir <path>]

Examples:
    python plot_tau_train_test_loss.py
    python plot_tau_train_test_loss.py --log_dir ../Code/DDIM_FMM/logs --output_dir ./tau_plots
'''

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000]

# Color scheme for each N
COLORS = {
    100:  '#e41a1c',
    200:  '#ff7f00',
    500:  '#984ea3',
    1000: '#4daf4a',
    2000: '#377eb8',
    4000: '#d62728',
    8000: '#a65628',
}

# Marker for each N
MARKERS = {
    100:  'o',
    200:  's',
    500:  '^',
    1000: 'D',
    2000: 'v',
    4000: 'P',
    8000: 'X',
}


def load_csv_files(log_dir, sizes=None):
    """
    Load all tau_loss_curve_N_*.csv files from log directories.
    
    Returns dict: {N: DataFrame}
    """
    if sizes is None:
        sizes = TRAIN_SIZES
    
    data = {}
    
    for N in sizes:
        # Look in tau-based directories
        patterns = [
            os.path.join(log_dir, f'DDIM_tau_{N}_incremental/', f'tau_loss_curve_N_{N}.csv'),
            os.path.join(log_dir, f'DDIM_tau_{N}_incremental_seed*/', f'tau_loss_curve_N_{N}.csv'),
            os.path.join(log_dir, f'DDIM_tau_{N}/', f'tau_loss_curve_N_{N}.csv'),
        ]
        
        found = False
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                csv_path = matches[0]
                df = pd.read_csv(csv_path)
                # Clean column names (strip whitespace)
                df.columns = df.columns.str.strip()
                data[N] = df
                print(f"  Loaded N={N}: {csv_path} ({len(df)} points)")
                found = True
                break
        
        if not found:
            print(f"  WARNING: No CSV found for N={N}")
    
    return data


def add_two_legends(ax, data_keys):
    """
    Add two legends:
      1. Line style legend (solid=train, dashed=test)
      2. Dataset size legend (color + marker for each N)
    """
    from matplotlib.lines import Line2D
    
    # Legend 1: Line style
    style_handles = [
        Line2D([0], [0], color='black', linewidth=2.0, linestyle='-', label='Train Loss'),
        Line2D([0], [0], color='black', linewidth=2.0, linestyle='--', label='Test Loss'),
    ]
    leg1 = ax.legend(handles=style_handles, loc='upper center', fontsize=16, framealpha=0.9)
    ax.add_artist(leg1)
    
    # Legend 2: Dataset size (color + marker)
    size_handles = []
    for N in sorted(data_keys):
        color = COLORS.get(N, '#333333')
        marker = MARKERS.get(N, 'o')
        size_handles.append(
            Line2D([0], [0], color=color, marker=marker, linewidth=2.0,
                   linestyle='-', markersize=7, label=f'N={N}')
        )
    ax.legend(handles=size_handles, loc='upper left', fontsize=16, framealpha=0.9)


def plot_train_test_loss_vs_tau(data, output_dir, log_y=False):
    """
    Main plot: L_train (solid) and L_test (dashed) vs tau for all N.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for N in sorted(data.keys()):
        df = data[N]
        tau = df['actual_tau'].values
        L_train = df['L_train'].values
        L_test = df['L_test'].values
        color = COLORS.get(N, '#333333')
        marker = MARKERS.get(N, 'o')
        
        ax.plot(tau, L_train, '-', color=color, linewidth=2.0,
                marker=marker, markersize=5)
        ax.plot(tau, L_test, '--', color=color, linewidth=2.0,
                marker=marker, markersize=5)
    
    ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.set_ylim(top=1.0)
    
    ax.set_xlabel(r'$\tau$', fontsize=20)
    ax.set_ylabel('Denoising Loss', fontsize=20)
    #ax.set_title(r'Train/Test Loss vs $\tau$ — Unconditional DDIM Channel Model', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=20)
    
    add_two_legends(ax, data.keys())
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'train_test_loss_vs_tau.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_train_test_loss_vs_tau_over_N(data, output_dir, log_y=False):
    """
    Plot with x-axis = tau / N to check if overfitting time scales with N.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for N in sorted(data.keys()):
        df = data[N]
        tau_over_N = df['actual_tau'].values / N
        L_train = df['L_train'].values
        L_test = df['L_test'].values
        color = COLORS.get(N, '#333333')
        marker = MARKERS.get(N, 'o')
        
        ax.plot(tau_over_N, L_train, '-', color=color, linewidth=2.0,
                marker=marker, markersize=5)
        ax.plot(tau_over_N, L_test, '--', color=color, linewidth=2.0,
                marker=marker, markersize=5)
    
    ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.set_ylim(top=1.0)
    
    ax.set_xlabel(r'$\tau / N$', fontsize=20)
    ax.set_ylabel('Denoising Loss', fontsize=20)
    #ax.set_title(r'Train/Test Loss vs $\tau/N$ — Overfitting Time Scaling', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=20)
    
    # Inline legends (independent of add_two_legends) — adjust loc= below
    from matplotlib.lines import Line2D
    _style = [
        Line2D([0], [0], color='black', linewidth=2.0, linestyle='-', label='Train Loss'),
        Line2D([0], [0], color='black', linewidth=2.0, linestyle='--', label='Test Loss'),
    ]
    _leg1 = ax.legend(handles=_style, loc='upper center', fontsize=16, framealpha=0.9)   # ← legend 1 position
    ax.add_artist(_leg1)
    _sizes = []
    for _N in sorted(data.keys()):
        _sizes.append(
            Line2D([0], [0], color=COLORS.get(_N, '#333333'), marker=MARKERS.get(_N, 'o'),
                   linewidth=2.0, linestyle='-', markersize=7, label=f'N={_N}'))
    ax.legend(handles=_sizes, loc='upper left', fontsize=16, framealpha=0.9)              # ← legend 2 position
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'train_test_loss_vs_tau_over_N.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_generalization_gap_vs_tau(data, output_dir):
    """
    Plot generalization gap (L_test - L_train) vs tau.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for N in sorted(data.keys()):
        df = data[N]
        tau = df['actual_tau'].values
        gap = df['generalization_gap'].values
        color = COLORS.get(N, '#333333')
        marker = MARKERS.get(N, 'o')
        
        ax.plot(tau, gap, '-', color=color, linewidth=2.0,
                marker=marker, markersize=5, label=f'N={N}')
    
    ax.set_xscale('log')
    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel(r'$\tau$', fontsize=20)
    ax.set_ylabel(r'Generalization Gap ($L_{test} - L_{train}$)', fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=20)
    
    ax.legend(loc='upper left', fontsize=16, framealpha=0.9)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'generalization_gap_vs_tau.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


def plot_separate_train_test(data, output_dir, log_y=False):
    """
    Two separate subplots: L_train only (left) and L_test only (right).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for N in sorted(data.keys()):
        df = data[N]
        tau = df['actual_tau'].values
        L_train = df['L_train'].values
        L_test = df['L_test'].values
        color = COLORS.get(N, '#333333')
        marker = MARKERS.get(N, 'o')
        
        ax1.plot(tau, L_train, '-', color=color, linewidth=2.0,
                 marker=marker, markersize=5, label=f'N={N}')
        ax2.plot(tau, L_test, '--', color=color, linewidth=2.0,
                 marker=marker, markersize=5, label=f'N={N}')
    
    for ax, title in [(ax1, r'Training Loss vs $\tau$'),
                      (ax2, r'Test Loss vs $\tau$')]:
        ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        ax.set_xlabel(r'$\tau$', fontsize=20)
        ax.set_ylabel('Denoising MSE Loss', fontsize=20)
        #ax.set_title(title, fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=20)
        ax.legend(loc='upper right', fontsize=16, framealpha=0.9)
    
    plt.tight_layout()
    
    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'train_test_loss_separate.{ext}')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot train/test loss vs tau")
    parser.add_argument("--log_dir", type=str,
                        default="../Code/DDIM_FMM/logs",
                        help="Directory containing DDIM_tau_* log folders")
    parser.add_argument("--output_dir", type=str,
                        default="./tau_plots",
                        help="Output directory for plots")
    parser.add_argument("--log_y", action="store_true",
                        help="Use log scale for y-axis")
    parser.add_argument("--sizes", type=int, nargs='+', default=None,
                        help="Specific sizes to plot (default: all available)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PLOTTING: Train/Test Loss vs Tau")
    print("=" * 60)
    print(f"  Log directory: {args.log_dir}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Load data
    print("Loading CSV files...")
    data = load_csv_files(args.log_dir, sizes=args.sizes)
    
    if not data:
        print("ERROR: No CSV files found. Run train_DDIM_tau.py first.")
        sys.exit(1)
    
    print(f"\nLoaded data for N = {sorted(data.keys())}")
    print()
    
    # Generate plots
    print("Generating plots...")
    plot_train_test_loss_vs_tau(data, args.output_dir, log_y=args.log_y)
    plot_train_test_loss_vs_tau_over_N(data, args.output_dir, log_y=args.log_y)
    plot_generalization_gap_vs_tau(data, args.output_dir)
    plot_separate_train_test(data, args.output_dir, log_y=args.log_y)
    
    print("\nDone! All plots saved to:", args.output_dir)
