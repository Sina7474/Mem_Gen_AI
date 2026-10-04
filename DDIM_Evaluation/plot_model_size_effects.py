"""
plot_model_size_effects.py — Visualize Task 11 Model Size Results
==================================================================
Reads the metrics CSV from compute_model_size_metrics.py and the loss CSVs
from training, producing:
  1. Quality (W1) vs tau for each N — one curve per n_feat
  2. Quality vs rescaled tau*W — collapse check
  3. f_mem vs tau for each N — one curve per n_feat
  4. f_mem vs rescaled tau*W/n — collapse check
  5. Train/test loss curves for each n_feat
  6. Generalization gap heatmap in (n, p) plane

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python plot_model_size_effects.py
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

N_FEAT_GRID = [16, 64, 128, 256, 512]
TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000]
LOGS_BASE = os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM', 'logs')
METRICS_CSV = Path("results/model_size/model_size_metrics.csv")
OUTPUT_DIR = Path("results/model_size/figures")

# Colors and markers for n_feat
COLORS_NFEAT = {
    16:  '#a65628',   # brown
    64:  '#377eb8',   # blue
    128: '#4daf4a',   # green
    256: '#e41a1c',   # red
    512: '#984ea3',   # purple
}

MARKERS_NFEAT = {
    16:  'v',
    64:  'o',
    128: 's',
    256: 'D',
    512: '^',
}

# Colors for N (when plotting per-width loss curves)
COLORS_N = {
    100:  '#e41a1c',
    200:  '#ff7f00',
    500:  '#984ea3',
    1000: '#4daf4a',
    2000: '#377eb8',
    4000: '#a65628',
}

MARKERS_N = {
    100:  'o',
    200:  's',
    500:  '^',
    1000: 'D',
    2000: 'v',
    4000: 'P',
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         DATA LOADING                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_metrics():
    """Load the model_size_metrics.csv."""
    if not METRICS_CSV.exists():
        print(f"ERROR: {METRICS_CSV} not found. Run compute_model_size_metrics.py first.")
        return None
    return pd.read_csv(METRICS_CSV)


def load_loss_csv(n_train, n_feat):
    """Load the tau loss curve CSV for a given (N, n_feat)."""
    nfeat_suffix = f'_nfeat{n_feat}' if n_feat != 256 else ''
    log_dir = os.path.join(LOGS_BASE, f"DDIM_tau_{n_train}{nfeat_suffix}_incremental")
    csv_path = os.path.join(log_dir, f"tau_loss_curve_N_{n_train}.csv")
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         PLOT 1: W1 vs tau (per N)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_w1_vs_tau_per_N(df, output_dir):
    """One figure per N: W1(Gen,Test) vs tau, one curve per n_feat."""
    for N in TRAIN_SIZES:
        sub = df[df['N'] == N]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for nf in N_FEAT_GRID:
            data = sub[sub['n_feat'] == nf].sort_values('tau')
            if data.empty:
                continue
            ax.plot(data['tau'], data['W1_Gen_Test'],
                    color=COLORS_NFEAT[nf], marker=MARKERS_NFEAT[nf],
                    markersize=6, linewidth=1.5,
                    label=f'W={nf} (p={data["p"].iloc[0]/1e6:.1f}M)')

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
        ax.set_ylabel(r'$W_1$(Gen, Test) on Effective Rank', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(fontsize=16, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ['pdf', 'png']:
            fig.savefig(output_dir / f'w1_vs_tau_N{N}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: w1_vs_tau_N{N}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                PLOT 2: W1 vs rescaled tau*W (collapse check)            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_w1_vs_tau_rescaled(df, output_dir):
    """W1(Gen,Test) vs tau*W for each N to check tau_gen ~ 1/W collapse."""
    for N in TRAIN_SIZES:
        sub = df[df['N'] == N]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for nf in N_FEAT_GRID:
            data = sub[sub['n_feat'] == nf].sort_values('tau')
            if data.empty:
                continue
            tau_rescaled = data['tau'].values * nf
            ax.plot(tau_rescaled, data['W1_Gen_Test'].values,
                    color=COLORS_NFEAT[nf], marker=MARKERS_NFEAT[nf],
                    markersize=6, linewidth=1.5,
                    label=f'W={nf}')

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau \cdot W$ (rescaled time)', fontsize=20)
        ax.set_ylabel(r'$W_1$(Gen, Test) on Effective Rank', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(fontsize=16, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ['pdf', 'png']:
            fig.savefig(output_dir / f'w1_vs_tauW_N{N}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: w1_vs_tauW_N{N}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                PLOT 3: f_mem vs tau (per N)                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_fmem_vs_tau_per_N(df, output_dir):
    """One figure per N: f_mem(%) vs tau, one curve per n_feat."""
    for N in TRAIN_SIZES:
        sub = df[df['N'] == N]
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        for nf in N_FEAT_GRID:
            data = sub[sub['n_feat'] == nf].sort_values('tau')
            if data.empty:
                continue
            fmem_pct = 100.0 * data['f_mem'].values
            ax.plot(data['tau'], fmem_pct,
                    color=COLORS_NFEAT[nf], marker=MARKERS_NFEAT[nf],
                    markersize=6, linewidth=1.5,
                    label=f'W={nf} (p={data["p"].iloc[0]/1e6:.1f}M)')

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
        ax.set_ylabel(r'$f_{\mathrm{mem}}$ (%)', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(fontsize=16, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ['pdf', 'png']:
            fig.savefig(output_dir / f'fmem_vs_tau_N{N}.{ext}', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: fmem_vs_tau_N{N}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║          PLOT 4: f_mem vs rescaled tau*W/n (collapse check)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_fmem_vs_tau_rescaled(df, output_dir):
    """Normalized f_mem vs tau*W/n for all (N, n_feat) to check collapse."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for N in TRAIN_SIZES:
        for nf in N_FEAT_GRID:
            data = df[(df['N'] == N) & (df['n_feat'] == nf)].sort_values('tau')
            if data.empty or data['f_mem'].max() < 1e-6:
                continue

            fmem = data['f_mem'].values
            fmem_max = fmem.max()
            fmem_norm = fmem / fmem_max if fmem_max > 0 else fmem

            tau_rescaled = data['tau'].values * nf / N

            ax.plot(tau_rescaled, fmem_norm,
                    color=COLORS_NFEAT[nf], marker=MARKERS_N.get(N, 'o'),
                    markersize=5, linewidth=1.0, alpha=0.7,
                    label=f'N={N}, W={nf}')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\tau \cdot W / n$', fontsize=20)
    ax.set_ylabel(r'$f_{\mathrm{mem}}(\tau) / f_{\mathrm{mem}}^{\max}$', fontsize=20)
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, alpha=0.3)

    # Create a compact legend (may have many entries)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 12:
        ax.legend(fontsize=12, loc='best', framealpha=0.9, ncol=2)

    plt.tight_layout()
    for ext in ['pdf', 'png']:
        fig.savefig(output_dir / f'fmem_vs_tauW_over_n_collapse.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: fmem_vs_tauW_over_n_collapse")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║          PLOT 5: Train/Test Loss Curves per n_feat                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_loss_curves_per_nfeat(output_dir):
    """For each n_feat, plot L_train (solid) and L_test (dashed), one color per N."""
    for nf in N_FEAT_GRID:
        fig, ax = plt.subplots(figsize=(10, 6))
        has_data = False

        for N in TRAIN_SIZES:
            df_loss = load_loss_csv(N, nf)
            if df_loss is None:
                continue
            has_data = True
            color = COLORS_N.get(N, '#333333')
            marker = MARKERS_N.get(N, 'o')

            ax.plot(df_loss['actual_tau'], df_loss['L_train'],
                    color=color, linestyle='-', linewidth=1.5,
                    marker=marker, markersize=4)
            ax.plot(df_loss['actual_tau'], df_loss['L_test'],
                    color=color, linestyle='--', linewidth=1.5,
                    marker=marker, markersize=4)

        if not has_data:
            plt.close(fig)
            continue

        # Legend
        handles_N = [Line2D([0], [0], color=COLORS_N.get(N, '#333'), marker=MARKERS_N.get(N, 'o'),
                            markersize=5, linewidth=1.5, label=f'N={N}')
                     for N in TRAIN_SIZES if load_loss_csv(N, nf) is not None]
        handles_style = [
            Line2D([0], [0], color='gray', linestyle='-', linewidth=1.5, label=r'$L_{\mathrm{train}}$'),
            Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label=r'$L_{\mathrm{test}}$'),
        ]
        ax.legend(handles=handles_N + handles_style, fontsize=16,
                  loc='upper right', framealpha=0.9)

        ax.set_xscale('log')
        ax.set_xlabel(r'$\tau$ (optimizer updates)', fontsize=20)
        ax.set_ylabel('Denoising Loss (MSE)', fontsize=20)
        ax.tick_params(axis='both', labelsize=20)
        ax.grid(True, alpha=0.3)

        p_val = sum(pp.numel() for pp in _get_model_params(nf))
        ax.set_title(f'W={nf} (p={p_val/1e6:.1f}M)', fontsize=20)

        plt.tight_layout()
        for ext in ['pdf', 'png']:
            fig.savefig(output_dir / f'train_test_loss_nfeat{nf}.{ext}',
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: train_test_loss_nfeat{nf}")


def _get_model_params(n_feat):
    """Get model parameter tensors for counting."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM'))
    from train_DDIM_tau import Unet
    model = Unet(in_channels=2, n_feat=n_feat)
    return model.parameters()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║         PLOT 6: Generalization Gap Heatmap in (n, p) plane              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_generalization_gap_heatmap(output_dir):
    """
    Heatmap of L_gen = L_test - L_train at selected tau values
    in the (N, n_feat) plane.
    """
    tau_targets = [5000, 20000, 100000, 200000]

    for tau_target in tau_targets:
        # Build matrix: rows = n_feat, cols = N
        gap_matrix = np.full((len(N_FEAT_GRID), len(TRAIN_SIZES)), np.nan)

        for i, nf in enumerate(N_FEAT_GRID):
            for j, N in enumerate(TRAIN_SIZES):
                df_loss = load_loss_csv(N, nf)
                if df_loss is None:
                    continue
                # Find row closest to tau_target
                idx = (df_loss['actual_tau'] - tau_target).abs().idxmin()
                gap_matrix[i, j] = df_loss.loc[idx, 'generalization_gap']

        if np.all(np.isnan(gap_matrix)):
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(gap_matrix, aspect='auto', origin='lower',
                       cmap='RdYlBu_r', interpolation='nearest')

        ax.set_xticks(range(len(TRAIN_SIZES)))
        ax.set_xticklabels([str(N) for N in TRAIN_SIZES], fontsize=16)
        ax.set_yticks(range(len(N_FEAT_GRID)))
        ax.set_yticklabels([f'W={nf}\n({_count_params(nf):.1f}M)' for nf in N_FEAT_GRID],
                           fontsize=14)
        ax.set_xlabel('Training set size $n$', fontsize=20)
        ax.set_ylabel('Model width $W$ (params $p$)', fontsize=20)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r'$L_{\mathrm{gen}} = L_{\mathrm{test}} - L_{\mathrm{train}}$',
                       fontsize=16)
        cbar.ax.tick_params(labelsize=14)

        # Annotate cells
        for i in range(len(N_FEAT_GRID)):
            for j in range(len(TRAIN_SIZES)):
                val = gap_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=12, color='white' if val > 0.1 else 'black')

        plt.tight_layout()
        for ext in ['pdf', 'png']:
            fig.savefig(output_dir / f'gen_gap_heatmap_tau{tau_target}.{ext}',
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: gen_gap_heatmap_tau{tau_target}")


def _count_params(n_feat):
    """Quick param count in millions."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM'))
    from train_DDIM_tau import Unet
    model = Unet(in_channels=2, n_feat=n_feat)
    return sum(p.numel() for p in model.parameters()) / 1e6


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(description="Plot Task 11 model-size results")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TASK 11: Plotting Model Size Effects")
    print("=" * 70)

    # Load metrics CSV
    df = load_metrics()

    # Plots from metrics CSV (W1, f_mem)
    if df is not None and not df.empty:
        print("\n** W1 vs tau (per N) **")
        plot_w1_vs_tau_per_N(df, output_dir)

        print("\n** W1 vs tau*W rescaled (per N) **")
        plot_w1_vs_tau_rescaled(df, output_dir)

        print("\n** f_mem vs tau (per N) **")
        plot_fmem_vs_tau_per_N(df, output_dir)

        print("\n** f_mem vs tau*W/n collapse **")
        plot_fmem_vs_tau_rescaled(df, output_dir)
    else:
        print("\n  (Skipping W1/f_mem plots — no metrics CSV)")

    # Plots from loss CSVs
    print("\n** Train/Test loss curves per n_feat **")
    plot_loss_curves_per_nfeat(output_dir)

    print("\n** Generalization gap heatmap **")
    plot_generalization_gap_heatmap(output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
