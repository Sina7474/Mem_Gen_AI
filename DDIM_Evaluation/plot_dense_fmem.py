#!/usr/bin/env python
"""
Plot f_mem vs τ for the dense retrain around τ=50,000 (W=256, N=50).

Shows both the dense-retrained checkpoints and the original checkpoints
to illustrate that the f_mem dip is a stochastic training transient.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "dense_fmem_results")
OUT_DIR = RESULTS_DIR


def main():
    with open(os.path.join(RESULTS_DIR, "dense_fmem.json")) as f:
        data = json.load(f)

    # Separate by source
    dense = [r for r in data if r["source"] == "dense"]
    orig  = [r for r in data if r["source"] == "original"]

    tau_d = np.array([r["tau"] for r in dense])
    fm_d  = np.array([r["f_mem"] for r in dense])
    lo_d  = np.array([r["ci_low"] for r in dense])
    hi_d  = np.array([r["ci_high"] for r in dense])

    tau_o = np.array([r["tau"] for r in orig])
    fm_o  = np.array([r["f_mem"] for r in orig])
    lo_o  = np.array([r["ci_low"] for r in orig])
    hi_o  = np.array([r["ci_high"] for r in orig])

    fig, ax = plt.subplots(figsize=(10, 5))

    # Dense retrain (connected line)
    order_d = np.argsort(tau_d)
    ax.plot(tau_d[order_d], fm_d[order_d], 'o-', color='#2166ac', ms=7, lw=1.8,
            label='Dense retrain (new run)', zorder=5)
    ax.fill_between(tau_d[order_d], lo_d[order_d], hi_d[order_d],
                    alpha=0.15, color='#2166ac')

    # Original checkpoints (distinct markers)
    ax.scatter(tau_o, fm_o, s=120, marker='D', c='#b2182b', edgecolors='k',
               linewidths=0.8, zorder=10, label='Original run')
    for i in range(len(tau_o)):
        ax.plot([tau_o[i], tau_o[i]], [lo_o[i], hi_o[i]], '-', color='#b2182b',
                lw=2, zorder=9)

    # Highlight the two dips
    # Original dip at τ=50k
    dip_orig = [r for r in orig if r["tau"] == 50000][0]
    ax.annotate(f'Original dip\nf_mem={dip_orig["f_mem"]:.3f}',
                xy=(50000, dip_orig["f_mem"]),
                xytext=(55000, 0.45), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#b2182b', lw=1.5),
                color='#b2182b', fontweight='bold')

    # Dense dip at τ=36k
    dip_dense = [r for r in dense if r["tau"] == 36000][0]
    ax.annotate(f'New dip\nf_mem={dip_dense["f_mem"]:.3f}',
                xy=(36000, dip_dense["f_mem"]),
                xytext=(31000, 0.45), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1.5),
                color='#2166ac', fontweight='bold')

    ax.set_xlabel('Training steps (τ)', fontsize=12)
    ax.set_ylabel('$f_{\\mathrm{mem}}$  (k = 1/3)', fontsize=12)
    ax.set_title('W=256, N=50: Dense checkpoint evaluation around τ=50,000',
                 fontsize=13)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(28000, 72000)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add a text box with the key finding
    textstr = ('Dips are stochastic training transients:\n'
               '• Original run: dip at τ=50k\n'
               '• New run: dip at τ=36k (different location)\n'
               '• Both recover within ~2000 steps')
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                 edgecolor='gray', alpha=0.9)
    ax.text(0.98, 0.35, textstr, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    for ext in ['png', 'pdf']:
        path = os.path.join(OUT_DIR, f'dense_fmem_vs_tau.{ext}')
        fig.savefig(path, dpi=200, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
