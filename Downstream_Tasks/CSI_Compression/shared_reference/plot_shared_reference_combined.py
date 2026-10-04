"""Combine the N=200 and N=500 CSI-compression tau sweeps in one figure.

The two generalization windows are rendered as directly labeled, double-headed
arrows.  This keeps their different horizontal extents legible without covering
the downstream curves or adding window entries to the legend.  The original
one-N figures are never read or overwritten; this script redraws the combined
figure from the metric table and writes to a distinct filename.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import plot_shared_reference as source
import sr_data as data


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SIZES = (200, 500)
COLORS = {200: "#1f77b4", 500: "#ff7f0e"}
MARKERS = {200: "o", 500: "s"}
KAPPA = 0.25
FMEM_THRESHOLD = 0.20
BENCHMARK_N = 5000

OUTPUT_STEM = "shared_reference_nmse_vs_tau_N200_N500_kappa1_4_fmem20"


def main():
    raw = pd.read_csv(source.RESULTS_CSV)
    agg = source.aggregate(raw)
    augmented = {}
    references = {}
    windows = {}

    for n_train in SIZES:
        subset = agg[agg["N"] == n_train]
        aug = subset[subset["experiment_type"] == "augmented"].copy()
        aug["tau"] = pd.to_numeric(aug["tau"], errors="coerce")
        aug = aug.dropna(subset=["tau"]).sort_values("tau")
        ref = subset[subset["experiment_type"] == "reference_only"]
        if aug.empty or ref.empty:
            raise ValueError(f"Incomplete aggregated data for N={n_train}")
        augmented[n_train] = aug
        references[n_train] = ref
        windows[n_train] = data.load_window_kappa(
            n_train, fmem_thresh=FMEM_THRESHOLD, kappa=KAPPA)

    benchmark = agg[agg["experiment_type"] == "full_real"]
    if benchmark.empty:
        raise ValueError("The N=5000 full-real benchmark is missing")

    fig, ax = plt.subplots(figsize=(10, 6))
    tau_min = min(float(frame["tau"].min()) for frame in augmented.values())
    tau_max = max(float(frame["tau"].max()) for frame in augmented.values())

    # Augmented curves and their across-seed standard-deviation bands.
    for n_train in SIZES:
        aug = augmented[n_train]
        tau = aug["tau"].to_numpy(dtype=float)
        mean = aug["nmse_mean"].to_numpy(dtype=float)
        std = aug["nmse_std"].to_numpy(dtype=float)
        ax.plot(
            tau,
            mean,
            color=COLORS[n_train],
            marker=MARKERS[n_train],
            ls="-",
            lw=2.2,
            ms=7,
            mfc="white",
            mew=1.5,
            zorder=5,
        )
        ax.fill_between(
            tau,
            mean - std,
            mean + std,
            color=COLORS[n_train],
            alpha=0.16,
            zorder=2,
        )

        # Mark each downstream-optimal checkpoint without adding legend clutter.
        optimum = int(np.argmin(mean))
        ax.plot(
            tau[optimum],
            mean[optimum],
            marker="*",
            ms=19,
            color="#d62728",
            zorder=7,
        )

    # N-specific real-only baselines use the same color as their augmented curve.
    for n_train in SIZES:
        ref = references[n_train]
        mean = float(ref["nmse_mean"].iloc[0])
        std = float(ref["nmse_std"].iloc[0])
        ax.axhline(mean, color=COLORS[n_train], ls="--", lw=2.0, zorder=4)
        ax.fill_between(
            [tau_min, tau_max],
            mean - std,
            mean + std,
            color=COLORS[n_train],
            alpha=0.08,
            zorder=1,
        )

    # One common full-real benchmark.
    benchmark_mean = float(benchmark["nmse_mean"].iloc[0])
    benchmark_std = float(benchmark["nmse_std"].iloc[0])
    ax.axhline(benchmark_mean, color="black", ls=":", lw=2.0, zorder=4)
    ax.fill_between(
        [tau_min, tau_max],
        benchmark_mean - benchmark_std,
        benchmark_mean + benchmark_std,
        color="black",
        alpha=0.08,
        zorder=1,
    )

    # Show each generalization window as a double-headed arrow rather than a
    # full-height shaded region.  The blue arrow sits above the blue curve; the
    # orange arrow sits below the orange curve and above the shared benchmark.
    arrow_y = {200: -3.62, 500: -6.08}
    label_y = {200: -3.50, 500: -5.96}
    for n_train in SIZES:
        tau_gen, tau_mem = windows[n_train]
        ax.annotate(
            "",
            xy=(tau_mem, arrow_y[n_train]),
            xytext=(tau_gen, arrow_y[n_train]),
            arrowprops={
                "arrowstyle": "<->",
                "color": COLORS[n_train],
                "lw": 2.4,
                "mutation_scale": 15,
            },
            zorder=8,
        )
        ax.text(
            np.sqrt(tau_gen * tau_mem),
            label_y[n_train],
            r"$T_{\mathrm{gen}}(%d)$" % n_train,
            color=COLORS[n_train],
            fontsize=15,
            ha="center",
            va="bottom",
            zorder=9,
        )

    # Five-row, one-column legend. Window arrows are labeled directly in the
    # axes and intentionally omitted from the legend.
    handles = [
        Line2D(
            [0], [0], color=COLORS[200], marker=MARKERS[200], mfc="white",
            lw=2.2, ms=7,
            label=r"$N(200)+N_{\mathrm{syn}}(4800)$",
        ),
        Line2D(
            [0], [0], color=COLORS[500], marker=MARKERS[500], mfc="white",
            lw=2.2, ms=7,
            label=r"$N(500)+N_{\mathrm{syn}}(4500)$",
        ),
        Line2D(
            [0], [0], color=COLORS[200], ls="--", lw=2.0,
            label=r"$N=200$",
        ),
        Line2D(
            [0], [0], color=COLORS[500], ls="--", lw=2.0,
            label=r"$N=500$",
        ),
        Line2D(
            [0], [0], color="black", ls=":", lw=2.0,
            label=rf"$N={BENCHMARK_N}$",
        ),
    ]

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$", fontsize=source.FS_LABEL)
    ax.set_ylabel("CRNet NMSE (dB)", fontsize=source.FS_LABEL)
    ax.tick_params(axis="both", labelsize=source.FS_TICK)
    ax.grid(True, which="both", alpha=0.30, zorder=0)
    ax.legend(
        handles=handles,
        ncol=1,
        fontsize=13.5,
        framealpha=0.94,
        edgecolor="grey",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        borderaxespad=0.0,
        handletextpad=0.55,
    )

    fig.tight_layout()
    source.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        output = source.FIG_DIR / f"{OUTPUT_STEM}.{extension}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved: {output}")
    plt.close(fig)

    for n_train in SIZES:
        tau_gen, tau_mem = windows[n_train]
        print(
            f"N={n_train}: generalization window "
            f"[{tau_gen:.1f}, {tau_mem:.1f}]")


if __name__ == "__main__":
    main()
