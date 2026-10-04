"""Figures of the paper, regenerated from the CSV tables.

Each function takes a table written by :mod:`memgen.evaluate`,
:mod:`memgen.phase` or one of the downstream modules and writes a PDF and a PNG
to ``results/figures``. No figure is stored in the repository.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from . import config as cfg  # noqa: E402
from .metrics import generalisation_window  # noqa: E402

__all__ = ["loss_curves", "fidelity_and_memorisation", "memorisation_collapse",
           "phase_diagram", "downstream_curve", "save"]

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
WIDTH_COLOUR = {64: "#377eb8", 128: "#ff7f00", 256: "#4daf4a"}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def save(fig, name: str) -> Path:
    """Write a figure as both PDF and PNG and return the PDF path."""
    out_dir = cfg.result_dir("figures")
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(pdf)
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)
    print(f"Wrote {pdf}")
    return pdf


def _series(frame: pd.DataFrame, by: str):
    """Iterate over the curves of a table, with a stable colour per key."""
    for colour, (key, group) in zip(
        PALETTE * 4, sorted(frame.groupby(by), key=lambda kv: kv[0])
    ):
        yield key, group.sort_values("tau"), colour


# --------------------------------------------------------------------------- #

def loss_curves(curves: list[Path], name: str = "loss_vs_tau"):
    """Train and test denoising loss versus training time.

    The test loss turns upward exactly where the model stops improving its
    distributional fit, which is the first signature of the window closing.
    """
    frame = pd.concat([pd.read_csv(p) for p in curves], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for n, group, colour in _series(frame, "N"):
        group = group[group["tau"] > 0]
        for ax, scale in zip(axes, (1.0, 1.0 / n)):
            ax.plot(group["tau"] * scale, group["loss_train"], color=colour,
                    label=f"$N={n}$")
            ax.plot(group["tau"] * scale, group["loss_test"], color=colour,
                    linestyle="--")
    for ax, xlabel in zip(axes, (r"$\tau$", r"$\tau / N$")):
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("denoising loss")
    axes[0].legend(title="solid: train, dashed: test")
    return save(fig, name)


def fidelity_and_memorisation(table: Path, by: str = "N",
                              name: str | None = None):
    """FCD and memorisation fraction against training time.

    The shaded band of each curve is the generalisation window: the FCD has
    reached the real-real floor but the memorisation fraction has not yet
    taken off.
    """
    frame = pd.read_csv(table)
    name = name or f"{Path(table).stem}_fidelity_memorisation"
    fig, (ax_fcd, ax_mem) = plt.subplots(1, 2, figsize=(9, 3.4))

    for key, group, colour in _series(frame, by):
        taus = group["tau"].to_numpy(float)
        mask = taus > 0
        label = f"${by}={key}$"
        ax_fcd.plot(taus[mask], group["fcd_gen_test"][mask], color=colour,
                    label=label)
        ax_fcd.axhline(group["fcd_train_test"].iloc[0], color=colour,
                       linestyle=":", alpha=0.6)
        ax_mem.plot(taus[mask], group["f_mem"][mask], color=colour, label=label)
        ax_mem.fill_between(taus[mask], group["f_mem_ci_low"][mask],
                            group["f_mem_ci_high"][mask], color=colour, alpha=0.2)

        window = generalisation_window(taus, group["fcd_gen_test"],
                                       group["f_mem"],
                                       float(group["fcd_train_test"].iloc[0]))
        if window.tau_gen:
            upper = window.tau_mem or taus.max()
            ax_mem.axvspan(window.tau_gen, upper, color=colour, alpha=0.08)

    for ax in (ax_fcd, ax_mem):
        ax.set_xscale("log")
        ax.set_xlabel(r"training time $\tau$")
    ax_fcd.set_yscale("log")
    ax_fcd.set_ylabel("FCD(gen, test)")
    ax_mem.set_ylabel(r"$f_{\mathrm{mem}}$")
    ax_fcd.legend()
    return save(fig, name)


def memorisation_collapse(table: Path, name: str | None = None):
    """Memorisation curves rescaled by the dataset size.

    Plotted against ``tau / N`` the curves of every dataset size collapse,
    which is the empirical statement that the onset of memorisation is
    proportional to the number of training channels.
    """
    frame = pd.read_csv(table)
    name = name or f"{Path(table).stem}_collapse"
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for n, group, colour in _series(frame, "N"):
        taus = group["tau"].to_numpy(float)
        mask = taus > 0
        ax.plot(taus[mask] / n, group["f_mem"][mask], color=colour,
                marker="o", markersize=3, label=f"$N={n}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau / N$")
    ax.set_ylabel(r"$f_{\mathrm{mem}}$")
    ax.legend()
    return save(fig, name)


def phase_diagram(boundary: Path, name: str = "phase_diagram"):
    """Critical dataset size against model capacity.

    Below a boundary the model memorises at the corresponding training time;
    above it the generalisation window stays open.
    """
    frame = pd.read_csv(boundary)
    frame = frame[frame["status"] == "measured"]
    if frame.empty:
        raise SystemExit("No measured boundary points to plot.")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for colour, (multiplier, group) in zip(PALETTE, frame.groupby("multiplier")):
        group = group.sort_values("width")
        params = [cfg.PARAM_COUNTS.get(int(w), int(w)) for w in group["width"]]
        ax.errorbar(params, group["N_c"],
                    yerr=[group["N_c"] - group["N_c_low"],
                          group["N_c_high"] - group["N_c"]],
                    marker="o", capsize=3, color=colour,
                    label=rf"$\tau = {multiplier:g}\,\tau_{{\mathrm{{gen}}}}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of parameters")
    ax.set_ylabel(r"critical dataset size $N_c$")
    ax.legend(title="memorising below the curve")
    return save(fig, name)


def downstream_curve(table: Path, value: str, ylabel: str,
                     name: str | None = None):
    """Downstream performance against the DDIM checkpoint used for augmentation.

    ``value`` is the metric column, for example ``test_nmse_db`` for CSI
    compression or ``snr_db`` for beam alignment. Horizontal lines mark the
    real-data-only reference and the full-real benchmark.
    """
    frame = pd.read_csv(table)
    name = name or f"{Path(table).stem}_vs_tau"
    augmented = frame[frame["experiment"] == "augmented"]
    sizes = sorted(augmented["N"].unique())

    fig, axes = plt.subplots(1, len(sizes), figsize=(3.4 * len(sizes), 3.2),
                             squeeze=False)
    for ax, n in zip(axes[0], sizes):
        group = augmented[augmented["N"] == n]
        stats = group.groupby("tau")[value].agg(["mean", "std"]).reset_index()
        ax.errorbar(stats["tau"], stats["mean"], yerr=stats["std"],
                    marker="o", capsize=3, color=PALETTE[0], label="augmented")
        for label, colour, style in (("reference_only", PALETTE[1], "--"),
                                     ("full_real", PALETTE[2], ":")):
            baseline = frame[(frame["experiment"] == label) & (frame["N"] == n)]
            if not baseline.empty:
                ax.axhline(baseline[value].mean(), color=colour, linestyle=style,
                           label=label.replace("_", " "))
        ax.set_xscale("log")
        ax.set_xlabel(r"DDIM checkpoint $\tau$")
        ax.set_title(f"$N={n}$")
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend()
    return save(fig, name)
