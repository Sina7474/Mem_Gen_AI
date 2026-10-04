"""Plot normalized model-size memorization with a tau*W/N inset.

The main axes reproduce ``wsize_all_fmem_norm_vs_tau``.  A smaller,
legend-free inset shows the same curves against ``tau*W/N``.  This script writes
to a new filename and therefore never replaces either source figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_wsize_all_NW import (
    AXIS_LABEL_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    _c,
    _legends,
    _ls,
    _mk,
)


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def _draw_curves(ax, df, sizes, widths, *, scaled_x: bool, inset: bool):
    present_widths = set()
    present_sizes = set()

    for n_train in sizes:
        for width in widths:
            sub = df[(df["N"] == n_train) & (df["W"] == width)].sort_values("tau")
            if sub.empty:
                continue

            tau = sub["tau"].to_numpy(dtype=float)
            fmem = sub["f_mem"].to_numpy(dtype=float)
            fmem_tau_max = fmem[-1]
            if fmem_tau_max <= 0:
                print(
                    f"  WARNING: f_mem(tau_max)=0 for N={n_train}, "
                    f"W={width}; skipping")
                continue

            present_widths.add(width)
            present_sizes.add(n_train)
            x = tau * float(width) / float(n_train) if scaled_x else tau
            y = fmem / fmem_tau_max

            ax.plot(
                x,
                y,
                ls=_ls(n_train),
                color=_c(width),
                lw=1.55 if inset else 2.2,
                marker=_mk(n_train),
                ms=3.0 if inset else 5,
                alpha=0.95,
                zorder=3,
            )

            if {"f_mem_ci_low", "f_mem_ci_high"}.issubset(sub.columns):
                y_low = sub["f_mem_ci_low"].to_numpy(dtype=float) / fmem_tau_max
                y_high = sub["f_mem_ci_high"].to_numpy(dtype=float) / fmem_tau_max
                if np.any(y_high - y_low > 0):
                    ax.fill_between(
                        x,
                        y_low,
                        y_high,
                        color=_c(width),
                        alpha=0.08 if inset else 0.10,
                        linewidth=0,
                        zorder=2,
                    )

    ax.axhline(1.0, color="0.6", ls=":", lw=0.9 if inset else 1.0,
               alpha=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.24 if inset else 0.30, zorder=0)
    return sorted(present_widths), sorted(present_sizes)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Normalized f_mem vs tau with a legend-free tau*W/N inset")
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=base / "results" / "wsize_fcd_fmem_kappa1_4.csv",
        help="Metric table (default: the kappa=1/4 table used by the source plots)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=base / "results" / "figures")
    parser.add_argument("--min_tau", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 1000])
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument(
        "--output_stem",
        default="wsize_all_fmem_norm_vs_tau_with_tauW_over_N_inset",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = df[
        (df["tau"] >= args.min_tau)
        & df["N"].isin(args.sizes)
        & df["W"].isin(args.widths)
    ].sort_values(["N", "W", "tau"])
    if df.empty:
        raise ValueError("No rows remain after applying the requested filters")

    fig, ax = plt.subplots(figsize=(10, 6))
    present_widths, present_sizes = _draw_curves(
        ax, df, args.sizes, args.widths, scaled_x=False, inset=False)

    ax.set_xlabel(r"$\tau$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        r"$f_{\mathrm{mem}}(\tau)\,/\,f_{\mathrm{mem}}(\tau_{\max})$",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    # Width and dataset-size encodings are defined by panel (a), so panel (c)
    # does not repeat the legends.

    # With the repeated legends removed, the upper-left region can hold a
    # larger inset.  The offsets keep it comfortably away from the panel edges
    # while avoiding the main memorization transitions.
    inset_ax = ax.inset_axes([0.09, 0.48, 0.46, 0.42], zorder=10)
    inset_ax.set_facecolor("white")
    inset_ax.patch.set_alpha(0.98)
    _draw_curves(
        inset_ax, df, args.sizes, args.widths, scaled_x=True, inset=True)
    inset_x_max = (df["tau"] * df["W"] / df["N"]).max()
    inset_ax.set_xlim(left=1000, right=inset_x_max * 1.02)
    inset_ax.set_ylim(0, 1.08)
    inset_ax.set_xlabel(r"$\tau W/N$", fontsize=22, labelpad=0)
    inset_ax.set_ylabel("")
    inset_ax.set_yticks([0.0, 0.5, 1.0])
    inset_ax.tick_params(axis="both", which="major", labelsize=20, pad=2)
    inset_ax.tick_params(axis="both", which="minor", labelsize=0)
    for spine in inset_ax.spines.values():
        spine.set_color("0.25")
        spine.set_linewidth(1.1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    for extension in ("png", "pdf"):
        output = args.output_dir / f"{args.output_stem}.{extension}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
