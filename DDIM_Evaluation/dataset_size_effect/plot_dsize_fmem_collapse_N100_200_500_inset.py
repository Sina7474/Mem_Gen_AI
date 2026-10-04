"""Small, legend-free N={100,200,500} memorization-collapse inset figure.

This redraws the subset from the kappa=1/4 metric table and writes to a new
filename, leaving ``dsize_fmem_collapse_N100_200_500`` untouched.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_dsize_fmem_collapse import _c, _m


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SIZES = (100, 200, 500)
FIGSIZE = (7.0, 4.2)
FONT_SIZE = 24
X_MIN = 10.0
OUTPUT_STEM = "dsize_fmem_collapse_N100_200_500_inset"


def main():
    base = Path(__file__).resolve().parent
    csv_path = base / "results" / "dsize_fmem_kappa1_4.csv"
    output_dir = base / "results" / "figures"

    df = pd.read_csv(csv_path)
    df = df[(df["N"].isin(SIZES)) & (df["tau"] >= 1)].sort_values(["N", "tau"])
    missing = sorted(set(SIZES) - set(df["N"].unique()))
    if missing:
        raise ValueError(f"Missing requested dataset sizes: {missing}")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for n_train in SIZES:
        subset = df[df["N"] == n_train].sort_values("tau")
        tau = subset["tau"].to_numpy(dtype=float)
        fmem = subset["f_mem"].to_numpy(dtype=float)
        final_fmem = fmem[-1]
        if final_fmem <= 0:
            raise ValueError(f"Cannot normalize N={n_train}: final f_mem <= 0")

        x = tau / float(n_train)
        y = fmem / final_fmem
        ax.plot(
            x,
            y,
            "-",
            color=_c(n_train),
            lw=2.2,
            marker=_m(n_train),
            ms=5,
            alpha=0.95,
        )

        y_low = subset["f_mem_ci_low"].to_numpy(dtype=float) / final_fmem
        y_high = subset["f_mem_ci_high"].to_numpy(dtype=float) / final_fmem
        ax.fill_between(
            x, y_low, y_high, color=_c(n_train), alpha=0.12, linewidth=0)

    ax.axhline(1.0, color="0.6", ls=":", lw=1.0, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlim(left=X_MIN)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"$\tau/N$", fontsize=FONT_SIZE)
    ax.set_ylabel(
        r"$f_{\mathrm{mem}}(\tau)\,/\,f_{\mathrm{mem}}(\tau_{\max})$",
        fontsize=FONT_SIZE,
    )
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.grid(True, alpha=0.3)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        output = output_dir / f"{OUTPUT_STEM}.{extension}"
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved: {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
