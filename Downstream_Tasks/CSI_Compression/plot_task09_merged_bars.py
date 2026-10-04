"""
plot_task09_merged_bars.py — Merged bar-chart figure (Task 09, paper version)
=============================================================================
Merges the two per-generator line figures

    downstream_csi_compression_augmentation_DDIM_N200.png
    downstream_csi_compression_augmentation_DDIM_N1000.png

into ONE single-panel grouped **bar chart**.

Layout
------
x-axis groups : number of real downstream training channels (200 / 500 / 1000)
within a group: 1 reference-only bar  +  2 generators x 3 checkpoints

    colour  ->  augmentation checkpoint  tau in {1k, 10k, 100k}
    hatch   ->  generator training size  N_gen in {200, 1000}
                (solid = N_gen 200,  hatched = N_gen 1000)

A dashed grey line marks the "full reference 5000" benchmark.

The original figures are NOT modified or overwritten; this script only writes
new files named  downstream_csi_compression_augmentation_bars_DDIM_N200_N1000.*

Usage:
    conda activate Mem_Gen
    cd Downstream_Tasks/CSI_Compression
    python plot_task09_merged_bars.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["hatch.linewidth"] = 1.6

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results" / "downstream_csi_compression_results.csv"
FIG_DIR = HERE / "figures"
OUT_STEM = "downstream_csi_compression_augmentation_bars_DDIM_N200_N1000"

# ── experiment grid ──────────────────────────────────────────────────────────
REAL_SIZES = [200, 500, 1000]          # x-axis groups
TAUS = [1000, 10000, 100000]           # augmentation checkpoints
GEN_SIZES = [200, 1000]                # DDIM generator training-set sizes

# ── styling (colours kept identical to the original line figures) ────────────
REF_COLOR = "#3a3a3a"
TAU_COLORS = {1000: "#1b9e77", 10000: "#d95f02", 100000: "#7570b3"}
TAU_LABEL = {1000: r"$\tau$ = 1k", 10000: r"$\tau$ = 10k",
             100000: r"$\tau$ = 100k"}
GEN_HATCH = {200: "", 1000: "///"}
BENCH_COLOR = "#555555"

FS_LABEL = 23      # axis labels
FS_TICK = 21       # tick labels
FS_LEGEND = 19     # legend
FS_BARTXT = 12.5   # numbers printed at the bar tips

BAR_W = 0.10       # bar width in x-units (group centres are 1.0 apart)
GAP_REF = 0.55     # gap (in bar widths) between reference bar and aug. block
GAP_GEN = 0.45     # gap (in bar widths) between the two generator blocks


# ─────────────────────────────────────────────────────────────────────────────
# data access
# ─────────────────────────────────────────────────────────────────────────────
def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {csv_path}. Run run_task09.py first.")
    return pd.read_csv(csv_path)


def benchmark_value(df: pd.DataFrame):
    row = df[df["group"] == "Benchmark"]
    return float(row["test_NMSE_dB"].iloc[0]) if not row.empty else None


def reference_only(df: pd.DataFrame, n_real: int) -> float:
    row = df[(df["group"] == "Real Only") & (df["N_downstream_real"] == n_real)]
    return float(row["test_NMSE_dB"].iloc[0]) if not row.empty else np.nan


def augmented(df: pd.DataFrame, gen_n: int, tau: int, n_real: int) -> float:
    row = df[(df["DDIM_train_N"].astype(str) == str(gen_n)) &
             (df["tau"].astype(str) == str(tau)) &
             (df["N_downstream_real"] == n_real)]
    return float(row["test_NMSE_dB"].iloc[0]) if not row.empty else np.nan


def bar_slots() -> np.ndarray:
    """Within-group x-offsets for the 7 bars, centred on the group tick.

    Order: [reference, (N_gen=200, tau) x3, (N_gen=1000, tau) x3].
    """
    n_tau = len(TAUS)
    cursor = -group_span() / 2.0

    offs = [cursor + 0.5]                               # reference bar centre
    cursor += 1.0 + GAP_REF
    for _ in GEN_SIZES:
        for t_idx in range(n_tau):
            offs.append(cursor + t_idx + 0.5)
        cursor += n_tau + GAP_GEN
    return np.array(offs) * BAR_W


def group_span() -> float:
    """Total width of one bar cluster, in bar-width units."""
    n_tau = len(TAUS)
    return 1.0 + GAP_REF + n_tau + GAP_GEN + n_tau


# ─────────────────────────────────────────────────────────────────────────────
# plotting
# ─────────────────────────────────────────────────────────────────────────────
def annotate(ax, rects, values, pad):
    """Print each value just past the tip of its (downward) bar, rotated."""
    for rect, val in zip(rects, values):
        if not np.isfinite(val):
            continue
        ax.text(rect.get_x() + rect.get_width() / 2.0, val - pad,
                f"{val:.2f}", ha="center", va="top", rotation=90,
                fontsize=FS_BARTXT, color="#1a1a1a", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.85))


def build_figure(df: pd.DataFrame, out_stem: str):
    bench = benchmark_value(df)
    offs = bar_slots()
    x = np.arange(len(REAL_SIZES), dtype=float)

    # y-range with head-room for the rotated value labels
    vals_all = [reference_only(df, n) for n in REAL_SIZES]
    for g in GEN_SIZES:
        for t in TAUS:
            vals_all += [augmented(df, g, t, n) for n in REAL_SIZES]
    if bench is not None:
        vals_all.append(bench)
    finite = [v for v in vals_all if np.isfinite(v)]
    y_floor = float(min(finite)) - 1.25
    pad = 0.020 * abs(y_floor)

    fig, ax = plt.subplots(figsize=(17.5, 7.2))

    # 1) reference-only bars
    ref_vals = [reference_only(df, n) for n in REAL_SIZES]
    rects = ax.bar(x + offs[0], ref_vals, BAR_W, color=REF_COLOR,
                   edgecolor="white", linewidth=1.0, zorder=3)
    annotate(ax, rects, ref_vals, pad)

    # 2) augmentation bars: colour = tau, hatch = generator size
    slot = 1
    for gen_n in GEN_SIZES:
        for tau in TAUS:
            vals = [augmented(df, gen_n, tau, n) for n in REAL_SIZES]
            rects = ax.bar(x + offs[slot], vals, BAR_W,
                           color=TAU_COLORS[tau], hatch=GEN_HATCH[gen_n],
                           edgecolor="white", linewidth=1.0, zorder=3)
            annotate(ax, rects, vals, pad)
            slot += 1

    # 3) benchmark line
    if bench is not None:
        ax.axhline(bench, color=BENCH_COLOR, linestyle="--", linewidth=2.2,
                   zorder=2)

    # 4) faint separators between the x-groups
    for xi in x[:-1]:
        ax.axvline(xi + 0.5, color="#d9d9d9", linewidth=1.0, zorder=1)

    # 5) cosmetics
    ax.axhline(0.0, color="black", linewidth=1.2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in REAL_SIZES])
    half = group_span() / 2.0 * BAR_W
    ax.set_xlim(x[0] - half - 1.4 * BAR_W, x[-1] + half + 1.4 * BAR_W)
    ax.set_ylim(y_floor, 0.0)
    ax.set_yticks(np.arange(np.ceil(y_floor), 0.5, 1.0))
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(axis="y", alpha=0.28, linestyle="-", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.set_xlabel("Number of real downstream training channels", fontsize=FS_LABEL)
    ax.set_ylabel("Test NMSE (dB)  —  lower is better", fontsize=FS_LABEL)

    # 6) legend: checkpoints (colour), generators (hatch), baselines
    h_tau = [Patch(facecolor=TAU_COLORS[t], edgecolor="white",
                   label=TAU_LABEL[t]) for t in TAUS]
    h_gen = [Patch(facecolor="#d6d6d6", edgecolor="#3d3d3d", linewidth=1.3,
                   hatch=GEN_HATCH[g], label=rf"Generator $N$ = {g}")
             for g in GEN_SIZES]
    h_base = [Patch(facecolor=REF_COLOR, edgecolor="white",
                    label="Reference only")]
    if bench is not None:
        h_base.append(Line2D([0], [0], color=BENCH_COLOR, linestyle="--",
                             linewidth=2.2,
                             label="Full reference 5000 (benchmark)"))

    ax.legend(handles=h_tau + h_gen + h_base, ncol=4,
              fontsize=FS_LEGEND, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.005),
              columnspacing=1.8, handlelength=2.0, handletextpad=0.7,
              labelspacing=0.55)

    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{out_stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Merged single-panel bar figure for Task 09 "
                    "(DDIM generators N=200 and N=1000)")
    parser.add_argument("--csv", type=str, default=str(RESULTS_CSV),
                        help="Path to the results CSV")
    parser.add_argument("--out", type=str, default=OUT_STEM,
                        help="Output file stem (written into figures/)")
    args = parser.parse_args()

    df = load_results(Path(args.csv))
    print(f"Loaded {len(df)} experiment rows from {args.csv}")
    build_figure(df, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
