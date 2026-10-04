"""
plot_shared_reference.py — Aggregate seeds and plot NMSE vs tau (guide s.14)
============================================================================
Reads results/shared_reference/shared_reference_results.csv, aggregates the four
CRNet seeds (mean ± std) and, for every N, draws:

  * NMSE (dB) vs tau on a log-tau axis, with a seed error band,
  * the shaded generalization window [tau_gen, tau_mem],
  * the reference-only horizontal baseline (mean ± std band),
  * the full-real D_5000 horizontal benchmark (mean ± std band),
  * a marker at the downstream-optimal checkpoint tau_down*,

and prints whether tau_down* falls before / inside / after the window.

It also writes an aggregated per-configuration CSV
(shared_reference_aggregated.csv) for the paper tables.

Usage:
    conda activate Mem_Gen
    cd Downstream_Tasks/CSI_Compression
    python shared_reference/plot_shared_reference.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator  # noqa: F401 (kept for parity)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import sr_data as D  # noqa: E402  (window recomputation at other f_mem thresholds)

RESULTS_DIR = HERE.parent / "results" / "shared_reference"
RESULTS_CSV = RESULTS_DIR / "shared_reference_results.csv"
FIG_DIR = RESULTS_DIR / "figures"

# Font criteria (consistent with the other paper figures)
FS_LABEL = 20
FS_TICK = 20
FS_LEGEND = 16

WINDOW_COLOR = "#2ca02c"
AUG_COLOR = "#1f77b4"
REF_COLOR = "#8c564b"
BENCH_COLOR = "black"


def aggregate(df):
    """Mean ± std of test NMSE across seeds, per configuration."""
    keys = ["experiment_type", "N", "tau", "regime", "tau_gen", "tau_mem",
            "FCD", "f_mem"]
    g = (df.groupby(keys, dropna=False)["test_nmse_db"]
         .agg(["mean", "std", "count"]).reset_index()
         .rename(columns={"mean": "nmse_mean", "std": "nmse_std",
                          "count": "n_seeds"}))
    g["nmse_std"] = g["nmse_std"].fillna(0.0)
    return g


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def plot_for_n(agg, n, fmem_thresh=None, kappa=None, fname_suffix=""):
    sub = agg[agg["N"] == n].copy()
    aug = sub[sub["experiment_type"] == "augmented"].copy()
    aug["tau"] = aug["tau"].map(_to_float)
    aug = aug.sort_values("tau")
    if aug.empty:
        print(f"  N={n}: no augmented rows yet, skipping plot.")
        return None

    ref = sub[sub["experiment_type"] == "reference_only"]
    bench = agg[agg["experiment_type"] == "full_real"]

    # Window: only the shaded window moves — the trained NMSE data is untouched.
    # The filename records the ratio-test and f_mem thresholds. Defaults reproduce
    # the canonical (kappa = 1/3, f_mem = 10%) window; tau_gen is kappa-free, and
    # only tau_mem (the right edge) moves.
    eff_kappa = D.KAPPA_MAIN if kappa is None else float(kappa)
    eff_fmem = 0.10 if fmem_thresh is None else float(fmem_thresh)
    tau_gen, tau_mem = D.load_window_kappa(int(n), fmem_thresh=eff_fmem,
                                           kappa=eff_kappa)
    benchmark_n = int(bench["N"].iloc[0]) if not bench.empty else 5000

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generalization window
    if np.isfinite(tau_gen) and np.isfinite(tau_mem):
        ax.axvspan(tau_gen, tau_mem, color=WINDOW_COLOR, alpha=0.13, zorder=0,
                   label="Generalization window")

    # Augmented curve with seed band
    tau = aug["tau"].to_numpy()
    m = aug["nmse_mean"].to_numpy()
    s = aug["nmse_std"].to_numpy()
    ax.plot(tau, m, "-o", color=AUG_COLOR, lw=2.0, ms=7, mfc="white",
            zorder=4,
            label=(rf"$N_{{\mathrm{{csi}}}}$ ({n}) + "
                   rf"$N_{{\mathrm{{syn}}}}$ ({benchmark_n - n})"))
    ax.fill_between(tau, m - s, m + s, color=AUG_COLOR, alpha=0.18, zorder=2)

    # Reference-only baseline band
    ref_line_y = None
    if not ref.empty:
        rm = float(ref["nmse_mean"].iloc[0]); rs = float(ref["nmse_std"].iloc[0])
        ref_line_y = rm
        ax.axhline(rm, color=REF_COLOR, ls="--", lw=2.0, zorder=3,
                   label=rf"$N_{{\mathrm{{csi}}}}={n}$")
        ax.fill_between([tau.min(), tau.max()], rm - rs, rm + rs,
                        color=REF_COLOR, alpha=0.12, zorder=1)

    # Full-real benchmark band
    if not bench.empty:
        bm = float(bench["nmse_mean"].iloc[0]); bs = float(bench["nmse_std"].iloc[0])
        ax.axhline(bm, color=BENCH_COLOR, ls=":", lw=2.0, zorder=3,
                   label=rf"$N_{{\mathrm{{csi}}}}={benchmark_n}$")
        ax.fill_between([tau.min(), tau.max()], bm - bs, bm + bs,
                        color=BENCH_COLOR, alpha=0.10, zorder=1)

    # Downstream-optimal checkpoint
    j = int(np.argmin(m))
    tau_star = tau[j]
    ax.plot([tau_star], [m[j]], marker="*", ms=20, color="#d62728", zorder=6,
            label="_nolegend_")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$", fontsize=FS_LABEL)
    ax.set_ylabel("CRNet NMSE (dB)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, which="both", alpha=0.30, zorder=0)
    legend_kwargs = {
        "fontsize": FS_LEGEND,
        "framealpha": 0.92,
        "edgecolor": "grey",
        "loc": "upper left",
    }
    if ref_line_y is not None:
        legend_kwargs.update({
            "bbox_to_anchor": (0.01, ref_line_y - 0.10),
            "bbox_transform": ax.get_yaxis_transform(),
            "borderaxespad": 0.0,
        })
    ax.legend(**legend_kwargs)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"shared_reference_nmse_vs_tau_N{n}{fname_suffix}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close(fig)

    # Regime of the optimum
    regime = ("before" if tau_star < tau_gen else
              "inside" if tau_star < tau_mem else "after")
    return {"N": n, "tau_star": tau_star, "nmse_star": m[j],
            "tau_gen": tau_gen, "tau_mem": tau_mem, "regime_of_optimum": regime}


def _run(df, agg, kappa, fmem_thresh):
    """Plot every N for one (kappa, f_mem) window setting.

    The filename carries kappa (=1/kden) and the f_mem percentage.
    """
    kden = int(round(1.0 / kappa)) if kappa > 0 else 0
    pct = int(round(fmem_thresh * 100))
    suffix = f"_kappa1_{kden}_fmem{pct}"
    tag = f"kappa = 1/{kden}, f_mem = {pct}%"

    print(f"\n--- Window setting: {tag} ---")
    summary = []
    for n in sorted(df["N"].unique()):
        if n == 5000:
            continue
        res = plot_for_n(agg, n, fmem_thresh=fmem_thresh, kappa=kappa,
                         fname_suffix=suffix)
        if res:
            summary.append(res)

    print("\n" + "=" * 68)
    print(f"Downstream-optimal checkpoint vs generalization window ({tag})")
    print("=" * 68)
    for r in summary:
        print(f"  N={r['N']:>4}:  tau*_down = {r['tau_star']:.0f}  "
              f"(NMSE {r['nmse_star']:.2f} dB)  | window "
              f"[{r['tau_gen']:.0f}, {r['tau_mem']:.0f}]  → optimum "
              f"{r['regime_of_optimum'].upper()} the window")
    print("=" * 68)
    if summary:
        out = RESULTS_DIR / f"shared_reference_optimum_summary{suffix}.csv"
        pd.DataFrame(summary).to_csv(out, index=False)
        print(f"  Summary CSV: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fmem", type=float, nargs="*", default=[0.10, 0.15, 0.20],
                    help="f_mem thresholds (fractions) plotted at the main "
                         "kappa = 1/3. One figure set per value; the legend and "
                         "filename always carry kappa. Default: 0.10 0.15 0.20")
    ap.add_argument("--kappa", type=float, nargs="*", default=[0.25],
                    help="Ratio-test thresholds kappa (memorized iff rho<kappa); "
                         "one figure set per value at f_mem=--kappa_fmem, tau_gen "
                         "unchanged. Default: 0.25 (i.e. 1/4). Pass nothing to "
                         "skip the kappa sweep.")
    ap.add_argument("--kappa_fmem", type=float, nargs="+", default=[0.10, 0.20],
                    help="f_mem threshold(s) used with each --kappa value; one "
                         "figure set per (kappa, f_mem) pair. Default: 0.10 0.20")
    args = ap.parse_args()

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"{RESULTS_CSV} not found. Run run_shared_reference.py first.")
    df = pd.read_csv(RESULTS_CSV)
    agg = aggregate(df)
    agg.to_csv(RESULTS_DIR / "shared_reference_aggregated.csv", index=False)
    print(f"Aggregated CSV: {RESULTS_DIR / 'shared_reference_aggregated.csv'}")

    # Main ratio-test threshold kappa = 1/3 across the requested f_mem levels.
    for t in args.fmem:
        _run(df, agg, kappa=D.KAPPA_MAIN, fmem_thresh=t)

    # Alternative ratio-test thresholds kappa across the requested f_mem levels.
    for k in (args.kappa or []):
        for t in args.kappa_fmem:
            _run(df, agg, kappa=k, fmem_thresh=t)


if __name__ == "__main__":
    main()
