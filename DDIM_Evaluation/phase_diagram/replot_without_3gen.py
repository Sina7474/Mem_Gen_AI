"""
replot_without_3gen.py
======================
Reproduce figures/phase_diagram_logistic.png EXACTLY, but with the
tau = 3*tau_gen boundary curve removed, saved as
figures/phase_diagram_logistic_eps10_without_3gen.png|pdf.

Nothing is recomputed: the already-written boundary + isotonic CSVs
(results/logistic_boundary.csv, results/logistic_boundary_isotonic.csv)
are reused, so every other curve keeps its identical N_c, 95% CI, colour
and position.  Only the multiplier == 3 rows are blanked (N_c -> NaN) so
that single curve (and its faint isotonic line) is not drawn, while the
full multiplier ordering [1, 2, 3, 4, 5, 7] is preserved — this keeps
tau = 5*tau_gen red (colour index 4), exactly as in the original figure.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import paper_phase_diagram as pdg
import phase_boundary_logistic as L

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
# The original phase_diagram_logistic.png drew only tau = 1, 2, 3, 5*tau_gen
# (c = 4 and c = 7 were censored / not shown). This figure is that original
# minus the tau = 3 curve, so only c = 1, 2, 5 stay visible. All other
# multipliers are blanked but kept in the frame to preserve the exact colour
# ordering (tau = 5 -> colour index 4 -> red), identical to the original.
KEEP_MULTS = [1.0, 2.0, 5.0]

def main():
    ap = argparse.ArgumentParser(
        description="Replot a logistic phase diagram while hiding selected "
                    "tau/tau_gen multipliers and preserving color ordering")
    ap.add_argument("--boundary", type=Path,
                    default=RESULTS_DIR / "logistic_boundary.csv")
    ap.add_argument("--isotonic", type=Path,
                    default=RESULTS_DIR / "logistic_boundary_isotonic.csv")
    ap.add_argument("--csv", type=Path, default=pdg.DEFAULT_CSV,
                    help="Sweep CSV used to recover tau_gen and the N grid")
    ap.add_argument("--eps", type=float, default=0.10)
    ap.add_argument("--keep-multipliers", type=float, nargs="+",
                    default=KEEP_MULTS)
    ap.add_argument("--tol", type=float, default=0.5)
    ap.add_argument("--const", type=float, default=None)
    ap.add_argument("--out-stem", type=Path,
                    default=FIGURES_DIR /
                    "phase_diagram_logistic_eps10_without_3gen")
    ap.add_argument("--max-y-tick", type=float, default=None)
    args = ap.parse_args()

    if not 0.0 < args.eps < 1.0:
        ap.error("--eps must lie strictly between 0 and 1")

    # Match the original plot: log x-axis at the true parameter count.
    L.EQUAL_SPACING = False

    bd = pd.read_csv(args.boundary)
    iso = pd.read_csv(args.isotonic)
    df = pd.read_csv(args.csv)
    widths = sorted(w for w in df.W.unique() if w in pdg.PARAM_COUNTS)
    n_grid = sorted(df.N.unique())
    n_ref = max(n_grid)
    _, taugen, const, _, _ = pdg.resolve_taugen(
        df, widths, n_ref, args.tol, args.const)

    # Blank hidden curves instead of deleting their rows. plot_phase_diagram()
    # then retains the original multiplier-to-color mapping.
    keep_bd = np.isclose(
        bd.multiplier.values[:, None], args.keep_multipliers).any(axis=1)
    bd.loc[~keep_bd, ["N_c", "N_c_lo", "N_c_hi"]] = np.nan
    keep_iso = np.isclose(
        iso.multiplier.values[:, None], args.keep_multipliers).any(axis=1)
    iso.loc[~keep_iso, "N_c_iso"] = np.nan

    args.out_stem.parent.mkdir(parents=True, exist_ok=True)
    L.plot_phase_diagram(
        bd, iso, taugen, const, args.eps, n_grid, args.out_stem,
        show_isotonic=True, max_y_tick=args.max_y_tick)

    print(f"WROTE  {args.out_stem}.png")
    print(f"WROTE  {args.out_stem}.pdf")


if __name__ == "__main__":
    main()
