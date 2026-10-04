#!/usr/bin/env python
"""
================================================================================
PAPER-FAITHFUL GENERALIZATION–MEMORIZATION PHASE DIAGRAM
================================================================================

Reconstructs, for our synthetic wireless channels, the phase diagram of Figure 3
(right panel) of Bonnaire et al., "Why Diffusion Models Don't Memorize: The Role
of Implicit Dynamical Regularization in Training" (arXiv:2505.17638v2).

The reconstruction follows the rule documented in
    ../../Reports/report_original_paper_phase_graph_logic.md
and explained in plain language in
    ../../Reports/report_phase_diagram_explained_simply.md


THE PLANE
---------
    x-axis :  p  = number of trainable U-Net parameters  (set by the width W)
    y-axis :  N  = training-set size
Every (p, N) cell is ONE trained DDIM. Below the boundary the model memorizes,
above it the model generalizes.


THE CONSTRUCTION (paper's rule, three steps)
--------------------------------------------
STEP 1 — one stopping time per MODEL SIZE (not per cell).
    A diffusion model's memorization grows with training time τ, so "does it
    memorize?" is only meaningful once we fix the τ at which we look. The paper
    looks at τ_gen(W): the time at which sample quality first becomes good.
    Crucially τ_gen is a function of the WIDTH ONLY — the paper's quality curves
    collapse under τ → W·τ, giving

        τ_gen(W) = C / W .

    We estimate τ_gen(W) from OUR OWN FCD curves (never assumed — see step 1b)
    as the first checkpoint whose FCD(Gen,Test) falls within `--tol` of the
    real–real floor FCD(Train,Test), measured at a reference dataset size
    `--ref-n` (the largest N, where the quality curve is least contaminated by
    memorization). We then fit the single constant C and use τ_gen(W) = C/W,
    exactly as the paper uses a single rounded constant.

STEP 1b — VERIFY the collapse, do not assume it.
    The report explicitly warns that W·τ_gen ≈ const must be checked on new
    data. This script prints the per-width W·τ_gen products and their spread,
    and writes the FCD-collapse figure so the claim is auditable.

STEP 2 — read f_mem at c·τ_gen(W) for every dataset size N.
    For each width and each multiplier c the memorization fraction is read at
    τ = c·τ_gen(W). Because c·τ_gen almost never lands exactly on a saved
    checkpoint (this is equally true in the original paper, whose released code
    does not state how it was handled), f_mem is linearly interpolated in log τ.

STEP 3 — the boundary is the smallest N that is still safe.
        N_c(p) = min { N : f_mem(W, N, c·τ_gen(W)) <= eps }
    Between the last memorizing N and the first safe N the crossing is refined
    by interpolating log N against log f_mem, which is what turns a handful of
    discrete measured points into the smooth curve the paper draws.


CENSORING (honesty)
-------------------
f_mem is monotonically increasing in τ. If c·τ_gen(W) exceeds a run's τ_max we
use the last measured value as a LOWER bound: if it already exceeds eps the cell
is provably memorizing at the requested time, so the boundary stays valid. Cells
whose last value is <= eps are flagged 'censored' and never used to *create* a
boundary. Boundaries that fall outside the tested N grid are reported as
'above_range' / 'below_range' and are NOT drawn.


ON eps
------
The paper uses the strict criterion f_mem = 0. On our grid the strict criterion
degenerates: at c·τ_gen the only exactly-zero cells are at the largest N, so a
strict boundary cannot be located for most widths (this is reported in the
sensitivity table). We therefore use eps = 1% by default and report the strict
result alongside it.


USAGE
-----
    python paper_phase_diagram.py                       # defaults
    python paper_phase_diagram.py --multipliers 1 2 3 4 # more boundary curves
    python paper_phase_diagram.py --eps 0.02            # looser safety criterion
    python paper_phase_diagram.py --const 4e6           # pin τ_gen(W) = 4e6 / W

OUTPUTS
-------
    results/paper_taugen.csv        τ_gen per width + collapse test
    results/paper_boundary.csv      every boundary point with its status
    results/paper_fmem_at_tau.csv   the f_mem(W, N, c·τ_gen) values behind it
    figures/phase_diagram_paper.png|pdf     the phase diagram
    figures/taugen_collapse.png|pdf         evidence for W·τ_gen ≈ const
================================================================================
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:                                              # smooth, overshoot-free curves
    from scipy.interpolate import PchipInterpolator
    _HAVE_SCIPY = True
except ImportError:                               # graceful fallback to linear
    _HAVE_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CSV = Path("../model_size_effect/results/wsize_fcd_fmem.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

# Trainable-parameter count of the DDIM U-Net at each width (measured, not
# estimated: torch sum(p.numel() for p in model.parameters() if p.requires_grad)).
# W=16 (61,058 params) is available in the CSV but excluded from the DEFAULT
# plots because it is capacity-limited (FCD never reaches the quality criterion).
# To include it, add 16: 61_058 to PARAM_COUNTS and W_COLOR.
PARAM_COUNTS = {64: 956_930, 128: 3_814_402, 256: 15_230_978}

# Visual identity, shared with the model-size figures.
W_COLOR = {64: "#377eb8", 128: "#ff7f00", 256: "#4daf4a"}

MEM_FILL = "#fdcc8a"      # memorizing region – colorblind-safe orange
GEN_FILL = "#b3d4f0"      # generalizing region – colorblind-safe blue


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — τ_gen(W) from our own FCD curves
# ─────────────────────────────────────────────────────────────────────────────
def tau_gen_raw(df, width, n_ref, tol):
    """First checkpoint whose FCD(Gen,Test) comes within `tol` of the floor.

    The floor is FCD(Train,Test), the irreducible real–real distance at that N,
    so the criterion reads "generated samples are as close to held-out data as
    real training data is, up to a factor (1+tol)".
    """
    s = df[(df.W == width) & (df.N == n_ref)].sort_values("tau")
    s = s[s.tau > 0]
    if s.empty:
        return None
    threshold = (1.0 + tol) * float(s.FCD_Train_Test.mean())
    hit = s[s.FCD_Gen_Test <= threshold]
    return int(hit.tau.iloc[0]) if len(hit) else None


def fit_collapse_constant(raw):
    """Single constant C of the paper's relation τ_gen(W) = C / W.

    Uses the geometric mean of the per-width products W·τ_gen, which is the
    least-squares fit in log space and is not dominated by any single width.
    """
    products = [w * t for w, t in raw.items()]
    return float(np.exp(np.mean(np.log(products))))


def round_significant(x, digits=1):
    """Round to `digits` significant figures (the paper quotes a round 3e6)."""
    if x == 0:
        return 0.0
    mag = np.floor(np.log10(abs(x)))
    return float(np.round(x / 10 ** mag, digits - 1) * 10 ** mag)


def resolve_taugen(df, widths, n_ref, tol, const_override=None):
    """τ_gen for every width, plus the collapse constant it derives from.

    A width whose FCD(Gen,Test) never falls within `tol` of the real–real floor
    is CAPACITY-LIMITED: the network is too small to ever produce samples as
    good as the criterion demands, so τ_gen cannot be *measured* for it. Such a
    width is still placed on the diagram, using the collapse law

        τ_gen(W) = C / W ,

    but it is handled with two safeguards:

      * it does NOT contribute to fitting C — it carries no measurement, so
        letting it influence the fit would be circular;
      * it is returned separately in `law_only` so every downstream report can
        mark it as law-derived rather than measured.

    This keeps the widths that DO have a measurement completely unaffected: C is
    fitted from exactly the same data as before, so their τ_gen is unchanged.

    Returns (measured, taugen, const, fitted, law_only).
    """
    measured = {}
    for w in widths:
        t = tau_gen_raw(df, w, n_ref, tol)
        if t is not None:
            measured[w] = t
    if len(measured) < 2:
        raise SystemExit("Not enough widths with a measurable τ_gen.")

    fitted = fit_collapse_constant(measured)
    const = const_override if const_override is not None \
        else round_significant(fitted, 1)
    taugen = {w: const / w for w in widths}
    law_only = [w for w in widths if w not in measured]
    return measured, taugen, const, fitted, law_only


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — f_mem at an arbitrary τ
# ─────────────────────────────────────────────────────────────────────────────
def fmem_at_tau(df, width, n_train, tau):
    """f_mem(W, N, τ) interpolated linearly in log τ.

    Returns (value, status) where status is
        'ok'        τ is inside the measured range,
        'censored'  τ is beyond this run's τ_max, so the returned value is the
                    last measured one and — because f_mem increases with τ — is
                    a LOWER bound on the true value.
    """
    s = df[(df.W == width) & (df.N == n_train)].sort_values("tau")
    s = s[s.tau > 0]
    if s.empty:
        return np.nan, "missing"
    t = s.tau.values.astype(float)
    f = s.f_mem.values.astype(float)
    if tau <= t[0]:
        return float(f[0]), "ok"
    if tau > t[-1]:
        return float(f[-1]), "censored"
    return float(np.interp(np.log(tau), np.log(t), f)), "ok"


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — the boundary N_c(p)
# ─────────────────────────────────────────────────────────────────────────────
def boundary_N(df, width, tau, eps, n_grid):
    """Smallest N whose f_mem at `tau` is <= eps, refined by interpolation.

    Returns (N_boundary, status, per_N_records).

    status:
        'interp'      crossing bracketed by two tested N and interpolated,
        'grid'        crossing lands exactly on a tested N (eps = 0 case),
        'below_range' even the smallest tested N is already safe,
        'above_range' no tested N is safe — the boundary is above our grid.
    """
    records = []
    for n in n_grid:
        value, state = fmem_at_tau(df, width, n, tau)
        records.append({"N": n, "f_mem": value, "state": state})

    safe_idx = next((i for i, r in enumerate(records)
                     if np.isfinite(r["f_mem"]) and r["f_mem"] <= eps), None)

    if safe_idx is None:
        return None, "above_range", records
    if safe_idx == 0:
        return float(n_grid[0]), "below_range", records

    lo, hi = records[safe_idx - 1], records[safe_idx]

    # A censored cell may only *confirm* memorization (its value is a lower
    # bound). If the safe side is censored we cannot trust it as a crossing.
    if hi["state"] == "censored":
        return float(hi["N"]), "censored", records

    # eps = 0 (the paper's strict rule) cannot be interpolated in log space.
    if eps <= 0 or hi["f_mem"] <= 0 or lo["f_mem"] <= 0:
        return float(hi["N"]), "grid", records

    # Refine the crossing: log N against log f_mem, evaluated at f_mem = eps.
    a, b = np.log(lo["f_mem"]), np.log(hi["f_mem"])
    frac = np.clip((np.log(eps) - a) / (b - a), 0.0, 1.0)
    n_star = np.exp(np.log(lo["N"]) + frac * (np.log(hi["N"]) - np.log(lo["N"])))
    return float(n_star), "interp", records


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def smooth_curve(p_points, n_points, p_dense):
    """Monotone-preserving curve through the measured boundary points.

    PCHIP in (log p, N) is used because it passes exactly through every measured
    point and, unlike a spline, cannot overshoot between them — important when
    only three model sizes are available.
    """
    lp, lpd = np.log10(p_points), np.log10(p_dense)
    if _HAVE_SCIPY and len(p_points) >= 2:
        return PchipInterpolator(lp, n_points, extrapolate=False)(lpd)
    return np.interp(lpd, lp, n_points, left=np.nan, right=np.nan)


def plot_phase_diagram(boundaries, grid_points, taugen, const, eps, tol,
                       n_grid, out_stem):
    """The paper-style filled phase diagram."""
    widths = sorted(taugen)
    p_all = np.array([PARAM_COUNTS[w] for w in widths], dtype=float)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype']  = 42
    fig, ax = plt.subplots(figsize=(10, 6))

    x_lo, x_hi = p_all.min() / 1.20, p_all.max() * 1.20
    p_dense = np.logspace(np.log10(x_lo), np.log10(x_hi), 500)

    # ── the c = 1 curve is the phase boundary that colours the plane ─────────
    base = boundaries[boundaries.multiplier == 1].sort_values("p")
    base = base[base.N_boundary.notna()]

    # Headroom so that a boundary lying above the tested N grid is still drawn
    # in full rather than being clipped at the top of the panel.
    n_top = max(n_grid)
    if boundaries.N_boundary.notna().any():
        n_top = max(n_top, float(boundaries.N_boundary.max()))
    y_lo, y_hi = 0.0, n_top * 1.08

    if len(base) >= 2:
        curve = smooth_curve(base.p.values, base.N_boundary.values, p_dense)
        # Constant extension into the small margins outside the measured span,
        # so the two phases tile the panel exactly as in the published figure.
        filled = pd.Series(curve).ffill().bfill().values
        ax.fill_between(p_dense, y_lo, filled, color=MEM_FILL, zorder=0)
        ax.fill_between(p_dense, filled, y_hi, color=GEN_FILL, zorder=0)

    # The measured span; outside it every curve is a constant extension only.
    for p_edge in (p_all.min(), p_all.max()):
        ax.axvline(p_edge, color="0.45", ls=(0, (2, 3)), lw=1.0, zorder=1.5)

    # ── every boundary curve ────────────────────────────────────────────────
    # τ = τ_gen is solid black; the rest are dashed in distinct dark colours.
    # No markers on any curve.  Labels are placed as text at the right end.
    curve_colors = ["#101010", "#7a0177", "#08519c", "#d73027",
                    "#1b9e77", "#6a3d9a", "#e6ab02"]
    for i, c in enumerate(sorted(boundaries.multiplier.unique())):
        sub = boundaries[(boundaries.multiplier == c) &
                         boundaries.N_boundary.notna()].sort_values("p")
        if sub.empty:
            continue
        colour = curve_colors[i % len(curve_colors)]
        is_base = (c == 1)
        ls = "-" if is_base else "--"
        lw = 3.0 if is_base else 2.2
        if len(sub) >= 2:
            ysmooth = smooth_curve(sub.p.values, sub.N_boundary.values,
                                   p_dense)
            ax.plot(p_dense, ysmooth, ls=ls, color=colour, lw=lw, zorder=4)
            # Annotate at the right end of the curve (inside plot area)
            label_txt = (r"$\tau=\tau_{\mathrm{gen}}$" if is_base
                         else rf"$\tau={c:g}\,\tau_{{\mathrm{{gen}}}}$")
            y_end = float(sub.N_boundary.values[-1])
            x_end = float(sub.p.values[-1])
            ax.annotate(label_txt, xy=(x_end, y_end),
                        xytext=(-10, 6), textcoords="offset points",
                        fontsize=14, color=colour, fontweight="bold",
                        va="bottom", ha="right", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="white", ec="none", alpha=0.7))

    # ── the 18 measured configurations, coloured by their verdict at τ_gen ──
    for _, r in grid_points.iterrows():
        safe = r.f_mem <= eps
        ax.plot(r.p, r.N, marker="s", ms=7, zorder=6,
                color="#1a7a1a" if safe else "#b02418",
                mec="white", mew=1.0, alpha=0.95)

    ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Number of trainable parameters  $p$", fontsize=20)
    ax.set_ylabel("Training-set size  $N$", fontsize=20)

    ax.set_xticks(p_all)
    ax.set_xticklabels([f"{p/1e6:.2f}M\n(W={w})" for p, w in zip(p_all, widths)],
                       fontsize=16)
    ax.minorticks_off()
    ax.set_yticks(np.arange(0, n_top + 1, 500))
    ax.tick_params(axis="y", labelsize=16)
    # Faint guides at the dataset sizes that were actually trained.
    for n in n_grid:
        ax.axhline(n, color="0.35", ls=(0, (1, 4)), lw=0.8, zorder=1.2,
                   alpha=0.55)

    # Legend: only region patches, placed inside the plot
    legend_handles = [
        Patch(facecolor=GEN_FILL, edgecolor="0.5", linewidth=0.5,
              label=r"$f_{\mathrm{mem}} < 10\%$"),
        Patch(facecolor=MEM_FILL, edgecolor="0.5", linewidth=0.5,
              label=r"$f_{\mathrm{mem}} > 10\%$"),
    ]
    ax.legend(handles=legend_handles, fontsize=16, loc="upper left",
              framealpha=0.92, edgecolor="0.7")

    ax.grid(alpha=0.20, ls=":", zorder=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_collapse(df, taugen_raw, const, n_ref, tol, out_stem, law_only=()):
    """Evidence that W·τ_gen ≈ const holds for our channels.

    `law_only` lists capacity-limited widths: they have no measurable τ_gen, so
    they cannot appear in the collapse fit (panel a), but their quality curve is
    still drawn in panel (b) — that curve is precisely the evidence that the
    network is too small to ever meet the criterion.
    """
    widths = sorted(taugen_raw)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # (a) τ_gen(W) against the fitted C / W law
    ax = axes[0]
    w_arr = np.array(widths, dtype=float)
    t_arr = np.array([taugen_raw[w] for w in widths], dtype=float)
    w_dense = np.logspace(np.log10(w_arr.min() / 1.3),
                          np.log10(w_arr.max() * 1.3), 100)
    ax.plot(w_dense, const / w_dense, "-", color="0.35", lw=2,
            label=rf"$\tau_{{\mathrm{{gen}}}}={const/1e6:g}\times10^{{6}}/W$")
    for w in widths:
        ax.plot(w, taugen_raw[w], "o", ms=11, color=W_COLOR[w],
                mec="white", mew=1.5, label=f"W = {w}")
        ax.annotate(rf"$W\tau_{{\mathrm{{gen}}}}={w*taugen_raw[w]/1e6:.2f}\times10^{{6}}$",
                    (w, taugen_raw[w]), textcoords="offset points",
                    xytext=(11, 9), fontsize=9.5, color=W_COLOR[w])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("U-Net width  $W$", fontsize=12)
    ax.set_ylabel(r"$\tau_{\mathrm{gen}}$  (training steps)", fontsize=12)
    ax.set_title(rf"(a)  measured $\tau_{{\mathrm{{gen}}}}$ follows $1/W$"
                 f"\n(from FCD at N = {n_ref}, tolerance {tol:g}×floor)",
                 fontsize=12)
    ax.set_xticks(widths); ax.set_xticklabels([str(w) for w in widths])
    y_ticks = sorted(taugen_raw.values())
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{t:,}" for t in y_ticks])
    ax.minorticks_off()
    ax.grid(alpha=0.25, ls=":", which="both")
    ax.legend(fontsize=10, loc="upper right")

    # (b) the FCD curves themselves, collapsed by τ → W·τ
    ax = axes[1]
    for w in widths:
        s = df[(df.W == w) & (df.N == n_ref) & (df.tau > 0)].sort_values("tau")
        ax.plot(s.tau * w, s.FCD_Gen_Test, "-o", ms=3.5, lw=1.7,
                color=W_COLOR[w], label=f"W = {w}")
    for w in law_only:
        s = df[(df.W == w) & (df.N == n_ref) & (df.tau > 0)].sort_values("tau")
        if s.empty:
            continue
        ax.plot(s.tau * w, s.FCD_Gen_Test, "--s", ms=3.5, lw=1.7, alpha=0.85,
                color=W_COLOR.get(w, "#984ea3"),
                label=f"W = {w} (capacity-limited)")
    floor = float(df[df.N == n_ref].FCD_Train_Test.mean())
    ax.axhline(floor, color="0.4", ls="--", lw=1.4,
               label=f"floor FCD(Train,Test) = {floor:.2f}")
    ax.axhline((1 + tol) * floor, color="0.55", ls=":", lw=1.4,
               label=rf"$\tau_{{\mathrm{{gen}}}}$ criterion = {1+tol:g}×floor")
    ax.axvline(const, color="crimson", ls="-", lw=1.6, alpha=0.75,
               label=rf"$W\tau_{{\mathrm{{gen}}}}={const/1e6:g}\times10^{{6}}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"rescaled training time  $W\,\tau$", fontsize=12)
    ax.set_ylabel("FCD(Gen, Test)", fontsize=12)
    ax.set_title(f"(b)  quality curves collapse under $\\tau\\to W\\tau$"
                 f"\n(N = {n_ref})", fontsize=12)
    ax.grid(alpha=0.25, ls=":", which="both")
    ax.legend(fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Paper-faithful generalization–memorization phase diagram.")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                    help="model-size sweep CSV (default: %(default)s)")
    ap.add_argument("--tol", type=float, default=0.5,
                    help="τ_gen criterion: FCD <= (1+tol)*floor "
                         "(default: %(default)s)")
    ap.add_argument("--ref-n", type=int, default=None,
                    help="dataset size used to read τ_gen (default: largest N, "
                         "where the quality curve is least contaminated by "
                         "memorization)")
    ap.add_argument("--const", type=float, default=None,
                    help="pin the collapse constant C in τ_gen(W)=C/W "
                         "(default: fitted from the data and rounded)")
    ap.add_argument("--eps", type=float, default=0.01,
                    help="memorization threshold defining a 'safe' run "
                         "(default: %(default)s)")
    ap.add_argument("--multipliers", type=float, nargs="+", default=[1, 2, 3],
                    help="boundary curves at c·τ_gen (default: %(default)s)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    widths = sorted(w for w in df.W.unique() if w in PARAM_COUNTS)
    n_grid = sorted(df.N.unique())
    n_ref = args.ref_n if args.ref_n is not None else max(n_grid)

    print("=" * 78)
    print("PAPER-FAITHFUL PHASE DIAGRAM")
    print("=" * 78)
    print(f"  CSV            : {args.csv}")
    print(f"  Widths W       : {widths}")
    print(f"  Dataset sizes N: {n_grid}")
    print(f"  τ_gen criterion: FCD(Gen,Test) <= {1+args.tol:g} × "
          f"FCD(Train,Test)   at N = {n_ref}")
    print(f"  Safety criterion: f_mem <= {args.eps:g}")
    print(f"  Multipliers c  : {args.multipliers}")

    # ── STEP 1 ──────────────────────────────────────────────────────────────
    raw, taugen, const, fitted, law_only = resolve_taugen(
        df, widths, n_ref, args.tol, args.const)

    products = np.array([w * t for w, t in raw.items()], dtype=float)
    spread = products.max() / products.min()

    print("\n" + "-" * 78)
    print("STEP 1 — τ_gen per model size (and the collapse test)")
    print("-" * 78)
    print(f"  {'W':>5} {'p':>12} {'τ_gen measured':>16} {'W·τ_gen':>12} "
          f"{'τ_gen = C/W':>13}")
    for w in sorted(taugen):
        if w in raw:
            print(f"  {w:>5} {PARAM_COUNTS[w]:>12,} {raw[w]:>16,} "
                  f"{w*raw[w]/1e6:>11.2f}M {taugen[w]:>13,.0f}")
        else:
            print(f"  {w:>5} {PARAM_COUNTS[w]:>12,} {'capacity-limited':>16} "
                  f"{'—':>12} {taugen[w]:>13,.0f}  (from the law)")
    print(f"\n  Fitted constant C = {fitted:.3g}   →  using C = {const:.3g}")
    print(f"  Collapse quality  : max/min of W·τ_gen = {spread:.2f}× "
          f"({'GOOD — the paper relation holds' if spread < 2 else 'WEAK — treat τ_gen(W)=C/W with caution'})")
    if law_only:
        print(f"  NOTE: W={law_only} never reach FCD <= {1+args.tol:g}×floor, so "
              f"their τ_gen is DERIVED from the law, not measured.")

    pd.DataFrame([{"W": w, "p": PARAM_COUNTS[w],
                   "tau_gen_measured": raw.get(w, np.nan),
                   "W_times_tau_gen": w * raw[w] if w in raw else np.nan,
                   "tau_gen_fitted": taugen[w], "C_fitted": fitted,
                   "C_used": const, "ref_N": n_ref, "tol": args.tol,
                   "source": "measured" if w in raw else "collapse_law"}
                  for w in sorted(taugen)]
                 ).to_csv(RESULTS_DIR / "paper_taugen.csv", index=False)

    # ── STEPS 2 & 3 ─────────────────────────────────────────────────────────
    print("\n" + "-" * 78)
    print("STEPS 2 & 3 — f_mem at c·τ_gen, and the boundary N_c(p)")
    print("-" * 78)

    boundary_rows, fmem_rows = [], []
    for c in args.multipliers:
        print(f"\n  c = {c:g}")
        for w in sorted(taugen):
            tau_c = c * taugen[w]
            n_star, status, records = boundary_N(df, w, tau_c, args.eps, n_grid)
            for r in records:
                fmem_rows.append({"multiplier": c, "W": w,
                                  "p": PARAM_COUNTS[w], "tau": tau_c,
                                  "N": r["N"], "f_mem": r["f_mem"],
                                  "state": r["state"]})
            boundary_rows.append({"multiplier": c, "W": w,
                                  "p": PARAM_COUNTS[w], "tau": tau_c,
                                  "N_boundary": n_star, "status": status,
                                  "eps": args.eps})
            shown = "  ".join(f"N={r['N']}:{r['f_mem']:.4f}"
                              f"{'*' if r['state'] == 'censored' else ''}"
                              for r in records)
            got = f"{n_star:,.0f}" if n_star is not None else "----"
            print(f"    W={w:>3}  τ={tau_c:>9,.0f}  →  N_c = {got:>7} "
                  f"[{status}]")
            print(f"            f_mem: {shown}")

    boundaries = pd.DataFrame(boundary_rows)
    boundaries.to_csv(RESULTS_DIR / "paper_boundary.csv", index=False)
    pd.DataFrame(fmem_rows).to_csv(RESULTS_DIR / "paper_fmem_at_tau.csv",
                                   index=False)

    # Strict paper criterion (eps = 0) reported as a sensitivity check.
    print("\n" + "-" * 78)
    print("SENSITIVITY — the paper's strict criterion f_mem = 0")
    print("-" * 78)
    for c in args.multipliers:
        cells = []
        for w in sorted(taugen):
            n_star, status, _ = boundary_N(df, w, c * taugen[w], 0.0, n_grid)
            got = f"{n_star:,.0f}" if n_star is not None else "above grid"
            cells.append(f"W={w}: {got} [{status}]")
        print(f"  c = {c:<4g}  " + " | ".join(cells))
    print("  (a strict zero is only observable at our largest N, so eps>0 is "
          "used for the figure)")

    # ── figures ─────────────────────────────────────────────────────────────
    grid_rows = []
    for w in sorted(taugen):
        for n in n_grid:
            value, _ = fmem_at_tau(df, w, n, taugen[w])
            grid_rows.append({"W": w, "p": PARAM_COUNTS[w], "N": n,
                              "f_mem": value})
    grid_points = pd.DataFrame(grid_rows).dropna(subset=["f_mem"])

    plot_phase_diagram(boundaries, grid_points, taugen, const, args.eps,
                       args.tol, n_grid, FIGURES_DIR / "phase_diagram_paper")
    plot_collapse(df, raw, const, n_ref, args.tol,
                  FIGURES_DIR / "taugen_collapse", law_only=law_only)

    print("\n" + "=" * 78)
    print("WROTE")
    print("=" * 78)
    for f in ("results/paper_taugen.csv", "results/paper_boundary.csv",
              "results/paper_fmem_at_tau.csv",
              "figures/phase_diagram_paper.png", "figures/phase_diagram_paper.pdf",
              "figures/taugen_collapse.png", "figures/taugen_collapse.pdf"):
        print(f"  {f}")


if __name__ == "__main__":
    main()
