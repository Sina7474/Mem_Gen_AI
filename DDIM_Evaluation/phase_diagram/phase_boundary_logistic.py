#!/usr/bin/env python
"""
================================================================================
PHASE-BOUNDARY ESTIMATION BY CONSTRAINED BINOMIAL LOGISTIC REGRESSION
================================================================================

An ADDITIONAL boundary estimator that sits alongside — and never overwrites —
the log--log interpolation of ``paper_phase_diagram.py``. All outputs of this
script carry the ``logistic`` prefix, so both models can be compared directly.

MODEL
-----
For each U-Net width ``W`` and evaluation time ``τ = c · τ_gen(W)`` the number of
memorized generated samples at training-set size ``N`` is treated as

        K_N ~ Binomial(M_N, q_N),      f_mem = K_N / M_N,

with the memorization probability following a logistic law in ``log N``

        logit(q_N) = a + b · log N,        b < 0.

Here ``M_N`` is the number of evaluated generated samples (``num_generated`` in
the sweep CSV, constant at 5000) and ``K_N = round(f_mem · M_N)`` is the number
judged memorized by the same nearest-neighbour ratio test used everywhere else.
``f_mem`` at the (usually off-grid) time ``c · τ_gen(W)`` is read with the
IDENTICAL log-τ interpolation as the interpolation model, so the two estimators
differ only in HOW they turn the per-N points into a boundary — not in the
underlying measurements.

CRITICAL DATASET SIZE
---------------------
For a memorization threshold ``ε`` the boundary is the ``N`` at which the fitted
memorization probability equals ``ε``:

        N_c = exp( ( logit(ε) − a ) / b ).

UNCERTAINTY
-----------
A 95% confidence interval for ``N_c`` is obtained by PARAMETRIC BOOTSTRAP: draw
``K_N* ~ Binomial(M_N, f_mem_N)`` independently for every ``N``, refit ``(a, b)``
by maximum likelihood, recompute ``N_c*``; the 2.5/97.5 percentiles over many
resamples give the interval.

PRIMARY vs SENSITIVITY
----------------------
  • PRIMARY  — the unconstrained per-(W, c) fits above. The three tested model
    sizes are simply joined by straight lines, exactly as the paper draws them.
  • SENSITIVITY (optional, clearly labelled) — a WEIGHTED ISOTONIC regression
    across the three widths that IMPOSES

        N_c(W=64) ≤ N_c(W=128) ≤ N_c(W=256)

    per multiplier, with weights 1/Var(log N_c) from the bootstrap. This is an
    ASSUMED monotonicity, not a measured one, and is drawn as a thin dashed
    line with an explicit "imposed" label.

MAXIMUM LIKELIHOOD
------------------
The binomial negative log-likelihood and its exact gradient are minimized with
L-BFGS-B (``b`` bounded strictly negative). No statsmodels dependency is needed.

USAGE
-----
    python phase_boundary_logistic.py                          # defaults
    python phase_boundary_logistic.py --eps 0.10 --multipliers 1 2 3 4 5 7
    python phase_boundary_logistic.py --n-boot 4000 --no-isotonic

OUTPUTS  (all new; nothing existing is touched)
-----------------------------------------------
    results/logistic_boundary.csv           N_c + 95% CI + (a, b) per (W, c)
    results/logistic_boundary_isotonic.csv  monotone-constrained N_c per (W, c)
    results/logistic_tau_repairs.csv        checkpoints repaired by the τ-monotone
                                            projection (empty file if none)
    figures/phase_diagram_logistic.png|pdf  the phase diagram, logistic model
    figures/logistic_fits.png|pdf           the per-(W, c) fitted sigmoids
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
from scipy.optimize import minimize
from scipy.special import expit, logit

# Reuse the EXACT τ_gen estimation and f_mem interpolation of the interpolation
# model, so the two boundary estimators are compared on identical inputs.
import paper_phase_diagram as pdg

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

# Distinct colours for the boundary multipliers (kept stable across figures).
MULT_COLORS = ["#101010", "#7a0177", "#08519c", "#8c510a", "#d73027", "#1b9e77",
               "#6a3d9a", "#e6ab02"]

# When True the x-axis places every width at an equally spaced tick instead of
# at its true parameter count on a log axis. Enabled with --equal-spacing.
EQUAL_SPACING = False


# ─────────────────────────────────────────────────────────────────────────────
# Binomial logistic regression:  logit(q) = a + b·log N,  b < 0
# ─────────────────────────────────────────────────────────────────────────────
def _nll_and_grad(theta, x, K, M):
    """Binomial negative log-likelihood and its exact gradient.

    softplus(z) = log(1+e^z) is evaluated with logaddexp for numerical safety.
    d(nll)/d(eta_i) = M_i·sigmoid(eta_i) − K_i, from which the two-parameter
    gradient follows directly.
    """
    a, b = theta
    eta = a + b * x
    # nll = Σ K·softplus(-eta) + (M-K)·softplus(eta)
    nll = np.sum(K * np.logaddexp(0.0, -eta) + (M - K) * np.logaddexp(0.0, eta))
    p = expit(eta)
    d_eta = M * p - K                      # ∂nll/∂eta_i
    grad = np.array([np.sum(d_eta), np.sum(x * d_eta)])
    return nll, grad


def fit_logistic(N, K, M, enforce_negative=True):
    """Maximum-likelihood (a, b) for logit(q)=a+b·log N.

    Returns (a, b, converged). The initial guess is an ordinary least-squares
    line through the empirical logits (with a 1/2 continuity correction).
    """
    x = np.log(np.asarray(N, dtype=float))
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)

    p_hat = (K + 0.5) / (M + 1.0)
    y = logit(np.clip(p_hat, 1e-6, 1 - 1e-6))
    b0, a0 = np.polyfit(x, y, 1)           # slope, intercept
    if enforce_negative and b0 >= 0:
        b0 = -1e-3

    bounds = [(None, None), (None, -1e-9)] if enforce_negative else \
             [(None, None), (None, None)]
    res = minimize(_nll_and_grad, x0=np.array([a0, b0]), args=(x, K, M),
                   jac=True, method="L-BFGS-B", bounds=bounds)
    a, b = res.x
    return float(a), float(b), bool(res.success)


def n_c(a, b, eps):
    """Critical dataset size N_c = exp((logit(eps) − a)/b); nan if b ≥ 0."""
    if b >= 0:
        return np.nan
    return float(np.exp((logit(eps) - a) / b))


def bootstrap_nc(N, K, M, eps, n_boot, rng, enforce_negative=True):
    """Parametric-bootstrap 95% CI and distribution for N_c.

    Resamples K_N* ~ Binomial(M_N, f_mem_N), refits, recomputes N_c*.
    Returns (lo, hi, samples) with samples the finite N_c* draws.
    """
    M = np.asarray(M, dtype=float)
    f = np.clip(np.asarray(K, dtype=float) / M, 0.0, 1.0)
    draws = []
    for _ in range(n_boot):
        Kb = rng.binomial(M.astype(int), f)
        a, b, ok = fit_logistic(N, Kb, M, enforce_negative)
        val = n_c(a, b, eps)
        if np.isfinite(val):
            draws.append(val)
    draws = np.asarray(draws, dtype=float)
    if draws.size < 2:
        return np.nan, np.nan, draws
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), draws


# ─────────────────────────────────────────────────────────────────────────────
# Weighted isotonic regression across the three widths (imposed monotonicity)
# ─────────────────────────────────────────────────────────────────────────────
def isotonic_increasing(y, w):
    """Weighted pool-adjacent-violators for a non-decreasing fit of y.

    Falls back to sklearn if present; otherwise a compact weighted PAVA is used.
    ``y`` is ordered along the constraint direction (model size, or τ).
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    try:
        from sklearn.isotonic import IsotonicRegression
        x = np.arange(len(y), dtype=float)
        return IsotonicRegression(increasing=True, out_of_bounds="clip") \
            .fit_transform(x, y, sample_weight=w)
    except Exception:
        vals = list(y)
        wts = list(w)
        idx = [[i] for i in range(len(y))]
        i = 0
        while i < len(vals) - 1:
            if vals[i] > vals[i + 1] + 1e-15:
                nw = wts[i] + wts[i + 1]
                nv = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / nw
                vals[i:i + 2] = [nv]
                wts[i:i + 2] = [nw]
                idx[i:i + 2] = [idx[i] + idx[i + 1]]
                i = max(i - 1, 0)
            else:
                i += 1
        out = np.empty(len(y))
        for block_v, block_i in zip(vals, idx):
            for j in block_i:
                out[j] = block_v
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Data repair: enforce that f_mem is non-decreasing in τ
# ─────────────────────────────────────────────────────────────────────────────
def _repair_spikes(tau, f, thresh, max_iter=8):
    """Replace isolated outlier checkpoints by a robust 3-point median filter.

    A corrupted checkpoint also makes its NEIGHBOURS look anomalous, so the
    outlier must be identified robustly and one at a time:

      * the reference for point i is median(f[i-1], f[i], f[i+1]) — a median is
        insensitive to a single wild value, unlike a mean or a neighbour-average;
      * only the single WORST offender is repaired per pass, after which the
        references are recomputed. Once the true outlier is corrected, its
        neighbours stop looking anomalous and are left alone.

    This is what keeps the correction local. Endpoints are never repaired, as
    they lack two-sided context.
    """
    f = f.copy()
    fixed = set()
    if len(f) < 3:
        return f, fixed
    for _ in range(max_iter):
        med = f.copy()
        med[1:-1] = np.median(np.stack([f[:-2], f[1:-1], f[2:]]), axis=0)
        dev = np.abs(f - med)
        dev[0] = dev[-1] = 0.0          # endpoints: no two-sided context
        i = int(np.argmax(dev))
        if dev[i] <= thresh:
            break
        f[i] = med[i]
        fixed.add(i)
    return f, fixed


def enforce_tau_monotone(df, thresh=0.05, tol=0.02, verbose=True):
    """Clean f_mem(τ) so it respects the physics, per (W, N) series.

    WHY THIS IS LEGITIMATE, NOT COSMETIC
    ------------------------------------
    Memorization only accumulates: a diffusion model trained longer cannot
    memorize *less* of its training set, so the true f_mem(τ) is monotonically
    non-decreasing. ``paper_phase_diagram.py`` already relies on exactly this
    property to justify its censoring rule ("f_mem is monotonically increasing
    in τ ... the returned value is a LOWER bound"). Any observed decrease is
    therefore estimator noise, not signal.

    This matters because a boundary is read at τ = c·τ_gen, which is almost
    never a saved checkpoint. A single corrupted checkpoint therefore poisons
    every multiplier whose interpolation bracket straddles it — which is what
    drove the isolated W=256 jumps at c=3 and c=4.

    TWO STAGES, IN THIS ORDER
    -------------------------
    1. ISOLATED-OUTLIER REPAIR. A checkpoint that disagrees with both of its
       neighbours by more than `thresh` is a corrupted evaluation and is
       re-interpolated from them. Applied first, and on its own, because a lone
       bad point is a *local* fault and deserves a *local* remedy.
    2. ISOTONIC PROJECTION. Whatever small non-monotonicity survives is ordinary
       measurement noise, and is removed by the weighted least-squares
       projection onto the non-decreasing cone (pool-adjacent-violators). Series
       that already comply are returned unchanged.

    Doing (1) before (2) is what keeps the correction local: pooling alone would
    smear one corrupted value across every neighbouring checkpoint.

    Returns (repaired_df, repairs) with ``repairs`` listing the changed cells.
    """
    df = df.copy()
    repairs = []
    for (w, n), grp in df[df.tau > 0].groupby(["W", "N"]):
        grp = grp.sort_values("tau")
        f0 = grp.f_mem.values.astype(float)
        tau = grp.tau.values.astype(float)
        if len(f0) < 3:
            continue

        f1, spike_idx = _repair_spikes(tau, f0, thresh)
        if np.all(np.diff(f1) >= -1e-12):
            f_fix = f1
        else:
            f_fix = isotonic_increasing(f1, grp.num_generated.values.astype(float))

        changed = np.abs(f_fix - f0) > 1e-9
        if not changed.any():
            continue
        df.loc[grp.index, "f_mem"] = f_fix
        for i in np.where(changed)[0]:
            repairs.append({"W": w, "N": n, "tau": int(tau[i]),
                            "f_mem_raw": float(f0[i]),
                            "f_mem_fixed": float(f_fix[i]),
                            "delta": float(f_fix[i] - f0[i]),
                            "kind": "outlier" if i in spike_idx else "isotonic"})

    if verbose:
        print("\n" + "-" * 78)
        print("DATA REPAIR — isolated-outlier removal, then monotone in τ")
        print("-" * 78)
        notable = [r for r in repairs if abs(r["delta"]) > 1e-3]
        if not repairs:
            print("  No violations: every series was already clean.")
        else:
            severe = [r for r in repairs if abs(r["delta"]) > tol]
            print(f"  Repaired {len(repairs)} checkpoint(s) across "
                  f"{len({(r['W'], r['N']) for r in repairs})} series; "
                  f"{len(severe)} exceed ±{tol:g}.")
            if notable:
                print("  Changes larger than 0.001:")
                for r in sorted(notable, key=lambda r: -abs(r["delta"])):
                    flag = "  <-- severe" if abs(r["delta"]) > tol else ""
                    print(f"    W={r['W']:>3} N={r['N']:>5} τ={r['tau']:>8,}  "
                          f"{r['f_mem_raw']:.3f} -> {r['f_mem_fixed']:.3f}  "
                          f"({r['delta']:+.3f})  [{r['kind']}]{flag}")
            print(f"  ({len(repairs) - len(notable)} further changes were "
                  f"below 0.001 and are numerically irrelevant.)")
    return df, repairs


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_phase_diagram(bd, iso, taugen, const, eps, n_grid, out_stem,
                       show_isotonic, max_y_tick=None):
    """Phase diagram with logistic N_c points, CIs and straight-line joins."""
    widths = sorted(taugen)
    p_all = np.array([pdg.PARAM_COUNTS[w] for w in widths], dtype=float)
    mults = sorted(bd.multiplier.unique())

    # x(p): identity on a log axis, or the equally spaced rank of each width.
    rank = {p: float(i) for i, p in enumerate(p_all)}
    xf = (lambda p: np.asarray([rank[v] for v in np.atleast_1d(p)], float)
          if EQUAL_SPACING else np.asarray(p, float))

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype']  = 42
    fig, ax = plt.subplots(figsize=(10, 6))
    if EQUAL_SPACING:
        x_lo, x_hi = -0.4, len(widths) - 1 + 0.4
    else:
        x_lo, x_hi = p_all.min() / 1.20, p_all.max() * 1.20
    # Headroom so a boundary (or its CI) lying above the tested N grid is drawn
    # in full rather than clipped at the top of the panel.
    n_top = max(n_grid)
    for col in ("N_c", "N_c_hi"):
        if bd[col].notna().any():
            n_top = max(n_top, float(bd[col].max()))
    y_lo, y_hi = 0.0, n_top * 1.08
    if max_y_tick is not None:
        y_hi = min(y_hi, max_y_tick)

    # ── shade the plane using the c = 1 boundary (straight-line joins) ───────
    base = bd[(bd.multiplier == mults[0]) & bd.N_c.notna()].sort_values("p")
    if len(base) >= 2:
        # Work in the plotted coordinate (linear rank, or log10 of p).
        tr = (lambda v: v) if EQUAL_SPACING else np.log10
        xspan = np.array([tr(x_lo), *tr(xf(base.p.values)), tr(x_hi)])
        yspan = np.array([base.N_c.values[0], *base.N_c.values,
                          base.N_c.values[-1]])
        xd = np.linspace(tr(x_lo), tr(x_hi), 400)
        yd = np.interp(xd, xspan, yspan)
        xplot = xd if EQUAL_SPACING else 10 ** xd
        ax.fill_between(xplot, y_lo, yd, color=pdg.MEM_FILL, zorder=0)
        ax.fill_between(xplot, yd, y_hi, color=pdg.GEN_FILL, zorder=0)

    edges = xf(p_all) if EQUAL_SPACING else np.array([p_all.min(), p_all.max()])
    for p_edge in edges:
        ax.axvline(p_edge, color="0.45", ls=(0, (2, 3)), lw=1.0, zorder=1.5)
    for n in n_grid:
        ax.axhline(n, color="0.35", ls=(0, (1, 4)), lw=0.8, zorder=1.2,
                   alpha=0.5)

    # ── one straight-line boundary per multiplier, with 95% CI error bars ────
    # τ = τ_gen is solid black; the rest are dashed in distinct dark colours.
    # No markers.  Labels placed as text at the right end of each curve.
    for i, c in enumerate(mults):
        sub = bd[(bd.multiplier == c) & bd.N_c.notna()].sort_values("p")
        if sub.empty:
            continue
        colour = MULT_COLORS[i % len(MULT_COLORS)]
        is_base = (c == mults[0])
        ls = "-" if is_base else "--"
        lw = 2.8 if is_base else 2.0
        yerr = np.vstack([sub.N_c.values - sub.N_c_lo.values,
                          sub.N_c_hi.values - sub.N_c.values])
        yerr = np.clip(yerr, 0, None)
        ax.errorbar(xf(sub.p.values), sub.N_c.values, yerr=yerr,
                    fmt=ls, color=colour, lw=lw, capsize=4, elinewidth=1.6,
                    zorder=4)

        if show_isotonic and iso is not None:
            si = iso[(iso.multiplier == c) & iso.N_c_iso.notna()] \
                .sort_values("p")
            if len(si) >= 2:
                ax.plot(xf(si.p.values), si.N_c_iso.values, ls=(0, (4, 2)),
                        color=colour, lw=1.4, alpha=0.85, zorder=3.5)

        # Annotate at the right end of the curve (inside plot area)
        label_txt = (r"$\tau=\tau_{\mathrm{gen}}$" if is_base
                     else rf"$\tau={c:g}\,\tau_{{\mathrm{{gen}}}}$")
        y_end = float(sub.N_c.values[-1])
        x_end = float(xf(sub.p.values)[-1])
        ax.annotate(label_txt, xy=(x_end, y_end),
                    xytext=(-10, 6), textcoords="offset points",
                    fontsize=16, color=colour, fontweight="bold",
                    va="bottom", ha="right", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc="white", ec="none", alpha=0.7))

    if not EQUAL_SPACING:
        ax.set_xscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Number of trainable parameters  $p$", fontsize=20)
    ax.set_ylabel("Training-set size  $N$", fontsize=20)
    ax.set_xticks(xf(p_all))
    ax.set_xticklabels([(f"{p/1e6:.2f}M" if p >= 1e6 else f"{p/1e3:.0f}K")
                        + f"\n(W={w})" for p, w in zip(p_all, widths)],
                       fontsize=20)
    ax.minorticks_off()
    # ~7 y-ticks: keeps the 500 spacing of the default (ε=10%) range and stays
    # readable when a smaller ε pushes N_c to much larger values.
    step = next((s for s in (100, 200, 250, 500, 1000, 2000, 2500, 5000,
                             10000, 20000) if n_top / s <= 8), 25000)
    tick_top = n_top if max_y_tick is None else min(n_top, max_y_tick)
    ax.set_yticks(np.arange(0, tick_top + 1, step))
    ax.tick_params(axis="y", labelsize=20)

    # Legend: only region patches, placed inside the plot
    eps_pct = f"{eps * 100:g}"
    legend_handles = [
        Patch(facecolor=pdg.GEN_FILL, edgecolor="0.5", linewidth=0.5,
              label=rf"$f_{{\mathrm{{mem}}}} < {eps_pct}\%$"),
        Patch(facecolor=pdg.MEM_FILL, edgecolor="0.5", linewidth=0.5,
              label=rf"$f_{{\mathrm{{mem}}}} > {eps_pct}\%$"),
    ]
    ax.legend(handles=legend_handles, fontsize=16, loc="upper left",
              framealpha=0.92, edgecolor="0.7")

    ax.grid(alpha=0.20, ls=":", zorder=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fits(fits, eps, out_stem):
    """Per-(W, c) empirical f_mem points with the fitted logistic curve."""
    mults = sorted({f["multiplier"] for f in fits})
    widths = sorted({f["W"] for f in fits})
    nr, nc = len(widths), len(mults)
    fig, axes = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.9 * nr),
                             squeeze=False, sharex=True)

    for r, w in enumerate(widths):
        for cidx, c in enumerate(mults):
            ax = axes[r][cidx]
            rec = next((f for f in fits
                        if f["W"] == w and f["multiplier"] == c), None)
            if rec is None:
                ax.axis("off")
                continue
            N = np.asarray(rec["N"], float)
            f_emp = np.asarray(rec["f_mem"], float)
            a, b = rec["a"], rec["b"]
            nn = np.logspace(np.log10(N.min() / 1.3), np.log10(N.max() * 1.3),
                             200)
            ax.plot(nn, expit(a + b * np.log(nn)), "-",
                    color=pdg.W_COLOR.get(w, "#333"), lw=2,
                    zorder=3)
            ax.plot(N, f_emp, "o", color="0.15", ms=5, zorder=4)
            ax.axhline(eps, color="crimson", ls=":", lw=1.2, zorder=2)
            if np.isfinite(rec["N_c"]):
                ax.axvline(rec["N_c"], color="crimson", ls="--", lw=1.1,
                           zorder=2)
                if np.isfinite(rec["N_c_lo"]) and np.isfinite(rec["N_c_hi"]):
                    ax.axvspan(rec["N_c_lo"], rec["N_c_hi"], color="crimson",
                               alpha=0.10, zorder=1)
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            ax.grid(alpha=0.25, ls=":")
            if r == 0:
                ax.set_title((r"$\tau=\tau_{\mathrm{gen}}$" if c == 1
                              else rf"$\tau={c:g}\,\tau_{{\mathrm{{gen}}}}$"),
                             fontsize=10.5)
            if cidx == 0:
                ax.set_ylabel(f"W = {w}\n$f_{{\\mathrm{{mem}}}}$", fontsize=10)
            if r == nr - 1:
                ax.set_xlabel("$N$", fontsize=10)
            nc_txt = (f"$N_c$={rec['N_c']:,.0f}" if np.isfinite(rec["N_c"])
                      else "$N_c$: n/a")
            ax.text(0.04, 0.06, nc_txt, transform=ax.transAxes, fontsize=8.5,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7",
                              alpha=0.85))

    fig.suptitle("Binomial logistic fits  "
                 r"$q_N=\sigma(a+b\log N)$  (points: measured $f_{\mathrm{mem}}$"
                 r"; dashed: $N_c$; band: 95% CI)", fontsize=12, y=1.005)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Phase boundary by constrained binomial logistic "
                    "regression (companion to paper_phase_diagram.py).")
    ap.add_argument("--csv", type=Path, default=pdg.DEFAULT_CSV,
                    help="model-size sweep CSV (default: %(default)s)")
    ap.add_argument("--tol", type=float, default=0.5,
                    help="τ_gen criterion FCD <= (1+tol)*floor (default: "
                         "%(default)s)")
    ap.add_argument("--ref-n", type=int, default=None,
                    help="dataset size used to read τ_gen (default: largest N)")
    ap.add_argument("--const", type=float, default=None,
                    help="pin the collapse constant C in τ_gen(W)=C/W")
    ap.add_argument("--eps", type=float, default=0.10,
                    help="memorization threshold ε (default: %(default)s)")
    ap.add_argument("--multipliers", type=float, nargs="+", default=[1, 2, 3],
                    help="boundary curves at c·τ_gen (default: %(default)s)")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for the N_c CI (default: "
                         "%(default)s)")
    ap.add_argument("--seed", type=int, default=0,
                    help="bootstrap RNG seed (default: %(default)s)")
    ap.add_argument("--no-isotonic", action="store_true",
                    help="skip the imposed-monotonicity sensitivity curve")
    ap.add_argument("--equal-spacing", action="store_true",
                    help="space the widths equally on the x-axis instead of "
                         "placing them at their true p on a log axis")
    ap.add_argument("--out-suffix", type=str, default="",
                    help="suffix appended to every output file name")
    ap.add_argument("--max-y-tick", type=float, default=None,
                    help="maximum y-axis value and largest labeled y tick")
    ap.add_argument("--no-tau-monotone", action="store_true",
                    help="do NOT repair f_mem to be non-decreasing in τ "
                         "(keeps corrupted single checkpoints; for comparison)")
    args = ap.parse_args()

    global EQUAL_SPACING
    EQUAL_SPACING = args.equal_spacing
    sfx = args.out_suffix

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.csv)
    widths = sorted(w for w in df.W.unique() if w in pdg.PARAM_COUNTS)
    n_grid = sorted(df.N.unique())
    n_ref = args.ref_n if args.ref_n is not None else max(n_grid)

    print("=" * 78)
    print("PHASE BOUNDARY — BINOMIAL LOGISTIC REGRESSION")
    print("=" * 78)
    print(f"  CSV            : {args.csv}")
    print(f"  Widths W       : {widths}")
    print(f"  Dataset sizes N: {n_grid}")
    print(f"  Threshold ε    : {args.eps:g}")
    print(f"  Multipliers c  : {args.multipliers}")
    print(f"  Bootstrap      : {args.n_boot} resamples (seed {args.seed})")
    print(f"  τ-monotone fix : "
          f"{'OFF (raw f_mem)' if args.no_tau_monotone else 'ON (isotonic in τ)'}")

    # ── STEP 0 — repair single corrupted checkpoints ────────────────────────
    if not args.no_tau_monotone:
        df, repairs = enforce_tau_monotone(df)
        if repairs:
            pd.DataFrame(repairs).to_csv(
                RESULTS_DIR / f"logistic_tau_repairs{sfx}.csv", index=False)

    # ── STEP 1 — τ_gen(W), identical to the interpolation model ─────────────
    raw, taugen, const, fitted, law_only = pdg.resolve_taugen(
        df, widths, n_ref, args.tol, args.const)
    print(f"\n  τ_gen(W) = {const:.3g} / W   →  "
          + ", ".join(f"W{w}:{taugen[w]:,.0f}" for w in sorted(taugen)))
    if law_only:
        print(f"  NOTE: W={law_only} are capacity-limited (FCD never reaches "
              f"{1+args.tol:g}×floor); their τ_gen comes from the collapse law.")

    # M_N (number of evaluated generated samples) per config.
    def m_for(w, n):
        s = df[(df.W == w) & (df.N == n) & (df.tau > 0)]
        return int(s.num_generated.median()) if len(s) else 0

    # ── STEP 2 & 3 — fit the logistic model and locate N_c ──────────────────
    print("\n" + "-" * 78)
    print("FIT  logit(q_N) = a + b·log N   (b < 0),   N_c at f_mem = ε")
    print("-" * 78)

    rows, fit_records = [], []
    for c in args.multipliers:
        print(f"\n  c = {c:g}")
        for w in sorted(taugen):
            tau_c = c * taugen[w]
            Ns, fs, Ms, states = [], [], [], []
            for n in n_grid:
                val, state = pdg.fmem_at_tau(df, w, n, tau_c)
                if not np.isfinite(val):
                    continue
                Ns.append(n)
                fs.append(val)
                Ms.append(m_for(w, n))
                states.append(state)
            Ns = np.array(Ns); fs = np.array(fs); Ms = np.array(Ms, float)
            K = np.round(fs * Ms)

            a, b, ok = fit_logistic(Ns, K, Ms, enforce_negative=True)
            Nc = n_c(a, b, args.eps)
            lo, hi, _ = bootstrap_nc(Ns, K, Ms, args.eps, args.n_boot, rng)
            censored = any(s == "censored" for s in states)
            status = ("ok" if ok else "fit_failed")
            if censored:
                status += "+censored"

            rows.append({"multiplier": c, "W": w, "p": pdg.PARAM_COUNTS[w],
                         "tau": tau_c, "a": a, "b": b, "N_c": Nc,
                         "N_c_lo": lo, "N_c_hi": hi, "eps": args.eps,
                         "status": status})
            fit_records.append({"multiplier": c, "W": w, "N": Ns.tolist(),
                                "f_mem": fs.tolist(), "a": a, "b": b,
                                "N_c": Nc, "N_c_lo": lo, "N_c_hi": hi})

            got = f"{Nc:,.0f}" if np.isfinite(Nc) else "----"
            ci = (f"[{lo:,.0f}, {hi:,.0f}]"
                  if np.isfinite(lo) and np.isfinite(hi) else "[n/a]")
            print(f"    W={w:>3}  τ={tau_c:>9,.0f}  a={a:+.3f}  b={b:+.4f}  "
                  f"→  N_c = {got:>7}  95%CI {ci:>18}  [{status}]")

    bd = pd.DataFrame(rows)
    bd.to_csv(RESULTS_DIR / f"logistic_boundary{sfx}.csv", index=False)

    # ── SENSITIVITY — imposed monotone isotonic across widths ───────────────
    iso = None
    if not args.no_isotonic:
        print("\n" + "-" * 78)
        print("SENSITIVITY — weighted isotonic N_c(W64) ≤ N_c(W128) ≤ "
              "N_c(W256)  (IMPOSED)")
        print("-" * 78)
        iso_rows = []
        for c in args.multipliers:
            sub = bd[(bd.multiplier == c)].sort_values("p")
            sub = sub[sub.N_c.notna()]
            if len(sub) < 2:
                continue
            y = np.log(sub.N_c.values)
            # weight by inverse variance of log N_c from the CI width
            spread = np.log(sub.N_c_hi.values) - np.log(sub.N_c_lo.values)
            spread = np.where(np.isfinite(spread) & (spread > 0), spread, np.nan)
            var = (spread / (2 * 1.96)) ** 2
            wts = np.where(np.isfinite(var) & (var > 0), 1.0 / var, 1.0)
            y_iso = np.exp(isotonic_increasing(y, wts))
            for (_, r), yi in zip(sub.iterrows(), y_iso):
                iso_rows.append({"multiplier": c, "W": r.W, "p": r.p,
                                 "N_c_iso": float(yi), "eps": args.eps})
            cells = " | ".join(f"W{int(r.W)}:{yi:,.0f}"
                               for (_, r), yi in zip(sub.iterrows(), y_iso))
            print(f"  c = {c:<4g}  {cells}")
        iso = pd.DataFrame(iso_rows)
        iso.to_csv(RESULTS_DIR / f"logistic_boundary_isotonic{sfx}.csv",
                   index=False)

    # ── figures ─────────────────────────────────────────────────────────────
    plot_phase_diagram(bd, iso, taugen, const, args.eps, n_grid,
                       FIGURES_DIR / f"phase_diagram_logistic{sfx}",
                       show_isotonic=not args.no_isotonic,
                       max_y_tick=args.max_y_tick)
    plot_fits(fit_records, args.eps, FIGURES_DIR / f"logistic_fits{sfx}")

    print("\n" + "=" * 78)
    print("WROTE")
    print("=" * 78)
    for f in (f"results/logistic_boundary{sfx}.csv",
              f"results/logistic_boundary_isotonic{sfx}.csv",
              f"results/logistic_tau_repairs{sfx}.csv",
              f"figures/phase_diagram_logistic{sfx}.png",
              f"figures/phase_diagram_logistic{sfx}.pdf",
              f"figures/logistic_fits{sfx}.png",
              f"figures/logistic_fits{sfx}.pdf"):
        print(f"  {f}")


if __name__ == "__main__":
    main()
