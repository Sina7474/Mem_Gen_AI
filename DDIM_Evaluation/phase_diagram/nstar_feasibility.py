#!/usr/bin/env python
"""
Feasibility test for the asymptotic boundary  N*(p) = lim_{tau->inf} N_c(p;tau).

The paper can draw N*(p) only because its N_c(tau) curves visibly FLATTEN before
the training budget runs out.  This script asks the same question of our data:

    for each width W, sweep tau over the whole measured range, refit the
    binomial logistic model at every tau, and look at the local log-log slope

        beta(tau) = d log N_c / d log tau .

    beta -> 0   =>  N_c has saturated, N* is measurable.
    beta ~ const>0  =>  N_c is still a growing power law, N* is NOT reachable
                        and quoting one would be an extrapolation, not a
                        measurement.

Outputs
    results/nstar_feasibility.csv
    figures/nstar_feasibility.png / .pdf

Usage (from DDIM_Evaluation/phase_diagram/):
    python nstar_feasibility.py
"""
import os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_phase_diagram as pdg
from phase_boundary_logistic import (
    fit_logistic, n_c, bootstrap_nc, enforce_tau_monotone,
)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

EPS = 0.10
TOL = 0.5
N_BOOT = 400          # lighter than the production run: this is a trend plot
W_COLOR = {64: "#0072B2", 128: "#D55E00", 256: "#009E73"}   # colourblind-safe


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    df = pd.read_csv(pdg.DEFAULT_CSV)
    widths = sorted(w for w in df.W.unique() if w in pdg.PARAM_COUNTS)
    n_grid = sorted(df.N.unique())
    n_ref = max(n_grid)

    df, _ = enforce_tau_monotone(df)
    _, taugen, const, _, _ = pdg.resolve_taugen(df, widths, n_ref, TOL, None)

    def m_for(w, n):
        s = df[(df.W == w) & (df.N == n) & (df.tau > 0)]
        return int(s.num_generated.median()) if len(s) else 0

    print("=" * 78)
    print("N*(p) FEASIBILITY  —  does N_c(tau) saturate inside our budget?")
    print("=" * 78)
    print(f"  tau_gen(W) = {const:.3g}/W  ->  "
          + ", ".join(f"W{w}:{taugen[w]:,.0f}" for w in widths))
    print(f"  epsilon = {EPS}")

    rows = []
    for w in widths:
        # Largest tau at which EVERY N of this width still has a measurement.
        tau_max_common = min(
            df[(df.W == w) & (df.N == n) & (df.tau > 0)].tau.max()
            for n in n_grid)
        tg = taugen[w]
        # Sweep from tau_gen up to the last uncensored tau.
        taus = np.unique(np.round(
            np.logspace(np.log10(tg), np.log10(tau_max_common), 14)))
        print(f"\n  W={w}   tau_gen={tg:,.0f}   uncensored up to "
              f"tau={tau_max_common:,.0f}  (c_max={tau_max_common/tg:.1f})")
        print(f"    {'tau':>9} {'c':>5} {'N_c':>7} {'95% CI':>17}")

        for t in taus:
            Ns, fs, Ms = [], [], []
            for n in n_grid:
                val, _ = pdg.fmem_at_tau(df, w, n, t)
                if not np.isfinite(val):
                    continue
                Ns.append(n); fs.append(val); Ms.append(m_for(w, n))
            Ns = np.array(Ns); Ms = np.array(Ms, float)
            K = np.round(np.array(fs) * Ms)

            a, b, ok = fit_logistic(Ns, K, Ms, enforce_negative=True)
            Nc = n_c(a, b, EPS)
            lo, hi, _ = bootstrap_nc(Ns, K, Ms, EPS, N_BOOT, rng)
            rows.append({"W": w, "p": pdg.PARAM_COUNTS[w], "tau": float(t),
                         "c": float(t / tg), "N_c": Nc,
                         "N_c_lo": lo, "N_c_hi": hi})
            print(f"    {t:>9,.0f} {t/tg:>5.1f} {Nc:>7,.0f} "
                  f"  [{lo:>6,.0f}, {hi:>6,.0f}]")

    res = pd.DataFrame(rows)

    # ── local log-log slope  beta = d log N_c / d log tau ──────────────────
    print("\n" + "-" * 78)
    print("LOCAL SLOPE  beta = d log N_c / d log tau   "
          "(beta -> 0 means saturation)")
    print("-" * 78)
    slope_rows = []
    for w in widths:
        s = res[res.W == w].sort_values("tau")
        lx, ly = np.log(s.tau.values), np.log(s.N_c.values)
        # slope over the FIRST and the LAST half of the sweep
        mid = len(lx) // 2
        b_early = np.polyfit(lx[:mid + 1], ly[:mid + 1], 1)[0]
        b_late = np.polyfit(lx[mid:], ly[mid:], 1)[0]
        b_all = np.polyfit(lx, ly, 1)[0]
        slope_rows.append({"W": w, "beta_early": b_early,
                           "beta_late": b_late, "beta_all": b_all})
        print(f"  W={w:>3}   beta(early)={b_early:+.3f}   "
              f"beta(late)={b_late:+.3f}   beta(all)={b_all:+.3f}")

    sl = pd.DataFrame(slope_rows)
    res.to_csv(RESULTS_DIR / "nstar_feasibility.csv", index=False)
    sl.to_csv(RESULTS_DIR / "nstar_feasibility_slopes.csv", index=False)

    saturated = (sl.beta_late.abs() < 0.05).all()
    print()
    if saturated:
        print("  VERDICT: N_c has flattened -> N*(p) IS measurable.")
    else:
        print("  VERDICT: beta(late) is still clearly positive for at least one")
        print("           width -> N_c is STILL GROWING at our largest tau.")
        print("           N*(p) is NOT measurable from this data; any value")
        print("           would be an extrapolation, not a measurement.")

    # ── figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    ax = axes[0]
    for w in widths:
        s = res[res.W == w].sort_values("tau")
        ax.plot(s.tau, s.N_c, "-o", color=W_COLOR[w], lw=2.2, ms=6,
                label=f"W = {w}")
        ax.fill_between(s.tau, s.N_c_lo, s.N_c_hi, color=W_COLOR[w], alpha=0.18)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ (optimizer updates)", fontsize=16)
    ax.set_ylabel(r"$N_c(\tau)$", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title(r"(a)  boundary vs training time", fontsize=15)

    ax = axes[1]
    width_bar = 0.35
    xpos = np.arange(len(widths))
    ax.bar(xpos - width_bar / 2, sl.beta_early, width_bar,
           label=r"early half", color="0.65", edgecolor="k", lw=0.6)
    ax.bar(xpos + width_bar / 2, sl.beta_late, width_bar,
           label=r"late half", color="#0072B2", edgecolor="k", lw=0.6)
    ax.axhline(0, color="k", lw=1.2)
    ax.axhspan(-0.05, 0.05, color="green", alpha=0.15)
    ax.text(0.98, 0.04, "saturation band", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=11, color="darkgreen")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"W = {w}" for w in widths], fontsize=14)
    ax.set_ylabel(r"$\beta=d\log N_c/d\log\tau$", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title(r"(b)  is the growth slowing down?", fontsize=15)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"nstar_feasibility.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote figures/nstar_feasibility.png / .pdf")
    print(f"  wrote results/nstar_feasibility.csv")


if __name__ == "__main__":
    main()
