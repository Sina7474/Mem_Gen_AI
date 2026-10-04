#!/usr/bin/env python
"""
Constrained vs Unconstrained logistic fit comparison.

For every (W, c), fits:
  logit(q_N) = a + b·log N
with:
  1. Constrained:   b < 0    (our production setting)
  2. Unconstrained: b ∈ ℝ    (no restriction)

Reports:
  - Unconstrained b̂ and its standard error (from observed Fisher information)
  - Constrained b̂
  - Resulting N_c for both fits
  - Difference in N_c

If all unconstrained b̂ < 0, then the constraint is non-binding and does not
alter the direction of results.

Usage (from DDIM_Evaluation/phase_diagram/):
    python validate_b_constraint.py
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit, logit
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_phase_diagram as pdg
from phase_boundary_logistic import (
    _nll_and_grad, fit_logistic, n_c, enforce_tau_monotone,
)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def observed_fisher(a, b, x, M):
    """Observed Fisher information matrix (Hessian of NLL at MLE).

    For binomial logistic regression with η_i = a + b·x_i:
        J_jk = Σ_i M_i · p_i · (1-p_i) · z_ij · z_ik
    where z_i = [1, x_i], p_i = sigmoid(η_i).
    Returns the 2×2 Fisher matrix.
    """
    eta = a + b * x
    p = expit(eta)
    w = M * p * (1 - p)  # weight per observation
    # Design matrix columns: [1, x_i]
    J = np.array([
        [np.sum(w),           np.sum(w * x)],
        [np.sum(w * x),       np.sum(w * x**2)],
    ])
    return J


def fit_unconstrained(N, K, M):
    """Fit logit(q)=a+b·logN with b∈ℝ (no constraint)."""
    x = np.log(np.asarray(N, dtype=float))
    K = np.asarray(K, dtype=float)
    M = np.asarray(M, dtype=float)

    p_hat = (K + 0.5) / (M + 1.0)
    y = logit(np.clip(p_hat, 1e-6, 1 - 1e-6))
    b0, a0 = np.polyfit(x, y, 1)

    res = minimize(_nll_and_grad, x0=np.array([a0, b0]), args=(x, K, M),
                   jac=True, method="L-BFGS-B",
                   bounds=[(None, None), (None, None)])
    a, b = res.x

    # Standard errors from inverse Fisher information
    J = observed_fisher(a, b, x, M)
    try:
        cov = np.linalg.inv(J)
        se_a = np.sqrt(cov[0, 0])
        se_b = np.sqrt(cov[1, 1])
    except np.linalg.LinAlgError:
        se_a = se_b = np.nan

    return float(a), float(b), float(se_a), float(se_b), bool(res.success)


def main():
    csv_path = pdg.DEFAULT_CSV
    df = pd.read_csv(csv_path)

    widths = sorted(w for w in df.W.unique() if w in pdg.PARAM_COUNTS)
    n_grid = sorted(df.N.unique())
    n_ref = max(n_grid)
    eps = 0.10
    multipliers = [1, 2, 3, 4, 5]
    tol = 0.5   # same as phase_boundary_logistic.py default
    const = None

    # Repair τ-monotone violations (same as production)
    df, _ = enforce_tau_monotone(df)

    # τ_gen
    raw, taugen, const_val, fitted, law_only = pdg.resolve_taugen(
        df, widths, n_ref, tol, const)

    def m_for(w, n):
        s = df[(df.W == w) & (df.N == n) & (df.tau > 0)]
        return int(s.num_generated.median()) if len(s) else 0

    print("=" * 90)
    print("VALIDATION: Constrained (b<0) vs Unconstrained (b∈ℝ) Logistic Fit")
    print("=" * 90)
    print(f"  Threshold ε = {eps}")
    print(f"  Multipliers c = {multipliers}")
    print(f"  Widths W = {widths}")
    print()

    rows = []
    all_b_negative = True

    print(f"{'c':>3} {'W':>4} {'b_uncon':>9} {'SE(b)':>8} {'b/SE':>6} "
          f"{'b_con':>9} {'Nc_uncon':>10} {'Nc_con':>10} {'ΔN_c':>8} "
          f"{'%diff':>7} {'b<0?':>5}")
    print("-" * 90)

    for c in multipliers:
        for w in sorted(taugen):
            tau_c = c * taugen[w]
            Ns, fs, Ms = [], [], []
            for n in n_grid:
                val, state = pdg.fmem_at_tau(df, w, n, tau_c)
                if not np.isfinite(val):
                    continue
                Ns.append(n)
                fs.append(val)
                Ms.append(m_for(w, n))
            Ns = np.array(Ns); fs = np.array(fs); Ms = np.array(Ms, float)
            K = np.round(fs * Ms)

            # Constrained fit (b < 0)
            a_con, b_con, ok_con = fit_logistic(Ns, K, Ms, enforce_negative=True)
            Nc_con = n_c(a_con, b_con, eps)

            # Unconstrained fit (b ∈ ℝ)
            a_unc, b_unc, se_a, se_b, ok_unc = fit_unconstrained(Ns, K, Ms)
            Nc_unc = n_c(a_unc, b_unc, eps) if b_unc < 0 else np.nan

            b_neg = b_unc < 0
            if not b_neg:
                all_b_negative = False

            # Difference
            if np.isfinite(Nc_con) and np.isfinite(Nc_unc):
                delta = Nc_unc - Nc_con
                pct = 100 * delta / Nc_con
            else:
                delta = np.nan
                pct = np.nan

            z = b_unc / se_b if se_b > 0 else np.nan

            rows.append({
                "c": c, "W": w, "tau_c": tau_c,
                "a_unconstrained": a_unc, "b_unconstrained": b_unc,
                "SE_b": se_b, "z_score": z,
                "a_constrained": a_con, "b_constrained": b_con,
                "Nc_unconstrained": Nc_unc, "Nc_constrained": Nc_con,
                "delta_Nc": delta, "pct_diff": pct,
                "b_negative": b_neg,
            })

            nc_u = f"{Nc_unc:,.0f}" if np.isfinite(Nc_unc) else "n/a"
            nc_c = f"{Nc_con:,.0f}" if np.isfinite(Nc_con) else "n/a"
            d_str = f"{delta:+,.0f}" if np.isfinite(delta) else "n/a"
            p_str = f"{pct:+.2f}%" if np.isfinite(pct) else "n/a"

            print(f"{c:>3} {w:>4} {b_unc:>+9.4f} {se_b:>8.4f} {z:>+6.2f} "
                  f"{b_con:>+9.4f} {nc_u:>10} {nc_c:>10} {d_str:>8} "
                  f"{p_str:>7} {'  ✓' if b_neg else '  ✗'}")

    print("-" * 90)
    print()

    # Summary
    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS_DIR / "constraint_validation.csv", index=False)

    if all_b_negative:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  ALL unconstrained b̂ < 0  →  constraint is NON-BINDING     ║")
        print("║  The hard constraint b<0 does NOT change results.           ║")
        print("╚══════════════════════════════════════════════════════════════╝")
    else:
        binding = result_df[~result_df.b_negative]
        print(f"WARNING: {len(binding)} cases have unconstrained b̂ ≥ 0:")
        print(binding[["c", "W", "b_unconstrained", "SE_b"]].to_string())

    # Summary statistics
    print(f"\n  Max |%diff in N_c|: {result_df.pct_diff.abs().max():.3f}%")
    print(f"  Mean |%diff|       : {result_df.pct_diff.abs().mean():.3f}%")
    print(f"  Max |ΔN_c|         : {result_df.delta_Nc.abs().max():.1f}")

    # ── Plot: b̂ ± 2·SE with zero line ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = []
    x_pos = []
    colors = []
    for i, row in result_df.iterrows():
        x_labels.append(f"c={row['c']}\nW={row['W']}")
        x_pos.append(i)
        colors.append('#2166ac' if row['b_negative'] else '#b2182b')

    b_vals = result_df.b_unconstrained.values
    se_vals = result_df.SE_b.values

    ax.errorbar(x_pos, b_vals, yerr=2*se_vals, fmt='none', ecolor='gray',
                capsize=4, capthick=1.2, zorder=3)
    ax.scatter(x_pos, b_vals, c=colors, s=80, zorder=5, edgecolors='k', lw=0.5)
    ax.scatter(x_pos, result_df.b_constrained.values, marker='x', c='red',
               s=60, zorder=4, label='Constrained b̂')
    ax.axhline(0, color='black', ls='--', lw=1.5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=7, rotation=0)
    ax.set_ylabel("b̂  (slope in logit scale)", fontsize=11)
    ax.set_title("Unconstrained b̂ ± 2·SE  (all should be < 0 for constraint to be non-binding)",
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    for ext in ['png', 'pdf']:
        fig.savefig(FIGURES_DIR / f"constraint_validation_b.{ext}",
                    dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Plot saved: {FIGURES_DIR}/constraint_validation_b.png")


if __name__ == "__main__":
    main()
