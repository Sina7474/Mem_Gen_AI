"""The generalisation-memorisation phase diagram.

For a fixed capacity ``W`` the memorisation fraction measured at a training
time proportional to ``tau_gen(W)`` decreases with the dataset size. Modelling
the number of memorised samples as binomial with

    logit(q_N) = a + b log N,    b < 0,

and solving ``q_N = eps`` gives the critical dataset size ``N_c(W)`` that
separates the generalising from the memorising phase. Confidence intervals come
from a parametric bootstrap over the binomial counts.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as cfg

__all__ = [
    "tau_gen_per_width",
    "fmem_at",
    "fit_logistic",
    "critical_size",
    "bootstrap_critical_size",
    "build_boundary",
]


def tau_gen_per_width(table: pd.DataFrame, reference_n: int | None = None,
                      quality_tol: float = cfg.QUALITY_TOL) -> dict[int, int]:
    """First checkpoint at which each width reaches the fidelity floor.

    The paper's collapse prediction is ``W * tau_gen ~ const``; the returned
    dictionary is what verifies it.
    """
    tau_gen: dict[int, int] = {}
    for width, group in table.groupby("width"):
        if reference_n is None:
            target_n = group["N"].max()
        else:
            target_n = reference_n
        curve = group[group["N"] == target_n].sort_values("tau")
        if curve.empty:
            continue
        floor = float(curve["fcd_train_test"].iloc[0])
        reached = curve[curve["fcd_gen_test"] <= floor * (1.0 + quality_tol)]
        if not reached.empty:
            tau_gen[int(width)] = int(reached["tau"].iloc[0])
    return tau_gen


def fmem_at(curve: pd.DataFrame, tau: float) -> float | None:
    """Memorisation fraction at an arbitrary ``tau``, interpolated in log-tau.

    Returns ``None`` when ``tau`` lies beyond the last measured checkpoint, so
    that censored observations never silently create a boundary.
    """
    curve = curve.sort_values("tau")
    taus = curve["tau"].to_numpy(dtype=float)
    values = curve["f_mem"].to_numpy(dtype=float)
    taus, values = taus[taus > 0], values[taus > 0]
    if tau > taus[-1]:
        return None
    return float(np.interp(np.log(tau), np.log(taus), values))


def fit_logistic(sizes, counts, totals,
                 enforce_negative: bool = True) -> tuple[float, float, bool]:
    """Maximum-likelihood fit of ``logit(q_N) = a + b log N``."""
    log_n = np.log(np.asarray(sizes, dtype=float))
    counts = np.asarray(counts, dtype=float)
    totals = np.asarray(totals, dtype=float)

    def negative_log_likelihood(theta):
        a, b = theta
        logits = np.clip(a + b * log_n, -30.0, 30.0)
        q = 1.0 / (1.0 + np.exp(-logits))
        q = np.clip(q, 1e-12, 1.0 - 1e-12)
        return -np.sum(counts * np.log(q) + (totals - counts) * np.log1p(-q))

    bounds = [(-100.0, 100.0), (-100.0, -1e-6 if enforce_negative else 100.0)]
    result = minimize(negative_log_likelihood, x0=np.array([0.0, -1.0]),
                      method="L-BFGS-B", bounds=bounds)
    return float(result.x[0]), float(result.x[1]), bool(result.success)


def critical_size(a: float, b: float, eps: float) -> float:
    """Dataset size at which the fitted memorisation fraction equals ``eps``."""
    if b >= 0:
        return float("nan")
    return float(np.exp((np.log(eps / (1.0 - eps)) - a) / b))


def bootstrap_critical_size(sizes, counts, totals, eps: float,
                            n_boot: int = 1_000,
                            seed: int = 0) -> tuple[float, float]:
    """Parametric bootstrap confidence interval for :func:`critical_size`."""
    rng = np.random.default_rng(seed)
    counts = np.asarray(counts, dtype=int)
    totals = np.asarray(totals, dtype=int)
    draws = []
    for _ in range(n_boot):
        resampled = rng.binomial(totals, np.clip(counts / totals, 0.0, 1.0))
        a, b, ok = fit_logistic(sizes, resampled, totals)
        if ok:
            value = critical_size(a, b, eps)
            if np.isfinite(value):
                draws.append(value)
    if len(draws) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def build_boundary(table: Path | pd.DataFrame,
                   eps: float = 0.10,
                   multipliers: tuple[float, ...] = (1.0, 2.0, 5.0),
                   reference_n: int | None = None,
                   n_boot: int = 1_000,
                   output: Path | None = None) -> Path:
    """Estimate ``N_c(W)`` for each capacity and each multiple of ``tau_gen``.

    Args:
        table: the CSV written by :func:`memgen.evaluate.evaluate_sweep`, or an
            already-loaded frame. It must contain several widths and sizes.
        eps: memorisation level defining the phase boundary.
        multipliers: training times ``c * tau_gen(W)`` at which the boundary is
            evaluated.
        reference_n: dataset size used to read off ``tau_gen(W)``; defaults to
            the largest available.
        n_boot: bootstrap resamples for the confidence interval.

    Returns:
        Path of the boundary CSV.
    """
    frame = pd.read_csv(table) if isinstance(table, (str, Path)) else table
    output = Path(output) if output else cfg.result_dir("tables") / "phase_boundary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    tau_gen = tau_gen_per_width(frame, reference_n)
    if not tau_gen:
        raise SystemExit("No width reached the fidelity floor; nothing to fit.")
    print("tau_gen per width (collapse check, W * tau_gen should be ~constant):")
    for width, tau in sorted(tau_gen.items()):
        print(f"  W={width:>4d}  tau_gen={tau:>7d}  W*tau_gen={width * tau:.3e}")

    rows = []
    for width, tau_g in sorted(tau_gen.items()):
        per_width = frame[frame["width"] == width]
        for multiplier in multipliers:
            tau = tau_g * multiplier
            sizes, counts, totals = [], [], []
            for n, curve in per_width.groupby("N"):
                value = fmem_at(curve, tau)
                if value is None:
                    continue
                total = int(curve["num_generated"].iloc[0])
                sizes.append(int(n))
                counts.append(int(round(value * total)))
                totals.append(total)

            if len(sizes) < 3:
                rows.append([width, multiplier, tau, np.nan, np.nan, np.nan,
                             np.nan, np.nan, len(sizes), "censored"])
                continue

            a, b, ok = fit_logistic(sizes, counts, totals)
            n_c = critical_size(a, b, eps)
            lo, hi = bootstrap_critical_size(sizes, counts, totals, eps, n_boot)
            inside = min(sizes) <= n_c <= max(sizes)
            status = "measured" if ok and inside else (
                "above_range" if n_c > max(sizes) else "below_range"
            )
            rows.append([width, multiplier, tau, n_c, lo, hi, a, b,
                         len(sizes), status])

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["width", "multiplier", "tau", "N_c", "N_c_low",
                         "N_c_high", "a", "b", "n_points", "status"])
        writer.writerows(rows)

    print(f"Wrote {output}")
    return output
