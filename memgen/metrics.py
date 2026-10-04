"""Fidelity and memorisation metrics.

Two numbers summarise a checkpoint:

``FCD``
    The Frechet Channel Distance, i.e. the Frechet distance between Gaussians
    fitted to the generated and to the held-out real channels in the flattened
    beamspace feature space. It is the channel analogue of the FID and plays
    the role of the distributional-fidelity axis. Reporting it against five
    disjoint test folds yields error bars, and ``FCD(train, test)`` gives the
    irreducible finite-sample floor.

``f_mem``
    The fraction of generated channels that fall much closer to their nearest
    training channel than to the second nearest, i.e. ``d1 / d2 < kappa``. It
    is the sample-level memorisation axis.

The generalisation window is the band of training times in which the FCD has
already descended onto its floor while ``f_mem`` is still negligible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy import linalg

from .config import (BOOTSTRAP_B, BOOTSTRAP_SEED, FMEM_THRESHOLD, FOLD_SEED,
                     KAPPA, NN_BATCH_SIZE, N_FOLDS, QUALITY_TOL)

__all__ = [
    "fit_gaussian",
    "frechet_distance",
    "fold_gaussians",
    "fcd_against_folds",
    "nearest_neighbour_ratio",
    "memorisation_fraction",
    "Window",
    "generalisation_window",
]


# --------------------------------------------------------------------------- #
# Frechet Channel Distance
# --------------------------------------------------------------------------- #

def fit_gaussian(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of a ``(M, D)`` feature matrix."""
    return features.mean(axis=0), np.cov(features, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """``||mu1 - mu2||^2 + tr(S1 + S2 - 2 (S1 S2)^{1/2})``."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def fold_gaussians(features: np.ndarray, n_folds: int = N_FOLDS,
                   seed: int = FOLD_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split the reference set into disjoint folds and fit a Gaussian to each."""
    order = np.random.default_rng(seed).permutation(len(features))
    return [fit_gaussian(features[fold]) for fold in np.array_split(order, n_folds)]


def fcd_against_folds(mu, sigma, folds) -> tuple[float, float]:
    """Mean and standard deviation of the FCD across the reference folds."""
    values = np.array([frechet_distance(mu, sigma, m, s) for m, s in folds])
    return float(values.mean()), float(values.std())


# --------------------------------------------------------------------------- #
# Memorisation
# --------------------------------------------------------------------------- #

def nearest_neighbour_ratio(generated: np.ndarray, train: np.ndarray,
                            batch_size: int = NN_BATCH_SIZE,
                            device=None, eps: float = 1e-12) -> dict[str, np.ndarray]:
    """Ratio ``d1 / d2`` of the two nearest training neighbours per sample.

    Returns a dict with the ``ratio``, the two distances ``d1``/``d2`` and the
    index of the nearest training channel (useful for qualitative inspection).
    """
    device = device or torch.device("cpu")
    train_t = torch.as_tensor(train, dtype=torch.float32, device=device)
    gen_t = torch.as_tensor(generated, dtype=torch.float32)

    ratios, d1s, d2s, nearest = [], [], [], []
    for start in range(0, len(gen_t), batch_size):
        batch = gen_t[start:start + batch_size].to(device)
        top2 = torch.topk(torch.cdist(batch, train_t, p=2), k=2,
                          largest=False, dim=1)
        d1, d2 = top2.values[:, 0], top2.values[:, 1]
        ratios.append((d1 / (d2 + eps)).cpu())
        d1s.append(d1.cpu())
        d2s.append(d2.cpu())
        nearest.append(top2.indices[:, 0].cpu())

    return {
        "ratio": torch.cat(ratios).numpy(),
        "d1": torch.cat(d1s).numpy(),
        "d2": torch.cat(d2s).numpy(),
        "nearest_index": torch.cat(nearest).numpy(),
    }


def memorisation_fraction(ratios: np.ndarray, kappa: float = KAPPA,
                          n_boot: int = BOOTSTRAP_B,
                          seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """``f_mem`` with a 95 % bootstrap confidence interval."""
    flags = (ratios < kappa).astype(np.float64)
    point = float(flags.mean())
    if n_boot <= 0:
        return point, point, point
    rng = np.random.default_rng(seed)
    draws = flags[rng.integers(0, len(flags), size=(n_boot, len(flags)))].mean(axis=1)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# --------------------------------------------------------------------------- #
# Generalisation window
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Window:
    """Boundaries of the generalisation window, in optimiser steps."""

    tau_gen: int | None
    tau_mem: int | None
    floor: float

    @property
    def is_open(self) -> bool:
        return self.tau_gen is not None and (
            self.tau_mem is None or self.tau_mem > self.tau_gen
        )

    @property
    def width(self) -> float:
        if self.tau_gen is None or self.tau_mem is None:
            return float("inf") if self.tau_gen is not None else 0.0
        return self.tau_mem / self.tau_gen


def generalisation_window(taus, fcd, fmem, floor: float,
                          quality_tol: float = QUALITY_TOL,
                          fmem_threshold: float = FMEM_THRESHOLD) -> Window:
    """Locate ``[tau_gen, tau_mem]`` on a single fidelity/memorisation curve.

    ``tau_gen`` is the first checkpoint whose FCD is within ``quality_tol`` of
    the real-real floor; ``tau_mem`` the first checkpoint at or after it whose
    memorisation fraction reaches ``fmem_threshold``. Either may be ``None``
    when the measured range does not contain the crossing.
    """
    taus = np.asarray(taus, dtype=float)
    order = np.argsort(taus)
    taus, fcd, fmem = taus[order], np.asarray(fcd)[order], np.asarray(fmem)[order]

    reached = np.flatnonzero(fcd <= floor * (1.0 + quality_tol))
    tau_gen = int(taus[reached[0]]) if reached.size else None

    exceeded = np.flatnonzero(fmem >= fmem_threshold)
    if tau_gen is not None:
        exceeded = exceeded[taus[exceeded] >= tau_gen]
    tau_mem = int(taus[exceeded[0]]) if exceeded.size else None

    return Window(tau_gen=tau_gen, tau_mem=tau_mem, floor=float(floor))
