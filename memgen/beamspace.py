"""Beamspace transforms and the feature map used by every metric.

The DDIM operates on per-sample normalised beamspace channels. The same
representation defines the feature space in which the Frechet Channel Distance
and the nearest-neighbour memorisation test are evaluated, so fidelity and
memorisation are always measured in identical coordinates.

A uniform planar array with ``nx`` columns and ``ny`` rows has the separable
codebook ``A = kron(F_ny, F_nx)``; a uniform linear array of ``n`` elements is
the special case ``(n, 1)``. The forward transform is ``Hv = Ar^H H At``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "dft_matrix",
    "upa_dft_codebook",
    "to_beamspace",
    "to_antenna",
    "normalise_per_sample",
    "features_from_antenna",
    "features_from_beamspace",
]


def dft_matrix(n: int) -> np.ndarray:
    """Unitary ``n x n`` DFT matrix."""
    k = np.arange(n)
    return np.exp(-2j * np.pi * k.reshape(-1, 1) * k / n) / np.sqrt(n)


def upa_dft_codebook(nx: int, ny: int) -> np.ndarray:
    """Separable DFT codebook ``kron(F_ny, F_nx)`` of a uniform planar array."""
    return np.kron(dft_matrix(ny), dft_matrix(nx)).astype(np.complex64)


def to_beamspace(h: np.ndarray, rx_array: tuple[int, int],
                 tx_array: tuple[int, int]) -> np.ndarray:
    """Antenna domain -> beamspace, ``Hv = Ar^H H At``.

    ``h`` has shape ``(..., Nr, Nt)`` with ``Nr = prod(rx_array)`` and
    ``Nt = prod(tx_array)``.
    """
    ar = upa_dft_codebook(*rx_array)
    at = upa_dft_codebook(*tx_array)
    return ar.conj().T @ h @ at


def to_antenna(hv: np.ndarray, rx_array: tuple[int, int],
               tx_array: tuple[int, int]) -> np.ndarray:
    """Beamspace -> antenna domain, ``H = Ar Hv At^H`` (inverse transform)."""
    ar = upa_dft_codebook(*rx_array)
    at = upa_dft_codebook(*tx_array)
    return ar @ hv @ at.conj().T


def normalise_per_sample(ri: np.ndarray) -> np.ndarray:
    """Scale each sample so that its peak magnitude is one.

    ``ri`` has shape ``(N, 2, rows, cols)`` holding real and imaginary parts.
    """
    mag = np.hypot(ri[:, 0], ri[:, 1]).reshape(len(ri), -1).max(axis=1)
    mag = np.where(mag > 1e-12, mag, 1.0)[:, None, None]
    out = ri.astype(np.float32, copy=True)
    out[:, 0] /= mag
    out[:, 1] /= mag
    return out


def features_from_antenna(arr: np.ndarray, rx_array: tuple[int, int],
                          tx_array: tuple[int, int]) -> np.ndarray:
    """Real channels ``(N, 2, rows, cols)`` -> flattened normalised beamspace."""
    h = (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64)
    hv = to_beamspace(h, rx_array, tx_array)
    ri = np.stack([hv.real, hv.imag], axis=1).astype(np.float32)
    return normalise_per_sample(ri).reshape(len(ri), -1).astype(np.float64)


def features_from_beamspace(samples: np.ndarray) -> np.ndarray:
    """Generated samples are already normalised beamspace, so only flatten."""
    return samples.reshape(len(samples), -1).astype(np.float64)
