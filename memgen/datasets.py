"""Loading, splitting and caching of the channel datasets.

All three datasets are reduced to the same contract: a shuffled pool of complex
antenna-domain channels plus UE coordinates, from which nested training subsets
and a disjoint held-out test split are drawn. The permutation is generated once
per dataset and reused by every ``N``, so training sets are nested and every run
is evaluated against the same held-out users.
"""

from __future__ import annotations

import numpy as np
import torch

from .beamspace import normalise_per_sample, to_beamspace
from .config import SPLIT_SEED, DatasetSpec

__all__ = ["ChannelPool", "load_pool", "beamspace_tensor"]

#: DICHASUS port ordering: maps the 32 flat measurement ports onto the physical
#: 4 x 8 antenna grid of the base station.
DICHASUS_ANTENNA_MAP = np.array(
    [
        [28, 5, 10, 14, 6, 2, 16, 18],
        [19, 4, 23, 17, 20, 11, 9, 27],
        [31, 29, 0, 13, 1, 12, 3, 7],
        [30, 26, 21, 25, 22, 15, 24, 8],
    ],
    dtype=int,
)


class ChannelPool:
    """A shuffled pool of antenna-domain channels with nested-subset access.

    Attributes:
        h: complex channels, shape ``(U, Nr, Nt)``.
        coords: UE positions, shape ``(U, 3)``.
        los: boolean mask, ``True`` for line-of-sight users.
    """

    def __init__(self, spec: DatasetSpec, h: np.ndarray, coords: np.ndarray,
                 los: np.ndarray):
        self.spec = spec
        self.h = h
        self.coords = coords
        self.los = los

    def __len__(self) -> int:
        return len(self.h)

    def train_indices(self, n: int) -> np.ndarray:
        """First ``n`` users of the shuffle, balanced over LoS/NLoS if paired.

        Candidates are restricted to the complement of :meth:`test_indices`, so
        every nested training subset is disjoint from the held-out split.
        """
        pool = np.setdiff1d(np.arange(len(self)), self.test_indices(),
                            assume_unique=True)
        if not self.spec.paired:
            if n > len(pool):
                raise ValueError(f"Requested N={n} but only {len(pool)} users.")
            return pool[:n]
        los = pool[self.los[pool]]
        nlos = pool[~self.los[pool]]
        n_los, n_nlos = n // 2, n - n // 2
        if n_los > len(los) or n_nlos > len(nlos):
            raise ValueError(f"Requested N={n} exceeds the available users.")
        return np.concatenate([los[:n_los], nlos[:n_nlos]])

    def test_indices(self) -> np.ndarray:
        """Tail of the shuffle, disjoint from every nested training subset."""
        start, stop = self.spec.test_fraction
        return np.arange(int(start * len(self)), int(stop * len(self)))

    def antenna_array(self, indices: np.ndarray) -> np.ndarray:
        """Real/imaginary antenna-domain array ``(n, 2, rows, cols)``."""
        h = self.h[indices]
        return np.stack([h.real, h.imag], axis=1).astype(np.float32)

    def beamspace_array(self, indices: np.ndarray) -> np.ndarray:
        """Per-sample normalised beamspace array ``(n, 2, rows, cols)``."""
        hv = to_beamspace(self.h[indices], self.spec.rx_array, self.spec.tx_array)
        ri = np.stack([hv.real, hv.imag], axis=1).astype(np.float32)
        return normalise_per_sample(ri)


def _load_sionna(path, subcarrier: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a Sionna RT ``.npz`` into ``(H, coords)``.

    The stored tensor has shape ``(U, Nr, 1, Nt, 1, n_sub, 4)``; the last axis
    holds the complex channel followed by the UE ``(x, y, z)`` position.
    """
    raw = np.load(path)["combined_array"][:, :, 0, :, 0, subcarrier, :]
    return raw[:, :, :, 0].astype(np.complex64), raw[:, 0, 0, 1:].real.astype(np.float32)


def _load_dichasus(path) -> tuple[np.ndarray, np.ndarray]:
    """Read the DICHASUS measurements and re-index onto the physical 4x8 grid."""
    raw = np.load(path)
    h = raw["h_downlink"]                                   # (U, 2, 1, 32)
    flat = (h[:, 0, 0, :] + 1j * h[:, 1, 0, :]).astype(np.complex64)
    return flat[:, DICHASUS_ANTENNA_MAP], raw["ue_locations"].astype(np.float32)


def _shuffle(spec: DatasetSpec, n_users: int) -> np.ndarray:
    """Deterministic master permutation, cached next to the run directories."""
    path = spec.index_file()
    if path.exists():
        indices = np.load(path)["indices"]
        if len(indices) != n_users:
            raise ValueError(
                f"{path} holds {len(indices)} indices but the dataset has "
                f"{n_users} users; delete the file to regenerate it."
            )
        return indices
    indices = np.random.RandomState(SPLIT_SEED).permutation(n_users)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, indices=indices)
    return indices


def load_pool(spec: DatasetSpec) -> ChannelPool:
    """Load a dataset and apply its master shuffle."""
    missing = [p for p in spec.paths() if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing data file(s):\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nSee docs/datasets.md for the expected layout."
        )

    if spec.measured:
        h, coords = _load_dichasus(spec.paths()[0])
        los = np.ones(len(h), dtype=bool)
    elif spec.paired:
        h_los, c_los = _load_sionna(spec.paths()[0], spec.subcarrier)
        h_nlos, c_nlos = _load_sionna(spec.paths()[1], spec.subcarrier)
        # Shuffle each propagation condition on its own so that every nested
        # subset keeps the 50/50 LoS/NLoS balance.
        h = np.concatenate([h_los, h_nlos])
        coords = np.concatenate([c_los, c_nlos])
        los = np.concatenate([np.ones(len(h_los), bool), np.zeros(len(h_nlos), bool)])
    else:
        h, coords = _load_sionna(spec.paths()[0], spec.subcarrier)
        los = np.ones(len(h), dtype=bool)

    order = _shuffle(spec, len(h))
    return ChannelPool(spec, h[order], coords[order], los[order])


def beamspace_tensor(pool: ChannelPool, indices: np.ndarray) -> torch.Tensor:
    """Normalised beamspace channels as a float32 tensor ready for the model."""
    return torch.from_numpy(pool.beamspace_array(indices))
