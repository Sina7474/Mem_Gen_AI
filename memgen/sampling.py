"""Checkpoint discovery and cached sampling from trained DDIMs.

Generation is the only expensive step of the evaluation pipeline, so samples
are written to ``<run_dir>/samples/`` the first time a checkpoint is visited
and reused by every metric afterwards.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

from .config import (DEVICE, GENERATION_BATCH, GENERATION_SEED, DatasetSpec,
                     set_seed)
from .model import DDIM, build_ddim

__all__ = ["list_checkpoints", "load_checkpoint", "generate", "cached_samples"]

_CKPT_RE = re.compile(r"checkpoint_tau_(\d+)\.pt$")


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """All ``(tau, path)`` pairs of a run, ordered by training time."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    found = []
    for path in ckpt_dir.glob("checkpoint_tau_*.pt"):
        match = _CKPT_RE.search(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)


def load_checkpoint(ddim: DDIM, path: Path, weights: str = "ema") -> bool:
    """Load the EMA (default) or raw weights of a checkpoint into ``ddim``."""
    key = "ema_model_state_dict" if weights == "ema" else "model_state_dict"
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    if key not in state:
        return False
    ddim.load_state_dict(state[key])
    return True


@torch.no_grad()
def generate(ddim: DDIM, n_samples: int, sample_shape: tuple[int, int, int],
             seed: int = GENERATION_SEED) -> np.ndarray:
    """Draw ``n_samples`` channels with the deterministic DDIM sampler.

    Seeding here fixes the initial noise, so a given checkpoint always yields
    the same channels and every metric is computed on the same sample set.
    """
    set_seed(seed)
    ddim.eval()
    chunks, drawn = [], 0
    while drawn < n_samples:
        size = min(GENERATION_BATCH, n_samples - drawn)
        chunks.append(ddim.sample(size, sample_shape, DEVICE).cpu().numpy())
        drawn += size
    return np.concatenate(chunks)[:n_samples]


def cached_samples(ddim: DDIM, run_dir: Path, tau: int, n_samples: int,
                   sample_shape: tuple[int, int, int], weights: str = "ema",
                   seed: int = GENERATION_SEED) -> tuple[np.ndarray, bool]:
    """Return ``(samples, cache_hit)``, generating and caching on a miss."""
    cache_dir = Path(run_dir) / "samples"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{weights}_tau{tau}_seed{seed}.npz"

    if cache.exists():
        try:
            stored = np.load(cache)["channels"]
            if len(stored) >= n_samples:
                return stored[:n_samples], True
        except (OSError, KeyError, ValueError):
            pass  # truncated or corrupt cache: regenerate below

    samples = generate(ddim, n_samples, sample_shape, seed=seed)
    np.savez_compressed(cache, channels=samples)
    return samples, False


def model_for(spec: DatasetSpec, width: int) -> DDIM:
    """Convenience constructor used by the evaluation entry points."""
    return build_ddim(spec.sample_shape, width, DEVICE)
