#!/usr/bin/env python3
"""Small data-free checks for core channel-analysis utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    beam = load_module("upa_beam_distance", "Code/DDIM_FMM/upa_beam_distance.py")
    erank = load_module("effective_rank_core", "DDIM_Evaluation/effective_rank_core.py")

    rng = np.random.default_rng(7)
    channels = (
        rng.standard_normal((4, 4, 32))
        + 1j * rng.standard_normal((4, 4, 32))
    ).astype(np.complex64)

    distance = beam.compute_upa_peak_beam_distance(
        channels, channels.copy(), N_tx_x=8, N_tx_y=4
    )
    if not np.allclose(distance.distances, 0.0):
        raise AssertionError("Identical channels must have zero peak-beam distance")

    transformed = erank.to_beamspace(channels, 2, 2, 8, 4)
    ranks, singular_values = erank.compute_effective_rank(transformed)
    if ranks.shape != (4,) or singular_values.shape != (4, 4):
        raise AssertionError("Unexpected effective-rank output shape")
    if not np.all(np.isfinite(ranks)):
        raise AssertionError("Effective-rank smoke test produced non-finite values")

    print("Smoke test passed: beam-distance and effective-rank utilities are operational.")


if __name__ == "__main__":
    main()
