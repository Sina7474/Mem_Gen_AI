"""Fidelity and memorisation versus training time.

One sweep replaces the per-ablation evaluation scripts: the dataset-size
scaling, the capacity scaling, the batch-size and full-batch controls and the
two replication datasets differ only in which runs they iterate over. Every row
of the resulting table is produced from a single set of generated channels, so
``FCD`` and ``f_mem`` for a given checkpoint are always mutually consistent.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from . import config as cfg
from .beamspace import features_from_antenna, features_from_beamspace
from .metrics import (fcd_against_folds, fit_gaussian, fold_gaussians,
                      memorisation_fraction, nearest_neighbour_ratio)
from .model import build_ddim
from .sampling import cached_samples, list_checkpoints, load_checkpoint

__all__ = ["RunSpec", "sweep_runs", "evaluate_sweep", "COLUMNS"]

COLUMNS = [
    "dataset", "N", "width", "batch_size", "tau", "epoch", "weights",
    "num_generated", "num_train", "num_test", "feature_dim", "n_folds",
    "fcd_gen_test", "fcd_gen_test_std", "fcd_train_test", "fcd_train_test_std",
    "f_mem", "f_mem_ci_low", "f_mem_ci_high", "kappa",
    "mean_ratio", "median_ratio",
]

#: Ablations reported in the paper, expressed as (N, W, B) grids.
SWEEPS: dict[str, str] = {
    "dataset-size": "N in {100..4000} at W=256, B=min(N,500)",
    "model-size": "W in {64,128,256} at N in {200,1000}",
    "batch-size": "B in {100,500,1000} at N=1000, W=256",
    "full-batch": "B=N at W=256 (noise-free gradient control)",
}


@dataclass(frozen=True)
class RunSpec:
    """One trained model to evaluate."""

    n: int
    width: int = cfg.DEFAULT_WIDTH
    batch_size: int | None = None

    def resolved_batch(self) -> int:
        return self.batch_size or cfg.batch_size_for(self.n)


def sweep_runs(sweep: str, spec: cfg.DatasetSpec,
               sizes: list[int] | None = None,
               widths: list[int] | None = None,
               batch_sizes: list[int] | None = None) -> list[RunSpec]:
    """Expand a named ablation into the list of runs it covers."""
    if sweep == "dataset-size":
        sizes = sizes or list(spec.sizes)
        width = (widths or [cfg.DEFAULT_WIDTH])[0]
        return [RunSpec(n, width) for n in sizes]
    if sweep == "model-size":
        sizes = sizes or [200, 1_000]
        widths = widths or [64, 128, 256]
        return [RunSpec(n, w) for n in sizes for w in widths]
    if sweep == "batch-size":
        sizes = sizes or [1_000]
        width = (widths or [cfg.DEFAULT_WIDTH])[0]
        batch_sizes = batch_sizes or [100, 500, 1_000]
        return [RunSpec(n, width, b) for n in sizes for b in batch_sizes]
    if sweep == "full-batch":
        sizes = sizes or [200, 500, 1_000]
        width = (widths or [cfg.DEFAULT_WIDTH])[0]
        return [RunSpec(n, width, n) for n in sizes]
    raise SystemExit(
        f"Unknown sweep {sweep!r}. Available: {', '.join(sorted(SWEEPS))}"
    )


def _reference_features(spec: cfg.DatasetSpec, run_dir: Path):
    """Beamspace features of the training and test splits of a run."""
    train = features_from_antenna(np.load(run_dir / "train.npy"),
                                  spec.rx_array, spec.tx_array)
    test = features_from_antenna(np.load(run_dir / "test.npy"),
                                 spec.rx_array, spec.tx_array)
    return train, test


def evaluate_sweep(sweep: str = "dataset-size",
                   dataset: str = cfg.DEFAULT_DATASET,
                   sizes: list[int] | None = None,
                   widths: list[int] | None = None,
                   batch_sizes: list[int] | None = None,
                   n_generated: int = cfg.N_GENERATED,
                   kappa: float = cfg.KAPPA,
                   weights: str = "ema",
                   seed: int = cfg.GENERATION_SEED,
                   output: Path | None = None) -> Path:
    """Evaluate every checkpoint of every run in an ablation.

    Returns the path of the written CSV table.
    """
    spec = cfg.get_dataset(dataset)
    runs = sweep_runs(sweep, spec, sizes, widths, batch_sizes)
    output = Path(output) if output else cfg.result_dir("tables") / f"{spec.key}_{sweep}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"{spec.label}  |  sweep: {sweep}  |  {len(runs)} run(s)")
    print(f"  samples/checkpoint: {n_generated}   kappa: {kappa:.4f}   "
          f"weights: {weights}")

    with output.open("w", newline="") as handle:
        csv.writer(handle).writerow(COLUMNS)

    for run in runs:
        batch_size = run.resolved_batch()
        run_dir = spec.run_dir(run.n, run.width, batch_size)
        checkpoints = list_checkpoints(run_dir)
        if not checkpoints:
            print(f"  skipping {run_dir.name}: no checkpoints found")
            continue

        feat_train, feat_test = _reference_features(spec, run_dir)
        folds = fold_gaussians(feat_test)
        floor_mean, floor_std = fcd_against_folds(*fit_gaussian(feat_train), folds)
        steps_per_epoch = max(1, -(-run.n // batch_size))

        print(f"  {run_dir.name}: floor FCD(train,test) = "
              f"{floor_mean:.4f} +/- {floor_std:.4f}")

        ddim = build_ddim(spec.sample_shape, run.width, cfg.DEVICE)
        rows = []
        for tau, path in tqdm(checkpoints, desc=f"    N={run.n} W={run.width}"):
            if not load_checkpoint(ddim, path, weights):
                continue
            samples, _ = cached_samples(ddim, run_dir, tau, n_generated,
                                        spec.sample_shape, weights, seed)
            feat_gen = features_from_beamspace(samples)

            fcd_mean, fcd_std = fcd_against_folds(*fit_gaussian(feat_gen), folds)
            nn_stats = nearest_neighbour_ratio(feat_gen, feat_train,
                                               device=cfg.DEVICE)
            f_mem, ci_low, ci_high = memorisation_fraction(nn_stats["ratio"], kappa)

            rows.append([
                spec.key, run.n, run.width, batch_size, tau,
                tau / steps_per_epoch, weights, len(feat_gen), len(feat_train),
                len(feat_test), feat_train.shape[1], cfg.N_FOLDS,
                fcd_mean, fcd_std, floor_mean, floor_std,
                f_mem, ci_low, ci_high, kappa,
                float(nn_stats["ratio"].mean()), float(np.median(nn_stats["ratio"])),
            ])

        with output.open("a", newline="") as handle:
            csv.writer(handle).writerows(rows)
        del ddim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Wrote {output}")
    return output
