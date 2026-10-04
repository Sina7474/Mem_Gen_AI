"""Training time is measured in optimiser steps.

The single trainer below covers every run reported in the paper: the three
datasets, the dataset-size sweep, the model-capacity sweep and the batch-size
and full-batch controls are all expressed as arguments rather than as separate
scripts. Checkpoints are written on a logarithmic grid of ``tau`` (about six
points per decade) together with the weight EMA that all sampling uses.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config as cfg
from .datasets import load_pool
from .model import build_ddim, evaluate_denoising_loss, find_t_eval

__all__ = ["train"]


def _ema_decay(step: int, base: float) -> float:
    """Warm-up schedule so early checkpoints track the live weights."""
    return min(base, (1.0 + step) / (10.0 + step))


@torch.no_grad()
def _ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(decay).add_(m.detach(), alpha=1.0 - decay)
    for e, m in zip(ema.buffers(), model.buffers()):
        e.copy_(m)


def train(dataset: str = cfg.DEFAULT_DATASET,
          n: int = 1_000,
          width: int = cfg.DEFAULT_WIDTH,
          batch_size: int | None = None,
          full_batch: bool = False,
          max_tau: int = cfg.DEFAULT_MAX_TAU,
          seed: int = cfg.DEFAULT_SEED,
          ema_decay: float = cfg.EMA_DECAY,
          resume: bool = True) -> Path:
    """Train one DDIM and checkpoint it along the ``tau`` grid.

    Args:
        dataset: key into :data:`memgen.config.DATASETS`.
        n: number of training channels.
        width: U-Net base width ``W``.
        batch_size: mini-batch size; defaults to ``min(N, 500)``.
        full_batch: use ``B = N`` (the noise-free gradient control).
        max_tau: number of optimiser steps to train for.
        seed: seed for the model initialisation and the data order.
        ema_decay: base decay of the weight EMA.
        resume: continue from the newest checkpoint if one exists.

    Returns:
        The run directory holding the checkpoints and the loss curve.
    """
    spec = cfg.get_dataset(dataset)
    batch_size = n if full_batch else (batch_size or cfg.batch_size_for(n))
    steps_per_epoch = math.ceil(n / batch_size)
    grid = cfg.tau_grid(max_tau)

    np.random.seed(seed)
    torch.manual_seed(seed)
    device = cfg.DEVICE

    run_dir = spec.run_dir(n, width, batch_size, seed)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- data
    pool = load_pool(spec)
    train_idx, test_idx = pool.train_indices(n), pool.test_indices()
    x_train = torch.from_numpy(pool.beamspace_array(train_idx))
    x_test = torch.from_numpy(pool.beamspace_array(test_idx))

    # Antenna-domain copies are what the downstream tasks and the metrics read.
    np.save(run_dir / "train.npy", pool.antenna_array(train_idx))
    np.save(run_dir / "test.npy", pool.antenna_array(test_idx))
    np.save(run_dir / "train_coords.npy", pool.coords[train_idx])
    np.save(run_dir / "test_coords.npy", pool.coords[test_idx])
    np.save(run_dir / "train_indices.npy", train_idx)

    # ---------------------------------------------------------------- model
    ddim = build_ddim(spec.sample_shape, width, device)
    ema = copy.deepcopy(ddim).to(device).eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    optimiser = torch.optim.Adam(ddim.parameters(), lr=cfg.LEARNING_RATE)

    t_eval, alpha_bar, noise_level = find_t_eval(ddim.alphabar_t)
    generator = torch.Generator().manual_seed(cfg.EVAL_NOISE_SEED)
    noise_train = torch.randn(x_train.shape, generator=generator)
    noise_test = torch.randn(x_test.shape, generator=generator)

    (run_dir / "config.json").write_text(json.dumps({
        "dataset": spec.key, "N": n, "width": width, "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch, "max_tau": max_tau, "seed": seed,
        "n_T": cfg.N_T, "betas": list(cfg.BETAS), "learning_rate": cfg.LEARNING_RATE,
        "ema_decay": ema_decay, "tau_grid": grid, "t_eval": t_eval,
        "alpha_bar_t_eval": alpha_bar, "noise_level_t_eval": noise_level,
        "num_train": len(train_idx), "num_test": len(test_idx),
        "parameters": sum(p.numel() for p in ddim.parameters()),
    }, indent=2))

    # --------------------------------------------------------------- resume
    step, epoch = 0, 0
    from .sampling import list_checkpoints  # local import avoids a cycle
    existing = list_checkpoints(run_dir)
    if resume and existing:
        tau_last, path = existing[-1]
        state = torch.load(path, map_location=device, weights_only=False)
        ddim.load_state_dict(state["model_state_dict"])
        ema.load_state_dict(state["ema_model_state_dict"])
        optimiser.load_state_dict(state["optimizer_state_dict"])
        step, epoch = int(state["tau"]), int(state["epoch"])
        grid = [t for t in grid if t > step]
        print(f"Resuming {run_dir.name} from tau={step}")

    curve = run_dir / "loss_curve.csv"
    if not curve.exists():
        with curve.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["N", "width", "batch_size", "tau", "epoch", "loss_train",
                 "loss_test", "generalisation_gap"]
            )

    def record(tau: int, epoch_float: float) -> None:
        loss_train = evaluate_denoising_loss(ddim, x_train, noise_train, t_eval, device)
        loss_test = evaluate_denoising_loss(ddim, x_test, noise_test, t_eval, device)
        torch.save({
            "model_state_dict": ddim.state_dict(),
            "ema_model_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "tau": tau, "epoch": epoch, "N": n, "width": width,
            "batch_size": batch_size, "dataset": spec.key,
        }, run_dir / "checkpoints" / f"checkpoint_tau_{tau}.pt")
        with curve.open("a", newline="") as handle:
            csv.writer(handle).writerow(
                [n, width, batch_size, tau, epoch_float, loss_train, loss_test,
                 loss_test - loss_train]
            )
        print(f"  tau={tau:>7d}  L_train={loss_train:.6f}  "
              f"L_test={loss_test:.6f}  gap={loss_test - loss_train:+.6f}")

    print(f"Training {spec.label}: N={n}, W={width}, B={batch_size}, "
          f"max_tau={max_tau} on {device}")
    if step == 0:
        record(0, 0.0)

    # ---------------------------------------------------------------- loop
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size,
                        shuffle=True, drop_last=False)
    pending = list(grid)
    started = time.time()
    while step < max_tau:
        epoch += 1
        ddim.train()
        for (batch,) in loader:
            loss = ddim(batch.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            step += 1
            _ema_update(ema, ddim, _ema_decay(step, ema_decay))

            if pending and step >= pending[0]:
                record(pending.pop(0), step / steps_per_epoch)
                ddim.train()
            if step >= max_tau:
                break

    torch.save(ema.state_dict(), run_dir / "ema_final.pt")
    print(f"Done in {(time.time() - started) / 60:.1f} min -> {run_dir}")
    return run_dir
