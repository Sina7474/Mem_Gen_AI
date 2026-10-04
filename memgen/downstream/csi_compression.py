"""CSI compression: is synthetic data useful, and for how long?

A CRNet autoencoder is trained on a fixed budget of ``K`` channels. The
reference arm uses only the ``N`` real channels that the DDIM itself was
trained on; the augmented arm keeps those ``N`` and fills the remaining
``K - N`` slots with channels sampled from a DDIM checkpoint. Sweeping the
checkpoint turns the downstream NMSE into a direct read-out of how useful the
generator is at each point of the generalisation window.

Validation and test channels are drawn from the held-out tail of the shuffle,
which the generator never saw, so no result can be explained by leakage.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .. import config as cfg
from ..beamspace import to_antenna
from ..datasets import load_pool
from ..model import build_ddim
from ..sampling import cached_samples, list_checkpoints, load_checkpoint
from .crnet import nmse_db, to_tensor, train_crnet  # noqa: F401  (re-exported)

__all__ = ["run"]

COLUMNS = [
    "experiment", "N", "tau", "K", "crnet_seed", "num_real", "num_synthetic",
    "best_epoch", "val_nmse_db", "test_nmse_db",
]


def _synthetic_to_complex(samples: np.ndarray, spec: cfg.DatasetSpec) -> np.ndarray:
    """Generated beamspace samples -> complex antenna-domain channels."""
    hv = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64)
    return to_antenna(hv, spec.rx_array, spec.tx_array)


def run(dataset: str = cfg.DEFAULT_DATASET,
        sizes: list[int] | None = None,
        taus: list[int] | None = None,
        k_total: int = 5_000,
        seeds: list[int] | None = None,
        epochs: int = 500,
        reduction: int = 4,
        width: int = cfg.DEFAULT_WIDTH,
        output: Path | None = None) -> Path:
    """Run the reference / augmented / full-real comparison.

    Args:
        dataset: generator dataset key.
        sizes: reference sizes ``N``; one DDIM run must exist per size.
        taus: DDIM checkpoints to augment from; defaults to every checkpoint.
        k_total: total number of training channels seen by every CRNet.
        seeds: CRNet seeds, averaged in the figures.
        epochs: CRNet training budget.
        reduction: CRNet compression ratio.
        width: U-Net width of the generator runs to draw samples from.

    Returns:
        Path of the results CSV.
    """
    spec = cfg.get_dataset(dataset)
    sizes = sizes or [200, 500, 1_000]
    seeds = seeds or [0, 1, 2, 3]
    output = Path(output) if output else cfg.result_dir("tables") / "csi_compression.csv"
    work_dir = cfg.result_dir("csi_compression")
    device = cfg.DEVICE

    pool = load_pool(spec)
    held_out = pool.test_indices()
    split = len(held_out) // 2
    x_val = to_tensor(pool.h[held_out[:split]])
    x_test = to_tensor(pool.h[held_out[split:]])
    nr, nt = spec.sample_shape[1], spec.sample_shape[2]

    print(f"CSI compression on {spec.label}")
    print(f"  budget K={k_total}, sizes {sizes}, seeds {seeds}")
    print(f"  validation {len(x_val)} / test {len(x_test)} held-out channels")

    rows: list[list] = []

    def train(experiment: str, n: int, tau: int | None,
              x_train: torch.Tensor, num_real: int, num_synth: int) -> None:
        for seed in seeds:
            tag = f"{experiment}_N{n}" + (f"_tau{tau}" if tau is not None else "")
            tag += f"_seed{seed}"
            record = train_crnet(
                x_train, x_val, x_test, tag,
                log_path=work_dir / "logs" / f"{tag}.json",
                ckpt_path=work_dir / "checkpoints" / f"{tag}.pt",
                nr=nr, nt=nt, reduction=reduction, epochs=epochs,
                seed=seed, device=device,
                extra={"experiment": experiment, "N": n, "tau": tau},
            )
            rows.append([experiment, n, tau if tau is not None else -1, k_total,
                         seed, num_real, num_synth, record["best_epoch"],
                         record["val_nmse_db"], record["test_nmse_db"]])

    # Full-real benchmark: the upper bound reachable with K real channels.
    benchmark = pool.h[pool.train_indices(k_total)]
    train("full_real", k_total, None, to_tensor(benchmark), k_total, 0)

    for n in sizes:
        run_dir = spec.run_dir(n, width, cfg.batch_size_for(n))
        checkpoints = list_checkpoints(run_dir)
        if not checkpoints:
            print(f"  skipping N={n}: no DDIM checkpoints in {run_dir}")
            continue
        if taus:
            checkpoints = [(t, p) for t, p in checkpoints if t in set(taus)]

        reference = pool.h[pool.train_indices(n)]
        x_reference = to_tensor(reference)
        train("reference_only", n, None, x_reference, n, 0)

        num_synth = k_total - n
        ddim = build_ddim(spec.sample_shape, width, device)
        for tau, path in checkpoints:
            if tau == 0 or not load_checkpoint(ddim, path):
                continue
            samples, _ = cached_samples(ddim, run_dir, tau, num_synth,
                                        spec.sample_shape)
            synthetic = to_tensor(_synthetic_to_complex(samples, spec))
            train("augmented", n, tau,
                  torch.cat([x_reference, synthetic]), n, num_synth)
        del ddim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"Wrote {output}")
    return output
