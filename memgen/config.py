"""Central configuration: paths, hyper-parameters and the dataset registry.

Every experiment in this repository is identified by a ``(dataset, N, W, B)``
tuple, where ``N`` is the training-set size, ``W`` the U-Net base width and
``B`` the mini-batch size. :class:`DatasetSpec` resolves that tuple to a run
directory so that training, evaluation and the downstream tasks all agree on
where artefacts live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Root directories
# --------------------------------------------------------------------------- #

#: Repository root (the directory containing this package).
REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where the raw channel files are expected. Override with ``MEMGEN_DATA_ROOT``.
DATA_ROOT = Path(os.environ.get("MEMGEN_DATA_ROOT", REPO_ROOT / "data"))

#: Where checkpoints, generated samples and caches are written.
#: Override with ``MEMGEN_RUN_ROOT``.
RUN_ROOT = Path(os.environ.get("MEMGEN_RUN_ROOT", REPO_ROOT / "runs"))

#: Where CSV tables and figures are written. Override with ``MEMGEN_RESULT_ROOT``.
RESULT_ROOT = Path(os.environ.get("MEMGEN_RESULT_ROOT", REPO_ROOT / "results"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every generator used by the package.

    With ``deterministic`` left on, two runs that share a seed produce
    bit-identical checkpoints. cuDNN otherwise selects convolution kernels by
    autotuning, and the resulting differences compound over training: measured
    on a W=256 run, weights diverged by 6.6e-2 after only 300 steps. Forcing
    deterministic kernels costs about 5 % in wall-clock time.

    Reproducibility is guaranteed for a fixed GPU model, driver and PyTorch
    version; results may still differ in the last bits across machines.

    Args:
        seed: value fed to NumPy and Torch.
        deterministic: disable cuDNN autotuning and nondeterministic kernels.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


# --------------------------------------------------------------------------- #
# Diffusion / optimisation hyper-parameters (identical across all experiments)
# --------------------------------------------------------------------------- #

N_T = 200                       #: number of diffusion steps
BETAS = (1e-4, 0.02)            #: linear beta schedule endpoints
LEARNING_RATE = 1e-4
DEFAULT_WIDTH = 256             #: U-Net base width W
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_TAU = 200_000
DEFAULT_SEED = 0
EMA_DECAY = 0.9999
EVAL_NOISE_SEED = 12345         #: fixed noise for the denoising-loss probe
SPLIT_SEED = 0                  #: seed of the master shuffle (nested subsets)

#: Checkpoint grid in optimiser steps tau (~6 points per decade).
TAU_GRID: tuple[int, ...] = (
    100, 150, 200, 300, 500, 700,
    1_000, 1_500, 2_000, 3_000, 5_000, 7_000,
    10_000, 15_000, 20_000, 30_000, 50_000, 70_000,
    100_000, 150_000, 200_000,
    250_000, 300_000, 350_000, 400_000,
)

#: U-Net parameter counts, used as the x-axis of the phase diagram.
PARAM_COUNTS = {64: 956_930, 128: 3_814_402, 256: 15_230_978}


# --------------------------------------------------------------------------- #
# Sampling / metric defaults
# --------------------------------------------------------------------------- #

N_GENERATED = 5_000             #: samples drawn per checkpoint
GENERATION_BATCH = 100
GENERATION_SEED = 0
N_FOLDS = 5                     #: disjoint test folds -> FCD error bars
FOLD_SEED = 0
KAPPA = 1.0 / 3.0               #: nearest-neighbour ratio threshold for f_mem
NN_BATCH_SIZE = 512
BOOTSTRAP_B = 1_000
BOOTSTRAP_SEED = 42

#: Generalisation window: tau_gen is the first checkpoint whose FCD is within
#: ``QUALITY_TOL`` of the real-real floor; tau_mem the first checkpoint whose
#: memorisation fraction exceeds ``FMEM_THRESHOLD``.
QUALITY_TOL = 0.10
FMEM_THRESHOLD = 0.01


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DatasetSpec:
    """Everything that distinguishes one channel dataset from another."""

    key: str
    label: str
    #: ``.npz`` files relative to :data:`DATA_ROOT` (LoS first, NLoS second).
    files: tuple[str, ...]
    #: Model input shape ``(2, rows, cols)``.
    sample_shape: tuple[int, int, int]
    #: Subcarrier taken from the Sionna tensor (unused by the measured set).
    subcarrier: int = 0
    #: UE / BS uniform planar array factorisation ``(Nx, Ny)``.
    rx_array: tuple[int, int] = (2, 2)
    tx_array: tuple[int, int] = (8, 4)
    #: ``True`` when the pool is split into equal LoS and NLoS halves.
    paired: bool = False
    #: ``True`` for the single-antenna DICHASUS measurements.
    measured: bool = False
    #: Fraction of the shuffled pool reserved for the test split.
    test_fraction: tuple[float, float] = (0.9, 1.0)
    sizes: tuple[int, ...] = field(default=(100, 200, 500, 1_000, 2_000, 4_000))

    @property
    def feature_dim(self) -> int:
        c, h, w = self.sample_shape
        return c * h * w

    def paths(self) -> list[Path]:
        return [DATA_ROOT / f for f in self.files]

    def run_dir(self, n: int, width: int = DEFAULT_WIDTH,
                batch_size: int | None = None, seed: int = DEFAULT_SEED) -> Path:
        """Directory holding checkpoints and caches for one training run."""
        batch_size = DEFAULT_BATCH_SIZE if batch_size is None else batch_size
        name = f"{self.key}_N{n}_W{width}_B{batch_size}"
        if seed != DEFAULT_SEED:
            name += f"_seed{seed}"
        return RUN_ROOT / self.key / name

    def index_file(self) -> Path:
        """Master shuffle shared by every ``N`` of this dataset."""
        return RUN_ROOT / self.key / "shared_indices.npz"


SIONNA_3P5GHZ = DatasetSpec(
    key="sionna_3p5ghz",
    label="Sionna RT, 3.5 GHz, LoS + NLoS",
    files=(
        "Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz",
        "Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz",
    ),
    sample_shape=(2, 4, 32),
    subcarrier=128,
    paired=True,
    sizes=(100, 200, 500, 1_000, 2_000, 4_000),
)

SIONNA_28GHZ = DatasetSpec(
    key="sionna_28ghz",
    label="Sionna RT, 28 GHz, LoS only",
    files=(
        "28GHz_Channel_UE_positions_full_LoS_iso_115_8759_"
        "51.546036612508296_-0.17853666925844522.npz",
    ),
    sample_shape=(2, 4, 32),
    subcarrier=0,
    sizes=(100, 200, 500, 1_000, 2_000),
)

DICHASUS_1P272GHZ = DatasetSpec(
    key="dichasus_1p272ghz",
    label="DICHASUS measurements, 1.272 GHz",
    files=("Measured_Dataset/dichasus_0c5x_downlink_1272MHz.npz",),
    sample_shape=(2, 4, 8),
    # Single-antenna UE: the 32 BS ports are re-indexed onto a physical 4 x 8
    # grid, so both sides reduce to uniform linear arrays.
    rx_array=(4, 1),
    tx_array=(8, 1),
    measured=True,
    sizes=(200, 500, 1_000, 2_000),
)

DATASETS: dict[str, DatasetSpec] = {
    d.key: d for d in (SIONNA_3P5GHZ, SIONNA_28GHZ, DICHASUS_1P272GHZ)
}

DEFAULT_DATASET = SIONNA_3P5GHZ.key


def get_dataset(key: str) -> DatasetSpec:
    """Look a dataset up by key, with a helpful error message."""
    try:
        return DATASETS[key]
    except KeyError:
        raise SystemExit(
            f"Unknown dataset {key!r}. Available: {', '.join(sorted(DATASETS))}"
        ) from None


def batch_size_for(n: int, cap: int = 500) -> int:
    """Protocol used by the dataset-size sweep: ``B = min(N, 500)``.

    Below the cap a full-batch update is the natural lowest-noise choice; above
    it the batch is frozen so that every ``N`` is optimised at the same
    per-step gradient-noise scale.
    """
    return min(n, cap)


def tau_grid(max_tau: int = DEFAULT_MAX_TAU) -> list[int]:
    grid = [t for t in TAU_GRID if t <= max_tau]
    if max_tau not in grid:
        grid.append(max_tau)
    return sorted(grid)


def result_dir(*parts: Sequence[str]) -> Path:
    """Create and return ``RESULT_ROOT/<parts...>``."""
    path = RESULT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
