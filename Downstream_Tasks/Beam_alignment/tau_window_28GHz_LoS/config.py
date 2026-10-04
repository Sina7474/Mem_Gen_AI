"""Single source of truth for the 28 GHz LoS tau-window experiment.

All paths written by this package are descendants of ``OUTPUT_ROOT``. Existing
beam-alignment, DDIM, and DDIM-evaluation trees are read-only inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


SOURCE_DIR = Path(__file__).resolve().parent
BEAM_DIR = SOURCE_DIR.parent
PROJECT_ROOT = BEAM_DIR.parent.parent
SRC_DIR = BEAM_DIR / "src"

DATASET_FILE = (
    PROJECT_ROOT
    / "dataset"
    / (
        "28GHz_Channel_UE_positions_full_LoS_iso_115_8759_"
        "51.546036612508296_-0.17853666925844522.npz"
    )
)
DDIM_LOGS = PROJECT_ROOT / "Code" / "DDIM_FMM" / "logs_ema_28GHz_LoS"
DDIM_GENERATOR_SCRIPT = (
    PROJECT_ROOT / "Code" / "DDIM_FMM" / "generate_downstream_28GHz_LoS.py"
)
FCD_SOURCE_CSV = (
    PROJECT_ROOT
    / "DDIM_Evaluation"
    / "dataset_size_effect_28GHz_LoS"
    / "results"
    / "dsize_fcd_fmem_28GHz_LoS.csv"
)
FCD_MODULE = PROJECT_ROOT / "DDIM_Evaluation" / "compute_fcd_vs_tau.py"
FMEM_MODULE = PROJECT_ROOT / "DDIM_Evaluation" / "compute_fmem_vs_tau.py"

OUTPUT_ROOT = BEAM_DIR / "28GHz_LoS" / "tau_window_kappa1_4_fmem20"
DATA_DIR = OUTPUT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
METRICS_DIR = OUTPUT_ROOT / "metrics"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
LOG_DIR = OUTPUT_ROOT / "logs"
FIGURE_DIR = OUTPUT_ROOT / "figures"

FMEM_CSV = METRICS_DIR / "fmem_kappa1_4.csv"
COMBINED_CSV = METRICS_DIR / "fcd_fmem_kappa1_4.csv"
WINDOW_CSV = METRICS_DIR / "window_boundaries.csv"
SELECTED_CSV = METRICS_DIR / "selected_checkpoints.csv"
RUN_RESULTS_CSV = METRICS_DIR / "run_results.csv"
OPTIMUM_CSV = METRICS_DIR / "optimum_summary.csv"
BASELINES_JSON = METRICS_DIR / "baselines.json"
PLOT_OVERRIDE_CSV = METRICS_DIR / "manual_plot_overrides.csv"
EFFECTIVE_PLOT_POINTS_CSV = METRICS_DIR / "effective_plot_points.csv"
SPLIT_META_JSON = DATA_DIR / "split_meta.json"
TEST_INDICES_FILE = DATA_DIR / "test_indices.npy"
MANIFEST_JSON = OUTPUT_ROOT / "manifest.json"

SCENE = "28GHz_LoS"
WIDTH = 256
REFERENCE_SIZES = (100, 500, 1000)
N_PROBES = (2, 4)
K_TOTAL = 5000
N_TEST = 5000

KAPPA = 1.0 / 4.0
FMEM_THRESHOLD = 0.20
REL_TOL = 0.50
N_GENERATED_FOR_FMEM = 5000
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 42
NN_BATCH_SIZE = 512

GENERATOR_SEED = 0
SYNTHETIC_SUBSET_SEED = 0
SPLIT_SEED = 42
TRAIN_SEED = 42
EVAL_SEED = 0

EPOCHS = 1000
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

TX_POWER_DBM = 20.0
BANDWIDTH_MHZ = 100.0
NOISE_PSD_DB = -161.0
MEASUREMENT_GAIN = 16.0

CHECKPOINT_GRID = (
    0,
    100,
    150,
    200,
    300,
    500,
    700,
    1000,
    1500,
    2000,
    3000,
    5000,
    7000,
    10000,
    15000,
    20000,
    30000,
    50000,
    70000,
    100000,
    150000,
    200000,
)


def ddim_run_dir(n: int) -> Path:
    """Return the existing width-256 28 GHz EMA run directory for ``n``."""
    batch_size = min(int(n), 500)
    return DDIM_LOGS / f"DDIM_tau_ema_28GHz_LoS_{n}_bs{batch_size}_incremental"


def ddim_checkpoint(n: int, tau: int) -> Path:
    return ddim_run_dir(n) / "checkpoints" / f"checkpoint_tau_{tau}.pth"


def canonical_generated_samples(n: int, tau: int) -> Path:
    return ddim_run_dir(n) / "generated_ema" / f"gen_ema_tau{tau}_seed0.npz"


def fallback_metric_samples(n: int, tau: int) -> Path:
    return SYNTHETIC_DIR / f"fmem_source_N{n}_tau{tau}_ema_seed0_count5000.npz"


def reference_indices_file(n: int) -> Path:
    return DATA_DIR / f"reference_indices_N{n}.npy"


def downstream_synthetic_file(n: int, tau: int) -> Path:
    count = K_TOTAL - int(n)
    return SYNTHETIC_DIR / f"N{n}_tau{tau}_ema_seed0_count{count}.npz"


def downstream_synthetic_meta_file(n: int, tau: int) -> Path:
    return downstream_synthetic_file(n, tau).with_suffix(".json")


def output_directories() -> tuple[Path, ...]:
    return (
        OUTPUT_ROOT,
        DATA_DIR,
        SYNTHETIC_DIR,
        METRICS_DIR,
        CHECKPOINT_DIR,
        LOG_DIR,
        FIGURE_DIR,
    )


def ensure_output_directories() -> None:
    for directory in output_directories():
        assert_output_path(directory)
        directory.mkdir(parents=True, exist_ok=True)


def assert_output_path(path: Path) -> None:
    """Refuse writes outside the new experiment output root."""
    resolved = Path(path).resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing write outside isolated output root: {resolved}")


def project_relative(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def resolve_project_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def fixed_configuration() -> dict[str, Any]:
    return {
        "experiment_name": "beam_alignment_tau_window_28GHz_LoS",
        "scene": SCENE,
        "width": WIDTH,
        "reference_sizes": list(REFERENCE_SIZES),
        "n_probes": list(N_PROBES),
        "k_total": K_TOTAL,
        "n_test": N_TEST,
        "kappa": KAPPA,
        "fmem_threshold": FMEM_THRESHOLD,
        "rel_tol": REL_TOL,
        "n_generated_for_fmem": N_GENERATED_FOR_FMEM,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "nn_batch_size": NN_BATCH_SIZE,
        "generator_seed": GENERATOR_SEED,
        "synthetic_subset_seed": SYNTHETIC_SUBSET_SEED,
        "split_seed": SPLIT_SEED,
        "train_seed": TRAIN_SEED,
        "eval_seed": EVAL_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "tx_power_dBm": TX_POWER_DBM,
        "bandwidth_MHz": BANDWIDTH_MHZ,
        "noise_PSD_dB": NOISE_PSD_DB,
        "measurement_gain": MEASUREMENT_GAIN,
        "checkpoint_grid": list(CHECKPOINT_GRID),
        "dataset_file": project_relative(DATASET_FILE),
        "ddim_logs": project_relative(DDIM_LOGS),
        "output_root": project_relative(OUTPUT_ROOT),
    }
