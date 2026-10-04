#!/usr/bin/env python3
"""Train and evaluate the isolated 28 GHz LoS SNR-versus-tau experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from . import config as C
    from . import window_data as W
except ImportError:
    import config as C
    import window_data as W


RESULT_FIELDS = [
    "configuration_id",
    "experiment_type",
    "N",
    "N_probe",
    "tau",
    "regime",
    "W",
    "kappa",
    "fmem_threshold",
    "tau_gen",
    "tau_mem",
    "FCD",
    "f_mem",
    "num_real_train",
    "num_synthetic_train",
    "num_total_train",
    "frob_norm",
    "train_seed",
    "generator_seed",
    "eval_seed",
    "epochs",
    "batch_size",
    "learning_rate",
    "average_snr_db",
    "average_rate_bps_hz",
    "reference_indices_file",
    "test_indices_file",
    "synthetic_data_file",
    "model_checkpoint",
    "run_log",
    "configuration_fingerprint",
    "checkpoint_sha256",
    "wall_time_seconds",
    "status",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def _atomic_torch_save(path: Path, value: Any) -> None:
    C.assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pt", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=C.PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else "unavailable",
        "dirty_worktree": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status_entry_count": len(status.stdout.splitlines()) if status.returncode == 0 else None,
    }


def _write_manifest(
    selected: list[dict[str, str]],
    sizes: list[int],
    n_probes: list[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    synthetic_files = sorted(
        {
            C.downstream_synthetic_file(W.as_int(row["N"]), W.as_int(row["tau"]))
            for row in selected
        }
    )
    inputs = [C.FMEM_CSV, C.COMBINED_CSV, C.WINDOW_CSV, C.SELECTED_CSV]
    inputs.extend(C.reference_indices_file(n) for n in sizes)
    inputs.append(C.TEST_INDICES_FILE)
    inputs.extend(synthetic_files)
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot build manifest; missing input(s): {missing}")

    now = _utc_now()
    configuration = {
        **C.fixed_configuration(),
        "active_sizes": sizes,
        "active_n_probes": n_probes,
        "actual_epochs": epochs,
        "actual_batch_size": batch_size,
        "actual_learning_rate": learning_rate,
    }
    environment = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or "CPU"
            ),
    }
    current_selected = [
        {
            "N": W.as_int(row["N"]),
            "tau": W.as_int(row["tau"]),
            "regime": row["regime"],
            "checkpoint_file": row["checkpoint_file"],
        }
        for row in selected
    ]
    current_hashes = {C.project_relative(path): W.sha256_file(path) for path in inputs}
    invocation_id = W.canonical_json_hash(
        {"configuration": configuration, "selected_checkpoints": current_selected}
    )
    invocation = {
        "invocation_id": invocation_id,
        "updated_utc": now,
        "configuration": configuration,
        "environment": environment,
        "selected_checkpoints": current_selected,
    }

    previous: dict[str, Any] = {}
    if C.MANIFEST_JSON.exists():
        previous = json.loads(C.MANIFEST_JSON.read_text())
    history = list(previous.get("run_invocations", []))
    if previous and not history and previous.get("configuration"):
        legacy_selected = previous.get("selected_checkpoints", [])
        legacy_id = W.canonical_json_hash(
            {
                "configuration": previous["configuration"],
                "selected_checkpoints": legacy_selected,
            }
        )
        history.append(
            {
                "invocation_id": legacy_id,
                "updated_utc": previous.get("created_utc", "unknown"),
                "configuration": previous["configuration"],
                "environment": previous.get("environment", {}),
                "selected_checkpoints": legacy_selected,
            }
        )
    history = [item for item in history if item.get("invocation_id") != invocation_id]
    history.append(invocation)

    cumulative_selected: dict[tuple[int, int], dict[str, Any]] = {}
    for item in previous.get("selected_checkpoints", []) + current_selected:
        cumulative_selected[(int(item["N"]), int(item["tau"]))] = item
    cumulative_hashes = dict(previous.get("input_sha256", {}))
    cumulative_hashes.update(current_hashes)

    manifest = {
        "created_utc": previous.get("created_utc", now),
        "updated_utc": now,
        "configuration": configuration,
        "environment": environment,
        "git": _git_metadata(),
        "selected_checkpoints": [
            cumulative_selected[key] for key in sorted(cumulative_selected)
        ],
        "input_sha256": cumulative_hashes,
        "run_invocations": history,
        "read_only_input_roots": [
            C.project_relative(C.DDIM_LOGS),
            C.project_relative(C.FCD_SOURCE_CSV.parent),
            C.project_relative(C.SRC_DIR),
        ],
        "isolated_output_root": C.project_relative(C.OUTPUT_ROOT),
    }
    W.atomic_write_json(C.MANIFEST_JSON, manifest)


def _base_result_row(
    *,
    configuration_id: str,
    experiment_type: str,
    n: int,
    n_probe: int,
    tau: int | None,
    regime: str,
    tau_gen: float,
    tau_mem: float,
    fcd: float | str,
    f_mem: float | str,
    n_real: int,
    n_synthetic: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    reference_file: Path,
    synthetic_file: Path | None,
    checkpoint: Path | None,
    run_log: Path,
    fingerprint: str,
) -> dict[str, Any]:
    return {
        "configuration_id": configuration_id,
        "experiment_type": experiment_type,
        "N": n,
        "N_probe": n_probe,
        "tau": "" if tau is None else tau,
        "regime": regime,
        "W": C.WIDTH,
        "kappa": C.KAPPA,
        "fmem_threshold": C.FMEM_THRESHOLD,
        "tau_gen": tau_gen,
        "tau_mem": tau_mem,
        "FCD": fcd,
        "f_mem": f_mem,
        "num_real_train": n_real,
        "num_synthetic_train": n_synthetic,
        "num_total_train": n_real + n_synthetic,
        "frob_norm": True if experiment_type in {"real_only", "augmented"} else "",
        "train_seed": C.TRAIN_SEED if experiment_type in {"real_only", "augmented"} else "",
        "generator_seed": C.GENERATOR_SEED if n_synthetic else "",
        "eval_seed": C.EVAL_SEED if experiment_type in {"real_only", "augmented"} else "",
        "epochs": epochs if experiment_type in {"real_only", "augmented"} else "",
        "batch_size": batch_size if experiment_type in {"real_only", "augmented"} else "",
        "learning_rate": learning_rate if experiment_type in {"real_only", "augmented"} else "",
        "average_snr_db": "",
        "average_rate_bps_hz": "",
        "reference_indices_file": C.project_relative(reference_file),
        "test_indices_file": C.project_relative(C.TEST_INDICES_FILE),
        "synthetic_data_file": C.project_relative(synthetic_file) if synthetic_file else "",
        "model_checkpoint": C.project_relative(checkpoint) if checkpoint else "",
        "run_log": C.project_relative(run_log),
        "configuration_fingerprint": fingerprint,
        "checkpoint_sha256": "",
        "wall_time_seconds": "",
        "status": "pending",
    }


def _configuration_payload(
    *,
    experiment_type: str,
    n: int,
    n_probe: int,
    tau: int | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    reference_file: Path,
    synthetic_file: Path | None,
) -> dict[str, Any]:
    return {
        "experiment_type": experiment_type,
        "N": n,
        "N_probe": n_probe,
        "tau": tau,
        "W": C.WIDTH,
        "K_total": C.K_TOTAL,
        "kappa": C.KAPPA,
        "fmem_threshold": C.FMEM_THRESHOLD,
        "frob_norm": True,
        "train_seed": C.TRAIN_SEED,
        "generator_seed": C.GENERATOR_SEED if synthetic_file else None,
        "eval_seed": C.EVAL_SEED,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "system": {
            "tx_power_dBm": C.TX_POWER_DBM,
            "bandwidth_MHz": C.BANDWIDTH_MHZ,
            "noise_PSD_dB": C.NOISE_PSD_DB,
            "measurement_gain": C.MEASUREMENT_GAIN,
        },
        "reference_indices_sha256": W.sha256_file(reference_file),
        "test_indices_sha256": W.sha256_file(C.TEST_INDICES_FILE),
        "synthetic_sha256": W.sha256_file(synthetic_file) if synthetic_file else None,
    }


def _load_cached_run(
    log_path: Path, checkpoint_path: Path, fingerprint: str, force: bool
) -> dict[str, Any] | None:
    if not log_path.exists() and not checkpoint_path.exists():
        return None
    if force:
        return None
    if not log_path.exists():
        raise RuntimeError(
            f"A checkpoint exists without provenance log: {checkpoint_path}. "
            "Use --force to replace this unverifiable isolated artifact."
        )
    record = json.loads(log_path.read_text())
    if record.get("configuration_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Cached configuration does not match current settings: {log_path}. "
            "Use --force to replace this isolated run."
        )
    if record.get("status") != "complete":
        print(f"  [resume] restarting incomplete isolated run: {log_path.name}")
        return None
    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Completed run log has no checkpoint: {checkpoint_path}. Use --force."
        )
    if record.get("checkpoint_sha256") != W.sha256_file(checkpoint_path):
        raise RuntimeError(f"Cached checkpoint checksum mismatch: {checkpoint_path}")
    if "result_row" not in record:
        raise RuntimeError(f"Cached run log has no result row: {log_path}")
    return record


def _train_and_evaluate(
    *,
    configuration_id: str,
    experiment_type: str,
    n: int,
    n_probe: int,
    tau: int | None,
    regime: str,
    tau_gen: float,
    tau_mem: float,
    fcd: float | str,
    f_mem: float | str,
    H_train: np.ndarray,
    H_test: np.ndarray,
    n_real: int,
    n_synthetic: int,
    reference_file: Path,
    synthetic_file: Path | None,
    sysd: dict[str, float],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    force: bool,
    train_one,
    prepare_training_channels,
    eval_snr,
) -> dict[str, Any]:
    checkpoint_path = C.CHECKPOINT_DIR / f"{configuration_id}.pt"
    log_path = C.LOG_DIR / f"{configuration_id}.json"
    payload = _configuration_payload(
        experiment_type=experiment_type,
        n=n,
        n_probe=n_probe,
        tau=tau,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        reference_file=reference_file,
        synthetic_file=synthetic_file,
    )
    fingerprint = W.canonical_json_hash(payload)
    row = _base_result_row(
        configuration_id=configuration_id,
        experiment_type=experiment_type,
        n=n,
        n_probe=n_probe,
        tau=tau,
        regime=regime,
        tau_gen=tau_gen,
        tau_mem=tau_mem,
        fcd=fcd,
        f_mem=f_mem,
        n_real=n_real,
        n_synthetic=n_synthetic,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        reference_file=reference_file,
        synthetic_file=synthetic_file,
        checkpoint=checkpoint_path,
        run_log=log_path,
        fingerprint=fingerprint,
    )

    cached = _load_cached_run(log_path, checkpoint_path, fingerprint, force)
    if cached is not None:
        cached_row = cached["result_row"]
        W.upsert_csv_row(C.RUN_RESULTS_CSV, RESULT_FIELDS, cached_row, "configuration_id")
        print(
            f"  [cached] {configuration_id}: "
            f"SNR={float(cached_row['average_snr_db']):.3f} dB"
        )
        return cached_row

    start = time.monotonic()
    started = _utc_now()
    W.atomic_write_json(
        log_path,
        {
            "configuration": payload,
            "configuration_fingerprint": fingerprint,
            "status": "running",
            "started_utc": started,
        },
    )
    row["status"] = "running"
    W.upsert_csv_row(C.RUN_RESULTS_CSV, RESULT_FIELDS, row, "configuration_id")
    try:
        _seed_everything(C.TRAIN_SEED)
        prepared, frob_used = prepare_training_channels(
            H_train, seed=C.TRAIN_SEED, frob_norm=True
        )
        if not frob_used or len(prepared) != n_real + n_synthetic:
            raise RuntimeError("Training preprocessing violated count/normalization invariants.")
        state, norm_factor = train_one(
            prepared,
            n_probe,
            sysd,
            nepoch=epochs,
            batch_size=batch_size,
            lr=learning_rate,
            seed=C.TRAIN_SEED,
            device=device,
        )
        _atomic_torch_save(checkpoint_path, state)

        # eval_snr seeds NumPy internally; seed Torch here because probing noise is Torch-based.
        _seed_everything(C.EVAL_SEED)
        snr_db, rate = eval_snr(
            state,
            n_probe,
            H_test,
            sysd,
            frob_norm=True,
            eval_seed=C.EVAL_SEED,
        )
        elapsed = time.monotonic() - start
        row.update(
            {
                "average_snr_db": float(snr_db),
                "average_rate_bps_hz": float(rate),
                "checkpoint_sha256": W.sha256_file(checkpoint_path),
                "wall_time_seconds": elapsed,
                "status": "complete",
            }
        )
        record = {
            "configuration": payload,
            "configuration_fingerprint": fingerprint,
            "status": "complete",
            "started_utc": started,
            "completed_utc": _utc_now(),
            "wall_time_seconds": elapsed,
            "training_norm_factor": float(norm_factor),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "result_row": row,
        }
        W.atomic_write_json(log_path, record)
        W.upsert_csv_row(C.RUN_RESULTS_CSV, RESULT_FIELDS, row, "configuration_id")
        print(f"  [complete] {configuration_id}: SNR={snr_db:.3f} dB, rate={rate:.4f}")
        return row
    except Exception as error:
        row.update(
            {
                "status": "failed",
                "wall_time_seconds": time.monotonic() - start,
            }
        )
        W.atomic_write_json(
            log_path,
            {
                "configuration": payload,
                "configuration_fingerprint": fingerprint,
                "status": "failed",
                "started_utc": started,
                "failed_utc": _utc_now(),
                "wall_time_seconds": time.monotonic() - start,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        W.upsert_csv_row(C.RUN_RESULTS_CSV, RESULT_FIELDS, row, "configuration_id")
        raise


def _record_baseline_rows(
    baselines: dict[str, float],
    sizes: list[int],
    n_probes: list[int],
    windows: dict[int, dict[str, str]],
) -> None:
    for n in sizes:
        reference = C.reference_indices_file(n)
        tau_gen = W.as_float(windows[n]["tau_gen"])
        tau_mem = W.as_float(windows[n]["tau_mem"])
        for n_probe in n_probes:
            for experiment_type, key in (("MRT_MRC", "MRT_MRC"), ("genie_DFT", "genie_DFT")):
                configuration_id = f"{experiment_type}_N{n}_nprobe{n_probe}"
                log_path = C.LOG_DIR / f"{configuration_id}.json"
                payload = {
                    "experiment_type": experiment_type,
                    "N": n,
                    "N_probe": n_probe,
                    "test_indices_sha256": W.sha256_file(C.TEST_INDICES_FILE),
                    "system": {
                        "tx_power_dBm": C.TX_POWER_DBM,
                        "bandwidth_MHz": C.BANDWIDTH_MHZ,
                        "noise_PSD_dB": C.NOISE_PSD_DB,
                        "measurement_gain": C.MEASUREMENT_GAIN,
                    },
                }
                fingerprint = W.canonical_json_hash(payload)
                row = _base_result_row(
                    configuration_id=configuration_id,
                    experiment_type=experiment_type,
                    n=n,
                    n_probe=n_probe,
                    tau=None,
                    regime="analytical",
                    tau_gen=tau_gen,
                    tau_mem=tau_mem,
                    fcd="",
                    f_mem="",
                    n_real=0,
                    n_synthetic=0,
                    epochs=0,
                    batch_size=0,
                    learning_rate=0.0,
                    reference_file=reference,
                    synthetic_file=None,
                    checkpoint=None,
                    run_log=log_path,
                    fingerprint=fingerprint,
                )
                row.update(
                    {
                        "average_snr_db": float(baselines[key]),
                        "status": "complete",
                        "wall_time_seconds": 0.0,
                    }
                )
                W.atomic_write_json(
                    log_path,
                    {
                        "configuration": payload,
                        "configuration_fingerprint": fingerprint,
                        "status": "complete",
                        "result_row": row,
                    },
                )
                W.upsert_csv_row(C.RUN_RESULTS_CSV, RESULT_FIELDS, row, "configuration_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(C.REFERENCE_SIZES))
    parser.add_argument("--n_probes", type=int, nargs="+", default=list(C.N_PROBES))
    parser.add_argument("--taus", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=C.LEARNING_RATE)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<id>")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain matching configurations only inside the isolated output root.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate all inputs only")
    args = parser.parse_args()

    sizes = sorted(set(args.sizes))
    n_probes = sorted(set(args.n_probes))
    if set(sizes) - set(C.REFERENCE_SIZES):
        parser.error(f"Supported sizes are {list(C.REFERENCE_SIZES)}")
    if set(n_probes) - set(C.N_PROBES):
        parser.error(f"Supported N_probe values are {list(C.N_PROBES)}")
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, and learning rate must be positive.")

    selected = W.load_selected_rows(sizes)
    if args.taus is not None:
        requested_taus = set(args.taus)
        selected = [row for row in selected if W.as_int(row["tau"]) in requested_taus]
        if not selected:
            parser.error("None of --taus belongs to the selected checkpoint table.")
    windows_rows = W.read_csv_rows(C.WINDOW_CSV)
    windows = {W.as_int(row["N"]): row for row in windows_rows if W.as_int(row["N"]) in sizes}
    if set(windows) != set(sizes):
        raise RuntimeError(f"Window table does not cover requested sizes: {sizes}")

    # Require stage 2 outputs and revalidate reference identity plus all overlap checks.
    shared = W.prepare_shared_data(dry_run=args.dry_run, require_existing=True)
    synthetic_by_key: dict[tuple[int, int], Path] = {}
    for row in selected:
        n, tau = W.as_int(row["N"]), W.as_int(row["tau"])
        path = C.downstream_synthetic_file(n, tau)
        W.validate_generated_file(path, C.K_TOTAL - n)
        with np.load(path) as payload:
            if len(payload["channels"]) != C.K_TOTAL - n:
                raise RuntimeError(f"Synthetic count mismatch: {path}")
        synthetic_by_key[(n, tau)] = path

    device = _device_from_arg(args.device)
    total_augmented = len(selected) * len(n_probes)
    total_real = len(sizes) * len(n_probes)
    print("=" * 78)
    print("28 GHz LoS beam alignment: average SNR versus DDIM tau")
    print("=" * 78)
    print(f"Device          : {device}")
    print(f"Sizes           : {sizes}")
    print(f"N_probe         : {n_probes}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Learning rate   : {args.learning_rate}")
    print(f"New train runs  : {total_real} real-only + {total_augmented} augmented")
    print(f"Shared test size: {len(shared.H_test)}")
    print(f"Output root     : {C.OUTPUT_ROOT}")

    if args.dry_run:
        print("\nPreflight passed; no models or result files were written.")
        return

    C.ensure_output_directories()
    _write_manifest(
        selected,
        sizes,
        n_probes,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        device,
    )

    if str(C.SRC_DIR) not in sys.path:
        sys.path.insert(0, str(C.SRC_DIR))
    from beam_align_core import (  # pylint: disable=import-outside-toplevel
        compute_baselines,
        eval_snr,
        make_system,
        prepare_training_channels,
        train_one,
    )
    from data_utils_28GHz_LoS import (  # pylint: disable=import-outside-toplevel
        beamspace_to_antenna,
    )

    sysd = make_system(
        tx_power_dBm=C.TX_POWER_DBM,
        BW_MHz=C.BANDWIDTH_MHZ,
        noise_PSD_dB=C.NOISE_PSD_DB,
        measurement_gain=C.MEASUREMENT_GAIN,
    )
    baselines = compute_baselines(shared.H_test, sysd)
    baseline_record = {
        "created_utc": _utc_now(),
        "test_indices_file": C.project_relative(C.TEST_INDICES_FILE),
        "test_indices_sha256": W.sha256_file(C.TEST_INDICES_FILE),
        "n_test": len(shared.H_test),
        "system": sysd,
        "average_snr_db": baselines,
    }
    W.atomic_write_json(C.BASELINES_JSON, baseline_record)
    _record_baseline_rows(baselines, sizes, n_probes, windows)
    print(
        f"Baselines: MRT+MRC={baselines['MRT_MRC']:.3f} dB, "
        f"Genie DFT={baselines['genie_DFT']:.3f} dB"
    )

    selected_by_n = {
        n: sorted(
            [row for row in selected if W.as_int(row["N"]) == n],
            key=lambda row: W.as_int(row["tau"]),
        )
        for n in sizes
    }

    for n in sizes:
        H_real = shared.H_reference[n]
        reference_file = C.reference_indices_file(n)
        window = windows[n]
        tau_gen = W.as_float(window["tau_gen"])
        tau_mem = W.as_float(window["tau_mem"])
        if len(H_real) != n:
            raise RuntimeError(f"N={n} real reference count is {len(H_real)}")

        for n_probe in n_probes:
            print(f"\n--- N={n}, N_probe={n_probe}: real-only ---")
            _train_and_evaluate(
                configuration_id=f"realonly_N{n}_nprobe{n_probe}_seed{C.TRAIN_SEED}",
                experiment_type="real_only",
                n=n,
                n_probe=n_probe,
                tau=None,
                regime="reference",
                tau_gen=tau_gen,
                tau_mem=tau_mem,
                fcd="",
                f_mem="",
                H_train=H_real,
                H_test=shared.H_test,
                n_real=n,
                n_synthetic=0,
                reference_file=reference_file,
                synthetic_file=None,
                sysd=sysd,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=device,
                force=args.force,
                train_one=train_one,
                prepare_training_channels=prepare_training_channels,
                eval_snr=eval_snr,
            )

            for selected_row in selected_by_n[n]:
                tau = W.as_int(selected_row["tau"])
                synthetic_file = synthetic_by_key[(n, tau)]
                with np.load(synthetic_file) as payload:
                    generated = payload["channels"]
                    H_beam = (generated[:, 0] + 1j * generated[:, 1]).astype(np.complex64)
                H_synthetic = beamspace_to_antenna(H_beam).astype(np.complex64)
                H_augmented = np.concatenate([H_real, H_synthetic], axis=0)
                if len(H_synthetic) != C.K_TOTAL - n or len(H_augmented) != C.K_TOTAL:
                    raise RuntimeError(
                        f"N={n}, tau={tau} violates fixed 5,000-sample budget."
                    )
                print(f"\n--- N={n}, N_probe={n_probe}, tau={tau} ({selected_row['regime']}) ---")
                _train_and_evaluate(
                    configuration_id=(
                        f"aug_N{n}_nprobe{n_probe}_tau{tau}_seed{C.TRAIN_SEED}"
                    ),
                    experiment_type="augmented",
                    n=n,
                    n_probe=n_probe,
                    tau=tau,
                    regime=selected_row["regime"],
                    tau_gen=W.as_float(selected_row["tau_gen"]),
                    tau_mem=W.as_float(selected_row["tau_mem"]),
                    fcd=W.as_float(selected_row["FCD"]),
                    f_mem=W.as_float(selected_row["f_mem"]),
                    H_train=H_augmented,
                    H_test=shared.H_test,
                    n_real=n,
                    n_synthetic=len(H_synthetic),
                    reference_file=reference_file,
                    synthetic_file=synthetic_file,
                    sysd=sysd,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    device=device,
                    force=args.force,
                    train_one=train_one,
                    prepare_training_channels=prepare_training_channels,
                    eval_snr=eval_snr,
                )
                del generated, H_beam, H_synthetic, H_augmented

    print("\nDone.")
    print(f"  Results CSV : {C.RUN_RESULTS_CSV}")
    print(f"  Checkpoints : {C.CHECKPOINT_DIR}")
    print(f"  Logs        : {C.LOG_DIR}")
    print(f"  Manifest    : {C.MANIFEST_JSON}")


if __name__ == "__main__":
    main()
