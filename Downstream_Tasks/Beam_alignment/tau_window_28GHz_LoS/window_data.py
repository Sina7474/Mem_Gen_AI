"""Shared data, provenance, CSV, and window helpers for the new experiment."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    from . import config as C
except ImportError:  # Direct execution from the source directory.
    import config as C


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _temp_path_for(path: Path, suffix: str | None = None) -> Path:
    C.assert_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=suffix or path.suffix, dir=path.parent
    )
    os.close(fd)
    return Path(name)


def atomic_write_json(path: Path, value: Any) -> None:
    tmp = _temp_path_for(path, ".json")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = _temp_path_for(path, ".npy")
    try:
        np.save(tmp, array)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = _temp_path_for(path, ".npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    tmp = _temp_path_for(path, ".csv")
    try:
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def upsert_csv_row(
    path: Path, fieldnames: Sequence[str], row: dict[str, Any], key: str
) -> None:
    rows = read_csv_rows(path) if path.exists() else []
    wanted = str(row[key])
    filtered = [existing for existing in rows if str(existing.get(key, "")) != wanted]
    filtered.append(row)
    filtered.sort(key=lambda item: str(item.get(key, "")))
    atomic_write_csv(path, fieldnames, filtered)


def as_int(value: Any) -> int:
    return int(float(value))


def as_float(value: Any) -> float:
    return float(value)


def available_checkpoint_taus(n: int) -> list[int]:
    ckpt_dir = C.ddim_run_dir(n) / "checkpoints"
    found: list[int] = []
    for path in ckpt_dir.glob("checkpoint_tau_*.pth"):
        match = re.fullmatch(r"checkpoint_tau_(\d+)\.pth", path.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(set(found))


def generated_source_path(n: int, tau: int) -> Path:
    canonical = C.canonical_generated_samples(n, tau)
    if canonical.exists():
        return canonical
    return C.fallback_metric_samples(n, tau)


def validate_generated_file(path: Path, minimum_count: int = 1) -> tuple[int, tuple[int, ...]]:
    if not path.exists():
        raise FileNotFoundError(f"Generated sample source is missing: {path}")
    with np.load(path) as payload:
        if "channels" not in payload.files:
            raise KeyError(f"Generated sample file has no 'channels' key: {path}")
        channels = payload["channels"]
        if channels.ndim != 4 or tuple(channels.shape[1:]) != (2, 4, 32):
            raise ValueError(f"Wrong generated shape in {path}: {channels.shape}")
        if len(channels) < minimum_count:
            raise ValueError(
                f"Only {len(channels)} samples in {path}; need at least {minimum_count}."
            )
        if channels.dtype != np.float32:
            raise TypeError(f"Expected float32 channels in {path}, got {channels.dtype}")
        if not np.isfinite(channels).all():
            raise ValueError(f"NaN or infinite generated values in {path}")
        return len(channels), tuple(channels.shape)


def load_fcd_source(sizes: Sequence[int]) -> dict[tuple[int, int], dict[str, str]]:
    if not C.FCD_SOURCE_CSV.exists():
        raise FileNotFoundError(f"Canonical 28 GHz FCD table is missing: {C.FCD_SOURCE_CSV}")
    wanted = {int(n) for n in sizes}
    result: dict[tuple[int, int], dict[str, str]] = {}
    for row in read_csv_rows(C.FCD_SOURCE_CSV):
        n = as_int(row["N"])
        if n not in wanted:
            continue
        tau = as_int(row["tau"])
        key = (n, tau)
        if key in result:
            raise RuntimeError(f"Duplicate canonical FCD row for N={n}, tau={tau}")
        result[key] = row
    for n in wanted:
        if not any(key[0] == n for key in result):
            raise ValueError(f"No canonical FCD rows found for N={n}")
    return result


def log_interpolated_crossing(
    x: Sequence[float], y: Sequence[float], target: float, descending: bool
) -> float:
    """Mirror the validated CSI/DDIM window crossing implementation."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if len(x_arr) != len(y_arr) or len(x_arr) < 2:
        return float("nan")
    if np.any(x_arr <= 0):
        raise ValueError("All tau values must be positive for log interpolation.")
    log_x = np.log(x_arr)
    for index in range(1, len(y_arr)):
        if descending and y_arr[index - 1] > target and y_arr[index] <= target:
            fraction = (y_arr[index - 1] - target) / (
                y_arr[index - 1] - y_arr[index]
            )
            return float(
                np.exp(log_x[index - 1] + fraction * (log_x[index] - log_x[index - 1]))
            )
        if (
            not descending
            and y_arr[index - 1] < target
            and y_arr[index] >= target
        ):
            fraction = (target - y_arr[index - 1]) / (
                y_arr[index] - y_arr[index - 1]
            )
            return float(
                np.exp(log_x[index - 1] + fraction * (log_x[index] - log_x[index - 1]))
            )
    return float("nan")


def classify_regime(tau: float, tau_gen: float, tau_mem: float) -> str:
    if tau < tau_gen:
        return "pre-window"
    if tau < tau_mem:
        return "in-window"
    return "post-window"


def _spread_bucket(bucket: Sequence[int], count: int) -> list[int]:
    values = sorted(set(int(v) for v in bucket))
    if len(values) <= count:
        return values
    indices = np.unique(np.rint(np.linspace(0, len(values) - 1, count)).astype(int))
    return [values[index] for index in indices]


def select_checkpoints(
    available: Sequence[int],
    tau_gen: float,
    tau_mem: float,
    n_pre: int = 2,
    n_inside: int = 3,
    n_post: int = 3,
) -> list[int]:
    positive = sorted(set(int(tau) for tau in available if int(tau) > 0))
    buckets = {
        "pre-window": [tau for tau in positive if tau < tau_gen],
        "in-window": [tau for tau in positive if tau_gen <= tau < tau_mem],
        "post-window": [tau for tau in positive if tau >= tau_mem],
    }
    requested = {"pre-window": n_pre, "in-window": n_inside, "post-window": n_post}
    chosen: list[int] = []
    for regime in ("pre-window", "in-window", "post-window"):
        bucket = buckets[regime]
        if not bucket:
            raise RuntimeError(
                f"No saved checkpoint belongs to {regime}; cannot cover all regimes."
            )
        if len(bucket) < requested[regime]:
            print(
                f"  [warning] {regime} has only {len(bucket)} saved checkpoint(s); "
                f"requested {requested[regime]}. Using every available point."
            )
        chosen.extend(_spread_bucket(bucket, requested[regime]))
    return sorted(set(chosen))


def load_selected_rows(sizes: Sequence[int] = C.REFERENCE_SIZES) -> list[dict[str, str]]:
    if not C.SELECTED_CSV.exists():
        raise FileNotFoundError(
            f"Selected-checkpoint table is missing: {C.SELECTED_CSV}. "
            "Run compute_window_metrics.py first."
        )
    wanted = {int(n) for n in sizes}
    rows = [row for row in read_csv_rows(C.SELECTED_CSV) if as_int(row["N"]) in wanted]
    seen: set[tuple[int, int]] = set()
    for row in rows:
        key = (as_int(row["N"]), as_int(row["tau"]))
        if key in seen:
            raise RuntimeError(f"Duplicate selected checkpoint row: {key}")
        seen.add(key)
    for n in wanted:
        regimes = {row["regime"] for row in rows if as_int(row["N"]) == n}
        required = {"pre-window", "in-window", "post-window"}
        if regimes != required:
            raise RuntimeError(f"N={n} selected regimes are {regimes}, expected {required}")
    return sorted(rows, key=lambda row: (as_int(row["N"]), as_int(row["tau"])))


@dataclass
class SharedData:
    H_all: np.ndarray
    coords: np.ndarray
    reference_indices: dict[int, np.ndarray]
    H_reference: dict[int, np.ndarray]
    test_indices: np.ndarray
    H_test: np.ndarray
    metadata: dict[str, Any]


def _persist_expected_array(path: Path, expected: np.ndarray, force: bool) -> None:
    if path.exists() and not force:
        existing = np.load(path)
        if not np.array_equal(existing, expected):
            raise RuntimeError(
                f"Existing isolated index file does not match deterministic split: {path}. "
                "Use --force only if you intentionally want to replace this new artifact."
            )
        return
    atomic_save_npy(path, expected)


def prepare_shared_data(
    *, force: bool = False, dry_run: bool = False, require_existing: bool = False
) -> SharedData:
    """Build and validate the exact DDIM references and one shared test split."""
    if str(C.SRC_DIR) not in sys.path:
        sys.path.insert(0, str(C.SRC_DIR))
    from data_utils_28GHz_LoS import (  # pylint: disable=import-outside-toplevel
        COORD_DECIMALS,
        coord_keys,
        load_sionna_raw_28ghz,
    )

    if not C.DATASET_FILE.exists():
        raise FileNotFoundError(f"Raw 28 GHz dataset is missing: {C.DATASET_FILE}")
    if require_existing:
        required = [C.TEST_INDICES_FILE, C.SPLIT_META_JSON]
        required.extend(C.reference_indices_file(n) for n in C.REFERENCE_SIZES)
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "The shared split has not been materialized. Run prepare_synthetic.py "
                f"first. Missing: {missing}"
            )

    H_all, coords = load_sionna_raw_28ghz(str(C.DATASET_FILE))
    keys = coord_keys(coords)
    reference_indices: dict[int, np.ndarray] = {}
    H_reference: dict[int, np.ndarray] = {}
    train_diffs: dict[str, float] = {}
    coord_diffs: dict[str, float] = {}

    for n in C.REFERENCE_SIZES:
        run = C.ddim_run_dir(n)
        index_file = run / "indices_28ghz_los.npy"
        train_file = run / "train.npy"
        train_coord_file = run / "train_coords.npy"
        for source in (index_file, train_file, train_coord_file):
            if not source.exists():
                raise FileNotFoundError(f"Required DDIM reference source is missing: {source}")

        master = np.load(index_file)
        selected = np.asarray(master[:n], dtype=np.int64)
        if len(selected) != n or len(np.unique(selected)) != n:
            raise RuntimeError(f"Invalid DDIM reference indices for N={n}")
        if selected.min() < 0 or selected.max() >= len(H_all):
            raise IndexError(f"DDIM reference index outside raw dataset for N={n}")

        H_ref = H_all[selected]
        train_ri = np.load(train_file)
        train_complex = (train_ri[:, 0] + 1j * train_ri[:, 1]).astype(np.complex64)
        if train_complex.shape != H_ref.shape:
            raise RuntimeError(
                f"DDIM/raw reference shape mismatch for N={n}: "
                f"{train_complex.shape} vs {H_ref.shape}"
            )
        max_diff = float(np.max(np.abs(train_complex - H_ref)))
        if max_diff > 1e-6:
            raise RuntimeError(
                f"Shared-reference violation for N={n}: max abs difference {max_diff:.3e}"
            )

        saved_coords = np.real(np.load(train_coord_file)).astype(np.float32)
        coord_diff = float(np.max(np.abs(saved_coords - coords[selected])))
        if coord_diff > 1e-6:
            raise RuntimeError(
                f"DDIM/raw coordinate mismatch for N={n}: max difference {coord_diff:.3e}"
            )

        reference_indices[n] = selected
        H_reference[n] = H_ref
        train_diffs[str(n)] = max_diff
        coord_diffs[str(n)] = coord_diff

    largest_n = max(C.REFERENCE_SIZES)
    largest_reference = reference_indices[largest_n]
    nested = all(
        np.array_equal(selected, largest_reference[: len(selected)])
        for selected in reference_indices.values()
    )
    if not nested:
        raise RuntimeError(
            "The DDIM reference sets are not nested prefixes of the largest set; "
            "shared-test union handling must be reviewed."
        )

    union_indices = np.unique(np.concatenate(list(reference_indices.values())))
    reference_keys = set(keys[union_indices].tolist())
    unseen = np.asarray(
        [index for index, key in enumerate(keys) if key not in reference_keys],
        dtype=np.int64,
    )
    if len(unseen) < C.N_TEST:
        raise ValueError(
            f"Only {len(unseen)} DDIM-unseen users are available; need {C.N_TEST}."
        )
    rng = np.random.default_rng(C.SPLIT_SEED)
    test_indices = np.sort(rng.permutation(unseen)[: C.N_TEST]).astype(np.int64)

    index_overlaps = {
        str(n): int(np.intersect1d(test_indices, selected).size)
        for n, selected in reference_indices.items()
    }
    coordinate_overlaps = {
        str(n): len(set(keys[test_indices].tolist()) & set(keys[selected].tolist()))
        for n, selected in reference_indices.items()
    }
    if any(index_overlaps.values()) or any(coordinate_overlaps.values()):
        raise RuntimeError(
            f"Leakage detected: index overlaps={index_overlaps}, "
            f"coordinate overlaps={coordinate_overlaps}"
        )

    metadata: dict[str, Any] = {
        "dataset_file": C.project_relative(C.DATASET_FILE),
        "scene_users": int(len(H_all)),
        "split_seed": C.SPLIT_SEED,
        "n_test": int(len(test_indices)),
        "coordinate_decimals": int(COORD_DECIMALS),
        "reference_counts": {str(n): int(len(v)) for n, v in reference_indices.items()},
        "reference_prefix_nested": bool(nested),
        "reference_union_count": int(len(union_indices)),
        "eligible_unseen_count": int(len(unseen)),
        "test_reference_index_overlaps": index_overlaps,
        "test_reference_coordinate_overlaps": coordinate_overlaps,
        "reference_train_max_abs_diff": train_diffs,
        "reference_coordinate_max_abs_diff": coord_diffs,
    }

    if not dry_run:
        C.ensure_output_directories()
        for n, selected in reference_indices.items():
            _persist_expected_array(C.reference_indices_file(n), selected, force)
        _persist_expected_array(C.TEST_INDICES_FILE, test_indices, force)
        metadata["reference_index_sha256"] = {
            str(n): sha256_file(C.reference_indices_file(n)) for n in C.REFERENCE_SIZES
        }
        metadata["test_indices_sha256"] = sha256_file(C.TEST_INDICES_FILE)
        if C.SPLIT_META_JSON.exists() and not force:
            existing = json.loads(C.SPLIT_META_JSON.read_text())
            # Adding a nested reference size (for example N=500 between the
            # existing N=100 and N=1000 prefixes) must retain the exact saved
            # test set. Validate immutable split fields and all previously
            # recorded reference counts, then extend the metadata additively.
            for field in ("split_seed", "n_test", "test_indices_sha256"):
                if existing.get(field) != metadata.get(field):
                    raise RuntimeError(
                        f"Existing split metadata field '{field}' does not match validation."
                    )
            for n_string, count in existing.get("reference_counts", {}).items():
                if metadata["reference_counts"].get(n_string) != count:
                    raise RuntimeError(
                        f"Existing reference count for N={n_string} changed unexpectedly."
                    )
            if existing != metadata:
                atomic_write_json(C.SPLIT_META_JSON, metadata)
        else:
            atomic_write_json(C.SPLIT_META_JSON, metadata)

    return SharedData(
        H_all=H_all,
        coords=coords,
        reference_indices=reference_indices,
        H_reference=H_reference,
        test_indices=test_indices,
        H_test=H_all[test_indices],
        metadata=metadata,
    )
