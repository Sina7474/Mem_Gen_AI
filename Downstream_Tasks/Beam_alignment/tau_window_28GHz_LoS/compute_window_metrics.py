#!/usr/bin/env python3
"""Compute kappa=1/4 f_mem, derive 20% windows, and select checkpoints.

This script reads the existing DDIM checkpoints/cached samples and canonical FCD
table, but writes only under the isolated tau-window output directory.
"""

from __future__ import annotations

import argparse
import gc
import sys
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


FMEM_FIELDS = [
    "scene",
    "N",
    "batch_size",
    "W",
    "tau",
    "weights",
    "num_generated",
    "num_train",
    "feature_dim",
    "kappa",
    "f_mem",
    "f_mem_ci_low",
    "f_mem_ci_high",
    "mean_ratio",
    "median_ratio",
    "mean_d1",
    "mean_d2",
    "generated_source",
    "generated_sha256",
    "train_source",
    "train_sha256",
]

COMBINED_FIELDS = [
    "scene",
    "N",
    "batch_size",
    "W",
    "tau",
    "epoch_float",
    "weights",
    "num_generated",
    "num_train",
    "num_test",
    "feature_dim",
    "n_folds",
    "FCD_Gen_Test",
    "FCD_Gen_Test_std",
    "FCD_Train_Test",
    "FCD_Train_Test_std",
    "kappa",
    "f_mem",
    "f_mem_ci_low",
    "f_mem_ci_high",
    "mean_ratio",
    "median_ratio",
    "mean_d1",
    "mean_d2",
    "generated_source",
    "generated_sha256",
    "train_source",
    "train_sha256",
]

WINDOW_FIELDS = [
    "N",
    "W",
    "kappa",
    "fmem_threshold",
    "rel_tol",
    "FCD_floor",
    "FCD_open_threshold",
    "tau_gen",
    "tau_mem",
    "window_width",
]

SELECTED_FIELDS = [
    "N",
    "tau",
    "regime",
    "tau_gen",
    "tau_mem",
    "FCD",
    "f_mem",
    "kappa",
    "fmem_threshold",
    "checkpoint_file",
    "generated_sample_source",
]


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def _target_taus(n: int, explicit: list[int] | None) -> list[int]:
    checkpoints = W.available_checkpoint_taus(n)
    if not checkpoints:
        raise FileNotFoundError(f"No DDIM checkpoints found for N={n}: {C.ddim_run_dir(n)}")
    requested = sorted(set(explicit if explicit is not None else C.CHECKPOINT_GRID))
    missing = sorted(set(requested) - set(checkpoints))
    if missing:
        raise FileNotFoundError(f"N={n} is missing requested checkpoint(s): {missing}")
    return requested


def _ensure_sample_source(
    n: int,
    tau: int,
    device: torch.device,
    generation_batch_size: int,
    generator_module_cache: dict[str, Any],
) -> Path:
    canonical = C.canonical_generated_samples(n, tau)
    if canonical.exists():
        W.validate_generated_file(canonical, C.N_GENERATED_FOR_FMEM)
        return canonical

    fallback = C.fallback_metric_samples(n, tau)
    if fallback.exists():
        W.validate_generated_file(fallback, C.N_GENERATED_FOR_FMEM)
        return fallback

    checkpoint = C.ddim_checkpoint(n, tau)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Neither cached samples nor a DDIM checkpoint exist for N={n}, tau={tau}."
        )
    print(f"  canonical cache missing; generating isolated fallback: {fallback}")
    C.ensure_output_directories()
    if "module" not in generator_module_cache:
        generator_module_cache["module"] = W.import_module_from_path(
            "_tau_window_generate_28ghz", C.DDIM_GENERATOR_SCRIPT
        )
    generator = generator_module_cache["module"]

    torch.manual_seed(C.GENERATOR_SEED)
    np.random.seed(C.GENERATOR_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(C.GENERATOR_SEED)
    model = generator.load_checkpoint(
        str(checkpoint), device, weights="ema", n_feat=C.WIDTH
    )
    channels = generator.sample_batched(
        model,
        C.N_GENERATED_FOR_FMEM,
        device,
        generation_batch_size,
    ).astype(np.float32)
    W.atomic_save_npz(fallback, channels=channels)
    del model, channels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    W.validate_generated_file(fallback, C.N_GENERATED_FOR_FMEM)
    return fallback


def _existing_rows_by_key(path: Path) -> tuple[list[dict[str, str]], dict[tuple[int, int], dict[str, str]]]:
    rows = W.read_csv_rows(path) if path.exists() else []
    keyed: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        key = (W.as_int(row["N"]), W.as_int(row["tau"]))
        if key in keyed:
            raise RuntimeError(f"Duplicate row in {path}: {key}")
        keyed[key] = row
    return rows, keyed


def _reusable_metric_row(
    row: dict[str, str] | None, source: Path, train_file: Path
) -> bool:
    if row is None:
        return False
    try:
        return (
            abs(W.as_float(row["kappa"]) - C.KAPPA) < 1e-12
            and W.as_int(row["num_generated"]) == C.N_GENERATED_FOR_FMEM
            and C.resolve_project_path(row["generated_source"]).resolve() == source.resolve()
            and row.get("generated_sha256") == W.sha256_file(source)
            and row.get("train_sha256") == W.sha256_file(train_file)
        )
    except (KeyError, ValueError, FileNotFoundError):
        return False


def _compute_fmem_rows(
    sizes: list[int],
    taus_by_n: dict[int, list[int]],
    device: torch.device,
    nn_batch_size: int,
    generation_batch_size: int,
    force: bool,
) -> list[dict[str, Any]]:
    fcd_module = W.import_module_from_path("_tau_window_fcd", C.FCD_MODULE)
    fmem_module = W.import_module_from_path("_tau_window_fmem", C.FMEM_MODULE)

    existing_rows, existing = _existing_rows_by_key(C.FMEM_CSV)
    output: dict[tuple[int, int], dict[str, Any]] = {
        (W.as_int(row["N"]), W.as_int(row["tau"])): row
        for row in existing_rows
    }
    generator_cache: dict[str, Any] = {}

    for n in sizes:
        run = C.ddim_run_dir(n)
        train_file = run / "train.npy"
        train_array = np.load(train_file)
        features_train = fcd_module.features_from_spatial_npy(train_array)
        if features_train.shape != (n, 256):
            raise RuntimeError(
                f"Unexpected reference feature shape for N={n}: {features_train.shape}"
            )
        train_tensor = torch.from_numpy(features_train.astype(np.float32))
        train_sha = W.sha256_file(train_file)

        print(f"\nN={n}: {len(taus_by_n[n])} checkpoint(s), train features {features_train.shape}")
        for tau in taus_by_n[n]:
            source = _ensure_sample_source(
                n, tau, device, generation_batch_size, generator_cache
            )
            old = existing.get((n, tau))
            if not force and _reusable_metric_row(old, source, train_file):
                print(f"  [cached metric] tau={tau}")
                output[(n, tau)] = old  # type: ignore[assignment]
                continue

            print(f"  computing tau={tau} from {C.project_relative(source)}")
            with np.load(source) as payload:
                samples = payload["channels"][: C.N_GENERATED_FOR_FMEM]
            features_gen = fcd_module.features_from_generated(samples)
            gen_tensor = torch.from_numpy(features_gen.astype(np.float32))
            nearest = fmem_module.nearest_ratio_l2(
                gen_tensor, train_tensor, batch_size=nn_batch_size, device=device
            )
            ratios = nearest["ratios"]
            is_memorized = ratios < C.KAPPA
            f_mem, ci_low, ci_high = fmem_module.compute_fmem_bootstrap(
                is_memorized, B=C.BOOTSTRAP_B, seed=C.BOOTSTRAP_SEED
            )
            row: dict[str, Any] = {
                "scene": C.SCENE,
                "N": n,
                "batch_size": min(n, 500),
                "W": C.WIDTH,
                "tau": tau,
                "weights": "ema",
                "num_generated": C.N_GENERATED_FOR_FMEM,
                "num_train": len(features_train),
                "feature_dim": features_train.shape[1],
                "kappa": C.KAPPA,
                "f_mem": float(f_mem),
                "f_mem_ci_low": float(ci_low),
                "f_mem_ci_high": float(ci_high),
                "mean_ratio": float(np.mean(ratios)),
                "median_ratio": float(np.median(ratios)),
                "mean_d1": float(np.mean(nearest["d1"])),
                "mean_d2": float(np.mean(nearest["d2"])),
                "generated_source": C.project_relative(source),
                "generated_sha256": W.sha256_file(source),
                "train_source": C.project_relative(train_file),
                "train_sha256": train_sha,
            }
            output[(n, tau)] = row
            # Preserve progress after every expensive nearest-neighbor pass.
            W.atomic_write_csv(
                C.FMEM_CSV,
                FMEM_FIELDS,
                [output[key] for key in sorted(output)],
            )
            del samples, features_gen, gen_tensor, nearest, ratios, is_memorized
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    rows = [output[key] for key in sorted(output)]
    W.atomic_write_csv(C.FMEM_CSV, FMEM_FIELDS, rows)
    return rows


def _merge_fcd(
    fmem_rows: list[dict[str, Any]], sizes: list[int], taus_by_n: dict[int, list[int]]
) -> list[dict[str, Any]]:
    fcd = W.load_fcd_source(sizes)
    fmem = {
        (W.as_int(row["N"]), W.as_int(row["tau"])): row
        for row in fmem_rows
    }
    existing_rows = W.read_csv_rows(C.COMBINED_CSV) if C.COMBINED_CSV.exists() else []
    target_keys = {(n, tau) for n in sizes for tau in taus_by_n[n]}
    output: dict[tuple[int, int], dict[str, Any]] = {
        (W.as_int(row["N"]), W.as_int(row["tau"])): row
        for row in existing_rows
        if (W.as_int(row["N"]), W.as_int(row["tau"])) not in target_keys
    }
    for key in sorted(target_keys):
        if key not in fcd:
            raise RuntimeError(f"Canonical FCD table has no row for {key}")
        if key not in fmem:
            raise RuntimeError(f"New f_mem table has no row for {key}")
        old = fcd[key]
        new = fmem[key]
        output[key] = {
            "scene": C.SCENE,
            "N": key[0],
            "batch_size": new["batch_size"],
            "W": C.WIDTH,
            "tau": key[1],
            "epoch_float": old.get("epoch_float", ""),
            "weights": "ema",
            "num_generated": new["num_generated"],
            "num_train": new["num_train"],
            "num_test": old.get("num_test", ""),
            "feature_dim": new["feature_dim"],
            "n_folds": old.get("n_folds", ""),
            "FCD_Gen_Test": old["FCD_Gen_Test"],
            "FCD_Gen_Test_std": old.get("FCD_Gen_Test_std", ""),
            "FCD_Train_Test": old["FCD_Train_Test"],
            "FCD_Train_Test_std": old.get("FCD_Train_Test_std", ""),
            "kappa": C.KAPPA,
            "f_mem": new["f_mem"],
            "f_mem_ci_low": new["f_mem_ci_low"],
            "f_mem_ci_high": new["f_mem_ci_high"],
            "mean_ratio": new["mean_ratio"],
            "median_ratio": new["median_ratio"],
            "mean_d1": new["mean_d1"],
            "mean_d2": new["mean_d2"],
            "generated_source": new["generated_source"],
            "generated_sha256": new["generated_sha256"],
            "train_source": new["train_source"],
            "train_sha256": new["train_sha256"],
        }
    rows = [output[key] for key in sorted(output)]
    W.atomic_write_csv(C.COMBINED_CSV, COMBINED_FIELDS, rows)
    return rows


def _derive_windows_and_selection(
    combined_rows: list[dict[str, Any]], sizes: list[int], taus_by_n: dict[int, list[int]]
) -> None:
    by_key = {
        (W.as_int(row["N"]), W.as_int(row["tau"])): row for row in combined_rows
    }
    old_windows = W.read_csv_rows(C.WINDOW_CSV) if C.WINDOW_CSV.exists() else []
    windows: dict[int, dict[str, Any]] = {
        W.as_int(row["N"]): row
        for row in old_windows
        if W.as_int(row["N"]) not in set(sizes)
    }
    old_selected = W.read_csv_rows(C.SELECTED_CSV) if C.SELECTED_CSV.exists() else []
    selected: dict[tuple[int, int], dict[str, Any]] = {
        (W.as_int(row["N"]), W.as_int(row["tau"])): row
        for row in old_selected
        if W.as_int(row["N"]) not in set(sizes)
    }

    for n in sizes:
        expected = taus_by_n[n]
        rows = [by_key[(n, tau)] for tau in expected if (n, tau) in by_key and tau > 0]
        if len(rows) != len([tau for tau in expected if tau > 0]):
            raise RuntimeError(f"Incomplete combined metric grid for N={n}")
        rows.sort(key=lambda row: W.as_int(row["tau"]))
        tau = np.asarray([W.as_int(row["tau"]) for row in rows], dtype=float)
        fcd = np.asarray([W.as_float(row["FCD_Gen_Test"]) for row in rows])
        fmem = np.asarray([W.as_float(row["f_mem"]) for row in rows])
        floors = np.asarray([W.as_float(row["FCD_Train_Test"]) for row in rows])
        if not np.allclose(floors, floors[0], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"FCD floor is inconsistent across tau for N={n}")
        floor = float(floors[0])
        open_threshold = floor * (1.0 + C.REL_TOL)
        tau_gen = W.log_interpolated_crossing(tau, fcd, open_threshold, descending=True)
        tau_mem = W.log_interpolated_crossing(
            tau, fmem, C.FMEM_THRESHOLD, descending=False
        )
        if not np.isfinite(tau_gen):
            raise RuntimeError(f"No finite FCD opening crossing was found for N={n}")
        if not np.isfinite(tau_mem):
            raise RuntimeError(
                f"No finite f_mem={C.FMEM_THRESHOLD:.2f} crossing was found for N={n}"
            )
        if tau_mem <= tau_gen:
            raise RuntimeError(
                f"Invalid N={n} window: tau_mem={tau_mem:.3f} <= tau_gen={tau_gen:.3f}"
            )
        windows[n] = {
            "N": n,
            "W": C.WIDTH,
            "kappa": C.KAPPA,
            "fmem_threshold": C.FMEM_THRESHOLD,
            "rel_tol": C.REL_TOL,
            "FCD_floor": floor,
            "FCD_open_threshold": open_threshold,
            "tau_gen": tau_gen,
            "tau_mem": tau_mem,
            "window_width": tau_mem - tau_gen,
        }

        print(f"\nSelecting regime-spanning checkpoints for N={n} ...")
        chosen = W.select_checkpoints(expected, tau_gen, tau_mem)
        print(
            f"N={n}: generalization window [{tau_gen:.1f}, {tau_mem:.1f}], "
            f"selected tau={chosen}"
        )
        for checkpoint_tau in chosen:
            metric = by_key[(n, checkpoint_tau)]
            checkpoint = C.ddim_checkpoint(n, checkpoint_tau)
            sample_source = C.resolve_project_path(metric["generated_source"])
            if not checkpoint.exists() or not sample_source.exists():
                raise FileNotFoundError(
                    f"Selected N={n}, tau={checkpoint_tau} source is missing."
                )
            selected[(n, checkpoint_tau)] = {
                "N": n,
                "tau": checkpoint_tau,
                "regime": W.classify_regime(checkpoint_tau, tau_gen, tau_mem),
                "tau_gen": tau_gen,
                "tau_mem": tau_mem,
                "FCD": metric["FCD_Gen_Test"],
                "f_mem": metric["f_mem"],
                "kappa": C.KAPPA,
                "fmem_threshold": C.FMEM_THRESHOLD,
                "checkpoint_file": C.project_relative(checkpoint),
                "generated_sample_source": C.project_relative(sample_source),
            }

    W.atomic_write_csv(C.WINDOW_CSV, WINDOW_FIELDS, [windows[key] for key in sorted(windows)])
    W.atomic_write_csv(
        C.SELECTED_CSV,
        SELECTED_FIELDS,
        [selected[key] for key in sorted(selected)],
    )


def _dry_run(sizes: list[int], taus_by_n: dict[int, list[int]]) -> None:
    fcd = W.load_fcd_source(sizes)
    print("DRY RUN — no files will be written")
    print(f"  output root : {C.OUTPUT_ROOT}")
    print(f"  kappa       : {C.KAPPA} (memorized iff rho < kappa)")
    print(f"  f_mem edge  : {C.FMEM_THRESHOLD}")
    print(f"  FCD source  : {C.FCD_SOURCE_CSV}")
    for n in sizes:
        print(f"\n  N={n} run: {C.ddim_run_dir(n)}")
        train = C.ddim_run_dir(n) / "train.npy"
        if not train.exists():
            raise FileNotFoundError(train)
        for tau in taus_by_n[n]:
            if (n, tau) not in fcd:
                raise RuntimeError(f"FCD source lacks N={n}, tau={tau}")
            source = W.generated_source_path(n, tau)
            status = "cached" if source.exists() else "will generate isolated fallback"
            if source.exists():
                W.validate_generated_file(source, C.N_GENERATED_FOR_FMEM)
            elif not C.ddim_checkpoint(n, tau).exists():
                raise FileNotFoundError(f"No generated data or checkpoint for N={n}, tau={tau}")
            print(f"    tau={tau:<6} {status}: {C.project_relative(source)}")
    print("\nPreflight passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(C.REFERENCE_SIZES))
    parser.add_argument(
        "--taus",
        type=int,
        nargs="+",
        default=None,
        help="Optional checkpoint subset. The production window requires the full grid.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<id>")
    parser.add_argument("--nn_batch_size", type=int, default=C.NN_BATCH_SIZE)
    parser.add_argument("--generation_batch_size", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Recompute target metric rows")
    parser.add_argument("--dry_run", action="store_true", help="Validate inputs only")
    args = parser.parse_args()

    sizes = sorted(set(args.sizes))
    unsupported = sorted(set(sizes) - set(C.REFERENCE_SIZES))
    if unsupported:
        parser.error(f"Unsupported main-experiment sizes: {unsupported}")
    if args.nn_batch_size <= 0 or args.generation_batch_size <= 0:
        parser.error("Batch sizes must be positive.")
    taus_by_n = {n: _target_taus(n, args.taus) for n in sizes}

    if args.dry_run:
        _dry_run(sizes, taus_by_n)
        return

    device = _device_from_arg(args.device)
    C.ensure_output_directories()
    print("=" * 78)
    print("28 GHz LoS window metrics: kappa=1/4, f_mem threshold=20%")
    print("=" * 78)
    print(f"Device       : {device}")
    print(f"Sizes        : {sizes}")
    print(f"Output root  : {C.OUTPUT_ROOT}")
    print(f"Force metric : {args.force}")

    fmem_rows = _compute_fmem_rows(
        sizes,
        taus_by_n,
        device,
        args.nn_batch_size,
        args.generation_batch_size,
        args.force,
    )
    combined_rows = _merge_fcd(fmem_rows, sizes, taus_by_n)
    _derive_windows_and_selection(combined_rows, sizes, taus_by_n)

    print("\nDone.")
    print(f"  f_mem      : {C.FMEM_CSV}")
    print(f"  FCD+f_mem  : {C.COMBINED_CSV}")
    print(f"  windows    : {C.WINDOW_CSV}")
    print(f"  checkpoints: {C.SELECTED_CSV}")


if __name__ == "__main__":
    main()
