#!/usr/bin/env python3
"""Materialize the shared split and fixed-budget downstream synthetic subsets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from . import config as C
    from . import window_data as W
except ImportError:
    import config as C
    import window_data as W


def _expected_subset_indices(source_count: int, count: int) -> np.ndarray:
    if source_count < C.N_GENERATED_FOR_FMEM:
        raise ValueError(
            f"Source has {source_count} samples; f_mem protocol requires "
            f"{C.N_GENERATED_FOR_FMEM}."
        )
    rng = np.random.default_rng(C.SYNTHETIC_SUBSET_SEED)
    # Restrict to the same first 5,000 samples used by the f_mem computation.
    return rng.permutation(C.N_GENERATED_FOR_FMEM)[:count].astype(np.int64)


def _validate_existing(
    output: Path,
    meta_file: Path,
    expected_indices: np.ndarray,
    expected_count: int,
    source: Path,
) -> None:
    W.validate_generated_file(output, expected_count)
    with np.load(output) as payload:
        channels = payload["channels"]
        if len(channels) != expected_count:
            raise ValueError(f"Wrong isolated subset count in {output}: {len(channels)}")
        if "source_indices" not in payload.files:
            raise KeyError(f"No source_indices stored in {output}")
        if not np.array_equal(payload["source_indices"], expected_indices):
            raise RuntimeError(f"Stored source indices do not match seed-0 selection: {output}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Synthetic metadata file is missing: {meta_file}")
    metadata = json.loads(meta_file.read_text())
    expected_source_hash = W.sha256_file(source)
    if metadata.get("source_sha256") != expected_source_hash:
        raise RuntimeError(f"Synthetic source hash mismatch for {output}")
    if metadata.get("output_sha256") != W.sha256_file(output):
        raise RuntimeError(f"Synthetic output hash mismatch for {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(C.REFERENCE_SIZES))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace only mismatched files inside the new isolated output root.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate without writing")
    args = parser.parse_args()

    sizes = sorted(set(args.sizes))
    unsupported = sorted(set(sizes) - set(C.REFERENCE_SIZES))
    if unsupported:
        parser.error(f"Unsupported main-experiment sizes: {unsupported}")

    selected = W.load_selected_rows(sizes)
    print("=" * 78)
    print("28 GHz LoS fixed-budget synthetic preparation")
    print("=" * 78)
    print(f"Sizes       : {sizes}")
    print(f"Subset seed : {C.SYNTHETIC_SUBSET_SEED}")
    print(f"Output root : {C.OUTPUT_ROOT}")
    print(f"Dry run     : {args.dry_run}")

    # This performs the expensive identity/leakage checks before any subset is made.
    shared = W.prepare_shared_data(force=args.force, dry_run=args.dry_run)
    print(
        f"Shared split validated: {len(shared.H_test)} test users, "
        f"reference sizes {sorted(shared.H_reference)}"
    )

    if not args.dry_run:
        C.ensure_output_directories()

    for row in selected:
        n = W.as_int(row["N"])
        tau = W.as_int(row["tau"])
        count = C.K_TOTAL - n
        source = C.resolve_project_path(row["generated_sample_source"])
        source_count, source_shape = W.validate_generated_file(
            source, C.N_GENERATED_FOR_FMEM
        )
        indices = _expected_subset_indices(source_count, count)
        output = C.downstream_synthetic_file(n, tau)
        meta_file = C.downstream_synthetic_meta_file(n, tau)

        if args.dry_run:
            action = "validate existing" if output.exists() else "create"
            print(
                f"  N={n:<4} tau={tau:<6} {action}: {count} of {source_count} "
                f"from {C.project_relative(source)}"
            )
            if output.exists() and not args.force:
                _validate_existing(output, meta_file, indices, count, source)
            continue

        if output.exists() and not args.force:
            _validate_existing(output, meta_file, indices, count, source)
            print(f"  [validated existing] N={n}, tau={tau}: {output.name}")
            continue

        with np.load(source) as payload:
            source_channels = payload["channels"][: C.N_GENERATED_FOR_FMEM]
            channels = np.asarray(source_channels[indices], dtype=np.float32)
        if channels.shape != (count, 2, 4, 32):
            raise RuntimeError(
                f"N={n}, tau={tau} subset has shape {channels.shape}; "
                f"expected {(count, 2, 4, 32)}"
            )
        if not np.isfinite(channels).all():
            raise ValueError(f"N={n}, tau={tau} subset contains non-finite values")

        W.atomic_save_npz(output, channels=channels, source_indices=indices)
        output_sha = W.sha256_file(output)
        metadata = {
            "N": n,
            "W": C.WIDTH,
            "tau": tau,
            "weights": "ema",
            "checkpoint_file": row["checkpoint_file"],
            "generator_seed": C.GENERATOR_SEED,
            "subset_seed": C.SYNTHETIC_SUBSET_SEED,
            "requested_count": count,
            "actual_count": int(len(channels)),
            "shape": list(channels.shape),
            "dtype": str(channels.dtype),
            "representation": "normalized_beamspace",
            "source_sample_file": C.project_relative(source),
            "source_sample_shape": list(source_shape),
            "source_indices_stored_in": C.project_relative(output),
            "source_sha256": W.sha256_file(source),
            "output_file": C.project_relative(output),
            "output_sha256": output_sha,
        }
        W.atomic_write_json(meta_file, metadata)
        _validate_existing(output, meta_file, indices, count, source)
        print(f"  [created] N={n}, tau={tau}: {output.name}")

    if args.dry_run:
        print("\nPreflight passed; no files were written.")
    else:
        print("\nDone.")
        print(f"  Split metadata : {C.SPLIT_META_JSON}")
        print(f"  Synthetic data : {C.SYNTHETIC_DIR}")


if __name__ == "__main__":
    main()

