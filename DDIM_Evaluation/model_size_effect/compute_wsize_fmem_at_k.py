"""Recompute model-size-sweep f_mem at a selectable ratio threshold k.

This is a cache-only companion to ``compute_wsize_fcd_fmem.py``.  It reads the
canonical metric table to recover the exact (N, W, tau) grid, reuses the saved
EMA-generated channels for every point, and changes only the nearest-neighbour
ratio threshold used for f_mem.  No checkpoint is loaded and no sample is
regenerated.

The output retains the canonical table schema and values, replacing only
``f_mem``, its bootstrap confidence interval, and ``k``.  It is therefore safe
to use with the existing plotting code while preserving the k=1/3 source CSV.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import compute_wsize_fcd_fmem as canonical


DEFAULT_SIZES = [200, 1000]
DEFAULT_WIDTHS = [64, 128, 256]


def cache_path(log_dir, tau, weights, seed):
    return Path(log_dir) / "generated_ema" / (
        f"gen_{weights}_tau{int(tau)}_seed{seed}.npz")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute cached model-size f_mem values at a selectable k")
    parser.add_argument("--k", type=float, default=0.25,
                        help="Nearest-neighbour ratio threshold (default: 0.25)")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    parser.add_argument("--min_tau", type=int, default=1,
                        help="Ignore tau below this value (default: 1)")
    parser.add_argument(
        "--input_csv", type=Path,
        default=Path("results/wsize_fcd_fmem.csv"),
        help="Canonical k=1/3 table defining the evaluation grid")
    parser.add_argument(
        "--output_csv", type=Path,
        default=Path("results/wsize_fcd_fmem_kappa1_4.csv"),
        help="Separate output table (default: %(default)s)")
    parser.add_argument(
        "--nearest_batch_size", type=int, default=canonical.NN_BATCH_SIZE,
        help="Generated-sample chunk size for nearest-neighbour distances")
    parser.add_argument(
        "--bootstrap_samples", type=int, default=canonical.BOOTSTRAP_B,
        help="Bootstrap resamples for confidence intervals. Set to 0 when a "
             "consumer needs only point estimates (default: %(default)s).")
    args = parser.parse_args()

    if not 0.0 < args.k < 1.0:
        parser.error("--k must lie strictly between 0 and 1")
    if args.min_tau < 0:
        parser.error("--min_tau must be non-negative")
    if args.nearest_batch_size < 1:
        parser.error("--nearest_batch_size must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap_samples must be non-negative")
    if args.input_csv.resolve() == args.output_csv.resolve():
        parser.error("--output_csv must differ from --input_csv")

    source = pd.read_csv(args.input_csv)
    required = {
        "N", "W", "tau", "weights", "num_generated",
        "f_mem", "f_mem_ci_low", "f_mem_ci_high", "k",
    }
    missing_columns = sorted(required - set(source.columns))
    if missing_columns:
        raise ValueError(
            f"Input CSV lacks required columns: {missing_columns}")

    selected = source[
        source["N"].isin(args.sizes)
        & source["W"].isin(args.widths)
        & (source["tau"] >= args.min_tau)
    ].copy().sort_values(["N", "W", "tau"])

    requested = {(n, w) for n in args.sizes for w in args.widths}
    present = set(zip(selected["N"].astype(int), selected["W"].astype(int)))
    missing_configs = sorted(requested - present)
    if missing_configs:
        raise ValueError(f"Configurations absent from input CSV: {missing_configs}")
    if selected.duplicated(["N", "W", "tau"]).any():
        raise ValueError("Input CSV has duplicate (N, W, tau) rows in selection")

    # Validate the complete cache grid before starting the expensive metric pass.
    missing_caches = []
    for row in selected.itertuples(index=False):
        log_dir = canonical.log_dir_for(int(row.N), int(row.W))
        path = cache_path(log_dir, row.tau, row.weights, canonical.GEN_SEED)
        if not path.exists():
            missing_caches.append(str(path))
    if missing_caches:
        preview = "\n".join(missing_caches[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_caches)} generated-sample caches; first entries:\n"
            f"{preview}")

    print("=" * 78)
    print("MODEL-SIZE f_mem RECOMPUTATION FROM CACHED EMA SAMPLES")
    print("=" * 78)
    print(f"  k             : {args.k:g}")
    print(f"  Dataset sizes : {sorted(args.sizes)}")
    print(f"  Model widths  : {sorted(args.widths)}")
    print(f"  Grid points   : {len(selected)}")
    print(f"  Device        : {canonical.DEVICE}")
    print(f"  Bootstrap     : {args.bootstrap_samples} resamples")
    print(f"  Input CSV     : {args.input_csv}")
    print(f"  Output CSV    : {args.output_csv}")
    print("=" * 78)

    output_groups = []
    for (n_train, width), group in selected.groupby(["N", "W"], sort=True):
        n_train, width = int(n_train), int(width)
        log_dir = canonical.log_dir_for(n_train, width)

        train_arr = np.load(Path(log_dir) / "train.npy")
        feat_train = canonical.features_from_spatial_npy(train_arr)
        train_t = torch.from_numpy(feat_train.astype(np.float32))

        updated = group.copy()
        for idx, row in tqdm(
                list(group.iterrows()), desc=f"N={n_train} W={width}"):
            path = cache_path(
                log_dir, row["tau"], row["weights"], canonical.GEN_SEED)
            with np.load(path) as cached:
                samples = cached["channels"]
            num_generated = int(row["num_generated"])
            if samples.shape[0] < num_generated:
                raise ValueError(
                    f"Cache {path} has {samples.shape[0]} samples; "
                    f"{num_generated} required")

            feat_gen = canonical.features_from_generated(
                samples[:num_generated])
            gen_t = torch.from_numpy(feat_gen.astype(np.float32))
            result = canonical.nearest_ratio_l2(
                gen_t, train_t, batch_size=args.nearest_batch_size,
                device=canonical.DEVICE)
            is_mem = result["ratios"] < args.k
            if args.bootstrap_samples:
                f_mem, ci_low, ci_high = canonical.compute_fmem_bootstrap(
                    is_mem, B=args.bootstrap_samples,
                    seed=canonical.BOOTSTRAP_SEED)
            else:
                f_mem = float(is_mem.mean())
                ci_low = ci_high = np.nan

            updated.at[idx, "f_mem"] = float(f_mem)
            updated.at[idx, "f_mem_ci_low"] = float(ci_low)
            updated.at[idx, "f_mem_ci_high"] = float(ci_high)
            updated.at[idx, "k"] = float(args.k)

        output_groups.append(updated)

    output = pd.concat(output_groups).sort_values(["N", "W", "tau"])
    if not (output["f_mem"] <= 1.0).all() or not (output["f_mem"] >= 0.0).all():
        raise ValueError("Computed f_mem values fall outside [0, 1]")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(output)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
