"""Recompute full-batch f_mem at a selectable nearest-neighbour threshold.

The script preserves ``results/fullbatch_fcd_fmem.csv``.  It uses that table to
recover the exact evaluation grid, loads the already cached EMA samples, and
replaces only ``f_mem`` and ``k`` in a separate output table.  Checkpoints are
not loaded and channels are never regenerated.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import compute_fullbatch_fcd_fmem as canonical


def generated_cache(log_dir, tau, weights):
    return Path(log_dir) / "generated_ema" / (
        f"gen_{weights}_tau{int(tau)}_seed{canonical.GEN_SEED}.npz")


def main():
    parser = argparse.ArgumentParser(
        description="Recompute cached full-batch f_mem values at a selectable k")
    parser.add_argument("--k", type=float, default=0.25)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 1000])
    parser.add_argument("--min_tau", type=int, default=0)
    parser.add_argument(
        "--input_csv", type=Path,
        default=Path("results/fullbatch_fcd_fmem.csv"))
    parser.add_argument(
        "--output_csv", type=Path,
        default=Path("results/fullbatch_fcd_fmem_kappa1_4.csv"))
    parser.add_argument(
        "--nearest_batch_size", type=int, default=canonical.NN_BATCH_SIZE)
    args = parser.parse_args()

    if not 0.0 < args.k < 1.0:
        parser.error("--k must lie strictly between 0 and 1")
    if args.min_tau < 0:
        parser.error("--min_tau must be non-negative")
    if args.nearest_batch_size < 1:
        parser.error("--nearest_batch_size must be positive")
    if args.input_csv.resolve() == args.output_csv.resolve():
        parser.error("--output_csv must differ from --input_csv")

    source = pd.read_csv(args.input_csv)
    required = {"N", "tau", "weights", "num_generated", "f_mem", "k"}
    missing_columns = sorted(required - set(source.columns))
    if missing_columns:
        raise ValueError(f"Input CSV lacks required columns: {missing_columns}")

    selected = source[
        source["N"].isin(args.sizes) & (source["tau"] >= args.min_tau)
    ].copy().sort_values(["N", "tau"])
    missing_sizes = sorted(set(args.sizes) - set(selected["N"].unique()))
    if missing_sizes:
        raise ValueError(f"Dataset sizes absent from input CSV: {missing_sizes}")
    if selected.duplicated(["N", "tau"]).any():
        raise ValueError("Input CSV has duplicate (N, tau) rows in selection")

    missing_caches = []
    for row in selected.itertuples(index=False):
        path = generated_cache(
            canonical.log_dir_for(int(row.N)), row.tau, row.weights)
        if not path.exists():
            missing_caches.append(str(path))
    if missing_caches:
        raise FileNotFoundError(
            f"Missing {len(missing_caches)} sample caches; first entries:\n"
            + "\n".join(missing_caches[:10]))

    print("=" * 78)
    print("FULL-BATCH f_mem RECOMPUTATION FROM CACHED EMA SAMPLES")
    print("=" * 78)
    print(f"  k             : {args.k:g}")
    print(f"  Dataset sizes : {sorted(args.sizes)}")
    print(f"  Grid points   : {len(selected)}")
    print(f"  Device        : {canonical.DEVICE}")
    print(f"  Output CSV    : {args.output_csv}")
    print("=" * 78)

    output_groups = []
    for n_train, group in selected.groupby("N", sort=True):
        n_train = int(n_train)
        log_dir = canonical.log_dir_for(n_train)
        train_arr = np.load(Path(log_dir) / "train.npy")
        feat_train = canonical.features_from_spatial_npy(train_arr)
        train_t = torch.from_numpy(feat_train.astype(np.float32))

        updated = group.copy()
        for idx, row in tqdm(
                list(group.iterrows()), desc=f"N=B={n_train}"):
            path = generated_cache(log_dir, row["tau"], row["weights"])
            with np.load(path) as cached:
                samples = cached["channels"]
            num_generated = int(row["num_generated"])
            if samples.shape[0] < num_generated:
                raise ValueError(
                    f"Cache {path} has {samples.shape[0]} samples; "
                    f"{num_generated} required")

            feat_gen = canonical.features_from_generated(samples[:num_generated])
            gen_t = torch.from_numpy(feat_gen.astype(np.float32))
            result = canonical.nearest_ratio_l2(
                gen_t, train_t, batch_size=args.nearest_batch_size,
                device=canonical.DEVICE)
            updated.at[idx, "f_mem"] = float(
                (result["ratios"] < args.k).mean())
            updated.at[idx, "k"] = float(args.k)

        output_groups.append(updated)

    output = pd.concat(output_groups).sort_values(["N", "tau"])
    if not output["f_mem"].between(0.0, 1.0).all():
        raise ValueError("Computed f_mem values fall outside [0, 1]")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"\nSaved {len(output)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
