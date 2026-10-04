#!/usr/bin/env python
"""
Re-evaluate suspected corrupted checkpoints with multiple generation seeds.

Loads each checkpoint, generates 5000 samples × 5 independent seeds, computes
f_mem for each seed, and reports the mean/std vs the original recorded value.

Usage:
    python verify_checkpoint_fmem.py
"""
import os, sys, numpy as np, torch

# ── import the same machinery used by the sweep ──────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model_size_effect'))
import compute_wsize_fcd_fmem as m

generate_samples = m._fcd.generate_samples

SEEDS = [42, 123, 456, 789, 2025]
N_GEN = 5000

CHECKPOINTS = [
    {"W": 256, "N": 50, "tau": 50000,
     "path": "Code/DDIM_FMM/logs_ema/DDIM_tau_ema_50_bs50_incremental",
     "original_fmem": 0.335,
     "note": "severe outlier (neighbours: 0.921, 0.974); repaired to 0.921"},
    {"W": 128, "N": 50, "tau": 100000,
     "path": "Code/DDIM_FMM/logs_ema/DDIM_tau_ema_50_nfeat128_bs50_incremental",
     "original_fmem": 0.901,
     "note": "moderate dip (neighbours: 0.960, 0.978); repaired to 0.960"},
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def main():
    print("=" * 78)
    print("CHECKPOINT RE-EVALUATION — multiple seeds")
    print("=" * 78)

    for cfg in CHECKPOINTS:
        W, N, tau = cfg["W"], cfg["N"], cfg["tau"]
        log_dir = os.path.join(PROJECT_ROOT, cfg["path"])
        ckpt_path = os.path.join(log_dir, "checkpoints", f"checkpoint_tau_{tau}.pth")

        print(f"\n{'─'*78}")
        print(f"  W={W}  N={N}  τ={tau:,}")
        print(f"  Original f_mem = {cfg['original_fmem']:.4f}")
        print(f"  Note: {cfg['note']}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Seeds: {SEEDS}   Samples/seed: {N_GEN}")
        print(f"{'─'*78}")

        if not os.path.exists(ckpt_path):
            print("  ERROR: checkpoint not found, skipping.")
            continue

        # Load train features for f_mem computation
        train_arr = np.load(os.path.join(log_dir, 'train.npy'))
        feat_train = m.features_from_spatial_npy(train_arr)
        train_t = torch.from_numpy(feat_train.astype(np.float32))

        fmem_values = []
        for seed in SEEDS:
            # Fresh model each time
            ddim = m.load_checkpoint_model(ckpt_path, W)
            if ddim is None:
                print(f"  Seed {seed}: FAILED to load model")
                continue

            # Generate fresh samples (bypass cache by calling generate_samples directly)
            samples = generate_samples(ddim, N_GEN, seed=seed)

            del ddim
            if m.DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            # Compute f_mem
            feat_gen = m.features_from_generated(samples)
            gen_t = torch.from_numpy(feat_gen.astype(np.float32))
            nn_result = m.nearest_ratio_l2(
                gen_t, train_t, batch_size=m.NN_BATCH_SIZE, device=m.DEVICE)
            is_mem = (nn_result["ratios"] < m.K_MAIN)
            f_mem = float(np.mean(is_mem.astype(float)))
            fmem_values.append(f_mem)
            print(f"  Seed {seed:>5}: f_mem = {f_mem:.4f}")

        if fmem_values:
            arr = np.array(fmem_values)
            print(f"\n  RESULT:  mean = {arr.mean():.4f}  std = {arr.std():.4f}")
            print(f"           range = [{arr.min():.4f}, {arr.max():.4f}]")
            print(f"           original = {cfg['original_fmem']:.4f}")
            diff = abs(arr.mean() - cfg['original_fmem'])
            if diff > 0.10:
                print(f"  VERDICT: original was DEFECTIVE (Δ = {diff:.3f}). "
                      f"Use new mean {arr.mean():.4f}.")
            else:
                print(f"  VERDICT: original was CORRECT (Δ = {diff:.3f}). "
                      f"Do NOT repair.")

    print(f"\n{'='*78}")
    print("DONE")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
