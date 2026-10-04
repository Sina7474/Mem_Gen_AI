#!/usr/bin/env python
"""
Evaluate f_mem at all dense checkpoints from the W=256, N=50 retrain around τ=50k.

Generates 5000 EMA samples per checkpoint, computes f_mem (k=1/3), and outputs
a CSV + JSON for plotting.

Usage:
    python eval_dense_fmem.py
"""
import os, sys, json, csv, numpy as np, torch

# ── import evaluation machinery ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model_size_effect'))
import compute_wsize_fcd_fmem as m

generate_samples = m._fcd.generate_samples
features_from_generated = m._fcd.features_from_generated
features_from_spatial_npy = m._fcd.features_from_spatial_npy
nearest_ratio_l2 = m._fmem.nearest_ratio_l2
compute_fmem_bootstrap = m._fmem.compute_fmem_bootstrap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Paths
DENSE_DIR = os.path.join(PROJECT_ROOT,
    "Code/DDIM_FMM/logs_ema/dense_retrain_W256_N50")
ORIG_DIR = os.path.join(PROJECT_ROOT,
    "Code/DDIM_FMM/logs_ema/DDIM_tau_ema_50_bs50_incremental")

W = 256
N_GEN = 5000
GEN_SEED = 42
K = 1.0 / 3.0

OUT_DIR = os.path.join(os.path.dirname(__file__), "dense_fmem_results")


def load_model(ckpt_path):
    """Load EMA weights into a fresh DDIM model."""
    ddim = m.create_model(W)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ddim.load_state_dict(ck["ema_model_state_dict"], strict=True)
    return ddim


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load training data features (same 50 samples used in the original run)
    train_npy = os.path.join(ORIG_DIR, "train.npy")
    feat_train = features_from_spatial_npy(np.load(train_npy))
    train_t = torch.from_numpy(feat_train.astype(np.float32))
    print(f"Training features: {train_t.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect checkpoint paths: dense + original flanking points
    ckpts = []
    # Original checkpoints at τ=30k and 70k for context
    for tau in [30000, 70000]:
        p = os.path.join(ORIG_DIR, "checkpoints", f"checkpoint_tau_{tau}.pth")
        if os.path.exists(p):
            ckpts.append((tau, p, "original"))

    # Dense checkpoints
    ckpt_dir = os.path.join(DENSE_DIR, "checkpoints")
    for fname in sorted(os.listdir(ckpt_dir)):
        if fname.startswith("checkpoint_tau_") and fname.endswith(".pth"):
            tau = int(fname.replace("checkpoint_tau_", "").replace(".pth", ""))
            ckpts.append((tau, os.path.join(ckpt_dir, fname), "dense"))

    # Also include original τ=50k for comparison
    p50 = os.path.join(ORIG_DIR, "checkpoints", "checkpoint_tau_50000.pth")
    if os.path.exists(p50):
        ckpts.append((50000, p50, "original"))

    ckpts.sort(key=lambda x: (x[0], x[2]))
    # Remove duplicate tau values (keep both original and dense for τ=50k)

    print(f"\nEvaluating {len(ckpts)} checkpoints:")
    for tau, path, src in ckpts:
        print(f"  τ={tau:>6d}  ({src})  {os.path.basename(path)}")

    results = []
    csv_path = os.path.join(OUT_DIR, "dense_fmem.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "source", "f_mem", "ci_low", "ci_high", "n_gen"])

    for i, (tau, ckpt_path, src) in enumerate(ckpts):
        print(f"\n[{i+1}/{len(ckpts)}] τ={tau} ({src})")

        # Load & generate
        ddim = load_model(ckpt_path)
        ddim.to(device)
        samples = generate_samples(ddim, N_GEN, seed=GEN_SEED)
        del ddim
        torch.cuda.empty_cache()

        # Features → nearest-neighbour ratio test
        feat_gen = features_from_generated(samples)
        gen_t = torch.from_numpy(feat_gen.astype(np.float32))
        nn = nearest_ratio_l2(gen_t, train_t, batch_size=512, device=device)
        is_mem = (nn["ratios"] < K)
        f_mem, ci_lo, ci_hi = compute_fmem_bootstrap(is_mem, B=1000, seed=42)

        print(f"  f_mem = {f_mem:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")

        results.append({
            "tau": tau, "source": src,
            "f_mem": float(f_mem), "ci_low": float(ci_lo), "ci_high": float(ci_hi),
        })

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([tau, src, f_mem, ci_lo, ci_hi, N_GEN])

    # Save JSON
    with open(os.path.join(OUT_DIR, "dense_fmem.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {OUT_DIR}/")
    print(f"{'='*60}")

    # Summary table
    print(f"\n{'τ':>6s}  {'source':>8s}  {'f_mem':>6s}  {'95% CI':>15s}")
    print("-" * 42)
    for r in results:
        print(f"{r['tau']:>6d}  {r['source']:>8s}  {r['f_mem']:.4f}  "
              f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")


if __name__ == "__main__":
    main()
