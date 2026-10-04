"""
compute_w1_vs_tau.py — W1 vs Training Time τ (optimizer updates)
================================================================
For each dataset size and each saved checkpoint, generate 1000 samples
and compute W1(Gen, Test) to track quality as a function of training time.

This mirrors Figure 2 (left) from the memorization paper, but uses
Wasserstein-1 on effective rank instead of FID.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation
    python compute_w1_vs_tau.py
"""

import sys
import os
from math import ceil
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import importlib.util

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cDDIM_cFMM_Repo', 'Effective_Rank_Github'))

from effective_rank_core import compute_effective_rank, compute_wasserstein, normalize_max_abs, to_beamspace

# Import model classes directly from train_DDIM.py to ensure architecture match
_train_path = os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM', 'train_DDIM.py')
_spec = importlib.util.spec_from_file_location("train_ddim_module", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
sys.modules['train_ddim_module'] = _train_mod
_spec.loader.exec_module(_train_mod)
Unet = _train_mod.Unet
DDIM = _train_mod.DDIM

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         CONFIGURATION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

TRAIN_SIZES = [100, 200, 500, 1000, 2000, 4000, 8000]
LOGS_BASE = os.path.join(os.path.dirname(__file__), '..', 'Code', 'DDIM_FMM', 'logs')
OUTPUT_DIR = Path("results/w1_vs_tau")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Generation settings
N_GEN = 1000          # samples per checkpoint (1000 for speed; paper uses 10K)
BATCH_SIZE_GEN = 100  # generation batch size
GEN_SEED = 0          # fixed seed for reproducible generation

# Training settings (must match train_DDIM.py)
BATCH_SIZE_TRAIN = 100
N_EPOCHS = 3000
CHECKPOINT_EPOCHS = list(range(0, 2800 + 1, 200)) + [3000]  # 0,200,...,2800,3000

# UPA antenna dimensions
ANTENNA = dict(Nrx_x=2, Nrx_y=2, Ntx_x=4, Ntx_y=8)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         HELPERS                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def create_model(device):
    """Create the same model architecture used in training."""
    n_T = 200
    nn_model = Unet(in_channels=2, n_feat=256)
    ddim = DDIM(nn_model=nn_model, betas=(1e-4, 0.02), n_T=n_T, device=device)
    return ddim


def generate_samples(ddim, n_samples, batch_size, device, seed=0):
    """Generate samples from a loaded model."""
    torch.manual_seed(seed)
    ddim.eval()
    all_samples = []
    n_batches = ceil(n_samples / batch_size)
    with torch.no_grad():
        for _ in range(n_batches):
            bs = min(batch_size, n_samples - len(all_samples))
            samples = ddim.sample(bs, (2, 4, 32), device)
            all_samples.append(samples.cpu().numpy())
    return np.concatenate(all_samples, axis=0)[:n_samples]


def samples_to_erank(samples_beamspace):
    """Convert beamspace samples (N, 2, 4, 32) to effective rank values."""
    H = (samples_beamspace[:, 0] + 1j * samples_beamspace[:, 1]).astype(np.complex64)
    H = normalize_max_abs(H)
    er, _ = compute_effective_rank(H, use_power=True)
    return er


def epoch_to_tau(epoch, n_train, batch_size=BATCH_SIZE_TRAIN):
    """Convert epoch to τ (number of optimizer updates)."""
    steps_per_epoch = ceil(n_train / batch_size)
    return epoch * steps_per_epoch


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         MAIN                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("W1 vs Training Time τ (optimizer updates)")
    print(f"  Device: {DEVICE}")
    print(f"  Generating {N_GEN} samples per checkpoint")
    print(f"  Checkpoint epochs: {CHECKPOINT_EPOCHS}")
    print("=" * 70)

    # Pre-compute test effective rank (same for all sizes — 4086 samples)
    print("\nLoading test data and computing test effective rank...")
    test_path = os.path.join(LOGS_BASE, 'DDIM_unconditional_3.5GHz_LoS+NLoS_0.0_1000_nT200_incremental', 'test.npy')
    test_arr = np.load(test_path)  # (4086, 2, 4, 32) spatial domain
    H_test_spatial = (test_arr[:, 0] + 1j * test_arr[:, 1]).astype(np.complex64)
    H_test_beam = to_beamspace(H_test_spatial, **ANTENNA)
    H_test_beam = normalize_max_abs(H_test_beam)
    er_test, _ = compute_effective_rank(H_test_beam, use_power=True)
    print(f"  Test erank computed: {len(er_test)} samples, mean={np.mean(er_test):.4f}")

    # Results storage
    results = {}  # results[n_train] = list of (tau, epoch, w1_gen_test, w1_gen_train)

    for n_train in TRAIN_SIZES:
        print(f"\n{'='*70}")
        print(f"  Dataset size N = {n_train}")
        print(f"  Steps/epoch = {ceil(n_train / BATCH_SIZE_TRAIN)}")
        print(f"{'='*70}")

        log_dir = os.path.join(LOGS_BASE, f'DDIM_unconditional_3.5GHz_LoS+NLoS_0.0_{n_train}_nT200_incremental')

        # Load training data erank
        train_arr = np.load(os.path.join(log_dir, 'train.npy'))
        H_train_spatial = (train_arr[:, 0] + 1j * train_arr[:, 1]).astype(np.complex64)
        H_train_beam = to_beamspace(H_train_spatial, **ANTENNA)
        H_train_beam = normalize_max_abs(H_train_beam)
        er_train, _ = compute_effective_rank(H_train_beam, use_power=True)

        results[n_train] = []

        # Create model once, then load different checkpoints
        ddim = create_model(DEVICE)

        for ep in CHECKPOINT_EPOCHS:
            # Determine checkpoint path
            if ep == 3000:
                ckpt_path = os.path.join(log_dir, 'model.pth')
            else:
                ckpt_path = os.path.join(log_dir, 'models_x_epoch', f'model_epoch_{ep}.pth')

            if not os.path.exists(ckpt_path):
                print(f"    Epoch {ep}: SKIP (no checkpoint)")
                continue

            # Load checkpoint
            ddim.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
            ddim.eval()

            # Generate samples
            samples = generate_samples(ddim, N_GEN, BATCH_SIZE_GEN, DEVICE, seed=GEN_SEED)

            # Compute erank of generated samples (already in beamspace from the model)
            er_gen = samples_to_erank(samples)

            # Compute W1
            w1_gen_test = compute_wasserstein(er_gen, er_test, n_select=None)
            w1_gen_train = compute_wasserstein(er_gen, er_train, n_select=None)

            tau = epoch_to_tau(ep, n_train)
            results[n_train].append((tau, ep, w1_gen_test, w1_gen_train))

            print(f"    Epoch {ep:4d} | τ = {tau:7,d} | W1(Gen,Test) = {w1_gen_test:.4f} | W1(Gen,Train) = {w1_gen_train:.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Saving results...")

    # Save as CSV
    import pandas as pd
    rows = []
    for n_train in TRAIN_SIZES:
        for (tau, ep, w1_test, w1_train) in results[n_train]:
            rows.append({
                'n_train': n_train,
                'epoch': ep,
                'tau': tau,
                'steps_per_epoch': ceil(n_train / BATCH_SIZE_TRAIN),
                'batch_size': BATCH_SIZE_TRAIN,
                'w1_gen_test': w1_test,
                'w1_gen_train': w1_train,
                'n_generated': N_GEN,
            })
    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / 'w1_vs_tau.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Save as npz for easy loading
    npz_path = OUTPUT_DIR / 'w1_vs_tau.npz'
    np.savez_compressed(npz_path, **{
        f'tau_{n}': np.array([r[0] for r in results[n]]) for n in TRAIN_SIZES
    }, **{
        f'w1_gen_test_{n}': np.array([r[2] for r in results[n]]) for n in TRAIN_SIZES
    }, **{
        f'w1_gen_train_{n}': np.array([r[3] for r in results[n]]) for n in TRAIN_SIZES
    })
    print(f"  Saved: {npz_path}")

    print("\n** Computation complete! Run plot_w1_vs_tau.py to generate plots. **")


if __name__ == "__main__":
    main()
