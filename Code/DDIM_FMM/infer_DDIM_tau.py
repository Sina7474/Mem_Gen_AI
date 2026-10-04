'''
Tau-Based DDIM Inference: Generate samples from tau checkpoints
================================================================
Generates synthetic beamspace channels from DDIM models saved at various
tau checkpoints. Supports different model widths (n_feat) for Task 11.

Usage:
    python infer_DDIM_tau.py <N> --tau <TAU1> [<TAU2> ...] [--n_feat W] [--n_gen 5000]

Examples:
    python infer_DDIM_tau.py 1000 --tau 5000 10000 50000 100000 200000
    python infer_DDIM_tau.py 200 --tau 5000 50000 200000 --n_feat 64
    python infer_DDIM_tau.py 4000 --tau 5000 50000 200000 --n_feat 512
'''

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
from pathlib import Path

# Import model architecture from training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_DDIM_tau import Unet, DDIM, BETAS, N_T, N_FEAT


def generate_samples(ddim, device, n_generate=5000, batch_size=200):
    """Generate n_generate unconditional beamspace channels."""
    ddim.eval()
    all_generated = []

    n_batches = (n_generate + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Generating", leave=False):
            current_batch = min(batch_size, n_generate - len(all_generated))
            x_gen = ddim.sample(current_batch, (2, 4, 32), device)
            all_generated.append(x_gen.cpu().numpy())

    return np.concatenate(all_generated, axis=0)[:n_generate]


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from tau-based DDIM checkpoints")
    parser.add_argument("N", type=int, help="Training set size (to locate checkpoint dir)")
    parser.add_argument("--tau", type=int, nargs="+", required=True,
                        help="Tau checkpoint(s) to generate from")
    parser.add_argument("--n_feat", type=int, default=N_FEAT,
                        help=f"Base channel width of U-Net (default: {N_FEAT})")
    parser.add_argument("--n_gen", type=int, default=5000,
                        help="Number of samples to generate per checkpoint (default: 5000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed used during training (for dir naming)")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Batch size for generation (default: 200)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine save directory (matches train_DDIM_tau.py naming)
    logs_dir = './logs/'
    seed_suffix = f"_seed{args.seed}" if args.seed != 0 else ""
    nfeat_suffix = f'_nfeat{args.n_feat}' if args.n_feat != 256 else ''
    save_dir = f'{logs_dir}DDIM_tau_{args.N}{nfeat_suffix}_incremental{seed_suffix}/'

    if not os.path.exists(save_dir):
        print(f"ERROR: Directory not found: {save_dir}")
        print(f"Make sure training was run with N={args.N}, n_feat={args.n_feat}")
        sys.exit(1)

    # Create model
    ddim = DDIM(
        nn_model=Unet(in_channels=2, n_feat=args.n_feat),
        betas=BETAS,
        n_T=N_T,
        device=device,
    )
    ddim.to(device)

    model_size = sum(p.numel() for p in ddim.parameters())
    print(f"Model: n_feat={args.n_feat}, params={model_size:,}")
    print(f"Directory: {save_dir}")
    print(f"Tau values: {args.tau}")
    print(f"Samples per tau: {args.n_gen}")

    # Output directory for generated samples
    gen_dir = os.path.join(save_dir, 'generated_samples')
    os.makedirs(gen_dir, exist_ok=True)

    # Generate for each tau checkpoint
    for tau in args.tau:
        # Find checkpoint file
        ckpt_path = os.path.join(save_dir, 'checkpoints', f'checkpoint_tau_{tau}.pth')

        if not os.path.exists(ckpt_path):
            # Try to find the closest checkpoint
            ckpt_dir = os.path.join(save_dir, 'checkpoints')
            if os.path.exists(ckpt_dir):
                available = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_tau_')]
                available_taus = sorted([int(f.split('_')[-1].replace('.pth', '')) for f in available])
                # Find closest tau that is >= requested
                candidates = [t for t in available_taus if t >= tau]
                if candidates:
                    closest = candidates[0]
                    ckpt_path = os.path.join(ckpt_dir, f'checkpoint_tau_{closest}.pth')
                    print(f"\n  [tau={tau}] Exact checkpoint not found, using tau={closest}")
                    tau = closest
                else:
                    print(f"\n  [tau={tau}] WARNING: No checkpoint found >= {tau}. Skipping.")
                    continue
            else:
                print(f"\n  [tau={tau}] WARNING: Checkpoint dir not found. Skipping.")
                continue

        # Load checkpoint
        print(f"\n  [tau={tau}] Loading {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "model_state_dict" in ckpt:
            ddim.load_state_dict(ckpt["model_state_dict"])
        else:
            ddim.load_state_dict(ckpt)

        # Generate
        print(f"  [tau={tau}] Generating {args.n_gen} samples...")
        samples = generate_samples(ddim, device, n_generate=args.n_gen,
                                   batch_size=args.batch_size)

        # Save
        out_path = os.path.join(gen_dir, f'generated_tau_{tau}.npz')
        np.savez_compressed(out_path, channels=samples)
        print(f"  [tau={tau}] Saved: {out_path} — shape {samples.shape}")

    print(f"\nDone! Generated samples saved to: {gen_dir}")


if __name__ == "__main__":
    main()
