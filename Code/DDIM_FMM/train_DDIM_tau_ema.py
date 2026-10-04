"""
train_DDIM_tau_ema.py
    — Tau-Based DDIM Training WITH PROPER WEIGHT-EMA and a DENSE checkpoint grid
================================================================================
This is the "Option 1A" trainer from
Teach_Docs/sample_quality_smoothness_diagnosis.md — the accurate (re-training)
route to paper-like smooth, saturating sample-quality curves.

Differences vs Code/DDIM_FMM/train_DDIM_tau.py:
  1. Maintains a shadow **EMA copy of the network weights**, updated every
     optimizer step, and SAVES it (`ema_model_state_dict`) in every checkpoint.
     Downstream compute_* scripts should sample from the EMA weights.
  2. Uses a much **denser τ grid** (≈ 6 points/decade) so the quality-vs-τ
     curves do not jump between widely-spaced checkpoints.
  3. Writes everything to a **separate base folder** (`./logs_ema/`) so nothing
     mixes with previous results.
  4. Reuses the **exact same train/test split** (shuffle indices) as the
     original incremental runs, so results remain directly comparable.

Everything else (architecture, DDIM schedule, normalization, t_eval, evaluation
loss, seeds) is imported verbatim from train_DDIM_tau.py.

EMA decay uses a standard warmup schedule so early checkpoints are not stuck at
the random initialization:
        decay(step) = min(base_decay, (1 + step) / (10 + step))

Usage:
    conda activate Mem_Gen
    cd Code/DDIM_FMM

    python train_DDIM_tau_ema.py 200  --max_tau 200000 --incremental
    python train_DDIM_tau_ema.py 1000 --max_tau 200000 --incremental
    python train_DDIM_tau_ema.py 2000 --max_tau 200000 --incremental
    python train_DDIM_tau_ema.py 4000 --max_tau 200000 --incremental

    # Batch-size sweep (same N, same tau grid, own log dir per batch size):
    python train_DDIM_tau_ema.py 1000 --max_tau 200000 --incremental --batch_size 512
    python train_DDIM_tau_ema.py 1000 --max_tau 200000 --incremental --batch_size 1024
"""

import os
import sys
import time
import json
import csv
import copy
import math
import argparse
import importlib.util

import numpy as np
import torch
from torch.utils.data import DataLoader

# ── Import reusable pieces from the original tau trainer ────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_train_path = os.path.join(_HERE, 'train_DDIM_tau.py')
_spec = importlib.util.spec_from_file_location("train_ddim_tau_module", _train_path)
_train_mod = importlib.util.module_from_spec(_spec)
sys.modules['train_ddim_tau_module'] = _train_mod
_spec.loader.exec_module(_train_mod)

Unet = _train_mod.Unet
DDIM = _train_mod.DDIM
CustomSionnaDataset = _train_mod.CustomSionnaDataset
find_t_eval = _train_mod.find_t_eval
evaluate_loss = _train_mod.evaluate_loss

# Reuse hyper-parameters so training matches the originals exactly
N_T = _train_mod.N_T
BATCH_SIZE = _train_mod.BATCH_SIZE
LRATE = _train_mod.LRATE
BETAS = _train_mod.BETAS
N_FEAT = _train_mod.N_FEAT
EVAL_NOISE_SEED = _train_mod.EVAL_NOISE_SEED
DEFAULT_MAX_TAU = _train_mod.DEFAULT_MAX_TAU
DEFAULT_SEED = _train_mod.DEFAULT_SEED


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Dense (≈6 points/decade) tau grid for smooth quality curves.
# Points beyond 200000 are only reached when --max_tau is raised (e.g. the
# N=4000 extension to 400000); for the default 200000 runs they are filtered out
# by the `t <= max_tau` rule below, so other sizes are unaffected.
DENSE_TAU_GRID = [
    100, 150, 200, 300, 500, 700,
    1000, 1500, 2000, 3000, 5000, 7000,
    10000, 15000, 20000, 30000, 50000, 70000,
    100000, 150000, 200000,
    250000, 300000, 350000, 400000,
]

# Separate output base directory (kept apart from ./logs/)
LOGS_EMA_DIR = './logs_ema/'

# EMA
DEFAULT_EMA_DECAY = 0.9999

# Reuse the existing master shuffle so the split is identical to prior runs.
SHARED_INDICES_DIR = os.path.join('logs', 'DDIM_tau_1000_incremental')


def ema_decay_at(step, base_decay):
    """Warmup EMA decay: ramps from ~0 up to base_decay so early checkpoints
    track the model instead of being stuck at the random init."""
    return min(base_decay, (1.0 + step) / (10.0 + step))


@torch.no_grad()
def ema_update(ema_model, model, decay):
    """In-place EMA of parameters; buffers (BatchNorm stats + DDIM schedules)
    are copied verbatim from the live model."""
    for e, m in zip(ema_model.parameters(), model.parameters()):
        e.mul_(decay).add_(m.detach(), alpha=1.0 - decay)
    for e, m in zip(ema_model.buffers(), model.buffers()):
        e.copy_(m)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tau_ema(n_train_samples=1000, max_tau=200000, incremental=True,
                  seed=0, n_feat=256, ema_decay=DEFAULT_EMA_DECAY, save_name=None,
                  resume_from=None, batch_size=BATCH_SIZE):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tau_grid = sorted([t for t in DENSE_TAU_GRID if t <= max_tau])
    if max_tau not in tau_grid:
        tau_grid.append(max_tau)

    steps_per_epoch = math.ceil(n_train_samples / batch_size)
    max_epochs = math.ceil(max_tau / steps_per_epoch)

    print("=" * 70)
    print("TAU-BASED DDIM TRAINING  —  PROPER WEIGHT-EMA + DENSE GRID")
    print("=" * 70)
    print(f"  N (training samples): {n_train_samples}")
    print(f"  n_feat (base width): {n_feat}")
    print(f"  batch_size: {batch_size}")
    print(f"  steps_per_epoch: {steps_per_epoch}")
    print(f"  max_tau: {max_tau}")
    print(f"  max_epochs needed: {max_epochs}")
    print(f"  dense tau_grid ({len(tau_grid)} pts): {tau_grid}")
    print(f"  ema base decay: {ema_decay}  (warmup schedule)")
    print(f"  seed: {seed}")
    print("=" * 70)

    # ── Directories (separate base dir) ────────────────────────────────────
    os.makedirs(LOGS_EMA_DIR, exist_ok=True)
    seed_suffix = f"_seed{seed}" if seed != 0 else ""
    nfeat_suffix = f'_nfeat{n_feat}' if n_feat != 256 else ''
    bs_suffix = f'_bs{batch_size}' if batch_size != BATCH_SIZE else ''
    if save_name:
        save_dir = f'{LOGS_EMA_DIR}{save_name}/'
    else:
        save_dir = f'{LOGS_EMA_DIR}DDIM_tau_ema_{n_train_samples}{nfeat_suffix}{bs_suffix}{"_incremental" if incremental else ""}{seed_suffix}/'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Reuse existing master shuffle so train/test split matches prior runs.
    indices_path = SHARED_INDICES_DIR
    if not os.path.exists(os.path.join(indices_path, 'indices_los.npy')):
        raise FileNotFoundError(
            f"Shared indices not found in {indices_path}. "
            f"Cannot guarantee a matching split; aborting.")
    print(f"  Reusing shuffle indices from: {indices_path}")

    with open(os.path.join(save_dir, 'training_log.txt'), 'w') as f:
        f.write("Tau-Based DDIM Training Log (EMA + dense grid)\n")
        f.write("=" * 60 + "\n")
        f.write(f"N: {n_train_samples}\nbatch_size: {batch_size}\n")
        f.write(f"steps_per_epoch: {steps_per_epoch}\nmax_tau: {max_tau}\n")
        f.write(f"tau_grid: {tau_grid}\nseed: {seed}\nn_T: {N_T}\n")
        f.write(f"n_feat: {n_feat}\nlrate: {LRATE}\nbetas: {BETAS}\n")
        f.write(f"ema_base_decay: {ema_decay}\n")
        f.write("=" * 60 + "\n\n")

    # ── Model + EMA shadow ────────────────────────────────────────────────
    ddim = DDIM(nn_model=Unet(in_channels=2, n_feat=n_feat),
                betas=BETAS, n_T=N_T, device=device).to(device)
    ema_ddim = copy.deepcopy(ddim).to(device)
    for p in ema_ddim.parameters():
        p.requires_grad_(False)
    ema_ddim.eval()

    model_size = sum(p.numel() for p in ddim.parameters())
    print(f"Model size: {model_size:,} parameters")

    # ── Data (same paths + reused indices) ────────────────────────────────
    data_path_LoS = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"
    data_path_NLoS = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz"

    print("\n** Loading TRAINING data **")
    dataset_train = CustomSionnaDataset(
        data_path_LoS, data_path_NLoS, 0.0, n_train_samples,
        "train", save_dir, indices_path)
    print(f"  Training samples: {len(dataset_train)}")

    print("\n** Loading TEST data **")
    dataset_test = CustomSionnaDataset(
        data_path_LoS, data_path_NLoS, 0.9, 1.0,
        "test", save_dir, indices_path)
    print(f"  Test samples: {len(dataset_test)}")

    # ── Evaluation tensors (same handling as original) ────────────────────
    train_tensor = torch.stack([dataset_train[i] for i in range(len(dataset_train))])
    if train_tensor.dim() == 5:
        train_tensor = train_tensor.squeeze(1)
    test_tensor = torch.stack([dataset_test[i] for i in range(len(dataset_test))])
    if test_tensor.dim() == 5:
        test_tensor = test_tensor.squeeze(1)

    train_eval = train_tensor.permute(0, 2, 3, 1) if train_tensor.shape[1] == 32 else train_tensor
    test_eval = test_tensor.permute(0, 2, 3, 1) if test_tensor.shape[1] == 32 else test_tensor
    assert train_eval.shape[1] == 2, f"Expected 2 channels, got {train_eval.shape[1]}"
    print(f"  train_eval: {train_eval.shape}, test_eval: {test_eval.shape}")

    print("\n** Finding evaluation timestep t_eval **")
    t_eval, actual_alpha_bar, noise_level = find_t_eval(ddim.alphabar_t, N_T)

    eval_gen = torch.Generator()
    eval_gen.manual_seed(EVAL_NOISE_SEED)
    eval_noise_train = torch.randn(train_eval.shape, generator=eval_gen)
    eval_noise_test = torch.randn(test_eval.shape, generator=eval_gen)
    torch.save(eval_noise_train, os.path.join(save_dir, 'eval_noise_train.pt'))
    torch.save(eval_noise_test, os.path.join(save_dir, 'eval_noise_test.pt'))

    config = {
        "N": int(n_train_samples), "batch_size": int(batch_size),
        "steps_per_epoch": int(steps_per_epoch), "max_tau": int(max_tau),
        "max_epochs": int(max_epochs), "tau_grid": [int(t) for t in tau_grid],
        "seed": int(seed), "n_T": int(N_T), "n_feat": int(n_feat),
        "lrate": float(LRATE), "betas": [float(b) for b in BETAS],
        "t_eval": int(t_eval), "alpha_bar_t_eval": float(actual_alpha_bar),
        "noise_level_t_eval": float(noise_level), "model_size": int(model_size),
        "incremental": bool(incremental), "eval_noise_seed": int(EVAL_NOISE_SEED),
        "num_train_samples": int(len(dataset_train)),
        "num_test_samples": int(len(dataset_test)),
        "ema_base_decay": float(ema_decay), "ema_warmup": True,
        "dense_grid": True, "shared_indices_dir": indices_path,
    }
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    csv_path = os.path.join(save_dir, f'tau_loss_curve_N_{n_train_samples}.csv')
    # In resume mode the CSV already exists with earlier tau rows: append, keep
    # the header. Fresh runs write a new file with the header.
    if resume_from and os.path.exists(csv_path):
        print(f"  Resume: appending to existing CSV {csv_path}")
    else:
        with open(csv_path, 'w', newline='') as csvfile:
            csv.writer(csvfile).writerow([
                'N', 'target_tau', 'actual_tau', 'epoch_float', 'steps_per_epoch',
                'L_train', 'L_test', 'generalization_gap', 't_eval',
                'num_train_eval', 'num_test_eval', 'checkpoint_path'])

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(ddim.parameters(), lr=LRATE)

    # ── Optional resume: restore model + EMA + optimizer + step counters ──
    start_step = 0
    start_epoch = 0
    if resume_from:
        print(f"\n  RESUMING from checkpoint: {resume_from}")
        rck = torch.load(resume_from, map_location=device, weights_only=False)
        ddim.load_state_dict(rck['model_state_dict'])
        ema_ddim.load_state_dict(rck['ema_model_state_dict'])
        if 'optimizer_state_dict' in rck:
            optim.load_state_dict(rck['optimizer_state_dict'])
        start_step = int(rck.get('global_step', 0))
        start_epoch = int(rck.get('epoch', 0))
        print(f"    restored global_step={start_step}, epoch={start_epoch}")
        # Only evaluate tau points strictly beyond where we resumed.
        tau_grid = [t for t in tau_grid if t > start_step]
        print(f"    remaining tau targets: {tau_grid}")

    def save_ckpt(path, step, ep):
        torch.save({
            "model_state_dict": ddim.state_dict(),
            "ema_model_state_dict": ema_ddim.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "global_step": step, "epoch": ep, "N": n_train_samples,
            "batch_size": batch_size, "steps_per_epoch": steps_per_epoch,
            "ema_base_decay": ema_decay,
        }, path)

    # ── tau = 0 checkpoint (EMA == model at init) ─────────────────────────
    if not resume_from:
        print(f"\n  [tau=0] Evaluating initial model...")
        L_train_0 = evaluate_loss(ddim, train_eval, eval_noise_train, t_eval, device)
        L_test_0 = evaluate_loss(ddim, test_eval, eval_noise_test, t_eval, device)
        print(f"    L_train={L_train_0:.6f}, L_test={L_test_0:.6f}")
        ckpt_path = os.path.join(save_dir, 'checkpoints', 'checkpoint_tau_0.pth')
        save_ckpt(ckpt_path, 0, 0)
        with open(csv_path, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow([
                n_train_samples, 0, 0, 0.0, steps_per_epoch, L_train_0, L_test_0,
                L_test_0 - L_train_0, t_eval, len(dataset_train), len(dataset_test),
                ckpt_path])

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = start_step
    tau_idx = 0
    epoch = start_epoch
    loss_ema = None
    start_time = time.time()

    while global_step < max_tau:
        epoch += 1
        ddim.train()
        for x in dataloader_train:
            x = x.permute(0, 2, 3, 1) if x.shape[1] == 32 else x
            x = x.to(device)

            optim.zero_grad()
            loss = ddim(x)
            loss.backward()
            optim.step()

            global_step += 1

            # EMA update (warmup decay)
            ema_update(ema_ddim, ddim, ema_decay_at(global_step, ema_decay))

            loss_ema = loss.item() if loss_ema is None else 0.95 * loss_ema + 0.05 * loss.item()

            if tau_idx < len(tau_grid) and global_step >= tau_grid[tau_idx]:
                target_tau = tau_grid[tau_idx]
                epoch_float = global_step / steps_per_epoch
                print(f"\n  [tau={global_step}] (target={target_tau}, "
                      f"epoch≈{epoch_float:.1f}) Evaluating...")

                L_train = evaluate_loss(ddim, train_eval, eval_noise_train, t_eval, device)
                L_test = evaluate_loss(ddim, test_eval, eval_noise_test, t_eval, device)
                gap = L_test - L_train
                print(f"    L_train={L_train:.6f}, L_test={L_test:.6f}, gap={gap:.6f}")

                ckpt_path = os.path.join(save_dir, 'checkpoints',
                                         f'checkpoint_tau_{global_step}.pth')
                save_ckpt(ckpt_path, global_step, epoch)

                metrics_path = os.path.join(save_dir, 'checkpoints',
                                            f'loss_metrics_tau_{global_step}.json')
                with open(metrics_path, 'w') as f:
                    json.dump({
                        "N": n_train_samples, "target_tau": target_tau,
                        "actual_tau": global_step, "epoch_float": epoch_float,
                        "L_train": L_train, "L_test": L_test,
                        "generalization_gap": gap,
                        "num_train_eval": len(dataset_train),
                        "num_test_eval": len(dataset_test), "t_eval": t_eval,
                    }, f, indent=2)

                with open(csv_path, 'a', newline='') as csvfile:
                    csv.writer(csvfile).writerow([
                        n_train_samples, target_tau, global_step, epoch_float,
                        steps_per_epoch, L_train, L_test, gap, t_eval,
                        len(dataset_train), len(dataset_test), ckpt_path])

                with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
                    f.write(f"[tau={global_step}] epoch≈{epoch_float:.1f}, "
                            f"L_train={L_train:.6f}, L_test={L_test:.6f}, "
                            f"gap={gap:.6f}, loss_ema={loss_ema:.6f}\n")

                tau_idx += 1
                ddim.train()

            if global_step % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  tau={global_step}/{max_tau}, loss_ema={loss_ema:.6f}, "
                      f"epoch={epoch}, elapsed={elapsed:.0f}s")

            if global_step >= max_tau:
                break

    # ── Final checkpoint if needed ────────────────────────────────────────
    if tau_idx == 0 or tau_grid[tau_idx - 1] != global_step:
        print(f"\n  [tau={global_step}] Final evaluation...")
        L_train = evaluate_loss(ddim, train_eval, eval_noise_train, t_eval, device)
        L_test = evaluate_loss(ddim, test_eval, eval_noise_test, t_eval, device)
        gap = L_test - L_train
        ckpt_path = os.path.join(save_dir, 'checkpoints',
                                 f'checkpoint_tau_{global_step}.pth')
        save_ckpt(ckpt_path, global_step, epoch)
        with open(csv_path, 'a', newline='') as csvfile:
            csv.writer(csvfile).writerow([
                n_train_samples, max_tau, global_step,
                global_step / steps_per_epoch, steps_per_epoch,
                L_train, L_test, gap, t_eval, len(dataset_train),
                len(dataset_test), ckpt_path])

    # Save final raw + EMA weights separately
    torch.save(ddim.state_dict(), os.path.join(save_dir, 'model_final.pth'))
    torch.save(ema_ddim.state_dict(), os.path.join(save_dir, 'ema_model_final.pth'))

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"  Total tau: {global_step}   Total epochs: {epoch}")
    print(f"  Elapsed: {elapsed_total:.1f}s ({elapsed_total/3600:.2f}h)")
    print(f"  Saved to: {save_dir}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tau-based DDIM training with proper weight-EMA + dense grid")
    parser.add_argument("N", type=int, help="Number of training samples")
    parser.add_argument("--max_tau", type=int, default=DEFAULT_MAX_TAU,
                        help=f"Maximum optimizer steps (default: {DEFAULT_MAX_TAU})")
    parser.add_argument("--incremental", action="store_true",
                        help="Use shared indices for nested subset consistency")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--n_feat", type=int, default=N_FEAT,
                        help=f"Base channel width of U-Net (default: {N_FEAT})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Mini-batch size (default: {BATCH_SIZE}). Changing it "
                             f"only changes how many samples are seen per optimizer "
                             f"step; tau (=optimizer steps) and the tau grid are "
                             f"unchanged, so FCD-vs-tau curves stay directly "
                             f"comparable across batch sizes. Non-default sizes get "
                             f"their own log dir (…_bs<SIZE>_…).")
    parser.add_argument("--ema_decay", type=float, default=DEFAULT_EMA_DECAY,
                        help=f"EMA base decay (default: {DEFAULT_EMA_DECAY})")
    parser.add_argument("--save_name", type=str, default=None,
                        help="Override the output directory name under logs_ema/ "
                             "(e.g. for an extended run kept separate from the original).")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a checkpoint .pth to resume training from "
                             "(restores model+EMA+optimizer+step; only trains tau "
                             "targets beyond the checkpoint's global_step).")
    args = parser.parse_args()

    print(f"Training with {args.N} samples ({args.N//2} LoS + {args.N//2} NLoS)")
    print(f"Mode: {'INCREMENTAL' if args.incremental else 'INDEPENDENT'}")
    print(f"Seed: {args.seed}  Max tau: {args.max_tau}  n_feat: {args.n_feat}")
    print(f"Batch size: {args.batch_size}")
    print(f"EMA base decay: {args.ema_decay}")

    train_tau_ema(
        n_train_samples=args.N,
        max_tau=args.max_tau,
        incremental=args.incremental,
        seed=args.seed,
        n_feat=args.n_feat,
        ema_decay=args.ema_decay,
        save_name=args.save_name,
        resume_from=args.resume_from,
        batch_size=args.batch_size,
    )
