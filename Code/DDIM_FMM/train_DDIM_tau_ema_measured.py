"""
train_DDIM_tau_ema_measured.py
    — Tau-Based DDIM Training WITH WEIGHT-EMA + DENSE grid, on the DICHASUS
      MEASURED dataset (single UE antenna, 4x8 BS UPA → shape (2, 4, 8)).
================================================================================
This is the MEASURED-data analogue of train_DDIM_tau_ema.py. It produces exactly
the checkpoint/log layout that the FCD-vs-τ and f_mem-vs-τ evaluation scripts
expect, so you can plot FCD and f_mem vs τ "like before".

What is identical to train_DDIM_tau_ema.py
------------------------------------------
  * τ (= optimizer update steps) is the training-time axis, dense grid up to
    --max_tau (default 200000);
  * a shadow EMA copy of the weights is kept and SAVED under the key
    `ema_model_state_dict` in every checkpoint (downstream scripts sample EMA);
  * checkpoints: <save_dir>/checkpoints/checkpoint_tau_{τ}.pth
  * per-τ metrics JSON + a CSV loss curve + eval noise tensors + config JSON;
  * outputs go to a separate base folder `./logs_ema/`.

What is DIFFERENT (measured data specifics)
-------------------------------------------
  * Model + beamspace pipeline imported from train_DDIM_measured.py:
      - single UE antenna  → BS-only 2-D DFT  F4^H @ h_spatial @ F8
      - 32 BS antennas re-indexed onto the physical 4x8 grid via ANTENNA_MAP
      - model input/output shape (2, 4, 8); generation size SAMPLE_SIZE=(2,4,8)
      - U-Net bottleneck (1, 2) instead of (1, 8)
  * ONE unlabelled pool of 27922 samples (no LoS/NLoS); a single shared shuffle
    `indices_all.npy`. Train = first N shuffled samples; Test = 90%–100% window.
  * Log dir: logs_ema/DDIM_tau_ema_measured_1p272GHz_N{N}[_nfeat{W}]_incremental/

Usage
-----
    conda activate Mem_Gen
    cd Code/DDIM_FMM

    python train_DDIM_tau_ema_measured.py 200  --max_tau 200000 --incremental
    python train_DDIM_tau_ema_measured.py 1000 --max_tau 200000 --incremental

    # other widths / seeds:
    python train_DDIM_tau_ema_measured.py 200 --max_tau 200000 --incremental --n_feat 128
    python train_DDIM_tau_ema_measured.py 200 --max_tau 200000 --incremental --seed 42
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

# ── Import reusable measured pieces (model, DDIM, dataset, shapes) ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_meas_path = os.path.join(_HERE, 'train_DDIM_measured.py')
_spec = importlib.util.spec_from_file_location("train_ddim_measured_module", _meas_path)
_meas = importlib.util.module_from_spec(_spec)
sys.modules['train_ddim_measured_module'] = _meas
_spec.loader.exec_module(_meas)

Unet = _meas.Unet
DDIM = _meas.DDIM
DichasusMeasuredDataset = _meas.DichasusMeasuredDataset
SPATIAL_SHAPE = _meas.SPATIAL_SHAPE      # (4, 8)
SAMPLE_SIZE = _meas.SAMPLE_SIZE          # (2, 4, 8)

# ── Hyper-parameters (match the simulated τ-EMA trainer) ───────────────────
N_T = 200
BATCH_SIZE = 100
LRATE = 1e-4
BETAS = (1e-4, 0.02)
N_FEAT = 256
EVAL_NOISE_SEED = 12345
DEFAULT_MAX_TAU = 200000
DEFAULT_SEED = 0
FREQ_TAG = "1p272GHz"

# Dense (≈6 points/decade) tau grid — identical to train_DDIM_tau_ema.py.
DENSE_TAU_GRID = [
    100, 150, 200, 300, 500, 700,
    1000, 1500, 2000, 3000, 5000, 7000,
    10000, 15000, 20000, 30000, 50000, 70000,
    100000, 150000, 200000,
    250000, 300000, 350000, 400000,
]

LOGS_EMA_DIR = './logs_ema/'
DEFAULT_EMA_DECAY = 0.9999
DATA_PATH = "../../dataset/Measured_Dataset/dichasus_0c5x_downlink_1272MHz.npz"


# ---------------------------------------------------------------------------
# Evaluation-timestep helpers (generic; mirror train_DDIM_tau.py)
# ---------------------------------------------------------------------------

def find_t_eval(alphabar_t, n_T):
    """Discrete DDIM timestep whose alpha_bar is closest to exp(-2*0.01)≈0.9802
    (paper's low-noise evaluation level)."""
    target_alpha_bar = np.exp(-2 * 0.01)
    alpha_values = alphabar_t[1:].cpu().numpy()
    t_eval = int(np.argmin(np.abs(alpha_values - target_alpha_bar)) + 1)
    actual_alpha_bar = float(alphabar_t[t_eval].item())
    noise_level = float(np.sqrt(1 - actual_alpha_bar))
    print(f"  t_eval = {t_eval} (out of {n_T})")
    print(f"  target alpha_bar = {target_alpha_bar:.6f}")
    print(f"  actual alpha_bar[t_eval] = {actual_alpha_bar:.6f}")
    print(f"  noise level sqrt(1-alpha_bar) = {noise_level:.6f}")
    return t_eval, actual_alpha_bar, noise_level


def evaluate_loss(ddim, data_tensor, eval_noise, t_eval, device, batch_size=256):
    """Mean denoising MSE at fixed t_eval with pre-generated noise."""
    ddim.eval()
    total_loss = 0.0
    n_samples = data_tensor.shape[0]
    n_batches = math.ceil(n_samples / batch_size)
    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            x = data_tensor[start:end].to(device)
            noise = eval_noise[start:end].to(device)
            loss = ddim.evaluate_denoising_loss(x, t_eval, noise)
            total_loss += loss.item() * (end - start)
    return total_loss / n_samples


# ---------------------------------------------------------------------------
# EMA helpers (identical to train_DDIM_tau_ema.py)
# ---------------------------------------------------------------------------

def ema_decay_at(step, base_decay):
    return min(base_decay, (1.0 + step) / (10.0 + step))


@torch.no_grad()
def ema_update(ema_model, model, decay):
    for e, m in zip(ema_model.parameters(), model.parameters()):
        e.mul_(decay).add_(m.detach(), alpha=1.0 - decay)
    for e, m in zip(ema_model.buffers(), model.buffers()):
        e.copy_(m)


# ---------------------------------------------------------------------------
# Eval-tensor shaping for the measured (2,4,8) pipeline
# ---------------------------------------------------------------------------

def _to_model_eval(tensor):
    """Dataset ToTensor turns (2,4,8) into (8,2,4) → batch (N,8,2,4). Permute back
    to (N,2,4,8) for the model. If already (N,2,4,8) leave it."""
    if tensor.dim() == 5:
        tensor = tensor.squeeze(1)
    if tensor.shape[1] == SPATIAL_SHAPE[1]:      # (N, 8, 2, 4)
        tensor = tensor.permute(0, 2, 3, 1)      # → (N, 2, 4, 8)
    return tensor


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tau_ema_measured(n_train_samples=1000, max_tau=200000, incremental=True,
                           seed=0, n_feat=256, ema_decay=DEFAULT_EMA_DECAY,
                           save_name=None, resume_from=None, batch_size=BATCH_SIZE):
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
    print("TAU-BASED DDIM TRAINING (MEASURED)  —  WEIGHT-EMA + DENSE GRID")
    print("=" * 70)
    print(f"  Dataset      : DICHASUS measured, shape {SAMPLE_SIZE}")
    print(f"  N (train)    : {n_train_samples}")
    print(f"  n_feat (W)   : {n_feat}")
    print(f"  batch_size   : {batch_size}")
    print(f"  steps/epoch  : {steps_per_epoch}")
    print(f"  max_tau      : {max_tau}")
    print(f"  max_epochs   : {max_epochs}")
    print(f"  dense grid   : {len(tau_grid)} pts → {tau_grid}")
    print(f"  ema decay    : {ema_decay} (warmup)")
    print(f"  seed         : {seed}")
    print("=" * 70)

    # ── Directories ────────────────────────────────────────────────────────
    os.makedirs(LOGS_EMA_DIR, exist_ok=True)
    seed_suffix = f"_seed{seed}" if seed != 0 else ""
    nfeat_suffix = f'_nfeat{n_feat}' if n_feat != 256 else ''
    bs_suffix = f'_bs{batch_size}' if batch_size != BATCH_SIZE else ''
    if save_name:
        save_dir = f'{LOGS_EMA_DIR}{save_name}/'
    else:
        save_dir = (f'{LOGS_EMA_DIR}DDIM_tau_ema_measured_{FREQ_TAG}_N{n_train_samples}'
                    f'{nfeat_suffix}{bs_suffix}{"_incremental" if incremental else ""}'
                    f'{seed_suffix}/')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Shared shuffle so all N (and train/test) use one consistent permutation.
    if incremental:
        indices_path = f'{LOGS_EMA_DIR}shared_indices_measured{seed_suffix}/'
        os.makedirs(indices_path, exist_ok=True)
    else:
        indices_path = save_dir
    print(f"  Indices dir  : {indices_path}")

    with open(os.path.join(save_dir, 'training_log.txt'), 'w') as f:
        f.write("Tau-Based DDIM Training Log — MEASURED (EMA + dense grid)\n")
        f.write("=" * 60 + "\n")
        f.write(f"dataset: DICHASUS measured 1.272 GHz, shape {SAMPLE_SIZE}\n")
        f.write(f"N: {n_train_samples}\nbatch_size: {batch_size}\n")
        f.write(f"steps_per_epoch: {steps_per_epoch}\nmax_tau: {max_tau}\n")
        f.write(f"tau_grid: {tau_grid}\nseed: {seed}\nn_T: {N_T}\n")
        f.write(f"n_feat: {n_feat}\nlrate: {LRATE}\nbetas: {BETAS}\n")
        f.write(f"ema_base_decay: {ema_decay}\n")
        f.write("note: new script — does not touch train_DDIM.py / train_DDIM_tau*.py\n")
        f.write("=" * 60 + "\n\n")

    # ── Model + EMA shadow ─────────────────────────────────────────────────
    ddim = DDIM(nn_model=Unet(in_channels=2, n_feat=n_feat),
                betas=BETAS, n_T=N_T, device=device).to(device)
    ema_ddim = copy.deepcopy(ddim).to(device)
    for p in ema_ddim.parameters():
        p.requires_grad_(False)
    ema_ddim.eval()

    model_size = sum(p.numel() for p in ddim.parameters())
    print(f"Model size: {model_size:,} parameters")

    # ── Data (single pool; train=first N, test=90-100%) ────────────────────
    print("\n** Loading TRAINING data **")
    dataset_train = DichasusMeasuredDataset(
        DATA_PATH, 0.0, n_train_samples, "train", save_dir, indices_path)
    print(f"  Training samples: {len(dataset_train)}")

    print("\n** Loading TEST data **")
    dataset_test = DichasusMeasuredDataset(
        DATA_PATH, 0.9, 1.0, "test", save_dir, indices_path)
    print(f"  Test samples: {len(dataset_test)}")

    # ── Evaluation tensors ────────────────────────────────────────────────
    train_tensor = torch.stack([dataset_train[i] for i in range(len(dataset_train))])
    test_tensor = torch.stack([dataset_test[i] for i in range(len(dataset_test))])
    train_eval = _to_model_eval(train_tensor)
    test_eval = _to_model_eval(test_tensor)
    assert train_eval.shape[1] == 2, f"Expected 2 channels, got {train_eval.shape[1]}"
    print(f"  train_eval: {tuple(train_eval.shape)}, test_eval: {tuple(test_eval.shape)}")

    print("\n** Finding evaluation timestep t_eval **")
    t_eval, actual_alpha_bar, noise_level = find_t_eval(ddim.alphabar_t, N_T)

    eval_gen = torch.Generator()
    eval_gen.manual_seed(EVAL_NOISE_SEED)
    eval_noise_train = torch.randn(train_eval.shape, generator=eval_gen)
    eval_noise_test = torch.randn(test_eval.shape, generator=eval_gen)
    torch.save(eval_noise_train, os.path.join(save_dir, 'eval_noise_train.pt'))
    torch.save(eval_noise_test, os.path.join(save_dir, 'eval_noise_test.pt'))

    config = {
        "dataset": "DICHASUS_measured_1p272GHz",
        "sample_size": list(SAMPLE_SIZE),
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
        "dft": "BS-only F4^H @ h_spatial @ F8 (single UE antenna)",
        "antenna_reindex": "ANTENNA_MAP (4x8)",
    }
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    csv_path = os.path.join(save_dir, f'tau_loss_curve_N_{n_train_samples}.csv')
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

    # ── Optional resume ───────────────────────────────────────────────────
    start_step, start_epoch = 0, 0
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
            x = _to_model_eval(x)                     # (B,8,2,4) → (B,2,4,8)
            x = x.to(device)

            optim.zero_grad()
            loss = ddim(x)
            loss.backward()
            optim.step()

            global_step += 1
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
    if tau_idx == 0 or (tau_grid and tau_grid[tau_idx - 1] != global_step):
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
        description="Tau-based DDIM training (MEASURED) with weight-EMA + dense grid")
    parser.add_argument("N", type=int, help="Number of training samples")
    parser.add_argument("--max_tau", type=int, default=DEFAULT_MAX_TAU,
                        help=f"Maximum optimizer steps (default: {DEFAULT_MAX_TAU})")
    parser.add_argument("--incremental", action="store_true",
                        help="Use shared nested indices across sizes")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--n_feat", type=int, default=N_FEAT,
                        help=f"Base U-Net width W (default: {N_FEAT})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Mini-batch size (default: {BATCH_SIZE}).")
    parser.add_argument("--ema_decay", type=float, default=DEFAULT_EMA_DECAY,
                        help=f"EMA base decay (default: {DEFAULT_EMA_DECAY})")
    parser.add_argument("--save_name", type=str, default=None,
                        help="Override output dir name under logs_ema/.")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Checkpoint .pth to resume from.")
    args = parser.parse_args()

    print(f"Training MEASURED with N={args.N}  W={args.n_feat}")
    print(f"Mode: {'INCREMENTAL' if args.incremental else 'INDEPENDENT'}")
    print(f"Seed: {args.seed}  Max tau: {args.max_tau}  Batch size: {args.batch_size}")
    print(f"EMA base decay: {args.ema_decay}")

    train_tau_ema_measured(
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
