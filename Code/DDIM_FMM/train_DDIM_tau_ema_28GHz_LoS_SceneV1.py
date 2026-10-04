"""
train_DDIM_tau_ema_28GHz_LoS_SceneV1.py
    — Tau-Based DDIM Training with weight-EMA on the 28 GHz LoS NEW SCENE (V1 UPA)
================================================================================
Third study scene: a DIFFERENT 28 GHz environment with potentially more complex
beam structure than the original 28 GHz scene. The hypothesis is that the simpler
original scene does not reveal the downstream effect of memorization because most
users already share a few dominant beam pairs.

This trainer is structurally identical to `train_DDIM_tau_ema_28GHz_LoS.py` — same
architecture, diffusion schedule, optimizer, EMA schedule, etc. — but loads a
DIFFERENT .npz file and writes to a separate output tree so results never mix.

Dataset
-------
    File: dataset/Final_Single_Scene_Channel_Sionna_V1_28GHz_LoS_UPA.npz
    Key:  'combined_array'  shape (12940, 4, 1, 32, 1, 1, 4)  complex64
    Format: identical to the original 28 GHz scene — last dim = [channel, x, y, z]
    UPA: Rx 2x2 = 4 antennas, Tx 8x4 = 32 antennas
    Users: 12 940 LoS users

Output tree
-----------
    logs_ema_28GHz_LoS_SceneV1/DDIM_tau_ema_28GHz_LoS_SceneV1_<N>_bs<B>_incremental/
        checkpoints/ …
        tau_loss_curve_N_<N>.csv
        training_config.json
        train.npy, test.npy, *_coords.npy
        indices_28ghz_los_scenev1.npy

Usage
-----
    conda activate Mem_Gen
    cd Code/DDIM_FMM

    python train_DDIM_tau_ema_28GHz_LoS_SceneV1.py 100 --max_tau 200000 --incremental --batch_size 100
    python train_DDIM_tau_ema_28GHz_LoS_SceneV1.py 200 --max_tau 200000 --incremental --batch_size 200
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
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

# ── Import the 3.5 GHz EMA trainer so the two scenes share identical code ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ema_path = os.path.join(_HERE, 'train_DDIM_tau_ema.py')
_spec = importlib.util.spec_from_file_location("train_ddim_tau_ema_module", _ema_path)
_ema_mod = importlib.util.module_from_spec(_spec)
sys.modules['train_ddim_tau_ema_module'] = _ema_mod
_spec.loader.exec_module(_ema_mod)

_base = _ema_mod._train_mod          # train_DDIM_tau.py module

# Architecture / diffusion / evaluation — verbatim from the 3.5 GHz study
Unet = _base.Unet
DDIM = _base.DDIM
upa_to_beamspace = _base.upa_to_beamspace
find_t_eval = _base.find_t_eval
evaluate_loss = _base.evaluate_loss

# Hyper-parameters — verbatim from the 3.5 GHz study
N_T = _base.N_T                      # 200
BATCH_SIZE = _base.BATCH_SIZE        # 100
LRATE = _base.LRATE                  # 1e-4
BETAS = _base.BETAS                  # (1e-4, 0.02)
N_FEAT = _base.N_FEAT                # 256
EVAL_NOISE_SEED = _base.EVAL_NOISE_SEED   # 12345
DEFAULT_MAX_TAU = _base.DEFAULT_MAX_TAU   # 200000
DEFAULT_SEED = _base.DEFAULT_SEED         # 0

# EMA machinery — verbatim
DENSE_TAU_GRID = _ema_mod.DENSE_TAU_GRID
DEFAULT_EMA_DECAY = _ema_mod.DEFAULT_EMA_DECAY   # 0.9999
ema_decay_at = _ema_mod.ema_decay_at
ema_update = _ema_mod.ema_update


# ---------------------------------------------------------------------------
# Scene-specific configuration  *** NEW SCENE ***
# ---------------------------------------------------------------------------

DATASET_PATH = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_28GHz_LoS_UPA.npz"

# The 28 GHz file stores a single subcarrier
SUBCARRIER_INDEX = 0

# *** SEPARATE *** output tree — never mixes with the original 28 GHz scene
LOGS_DIR = './logs_ema_28GHz_LoS_SceneV1/'

# Shared shuffle for nested training subsets
SHARED_INDICES_DIR = os.path.join(LOGS_DIR, 'shared_indices_28GHz_LoS_SceneV1')
INDICES_FILE = 'indices_28ghz_los_scenev1.npy'

SPLIT_SEED = 0
TEST_PERCENT_START = 0.9
TEST_PERCENT_END = 1.0


def get_shared_indices(n_users):
    """Return the master shuffle, creating it deterministically on first call."""
    os.makedirs(SHARED_INDICES_DIR, exist_ok=True)
    path = os.path.join(SHARED_INDICES_DIR, INDICES_FILE)
    if os.path.exists(path):
        indices = np.load(path)
        if len(indices) != n_users:
            raise ValueError(
                f"Shared indices in {path} have length {len(indices)} but the "
                f"dataset has {n_users} users. Delete the file to regenerate.")
        return indices, path
    rng = np.random.RandomState(SPLIT_SEED)
    indices = rng.permutation(n_users)
    np.save(path, indices)
    print(f"  Created shared shuffle (SPLIT_SEED={SPLIT_SEED}) -> {path}")
    return indices, path


# ---------------------------------------------------------------------------
# Dataset — 28 GHz SceneV1, LoS only
# ---------------------------------------------------------------------------

class LoSOnly28GHzSceneV1Dataset(Dataset):
    """28 GHz LoS-only channels from the NEW scene V1 .npz file.

    Identical preprocessing to the original 28 GHz dataset class:
    - same UPA codebook (Rx 2x2, Tx 4x8)
    - same per-sample peak normalization
    - same beamspace transform

    Args:
        data_path:       path to the .npz containing `combined_array`.
        percent_start:   split start as a fraction of all users.
        percent_end:     split end. If > 1.0 it is interpreted as an ABSOLUTE
                         number of training samples.
        flag_data_split: 'train' or 'test'.
        save_dir:        run directory where caches go.
    """

    def __init__(self, data_path, percent_start, percent_end,
                 flag_data_split, save_dir):
        self.data_path = data_path
        self.percent_start = percent_start
        self.percent_end = percent_end
        self.flag_data_split = flag_data_split
        self.save_dir = save_dir
        self.load_dataset()

    def load_dataset(self):
        raw = np.load(self.data_path)["combined_array"]
        # (U, 4, 1, 32, 1, 1, 4) -> (U, 4, 32, 4)
        data = raw[:, :, 0, :, 0, SUBCARRIER_INDEX, :]
        n_users = data.shape[0]
        print(f"Number of 28 GHz LoS SceneV1 users: {n_users}")

        indices, _ = get_shared_indices(n_users)
        np.save(os.path.join(self.save_dir, INDICES_FILE), indices)
        data = data[indices]

        if self.percent_end > 1.0:
            n = int(self.percent_end)
            if n > n_users:
                raise ValueError(f"Requested N={n} but only {n_users} users exist.")
            data = data[:n]
        else:
            i0 = int(self.percent_start * n_users)
            i1 = int(self.percent_end * n_users)
            data = data[i0:i1]

        # last dim = [channel, x, y, z]
        H_set = data[:, :, :, 0]        # (n, 4, 32) complex
        coords = data[:, 0, 0, 1:]      # (n, 3) UE position

        # (n, 2, 4, 32) real / imaginary
        array1 = np.stack((np.real(H_set), np.imag(H_set)), axis=1)
        array2 = array1.copy()

        np.save(os.path.join(self.save_dir, self.flag_data_split + '.npy'), array1)

        # Antenna domain -> beamspace (same UPA DFT codebook as 3.5 GHz)
        for i in range(array1.shape[0]):
            dft = upa_to_beamspace(array2[i, 0] + 1j * array2[i, 1],
                                   Nrx_x=2, Nrx_y=2, Ntx_x=8, Ntx_y=4)
            array1[i, 0] = np.real(dft)
            array1[i, 1] = np.imag(dft)

        np.save(os.path.join(self.save_dir,
                             self.flag_data_split + '_coords.npy'),
                np.real(coords))

        # Per-sample normalization by the peak magnitude
        self.data = []
        for i in range(array1.shape[0]):
            mag = np.sqrt(array1[i, 0] ** 2 + array1[i, 1] ** 2)
            peak = np.max(mag)
            array1[i, 0] /= peak
            array1[i, 1] /= peak
            self.data.append(array1[i, :2, :, :])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return ToTensor()(self.data[idx]).float()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_tau_ema_28ghz_scenev1(n_train_samples=100, max_tau=200000,
                                incremental=True, seed=DEFAULT_SEED,
                                n_feat=N_FEAT, ema_decay=DEFAULT_EMA_DECAY,
                                save_name=None, resume_from=None,
                                batch_size=BATCH_SIZE):
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
    print("TAU-BASED DDIM TRAINING  —  28 GHz LoS SCENE V1 (NEW)  (EMA + dense grid)")
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

    # ── Directories ───────────────────────────────────────────────────────
    os.makedirs(LOGS_DIR, exist_ok=True)
    seed_suffix = f"_seed{seed}" if seed != DEFAULT_SEED else ""
    nfeat_suffix = f'_nfeat{n_feat}' if n_feat != N_FEAT else ''
    bs_suffix = f'_bs{batch_size}'
    if save_name:
        save_dir = os.path.join(LOGS_DIR, save_name) + '/'
    else:
        save_dir = (f'{LOGS_DIR}DDIM_tau_ema_28GHz_LoS_SceneV1_{n_train_samples}'
                    f'{nfeat_suffix}{bs_suffix}'
                    f'{"_incremental" if incremental else ""}{seed_suffix}/')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    with open(os.path.join(save_dir, 'training_log.txt'), 'w') as f:
        f.write("Tau-Based DDIM Training Log — 28 GHz LoS SceneV1 (EMA + dense grid)\n")
        f.write("=" * 60 + "\n")
        f.write(f"dataset: {DATASET_PATH}\n")
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

    # ── Data ──────────────────────────────────────────────────────────────
    print("\n** Loading TRAINING data (28 GHz LoS SceneV1) **")
    dataset_train = LoSOnly28GHzSceneV1Dataset(
        DATASET_PATH, 0.0, n_train_samples, "train", save_dir)
    print(f"  Training samples: {len(dataset_train)}")

    print("\n** Loading TEST data (28 GHz LoS SceneV1) **")
    dataset_test = LoSOnly28GHzSceneV1Dataset(
        DATASET_PATH, TEST_PERCENT_START, TEST_PERCENT_END, "test", save_dir)
    print(f"  Test samples: {len(dataset_test)}")

    # ── Evaluation tensors ────────────────────────────────────────────────
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
        "scene": "28GHz_LoS_SceneV1", "dataset": DATASET_PATH,
        "subcarrier_index": int(SUBCARRIER_INDEX),
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
        "dense_grid": True, "shared_indices_dir": SHARED_INDICES_DIR,
        "split_seed": int(SPLIT_SEED),
        "test_percent_range": [TEST_PERCENT_START, TEST_PERCENT_END],
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
        tau_grid = [t for t in tau_grid if t > start_step]
        print(f"    remaining tau targets: {tau_grid}")

    def save_ckpt(path, step, ep):
        torch.save({
            "model_state_dict": ddim.state_dict(),
            "ema_model_state_dict": ema_ddim.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "global_step": step, "epoch": ep, "N": n_train_samples,
            "batch_size": batch_size, "steps_per_epoch": steps_per_epoch,
            "ema_base_decay": ema_decay, "scene": "28GHz_LoS_SceneV1",
        }, path)

    # ── tau = 0 checkpoint ────────────────────────────────────────────────
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

    torch.save(ddim.state_dict(), os.path.join(save_dir, 'model_final.pth'))
    torch.save(ema_ddim.state_dict(), os.path.join(save_dir, 'ema_model_final.pth'))

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training complete! (28 GHz LoS SceneV1)")
    print(f"  Total tau: {global_step}   Total epochs: {epoch}")
    print(f"  Elapsed: {elapsed_total:.1f}s ({elapsed_total/3600:.2f}h)")
    print(f"  Saved to: {save_dir}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tau-based DDIM training with weight-EMA on the 28 GHz "
                    "LoS SceneV1 (NEW scene for memorization study)")
    parser.add_argument("N", type=int, help="Number of training samples")
    parser.add_argument("--max_tau", type=int, default=DEFAULT_MAX_TAU,
                        help=f"Maximum optimizer steps (default: {DEFAULT_MAX_TAU})")
    parser.add_argument("--incremental", action="store_true",
                        help="Use shared (nested) subsets.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n_feat", type=int, default=N_FEAT,
                        help=f"Base channel width W (default: {N_FEAT})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Mini-batch size (default: {BATCH_SIZE})")
    parser.add_argument("--ema_decay", type=float, default=DEFAULT_EMA_DECAY)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    print(f"Scene: 28 GHz LoS SceneV1 (NEW) — {DATASET_PATH}")
    print(f"Training with {args.N} samples (all LoS)")
    print(f"Mode: {'INCREMENTAL (nested subsets)' if args.incremental else 'INDEPENDENT'}")
    print(f"Seed: {args.seed}  Max tau: {args.max_tau}  n_feat: {args.n_feat}")
    print(f"Batch size: {args.batch_size}")
    print(f"EMA base decay: {args.ema_decay}")

    train_tau_ema_28ghz_scenev1(
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
