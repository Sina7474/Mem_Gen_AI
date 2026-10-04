'''
Tau-Based DDIM Training for Wireless Channel Generation
=========================================================
Trains the unconditional DDIM model with checkpointing and evaluation
based on optimizer update steps (tau), not epochs.

This is the channel-DDIM analogue of the paper's (2505.17638v2) training
methodology, where tau = total number of optimizer parameter updates.

Relation: tau = epoch * steps_per_epoch, where steps_per_epoch = ceil(N / batch_size)

Usage:
    python train_DDIM_tau.py <N> --max_tau <MAX_TAU> [--incremental] [--seed S]

Examples:
    python train_DDIM_tau.py 1000 --max_tau 200000 --incremental
    python train_DDIM_tau.py 100 --max_tau 200000 --incremental --seed 42
'''

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import ToTensor
import os
import sys
import time
import json
import csv
import math

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default tau grid for evaluation (logarithmic spacing)
TAU_GRID = [
    100, 200, 500, 1000, 2000, 5000, 10000,
    20000, 50000, 100000, 200000, 500000, 1000000,
]

DEFAULT_MAX_TAU = 200000
DEFAULT_SEED = 0
BATCH_SIZE = 100
N_T = 200  # Diffusion timesteps
N_FEAT = 256
LRATE = 1e-4
BETAS = (1e-4, 0.02)

# Evaluation noise seed (fixed for deterministic evaluation)
EVAL_NOISE_SEED = 12345


# ---------------------------------------------------------------------------
# Model Architecture (Unconditional U-Net) — same as train_DDIM.py
# ---------------------------------------------------------------------------

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class Unet(nn.Module):
    """Unconditional U-Net: only time embedding, no context/location conditioning."""
    def __init__(self, in_channels, n_feat=256):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d((1, 8)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=(1, 8), stride=(1, 8)),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1, down2)
        up3 = self.up2(up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


# ---------------------------------------------------------------------------
# DDIM Schedules and Model
# ---------------------------------------------------------------------------

def ddim_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    DDIM_coeff = sqrtmab - torch.sqrt(alpha_t) * torch.sqrt(1 - alphabar_t / alpha_t)

    return {
        "alpha_t": alpha_t,
        "oneover_sqrta": oneover_sqrta,
        "sqrt_beta_t": sqrt_beta_t,
        "alphabar_t": alphabar_t,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
        "DDIM_coeff": DDIM_coeff,
    }


class DDIM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDIM, self).__init__()
        self.nn_model = nn_model.to(device)

        for k, v in ddim_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

    def forward(self, x):
        """Training forward: random timestep, random noise, predict noise."""
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.n_T))

    def evaluate_denoising_loss(self, x, t_eval, noise):
        """
        Evaluate denoising MSE at a fixed timestep t_eval with given noise.
        
        This is the paper-style evaluation: fix a diffusion time and measure
        how well the model predicts the noise.
        
        Args:
            x: clean data, shape (B, 2, 4, 32)
            t_eval: integer timestep index (1 to n_T)
            noise: pre-generated noise, same shape as x
            
        Returns:
            MSE loss (scalar)
        """
        _ts = torch.full((x.shape[0],), t_eval, dtype=torch.long, device=self.device)
        
        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )
        
        eps_hat = self.nn_model(x_t, _ts.float() / self.n_T)
        return F.mse_loss(eps_hat, noise)

    def sample(self, n_sample, size, device):
        """Unconditional DDIM sampling (deterministic, no guidance)."""
        x_i = torch.randn(n_sample, *size).to(device)

        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            eps = self.nn_model(x_i, t_is)
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.DDIM_coeff[i])

        return x_i


# ---------------------------------------------------------------------------
# Beamspace Transform Utilities
# ---------------------------------------------------------------------------

def dft_matrix(N: int) -> np.ndarray:
    n = np.arange(N)
    k = n.reshape(-1, 1)
    return np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)


def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    Ax = dft_matrix(Nx)
    Ay = dft_matrix(Ny)
    return np.kron(Ay, Ax)


def upa_to_beamspace(H: np.ndarray, Nrx_x: int, Nrx_y: int, Ntx_x: int, Ntx_y: int) -> np.ndarray:
    Ar = upa_dft_codebook(Nrx_x, Nrx_y)
    At = upa_dft_codebook(Ntx_x, Ntx_y)
    Hv = Ar.conj().T @ H @ At
    return Hv


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

SUBCARRIER_INDEX = 128


class CustomSionnaDataset(Dataset):
    def __init__(self, data_path_LoS, data_path_NLoS, percent_start, percent_end,
                 flag_data_split, save_dir, indices_dir):
        self.data_path_LoS = data_path_LoS
        self.data_path_NLoS = data_path_NLoS
        self.percent_start = percent_start
        self.percent_end = percent_end
        self.flag_data_split = flag_data_split
        self.idx_start_train = 0
        self.idx_end_train = None
        self.total_length = None
        self.idx_start = None
        self.idx_end = None
        self.indices_dir = indices_dir
        self.save_dir = save_dir
        self.load_dataset()

    def load_dataset(self):
        np_array_LoS = np.load(self.data_path_LoS)
        np_array_NLoS = np.load(self.data_path_NLoS)

        data_LoS = np_array_LoS["combined_array"][:, :, 0, :, 0, SUBCARRIER_INDEX, :]
        data_NLoS = np_array_NLoS["combined_array"][:, :, 0, :, 0, SUBCARRIER_INDEX, :]

        print(f"Number of LoS samples: {data_LoS.shape[0]}")
        print(f"Number of NLoS samples: {data_NLoS.shape[0]}")

        # Handle shuffled indices
        if os.path.exists(os.path.join(self.indices_dir, 'indices_los.npy')):
            indices_los = np.load(os.path.join(self.indices_dir, 'indices_los.npy'))
            indices_nlos = np.load(os.path.join(self.indices_dir, 'indices_nlos.npy'))
            np.save(os.path.join(self.save_dir, 'indices_los.npy'), indices_los)
            np.save(os.path.join(self.save_dir, 'indices_nlos.npy'), indices_nlos)
        else:
            indices_los = np.arange(data_LoS.shape[0])
            indices_nlos = np.arange(data_NLoS.shape[0])
            np.random.shuffle(indices_los)
            np.random.shuffle(indices_nlos)
            np.save(os.path.join(self.indices_dir, 'indices_los.npy'), indices_los)
            np.save(os.path.join(self.indices_dir, 'indices_nlos.npy'), indices_nlos)
            np.save(os.path.join(self.save_dir, 'indices_los.npy'), indices_los)
            np.save(os.path.join(self.save_dir, 'indices_nlos.npy'), indices_nlos)

        data_LoS = data_LoS[indices_los]
        data_NLoS = data_NLoS[indices_nlos]

        if self.percent_end > 1.0:
            total_samples = int(self.percent_end)
            n_los = total_samples // 2
            n_nlos = total_samples - n_los
            data_LoS = data_LoS[:n_los]
            data_NLoS = data_NLoS[:n_nlos]
            data = np.concatenate((data_LoS, data_NLoS), axis=0)
        else:
            data = np.concatenate((data_LoS, data_NLoS), axis=0)
            self.total_length = data.shape[0]

            if os.path.exists(os.path.join(self.indices_dir, 'indices.npy')):
                indices = np.load(os.path.join(self.indices_dir, 'indices.npy'))
                np.save(os.path.join(self.save_dir, 'indices.npy'), indices)
            else:
                indices = np.arange(data.shape[0])
                np.random.shuffle(indices)
                np.save(os.path.join(self.indices_dir, 'indices.npy'), indices)
                np.save(os.path.join(self.save_dir, 'indices.npy'), indices)

            data = data[indices]
            self.idx_start = int(self.percent_start * self.total_length)
            self.idx_end = int(self.percent_end * self.total_length)
            data = data[self.idx_start:self.idx_end, :, :, :]

        H_set = data[:, :, :, 0]
        coords = data[:, 0, 0, 1:]

        array1 = np.stack((np.real(H_set), np.imag(H_set)), axis=1)
        array2 = array1.copy()

        np.save(os.path.join(self.save_dir, self.flag_data_split + '.npy'), array1)

        # Transform to beamspace
        for i in range(array1.shape[0]):
            dft_data = upa_to_beamspace(array2[i, 0] + 1j * array2[i, 1],
                                        Nrx_x=2, Nrx_y=2, Ntx_x=8, Ntx_y=4)
            array1[i, 0] = np.real(dft_data)
            array1[i, 1] = np.imag(dft_data)

        np.save(os.path.join(self.save_dir, self.flag_data_split + '_coords.npy'), coords)

        # Normalize per sample
        self.data = []
        for i in range(array1.shape[0]):
            magnitude = np.sqrt(array1[i, 0, :, :] ** 2 + array1[i, 1, :, :] ** 2)
            max_magnitude = np.max(magnitude)
            array1[i, 0, :, :] /= max_magnitude
            array1[i, 1, :, :] /= max_magnitude
            self.data.append(array1[i, :2, :, :])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = ToTensor()(self.data[idx]).float()
        return data_item


# ---------------------------------------------------------------------------
# Evaluation at fixed t_eval
# ---------------------------------------------------------------------------

def find_t_eval(alphabar_t, n_T):
    """
    Find the discrete DDIM timestep whose alpha_bar is closest to the
    paper's target: alpha_bar = exp(-2 * 0.01) ≈ 0.9802.
    
    This corresponds to a low-noise level (t close to 0 in continuous time).
    """
    target_alpha_bar = np.exp(-2 * 0.01)  # ≈ 0.9802
    
    # alphabar_t is indexed [0, 1, ..., n_T] where index 0 is t=0
    # We want timesteps 1 to n_T (0 is unused in training)
    alpha_values = alphabar_t[1:].cpu().numpy()  # shape (n_T,)
    
    t_eval = int(np.argmin(np.abs(alpha_values - target_alpha_bar)) + 1)  # +1 because we skip index 0
    
    actual_alpha_bar = float(alphabar_t[t_eval].item())
    noise_level = float(np.sqrt(1 - actual_alpha_bar))
    
    print(f"  t_eval = {t_eval} (out of {n_T})")
    print(f"  target alpha_bar = {target_alpha_bar:.6f}")
    print(f"  actual alpha_bar[t_eval] = {actual_alpha_bar:.6f}")
    print(f"  noise level sqrt(1-alpha_bar) = {noise_level:.6f}")
    
    return t_eval, actual_alpha_bar, noise_level


def evaluate_loss(ddim, data_tensor, eval_noise, t_eval, device, batch_size=256):
    """
    Evaluate denoising MSE on a dataset at fixed t_eval with pre-generated noise.
    
    Args:
        ddim: DDIM model in eval mode
        data_tensor: (N, 2, 4, 32) tensor of clean normalized beamspace channels
        eval_noise: (N, 2, 4, 32) pre-generated noise tensor
        t_eval: fixed evaluation timestep
        device: torch device
        batch_size: evaluation batch size
        
    Returns:
        mean MSE loss over all samples
    """
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
# Training
# ---------------------------------------------------------------------------

def train_tau(n_train_samples=1000, max_tau=200000, incremental=False, seed=0, n_feat=256):
    """
    Tau-based training: train until max_tau optimizer steps, evaluate at TAU_GRID points.
    
    Args:
        n_train_samples: number of training samples (half LoS + half NLoS)
        max_tau: maximum optimizer update steps
        incremental: use shared indices for nested subset consistency
        seed: random seed
        n_feat: base channel width of U-Net (controls model capacity)
    """
    # Apply the requested seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine tau grid (only points <= max_tau)
    tau_grid = sorted([t for t in TAU_GRID if t <= max_tau])
    if max_tau not in tau_grid:
        tau_grid.append(max_tau)
    
    steps_per_epoch = math.ceil(n_train_samples / BATCH_SIZE)
    max_epochs = math.ceil(max_tau / steps_per_epoch)
    
    print(f"=" * 70)
    print(f"TAU-BASED DDIM TRAINING")
    print(f"=" * 70)
    print(f"  N (training samples): {n_train_samples}")
    print(f"  n_feat (base width): {n_feat}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  steps_per_epoch: {steps_per_epoch}")
    print(f"  max_tau: {max_tau}")
    print(f"  max_epochs needed: {max_epochs}")
    print(f"  tau_grid: {tau_grid}")
    print(f"  seed: {seed}")
    print(f"=" * 70)

    # Setup directories
    logs_main_dir = './logs/'
    os.makedirs(logs_main_dir, exist_ok=True)
    
    seed_suffix = f"_seed{seed}" if seed != 0 else ""
    
    # Directory naming includes n_feat when not default (256)
    nfeat_suffix = f'_nfeat{n_feat}' if n_feat != 256 else ''
    
    if incremental:
        save_dir = f'{logs_main_dir}DDIM_tau_{n_train_samples}{nfeat_suffix}_incremental{seed_suffix}/'
        # Try shared_indices first; if empty, fall back to existing training directory
        shared_idx = f'{logs_main_dir}shared_indices{seed_suffix}/'
        existing_idx = f'{logs_main_dir}DDIM_unconditional_3.5GHz_LoS+NLoS_0.0_1000_nT200_incremental{seed_suffix}/'
        if os.path.exists(os.path.join(shared_idx, 'indices_los.npy')):
            indices_path = shared_idx
        elif os.path.exists(os.path.join(existing_idx, 'indices_los.npy')):
            indices_path = existing_idx
            print(f"  Using indices from existing training: {indices_path}")
        else:
            indices_path = shared_idx
            os.makedirs(indices_path, exist_ok=True)
    else:
        save_dir = f'{logs_main_dir}DDIM_tau_{n_train_samples}{nfeat_suffix}{seed_suffix}/'
        indices_path = save_dir

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    # Initialize training log
    with open(os.path.join(save_dir, 'training_log.txt'), 'w') as f:
        f.write(f"Tau-Based DDIM Training Log\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"N: {n_train_samples}\n")
        f.write(f"batch_size: {BATCH_SIZE}\n")
        f.write(f"steps_per_epoch: {steps_per_epoch}\n")
        f.write(f"max_tau: {max_tau}\n")
        f.write(f"max_epochs: {max_epochs}\n")
        f.write(f"tau_grid: {tau_grid}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"n_T: {N_T}\n")
        f.write(f"n_feat: {n_feat}\n")
        f.write(f"lrate: {LRATE}\n")
        f.write(f"betas: {BETAS}\n")
        f.write(f"{'=' * 60}\n\n")

    # Create model
    ddim = DDIM(
        nn_model=Unet(in_channels=2, n_feat=n_feat),
        betas=BETAS,
        n_T=N_T,
        device=device,
    )
    ddim.to(device)

    model_size = sum(p.numel() for p in ddim.parameters())
    print(f"Model size: {model_size:,} parameters")

    # Dataset paths
    data_path_LoS = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"
    data_path_NLoS = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz"

    # Load training data
    print("\n** Loading TRAINING data **")
    dataset_train = CustomSionnaDataset(
        data_path_LoS, data_path_NLoS,
        0.0, n_train_samples, "train", save_dir, indices_path
    )
    print(f"  Training samples: {len(dataset_train)}")

    # Load test data (90%-100% split)
    print("\n** Loading TEST data **")
    dataset_test = CustomSionnaDataset(
        data_path_LoS, data_path_NLoS,
        0.9, 1.0, "test", save_dir, indices_path
    )
    print(f"  Test samples: {len(dataset_test)}")

    # Prepare evaluation tensors
    print("\n** Preparing evaluation tensors **")
    train_tensor = torch.stack([dataset_train[i] for i in range(len(dataset_train))])
    # Fix dimension: dataset returns (1, 2, 4, 32) from ToTensor, squeeze extra dim
    if train_tensor.dim() == 5:
        train_tensor = train_tensor.squeeze(1)
    # Transpose to match model input: (N, 2, 4, 32) -> needs (N, ch, H, W)
    # Actually ToTensor on (2, 4, 32) gives (1, 2, 4, 32) or (2, 4, 32) depending
    # Let's check and handle
    print(f"  train_tensor shape: {train_tensor.shape}")
    
    test_tensor = torch.stack([dataset_test[i] for i in range(len(dataset_test))])
    if test_tensor.dim() == 5:
        test_tensor = test_tensor.squeeze(1)
    print(f"  test_tensor shape: {test_tensor.shape}")

    # The model expects input as (B, 2, 4, 32) but the dataset uses
    # np.transpose(x, (0, 2, 3, 1)) in the original train loop.
    # Let's verify: the original forward does x = np.transpose(x, (0, 2, 3, 1))
    # which means DataLoader gives (B, 1, 2, 4, 32) or (B, 2, 4, 32)
    # and then transposes to (B, 4, 32, 2) — wait that doesn't make sense for Conv2d.
    # Let me re-check the original code: "x = np.transpose(x, (0, 2, 3, 1))"
    # DataLoader gives (B, C, H, W) from ToTensor. If original data is (2, 4, 32),
    # ToTensor makes it (1, 2, 4, 32) (adds channel dim for 3D input? No.)
    # Actually ToTensor on a numpy array of shape (2, 4, 32) treats it as (C, H, W)
    # and returns (C, H, W) = (2, 4, 32). So DataLoader batch is (B, 2, 4, 32).
    # Then np.transpose(x, (0, 2, 3, 1)) gives (B, 4, 32, 2).
    # That means the model input is (B, 4, 32, 2)? But Unet has in_channels=2...
    # Wait, that can't be right. Let me re-read:
    # In forward: self.nn_model(x_t, _ts / self.n_T) where x_t has same shape as x.
    # The model is Unet(in_channels=2, ...) which uses Conv2d(2, ...).
    # So input must be (B, 2, H, W). The transpose (0,2,3,1) makes (B,4,32,2) which
    # would be in_channels=4? No... 
    # Actually wait: np.transpose on a torch tensor? That's odd but it works.
    # (B, 2, 4, 32) -> transpose (0,2,3,1) -> (B, 4, 32, 2). In_channels=2 won't work.
    # Unless I'm misreading. Let me look again at the data format.
    # The dataset stores self.data[i] as array1[i, :2, :, :] which is shape (2, 4, 32).
    # ToTensor on (2, 4, 32) ndarray... ToTensor expects (H, W, C) for PIL or
    # for numpy it permutes (H, W, C) -> (C, H, W). But for 3D arrays:
    # Actually ToTensor for shape (H, W, C) does permute. But (2, 4, 32) is ambiguous.
    # Let me check: ToTensor with ndarray of shape (2, 4, 32) - it would interpret as
    # (H=2, W=4, C=32) and output (32, 2, 4). That seems weird too.
    # 
    # OK let me just look at what np.transpose(x, (0, 2, 3, 1)) does:
    # If batch from DataLoader is (B, 32, 2, 4) [from ToTensor interpretation],
    # then transpose (0,2,3,1) -> (B, 2, 4, 32). Yes! That's what happens:
    # - self.data[i] = (2, 4, 32) numpy
    # - ToTensor treats it as (H=2, W=4, C=32) -> output (C=32, H=2, W=4) = (32, 2, 4)  
    # - DataLoader batch: (B, 32, 2, 4)
    # - np.transpose(x, (0, 2, 3, 1)) -> (B, 2, 4, 32)
    # So the model input is indeed (B, 2, 4, 32) with in_channels=2, H=4, W=32. 
    
    # Apply same transform as training loop
    # DataLoader gives (B, 32, 2, 4), then transpose to (B, 2, 4, 32)
    train_eval = train_tensor.permute(0, 2, 3, 1) if train_tensor.shape[1] == 32 else train_tensor
    test_eval = test_tensor.permute(0, 2, 3, 1) if test_tensor.shape[1] == 32 else test_tensor
    
    # Verify shapes
    print(f"  train_eval shape (for model): {train_eval.shape}")
    print(f"  test_eval shape (for model): {test_eval.shape}")
    assert train_eval.shape[1] == 2, f"Expected 2 channels, got {train_eval.shape[1]}"

    # Find t_eval (paper-style: fixed low-noise timestep)
    print("\n** Finding evaluation timestep t_eval **")
    t_eval, actual_alpha_bar, noise_level = find_t_eval(ddim.alphabar_t, N_T)

    # Generate deterministic evaluation noise
    print("\n** Generating deterministic evaluation noise **")
    eval_gen = torch.Generator()
    eval_gen.manual_seed(EVAL_NOISE_SEED)
    
    eval_noise_train = torch.randn(train_eval.shape, generator=eval_gen)
    eval_noise_test = torch.randn(test_eval.shape, generator=eval_gen)
    
    # Save evaluation noise for reproducibility
    torch.save(eval_noise_train, os.path.join(save_dir, 'eval_noise_train.pt'))
    torch.save(eval_noise_test, os.path.join(save_dir, 'eval_noise_test.pt'))
    print(f"  Saved eval noise: train {eval_noise_train.shape}, test {eval_noise_test.shape}")

    # Save training config
    config = {
        "N": int(n_train_samples),
        "batch_size": int(BATCH_SIZE),
        "steps_per_epoch": int(steps_per_epoch),
        "max_tau": int(max_tau),
        "max_epochs": int(max_epochs),
        "tau_grid": [int(t) for t in tau_grid],
        "seed": int(seed),
        "n_T": int(N_T),
        "n_feat": int(n_feat),
        "lrate": float(LRATE),
        "betas": [float(b) for b in BETAS],
        "t_eval": int(t_eval),
        "alpha_bar_t_eval": float(actual_alpha_bar),
        "noise_level_t_eval": float(noise_level),
        "model_size": int(model_size),
        "incremental": bool(incremental),
        "eval_noise_seed": int(EVAL_NOISE_SEED),
        "num_train_samples": int(len(dataset_train)),
        "num_test_samples": int(len(dataset_test)),
    }
    with open(os.path.join(save_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Setup CSV file for loss curves
    csv_path = os.path.join(save_dir, f'tau_loss_curve_N_{n_train_samples}.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'N', 'target_tau', 'actual_tau', 'epoch_float',
            'steps_per_epoch', 'L_train', 'L_test',
            'generalization_gap', 't_eval', 'num_train_eval',
            'num_test_eval', 'checkpoint_path'
        ])

    # Setup DataLoader
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    # Optimizer
    optim = torch.optim.Adam(ddim.parameters(), lr=LRATE)

    # --- Tau-based training loop ---
    global_step = 0
    tau_idx = 0  # Index into tau_grid for next evaluation point
    epoch = 0
    
    # Track training loss EMA for logging
    loss_ema = None
    
    print(f"\n** Starting tau-based training **")
    print(f"  Will evaluate at tau = {tau_grid}")
    
    start_time = time.time()
    
    # Initial evaluation at tau=0 (before any training)
    print(f"\n  [tau=0] Evaluating initial model (before training)...")
    L_train_0 = evaluate_loss(ddim, train_eval, eval_noise_train, t_eval, device)
    L_test_0 = evaluate_loss(ddim, test_eval, eval_noise_test, t_eval, device)
    print(f"    L_train={L_train_0:.6f}, L_test={L_test_0:.6f}")
    
    # Save initial checkpoint
    ckpt_path = os.path.join(save_dir, 'checkpoints', 'checkpoint_tau_0.pth')
    torch.save({
        "model_state_dict": ddim.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "global_step": 0,
        "epoch": 0,
        "N": n_train_samples,
        "batch_size": BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch,
    }, ckpt_path)
    
    # Save initial metrics
    with open(os.path.join(save_dir, 'checkpoints', 'loss_metrics_tau_0.json'), 'w') as f:
        json.dump({
            "N": n_train_samples,
            "target_tau": 0,
            "actual_tau": 0,
            "epoch_float": 0.0,
            "L_train": L_train_0,
            "L_test": L_test_0,
            "generalization_gap": L_test_0 - L_train_0,
            "num_train_eval": len(dataset_train),
            "num_test_eval": len(dataset_test),
            "t_eval": t_eval,
        }, f, indent=2)
    
    # Write initial row to CSV
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            n_train_samples, 0, 0, 0.0,
            steps_per_epoch, L_train_0, L_test_0,
            L_test_0 - L_train_0, t_eval, len(dataset_train),
            len(dataset_test), ckpt_path
        ])
    
    with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
        f.write(f"[tau=0] L_train={L_train_0:.6f}, L_test={L_test_0:.6f}\n")

    # Main training loop
    while global_step < max_tau:
        epoch += 1
        ddim.train()
        
        for x in dataloader_train:
            # Apply same transpose as original training
            x = x.permute(0, 2, 3, 1) if x.shape[1] == 32 else x
            x = x.to(device)
            
            optim.zero_grad()
            loss = ddim(x)
            loss.backward()
            optim.step()
            
            global_step += 1
            
            # Update EMA loss
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            
            # Check if we hit a tau grid point
            if tau_idx < len(tau_grid) and global_step >= tau_grid[tau_idx]:
                target_tau = tau_grid[tau_idx]
                epoch_float = global_step / steps_per_epoch
                
                print(f"\n  [tau={global_step}] (target={target_tau}, "
                      f"epoch≈{epoch_float:.1f}) Evaluating...")
                
                # Evaluate
                L_train = evaluate_loss(ddim, train_eval, eval_noise_train, t_eval, device)
                L_test = evaluate_loss(ddim, test_eval, eval_noise_test, t_eval, device)
                gap = L_test - L_train
                
                print(f"    L_train={L_train:.6f}, L_test={L_test:.6f}, gap={gap:.6f}")
                
                # Save checkpoint
                ckpt_path = os.path.join(save_dir, 'checkpoints',
                                         f'checkpoint_tau_{global_step}.pth')
                torch.save({
                    "model_state_dict": ddim.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "N": n_train_samples,
                    "batch_size": BATCH_SIZE,
                    "steps_per_epoch": steps_per_epoch,
                }, ckpt_path)
                
                # Save loss metrics JSON
                metrics_path = os.path.join(save_dir, 'checkpoints',
                                            f'loss_metrics_tau_{global_step}.json')
                with open(metrics_path, 'w') as f:
                    json.dump({
                        "N": n_train_samples,
                        "target_tau": target_tau,
                        "actual_tau": global_step,
                        "epoch_float": epoch_float,
                        "L_train": L_train,
                        "L_test": L_test,
                        "generalization_gap": gap,
                        "num_train_eval": len(dataset_train),
                        "num_test_eval": len(dataset_test),
                        "t_eval": t_eval,
                    }, f, indent=2)
                
                # Append to CSV
                with open(csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        n_train_samples, target_tau, global_step, epoch_float,
                        steps_per_epoch, L_train, L_test,
                        gap, t_eval, len(dataset_train),
                        len(dataset_test), ckpt_path
                    ])
                
                # Log
                with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
                    f.write(f"[tau={global_step}] epoch≈{epoch_float:.1f}, "
                            f"L_train={L_train:.6f}, L_test={L_test:.6f}, "
                            f"gap={gap:.6f}, loss_ema={loss_ema:.6f}\n")
                
                tau_idx += 1
                ddim.train()  # Back to training mode
            
            # Periodic logging
            if global_step % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"  tau={global_step}/{max_tau}, loss_ema={loss_ema:.6f}, "
                      f"epoch={epoch}, elapsed={elapsed:.0f}s")
            
            # Check if we've reached max_tau
            if global_step >= max_tau:
                break
    
    # Final evaluation if not already done at max_tau
    if tau_idx == 0 or tau_grid[tau_idx - 1] != global_step:
        print(f"\n  [tau={global_step}] Final evaluation...")
        L_train = evaluate_loss(ddim, train_eval, eval_noise_train, t_eval, device)
        L_test = evaluate_loss(ddim, test_eval, eval_noise_test, t_eval, device)
        gap = L_test - L_train
        print(f"    L_train={L_train:.6f}, L_test={L_test:.6f}, gap={gap:.6f}")
        
        ckpt_path = os.path.join(save_dir, 'checkpoints',
                                 f'checkpoint_tau_{global_step}.pth')
        torch.save({
            "model_state_dict": ddim.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "N": n_train_samples,
            "batch_size": BATCH_SIZE,
            "steps_per_epoch": steps_per_epoch,
        }, ckpt_path)
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                n_train_samples, max_tau, global_step,
                global_step / steps_per_epoch,
                steps_per_epoch, L_train, L_test,
                gap, t_eval, len(dataset_train),
                len(dataset_test), ckpt_path
            ])

    # Save final model separately
    torch.save(ddim.state_dict(), os.path.join(save_dir, 'model_final.pth'))
    
    elapsed_total = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Training complete!")
    print(f"  Total tau (optimizer steps): {global_step}")
    print(f"  Total epochs: {epoch}")
    print(f"  Elapsed time: {elapsed_total:.1f}s ({elapsed_total/3600:.2f}h)")
    print(f"  Results saved to: {save_dir}")
    print(f"  CSV: {csv_path}")
    print(f"{'=' * 70}")
    
    with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
        f.write(f"\n{'=' * 60}\n")
        f.write(f"Training complete.\n")
        f.write(f"Total tau: {global_step}\n")
        f.write(f"Total epochs: {epoch}\n")
        f.write(f"Elapsed time: {elapsed_total:.1f}s\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tau-based DDIM training")
    parser.add_argument("N", type=int, help="Number of training samples")
    parser.add_argument("--max_tau", type=int, default=DEFAULT_MAX_TAU,
                        help=f"Maximum optimizer steps (default: {DEFAULT_MAX_TAU})")
    parser.add_argument("--incremental", action="store_true",
                        help="Use shared indices for nested subset consistency")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--n_feat", type=int, default=N_FEAT,
                        help=f"Base channel width of U-Net (default: {N_FEAT})")
    
    args = parser.parse_args()
    
    print(f"Training with {args.N} samples ({args.N//2} LoS + {args.N//2} NLoS)")
    print(f"Mode: {'INCREMENTAL' if args.incremental else 'INDEPENDENT'}")
    print(f"Seed: {args.seed}")
    print(f"Max tau: {args.max_tau}")
    print(f"n_feat (base width): {args.n_feat}")
    
    train_tau(
        n_train_samples=args.N,
        max_tau=args.max_tau,
        incremental=args.incremental,
        seed=args.seed,
        n_feat=args.n_feat,
    )
