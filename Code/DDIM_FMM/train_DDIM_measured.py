'''
Unconditional DDIM Training on the DICHASUS Measured CSI Dataset
================================================================
New, standalone script — does NOT touch / import from train_DDIM.py.

Dataset
-------
Source : DICHASUS `dichasus-0c5x` measured downlink MISO CSI
File   : ../../dataset/Measured_Dataset/dichasus_0c5x_downlink_1272MHz.npz
Arrays :
    h_downlink   (27922, 2, 1, 32)  float32   [samples, re/im, UE ant=1, BS ant=32]
    ue_locations (27922, 3)         float64   [x, y, z] metadata (NOT used for conditioning)

Key differences vs the Sionna-based train_DDIM.py
-------------------------------------------------
  * UE has a SINGLE antenna  -> no UE-side (left) DFT, only BS-side (right) DFT.
  * 32 BS antennas are NOT in raster order -> must be re-indexed onto the physical
    4x8 grid via ANTENNA_MAP before any spatial DFT.
  * Channel is single-subcarrier already -> no subcarrier selection.
  * No LoS/NLoS labels -> all 27922 samples form ONE shuffled, unlabelled pool.
  * Model input/output spatial shape is (4, 8) -> tensor (2, 4, 8), so the U-Net
    bottleneck pooling / transpose-conv are (1, 2) instead of (1, 8).

Beamspace pipeline (per sample)
-------------------------------
    h_complex = h[i,0,0,:] + 1j*h[i,1,0,:]        # (32,)  complex, flat
    h_spatial = h_complex[ANTENNA_MAP]            # (4, 8) complex, physically ordered
    h_beam    = F4.conj().T @ h_spatial @ F8      # (4, 8) complex beamspace (BS-only DFT)
    array_ri  = stack(real, imag)                 # (2, 4, 8)
    array_ri /= max(|beam|)                        # per-sample amplitude normalisation

Usage
-----
    python train_DDIM_measured.py 200                 # N=200,  W=256, seed=0
    python train_DDIM_measured.py 1000                # N=1000, W=256, seed=0
    python train_DDIM_measured.py 200 --w 128         # N=200,  W=128
    python train_DDIM_measured.py 200 --seed 42       # N=200,  seed=42
    python train_DDIM_measured.py 200 --incremental   # shared nested indices
'''

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms import ToTensor
import os
import time

# Set the SEED for reproducibility (overridden by --seed argument)
DEFAULT_SEED = 0
np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)

# ---------------------------------------------------------------------------
# BS antenna re-indexing map (flat 32-vector -> physical 4x8 UPA grid)
# rows = elevation (4), cols = azimuth (8)
# ---------------------------------------------------------------------------
ANTENNA_MAP = np.array([
    [28,  5, 10, 14,  6,  2, 16, 18],
    [19,  4, 23, 17, 20, 11,  9, 27],
    [31, 29,  0, 13,  1, 12,  3,  7],
    [30, 26, 21, 25, 22, 15, 24,  8],
], dtype=int)   # shape: (4, 8)

# Model spatial shape after the beamspace transform.
SPATIAL_SHAPE = (4, 8)          # (rows, cols)
SAMPLE_SIZE = (2, 4, 8)         # (re/im channels, rows, cols)


# ---------------------------------------------------------------------------
# Model Architecture (Unconditional U-Net) — adapted for (2, 4, 8) input
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
    """Unconditional U-Net for (2, 4, 8) inputs: only time embedding, no conditioning.

    Spatial dimension flow (rows, cols):
        init_conv               -> (4, 8)
        down1 -> MaxPool2d(2)    -> (2, 4)
        down2 -> MaxPool2d(2)    -> (1, 2)
        to_vec -> AvgPool2d(1,2) -> (1, 1)
        up0   -> ConvTranspose (1,2) -> (1, 2)
        up1   -> ConvTranspose (2,2) -> (2, 4)
        up2   -> ConvTranspose (2,2) -> (4, 8)
        out                      -> (4, 8)
    """
    def __init__(self, in_channels, n_feat=256):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        # CHANGED for (4, 8) input: bottleneck is (1, 2) -> pool the width of 2.
        self.to_vec = nn.Sequential(nn.AvgPool2d((1, 2)), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)

        # CHANGED for (4, 8) input: restore the (1, 2) bottleneck.
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=(1, 2), stride=(1, 2)),
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
        """Unconditional forward: predict noise from noisy input and timestep."""
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.n_T))

    def evaluate_denoising_loss(self, x, t_eval, noise):
        """Paper-style evaluation: denoising MSE at a FIXED timestep ``t_eval``
        with a FIXED, pre-generated ``noise`` tensor (deterministic).

        Args:
            x:       clean data, shape (B, 2, 4, 8)
            t_eval:  integer timestep index (1 .. n_T)
            noise:   pre-generated noise, same shape as ``x``
        Returns:
            MSE loss (scalar tensor).
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
# Beamspace Transform Utilities (BS side only — single UE antenna)
# ---------------------------------------------------------------------------

def dft_matrix(N: int) -> np.ndarray:
    """Unitary DFT matrix of size N x N."""
    n = np.arange(N)
    k = n.reshape(-1, 1)
    return np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)


def bs_beamspace(h_spatial: np.ndarray) -> np.ndarray:
    """Apply a 2-D DFT to a physically-ordered (4, 8) BS channel.

    Only the BS side is transformed because the UE has a single antenna,
    so the left (UE-side) DFT of the original ``Ar.conj().T @ H @ At`` drops out.
    """
    F4 = dft_matrix(SPATIAL_SHAPE[0])   # (4, 4) — elevation
    F8 = dft_matrix(SPATIAL_SHAPE[1])   # (8, 8) — azimuth
    return F4.conj().T @ h_spatial @ F8   # (4, 8) complex beamspace


# ---------------------------------------------------------------------------
# Dataset — single unlabelled pool of 27922 measured samples
# ---------------------------------------------------------------------------

class DichasusMeasuredDataset(Dataset):
    def __init__(self, data_path, percent_start, percent_end, flag_data_split, save_dir, indices_dir):
        self.data_path = data_path
        self.percent_start = percent_start
        self.percent_end = percent_end
        self.flag_data_split = flag_data_split
        self.save_dir = save_dir
        self.indices_dir = indices_dir
        self.total_length = None
        self.load_dataset()

    def load_dataset(self):
        raw = np.load(self.data_path)
        h_raw = raw["h_downlink"]      # (27922, 2, 1, 32) float32
        pos_raw = raw["ue_locations"]  # (27922, 3)        float64

        n_total = h_raw.shape[0]
        print(f"Number of measured samples: {n_total}")
        with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Number of measured samples: {n_total}\n")
            f.write(f"Carrier frequency tag: 1p272GHz (single subcarrier)\n")

        # ── One shared shuffle index over the whole pool ─────────────────────
        if os.path.exists(os.path.join(self.indices_dir, 'indices_all.npy')):
            print(f"Shuffled indices found in {self.indices_dir}. Loading indices.")
            indices_all = np.load(os.path.join(self.indices_dir, 'indices_all.npy'))
            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Shuffled indices loaded from {self.indices_dir}\n")
            np.save(os.path.join(self.save_dir, 'indices_all.npy'), indices_all)
        else:
            print("No shuffled indices found. Creating and saving them.")
            indices_all = np.arange(n_total)
            np.random.shuffle(indices_all)
            np.save(os.path.join(self.indices_dir, 'indices_all.npy'), indices_all)
            np.save(os.path.join(self.save_dir, 'indices_all.npy'), indices_all)
            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Shuffled indices created and saved to {self.indices_dir}\n")

        # Apply the shared permutation to both channel and coordinates.
        h_raw = h_raw[indices_all]
        pos_raw = pos_raw[indices_all]

        # ── Select the samples for this split ────────────────────────────────
        if self.percent_end > 1.0:
            # Training split: use the first N shuffled samples (nested subsets).
            n_samples = int(self.percent_end)
            h_sel = h_raw[:n_samples]
            pos_sel = pos_raw[:n_samples]
            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Using first {n_samples} shuffled samples for training.\n")
        else:
            # Validation split: fixed percentage window (never overlaps training).
            self.total_length = n_total
            idx_start = int(self.percent_start * self.total_length)
            idx_end = int(self.percent_end * self.total_length)
            h_sel = h_raw[idx_start:idx_end]
            pos_sel = pos_raw[idx_start:idx_end]
            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Validation window: [{idx_start}:{idx_end}] of {self.total_length}\n")

        n_sel = h_sel.shape[0]

        # ── Build complex channel and re-index onto the physical 4x8 grid ────
        # h_sel[i, 0, 0, :] = real, h_sel[i, 1, 0, :] = imag  -> (32,) complex
        h_complex = h_sel[:, 0, 0, :] + 1j * h_sel[:, 1, 0, :]   # (n_sel, 32)
        h_spatial = h_complex[:, ANTENNA_MAP]                     # (n_sel, 4, 8)

        # Save raw spatial-domain data (re/im stacked) for reference.
        array_spatial = np.stack((np.real(h_spatial), np.imag(h_spatial)), axis=1)  # (n_sel, 2, 4, 8)
        np.save(os.path.join(self.save_dir, self.flag_data_split + '.npy'), array_spatial)
        np.save(os.path.join(self.save_dir, self.flag_data_split + '_coords.npy'), pos_sel)
        print(f"** {self.flag_data_split} data saved to {os.path.join(self.save_dir, self.flag_data_split + '.npy')}")

        # ── Beamspace transform (BS-only DFT) + per-sample normalisation ─────
        self.data = []
        for i in range(n_sel):
            h_beam = bs_beamspace(h_spatial[i])                  # (4, 8) complex
            array_ri = np.stack([np.real(h_beam), np.imag(h_beam)], axis=0)  # (2, 4, 8)

            magnitude = np.sqrt(array_ri[0] ** 2 + array_ri[1] ** 2)         # (4, 8)
            max_mag = np.max(magnitude)
            array_ri = array_ri / max_mag                                    # in [-1, 1]

            self.data.append(array_ri.astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = ToTensor()(self.data[idx]).float()
        return data_item


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_train_samples=1000, n_feat=256, incremental=False, seed=0):
    # Apply the requested seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_epoch = 3000
    batch_size = 100
    n_T = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    lrate = 1e-4
    freq_tag = "1p272GHz"

    logs_main_dir = './logs/'
    os.makedirs(logs_main_dir, exist_ok=True)

    day_of_experiment = time.strftime("%d_%m_%Y_%H_%M")
    print(f"Day of experiment: {day_of_experiment}")

    val_split_init = 0.7
    val_split_end = 0.9

    seed_suffix = f"_seed{seed}" if seed != 0 else ""

    # Log directory convention:
    #   logs/DDIM_measured_1p272GHz_N{N}_nT{n_T}_W{W}[_incremental][_seed{seed}]/
    incr_suffix = "_incremental" if incremental else ""
    save_dir = (
        f'{logs_main_dir}DDIM_measured_{freq_tag}_N{n_train_samples}_nT{n_T}_W{n_feat}'
        f'{incr_suffix}{seed_suffix}/'
    )

    if incremental:
        # Incremental mode: all sizes share the same shuffled indices (nested subsets).
        indices_path = f'{logs_main_dir}shared_indices_measured{seed_suffix}/'
        os.makedirs(indices_path, exist_ok=True)
    else:
        # Independent mode: this size creates/uses its own shuffled indices.
        indices_path = save_dir

    print(f"** Storing results in {save_dir}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    os.makedirs(os.path.join(save_dir, 'models_x_epoch'), exist_ok=True)

    ddim = DDIM(
        nn_model=Unet(in_channels=2, n_feat=n_feat),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
    )
    ddim.to(device)

    model_size = sum(p.numel() for p in ddim.parameters())
    print(f"Model size: {model_size} parameters")

    # Dataset path (relative to Code/DDIM_FMM/)
    data_path = "../../dataset/Measured_Dataset/dichasus_0c5x_downlink_1272MHz.npz"

    print("** TRAINING **")
    dataset_train = DichasusMeasuredDataset(data_path, 0.0, n_train_samples, "train", save_dir, indices_path)
    print("** VALIDATION **")
    dataset_val = DichasusMeasuredDataset(data_path, val_split_init, val_split_end, "val", save_dir, indices_path)

    print(f"Number of samples in training split: {len(dataset_train)}")
    print(f"Number of samples in validation split: {len(dataset_val)}")
    with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
        f.write(f"Number of samples in training split: {len(dataset_train)}\n")
        f.write(f"Number of samples in validation split: {len(dataset_val)}\n")

    # Save training config
    with open(os.path.join(save_dir, 'training_config.txt'), 'w') as f:
        f.write(f"n_epoch: {n_epoch}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"n_T: {n_T}\n")
        f.write(f"device: {device}\n")
        f.write(f"n_feat (W): {n_feat}\n")
        f.write(f"lrate: {lrate}\n")
        f.write(f"n_train_samples (N): {n_train_samples}\n")
        f.write(f"val_split_init: {val_split_init}\n")
        f.write(f"val_split_end: {val_split_end}\n")
        f.write(f"data_path: {data_path}\n")
        f.write(f"indices_path: {indices_path}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"carrier_frequency: 1.272 GHz (single subcarrier)\n")
        f.write(f"spatial_shape: {SPATIAL_SHAPE} (rows=elevation, cols=azimuth)\n")
        f.write(f"model_input_shape: {SAMPLE_SIZE}\n")
        f.write(f"dft_config: BS-only 2-D DFT, F4.conj().T @ h_spatial @ F8\n")
        f.write(f"antenna_reindex: ANTENNA_MAP (4x8) applied before DFT\n")
        f.write(f"conditioning: unconditional\n")
        f.write(f"model_size: {model_size}\n")
        f.write(f"note: new script - does not touch train_DDIM.py\n")

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)

    optim = torch.optim.Adam(ddim.parameters(), lr=lrate)

    # Track losses for plotting
    train_loss_history = []
    val_loss_history = []
    val_epochs = []

    best_val_loss = float('inf')
    for ep in range(n_epoch):
        print(f'** Epoch {ep}/{n_epoch} **')
        ddim.train()

        pbar = tqdm(dataloader_train)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x = np.transpose(x, (0, 2, 3, 1))   # (B, 8, 2, 4) -> (B, 2, 4, 8)
            x = x.to(device)
            loss = ddim(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"Training loss: {loss_ema:.4f}")
            optim.step()

        train_loss_history.append(loss_ema)

        with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
            f.write(f'Epoch {ep}/{n_epoch}, Training loss: {loss_ema:.4f}\n')

        # Evaluate every 200 epochs
        if ep % 200 == 0:
            ddim.eval()
            print(f"Evaluating model at epoch {ep}...")
            with torch.no_grad():
                list_mse_losses = []
                pbar = tqdm(dataloader_val)
                start_time = time.time()
                for x_val in pbar:
                    x_val = np.transpose(x_val, (0, 2, 3, 1))
                    x_val = x_val.to(device)

                    # Generate unconditional samples (note the (2, 4, 8) size)
                    x_gen = ddim.sample(x_val.shape[0], SAMPLE_SIZE, device)

                    mse = F.mse_loss(x_gen, x_val)
                    pbar.set_description(f"validation loss: {mse.item():.4f}")
                    list_mse_losses.append(mse.item())

                end_time = time.time()
                inference_time = end_time - start_time
                validation_loss = np.mean(list_mse_losses)
                val_loss_history.append(validation_loss)
                val_epochs.append(ep)
                print(f"  Inference time: {inference_time:.2f} seconds")
                print(f"  Validation loss (MSE): {validation_loss:.4f}")

                with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
                    f.write(f'Epoch {ep}, Validation loss: {validation_loss:.4f}, Inference time: {inference_time:.2f} seconds\n')

                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    print(f"  New best validation loss: {best_val_loss:.4f}")
                    torch.save(ddim.state_dict(), os.path.join(save_dir, "model.pth"))
                    print(f'  Saved model at {os.path.join(save_dir, "model.pth")}')
                    with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
                        f.write(f'  New best model saved with validation loss: {best_val_loss:.4f}\n')

            # Save checkpoint every 200 epochs
            torch.save(ddim.state_dict(), os.path.join(save_dir, 'models_x_epoch', f'model_epoch_{ep}.pth'))

    # ── Save loss histories and plot ─────────────────────────────────────────
    import matplotlib.pyplot as plt

    np.save(os.path.join(save_dir, 'train_loss_history.npy'), np.array(train_loss_history))
    np.save(os.path.join(save_dir, 'val_loss_history.npy'), np.array(val_loss_history))
    np.save(os.path.join(save_dir, 'val_epochs.npy'), np.array(val_epochs))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(n_epoch), train_loss_history, label='Training Loss (EMA)', alpha=0.8)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title(f'DDIM Measured (1.272 GHz) — N={n_train_samples}, W={n_feat}')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(val_epochs, val_loss_history, 'r-o', label='Validation Loss', markersize=4)
    ax2.set_ylabel('Validation MSE Loss')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {os.path.join(save_dir, 'loss_curves.png')}")


if __name__ == "__main__":
    import sys
    incremental = "--incremental" in sys.argv
    seed = 0
    n_feat = 256
    argv_filtered = []
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--incremental":
            i += 1
            continue
        if sys.argv[i] == "--seed" and i + 1 < len(sys.argv):
            seed = int(sys.argv[i + 1])
            i += 2
            continue
        if sys.argv[i] == "--w" and i + 1 < len(sys.argv):
            n_feat = int(sys.argv[i + 1])
            i += 2
            continue
        argv_filtered.append(sys.argv[i])
        i += 1
    n_samples = int(argv_filtered[0]) if argv_filtered else 1000
    mode_str = "INCREMENTAL (nested subsets)" if incremental else "INDEPENDENT (separate shuffle per size)"
    print(f"Training on DICHASUS measured data: N={n_samples}, W={n_feat}")
    print(f"Mode: {mode_str}")
    print(f"Seed: {seed}")
    train(n_train_samples=n_samples, n_feat=n_feat, incremental=incremental, seed=seed)
