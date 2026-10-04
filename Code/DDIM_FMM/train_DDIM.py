'''
Unconditional DDIM Training for Wireless Channel Generation
============================================================
Based on "Site-Specific MIMO Channel Generation via Diffusion and Flow Matching"
but with UE-location conditioning removed (unconditional generation).

This script trains on LoS + NLoS combined data using the middle subcarrier (index 128)
which corresponds to the exact carrier frequency.

Usage:
    python train_DDIM.py
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
# Model Architecture (Unconditional U-Net)
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
        """Unconditional forward: predict noise from noisy input and timestep."""
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)
        noise = torch.randn_like(x)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )

        return self.loss_mse(noise, self.nn_model(x_t, _ts / self.n_T))

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

SUBCARRIER_INDEX = 128  # Middle subcarrier → exact carrier frequency


class CustomSionnaDataset(Dataset):
    def __init__(self, data_path_LoS, data_path_NLoS, percent_start, percent_end, flag_data_split, save_dir, indices_dir):
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

        # Select middle subcarrier (index 128) for exact carrier frequency
        data_LoS = np_array_LoS["combined_array"][:, :, 0, :, 0, SUBCARRIER_INDEX, :]
        data_NLoS = np_array_NLoS["combined_array"][:, :, 0, :, 0, SUBCARRIER_INDEX, :]

        print(f"Number of LoS samples: {data_LoS.shape[0]}")
        print(f"Number of NLoS samples: {data_NLoS.shape[0]}")
        print(f"Total number of samples (LoS + NLoS): {data_LoS.shape[0] + data_NLoS.shape[0]}")
        with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Number of LoS samples: {data_LoS.shape[0]}\n")
            f.write(f"Number of NLoS samples: {data_NLoS.shape[0]}\n")
            f.write(f"Total number of samples (LoS + NLoS): {data_LoS.shape[0] + data_NLoS.shape[0]}\n")
            f.write(f"Subcarrier index used: {SUBCARRIER_INDEX} (middle subcarrier)\n")

        # Handle shuffled indices
        if os.path.exists(os.path.join(self.indices_dir, 'indices_los.npy')):
            print(f"Shuffled indices found in {self.indices_dir}. Loading indices.")
            indices_los = np.load(os.path.join(self.indices_dir, 'indices_los.npy'))
            indices_nlos = np.load(os.path.join(self.indices_dir, 'indices_nlos.npy'))

            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Shuffled indices loaded from {self.indices_dir}\n")

            np.save(os.path.join(self.save_dir, 'indices_los.npy'), indices_los)
            np.save(os.path.join(self.save_dir, 'indices_nlos.npy'), indices_nlos)
        else:
            print(f"No shuffled indices found. Creating and saving them.")
            indices_los = np.arange(data_LoS.shape[0])
            indices_nlos = np.arange(data_NLoS.shape[0])
            np.random.shuffle(indices_los)
            np.random.shuffle(indices_nlos)

            # Save to indices_dir (shared location) so other sizes can reuse them
            np.save(os.path.join(self.indices_dir, 'indices_los.npy'), indices_los)
            np.save(os.path.join(self.indices_dir, 'indices_nlos.npy'), indices_nlos)
            # Also save a copy to save_dir for record-keeping
            np.save(os.path.join(self.save_dir, 'indices_los.npy'), indices_los)
            np.save(os.path.join(self.save_dir, 'indices_nlos.npy'), indices_nlos)

            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Shuffled indices created and saved to {self.indices_dir}\n")

        data_LoS = data_LoS[indices_los]
        data_NLoS = data_NLoS[indices_nlos]

        if self.percent_end > 1.0:
            total_samples = int(self.percent_end)
            n_los = total_samples // 2
            n_nlos = total_samples - n_los
            data_LoS = data_LoS[:n_los]
            data_NLoS = data_NLoS[:n_nlos]

            with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                f.write(f"Using {n_los} LoS and {n_nlos} NLoS for total of {total_samples} samples.\n")

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
                np.save(os.path.join(self.save_dir, 'indices.npy'), indices)

                with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
                    f.write(f"Combined shuffled indices created and saved.\n")

            data = data[indices]

            self.idx_start = int(self.percent_start * self.total_length)
            self.idx_end = int(self.percent_end * self.total_length)
            data = data[self.idx_start:self.idx_end, :, :, :]

        # Extract channel and coordinates (coordinates saved as metadata only)
        H_set = data[:, :, :, 0]  # complex channel, shape: (N, 4, 32)
        coords = data[:, 0, 0, 1:]  # x, y, z coordinates, shape: (N, 3)

        # Save raw spatial-domain data
        array1 = np.stack((np.real(H_set), np.imag(H_set)), axis=1)  # (N, 2, 4, 32)
        array2 = array1.copy()

        np.save(os.path.join(self.save_dir, self.flag_data_split + '.npy'), array1)
        print(f"** {self.flag_data_split} data saved to {os.path.join(self.save_dir, self.flag_data_split + '.npy')}")

        # Transform to beamspace
        for i in range(array1.shape[0]):
            dft_data = upa_to_beamspace(array2[i, 0] + 1j * array2[i, 1], Nrx_x=2, Nrx_y=2, Ntx_x=8, Ntx_y=4)
            array1[i, 0] = np.real(dft_data)
            array1[i, 1] = np.imag(dft_data)

        # Save coordinates as metadata (not used for conditioning)
        np.save(os.path.join(self.save_dir, self.flag_data_split + '_coords.npy'), coords)

        # Normalize per sample and store
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
# Training
# ---------------------------------------------------------------------------

def train(n_train_samples=1000, incremental=False, seed=0):
    # Apply the requested seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_epoch = 3000
    batch_size = 100
    n_T = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_feat = 256
    lrate = 1e-4
    flag_frequency = 3.5

    logs_main_dir = './logs/'
    os.makedirs(logs_main_dir, exist_ok=True)

    day_of_experiment = time.strftime("%d_%m_%Y_%H_%M")
    print(f"Day of experiment: {day_of_experiment}")

    train_split_init = 0.0
    train_split_end = n_train_samples  # Use fixed number of samples (half LoS + half NLoS)
    val_split_init = 0.7
    val_split_end = 0.9

    seed_suffix = f"_seed{seed}" if seed != 0 else ""

    if incremental:
        # Incremental mode: all sizes share the same shuffled indices (nested subsets)
        # 100 ⊂ 200 ⊂ 500 ⊂ 1000 ⊂ 2000 ⊂ 4000 ⊂ 8000
        save_dir = f'{logs_main_dir}DDIM_unconditional_{flag_frequency}GHz_LoS+NLoS_{train_split_init}_{train_split_end}_nT{n_T}_incremental{seed_suffix}/'
        indices_path = f'{logs_main_dir}shared_indices{seed_suffix}/'
        os.makedirs(indices_path, exist_ok=True)
    else:
        # Independent mode: each size creates its own shuffled indices
        save_dir = f'{logs_main_dir}DDIM_unconditional_{flag_frequency}GHz_LoS+NLoS_{train_split_init}_{train_split_end}_nT{n_T}{seed_suffix}/'
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

    # Dataset paths
    data_path_LoS = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"
    data_path_NLoS = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz"

    print("** TRAINING **")
    dataset_train = CustomSionnaDataset(data_path_LoS, data_path_NLoS, train_split_init, train_split_end, "train", save_dir, indices_path)
    print("** VALIDATION **")
    dataset_val = CustomSionnaDataset(data_path_LoS, data_path_NLoS, val_split_init, val_split_end, "val", save_dir, indices_path)

    print(f"Number of samples in training split: {len(dataset_train)}")
    with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
        f.write(f"Number of samples in training split: {len(dataset_train)}\n")

    print(f"Number of samples in validation split: {len(dataset_val)}")
    with open(os.path.join(save_dir, 'training_log.txt'), 'a') as f:
        f.write(f"Number of samples in validation split: {len(dataset_val)}\n")

    # Save training config
    with open(os.path.join(save_dir, 'training_config.txt'), 'w') as f:
        f.write(f"n_epoch: {n_epoch}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"n_T: {n_T}\n")
        f.write(f"device: {device}\n")
        f.write(f"n_feat: {n_feat}\n")
        f.write(f"lrate: {lrate}\n")
        f.write(f"train_split_init: {train_split_init}\n")
        f.write(f"train_split_end: {train_split_end}\n")
        f.write(f"val_split_init: {val_split_init}\n")
        f.write(f"val_split_end: {val_split_end}\n")
        f.write(f"data_path_LoS: {data_path_LoS}\n")
        f.write(f"data_path_NLoS: {data_path_NLoS}\n")
        f.write(f"indices_path: {indices_path}\n")
        f.write(f"flag_frequency: {flag_frequency}\n")
        f.write(f"model_size: {model_size}\n")
        f.write(f"subcarrier_index: {SUBCARRIER_INDEX}\n")
        f.write(f"conditioning: unconditional\n")

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
            x = np.transpose(x, (0, 2, 3, 1))
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

                    # Generate unconditional samples
                    x_gen = ddim.sample(x_val.shape[0], (2, 4, 32), device)

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
    ax1.set_title('DDIM Unconditional — Training & Validation Loss')
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
        argv_filtered.append(sys.argv[i])
        i += 1
    n_samples = int(argv_filtered[0]) if argv_filtered else 1000
    mode_str = "INCREMENTAL (nested subsets)" if incremental else "INDEPENDENT (separate shuffle per size)"
    print(f"Training with {n_samples} samples ({n_samples//2} LoS + {n_samples//2} NLoS)")
    print(f"Mode: {mode_str}")
    print(f"Seed: {seed}")
    train(n_train_samples=n_samples, incremental=incremental, seed=seed)
