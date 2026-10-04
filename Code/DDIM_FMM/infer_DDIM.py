'''
Unconditional DDIM Inference for Wireless Channel Generation
=============================================================
Based on "Site-Specific MIMO Channel Generation via Diffusion and Flow Matching"
but with UE-location conditioning removed (unconditional generation).

Usage:
    python infer_DDIM.py generate <n_train_samples>
    
Examples:
    python infer_DDIM.py generate 1000
    python infer_DDIM.py generate 500
'''

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import scipy
import matplotlib.cm as cm
import time
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from upa_beam_distance import compute_upa_peak_beam_distance

# Set the SEED for reproducibility (overridden by --seed argument)
DEFAULT_SEED = 0
np.random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)

train_split_init = 0.0
train_split_end = 1000  # Must match training config (500 LoS + 500 NLoS)
val_split_init = 0.7
val_split_end = 0.9
test_split_init = 0.9
test_split_end = 1.0

num_samples_to_generate = 5000  # Generate 5000 synthetic channels


# ---------------------------------------------------------------------------
# Model Architecture (Unconditional U-Net) — must match training
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

SUBCARRIER_INDEX = 128


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

        data_LoS = np_array_LoS["combined_array"][:, :, 0, :, 0, SUBCARRIER_INDEX, :]
        data_NLoS = np_array_NLoS["combined_array"][:, :, 0, :, 0, SUBCARRIER_INDEX, :]

        print(f"Number of LoS samples: {data_LoS.shape[0]}")
        print(f"Number of NLoS samples: {data_NLoS.shape[0]}")
        print(f"Total number of samples (LoS + NLoS): {data_LoS.shape[0] + data_NLoS.shape[0]}")
        with open(os.path.join(self.save_dir, 'training_log.txt'), 'a') as f:
            f.write(f"Number of LoS samples: {data_LoS.shape[0]}\n")
            f.write(f"Number of NLoS samples: {data_NLoS.shape[0]}\n")
            f.write(f"Total number of samples (LoS + NLoS): {data_LoS.shape[0] + data_NLoS.shape[0]}\n")

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
        print(f"** {self.flag_data_split} data saved to {os.path.join(self.save_dir, self.flag_data_split + '.npy')}")

        for i in range(array1.shape[0]):
            dft_data = upa_to_beamspace(array2[i, 0] + 1j * array2[i, 1], Nrx_x=2, Nrx_y=2, Ntx_x=8, Ntx_y=4)
            array1[i, 0] = np.real(dft_data)
            array1[i, 1] = np.imag(dft_data)

        np.save(os.path.join(self.save_dir, self.flag_data_split + '_coords.npy'), coords)

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
# Inference / Generation
# ---------------------------------------------------------------------------

def generate_and_save(ddim, device, save_dir, n_generate=5000, batch_size=100):
    """
    Generate n_generate unconditional synthetic channels and save them.
    Also load and save the test ground truth for later evaluation.
    No cosine similarity computation (not meaningful for unconditional DDIM).
    """
    indices_path = save_dir
    output_dir = os.path.join(save_dir, 'generated_channels')
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Generate synthetic channels ---
    print(f"** Generating {n_generate} unconditional synthetic channels **")
    ddim.eval()
    all_generated = []

    n_batches = (n_generate + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(n_batches), desc="Generating"):
            current_batch = min(batch_size, n_generate - len(all_generated))
            x_gen = ddim.sample(current_batch, (2, 4, 32), device)
            all_generated.append(x_gen.cpu().numpy())

    generated_channels = np.concatenate(all_generated, axis=0)[:n_generate]
    print(f"  Generated shape: {generated_channels.shape}")  # (5000, 2, 4, 32)

    # Save generated channels
    gen_path = os.path.join(output_dir, 'generated_beamspace.npz')
    np.savez_compressed(gen_path, channels=generated_channels)
    print(f"  Saved generated channels to {gen_path}")

    # --- 2. Load and save test ground truth (from test split, NOT training data) ---
    print("** Loading test ground truth (test split: 90%-100% of data) **")
    data_path_LoS = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz"
    data_path_NLoS = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz"

    dataset_test = CustomSionnaDataset(
        data_path_LoS, data_path_NLoS,
        test_split_init, test_split_end,
        "test", save_dir, indices_path
    )

    # Collect all test data as beamspace tensors
    test_channels = []
    for idx in range(len(dataset_test)):
        item = dataset_test[idx]  # ToTensor permutes (2,4,32) → (32,2,4)
        test_channels.append(item.numpy())
    test_channels = np.array(test_channels)
    # Undo ToTensor permutation: (N, 32, 2, 4) → (N, 2, 4, 32)
    test_channels = np.transpose(test_channels, (0, 2, 3, 1))
    print(f"  Test ground truth shape: {test_channels.shape}")

    gt_path = os.path.join(output_dir, 'test_ground_truth_beamspace.npz')
    np.savez_compressed(gt_path, channels=test_channels)
    print(f"  Saved test ground truth to {gt_path}")

    # --- 3. Save data split documentation ---
    n_train = int(train_split_end)
    doc_path = os.path.join(output_dir, 'data_split_info.txt')
    with open(doc_path, 'w') as f:
        f.write("=== Data Split Documentation ===\n")
        f.write(f"Training: {n_train} samples ({n_train//2} LoS + {n_train//2} NLoS, absolute count)\n")
        f.write(f"Validation: {val_split_init*100:.0f}% - {val_split_end*100:.0f}% of total data\n")
        f.write(f"Test: {test_split_init*100:.0f}% - {test_split_end*100:.0f}% of total data\n")
        f.write(f"\nGenerated samples: {n_generate}\n")
        f.write(f"Test ground truth samples: {len(dataset_test)}\n")
        f.write(f"\nIMPORTANT: Generated channels are unconditional (no location conditioning).\n")
        f.write(f"The test ground truth is from the HELD-OUT test split (no overlap with training).\n")
        f.write(f"\nFiles:\n")
        f.write(f"  generated_beamspace.npz — shape: {generated_channels.shape}, key: 'channels'\n")
        f.write(f"  test_ground_truth_beamspace.npz — shape: {test_channels.shape}, key: 'channels'\n")
        f.write(f"\nFormat: (N, 2, Nr=4, Nt=32) — axis 1: [real, imag] of beamspace channel\n")
        f.write(f"Normalization: per-sample max-magnitude normalized\n")
    print(f"  Documentation saved to {doc_path}")

    # --- 4. Quick visual comparison ---
    print("** Creating visual comparison plots **")
    images_dir = os.path.join(save_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    n_plot = min(5, len(test_channels))
    fig, axes = plt.subplots(n_plot, 2, figsize=(12, 3*n_plot))
    for i in range(n_plot):
        mag_gt = np.sqrt(test_channels[i, 0]**2 + test_channels[i, 1]**2)
        mag_gen = np.sqrt(generated_channels[i, 0]**2 + generated_channels[i, 1]**2)

        axes[i, 0].imshow(mag_gt / (np.max(mag_gt) + 1e-12), cmap='viridis', aspect='auto')
        axes[i, 0].set_title(f'Ground Truth #{i}')
        axes[i, 0].set_ylabel('Rx beam')
        axes[i, 1].imshow(mag_gen / (np.max(mag_gen) + 1e-12), cmap='viridis', aspect='auto')
        axes[i, 1].set_title(f'Generated #{i}')
    axes[-1, 0].set_xlabel('Tx beam')
    axes[-1, 1].set_xlabel('Tx beam')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'generated_vs_ground_truth.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("** Inference complete! **")
    print(f"   Generated channels: {gen_path}")
    print(f"   Ground truth:       {gt_path}")


def main(mode, n_train_samples=1000, incremental=False, seed=0):
    """
    Main function.

    Parameters:
    - mode: 'generate' to generate 5000 samples and evaluate.
    - n_train_samples: number of training samples used (to locate the correct model directory).
    - incremental: if True, use the incremental (nested subset) folder naming.
    - seed: random seed used during training (to locate the correct model directory).
    """
    # Apply the requested seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    global train_split_end
    train_split_end = n_train_samples

    batch_size = 100
    n_T = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_feat = 256
    flag_frequency = 3.5
    seed_suffix = f"_seed{seed}" if seed != 0 else ""
    if incremental:
        save_dir = f"./logs/DDIM_unconditional_{flag_frequency}GHz_LoS+NLoS_0.0_{n_train_samples}_nT200_incremental{seed_suffix}/"
    else:
        save_dir = f"./logs/DDIM_unconditional_{flag_frequency}GHz_LoS+NLoS_0.0_{n_train_samples}_nT200{seed_suffix}/"

    os.makedirs(save_dir, exist_ok=True)

    # Read training config to get parameters
    training_config_path = os.path.join(save_dir, 'training_config.txt')
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('batch_size'):
                    batch_size = int(line.split(':')[1].strip())
                elif line.startswith('n_feat'):
                    n_feat = int(line.split(':')[1].strip())
                elif line.startswith('n_T'):
                    n_T = int(line.split(':')[1].strip())
                elif line.startswith('flag_frequency'):
                    flag_frequency = float(line.split(':')[1].strip())

    # Initialize unconditional model
    ddim = DDIM(
        nn_model=Unet(in_channels=2, n_feat=n_feat),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
    )
    ddim.to(device)

    # Load the trained model
    model_path = os.path.join(save_dir, "model.pth")
    ddim.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")

    model_size = sum(p.numel() for p in ddim.parameters() if p.requires_grad)
    print(f"Model size: {model_size} parameters")

    if mode == 'generate':
        generate_and_save(ddim, device, save_dir, n_generate=num_samples_to_generate, batch_size=batch_size)
    else:
        print("Invalid mode. Use 'generate'.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer_DDIM.py generate <n_train_samples> [--incremental] [--seed <N>]")
        print("Example: python infer_DDIM.py generate 1000")
        print("Example: python infer_DDIM.py generate 200 --incremental --seed 42")
        sys.exit(1)

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
    mode = argv_filtered[0]
    n_train = int(argv_filtered[1]) if len(argv_filtered) > 1 else 1000
    mode_str = "INCREMENTAL" if incremental else "INDEPENDENT"
    print(f"Inference for model trained on {n_train} samples [{mode_str}] (seed={seed})")
    main(mode, n_train_samples=n_train, incremental=incremental, seed=seed)
