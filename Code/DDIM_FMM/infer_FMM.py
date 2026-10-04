'''
Unconditional Flow Matching Model (FMM) Inference for Wireless Channel Generation
==================================================================================
Based on "Site-Specific MIMO Channel Generation via Diffusion and Flow Matching"
but with UE-location conditioning removed (unconditional generation).

Usage:
    python infer_FMM.py generate
'''

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToTensor
import time
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from upa_beam_distance import compute_upa_peak_beam_distance

# Set the SEED for reproducibility
np.random.seed(0)
torch.manual_seed(0)

train_split_init = 0.0
train_split_end = 200
val_split_init = 0.7
val_split_end = 0.9
test_split_init = 0.9
test_split_end = 1.0

num_samples_2_plot = 100000


# ---------------------------------------------------------------------------
# Model Architecture (Unconditional U-Net for Flow Matching) — must match training
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


class UnetFlowMatching(nn.Module):
    """Unconditional U-Net for Flow Matching: only time embedding, no context."""
    def __init__(self, in_channels, n_feat=256):
        super(UnetFlowMatching, self).__init__()

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
# Flow Matching Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_flow_matching(model, n_sample, shape, device, steps=50):
    """Unconditional flow matching sampling via Euler integration."""
    model.eval()
    C, H, W = shape

    x = torch.randn((n_sample, C, H, W), device=device)
    dt = 1.0 / float(steps)

    for i in range(steps):
        t_val = torch.full((n_sample, 1), float(i) / float(steps), device=device, dtype=torch.float32)
        v = model(x, t_val)
        x = x + v * dt

    return x


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

def generate_H_test(model, flag_frequency, flag_data_split, steps, device, save_dir, batch_size=5000, ws_test=[0.0]):
    list_distances = []
    list_distances_upa = []
    indices_path = save_dir

    hist_bins = 32
    hist_pred = torch.zeros(hist_bins, device=device)
    hist_gt = torch.zeros(hist_bins, device=device)

    print("** TESTING **")
    data_path_LoS = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_3p5GHz_LoS.npz"
    data_path_NLoS = "../../dataset/Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz"

    if flag_data_split == "test":
        dataset_test = CustomSionnaDataset(data_path_LoS, data_path_NLoS, test_split_init, test_split_end, flag_data_split, save_dir, indices_path)
    elif flag_data_split == "train":
        dataset_test = CustomSionnaDataset(data_path_LoS, data_path_NLoS, train_split_init, train_split_end, flag_data_split, save_dir, indices_path)
    elif flag_data_split == "val":
        dataset_test = CustomSionnaDataset(data_path_LoS, data_path_NLoS, val_split_init, val_split_end, flag_data_split, save_dir, indices_path)

    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
    print(f"Number of samples in {flag_data_split} dataloader: {len(dataloader_test.dataset)}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)

    list_times = []
    list_cosine_similarity = []

    model.eval()
    beg_eval_time = time.time()
    with torch.no_grad():
        ccount = 0
        pbar = tqdm(dataloader_test)
        for x_test in pbar:
            x_test = np.transpose(x_test, (0, 2, 3, 1))
            x_test = x_test.to(device)
            sublist_cosine_similarity = [[] for _ in range(batch_size)]

            for jj in range(64):
                start_time = time.time()
                # Unconditional sampling
                x_gen = sample_flow_matching(model, batch_size, (2, 4, 32), device, steps=steps)
                end_time = time.time()

                x_gen_ifft = x_gen
                if jj <= 0:
                    list_times.append(end_time - start_time)
                    res = compute_upa_peak_beam_distance(
                        x_test.cpu().numpy(), x_gen_ifft.cpu().numpy(),
                        N_tx_x=8, N_tx_y=4
                    )
                    list_distances_upa.extend(res.distances.tolist())

                for s in range(batch_size):
                    H_rt_complex = (
                        x_test.cpu().numpy()[s, 0, :, :]
                        + 1j * x_test.cpu().numpy()[s, 1, :, :]
                    )
                    H_gen_complex = (
                        x_gen_ifft.cpu().numpy()[s, 0, :, :]
                        + 1j * x_gen_ifft.cpu().numpy()[s, 1, :, :]
                    )

                    P_rt = np.sum(np.abs(H_rt_complex) ** 2, axis=0)
                    P_gen = np.sum(np.abs(H_gen_complex) ** 2, axis=0)

                    p_rt = P_rt / (np.sum(P_rt) + 1e-12)
                    p_gen = P_gen / (np.sum(P_gen) + 1e-12)

                    cos_sim = cosine_similarity(
                        p_rt.reshape(1, -1),
                        p_gen.reshape(1, -1)
                    )[0, 0]
                    sublist_cosine_similarity[s].append(cos_sim)

            for s in range(batch_size):
                list_cosine_similarity.append(np.mean(sublist_cosine_similarity[s]))

            for s in range(batch_size):
                magnitude_ground_truth = np.sqrt(x_test.cpu().numpy()[s, 0, :, :] ** 2 + x_test.cpu().numpy()[s, 1, :, :] ** 2)
                magnitude_ground_truth_normalized = magnitude_ground_truth / np.max(magnitude_ground_truth)
                index_max = np.unravel_index(np.argmax(magnitude_ground_truth_normalized), magnitude_ground_truth_normalized.shape)

                magnitude_x_gen = np.sqrt(x_gen_ifft.cpu().numpy()[s, 0, :, :] ** 2 + x_gen_ifft.cpu().numpy()[s, 1, :, :] ** 2)
                magnitude_x_gen_normalized = magnitude_x_gen / np.max(magnitude_x_gen)
                index_max_gen = np.unravel_index(np.argmax(magnitude_x_gen_normalized), magnitude_x_gen_normalized.shape)

                peak_index_diff = np.abs((index_max[1] - index_max_gen[1]))
                peak_index_diff = min(peak_index_diff, hist_bins - peak_index_diff)
                list_distances.append(peak_index_diff)
                hist_pred[index_max_gen[1]] += 1
                hist_gt[index_max[1]] += 1

                if ccount % 500 == 0 and flag_data_split == "test":
                    plt.figure(figsize=(10, 5))
                    plt.subplot(2, 1, 1)
                    plt.imshow(magnitude_ground_truth_normalized, cmap='viridis')
                    plt.text(0, -2, f'Max at column {index_max[1]}', color='red', fontsize=12, ha='left', va='top')
                    plt.title(f"Label_H_test_{ccount}")
                    plt.axis('off')
                    plt.subplot(2, 1, 2)
                    plt.imshow(magnitude_x_gen_normalized, cmap='viridis')
                    plt.text(0, -2, f'Max at column {index_max_gen[1]}', color='red', fontsize=12, ha='left', va='top')
                    plt.title(f"H_test_{ccount}")
                    plt.axis('off')
                    plt.savefig(os.path.join(save_dir, f'images/H_test_{ccount}_vs_Label_H_test_{ccount}.png'), dpi=300)
                    plt.close()

                ccount += 1
                if ccount > num_samples_2_plot:
                    break

            print(f"Number of samples in list_cosine_similarity: {len(list_cosine_similarity)}")
            print(f"Number of elements in UPA list_distances: {len(list_distances_upa)}")

            if ccount > num_samples_2_plot:
                print(f"Reached the limit of {num_samples_2_plot} samples. Stopping.")
                break

    print(f"Total evaluation time: {time.time() - beg_eval_time:.2f} seconds")

    if flag_data_split == "test":
        cos_sim_array = np.array(list_cosine_similarity)
        np.save(os.path.join(save_dir, 'images/list_cosine_similarity.npy'), cos_sim_array)

        fig_cos, axes_cos = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.ecdfplot(x=cos_sim_array, ax=axes_cos[0])
        axes_cos[0].set_xlabel('Cosine Similarity')
        axes_cos[0].set_ylabel('Cumulative Probability')
        axes_cos[0].set_title('CDF of Cosine Similarity')
        axes_cos[0].grid(True)
        axes_cos[1].hist(cos_sim_array, bins=50, edgecolor='black')
        axes_cos[1].set_xlabel('Cosine Similarity')
        axes_cos[1].set_ylabel('Frequency')
        axes_cos[1].set_title('Histogram of Cosine Similarity')
        axes_cos[1].grid(True)
        fig_cos.tight_layout()
        fig_cos.savefig(os.path.join(save_dir, 'images/cdf_and_histogram_cosine_similarity.png'), dpi=300, bbox_inches='tight')
        plt.close(fig_cos)

        with open(os.path.join(save_dir, 'images/cosine_similarity_stats.txt'), 'w') as f:
            f.write(f"Mean cosine similarity: {np.mean(cos_sim_array):.6f}\n")
            f.write(f"Std cosine similarity:  {np.std(cos_sim_array):.6f}\n")
            f.write(f"Min cosine similarity:  {np.min(cos_sim_array):.6f}\n")
            f.write(f"Max cosine similarity:  {np.max(cos_sim_array):.6f}\n")

        hist_gt_np = hist_gt.cpu().numpy()
        hist_pred_np = hist_pred.cpu().numpy()
        x_axis = np.arange(hist_bins)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.ecdfplot(x=list_distances, ax=axes[0])
        axes[0].set_xlabel('Peak Index Difference')
        axes[0].set_ylabel('Cumulative Probability')
        axes[0].grid(True)
        axes[1].bar(x_axis, hist_pred_np, label='Predicted Peak Index')
        axes[1].bar(x_axis, hist_gt_np, alpha=0.5, label='Ground Truth Peak Index')
        axes[1].legend()
        axes[1].set_xlabel('Peak Index')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'images/cdf_and_peak_index_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sns.ecdfplot(x=list_distances_upa, ax=axes[0])
        axes[0].set_xlabel('Peak Index Difference')
        axes[0].set_ylabel('Cumulative Probability')
        axes[0].grid(True)
        axes[1].bar(x_axis, hist_pred_np, label='Predicted Peak Index')
        axes[1].bar(x_axis, hist_gt_np, alpha=0.5, label='Ground Truth Peak Index')
        axes[1].legend()
        axes[1].set_xlabel('Peak Index')
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'images/UPA_cdf_and_peak_index_histogram.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

        np.save(os.path.join(save_dir, 'images/UPA_list_distances.npy'), np.array(list_distances_upa))
        np.save(os.path.join(save_dir, 'images/list_distances.npy'), np.array(list_distances))
        np.save(os.path.join(save_dir, 'images/list_times.npy'), np.array(list_times))

        with open(os.path.join(save_dir, 'images/histogram_data.txt'), 'w') as f:
            f.write("Peak Index, Predicted Frequency, Ground Truth Frequency\n")
            for i in range(hist_bins):
                f.write(f"{i}, {hist_pred_np[i]}, {hist_gt_np[i]}\n")


def main(mode):
    """
    Main function.

    Parameters:
    - mode: 'generate' to generate samples and evaluate.
    """
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_feat = 256
    steps = 50
    flag_frequency = 3.5
    flag_data_split = "test"
    save_dir = "./logs/FMM_unconditional_3.5GHz_LoS+NLoS_0.0_200_steps50/"
    ws_test = [0.0]

    if not os.path.exists(os.path.join(save_dir, 'images')):
        os.makedirs(os.path.join(save_dir, 'images'))

    # Read training config
    training_config_path = os.path.join(save_dir, 'training_config.txt')
    if os.path.exists(training_config_path):
        with open(training_config_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('batch_size'):
                    batch_size = int(line.split(':')[1].strip())
                elif line.startswith('n_feat'):
                    n_feat = int(line.split(':')[1].strip())
                elif line.startswith('steps'):
                    steps = int(line.split(':')[1].strip())
                elif line.startswith('flag_frequency'):
                    flag_frequency = float(line.split(':')[1].strip())

    model = UnetFlowMatching(in_channels=2, n_feat=n_feat).to(device)

    model.load_state_dict(torch.load(os.path.join(save_dir, "model.pth"), map_location=torch.device('cpu')))
    print(f"Model loaded from {os.path.join(save_dir, 'model.pth')}")

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size} parameters")

    if mode == 'generate':
        generate_H_test(model, flag_frequency, flag_data_split, steps, device, save_dir, batch_size=batch_size, ws_test=ws_test)
    else:
        print("Invalid mode selected. Choose 'generate'.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_FMM.py <mode>")
        print("Modes: 'generate'")
        sys.exit(1)

    mode = sys.argv[1]
    main(mode)
