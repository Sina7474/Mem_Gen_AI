"""
crnet_core.py — CRNet model + training core for CSI compression
================================================================
Copied and adapted from the reference implementation:
    https://github.com/Kylin9511/CRNet
Original CRNet architecture:
    W. Liu et al., "CRNet: An Efficient Multi-task Learning Architecture for
    Large-Scale MIMO CSI Feedback", IEEE TCCN, 2022.
    https://github.com/Kylin9511/CRNet

This module contains the reusable, task-agnostic pieces:
    - UPA beamspace <-> antenna transforms (unitary 2-D DFT)
    - Preprocessing to CRNet tensor format (per-sample max-amp normalization)
    - NMSE (dB) metric
    - CRNet autoencoder (ConvBN / CRBlock / CRNet)
    - train_crnet(): trains one fresh model, returns metrics dict

The Task-09-specific experiment orchestration lives in run_task09.py.
"""

import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# UPA / BEAMSPACE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def upa_dft_codebook(Nx: int, Ny: int) -> np.ndarray:
    """Unitary 2-D DFT codebook for a UPA with Nx x Ny elements.
    Returns (Nx*Ny, Nx*Ny) = kron(F_Nx, F_Ny), where F_N = DFT_N / sqrt(N).

    NOTE: With (ntx_x=4, ntx_y=8) this yields kron(F_4, F_8), which is
    identical to the DDIM training convention upa_dft_codebook(8,4)=kron(dft(4),
    dft(8)). Likewise (nrx_x=2, nrx_y=2) gives kron(F_2, F_2). The generated
    beamspace channels therefore round-trip consistently through these codebooks.
    """
    Fx = np.fft.fft(np.eye(Nx, dtype=np.complex64), axis=0) / np.sqrt(Nx)
    Fy = np.fft.fft(np.eye(Ny, dtype=np.complex64), axis=0) / np.sqrt(Ny)
    return np.kron(Fx, Fy).astype(np.complex64)


def beamspace_to_antenna(Hv: np.ndarray, nrx_x: int, nrx_y: int,
                         ntx_x: int, ntx_y: int) -> np.ndarray:
    """Inverse UPA beamspace transform: H = Ar @ Hv @ At^H.
    Hv: (..., Nr, Nt) complex. Returns same shape in antenna domain.
    """
    Ar = upa_dft_codebook(nrx_x, nrx_y)
    At = upa_dft_codebook(ntx_x, ntx_y)
    return (Ar @ Hv) @ At.conj().T


def antenna_to_beamspace(H: np.ndarray, nrx_x: int, nrx_y: int,
                         ntx_x: int, ntx_y: int) -> np.ndarray:
    """Forward UPA beamspace transform: Hv = Ar^H @ H @ At.
    H: (..., Nr, Nt) complex. Returns same shape in beamspace.
    """
    Ar = upa_dft_codebook(nrx_x, nrx_y)
    At = upa_dft_codebook(ntx_x, ntx_y)
    return (Ar.conj().T @ H) @ At


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING  (N, Nr, Nt) complex → (N, 2, Nr, Nt) float32 in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────
def to_tensor(H: np.ndarray) -> torch.Tensor:
    """Per-sample max-amplitude normalization, real/imag stacking, mapping to [0, 1].

    Because the UPA DFT is unitary, NMSE computed on these tensors equals the
    beamspace NMSE ||Hv_pred - Hv||^2 / ||Hv||^2.
    """
    amp = np.abs(H).max(axis=(-2, -1), keepdims=True) + 1e-12
    Hn = H / amp
    ri = np.stack([Hn.real, Hn.imag], axis=1).astype(np.float32)
    return torch.from_numpy((ri + 1.0) / 2.0)


def nmse_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    """NMSE in dB. Both tensors are max-normalized in [0, 1].
    Equivalent to beamspace NMSE due to the unitary UPA DFT. Lower is better.
    """
    p = pred.float() * 2.0 - 1.0
    t = target.float() * 2.0 - 1.0
    num = ((p - t) ** 2).sum(dim=list(range(1, t.ndim)))
    den = (t ** 2).sum(dim=list(range(1, t.ndim))) + 1e-12
    return 10.0 * torch.log10((num / den).mean()).item()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class ConvBN(nn.Sequential):
    """Conv2d + BatchNorm2d with auto same-padding."""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1):
        padding = (
            [(k - 1) // 2 for k in kernel_size]
            if not isinstance(kernel_size, int)
            else (kernel_size - 1) // 2
        )
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size,
                               stride=stride, padding=padding, bias=False)),
            ("bn", nn.BatchNorm2d(out_planes)),
        ]))


class CRBlock(nn.Module):
    """CRBlock: two parallel paths merged with 1x1 conv + residual shortcut."""
    def __init__(self):
        super().__init__()
        self.path1 = nn.Sequential(
            ConvBN(2, 7, 3),          nn.LeakyReLU(0.3, inplace=True),
            ConvBN(7, 7, [1, 9]),     nn.LeakyReLU(0.3, inplace=True),
            ConvBN(7, 7, [3, 1]),
        )
        self.path2 = nn.Sequential(
            ConvBN(2, 7, [1, 5]),     nn.LeakyReLU(0.3, inplace=True),
            ConvBN(7, 7, [3, 1]),
        )
        self.merge = ConvBN(14, 2, 1)
        self.relu = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, x):
        return self.relu(self.merge(torch.cat([self.path1(x), self.path2(x)], dim=1)) + x)


class CRNet(nn.Module):
    """CRNet autoencoder for (2, Nr, Nt) channel tensors.

    Encoder: dual-path conv → 1x1 merge → FC bottleneck (latent = total // reduction)
    Decoder: FC expand → 5x5 conv → 2x CRBlock → sigmoid
    """
    def __init__(self, nr: int = 4, nt: int = 32, reduction: int = 4):
        super().__init__()
        total = 2 * nr * nt
        self.nr, self.nt = nr, nt

        self.enc_path1 = nn.Sequential(
            ConvBN(2, 2, 3),        nn.LeakyReLU(0.3, inplace=True),
            ConvBN(2, 2, [1, 9]),   nn.LeakyReLU(0.3, inplace=True),
            ConvBN(2, 2, [3, 1]),
        )
        self.enc_path2 = ConvBN(2, 2, 3)
        self.enc_merge = nn.Sequential(
            nn.LeakyReLU(0.3, inplace=True),
            ConvBN(4, 2, 1),
            nn.LeakyReLU(0.3, inplace=True),
        )
        self.enc_fc = nn.Linear(total, total // reduction)
        self.dec_fc = nn.Linear(total // reduction, total)
        self.dec_feature = nn.Sequential(
            ConvBN(2, 2, 5), nn.LeakyReLU(0.3, inplace=True),
            CRBlock(), CRBlock(),
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        z = self.enc_merge(
            torch.cat([self.enc_path1(x), self.enc_path2(x)], dim=1)
        )
        z = self.enc_fc(z.view(N, -1))
        return self.sigmoid(self.dec_feature(self.dec_fc(z).view(N, 2, self.nr, self.nt)))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_crnet(
    X_train: torch.Tensor,
    X_val: torch.Tensor,
    X_test: torch.Tensor,
    tag: str,
    log_path: Path,
    ckpt_path: Path,
    nr: int = 4,
    nt: int = 32,
    reduction: int = 4,
    epochs: int = 500,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device = None,
    skip_done: bool = True,
    extra_meta: dict = None,
) -> dict:
    """Train a fresh CRNet and return a metrics dict. Results cached to log_path.

    Returns dict with keys: test_nmse_db, val_nmse_db, best_epoch, n_train,
    n_val, n_test, train_time_s (plus any extra_meta).
    """
    if skip_done and log_path.exists():
        d = json.loads(log_path.read_text())
        print(f"  [CACHED] {tag}: test={d['test_nmse_db']:.2f} dB")
        return d

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loader(ds, shuffle):
        return DataLoader(TensorDataset(ds), batch_size=batch_size,
                          shuffle=shuffle, pin_memory=True, num_workers=2)

    tr_ld = make_loader(X_train, True)
    vl_ld = make_loader(X_val, False)
    te_ld = make_loader(X_test, False)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = CRNet(nr=nr, nt=nt, reduction=reduction).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    crit = nn.MSELoss()

    best_val, best_state, best_epoch = float("inf"), None, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        for (x,) in tr_ld:
            x = x.to(device)
            opt.zero_grad()
            crit(model(x), x).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vl = sum(crit(model(x.to(device)), x.to(device)).item()
                     for (x,) in vl_ld) / len(vl_ld)

        if vl < best_val:
            best_val = vl
            best_epoch = ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        sch.step()

        if ep % 50 == 0:
            print(f"    [{tag}] ep {ep:4d}/{epochs}  val_mse={vl:.6f}  "
                  f"elapsed={time.time()-t0:.0f}s")

    model.load_state_dict(best_state)
    model.eval()

    # Test NMSE
    preds, targets = [], []
    with torch.no_grad():
        for (x,) in te_ld:
            preds.append(model(x.to(device)).cpu())
            targets.append(x)
    test_nmse = nmse_db(torch.cat(preds), torch.cat(targets))

    # Validation NMSE (dB) at best model
    vpreds, vtargets = [], []
    with torch.no_grad():
        for (x,) in vl_ld:
            vpreds.append(model(x.to(device)).cpu())
            vtargets.append(x)
    val_nmse = nmse_db(torch.cat(vpreds), torch.cat(vtargets))

    elapsed = time.time() - t0
    print(f"  [{tag}] Test NMSE: {test_nmse:.2f} dB  Val NMSE: {val_nmse:.2f} dB  "
          f"(best_ep={best_epoch}, time={elapsed:.0f}s)")

    torch.save(best_state, str(ckpt_path))
    meta = {
        "tag": tag, "n_train": len(X_train), "n_val": len(X_val),
        "n_test": len(X_test), "reduction": reduction, "epochs": epochs,
        "seed": seed, "best_epoch": best_epoch,
        "test_nmse_db": test_nmse, "val_nmse_db": val_nmse,
        "train_time_s": elapsed,
    }
    if extra_meta:
        meta.update(extra_meta)
    log_path.write_text(json.dumps(meta, indent=2))
    return meta
