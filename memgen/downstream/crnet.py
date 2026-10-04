"""CRNet autoencoder for CSI compression.

The architecture follows the reference implementation of

    Z. Lu, J. Wang and J. Song, "Multi-resolution CSI feedback with deep
    learning in massive MIMO system", IEEE ICC 2020,
    https://github.com/Kylin9511/CRNet

and is used here only as a fixed downstream consumer of the generated
channels: it is never tuned per experiment, so differences in test NMSE are
attributable to the training data alone.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["CRNet", "to_tensor", "nmse_db", "train_crnet"]


def to_tensor(h: np.ndarray) -> torch.Tensor:
    """Complex channels ``(N, Nr, Nt)`` -> CRNet input ``(N, 2, Nr, Nt)``.

    Each channel is scaled by its own peak amplitude and mapped to ``[0, 1]``.
    Because the array DFT is unitary, the NMSE computed on these tensors equals
    the beamspace NMSE.
    """
    amplitude = np.abs(h).max(axis=(-2, -1), keepdims=True) + 1e-12
    ri = np.stack([(h / amplitude).real, (h / amplitude).imag], axis=1)
    return torch.from_numpy(((ri + 1.0) / 2.0).astype(np.float32))


def nmse_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Normalised mean-squared reconstruction error in dB (lower is better)."""
    p = pred.float() * 2.0 - 1.0
    t = target.float() * 2.0 - 1.0
    axes = list(range(1, t.ndim))
    num = ((p - t) ** 2).sum(dim=axes)
    den = (t ** 2).sum(dim=axes) + 1e-12
    return 10.0 * torch.log10((num / den).mean()).item()


class ConvBN(nn.Sequential):
    """Convolution with same-padding followed by batch normalisation."""

    def __init__(self, in_planes: int, out_planes: int, kernel_size, stride=1):
        padding = ((kernel_size - 1) // 2 if isinstance(kernel_size, int)
                   else [(k - 1) // 2 for k in kernel_size])
        super().__init__(OrderedDict([
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size,
                               stride=stride, padding=padding, bias=False)),
            ("bn", nn.BatchNorm2d(out_planes)),
        ]))


class CRBlock(nn.Module):
    """Two parallel convolution paths merged by a 1x1 convolution, plus a skip."""

    def __init__(self):
        super().__init__()
        self.path1 = nn.Sequential(
            ConvBN(2, 7, 3), nn.LeakyReLU(0.3, inplace=True),
            ConvBN(7, 7, [1, 9]), nn.LeakyReLU(0.3, inplace=True),
            ConvBN(7, 7, [3, 1]),
        )
        self.path2 = nn.Sequential(
            ConvBN(2, 7, [1, 5]), nn.LeakyReLU(0.3, inplace=True),
            ConvBN(7, 7, [3, 1]),
        )
        self.merge = ConvBN(14, 2, 1)
        self.relu = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, x):
        merged = self.merge(torch.cat([self.path1(x), self.path2(x)], dim=1))
        return self.relu(merged + x)


class CRNet(nn.Module):
    """Autoencoder compressing ``(2, Nr, Nt)`` channels by ``reduction``."""

    def __init__(self, nr: int = 4, nt: int = 32, reduction: int = 4):
        super().__init__()
        self.nr, self.nt = nr, nt
        total = 2 * nr * nt

        self.enc_path1 = nn.Sequential(
            ConvBN(2, 2, 3), nn.LeakyReLU(0.3, inplace=True),
            ConvBN(2, 2, [1, 9]), nn.LeakyReLU(0.3, inplace=True),
            ConvBN(2, 2, [3, 1]),
        )
        self.enc_path2 = ConvBN(2, 2, 3)
        self.enc_merge = nn.Sequential(
            nn.LeakyReLU(0.3, inplace=True), ConvBN(4, 2, 1),
            nn.LeakyReLU(0.3, inplace=True),
        )
        self.enc_fc = nn.Linear(total, total // reduction)
        self.dec_fc = nn.Linear(total // reduction, total)
        self.dec_feature = nn.Sequential(
            ConvBN(2, 2, 5), nn.LeakyReLU(0.3, inplace=True), CRBlock(), CRBlock()
        )
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        z = self.enc_merge(torch.cat([self.enc_path1(x), self.enc_path2(x)], dim=1))
        z = self.enc_fc(z.view(n, -1))
        return self.sigmoid(self.dec_feature(self.dec_fc(z).view(n, 2, self.nr, self.nt)))


def _nmse_over(model: CRNet, loader: DataLoader, device) -> float:
    preds, targets = [], []
    with torch.no_grad():
        for (x,) in loader:
            preds.append(model(x.to(device)).cpu())
            targets.append(x)
    return nmse_db(torch.cat(preds), torch.cat(targets))


def train_crnet(x_train: torch.Tensor, x_val: torch.Tensor, x_test: torch.Tensor,
                tag: str, log_path: Path, ckpt_path: Path, *,
                nr: int = 4, nt: int = 32, reduction: int = 4,
                epochs: int = 500, batch_size: int = 512, lr: float = 1e-3,
                seed: int = 0, device=None, reuse: bool = True,
                extra: dict | None = None) -> dict:
    """Train one CRNet to convergence and report its test NMSE.

    The run is cached in ``log_path``, so re-running an experiment only trains
    the configurations that are still missing.
    """
    log_path, ckpt_path = Path(log_path), Path(ckpt_path)
    if reuse and log_path.exists():
        record = json.loads(log_path.read_text())
        print(f"  [cached] {tag}: {record['test_nmse_db']:.2f} dB")
        return record

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def loader(data, shuffle):
        return DataLoader(TensorDataset(data),
                          batch_size=min(batch_size, max(1, len(data))),
                          shuffle=shuffle, pin_memory=True)

    train_loader, val_loader, test_loader = (
        loader(x_train, True), loader(x_val, False), loader(x_test, False)
    )

    model = CRNet(nr=nr, nt=nt, reduction=reduction).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_loss, best_state, best_epoch = float("inf"), None, 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for (x,) in train_loader:
            x = x.to(device)
            optimiser.zero_grad()
            criterion(model(x), x).backward()
            optimiser.step()
        schedule.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([
                criterion(model(x.to(device)), x.to(device)).item()
                for (x,) in val_loader
            ]))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    record = {
        "tag": tag, "seed": seed, "epochs": epochs, "best_epoch": best_epoch,
        "num_train": len(x_train), "num_val": len(x_val), "num_test": len(x_test),
        "test_nmse_db": _nmse_over(model, test_loader, device),
        "val_nmse_db": _nmse_over(model, val_loader, device),
        "train_time_s": time.time() - started,
        **(extra or {}),
    }
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(record, indent=2))
    print(f"  {tag}: {record['test_nmse_db']:.2f} dB "
          f"(best epoch {best_epoch}/{epochs})")
    return record
