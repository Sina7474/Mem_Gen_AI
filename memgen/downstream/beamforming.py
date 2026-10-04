"""Learned probing beams and beam synthesis for the beam-alignment task.

The receiver and transmitter sound the channel with a few unit-modulus probing
beams, the measured beam powers are fed back, and a small network synthesises
the transmit and receive beams used for data transmission. Probing codebooks
and synthesiser are trained jointly to maximise beamforming gain, so the
probing directions adapt to the propagation of the site being modelled.

This is the downstream consumer used to answer a single question: do channels
sampled from the diffusion model train a beam predictor as well as real ones?
The method follows

    Y. Heng, J. Mo and J. G. Andrews, "Learning site-specific probing beams for
    fast mmWave beam alignment", IEEE Trans. Wireless Commun., 2022,

restricted to the configuration used in the paper: learned transmit and
receive probing, diagonal power feedback, and an MLP beam synthesiser.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "db_to_pow",
    "pow_to_db",
    "AnalogProbing",
    "BeamSynthesiser",
    "BeamAlignmentNet",
    "beamforming_gain",
    "beamforming_gain_numpy",
]


def db_to_pow(x):
    """Decibels to linear power."""
    return 10.0 ** (np.asarray(x, dtype=np.float64) / 10.0)


def pow_to_db(x):
    """Linear power to decibels."""
    return 10.0 * np.log10(np.asarray(x, dtype=np.float64))


def _unit_modulus(real: torch.Tensor, imag: torch.Tensor,
                  n_antenna: int) -> torch.Tensor:
    """Project onto the per-element constant-modulus set of an analog array.

    A phase shifter can only rotate, not scale, so every entry is forced to
    magnitude ``1/sqrt(n_antenna)``; the array then has unit total power.
    """
    weights = torch.complex(real, imag)
    return weights / torch.abs(weights) / math.sqrt(n_antenna)


class AnalogProbing(nn.Module):
    """Learned transmit and receive probing codebooks with a noisy read-out.

    Args:
        n_rx: receive antennas.
        n_tx: transmit antennas.
        n_probe: probing beams per side.
        noise_power: measurement noise power, relative to transmit power.
        norm_factor: scale the channels were divided by before training, so
            that the injected noise keeps its physical meaning.
    """

    def __init__(self, n_rx: int, n_tx: int, n_probe: int,
                 noise_power: float, norm_factor: float = 1.0):
        super().__init__()
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.n_probe = n_probe
        self.register_buffer("noise_power", torch.tensor(float(noise_power)))
        self.register_buffer("norm_factor", torch.tensor(float(norm_factor)))

        self.tx_real = nn.Parameter(torch.empty(n_tx, n_probe))
        self.tx_imag = nn.Parameter(torch.empty(n_tx, n_probe))
        self.rx_real = nn.Parameter(torch.empty(n_rx, n_probe))
        self.rx_imag = nn.Parameter(torch.empty(n_rx, n_probe))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (self.tx_real, self.tx_imag, self.rx_real, self.rx_imag):
            nn.init.uniform_(weight, -1.0, 1.0)

    def tx_codebook(self) -> torch.Tensor:
        return _unit_modulus(self.tx_real, self.tx_imag, self.n_tx)

    def rx_codebook(self) -> torch.Tensor:
        return _unit_modulus(self.rx_real, self.rx_imag, self.n_rx)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sound ``h`` with every probing pair.

        Thermal noise enters at the receive antennas, before analog combining,
        so the combiner shapes the noise exactly as it shapes the signal.

        Args:
            h: channels ``(B, n_rx, n_tx)``, already divided by ``norm_factor``.

        Returns:
            Noiseless and noisy measurements, both ``(B, n_probe, n_probe)``.
        """
        received = h @ self.tx_codebook()            # (B, n_rx, n_probe)

        sigma = torch.sqrt(self.noise_power / 2.0) / self.norm_factor
        noise = torch.complex(torch.randn_like(received.real),
                              torch.randn_like(received.imag)) * sigma

        rx = self.rx_codebook().conj().transpose(0, 1)
        return rx @ received, rx @ (received + noise)


class BeamSynthesiser(nn.Module):
    """Map fed-back beam powers to one unit-modulus analog beam."""

    def __init__(self, n_measurement: int, n_antenna: int):
        super().__init__()
        self.n_antenna = n_antenna
        hidden = 5 * n_antenna
        self.net = nn.Sequential(
            nn.Linear(n_measurement, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * n_antenna),
        )

    def forward(self, powers: torch.Tensor) -> torch.Tensor:
        out = self.net(powers)
        return _unit_modulus(out[:, :self.n_antenna], out[:, self.n_antenna:],
                             self.n_antenna)


class BeamAlignmentNet(nn.Module):
    """Probe, feed back, and synthesise the transmit and receive beams.

    With equal numbers of transmit and receive probing beams the receiver only
    reports the diagonal of the measurement matrix, i.e. one power per probing
    pair, which is the cheapest feedback the architecture supports.
    """

    def __init__(self, n_rx: int, n_tx: int, n_probe: int,
                 noise_power: float, norm_factor: float = 1.0):
        super().__init__()
        self.probing = AnalogProbing(n_rx, n_tx, n_probe,
                                     noise_power, norm_factor)
        self.tx_head = BeamSynthesiser(n_probe, n_tx)
        self.rx_head = BeamSynthesiser(n_probe, n_rx)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, measured = self.probing(h)
        powers = torch.diagonal(torch.abs(measured) ** 2, dim1=1, dim2=2)
        return self.tx_head(powers), self.rx_head(powers)

    def probing_codebooks(self) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            return (self.probing.tx_codebook().cpu().numpy(),
                    self.probing.rx_codebook().cpu().numpy())


def beamforming_gain(h: torch.Tensor, tx_beam: torch.Tensor,
                     rx_beam: torch.Tensor) -> torch.Tensor:
    """``|rx^H h tx|^2`` for a batch of channels and beam pairs."""
    y = rx_beam.conj().unsqueeze(1) @ h @ tx_beam.unsqueeze(-1)
    return torch.abs(y.reshape(-1)) ** 2


def beamforming_gain_numpy(h: np.ndarray, tx_beam: np.ndarray,
                           rx_beam: np.ndarray) -> np.ndarray:
    """``|rx^H h tx|^2`` evaluated on the raw, unnormalised channels."""
    y = rx_beam.conj()[:, None, :] @ h @ tx_beam[:, :, None]
    return (np.abs(y.reshape(-1)) ** 2).astype(np.float64)
