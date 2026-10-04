"""Beam alignment: does synthetic data help a site-specific beam predictor?

A learned probing/beam-synthesis autoencoder (DL-GF) observes a handful of
probing measurements and predicts the transmit and receive beams. Exactly as in
:mod:`memgen.downstream.csi_compression`, the reference arm trains on the ``N``
real channels the DDIM saw, the augmented arm tops them up with channels
sampled from a DDIM checkpoint, and the resulting beamforming SNR is reported
against that checkpoint.

Scale handling follows the reference DL-GF pipeline: training channels are
Frobenius-normalised per sample whenever real and synthetic channels are mixed,
the network input is divided by the training max-abs, and the beamforming gain
is always evaluated on the raw physical test channels so the reported SNR stays
meaningful.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .. import config as cfg
from ..beamspace import to_antenna, upa_dft_codebook
from ..datasets import load_pool
from ..model import build_ddim
from ..sampling import cached_samples, list_checkpoints, load_checkpoint
from .dlgf import BF_loss, Joint_BF_Autoencoder, dB_2_pow, eval_model, pow_2_dB

__all__ = ["System", "run", "baselines"]

COLUMNS = [
    "experiment", "N", "tau", "n_probe", "seed", "num_real", "num_synthetic",
    "snr_db", "rate_bps_hz",
]


class System:
    """Link budget shared by every beam-alignment experiment."""

    def __init__(self, tx_power_dbm: float = 20.0, bandwidth_mhz: float = 100.0,
                 noise_psd_dbm_hz: float = -161.0, measurement_gain: float = 16.0):
        self.tx_power_dbm = tx_power_dbm
        self.noise_power_dbm = noise_psd_dbm_hz + 10.0 * np.log10(bandwidth_mhz * 1e6)
        self.noise_lin = dB_2_pow(self.noise_power_dbm - tx_power_dbm)
        self.measurement_noise = self.noise_lin / measurement_gain


def _frobenius_normalise(h: np.ndarray) -> np.ndarray:
    norm = np.sqrt((np.abs(h) ** 2).sum(axis=(1, 2), keepdims=True))
    return (h / np.where(norm == 0, 1.0, norm)).astype(np.complex64)


def _build_model(n_rx: int, n_tx: int, n_probe: int, system: System,
                 norm_factor: float) -> Joint_BF_Autoencoder:
    """Fully learned Tx/Rx probing with diagonal feedback and an MLP synthesiser."""
    return Joint_BF_Autoencoder(
        num_antenna_Tx=n_tx, num_antenna_Rx=n_rx,
        num_probing_beam_Tx=n_probe, num_probing_beam_Rx=n_probe,
        noise_power=system.measurement_noise, norm_factor=norm_factor,
        feedback="diagonal", num_feedback=None,
        learned_probing="TxRx", beam_synthesizer="MLP",
    )


def train_predictor(h_train: np.ndarray, n_probe: int, system: System, *,
                    epochs: int = 1_000, batch_size: int = 256,
                    lr: float = 1e-3, seed: int = 0, device=None):
    """Train one beam predictor; returns ``(state_dict, norm_factor)``."""
    device = device or cfg.DEVICE
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    h_train = np.asarray(h_train, dtype=np.complex64)[rng.permutation(len(h_train))]

    norm_factor = float(np.abs(h_train).max())
    x = torch.from_numpy((h_train / norm_factor).astype(np.complex64)).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x),
        batch_size=min(batch_size, len(x)), shuffle=True)

    model = _build_model(h_train.shape[1], h_train.shape[2], n_probe,
                         system, norm_factor).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    loss_fn = BF_loss(noise_power_dBm=system.noise_power_dbm,
                      Tx_power_dBm=system.tx_power_dbm)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimiser.zero_grad()
            tx_beam, rx_beam, _ = model(batch)
            loss_fn(batch, tx_beam, rx_beam).backward()
            optimiser.step()
    return {k: v.cpu() for k, v in model.state_dict().items()}, norm_factor


def evaluate_predictor(state_dict, n_probe: int, h_test: np.ndarray,
                       system: System, frobenius: bool,
                       seed: int = 0) -> tuple[float, float]:
    """Average SNR (dB) and achievable rate on the raw physical test channels."""
    model = _build_model(h_test.shape[1], h_test.shape[2], n_probe, system, 1.0)
    model.load_state_dict(state_dict)
    model.eval()
    norm_factor = model.joint_beamformer.norm_factor.item()

    h_in = _frobenius_normalise(h_test) if frobenius else h_test.astype(np.complex64)
    np.random.seed(seed)
    with torch.no_grad():
        gain, _ = eval_model(model, torch.from_numpy(h_in / norm_factor), h_test,
                             noise_power=system.measurement_noise,
                             prediction_mode="GF", feedback_mode="diagonal")
    snr_lin = gain / system.noise_lin
    return float(np.mean(pow_2_dB(snr_lin))), float(np.mean(np.log2(1.0 + snr_lin)))


def baselines(h_test: np.ndarray, system: System,
              spec: cfg.DatasetSpec) -> dict[str, float]:
    """Matched-filter upper bound and genie-aided DFT beam selection, in dB."""
    singular = np.linalg.svd(h_test, compute_uv=False)
    mrt_mrc = pow_2_dB((singular[:, 0] ** 2) / system.noise_lin)

    cb_rx = upa_dft_codebook(*spec.rx_array)
    cb_tx = upa_dft_codebook(*spec.tx_array)
    gains = np.abs(cb_rx.conj().T @ h_test @ cb_tx) ** 2
    genie = pow_2_dB(gains.reshape(len(gains), -1).max(axis=1) / system.noise_lin)
    return {"mrt_mrc_db": float(np.mean(mrt_mrc)),
            "genie_dft_db": float(np.mean(genie))}


def run(dataset: str = "sionna_28ghz",
        sizes: list[int] | None = None,
        taus: list[int] | None = None,
        n_probes: list[int] | None = None,
        k_total: int = 5_000,
        n_test: int = 5_000,
        seeds: list[int] | None = None,
        epochs: int = 1_000,
        width: int = cfg.DEFAULT_WIDTH,
        output: Path | None = None) -> Path:
    """Sweep the DDIM checkpoint used to augment the beam-alignment training set.

    Returns the path of the results CSV.
    """
    spec = cfg.get_dataset(dataset)
    sizes = sizes or [100, 500]
    n_probes = n_probes or [2, 4]
    seeds = seeds or [0]
    output = Path(output) if output else cfg.result_dir("tables") / "beam_alignment.csv"
    system = System()

    pool = load_pool(spec)
    h_test = pool.h[pool.test_indices()[:n_test]]
    print(f"Beam alignment on {spec.label}")
    print(f"  {len(h_test)} held-out test channels, probes {n_probes}")
    for name, value in baselines(h_test, system, spec).items():
        print(f"  baseline {name}: {value:.2f} dB")

    rows: list[list] = []

    def train_and_eval(experiment: str, n: int, tau: int | None,
                       h_train: np.ndarray, frobenius: bool,
                       num_real: int, num_synth: int) -> None:
        for n_probe in n_probes:
            for seed in seeds:
                state, _ = train_predictor(h_train, n_probe, system,
                                           epochs=epochs, seed=seed)
                snr, rate = evaluate_predictor(state, n_probe, h_test, system,
                                               frobenius)
                rows.append([experiment, n, tau if tau is not None else -1,
                             n_probe, seed, num_real, num_synth, snr, rate])
                print(f"  {experiment} N={n} tau={tau} probes={n_probe} "
                      f"seed={seed}: {snr:.2f} dB")

    for n in sizes:
        run_dir = spec.run_dir(n, width, cfg.batch_size_for(n))
        checkpoints = list_checkpoints(run_dir)
        if not checkpoints:
            print(f"  skipping N={n}: no DDIM checkpoints in {run_dir}")
            continue
        if taus:
            checkpoints = [(t, p) for t, p in checkpoints if t in set(taus)]

        reference = pool.h[pool.train_indices(n)]
        train_and_eval("reference_only", n, None,
                       _frobenius_normalise(reference), True, n, 0)

        num_synth = k_total - n
        ddim = build_ddim(spec.sample_shape, width, cfg.DEVICE)
        for tau, path in checkpoints:
            if tau == 0 or not load_checkpoint(ddim, path):
                continue
            samples, _ = cached_samples(ddim, run_dir, tau, num_synth,
                                        spec.sample_shape)
            hv = (samples[:, 0] + 1j * samples[:, 1]).astype(np.complex64)
            synthetic = to_antenna(hv, spec.rx_array, spec.tx_array)
            mixture = np.concatenate([_frobenius_normalise(reference),
                                      _frobenius_normalise(synthetic)])
            train_and_eval("augmented", n, tau, mixture, True, n, num_synth)
        del ddim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print(f"Wrote {output}")
    return output
