"""
beam_align_core.py — DL-GF beam-alignment training / evaluation / baselines
===========================================================================
Thin orchestration layer on top of the copied DL-GF modules (DL_utils.py,
beam_utils.py). Provides a programmatic API so run_task10.py can train and
evaluate many experiments without shelling out to the original argparse CLI.

Design choices (documented in the Task 10 report):
  - Antenna arrays: BS (Tx) = 4x8 = 32, UE (Rx) = 2x2 = 4.
  - Model: Joint_BF_Autoencoder, learned_probing='TxRx' (fully-learned Tx & Rx
    probing beamformers — no DFT/sqrt assumption, so it works for Nt=32 which is
    not a perfect square), feedback='diagonal', beam_synthesizer='MLP',
    BF_loss (maximize beamforming gain).
  - Scale handling (identical philosophy to the reference DL-GF pipeline):
      * Training channels are max-abs normalized to O(1) before the network.
      * When real + synthetic are MIXED, per-sample Frobenius normalization is
        applied first so the physical-scale real channels and the unit-scale
        synthetic channels contribute comparably.
      * At evaluation the RAW physical test channels are used to compute the
        beamforming gain, so the reported SNR is physically meaningful. The
        model input is separately max-abs normalized (matching training scale).
  - Metric: Average SNR (dB) on the held-out test set, per num_probing_beam.
    achievable_rate = log2(1 + SNR_lin) is also recorded.
  - Baselines (npb-independent horizontal lines):
      * MRT + MRC   : per-sample optimal gain = sigma_max(H)^2 (matched filter).
      * Genie DFT   : best beam pair from the unitary UPA DFT codebooks
                      (Rx 2x2, Tx 4x8), exhaustive search per sample.
"""

import os
import numpy as np
import torch
import torch.optim as optim

from DL_utils import Joint_BF_Autoencoder, BF_loss, eval_model
from beam_utils import pow_2_dB, dB_2_pow

# UPA geometry (must match data_utils)
NRX_X, NRX_Y = 2, 2      # Nr = 4
NTX_X, NTX_Y = 4, 8      # Nt = 32


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM / NOISE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
def make_system(tx_power_dBm=20, BW_MHz=100.0, noise_PSD_dB=-161.0,
                measurement_gain=16.0):
    """Return a dict of derived system constants (matches DL-GF defaults)."""
    noise_power_dBm = noise_PSD_dB + 10.0 * np.log10(BW_MHz * 1e6)
    noise_lin = dB_2_pow(noise_power_dBm - tx_power_dBm)          # ref. unit Tx
    meas_noise = noise_lin / measurement_gain                    # probing noise
    return {
        "tx_power_dBm": tx_power_dBm,
        "noise_power_dBm": noise_power_dBm,
        "measurement_gain": measurement_gain,
        "noise_lin": noise_lin,
        "meas_noise": meas_noise,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING-DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
def prepare_training_channels(h_train, seed, frob_norm):
    """Prepare a single-source training set (real-only OR synthetic-only).

    Parameters
    ----------
    h_train   : (N, Nr, Nt) complex — the training channels (all from one source).
    seed      : int — deterministic shuffle seed.
    frob_norm : bool — if True, apply per-sample Frobenius normalization so every
                channel has unit Frobenius norm. This mirrors the reference DL-GF
                pipeline, which Frobenius-normalizes SYNTHETIC (unit-scale
                beamspace) channels but keeps RAW real channels at physical scale.

    Returns (h_all complex64, frob_norm_used bool).
    """
    h_all = np.asarray(h_train, dtype=np.complex64)
    if frob_norm:
        frob = np.sqrt(np.sum(np.abs(h_all) ** 2, axis=(1, 2), keepdims=True))
        frob = np.where(frob == 0, 1.0, frob)
        h_all = (h_all / frob).astype(np.complex64)
    # Deterministic shuffle so LoS/NLoS are interleaved across batches
    rng = np.random.default_rng(seed)
    h_all = h_all[rng.permutation(len(h_all))]
    return h_all, bool(frob_norm)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE (experiment, npb) MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_one(h_train, npb, sysd, nepoch, batch_size, lr, seed, device):
    """Train a Joint_BF_Autoencoder for one num_probing_beam value.

    Returns (state_dict on CPU, norm_factor_train float).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Nr, Nt = h_train.shape[1], h_train.shape[2]
    norm_factor = float(np.max(np.abs(h_train)))
    h_scaled = (h_train / norm_factor).astype(np.complex64)

    X = torch.from_numpy(h_scaled).to(device)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X),
        batch_size=min(batch_size, len(X)), shuffle=True,
    )

    model = Joint_BF_Autoencoder(
        num_antenna_Tx=Nt, num_antenna_Rx=Nr,
        num_probing_beam_Tx=npb, num_probing_beam_Rx=npb,
        noise_power=sysd["meas_noise"], norm_factor=norm_factor,
        feedback="diagonal", num_feedback=None,
        learned_probing="TxRx", beam_synthesizer="MLP",
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)
    loss_fn = BF_loss(noise_power_dBm=sysd["noise_power_dBm"],
                      Tx_power_dBm=sysd["tx_power_dBm"])

    model.train()
    for _ in range(nepoch):
        for (xb,) in loader:
            opt.zero_grad()
            tx_beam, rx_beam, _ = model(xb)
            loss = loss_fn(xb, tx_beam, rx_beam)
            loss.backward()
            opt.step()

    return {k: v.cpu() for k, v in model.state_dict().items()}, norm_factor


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE ONE MODEL  →  Average SNR (dB) + achievable rate
# ─────────────────────────────────────────────────────────────────────────────
def eval_snr(state_dict, npb, h_test_raw, sysd, frob_norm=False, eval_seed=0):
    """Load a trained model and compute Average SNR (dB) on raw test channels.

    Test channels are preprocessed to EXACTLY MATCH the training pipeline:
      - If frob_norm=True (synthetic-data training): apply per-sample Frobenius
        normalization to the raw test channels, then scale by the training
        norm_factor retrieved from the model's loaded state dict.
      - If frob_norm=False (real-data training): no Frobenius normalization;
        scale by the training norm_factor from state dict.
    The training norm_factor is stored as a buffer inside the Joint_BF_Autoencoder
    state dict (joint_beamformer.norm_factor), so load_state_dict restores it.

    The beamforming gain is always computed on the RAW physical test channels
    (h_test_raw, un-modified), so the reported SNR is physically meaningful.

    This mirrors DLGF_Plotting_Sina_UPA.py::load_test_channels() exactly:
        if frob_norm_train and is_raw_test:
            h_for_model = h_test / frob_per_sample
        h_test_sc = (h_for_model.T / norm_factor_train).T
    """
    Nr, Nt = h_test_raw.shape[1], h_test_raw.shape[2]

    # Build model skeleton and restore all weights + buffers (incl. norm_factor)
    model = Joint_BF_Autoencoder(
        num_antenna_Tx=Nt, num_antenna_Rx=Nr,
        num_probing_beam_Tx=npb, num_probing_beam_Rx=npb,
        noise_power=sysd["meas_noise"], norm_factor=1.0,
        feedback="diagonal", num_feedback=None,
        learned_probing="TxRx", beam_synthesizer="MLP",
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Retrieve the training norm_factor from the restored buffer.
    # This is the max-abs of the (Frobenius-normalized) training data —
    # the same value the model's probing noise was calibrated for.
    norm_factor_train = model.joint_beamformer.norm_factor.item()

    # Step 1: preprocessing matching training
    if frob_norm:
        # Apply per-sample Frobenius normalization to the raw test channels.
        # Raw real Sionna channels have Frobenius norm ~7e-4 while synthetic
        # training channels were normalized to ||H||_F = 1.  Without this step
        # the test inputs are ~4000x smaller than training inputs, the probing
        # measurements are drowned in noise, and beam predictions are random.
        frob = np.sqrt(np.sum(np.abs(h_test_raw) ** 2, axis=(1, 2), keepdims=True))
        frob = np.where(frob == 0, 1.0, frob)
        h_for_model = (h_test_raw / frob).astype(np.complex64)
    else:
        h_for_model = h_test_raw.astype(np.complex64)

    # Step 2: scale by training norm_factor (NOT test set's own max-abs)
    h_scaled = (h_for_model / norm_factor_train).astype(np.complex64)

    np.random.seed(eval_seed)   # deterministic probing noise realization
    torch_x = torch.from_numpy(h_scaled)
    with torch.no_grad():
        predicted_bf_gain, _ = eval_model(
            model, torch_x, h_test_raw,  # h_test_raw: raw physical for BF gain
            noise_power=sysd["meas_noise"],
            prediction_mode="GF", feedback_mode="diagonal",
        )

    snr_lin = predicted_bf_gain / sysd["noise_lin"]
    snr_db = float(np.mean(pow_2_dB(snr_lin)))
    rate = float(np.mean(np.log2(1.0 + snr_lin)))
    return snr_db, rate


# ─────────────────────────────────────────────────────────────────────────────
# THEORETICAL / CLASSICAL BASELINES  (npb-independent)
# ─────────────────────────────────────────────────────────────────────────────
def compute_baselines(h_test_raw, sysd):
    """Return {'MRT_MRC': dB, 'genie_DFT': dB} average SNR over the test set."""
    from data_utils import upa_dft_codebook

    noise_lin = sysd["noise_lin"]

    # --- MRT + MRC : per-sample optimal matched-filter gain = sigma_max(H)^2 ---
    svals = np.linalg.svd(h_test_raw, compute_uv=False)      # (N, min(Nr,Nt))
    sigma_max_sq = svals[:, 0] ** 2                          # (N,)
    mrt_mrc_db = float(np.mean(pow_2_dB(sigma_max_sq / noise_lin)))

    # --- Genie-aided DFT : best beam pair from unitary UPA DFT codebooks -------
    Cb_rx = upa_dft_codebook(NRX_X, NRX_Y)                   # (Nr, Nr)
    Cb_tx = upa_dft_codebook(NTX_X, NTX_Y)                   # (Nt, Nt)
    bf = np.abs(Cb_rx.conj().T @ h_test_raw @ Cb_tx) ** 2    # (N, Nr, Nt)
    best = bf.reshape(bf.shape[0], -1).max(axis=1)           # (N,)
    genie_db = float(np.mean(pow_2_dB(best / noise_lin)))

    return {"MRT_MRC": mrt_mrc_db, "genie_DFT": genie_db}
