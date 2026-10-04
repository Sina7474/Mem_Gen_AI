"""
generate_downstream_28GHz_LoS.py — Synthetic channels for the 28 GHz downstream task
====================================================================================
Samples synthetic channels from the trained **28 GHz LoS-only** DDIM checkpoints
so they can be used as training data for the downstream beam-alignment study
(Downstream_Tasks/Beam_alignment/run_task10_28GHz_LoS.py).

This is the 28 GHz analogue of `generate_10k.py` (3.5 GHz), with two
differences that follow from how the 28 GHz models were trained:

  1. **EMA weights are used.** The 28 GHz runs were trained with
     `train_DDIM_tau_ema_28GHz_LoS.py`, whose checkpoints store both
     `model_state_dict` and `ema_model_state_dict`. Every downstream/eval script
     in this project samples from the EMA copy, so this script does the same
     (override with `--weights raw` if you ever need the live weights).
  2. **Log tree.** Checkpoints live in
     `logs_ema_28GHz_LoS/DDIM_tau_ema_28GHz_LoS_<N>_bs<B>_incremental/`
     rather than `logs/DDIM_tau_<N>_incremental/`.

The architecture, the DDIM schedule (n_T = 200, betas = (1e-4, 0.02)) and the
deterministic DDIM sampler are imported verbatim from `train_DDIM_tau.py`, so
sampling is identical to the 3.5 GHz study.

Output
------
    logs_ema_28GHz_LoS/DDIM_tau_ema_28GHz_LoS_<N>_bs<B>_incremental/
        generated_downstream/generated_tau_<tau>_<M>.npz
            key   : 'channels'
            shape : (M, 2, 4, 32)  float32
            domain: NORMALIZED BEAMSPACE (exactly what the DDIM was trained on)

The downstream loader inverts the UPA DFT transform to get back to the antenna
domain, using the same codebook convention as training.

Files written by the τ-evaluation pipeline (`generated_ema/…`) are never touched.

Usage
-----
    conda activate Mem_Gen
    cd Code/DDIM_FMM

    # everything the downstream task needs (default N and tau sets)
    python generate_downstream_28GHz_LoS.py

    # a single checkpoint
    python generate_downstream_28GHz_LoS.py --N 200 --tau 10000

    # larger sampling batch if the GPU has room
    python generate_downstream_28GHz_LoS.py --batch_size 500

Arguments
---------
    --N           DDIM generator training sizes      (default: 200 1000)
    --tau         Checkpoint horizons to sample from (default: 1000 10000 100000)
    --n_generate  Channels to generate per checkpoint (default: 5000)
    --batch_size  Sampling batch size                (default: 250)
    --weights     'ema' (default) or 'raw'
    --force       Overwrite existing output files
    --seed        RNG seed (default: 0)
"""

import argparse
import importlib.util
import math
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

# ── Reuse the exact architecture + DDIM schedule used for training ──────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "train_ddim_tau_module", os.path.join(_HERE, "train_DDIM_tau.py"))
_train_mod = importlib.util.module_from_spec(_spec)
sys.modules["train_ddim_tau_module"] = _train_mod
_spec.loader.exec_module(_train_mod)

Unet = _train_mod.Unet
DDIM = _train_mod.DDIM
N_T = _train_mod.N_T          # 200
BETAS = _train_mod.BETAS      # (1e-4, 0.02)
N_FEAT = _train_mod.N_FEAT    # 256

LOGS_DIR = os.path.join(_HERE, "logs_ema_28GHz_LoS")
OUT_SUBDIR = "generated_downstream"
CHANNEL_SHAPE = (2, 4, 32)


def run_dir(n_train: int, n_feat: int = N_FEAT) -> str:
    """Directory of the 28 GHz run for a given generator size N.

    Reproduces the naming rule of train_DDIM_tau_ema_28GHz_LoS.py:
    batch size B = min(N, 500), width suffix only when W != 256.
    """
    bs = min(n_train, 500)
    nfeat_suffix = f"_nfeat{n_feat}" if n_feat != N_FEAT else ""
    return os.path.join(
        LOGS_DIR,
        f"DDIM_tau_ema_28GHz_LoS_{n_train}{nfeat_suffix}_bs{bs}_incremental")


def output_path(n_train: int, tau: int, n_generate: int, n_feat: int = N_FEAT) -> str:
    return os.path.join(run_dir(n_train, n_feat), OUT_SUBDIR,
                        f"generated_tau_{tau}_{n_generate}.npz")


def load_checkpoint(ckpt_path: str, device, weights: str = "ema", n_feat: int = N_FEAT):
    """Build a DDIM model and load the requested weight set from a checkpoint."""
    ddim = DDIM(nn_model=Unet(in_channels=2, n_feat=n_feat),
                betas=BETAS, n_T=N_T, device=device).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    key = "ema_model_state_dict" if weights == "ema" else "model_state_dict"
    if key not in ck:
        raise KeyError(f"'{key}' missing in {ckpt_path}. Keys: {list(ck.keys())}")
    ddim.load_state_dict(ck[key], strict=True)
    ddim.eval()
    return ddim


@torch.no_grad()
def sample_batched(ddim, n_sample: int, device, batch_size: int) -> np.ndarray:
    """Deterministic DDIM sampling in GPU-sized batches -> (n_sample, 2, 4, 32)."""
    out = []
    produced = 0
    n_batches = math.ceil(n_sample / batch_size)
    for _ in tqdm(range(n_batches), desc="  sampling", leave=False):
        cur = min(batch_size, n_sample - produced)
        if cur <= 0:
            break
        x = ddim.sample(cur, CHANNEL_SHAPE, device)
        out.append(x.cpu().numpy())
        produced += cur
    return np.concatenate(out, axis=0)[:n_sample]


def generate_one(n_train, tau, n_generate, batch_size, device, weights,
                 force, n_feat=N_FEAT) -> bool:
    rdir = run_dir(n_train, n_feat)
    ckpt = os.path.join(rdir, "checkpoints", f"checkpoint_tau_{tau}.pth")
    out_p = output_path(n_train, tau, n_generate, n_feat)

    if not os.path.exists(ckpt):
        print(f"  [SKIP] checkpoint not found: {ckpt}")
        return False
    if os.path.exists(out_p) and not force:
        have = np.load(out_p)["channels"].shape[0]
        print(f"  [SKIP] exists with {have} samples: {out_p}")
        return True

    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    print(f"  loading {weights.upper()} weights: {ckpt}")
    ddim = load_checkpoint(ckpt, device, weights=weights, n_feat=n_feat)

    print(f"  generating {n_generate} channels (batch_size={batch_size}) ...")
    channels = sample_batched(ddim, n_generate, device, batch_size)

    np.savez_compressed(out_p, channels=channels.astype(np.float32))
    print(f"  saved -> {out_p}  shape={channels.shape}")

    del ddim
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Generate synthetic 28 GHz LoS channels from DDIM EMA checkpoints")
    ap.add_argument("--N", type=int, nargs="+", default=[200, 1000],
                    help="DDIM generator training sizes (default: 200 1000)")
    ap.add_argument("--tau", type=int, nargs="+", default=[1000, 10000, 100000],
                    help="Checkpoint horizons (default: 1000 10000 100000)")
    ap.add_argument("--n_generate", type=int, default=5000,
                    help="Channels per checkpoint (default: 5000)")
    ap.add_argument("--batch_size", type=int, default=250,
                    help="Sampling batch size (default: 250)")
    ap.add_argument("--n_feat", type=int, default=N_FEAT,
                    help=f"U-Net width W of the generators (default: {N_FEAT})")
    ap.add_argument("--weights", choices=["ema", "raw"], default="ema",
                    help="Which weight set to sample from (default: ema)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("GENERATING SYNTHETIC 28 GHz LoS CHANNELS (downstream beam alignment)")
    print("=" * 70)
    print(f"  device      : {device}")
    print(f"  weights     : {args.weights}")
    print(f"  N values    : {args.N}")
    print(f"  tau values  : {args.tau}")
    print(f"  per ckpt    : {args.n_generate} channels")
    print(f"  batch size  : {args.batch_size}")
    print("=" * 70)

    total = len(args.N) * len(args.tau)
    done = 0
    for n_train in args.N:
        for tau in args.tau:
            done += 1
            print(f"\n[{done}/{total}] N={n_train}, tau={tau}")
            generate_one(n_train, tau, args.n_generate, args.batch_size,
                         device, args.weights, args.force, args.n_feat)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    missing = 0
    for n_train in args.N:
        for tau in args.tau:
            p = output_path(n_train, tau, args.n_generate, args.n_feat)
            ok = os.path.exists(p)
            missing += (not ok)
            print(f"  [{'OK     ' if ok else 'MISSING'}] {p}")
    print("=" * 70)
    if missing:
        print(f"  {missing} file(s) missing — check the checkpoints above.")
        sys.exit(1)
    print("  All files generated. Next:")
    print("    cd ../../Downstream_Tasks/Beam_alignment")
    print("    python run_task10_28GHz_LoS.py")


if __name__ == "__main__":
    main()
