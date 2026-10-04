"""
generate_downstream_28GHz_LoS_SceneV1.py
    — Generate synthetic channels from the 28 GHz LoS SceneV1 DDIM checkpoints
================================================================================
Generates channels for the downstream beam-alignment study using the NEW 28 GHz
scene (SceneV1). Structurally identical to `generate_downstream_28GHz_LoS.py`
but points to the separate log tree `logs_ema_28GHz_LoS_SceneV1/`.

Output
------
    logs_ema_28GHz_LoS_SceneV1/DDIM_tau_ema_28GHz_LoS_SceneV1_<N>_bs<B>_incremental/
        generated_downstream/generated_tau_<tau>_<M>.npz
            key   : 'channels'
            shape : (M, 2, 4, 32)  float32
            domain: NORMALIZED BEAMSPACE

Usage
-----
    conda activate Mem_Gen
    cd Code/DDIM_FMM

    # Generate all needed pools for downstream beam alignment
    python generate_downstream_28GHz_LoS_SceneV1.py --N 100 200 --tau 1000 10000 100000 200000

    # Single checkpoint
    python generate_downstream_28GHz_LoS_SceneV1.py --N 100 --tau 10000
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

# *** SEPARATE LOG TREE for the new scene ***
LOGS_DIR = os.path.join(_HERE, "logs_ema_28GHz_LoS_SceneV1")
OUT_SUBDIR = "generated_downstream"
CHANNEL_SHAPE = (2, 4, 32)


def run_dir(n_train: int, n_feat: int = N_FEAT) -> str:
    """Directory of the SceneV1 run for a given generator size N."""
    bs = min(n_train, 500)
    nfeat_suffix = f"_nfeat{n_feat}" if n_feat != N_FEAT else ""
    return os.path.join(
        LOGS_DIR,
        f"DDIM_tau_ema_28GHz_LoS_SceneV1_{n_train}{nfeat_suffix}_bs{bs}_incremental")


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
        description="Generate synthetic 28 GHz LoS SceneV1 channels from DDIM EMA checkpoints")
    ap.add_argument("--N", type=int, nargs="+", default=[100, 200],
                    help="DDIM generator training sizes (default: 100 200)")
    ap.add_argument("--tau", type=int, nargs="+", default=[1000, 10000, 100000, 200000],
                    help="Checkpoint horizons (default: 1000 10000 100000 200000)")
    ap.add_argument("--n_generate", type=int, default=5000,
                    help="Channels per checkpoint (default: 5000)")
    ap.add_argument("--batch_size", type=int, default=250,
                    help="Sampling batch size (default: 250)")
    ap.add_argument("--n_feat", type=int, default=N_FEAT,
                    help=f"U-Net width W (default: {N_FEAT})")
    ap.add_argument("--weights", choices=["ema", "raw"], default="ema")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("GENERATING SYNTHETIC 28 GHz LoS SceneV1 CHANNELS (downstream)")
    print("=" * 70)
    print(f"  device      : {device}")
    print(f"  weights     : {args.weights}")
    print(f"  N values    : {args.N}")
    print(f"  tau values  : {args.tau}")
    print(f"  n_generate  : {args.n_generate}")
    print(f"  batch_size  : {args.batch_size}")
    print(f"  n_feat (W)  : {args.n_feat}")
    print(f"  logs dir    : {LOGS_DIR}")
    print("=" * 70)

    for n in args.N:
        print(f"\n{'─'*40}\n  N = {n}\n{'─'*40}")
        for tau in args.tau:
            print(f"\n  tau = {tau}:")
            generate_one(n, tau, args.n_generate, args.batch_size, device,
                         args.weights, args.force, args.n_feat)

    print("\nAll done.")


if __name__ == "__main__":
    main()
