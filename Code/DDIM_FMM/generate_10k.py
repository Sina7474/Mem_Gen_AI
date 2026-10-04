"""
generate_10k.py — Generate 10 000 synthetic channels from existing DDIM checkpoints
=====================================================================================
Loads existing tau-checkpoint .pth files (already trained by train_DDIM_tau.py) and
generates 10 000 normalised beamspace channels for each (N, tau) combination.

!! WARNING — files produced before 2026-08-09 are INVALID !!
    Until then, load_checkpoint() called load_state_dict(..., strict=False) with a
    checkpoint dict whose keys are ('model_state_dict', 'optimizer_state_dict', …).
    None of those keys match the module, so with strict=False nothing was restored
    and every sample was drawn from a randomly initialised U-Net.  Any
    `generated_tau_*_10k.npz` created before that date must be regenerated with
    --force before it is used for anything.

    Note also that these are the NON-EMA runs under `logs/`.  The memorization
    metrics (f_mem, FCD) in DDIM_Evaluation/dataset_size_effect are measured on
    the EMA runs under `logs_ema/`, so downstream experiments that need to line up
    with those metrics should use the EMA sample pools instead
    (`logs_ema/DDIM_tau_ema_<N>_bs<B>_incremental/generated_ema/`).

Saved to:
    logs/DDIM_tau_{N}_incremental/generated_tau/generated_tau_{tau}_10k.npz
    key: 'channels', shape: (10000, 2, 4, 32), float32

The 5000-sample files (generated_tau_{tau}.npz) are NEVER touched.

Usage:
    conda activate Mem_Gen
    cd Code/DDIM_FMM
    python generate_10k.py                          # all N and tau
    python generate_10k.py --N 200                  # one dataset size
    python generate_10k.py --N 200 --tau 10000      # one specific checkpoint
    python generate_10k.py --batch_size 200         # larger GPU batch

Arguments:
    --N           DDIM training dataset sizes to process (default: 100 200 1000)
    --tau         Tau checkpoints to generate from (default: 1000 10000 100000)
    --n_generate  Number of channels to generate (default: 10000)
    --batch_size  Sampling batch size (default: 100; increase if GPU has memory)
    --force       Overwrite existing 10k files (default: skip if already exists)
    --seed        RNG seed for reproducibility (default: 0)
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ── Model architecture (identical to train_DDIM_tau.py) ──────────────────────

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = (in_channels == out_channels)
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels), nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels), nn.GELU())

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            out = (x + x2) if self.same_channels else (x1 + x2)
            return out / 1.414
        return self.conv2(self.conv1(x))


class UnetDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.model = nn.Sequential(ResidualConvBlock(in_ch, out_ch), nn.MaxPool2d(2))

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
            ResidualConvBlock(out_ch, out_ch),
            ResidualConvBlock(out_ch, out_ch))

    def forward(self, x, skip):
        return self.model(torch.cat((x, skip), 1))


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, x):
        return self.model(x.view(-1, self.input_dim))


class Unet(nn.Module):
    def __init__(self, in_channels=2, n_feat=256):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.to_vec = nn.Sequential(nn.AvgPool2d((1, 8)), nn.GELU())
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, n_feat)
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, kernel_size=(1, 8), stride=(1, 8)),
            nn.GroupNorm(8, 2 * n_feat), nn.ReLU())
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat), nn.ReLU(),
            nn.Conv2d(n_feat, in_channels, 3, 1, 1))

    def forward(self, x, t):
        x = self.init_conv(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        hv = self.to_vec(d2)
        te1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        te2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        u1 = self.up0(hv)
        u2 = self.up1(u1 + te1, d2)
        u3 = self.up2(u2 + te2, d1)
        return self.out(torch.cat((u3, x), 1))


# ── DDIM schedule + sampler (identical to train_DDIM_tau.py) ─────────────────

def ddim_schedules(beta1, beta2, T):
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    alpha_t = 1 - beta_t
    alphabar_t = torch.cumsum(torch.log(alpha_t), dim=0).exp()
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)
    sqrtmab = torch.sqrt(1 - alphabar_t)
    DDIM_coeff = sqrtmab - torch.sqrt(alpha_t) * torch.sqrt(1 - alphabar_t / alpha_t)
    return {"oneover_sqrta": oneover_sqrta, "sqrtab": sqrtab,
            "sqrtmab": sqrtmab, "DDIM_coeff": DDIM_coeff}


class DDIM(nn.Module):
    def __init__(self, nn_model, betas=(1e-4, 0.02), n_T=200, device="cpu"):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.n_T = n_T
        self.device = device
        for k, v in ddim_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

    def sample(self, n_sample, size, device, batch_size=100):
        """Generate n_sample channels in batches to fit GPU memory."""
        all_out = []
        n_batches = math.ceil(n_sample / batch_size)
        self.eval()
        with torch.no_grad():
            for _ in tqdm(range(n_batches), desc="  sampling", leave=False):
                cur = min(batch_size, n_sample - len(all_out) * batch_size
                          if all_out else batch_size)
                # recompute cur correctly
                generated_so_far = sum(x.shape[0] for x in all_out)
                cur = min(batch_size, n_sample - generated_so_far)
                if cur <= 0:
                    break
                x_i = torch.randn(cur, *size).to(device)
                for i in range(self.n_T, 0, -1):
                    t_is = torch.tensor([i / self.n_T]).to(device).repeat(cur, 1, 1, 1)
                    eps = self.nn_model(x_i, t_is)
                    x_i = self.oneover_sqrta[i] * (x_i - eps * self.DDIM_coeff[i])
                all_out.append(x_i.cpu().numpy())
        return np.concatenate(all_out, axis=0)[:n_sample]


# ── Main ──────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path, device):
    """Load a DDIM model from a tau checkpoint .pth file.

    The tau checkpoints written by train_DDIM_tau.py are dicts of the form
    {'model_state_dict': ..., 'optimizer_state_dict': ..., 'global_step': ...}.
    Loading must be STRICT: a lenient load silently leaves the U-Net randomly
    initialized, and the sampler then emits noise that looks superficially like
    a channel but carries no information from the checkpoint at all.
    """
    ddim = DDIM(
        nn_model=Unet(in_channels=2, n_feat=256),
        betas=(1e-4, 0.02), n_T=200, device=device,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
    elif isinstance(state, dict) and "ema_model_state_dict" in state:
        sd = state["ema_model_state_dict"]
    else:
        sd = state

    if not any(k.startswith("nn_model.") for k in sd.keys()):
        raise KeyError(
            f"Unrecognised checkpoint layout in {ckpt_path}; "
            f"top-level keys: {list(sd.keys())[:8]}")

    ddim.load_state_dict(sd, strict=True)
    ddim.eval()
    return ddim


def generate_10k(n_train, tau, logs_base, n_generate, batch_size, device, force):
    log_dir = os.path.join(logs_base, f"DDIM_tau_{n_train}_incremental")
    ckpt_path = os.path.join(log_dir, "checkpoints", f"checkpoint_tau_{tau}.pth")
    out_dir = os.path.join(log_dir, "generated_tau")
    out_path = os.path.join(out_dir, f"generated_tau_{tau}_10k.npz")

    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return False

    if os.path.exists(out_path) and not force:
        existing = np.load(out_path)["channels"].shape[0]
        print(f"  [SKIP] Already exists with {existing} samples: {out_path}")
        return True

    os.makedirs(out_dir, exist_ok=True)
    print(f"  Loading checkpoint: {ckpt_path}")
    ddim = load_checkpoint(ckpt_path, device)

    print(f"  Generating {n_generate} channels (batch_size={batch_size}) ...")
    channels = ddim.sample(n_generate, (2, 4, 32), device, batch_size=batch_size)
    # channels shape: (n_generate, 2, 4, 32), float32 beamspace normalised

    np.savez_compressed(out_path, channels=channels.astype(np.float32))
    print(f"  Saved → {out_path}  shape={channels.shape}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Generate 10k synthetic DDIM channels")
    ap.add_argument("--N", type=int, nargs="+", default=[100, 200, 1000],
                    help="DDIM training dataset sizes (default: 100 200 1000)")
    ap.add_argument("--tau", type=int, nargs="+", default=[1000, 10000, 100000],
                    help="Tau checkpoints (default: 1000 10000 100000)")
    ap.add_argument("--n_generate", type=int, default=10000,
                    help="Number of channels to generate (default: 10000)")
    ap.add_argument("--batch_size", type=int, default=100,
                    help="Sampling batch size (default: 100)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed (default: 0)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Generating {args.n_generate} channels per checkpoint")
    print(f"N values : {args.N}")
    print(f"Tau values: {args.tau}\n")

    here = os.path.dirname(os.path.abspath(__file__))
    logs_base = os.path.join(here, "logs")

    total = len(args.N) * len(args.tau)
    done = 0
    for n_train in args.N:
        for tau in args.tau:
            done += 1
            print(f"[{done}/{total}] N={n_train}, tau={tau}")
            generate_10k(n_train, tau, logs_base, args.n_generate,
                         args.batch_size, device, args.force)

    print(f"\nDone. Generated files are saved as:")
    for n_train in args.N:
        for tau in args.tau:
            p = os.path.join(logs_base, f"DDIM_tau_{n_train}_incremental",
                             "generated_tau", f"generated_tau_{tau}_10k.npz")
            status = "OK" if os.path.exists(p) else "MISSING"
            print(f"  [{status}] {p}")


if __name__ == "__main__":
    main()
