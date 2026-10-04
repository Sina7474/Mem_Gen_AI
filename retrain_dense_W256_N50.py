#!/usr/bin/env python
"""
Dense re-training around τ=50,000 for W=256, N=50 to investigate the f_mem dip.

Resumes from the τ=30,000 checkpoint and saves checkpoints densely between
30k and 70k. Output goes to a SEPARATE directory so the original run is
untouched.

Usage (from project root):
    python retrain_dense_W256_N50.py
"""
import os, sys

# ── Monkey-patch DENSE_TAU_GRID before importing train_tau_ema ─────────────
DDIM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Code', 'DDIM_FMM')
sys.path.insert(0, DDIM_DIR)

import train_DDIM_tau_ema as trainer

# Replace the tau grid with a dense version around [30k, 70k]
trainer.DENSE_TAU_GRID = list(range(32000, 70001, 2000))
# → [32000, 34000, 36000, 38000, 40000, 42000, 44000, 46000, 48000,
#    50000, 52000, 54000, 56000, 58000, 60000, 62000, 64000, 66000, 68000, 70000]

RESUME_CKPT = os.path.join(
    DDIM_DIR,
    "logs_ema/DDIM_tau_ema_50_bs50_incremental/checkpoints/checkpoint_tau_30000.pth",
)
SAVE_NAME = "dense_retrain_W256_N50"


def main():
    os.chdir(DDIM_DIR)  # training script uses relative paths for dataset
    trainer.train_tau_ema(
        n_train_samples=50,
        max_tau=70000,
        incremental=True,
        seed=0,
        n_feat=256,
        save_name=SAVE_NAME,
        resume_from=RESUME_CKPT,
        batch_size=50,
    )


if __name__ == "__main__":
    main()
