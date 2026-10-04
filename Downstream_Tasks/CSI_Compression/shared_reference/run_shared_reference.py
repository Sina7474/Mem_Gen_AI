"""
run_shared_reference.py — Shared-Reference CRNet CSI-compression experiment
===========================================================================
Implements Phase 1 (the main fair comparison) of

    reports/CRNet_Shared_Reference_Experiment_Implementation_Guide.md

for W = 256, N in {200, 1000}, K_total = 5000, CRNet seeds {0,1,2,3}. Section 11
(fixed 20/80 control) is intentionally NOT run here, per the user's request.

For every N the script:
  1. Rebuilds D_N from the raw data + the DDIM shuffle indices and asserts it is
     byte-identical to the generator's train.npy  (N_csi == N, I_DDIM == I_CRNet).
  2. Trains a reference-only CRNet on D_N.
  3. Trains a full-real benchmark CRNet on D_5000 (shared across N, trained once).
  4. For 6-8 N-specific DDIM checkpoints spanning the pre/in/post-window regimes,
     builds  D_N ∪ synthetic(K_total-N)  and trains an augmented CRNet.
  5. Repeats every configuration over the four seeds and logs one CSV row per run.

Representation: every channel (real reference, synthetic, val, test) is mapped to
the same normalised beamspace the DDIM used, so real and synthetic share one
basis and the NMSE metric is consistent.

Usage:
    conda activate Mem_Gen
    cd Downstream_Tasks/CSI_Compression
    python shared_reference/run_shared_reference.py            # full run
    python shared_reference/run_shared_reference.py --debug    # quick smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import sr_data as D                       # noqa: E402
from sr_train import train_crnet_convergence  # noqa: E402


# ── Experiment configuration (guide sections 3, 9, 16) ───────────────────────
W = 256
REFERENCE_SIZES = [200, 500, 1000]
BENCHMARK_N = 5000
K_TOTAL = 5000
SEEDS = [0, 1, 2, 3]
GENERATOR_SEED = 0

# CRNet hyper-parameters (identical to the legacy Task-09 trainer)
NR, NT, REDUCTION = 4, 32, 4
MAX_EPOCHS = 500
# Early stopping is DISABLED (patience=None): the legacy trainer ran the full
# 500 epochs of the cosine schedule and restored best-validation weights. CRNet
# only reaches its ~-6 dB floor near the end of that schedule, so any early stop
# leaves it badly under-trained. Kept configurable via --patience for auditing.
PATIENCE = None
BATCH_SIZE = 512
LR = 1e-3

# Window definition (matches analyze_generalization_window.py defaults)
FMEM_THRESH = 0.1
REL_TOL = 0.5

RESULTS_DIR = HERE.parent / "results" / "shared_reference"
LOGS_DIR = RESULTS_DIR / "logs"
CKPT_DIR = RESULTS_DIR / "checkpoints"
RESULTS_CSV = RESULTS_DIR / "shared_reference_results.csv"

CSV_COLUMNS = [
    "configuration_id", "experiment_type", "N", "N_csi", "W", "K_total",
    "tau", "regime", "tau_gen", "tau_mem", "FCD", "f_mem",
    "sampling_mode", "crnet_seed", "generator_seed",
    "num_real_train", "num_synthetic_train", "num_total_train",
    "best_epoch", "max_epoch", "early_stopped", "converged_flag",
    "validation_nmse_db", "test_nmse_db", "final_lr",
    "reference_indices_file", "synthetic_data_file", "model_checkpoint",
]


def append_row(row):
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="Shared-reference CRNet experiment")
    ap.add_argument("--debug", action="store_true",
                    help="Quick smoke test: 1 seed, few epochs, 2 checkpoints/N")
    ap.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Override CRNet seeds")
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="Override reference sizes N")
    ap.add_argument("--patience", type=int, default=None,
                    help="Early-stopping patience (default: disabled, i.e. run "
                         "the full epoch budget like the legacy trainer)")
    ap.add_argument("--no_skip", dest="skip_done", action="store_false",
                    help="Retrain even if a cached run log exists")
    ap.set_defaults(skip_done=True)
    args = ap.parse_args()

    seeds = args.seeds if args.seeds else ([0] if args.debug else SEEDS)
    sizes = args.sizes if args.sizes else REFERENCE_SIZES
    max_epochs = args.epochs if args.epochs else (20 if args.debug else MAX_EPOCHS)
    patience = args.patience if args.patience else PATIENCE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for d in (RESULTS_DIR, LOGS_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("Shared-Reference CRNet CSI-Compression Experiment (Phase 1)")
    print("=" * 74)
    print(f"  Device        : {device}")
    print(f"  W             : {W}")
    print(f"  N (reference) : {sizes}")
    print(f"  K_total       : {K_TOTAL}")
    print(f"  Seeds         : {seeds}")
    print(f"  Max epochs    : {max_epochs} (patience "
          f"{patience if patience else 'off — full run'})")
    print("=" * 74)

    # ── Shared, leakage-safe split ────────────────────────────────────────
    split = D.build_shared_split(
        reference_sizes=tuple(sizes), benchmark_n=BENCHMARK_N,
        n_val_per_side=500, n_test_per_side=1000, eval_tail_per_side=1500,
        seed=42)
    X_val = D.real_to_tensor(split.H_val)
    X_test = D.real_to_tensor(split.H_test)
    print(f"Shared val set : {len(X_val)}  |  shared test set: {len(X_test)} "
          f"(both DDIM-unseen, disjoint from every D_N)")

    # Persist the split metadata + reference indices for provenance.
    (RESULTS_DIR / "split_meta.json").write_text(json.dumps({
        **split.meta,
        "val_idx_los": split.val_idx_los.tolist(),
        "val_idx_nlos": split.val_idx_nlos.tolist(),
        "test_idx_los": split.test_idx_los.tolist(),
        "test_idx_nlos": split.test_idx_nlos.tolist(),
    }, indent=2))

    done = set()
    if RESULTS_CSV.exists() and args.skip_done:
        with open(RESULTS_CSV) as f:
            for r in csv.DictReader(f):
                done.add(r["configuration_id"])

    def train_and_log(cfg_id, *, exp_type, n, tau, regime, tau_gen, tau_mem,
                      fcd, fmem, sampling_mode, seed, gen_seed, X_train,
                      n_real, n_syn, ref_file, syn_file):
        if cfg_id in done:
            print(f"  [SKIP-CSV] {cfg_id}")
            return
        log_path = LOGS_DIR / f"{cfg_id}.json"
        ckpt_path = CKPT_DIR / f"{cfg_id}.pt"
        rec = train_crnet_convergence(
            X_train, X_val, X_test, tag=cfg_id, log_path=log_path,
            ckpt_path=ckpt_path, nr=NR, nt=NT, reduction=REDUCTION,
            max_epochs=max_epochs, patience=patience, batch_size=BATCH_SIZE,
            lr=LR, seed=seed, device=device, skip_done=args.skip_done,
            extra_meta={"experiment_type": exp_type, "N": n, "tau": tau})
        append_row({
            "configuration_id": cfg_id, "experiment_type": exp_type,
            "N": n, "N_csi": n, "W": W, "K_total": K_TOTAL,
            "tau": tau if tau is not None else "none", "regime": regime,
            "tau_gen": round(tau_gen, 1) if tau_gen == tau_gen else "nan",
            "tau_mem": round(tau_mem, 1) if tau_mem == tau_mem else "nan",
            "FCD": round(fcd, 5) if fcd == fcd else "nan",
            "f_mem": round(fmem, 5) if fmem == fmem else "nan",
            "sampling_mode": sampling_mode, "crnet_seed": seed,
            "generator_seed": gen_seed if gen_seed is not None else "none",
            "num_real_train": n_real, "num_synthetic_train": n_syn,
            "num_total_train": n_real + n_syn,
            "best_epoch": rec["best_epoch"], "max_epoch": rec["max_epoch"],
            "early_stopped": rec["early_stopped"],
            "converged_flag": rec["converged_flag"],
            "validation_nmse_db": round(rec["validation_nmse_db"], 4),
            "test_nmse_db": round(rec["test_nmse_db"], 4),
            "final_lr": rec["final_lr"],
            "reference_indices_file": ref_file,
            "synthetic_data_file": syn_file if syn_file else "none",
            "model_checkpoint": str(ckpt_path),
        })
        done.add(cfg_id)

    # ── Full-real benchmark on D_5000 (shared across N; trained once) ──────
    H_bench, bl_los, bl_nlos = split.reference(BENCHMARK_N, verify=False)
    X_bench = D.real_to_tensor(H_bench)
    bench_ref_file = str(RESULTS_DIR / "ref_indices_N5000.npz")
    np.savez(bench_ref_file, los_sel=bl_los, nlos_sel=bl_nlos)
    for seed in seeds:
        train_and_log(
            f"fullreal_N5000_seed{seed}", exp_type="full_real", n=BENCHMARK_N,
            tau=None, regime="benchmark", tau_gen=float("nan"),
            tau_mem=float("nan"), fcd=float("nan"), fmem=float("nan"),
            sampling_mode="natural_mixture", seed=seed, gen_seed=None,
            X_train=X_bench, n_real=BENCHMARK_N, n_syn=0,
            ref_file=bench_ref_file, syn_file=None)

    # ── Per-N: reference-only + augmented across checkpoints ───────────────
    for n in sizes:
        H_ref, los_sel, nlos_sel = split.reference(n, verify=True)
        X_ref = D.real_to_tensor(H_ref)
        ref_file = str(RESULTS_DIR / f"ref_indices_N{n}.npz")
        np.savez(ref_file, los_sel=los_sel, nlos_sel=nlos_sel)

        tau_gen, tau_mem = D.load_window(n, FMEM_THRESH, REL_TOL)
        checkpoints = D.select_checkpoints(n, tau_gen, tau_mem)
        if args.debug:
            checkpoints = checkpoints[:2]
        print(f"\n{'─'*74}\n  N={n}: window [{tau_gen:.0f}, {tau_mem:.0f}] "
              f"→ checkpoints {checkpoints}")

        # Reference-only baseline
        for seed in seeds:
            train_and_log(
                f"refonly_N{n}_seed{seed}", exp_type="reference_only", n=n,
                tau=None, regime="reference", tau_gen=tau_gen, tau_mem=tau_mem,
                fcd=float("nan"), fmem=float("nan"),
                sampling_mode="natural_mixture", seed=seed, gen_seed=None,
                X_train=X_ref, n_real=n, n_syn=0, ref_file=ref_file,
                syn_file=None)

        # Augmented at each checkpoint
        n_syn = K_TOTAL - n
        for tau in checkpoints:
            ch, syn_file = D.load_synthetic(n, tau, n_syn, GENERATOR_SEED)
            X_syn = D.synth_to_tensor(ch)
            X_aug = torch.cat([X_ref, X_syn], dim=0)
            assert len(X_ref) == n and len(X_syn) == n_syn and len(X_aug) == K_TOTAL
            regime = D.classify_regime(tau, tau_gen, tau_mem)
            fcd, fmem = D.load_checkpoint_metrics(n, tau)
            for seed in seeds:
                train_and_log(
                    f"aug_N{n}_tau{tau}_seed{seed}", exp_type="augmented", n=n,
                    tau=tau, regime=regime, tau_gen=tau_gen, tau_mem=tau_mem,
                    fcd=fcd, fmem=fmem, sampling_mode="natural_mixture",
                    seed=seed, gen_seed=GENERATOR_SEED, X_train=X_aug,
                    n_real=n, n_syn=n_syn, ref_file=ref_file, syn_file=syn_file)

    print(f"\n{'='*74}\nDONE. Results CSV: {RESULTS_CSV}")
    print(f"Run  python shared_reference/plot_shared_reference.py  for figures.")
    print("=" * 74)


if __name__ == "__main__":
    main()
