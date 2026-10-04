"""
analyze_generalization_window.py
    — Numerical generalization-window duration (in τ) per batch size
================================================================================
For a fixed dataset (N = 1000) and model (n_feat = 256), each batch size passes
through the same three phases as training time τ grows:

    (1) UNDER-FIT   : FCD(Gen,Test) is high, samples are noise            (f_mem ≈ 0)
    (2) GENERALISE  : FCD(Gen,Test) sits on the real–real floor           (f_mem ≈ 0)
    (3) MEMORISE    : FCD(Gen,Test) drifts up, samples copy the trainset  (f_mem ↑)

The **generalization window** is the τ-band of phase (2): it opens when FCD(Gen,Test)
first descends onto the real–real floor (good samples) and closes when memorization
begins.  We quantify:

    τ_open   = first τ (log-interpolated) where FCD(Gen,Test) ≤ floor · (1 + rel_tol)
    τ_close  = τ (log-interpolated) where f_mem(τ) first crosses `fmem_thresh`
    Δτ       = τ_close − τ_open        (window duration, in optimizer steps)

The paper's claim (repetition does not change *when/how* a model memorises relative
to fitting) predicts Δτ should be ~constant across batch sizes even though the
absolute τ_open / τ_close shift.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/batch_size_effect
    python analyze_generalization_window.py
    python analyze_generalization_window.py --fmem_thresh 0.1 --rel_tol 0.5
"""

import os
import argparse
import numpy as np
import pandas as pd


def _log_interp_crossing(x, y, y_target, descending):
    """First τ where y crosses y_target, interpolated linearly in log(x).

    descending=True  → crossing from y > y_target down to y ≤ y_target
    descending=False → crossing from y < y_target up   to y ≥ y_target
    Returns np.nan if no such crossing exists.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lx = np.log(x)
    for i in range(1, len(y)):
        if descending and y[i - 1] > y_target and y[i] <= y_target:
            f = (y[i - 1] - y_target) / (y[i - 1] - y[i])
            return float(np.exp(lx[i - 1] + f * (lx[i] - lx[i - 1])))
        if (not descending) and y[i - 1] < y_target and y[i] >= y_target:
            f = (y_target - y[i - 1]) / (y[i] - y[i - 1])
            return float(np.exp(lx[i - 1] + f * (lx[i] - lx[i - 1])))
    return np.nan


def main():
    ap = argparse.ArgumentParser(
        description="Numerical generalization-window duration (τ) per batch size")
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--fmem_thresh", type=float, default=0.1,
                    help="f_mem value that marks the onset of memorization")
    ap.add_argument("--rel_tol", type=float, default=0.5,
                    help="'near floor' band: FCD ≤ floor·(1+rel_tol). Default 0.5 = 50%%")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Ignore τ below this (drops the τ=0 random-init spike)")
    ap.add_argument("--csv_path", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', f'bs_fcd_fmem_N{args.N}.csv')
    df = pd.read_csv(csv_path)
    df = df[df['tau'] >= args.min_tau].sort_values(['batch_size', 'tau'])

    floor = float(df['FCD_Train_Test'].iloc[0])
    floor_thresh = floor * (1.0 + args.rel_tol)

    print("=" * 78)
    print(f"Generalization-window analysis  (N={args.N}, n_feat=256)")
    print(f"  real–real floor  FCD(Train,Test) = {floor:.3f}")
    print(f"  'near floor' band: FCD(Gen,Test) ≤ {floor_thresh:.3f}"
          f"  (= floor × {1 + args.rel_tol:g})")
    print(f"  memorization onset: f_mem crosses {args.fmem_thresh:g}")
    print("=" * 78)

    header = (f"{'batch':>6} | {'τ_open':>9} | {'τ_close':>9} | {'Δτ (steps)':>11} | "
              f"{'Δτ / τ_open':>11}")
    print(header)
    print("-" * len(header))

    rows = []
    for bs, g in df.groupby('batch_size'):
        g = g.sort_values('tau')
        tau = g['tau'].to_numpy()
        fcd = g['FCD_Gen_Test'].to_numpy()
        fmem = g['f_mem'].to_numpy()

        tau_open = _log_interp_crossing(tau, fcd, floor_thresh, descending=True)
        tau_close = _log_interp_crossing(tau, fmem, args.fmem_thresh, descending=False)
        dtau = tau_close - tau_open
        rows.append((bs, tau_open, tau_close, dtau))
        print(f"{bs:>6} | {tau_open:>9.0f} | {tau_close:>9.0f} | {dtau:>11.0f} | "
              f"{dtau / tau_open:>11.2f}")

    print("-" * len(header))
    dtaus = np.array([r[3] for r in rows], float)
    print(f"Δτ across batch sizes:  mean = {dtaus.mean():.0f}   "
          f"std = {dtaus.std(ddof=0):.0f}   "
          f"spread(max/min) = {dtaus.max() / dtaus.min():.2f}×")
    print("=" * 78)


if __name__ == "__main__":
    main()
