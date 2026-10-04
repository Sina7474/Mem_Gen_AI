"""
analyze_generalization_window.py
    — Numerical generalization-window duration (in τ) per dataset size N
================================================================================
For a fixed model width (W = 256) and the batch protocol B = min(N, 500), each
dataset size passes through the same three phases as training time τ grows:

    (1) UNDER-FIT   : FCD(Gen,Test) is high, samples are noise            (f_mem ≈ 0)
    (2) GENERALISE  : FCD(Gen,Test) sits on that N's real–real floor      (f_mem ≈ 0)
    (3) MEMORISE    : FCD(Gen,Test) drifts up, samples copy the trainset  (f_mem ↑)

The **generalization window** is the τ-band of phase (2): it opens when FCD(Gen,Test)
first descends onto that N's real–real floor (good samples) and closes when
memorization begins.  Because every N has a DIFFERENT training set, the floor
FCD(Train,Test) is taken per-N (not shared).

    τ_open   = first τ (log-interpolated) where FCD(Gen,Test) ≤ floor_N · (1 + rel_tol)
    τ_close  = τ (log-interpolated) where f_mem(τ) first crosses `fmem_thresh`
    Δτ       = τ_close − τ_open        (window duration, in optimizer steps)

The paper's prediction (t_mem grows with the number of unique samples N at fixed
capacity) is that τ_close — and hence the window — moves to LARGER τ as N grows.

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/dataset_size_effect
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
        description="Numerical generalization-window duration (τ) per dataset size N")
    ap.add_argument("--fmem_thresh", type=float, default=0.1,
                    help="f_mem value that marks the onset of memorization")
    ap.add_argument("--rel_tol", type=float, default=0.5,
                    help="'near floor' band: FCD ≤ floor_N·(1+rel_tol). Default 0.5 = 50%%")
    ap.add_argument("--min_tau", type=int, default=1,
                    help="Ignore τ below this (drops the τ=0 random-init spike)")
    ap.add_argument("--csv_path", type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv_path or os.path.join(base, 'results', 'dsize_fcd_fmem.csv')
    df = pd.read_csv(csv_path)
    df = df[df['tau'] >= args.min_tau].sort_values(['N', 'tau'])

    print("=" * 92)
    print(f"Generalization-window analysis  (W=256, B=min(N,500))")
    print(f"  'near floor' band : FCD(Gen,Test) ≤ floor_N × {1 + args.rel_tol:g}"
          f"   (floor_N is that N's own FCD(Train,Test))")
    print(f"  memorization onset: f_mem crosses {args.fmem_thresh:g}")
    print("=" * 92)

    header = (f"{'N':>6} | {'B':>4} | {'floor_N':>8} | {'τ_open':>9} | "
              f"{'τ_close':>9} | {'Δτ (steps)':>11} | {'Δτ/τ_open':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for N, g in df.groupby('N'):
        g = g.sort_values('tau')
        tau  = g['tau'].to_numpy()
        fcd  = g['FCD_Gen_Test'].to_numpy()
        fmem = g['f_mem'].to_numpy()
        bs   = int(g['batch_size'].iloc[0])
        floor = float(g['FCD_Train_Test'].iloc[0])
        floor_thresh = floor * (1.0 + args.rel_tol)

        tau_open  = _log_interp_crossing(tau, fcd, floor_thresh, descending=True)
        tau_close = _log_interp_crossing(tau, fmem, args.fmem_thresh, descending=False)
        dtau = tau_close - tau_open
        rows.append((N, bs, floor, tau_open, tau_close, dtau))
        ratio = dtau / tau_open if tau_open and not np.isnan(tau_open) else float('nan')
        print(f"{N:>6} | {bs:>4} | {floor:>8.3f} | {tau_open:>9.0f} | "
              f"{tau_close:>9.0f} | {dtau:>11.0f} | {ratio:>9.2f}")

    print("-" * len(header))
    dtaus = np.array([r[5] for r in rows], float)
    print(f"Δτ across N:  mean = {dtaus.mean():.0f}   "
          f"std = {dtaus.std(ddof=0):.0f}   "
          f"spread(max/min) = {dtaus.max() / dtaus.min():.2f}×")
    print("=" * 92)


if __name__ == "__main__":
    main()
