"""
build_phase_diagram_table.py
    — Aggregate quality-vs-τ and f_mem-vs-τ curves into a generalization–memorization
      phase-diagram summary (Task 16).
================================================================================
For every trained DDIM configuration (model width W, training-set size N) this
script estimates two characteristic training times, in optimizer-update units τ:

    τ_gen  = first τ at which the generated-channel QUALITY first reaches the
             real–real floor (the model has "generalized" / samples look real);
    τ_mem  = first τ at which the memorization fraction f_mem first exceeds a
             small threshold (the model has started to "memorize" training data).

The generalization window is [τ_gen, τ_mem]. A configuration is SAFE when the
model becomes high-quality BEFORE it starts to memorize (τ_mem > τ_gen, i.e.
f_mem(τ_gen) < threshold), and UNSAFE otherwise.

--------------------------------------------------------------------------------
Data sources (mapped to the task's abstract column names)
--------------------------------------------------------------------------------
We do not have a dedicated BS-BFD file; our FID-style sample-quality metric is the
Fréchet Channel Distance (FCD) computed on channel features. We therefore use:

    quality_gen_test        <-  FCD_Gen_Test         (generated vs held-out test)
    quality_floor_real_real <-  FCD_Train_Test       (train vs test = real–real floor)
    quality_metric_name     =   "FCD"
    f_mem                   <-  f_mem
    f_mem_ci_low/high       <-  f_mem_ci_low/high

Two CSVs are merged into a single long table keyed by (W, N, τ):

  • ../dataset_size_effect/results/dsize_fcd_fmem.csv
        width is implicit (n_feat = 256); N ∈ {200, 500, 1000, 2000, 4000}.
        This is the canonical W = 256 dataset-size sweep.
  • ../model_size_effect/results/wsize_fcd_fmem.csv
        explicit (N, W); N ∈ {200, 1000}, W ∈ {64, 128, 256}.
        Used for W ∈ {64, 128}. Its W = 256 rows are redundant with the sweep
        above and are dropped by default (--prefer_dsize_for_w256).

--------------------------------------------------------------------------------
Available (W, N) grid  (sparse — do NOT pretend it is dense)
--------------------------------------------------------------------------------
    W = 64   (p ≈ 0.96 M) :  N = 200, 1000
    W = 128  (p ≈ 3.81 M) :  N = 200, 1000
    W = 256  (p ≈ 15.23 M):  N = 200, 500, 1000, 2000, 4000

--------------------------------------------------------------------------------
Outputs
--------------------------------------------------------------------------------
    results/phase_diagram_summary.csv       one row per (W, N) with τ_gen, τ_mem,
                                            f_mem(τ_gen / 3τ_gen / 8τ_gen), labels.
    results/phase_diagram_sensitivity.csv   the same classification recomputed over
                                            a grid of (QUALITY_TOL, FMEM_THRESHOLD).

Usage:
    conda activate Mem_Gen
    cd DDIM_Evaluation/phase_diagram
    python build_phase_diagram_table.py
    python build_phase_diagram_table.py --fmem_threshold 0.01 --quality_tol 0.10
"""

import os
import argparse
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants --
# Trainable-parameter counts of the U-Net for each base width W (n_feat), obtained
# with sum(p.numel() for p in Unet(in_channels=2, n_feat=W).parameters()).
PARAM_COUNTS = {
    64:  956930,
    128: 3814402,
    256: 15230978,
}

# Main thresholds (see Task 16 §4.4, §4.5). τ = optimizer updates.
QUALITY_TOL_MAIN = 0.10       # quality is "good" when ≤ floor * (1 + 0.10)
STABILITY_TOL    = 0.25       # the next checkpoint must still be ≤ floor*(1+0.25)
FMEM_THRESHOLD_MAIN = 0.01    # memorization "started" when f_mem ≥ 1%

# Sensitivity grids (Task 16 §13).
QUALITY_TOL_VALUES     = [0.05, 0.10, 0.20]
FMEM_THRESHOLD_VALUES  = [0.001, 0.01, 0.05]


# --------------------------------------------------------------- data loading --
def load_long_table(dsize_csv, wsize_csv, prefer_dsize_for_w256=True):
    """Merge the two source CSVs into one long table with columns
    [W, N, tau, quality_gen_test, quality_floor_real_real, f_mem,
     f_mem_ci_low, f_mem_ci_high]."""
    frames = []

    # dataset_size_effect: width is implicit in n_feat (all 256 here).
    if os.path.exists(dsize_csv):
        d = pd.read_csv(dsize_csv)
        d = d.rename(columns={
            'FCD_Gen_Test':   'quality_gen_test',
            'FCD_Train_Test': 'quality_floor_real_real',
        })
        d['W'] = d['n_feat'].astype(int)          # 256 for this sweep
        frames.append(d[['W', 'N', 'tau', 'quality_gen_test',
                         'quality_floor_real_real', 'f_mem',
                         'f_mem_ci_low', 'f_mem_ci_high']])
    else:
        print(f"  WARNING: dataset-size CSV not found: {dsize_csv}")

    # model_size_effect: explicit (N, W).
    if os.path.exists(wsize_csv):
        w = pd.read_csv(wsize_csv)
        w = w.rename(columns={
            'FCD_Gen_Test':   'quality_gen_test',
            'FCD_Train_Test': 'quality_floor_real_real',
        })
        w['W'] = w['W'].astype(int)
        if prefer_dsize_for_w256:
            # Drop the W=256 rows here; the dataset-size sweep is the canonical
            # source for W=256 (avoids duplicate (W,N,tau) keys).
            w = w[w['W'] != 256]
        frames.append(w[['W', 'N', 'tau', 'quality_gen_test',
                        'quality_floor_real_real', 'f_mem',
                        'f_mem_ci_low', 'f_mem_ci_high']])
    else:
        print(f"  WARNING: model-size CSV not found: {wsize_csv}")

    if not frames:
        raise FileNotFoundError("No source CSVs found; cannot build phase diagram.")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=['W', 'N', 'tau'])
    df = df.sort_values(['W', 'N', 'tau']).reset_index(drop=True)
    return df


# ------------------------------------------------------------- core estimators --
def estimate_tau_gen(tau, quality, floor, quality_tol, stability_tol):
    """First τ (excluding τ=0) where quality reaches the real–real floor and the
    NEXT checkpoint does not strongly degrade (Task 16 §4.4).

    Returns (tau_gen, quality_at_tau_gen, rule_str) or (nan, nan, reason)."""
    good_thr   = floor * (1.0 + quality_tol)
    stable_thr = floor * (1.0 + stability_tol)

    order = np.argsort(tau)
    tau, quality = tau[order], quality[order]
    # Exclude τ = 0 (random-init spike) from the search.
    mask = tau > 0
    tau, quality = tau[mask], quality[mask]
    if len(tau) == 0:
        return np.nan, np.nan, 'no_positive_tau'

    for k in range(len(tau)):
        if quality[k] <= good_thr:
            is_last = (k == len(tau) - 1)
            if is_last or quality[k + 1] <= stable_thr:
                return float(tau[k]), float(quality[k]), \
                    f'quality<=floor*(1+{quality_tol}) & stable'
    return np.nan, np.nan, 'never_reached_floor'


def estimate_tau_mem(tau, fmem, fmem_threshold):
    """First τ (excluding τ=0) where f_mem ≥ threshold (Task 16 §4.5)."""
    order = np.argsort(tau)
    tau, fmem = tau[order], fmem[order]
    mask = tau > 0
    tau, fmem = tau[mask], fmem[mask]
    for k in range(len(tau)):
        if fmem[k] >= fmem_threshold:
            return float(tau[k])
    return np.nan


def fmem_at(tau_target, tau, fmem, monotone=False):
    """Log-linear interpolation of f_mem at τ_target (Task 16 §9.2).
    Returns NaN if τ_target is beyond the measured τ range (no extrapolation)."""
    if not np.isfinite(tau_target):
        return np.nan
    order = np.argsort(tau)
    tau, fmem = tau[order].astype(float), fmem[order].astype(float)
    mask = tau > 0
    tau, fmem = tau[mask], fmem[mask]
    if len(tau) == 0:
        return np.nan
    if monotone:
        fmem = np.maximum.accumulate(fmem)
    if tau_target < tau[0] or tau_target > tau[-1]:
        return np.nan            # do not extrapolate beyond [τ_min, τ_max]
    return float(np.interp(np.log10(tau_target), np.log10(tau), fmem))


# ------------------------------------------------------------- summary builder --
def build_summary(df, quality_tol, stability_tol, fmem_threshold, monotone=False):
    rows = []
    for (W, N), g in df.groupby(['W', 'N']):
        g = g.sort_values('tau')
        tau   = g['tau'].values.astype(float)
        qual  = g['quality_gen_test'].values.astype(float)
        fmem  = g['f_mem'].values.astype(float)
        floor = float(g['quality_floor_real_real'].iloc[0])  # shared per (W,N)

        tau_gen, q_at_gen, rule = estimate_tau_gen(
            tau, qual, floor, quality_tol, stability_tol)
        tau_mem = estimate_tau_mem(tau, fmem, fmem_threshold)

        fmem_gen  = fmem_at(tau_gen,           tau, fmem, monotone)
        fmem_3gen = fmem_at(3.0 * tau_gen,     tau, fmem, monotone)
        fmem_8gen = fmem_at(8.0 * tau_gen,     tau, fmem, monotone)

        tau_max      = float(tau[tau > 0].max()) if np.any(tau > 0) else np.nan
        fmem_tau_max = float(fmem[np.argmax(tau)])

        # Best observed quality (over τ>0) — a fallback anchor that is ALWAYS
        # defined even when the strict floor rule is never satisfied. This lets
        # every (W,N) point be placed and classified. tau_best is the τ at which
        # the generated-test quality is closest to the real-real floor.
        pos = tau > 0
        tau_p, qual_p = tau[pos], qual[pos]
        j = int(np.argmin(qual_p))
        tau_best        = float(tau_p[j])
        quality_best    = float(qual_p[j])
        quality_ratio_best = quality_best / floor if floor > 0 else np.nan
        fmem_best       = fmem_at(tau_best, tau, fmem, monotone)

        # Classification (safe = high quality reached before memorization).
        is_safe_gen = (np.isfinite(fmem_gen) and fmem_gen < fmem_threshold)
        is_safe_3   = (np.isfinite(fmem_3gen) and fmem_3gen < fmem_threshold)
        is_safe_8   = (np.isfinite(fmem_8gen) and fmem_8gen < fmem_threshold)

        # Unified phase label placing EVERY point (Task 16 §10.1 marker states):
        #   safe            : strict τ_gen reached, f_mem(τ_gen) < threshold
        #   unsafe          : strict τ_gen reached, f_mem(τ_gen) ≥ threshold
        #   near_floor_safe : strict τ_gen NOT reached (quality plateaus just
        #                     above floor*(1+tol)), but f_mem at best-quality τ
        #                     is still < threshold  → effectively generalizing
        #   near_floor_unsafe: same, but memorization already present at best τ
        if np.isfinite(tau_gen):
            phase_label = 'safe' if is_safe_gen else 'unsafe'
        else:
            safe_best = (np.isfinite(fmem_best) and fmem_best < fmem_threshold)
            phase_label = 'near_floor_safe' if safe_best else 'near_floor_unsafe'

        window_len   = (tau_mem - tau_gen) if (np.isfinite(tau_mem) and
                                               np.isfinite(tau_gen)) else np.nan
        window_ratio = (tau_mem / tau_gen) if (np.isfinite(tau_mem) and
                                               np.isfinite(tau_gen) and
                                               tau_gen > 0) else np.nan

        notes = []
        if not np.isfinite(tau_gen):
            notes.append('tau_gen not reached')
        if not np.isfinite(tau_mem):
            notes.append('tau_mem not reached (f_mem<thr over whole run)')
        if not np.isfinite(fmem_3gen) and np.isfinite(tau_gen):
            notes.append('3*tau_gen beyond tau_max')
        if not np.isfinite(fmem_8gen) and np.isfinite(tau_gen):
            notes.append('8*tau_gen beyond tau_max')

        rows.append({
            'W': int(W),
            'parameter_count_p': PARAM_COUNTS.get(int(W), np.nan),
            'N': int(N),
            'quality_metric_name': 'FCD',
            'quality_floor': floor,
            'tau_gen': tau_gen,
            'tau_gen_rule': rule,
            'quality_at_tau_gen': q_at_gen,
            'tau_best_quality': tau_best,
            'quality_at_best': quality_best,
            'quality_ratio_best_over_floor': quality_ratio_best,
            'f_mem_at_best_quality': fmem_best,
            'tau_mem_threshold': fmem_threshold,
            'tau_mem': tau_mem,
            'f_mem_at_tau_gen': fmem_gen,
            'f_mem_at_3tau_gen': fmem_3gen,
            'f_mem_at_8tau_gen': fmem_8gen,
            'is_safe_at_tau_gen': bool(is_safe_gen),
            'is_safe_at_3tau_gen': bool(is_safe_3),
            'is_safe_at_8tau_gen': bool(is_safe_8),
            'phase_label': phase_label,
            'window_length_tau': window_len,
            'window_ratio_tau_mem_over_tau_gen': window_ratio,
            'tau_max': tau_max,
            'f_mem_at_tau_max': fmem_tau_max,
            'is_no_mem_at_tau_max': bool(fmem_tau_max < fmem_threshold),
            'data_available': True,
            'notes': '; '.join(notes) if notes else '',
        })

    cols = ['W', 'parameter_count_p', 'N', 'quality_metric_name', 'quality_floor',
            'tau_gen', 'tau_gen_rule', 'quality_at_tau_gen', 'tau_best_quality',
            'quality_at_best', 'quality_ratio_best_over_floor',
            'f_mem_at_best_quality', 'tau_mem_threshold',
            'tau_mem', 'f_mem_at_tau_gen', 'f_mem_at_3tau_gen', 'f_mem_at_8tau_gen',
            'is_safe_at_tau_gen', 'is_safe_at_3tau_gen', 'is_safe_at_8tau_gen',
            'phase_label', 'window_length_tau',
            'window_ratio_tau_mem_over_tau_gen', 'tau_max',
            'f_mem_at_tau_max', 'is_no_mem_at_tau_max', 'data_available', 'notes']
    return pd.DataFrame(rows, columns=cols).sort_values(['W', 'N']).reset_index(drop=True)


def build_sensitivity(df, stability_tol, monotone=False):
    """Recompute τ_gen, τ_mem, safe label over the (QUALITY_TOL, FMEM_THRESHOLD)
    grid (Task 16 §13)."""
    rows = []
    for q_tol in QUALITY_TOL_VALUES:
        for f_thr in FMEM_THRESHOLD_VALUES:
            for (W, N), g in df.groupby(['W', 'N']):
                g = g.sort_values('tau')
                tau   = g['tau'].values.astype(float)
                qual  = g['quality_gen_test'].values.astype(float)
                fmem  = g['f_mem'].values.astype(float)
                floor = float(g['quality_floor_real_real'].iloc[0])

                tau_gen, _, _ = estimate_tau_gen(tau, qual, floor, q_tol,
                                                 stability_tol)
                tau_mem = estimate_tau_mem(tau, fmem, f_thr)
                fmem_gen = fmem_at(tau_gen, tau, fmem, monotone)
                is_safe = (np.isfinite(fmem_gen) and fmem_gen < f_thr)
                w_ratio = (tau_mem / tau_gen) if (np.isfinite(tau_mem) and
                          np.isfinite(tau_gen) and tau_gen > 0) else np.nan
                rows.append({
                    'W': int(W), 'N': int(N),
                    'quality_tol': q_tol, 'fmem_threshold': f_thr,
                    'tau_gen': tau_gen, 'tau_mem': tau_mem,
                    'is_safe': bool(is_safe), 'window_ratio': w_ratio,
                })
    return pd.DataFrame(rows).sort_values(
        ['quality_tol', 'fmem_threshold', 'W', 'N']).reset_index(drop=True)


# ---------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(
        description="Build the generalization–memorization phase-diagram tables.")
    ap.add_argument('--quality_tol', type=float, default=QUALITY_TOL_MAIN)
    ap.add_argument('--stability_tol', type=float, default=STABILITY_TOL)
    ap.add_argument('--fmem_threshold', type=float, default=FMEM_THRESHOLD_MAIN)
    ap.add_argument('--monotone_fmem', action='store_true',
                    help="Use cumulative-max (monotone) f_mem as a sensitivity check")
    ap.add_argument('--dsize_csv', type=str, default=None)
    ap.add_argument('--wsize_csv', type=str, default=None)
    ap.add_argument('--output_dir', type=str, default=None)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    dsize_csv = args.dsize_csv or os.path.join(
        base, '..', 'dataset_size_effect', 'results', 'dsize_fcd_fmem.csv')
    wsize_csv = args.wsize_csv or os.path.join(
        base, '..', 'model_size_effect', 'results', 'wsize_fcd_fmem.csv')
    out_dir = args.output_dir or os.path.join(base, 'results')
    os.makedirs(out_dir, exist_ok=True)

    print("Loading and merging source curves ...")
    df = load_long_table(dsize_csv, wsize_csv)
    grid = df.groupby('W')['N'].unique().apply(lambda a: sorted(a.tolist()))
    print("  Available (W -> N) grid:")
    for W, Ns in grid.items():
        print(f"    W={W} (p={PARAM_COUNTS.get(int(W), '?')}): N={Ns}")

    print(f"\nEstimating τ_gen / τ_mem  "
          f"(QUALITY_TOL={args.quality_tol}, STABILITY_TOL={args.stability_tol}, "
          f"FMEM_THRESHOLD={args.fmem_threshold}, monotone={args.monotone_fmem})")
    summary = build_summary(df, args.quality_tol, args.stability_tol,
                            args.fmem_threshold, args.monotone_fmem)
    sens = build_sensitivity(df, args.stability_tol, args.monotone_fmem)

    summary_path = os.path.join(out_dir, 'phase_diagram_summary.csv')
    sens_path    = os.path.join(out_dir, 'phase_diagram_sensitivity.csv')
    summary.to_csv(summary_path, index=False)
    sens.to_csv(sens_path, index=False)

    # Console preview of the key columns.
    show = ['W', 'N', 'tau_gen', 'tau_mem', 'f_mem_at_tau_gen',
            'phase_label', 'window_ratio_tau_mem_over_tau_gen',
            'quality_ratio_best_over_floor', 'f_mem_at_tau_max', 'notes']
    with pd.option_context('display.width', 160, 'display.max_columns', None):
        print("\n=== phase_diagram_summary.csv ===")
        print(summary[show].to_string(index=False))
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {sens_path}")
    print("Done.")


if __name__ == '__main__':
    main()
