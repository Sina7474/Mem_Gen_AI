# FullDim_Distance — Full 256-D Distribution Distance (Sinkhorn-OT + Sliced-Wasserstein)

Compare the **DDIM-generated** channel distribution against the **held-out TEST**
distribution **directly in the full 256-D beamspace**, without collapsing each
sample to a scalar.

Every channel `H ∈ (2,4,32)` is flattened to a vector `x ∈ R^256`. We then measure
the distance between the two 256-D empirical distributions with two complementary
multivariate optimal-transport metrics:

| Metric | What it is | Why |
|--------|-----------|-----|
| **Sinkhorn-W2** | Entropic-regularized 2-Wasserstein (true multivariate OT), stabilized log-domain GPU solver | Full joint 256-D geometry, tractable for n≈5000 |
| **SWD-W2** | Sliced-Wasserstein-2: average of 1-D W2 over many random 256-D projections | Exact multivariate generalization of the **1-D Wasserstein** you used on effective-rank |

### Why this (vs FCD / effective-rank)
- **effective-rank** → 1 scalar per sample → 1-D Wasserstein (information collapsed).
- **FCD** → mean + covariance → Gaussian assumption (information collapsed).
- **Here** → the full 256-D distribution is kept; nothing is reduced to a scalar.

### Data reuse (directly comparable to FCD)
`compute_fulldim_distance_vs_tau.py` **imports** `../compute_fcd_vs_tau.py` and reuses
its functions, so the inputs are **identical** to the FCD run:
- **Generated** samples: the SAME cached `generated_ema/gen_ema_tau*_seed0.npz`.
- **Reference**: the SAME `test.npy` / `train.npy` → beamspace → per-sample norm → flatten 256.
- **Protocol**: TEST split into **5 disjoint folds** → 2σ error bars.
  - `Metric(Gen , Test)` = quality curve (want LOW, saturating).
  - `Metric(Train, Test)` = real–real floor (irreducible finite-sample distance).

## Files
- `compute_fulldim_distance_vs_tau.py` — computes both metrics vs τ → `results/fulldim_distance_vs_tau_<weights>.csv`
- `plot_fulldim_distance_vs_tau.py` — Figure-2-style plots (2 panels + per-metric figures), optional `f_mem` overlay

## Run

```bash
conda activate Mem_Gen
cd DDIM_Evaluation/FullDim_Distance

# 1) Quick smoke test (few taus, few samples) — verify everything works
python compute_fulldim_distance_vs_tau.py --debug

# 2) Full compute (EMA weights, N = 200 1000 4000; reuses cached generated data)
python compute_fulldim_distance_vs_tau.py

# 3) Plot both metrics with the f_mem overlay (same f_mem CSV as FCD)
python plot_fulldim_distance_vs_tau.py --fmem_csv ../results/fmem_vs_tau/fmem_vs_tau.csv
```

Outputs land in `results/` (CSV) and `results/figures/` (PNG + PDF).

### Useful options
```bash
# one size only
python compute_fulldim_distance_vs_tau.py --sizes 1000

# tune the metrics
python compute_fulldim_distance_vs_tau.py --sinkhorn_reg_frac 0.05 --sinkhorn_iters 300 \
                                          --n_proj 512 --n_quantiles 256
```

## Notes
- GPU is used automatically if available (Sinkhorn cost matrix + SWD projections).
- `Sinkhorn-W2` uses an auto-scaled entropic regularization `ε = reg_frac × median(cost)`
  so it is robust to the beamspace magnitude scale.
- `SWD-W2` uses quantile matching, so the generated (5000) and each test fold need
  not have equal sample counts.
