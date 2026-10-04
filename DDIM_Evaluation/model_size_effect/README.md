# Model-Size Effect — DDIM Sample Fidelity & Memorization vs Model Width W

Companion experiment to [`../dataset_size_effect/`](../dataset_size_effect/). There
we fixed the model width (**W = 256**) and swept the dataset size **N**. Here we do
the **orthogonal** experiment: fix a limited set of dataset sizes and sweep the
**model capacity (U-Net width W)** to see how capacity alone moves sample quality
(FCD) and memorization (`f_mem`).

## Experiment design

| Knob            | Values                                    |
| --------------- | ----------------------------------------- |
| Dataset size N  | **200, 1000** (core sweep) + **50, 500, 2000, 4000** (phase-diagram completion) |
| Model width W   | **64, 128, 256** (= U-Net `n_feat`)       |
| Batch size B    | **min(N, 500)**  → N=200→B=200, N≥500→B=500 |
| Weights         | EMA                                        |
| Metrics         | FCD(Gen,Test), FCD(Train,Test) floor, `f_mem` (%) |

The batch rule **B = min(N, 500)** is the SAME as the dataset-size sweep, so the
two experiments are mutually consistent and the **W = 256** runs are literally the
same checkpoints already trained there (reused, not retrained).

### Training budget (τ_max) per config

Most configs train to **τ = 200 000**. The two **low-capacity N = 1000** runs need
longer to reach/traverse memorization, so they are **extended to τ = 400 000**:

| (N, W)       | τ_max       | Log directory                                          | Status |
| ------------ | ----------- | ------------------------------------------------------ | ------ |
| (200,  64)   | 200 000     | `logs_ema/DDIM_tau_ema_200_nfeat64_bs200_incremental`  | train  |
| (200,  128)  | 200 000     | `logs_ema/DDIM_tau_ema_200_nfeat128_bs200_incremental` | train  |
| (200,  256)  | 200 000     | `logs_ema/DDIM_tau_ema_200_bs200_incremental`          | **reused** |
| (1000, 64)   | **400 000** | `logs_ema/DDIM_tau_ema_1000_nfeat64_bs500_incremental` | train  |
| (1000, 128)  | **400 000** | `logs_ema/DDIM_tau_ema_1000_nfeat128_bs500_incremental`| train  |
| (1000, 256)  | 200 000     | `logs_ema/DDIM_tau_ema_1000_bs500_incremental`         | **reused** |

> Width **W = 256** is the trainer default, so those two directories carry **no**
> `_nfeat` suffix. All other widths get a `_nfeat<W>` suffix. Every run here uses a
> non-default batch size, so all carry a `_bs<B>` suffix.

---

## Step 1 — Train the 4 new models

Only the four non-256 configs need training (W = 256 is reused from the
dataset-size sweep). Run from the trainer directory:

```bash
conda activate Mem_Gen
cd Code/DDIM_FMM

# ── N = 200 (B = 200), τ_max = 200000 ──────────────────────────────────────
python train_DDIM_tau_ema.py 200  --incremental --batch_size 200 --n_feat 64  --max_tau 200000
python train_DDIM_tau_ema.py 200  --incremental --batch_size 200 --n_feat 128 --max_tau 200000

# ── N = 1000 (B = 500), τ_max = 400000  (extended for low capacity) ────────
python train_DDIM_tau_ema.py 1000 --incremental --batch_size 500 --n_feat 64  --max_tau 400000
python train_DDIM_tau_ema.py 1000 --incremental --batch_size 500 --n_feat 128 --max_tau 400000
```

Notes:
- **Resuming.** If a run is interrupted, resume it (do **not** restart from scratch)
  by pointing `--resume_from` at the latest checkpoint, e.g.:
  ```bash
  python train_DDIM_tau_ema.py 1000 --incremental --batch_size 500 --n_feat 64 --max_tau 400000 \
      --resume_from logs_ema/DDIM_tau_ema_1000_nfeat64_bs500_incremental/checkpoints/checkpoint_tau_200000.pth
  ```
  Only τ targets beyond the checkpoint's step are (re)evaluated.
- The **W = 256** models are already trained; do **not** retrain them.

---

## Step 1b — Phase-diagram completion runs (N = 500, 2000, 4000)

The paper-style phase diagram in [`../phase_diagram/`](../phase_diagram/) needs the
narrow widths **W ∈ {64, 128}** evaluated at more dataset sizes so its
memorizing/generalizing boundary `n*(p)` is a real rising curve rather than a
2-point (capped) estimate. These **6 extra runs** fill the grid at
**N ∈ {500, 2000, 4000}**. Batch is still **B = min(N, 500) = 500**, and each N uses
the **same `τ_max` as its W = 256 counterpart** in the dataset-size sweep, so the
per-N τ grids stay identical and directly comparable.

| (N, W)       | τ_max       | Log directory                                          | Status |
| ------------ | ----------- | ------------------------------------------------------ | ------ |
| (50,   64)   | 200 000     | `logs_ema/DDIM_tau_ema_50_nfeat64_bs50_incremental`    | train  |
| (50,   128)  | 200 000     | `logs_ema/DDIM_tau_ema_50_nfeat128_bs50_incremental`   | train  |
| (50,   256)  | 200 000     | `logs_ema/DDIM_tau_ema_50_bs50_incremental`            | train  |
| (500,  64)   | 200 000     | `logs_ema/DDIM_tau_ema_500_nfeat64_bs500_incremental`  | train  |
| (500,  128)  | 200 000     | `logs_ema/DDIM_tau_ema_500_nfeat128_bs500_incremental` | train  |
| (500,  256)  | 200 000     | `logs_ema/DDIM_tau_ema_500_bs500_incremental`          | **reused** |
| (2000, 64)   | 300 000     | `logs_ema/DDIM_tau_ema_2000_nfeat64_bs500_incremental` | train  |
| (2000, 128)  | 300 000     | `logs_ema/DDIM_tau_ema_2000_nfeat128_bs500_incremental`| train  |
| (2000, 256)  | 300 000     | `logs_ema/DDIM_tau_ema_2000_bs500_incremental`         | **reused** |
| (4000, 64)   | 400 000     | `logs_ema/DDIM_tau_ema_4000_nfeat64_bs500_incremental` | train  |
| (4000, 128)  | 400 000     | `logs_ema/DDIM_tau_ema_4000_nfeat128_bs500_incremental`| train  |
| (4000, 256)  | 400 000     | `logs_ema/DDIM_tau_ema_4000_bs500_incremental`         | **reused** |

Train the **6 new narrow-width models** (W = 256 at these N is reused, not retrained):

```bash
conda activate Mem_Gen
cd Code/DDIM_FMM

# ── N = 50 (B = 50), all three widths ──────────────────────────────────
python train_DDIM_tau_ema.py 50 --incremental --batch_size 50 --n_feat 64  --max_tau 200000
python train_DDIM_tau_ema.py 50 --incremental --batch_size 50 --n_feat 128 --max_tau 200000
python train_DDIM_tau_ema.py 50 --incremental --batch_size 50 --n_feat 256 --max_tau 200000

# ── W = 64  (B = 500) ──────────────────────────────────────────────────────
python train_DDIM_tau_ema.py 500  --incremental --batch_size 500 --n_feat 64  --max_tau 200000
python train_DDIM_tau_ema.py 2000 --incremental --batch_size 500 --n_feat 64  --max_tau 300000
python train_DDIM_tau_ema.py 4000 --incremental --batch_size 500 --n_feat 64  --max_tau 400000

# ── W = 128 (B = 500) ──────────────────────────────────────────────────────
python train_DDIM_tau_ema.py 500  --incremental --batch_size 500 --n_feat 128 --max_tau 200000
python train_DDIM_tau_ema.py 2000 --incremental --batch_size 500 --n_feat 128 --max_tau 300000
python train_DDIM_tau_ema.py 4000 --incremental --batch_size 500 --n_feat 128 --max_tau 400000
```

The `compute_wsize_fcd_fmem.py` grid (`DEFAULT_N = [200, 500, 1000, 2000, 4000]`)
already includes these configs and **silently skips any run not yet trained**
(WARNING + continue), so you can compute partial results at any time and the CSV
fills in as runs finish.

---

## Step 2 — Compute FCD + `f_mem` vs τ

From this folder. Generation of the 5000 EMA samples happens once per checkpoint
and is cached in each run's `generated_ema/`; re-runs are near-instant.

```bash
conda activate Mem_Gen
cd DDIM_Evaluation/model_size_effect

# All (N, W) configs present in logs_ema → results/wsize_fcd_fmem.csv
# (untrained configs are skipped automatically with a WARNING)
python compute_wsize_fcd_fmem.py

# Resume an interrupted sweep WITHOUT losing the rows already written:
#   the CSV is truncated on a normal run, but --append keeps it and only adds
#   the requested configs (header written only when the file does not exist).
python compute_wsize_fcd_fmem.py --append --configs 2000:64 2000:128 4000:64

# Write to a separate CSV instead of the default one:
python compute_wsize_fcd_fmem.py --out wsize_fcd_fmem_extra.csv --configs 50:64

# (optional) a single config, or a subset:
python compute_wsize_fcd_fmem.py --configs 1000:64
python compute_wsize_fcd_fmem.py --configs 200:64 200:128 200:256

# (optional) quick smoke test on a few τ points:
python compute_wsize_fcd_fmem.py --debug
```

**Metrics (identical to the dataset-size sweep):**
- **FCD(Gen,Test)** — Fréchet Channel Distance on 256-D per-sample-normalized
  UPA-DFT beamspace features; **±2σ** over **5 disjoint test folds**.
- **FCD(Train,Test)** — the real–real floor for that N (shared across W: same train
  set), also with a ±2σ fold band.
- **`f_mem`** — fraction of the 5000 generated samples with nearest-neighbour ratio
  ρ = d₁/d₂ < 1/3 against the training set, with a **95 % bootstrap CI** (1000
  resamples, seed 42) stored as `f_mem_ci_low` / `f_mem_ci_high`.

Output: `results/wsize_fcd_fmem.csv` (one row per (N, W) × τ checkpoint).

---

## Step 3 — Plot (one figure per N, color = width W)

```bash
conda activate Mem_Gen
cd DDIM_Evaluation/model_size_effect

# One figure per dataset size N present in the CSV
python plot_wsize_fcd_fmem.py

# (optional) restrict to a single N or a subset of widths
python plot_wsize_fcd_fmem.py --sizes 1000
python plot_wsize_fcd_fmem.py --widths 64 128
```

Each figure (`results/figures/wsize_fcd_fmem_N<N>.png/.pdf`) shows, for that N:
- **solid** FCD(Gen,Test) per width W (color) with a 2σ band — sample quality (↓);
- one **dotted** black real–real floor FCD(Train,Test) (shared by all W at that N);
- **dashed** `f_mem`(%) per width on the right axis with a 95 % bootstrap-CI band.

Two stacked legends: width W (color) on top, quantity (line style) directly below.
Styling matches `../tau_plots/` (axis labels 20, ticks 20, legend 16).

---

## Files

| File                          | Purpose                                             |
| ----------------------------- | --------------------------------------------------- |
| `compute_wsize_fcd_fmem.py`   | Compute FCD + `f_mem` vs τ for all (N, W) configs   |
| `plot_wsize_fcd_fmem.py`      | One FCD+`f_mem` figure per N (color = width W)       |
| `results/wsize_fcd_fmem.csv`  | Metrics table (one row per (N, W) × τ)              |
| `results/figures/`            | Output PNG/PDF figures                              |

All heavy lifting (FCD, generation/caching, beamspace features, `f_mem`, bootstrap
CI) is imported verbatim from the validated parent scripts
`../compute_fcd_vs_tau.py` and `../compute_fmem_vs_tau.py`; no metric logic is
re-implemented here. The only model-size-specific addition is a **width-aware model
build** (`create_model(n_feat=W)`), because the network width now varies per config.
