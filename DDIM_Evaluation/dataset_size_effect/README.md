# Dataset-Size Sweep on Sample Quality & Memorization (W = 256, B = min(N, 500))

**The main scaling experiment.** Hold the model fixed (U-Net width `n_feat = 256`)
and sweep the **dataset size**

$$N \in \{200,\; 500,\; 1000,\; 2000,\; 4000\}$$

For every `N`, the training mini-batch size follows the fixed protocol

$$B = \min(N,\, 500)$$

and for every `N` we measure, as a function of training time `τ` (optimizer steps),
the sample-quality curve **FCD(Gen, Test)**, its real–real floor **FCD(Train, Test)**,
and the **memorization** curve **f_mem(τ)**.

This is the experiment that tests the paper's central prediction that the onset of
memorization scales with the number of *unique* training points `N` (larger `N` ⇒
memorization is pushed to later `τ`), at a **fixed model capacity**.

---

## 1. The batch-size rule `B = min(N, 500)` and why it is fair

| N    | B = min(N, 500) | steps / epoch = ⌈N/B⌉ | regime                         |
| ---- | --------------- | --------------------- | ------------------------------ |
| 200  | 200             | 1                     | full-batch (B = N)             |
| 500  | 500             | 1                     | full-batch (B = N)             |
| 1000 | 500             | 2                     | capped batch                   |
| 2000 | 500             | 4                     | capped batch                   |
| 4000 | 500             | 8                     | capped batch                   |

**Why cap the batch at 500?** The batch size sets the **gradient-noise scale** of
SGD. If we let the batch grow with `N` (e.g. full-batch everywhere) then two things
would change at once — the dataset size *and* the optimization noise — and we could
not attribute a shift in memorization onset to `N` alone. Freezing `B = 500` for all
`N ≥ 500` keeps the per-step gradient-noise scale **identical** across the large sets,
so any change in the curves is due to the **dataset size**, which is exactly the
variable this experiment isolates. For the small sets (`N < 500`) a full-batch update
is the natural lowest-noise choice, so `B = N`.

**Why τ (optimizer steps) is the x-axis.** The dense `τ` grid is defined in *step*
units and is identical for every run, so all five sizes are checkpointed at exactly
the same `τ` values and overlay directly. (For `N ≥ 1000`, `τ` is not equal to epochs
because ⌈N/500⌉ > 1; the CSV also stores `epoch_float = τ / steps_per_epoch` if an
epoch axis is ever needed.)

Everything else is held identical across all five runs:

| Held fixed                         | Value                                             |
| ---------------------------------- | ------------------------------------------------- |
| Model width `n_feat`               | 256                                               |
| Diffusion steps `N_T`              | 200                                               |
| Learning rate                      | 1e-4 (Adam, fixed)                                |
| EMA base decay                     | 0.9999 (warmup schedule)                          |
| Max τ                              | 200000 optimizer steps                            |
| τ grid                             | dense grid (≈6 pts/decade), same for all runs     |
| Train / test split                 | shared master shuffle (nested subsets)            |
| **Varied**                         | **dataset size N (with B = min(N, 500))**         |

---

## 2. Which runs already exist vs. which you must train

The trainer writes each `(N, B)` run to its own log directory. Width 256 is the
trainer default (no `_nfeat` suffix); the batch size adds a `_bs<B>` suffix.

| N    | B   | Log directory                                          | Status                         |
| ---- | --- | ------------------------------------------------------ | ------------------------------ |
| 200  | 200 | `logs_ema/DDIM_tau_ema_200_bs200_incremental`          | **already trained** (= the N=200 full-batch run — reused verbatim) |
| 500  | 500 | `logs_ema/DDIM_tau_ema_500_bs500_incremental`          | **you train this**             |
| 1000 | 500 | `logs_ema/DDIM_tau_ema_1000_bs500_incremental`         | **you train this**             |
| 2000 | 500 | `logs_ema/DDIM_tau_ema_2000_bs500_incremental`         | **you train this**             |
| 4000 | 500 | `logs_ema/DDIM_tau_ema_4000_bs500_incremental`         | **you train this**             |

Because `N = 200` with `B = min(200,500) = 200` is exactly the batch used by the
earlier full-batch experiment, its checkpoints **and** its cached EMA samples are
reused with no retraining and no regeneration.

---

## 3. Files in this folder

| File                                                     | Role                                                                                    |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [compute_dsize_fcd_fmem.py](compute_dsize_fcd_fmem.py)   | Computes FCD(Gen,Test), the per-N real–real floor FCD(Train,Test), and f_mem(τ) per N   |
| [plot_dsize_fcd_fmem.py](plot_dsize_fcd_fmem.py)         | Draws the paper-style overlay (FCD solid, floor dotted, f_mem dashed), one color per N  |
| `results/dsize_fcd_fmem.csv`                             | One row per (N × τ checkpoint)                                                           |
| `results/figures/dsize_fcd_fmem.png / .pdf`             | Final figure                                                                            |

**No metric logic is re-implemented here.** The compute script imports the exact,
already-validated routines from the parent evaluation scripts:

- FCD + generation/caching ← [`../compute_fcd_vs_tau.py`](../compute_fcd_vs_tau.py)
- nearest-neighbour ratio → f_mem ← [`../compute_fmem_vs_tau.py`](../compute_fmem_vs_tau.py)

Both metrics use the **identical 256-D representation** (per-sample-normalized UPA-DFT
beamspace, flattened) and the **identical EMA-generated samples**, so for a given `N`
the FCD and f_mem curves are mutually self-consistent, and the five sizes are mutually
comparable.

---

## 4. Metrics (one-line definitions)

- **FCD(Gen, Test)** — Fréchet Channel Distance between generated and held-out test
  channels in 256-D beamspace: `||μ_g − μ_r||² + Tr(Σ_g + Σ_r − 2(Σ_g Σ_r)^½)`.
  Lower = better quality. Error band = 2σ over 5 disjoint test folds.
- **FCD(Train, Test)** — the same distance between the *training* set and the test
  set: the irreducible **real–real floor**. This is drawn per `N` (each `N` has a
  different training set, hence a different floor).
- **f_mem(τ)** — fraction of generated samples whose nearest-neighbour ratio
  `ρ = d₁/d₂ < 1/3` against the training set (memorization test), on the same samples.

The **generalization window** for each `N` is the `τ`-band where FCD(Gen,Test) has
descended onto its floor while f_mem(τ) is still ≈ 0. The paper's prediction is that
this window **starts at roughly the same place but ends later for larger `N`**
(memorization onset moves right with `N`).

---

## 5. How to reproduce (full pipeline)

```bash
conda activate Mem_Gen

# ── 1. Train the four new sizes (W=256, B=500, same dense τ grid) ─────────────
#      (N=200 with B=200 is already trained — nothing to do there.)
cd Code/DDIM_FMM
python train_DDIM_tau_ema.py 500  --max_tau 200000 --incremental --batch_size 500
python train_DDIM_tau_ema.py 1000 --max_tau 200000 --incremental --batch_size 500
python train_DDIM_tau_ema.py 2000 --max_tau 200000 --incremental --batch_size 500
python train_DDIM_tau_ema.py 4000 --max_tau 200000 --incremental --batch_size 500

# ── 2. Compute FCD + f_mem for all five sizes (N=200 reuses its cache) ────────
cd DDIM_Evaluation/dataset_size_effect
python compute_dsize_fcd_fmem.py --sizes 200 500 1000 2000 4000

# ── 3. Plot the overlay ───────────────────────────────────────────────────────
python plot_dsize_fcd_fmem.py
```

Output figure: `results/figures/dsize_fcd_fmem.png`.

> **Tip.** Each training run is independent; you can launch them one at a time or on
> separate GPUs. The compute step will simply skip any size whose log directory is
> not yet present, so you can also run steps 2–3 incrementally as each model finishes.

---

## 6. Reading the figure

- **Color** = dataset size `N` (200 blue, 500 orange, 1000 green, 2000 purple, 4000 red).
- **Solid** = FCD(Gen,Test) quality (left axis, ↓ better), with a 2σ band.
- **Dotted** = each `N`'s own real–real floor.
- **Dashed** = f_mem(τ) memorization (right axis).

Compare, across `N`: at what `τ` each solid curve reaches its floor (onset of good
generation), and at what `τ` its dashed f_mem curve lifts off zero (onset of
memorization). The paper predicts memorization onset **grows with `N`**, so the dashed
curves should fan out to the right as `N` increases while every model shares the same
fixed capacity.
