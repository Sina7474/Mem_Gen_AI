# Batch-Size Effect on Sample Quality & Memorization (N = 1000)

**Question.** For a fixed dataset (N = 1000, LoS+NLoS) and a fixed model (U-Net width
`n_feat = 256`), how does the **training mini-batch size** change the sample-quality
curve **FCD(Gen, Test)** and the **memorization** curve **f_mem(τ)** as a function of
training time τ?

We compare three batch sizes: **100 (already trained), 512, 1024.**

---

## 1. Why this comparison is fair

The x-axis is **τ = number of optimizer (gradient) steps**, *not* epochs. The dense τ
grid used for checkpoints is defined in step units, so **every batch size is checkpointed
at exactly the same τ values** and the curves overlay directly.

Everything else is held identical across the three runs:

| Held fixed                         | Value                                             |
| ---------------------------------- | ------------------------------------------------- |
| Dataset size N                     | 1000 (500 LoS + 500 NLoS)                         |
| Train / test split                 | shared indices `logs/DDIM_tau_1000_incremental`   |
| Model width `n_feat`               | 256                                               |
| Diffusion steps `N_T`              | 200                                               |
| Learning rate                      | 1e-4 (Adam, fixed)                                |
| EMA base decay                     | 0.9999 (warmup schedule)                          |
| Max τ                              | 200000 optimizer steps                            |
| τ grid                             | dense grid (≈6 pts/decade), same for all runs     |
| **Varied**                         | **batch size ∈ {100, 512, 1024}**                 |

The **only** difference is how many samples the model sees per gradient update.
Because N = 1000, batch 1024 means every step is effectively a **full-batch** update;
batch 512 is two mini-batches per epoch; batch 100 is ten mini-batches per epoch.

> **Wall-clock note.** All three runs take the same number of gradient steps
> (200000), but a larger batch processes more samples per step, so bigger batches
> take proportionally longer in wall-clock time (batch 1024 sees ≈10× the samples
> per step vs. batch 100). This is expected and does not affect the τ axis.

---

## 2. Files in this folder

| File                                                                 | Role                                                                                   |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| [compute_bs_fcd_fmem.py](compute_bs_fcd_fmem.py)                      | Computes FCD(Gen,Test), the real–real floor FCD(Train,Test), and f_mem(τ) per batch size |
| [plot_bs_fcd_fmem.py](plot_bs_fcd_fmem.py)                            | Draws the paper-style overlay (FCD solid, floor dotted, f_mem dashed)                  |
| `results/bs_fcd_fmem_N1000.csv`                                       | One row per (batch size × τ checkpoint)                                                 |
| `results/figures/bs_fcd_fmem_N1000.png / .pdf`                        | Final figure                                                                           |

**No metric logic is re-implemented here.** The compute script imports the exact,
already-validated routines from the parent evaluation scripts:

- FCD + generation/caching ← `../compute_fcd_vs_tau.py`
- nearest-neighbour ratio → f_mem ← `../compute_fmem_vs_tau.py`

Both metrics use the **identical 256-D representation** (per-sample-normalized UPA-DFT
beamspace, flattened) and the **identical EMA-generated samples**, so for a given batch
size the FCD and f_mem curves are mutually self-consistent, and the three batch sizes are
mutually comparable.

---

## 3. Where the trained models live

The trainer `Code/DDIM_FMM/train_DDIM_tau_ema.py` now takes a `--batch_size` flag
(default 100, fully backward-compatible). Non-default batch sizes get their **own** log
directory so nothing overwrites the original run:

| Batch size | Log directory                                                       | Status                    |
| ---------- | ------------------------------------------------------------------- | ------------------------- |
| 100        | `logs_ema/DDIM_tau_ema_1000_incremental`                            | already trained (reused)  |
| 512        | `logs_ema/DDIM_tau_ema_1000_bs512_incremental`                      | **you train this**        |
| 1024       | `logs_ema/DDIM_tau_ema_1000_bs1024_incremental`                     | **you train this**        |

The batch-100 EMA samples are already cached in its `generated_ema/` folder, so
`compute_bs_fcd_fmem.py` reuses them verbatim — **nothing is regenerated for batch 100.**

---

## 4. Metrics (one-line definitions)

- **FCD(Gen, Test)** — Fréchet Channel Distance between generated and held-out test
  channels in 256-D beamspace: `||μ_g − μ_r||² + Tr(Σ_g + Σ_r − 2(Σ_g Σ_r)^½)`.
  Lower = better quality. Error band = 2σ over 5 disjoint test folds.
- **FCD(Train, Test)** — the same distance between the *training* set and the test set:
  the irreducible **real–real floor**. Identical for all batch sizes (same split), drawn
  once as a black dotted line.
- **f_mem(τ)** — fraction of generated samples whose nearest-neighbour ratio
  `ρ = d₁/d₂ < 1/3` against the training set (memorization test), on the same samples.

The **generalization window** for each batch size is the τ-band where FCD(Gen,Test) has
descended onto the floor while f_mem(τ) is still ≈ 0.

---

## 5. How to reproduce (full pipeline)

```bash
conda activate Mem_Gen

# ── 1. Train the two new batch sizes (N=1000, width 256, same τ grid) ─────────
cd Code/DDIM_FMM
python train_DDIM_tau_ema.py 1000 --max_tau 200000 --incremental --batch_size 512
python train_DDIM_tau_ema.py 1000 --max_tau 200000 --incremental --batch_size 1024

# ── 2. Compute FCD + f_mem for all three batch sizes (100 reuses its cache) ───
cd DDIM_Evaluation/batch_size_effect
python compute_bs_fcd_fmem.py --N 1000 --batch_sizes 100 512 1024

# ── 3. Plot the overlay ───────────────────────────────────────────────────────
python plot_bs_fcd_fmem.py --N 1000
```

Output figure: `results/figures/bs_fcd_fmem_N1000.png`.

---

## 6. Reading the figure

- **Color** = batch size (100 blue, 512 orange, 1024 red).
- **Solid** = FCD(Gen,Test) quality (left axis, ↓ better), with a 2σ band.
- **Black dotted** = the shared real–real floor.
- **Dashed** = f_mem(τ) memorization (right axis).

Compare, across batch sizes: how fast each solid curve reaches the floor (speed of
generalization in gradient steps), how low it gets, and at what τ its dashed f_mem curve
lifts off zero (onset of memorization).
