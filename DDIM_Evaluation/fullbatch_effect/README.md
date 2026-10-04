# Full-Batch Effect on Sample Quality & Memorization (batch size = dataset size)

**Question.** For a fixed model width (`n_feat = 256`), how do the sample-quality
curve **FCD(Gen, Test)** and the memorization curve **f_mem(τ)** behave when each
dataset size N is trained with a **full-batch** gradient (batch size = N)?

We sweep three matched (dataset size, batch size) pairs:

| Dataset size N | Batch size B | Regime                         |
| -------------- | ------------ | ------------------------------ |
| 200            | 200          | full-batch gradient descent    |
| 1000           | 1000         | full-batch gradient descent    |
| 4000           | 4000         | full-batch gradient descent    |

---

## 1. What "batch = N" means and why it is interesting

When the batch size equals the dataset size, **every optimizer step processes the
entire training set at once**, so:

$$\text{steps per epoch} = \lceil N / B \rceil = \lceil N / N \rceil = 1
\quad\Longrightarrow\quad \tau \;(\text{optimizer steps}) \;=\; \text{epochs}.$$

Each gradient is the **exact full-dataset gradient** — there is **no mini-batch
stochasticity**. This isolates the effect of dataset size on memorization from the
effect of gradient noise, and it is the cleanest way to ask:

> *"At the same number of data repetitions per sample, does a larger dataset delay
> memorization?"*

Because τ = epochs here, the τ axis is directly the number of times each sample has
been used in an update — the natural x-axis for the paper's memorization scaling law
$\tau_{\mathrm{mem}} \propto N$.

Everything else is held identical to the earlier experiments:

| Held fixed                         | Value                                             |
| ---------------------------------- | ------------------------------------------------- |
| Model width `n_feat`               | 256                                               |
| Train / test split                 | shared indices `logs/DDIM_tau_1000_incremental`   |
| Diffusion steps `N_T`              | 200                                               |
| Learning rate                      | 1e-4 (Adam, fixed)                                |
| EMA base decay                     | 0.9999 (warmup schedule)                          |
| Max τ                              | 200000 optimizer steps                            |
| τ grid                             | dense grid (≈6 pts/decade), same for all runs     |
| **Varied together**                | **N = B ∈ {200, 1000, 4000}**                     |

---

## 2. Files in this folder

| File                                                                       | Role                                                                                        |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [compute_fullbatch_fcd_fmem.py](compute_fullbatch_fcd_fmem.py)             | Computes FCD(Gen,Test), the per-N real–real floor FCD(Train,Test), and f_mem(τ) for each N  |
| [plot_fullbatch_fcd_fmem.py](plot_fullbatch_fcd_fmem.py)                   | Draws the paper-style overlay (FCD solid, floor dotted, f_mem dashed)                       |
| `results/fullbatch_fcd_fmem.csv`                                           | One row per (N × τ checkpoint)                                                               |
| `results/figures/fullbatch_fcd_fmem.png / .pdf`                            | Final figure                                                                                |

**No metric logic is re-implemented here.** The compute script imports the exact,
already-validated routines from the parent evaluation scripts:

- FCD + generation/caching ← `../compute_fcd_vs_tau.py`
- nearest-neighbour ratio → f_mem ← `../compute_fmem_vs_tau.py`

Both metrics use the **identical 256-D representation** (per-sample-normalized UPA-DFT
beamspace, flattened) and the **identical EMA-generated samples**, so for a given N the
FCD and f_mem curves are mutually self-consistent, and the three N values are mutually
comparable.

---

## 3. Where the trained models live

The trainer `Code/DDIM_FMM/train_DDIM_tau_ema.py` takes a `--batch_size` flag. For a
full-batch run we pass `--batch_size N`; the trainer then writes a `_bs<N>` suffix so
these runs stay in their **own** directories, separate from the batch-100 originals:

| N (=batch) | Log directory                                                   | Status              |
| ---------- | --------------------------------------------------------------- | ------------------- |
| 200        | `logs_ema/DDIM_tau_ema_200_bs200_incremental`                   | **you train this**  |
| 1000       | `logs_ema/DDIM_tau_ema_1000_bs1000_incremental`                 | **you train this**  |
| 4000       | `logs_ema/DDIM_tau_ema_4000_bs4000_incremental`                 | **you train this**  |

Generated EMA samples are cached in each run's `generated_ema/` folder on first use and
reused verbatim on any re-run.

---

## 4. Metrics (one-line definitions)

- **FCD(Gen, Test)** — Fréchet Channel Distance between generated and held-out test
  channels in 256-D beamspace: `||μ_g − μ_r||² + Tr(Σ_g + Σ_r − 2(Σ_g Σ_r)^½)`.
  Lower = better quality. Error band = 2σ over 5 disjoint test folds.
- **FCD(Train, Test)** — the same distance between the *training* set and the test set:
  the irreducible **real–real floor**. **Differs per N** (each N has a different train
  set), so each N gets its own dotted floor in its own color.
- **f_mem(τ)** — fraction of generated samples whose nearest-neighbour ratio
  `ρ = d₁/d₂ < 1/3` against the training set (memorization test), on the same samples.

The **generalization window** for each N is the τ-band where FCD(Gen,Test) has descended
onto its floor while f_mem(τ) is still ≈ 0.

---

## 5. How to reproduce (full pipeline)

```bash
conda activate Mem_Gen

# ── 1. Train each size with batch size = size (full-batch), width 256 ─────────
cd Code/DDIM_FMM
python train_DDIM_tau_ema.py 200  --max_tau 200000 --incremental --batch_size 200
python train_DDIM_tau_ema.py 1000 --max_tau 200000 --incremental --batch_size 1000
python train_DDIM_tau_ema.py 4000 --max_tau 200000 --incremental --batch_size 4000

# ── 2. Compute FCD + f_mem for all three full-batch runs ──────────────────────
cd DDIM_Evaluation/fullbatch_effect
python compute_fullbatch_fcd_fmem.py --sizes 200 1000 4000

# ── 3. Plot the overlay ───────────────────────────────────────────────────────
python plot_fullbatch_fcd_fmem.py
```

Output figure: `results/figures/fullbatch_fcd_fmem.png`.

---

## 6. Reading the figure

- **Color** = dataset size N (200 orange, 1000 green, 4000 red).
- **Solid** = FCD(Gen,Test) quality (left axis, ↓ better), with a 2σ band.
- **Dotted (same color)** = that N's real–real floor.
- **Dashed (same color)** = that N's f_mem(τ) memorization (right axis).

Because τ = epochs here, compare across N: at the same number of data repetitions,
does the larger dataset (red) keep f_mem near zero for longer (later memorization
onset), and does its quality curve settle onto a lower floor? This is the full-batch
counterpart of the paper's $\tau_{\mathrm{mem}} \propto N$ result, now free of
mini-batch gradient noise.
