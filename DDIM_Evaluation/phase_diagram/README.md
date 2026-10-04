# Generalization–Memorization Phase Diagram

Reconstruction, for our synthetic wireless channels, of the phase diagram in
**Figure 3 (right panel)** of Bonnaire et al., *Why Diffusion Models Don't
Memorize: The Role of Implicit Dynamical Regularization in Training*
(arXiv:2505.17638v2).

The scripts below implement the construction used for the journal analysis.

---

## The plane

| Axis | Symbol | Meaning | Our values |
| ---- | ------ | ------- | ---------- |
| x | **p** | trainable U-Net parameters (set by width W) | 0.96 M (W=64), 3.81 M (W=128), 15.23 M (W=256) |
| y | **N** | training-set size | 50, 200, 500, 1000, 2000, 4000 |

Every (p, N) cell is one trained DDIM — **18 models**. Below the boundary the
model memorizes at the chosen stopping time; above it, it generalizes.

---

## The construction

### Step 1 — one stopping time per **model size**

Memorization grows with training time τ, so "does this model memorize?" only
becomes meaningful once we fix **when** to look. The paper looks at
$\tau_{\text{gen}}(W)$: the time at which sample quality first becomes good.
Crucially it is a function of the **width only** — the paper's quality curves
collapse under $\tau \to W\tau$, giving $\tau_{\text{gen}}(W) = C/W$.

We estimate it from **our own** FCD curves: the first checkpoint whose
`FCD(Gen,Test)` falls within `--tol` of the real–real floor `FCD(Train,Test)`,
read at the largest N (where the quality curve is least contaminated by
memorization).

### Step 1b — **verify** the collapse, never assume it

The report warns explicitly that $W\tau_{\text{gen}} \approx$ const must be
re-checked on new data. **For our channels it holds:**

| W | p | τ_gen measured | W·τ_gen | τ_gen = C/W |
| --: | --: | --: | --: | --: |
| 64 | 956,930 | 70,000 | 4.48 × 10⁶ | 62,500 |
| 128 | 3,814,402 | 30,000 | 3.84 × 10⁶ | 31,250 |
| 256 | 15,230,978 | 15,000 | 3.84 × 10⁶ | 15,625 |

Spread of `W·τ_gen` is only **1.17×** across a 16× range of model size. The
fitted constant is `C = 4.04 × 10⁶`, used rounded as **C = 4 × 10⁶** (the paper
likewise quotes a round `3 × 10⁶` for CelebA). Figure `taugen_collapse.png`
panel (b) shows the three FCD curves falling on top of each other and reaching
their minimum exactly at `Wτ ≈ 4 × 10⁶`.

### Step 2 — read `f_mem` at `c·τ_gen(W)` for every N

`c·τ_gen` almost never lands on a saved checkpoint (equally true in the original
paper, whose code does not state how this was handled), so `f_mem` is linearly
interpolated **in log τ**.

### Step 3 — the boundary is the smallest safe N

$$N_c(p) = \min\{\,N : f_{\text{mem}}(W, N, c\,\tau_{\text{gen}}(W)) \le \epsilon\,\}$$

The crossing is refined by interpolating **log N against log f_mem**, which is
what turns a handful of discrete measured points into the smooth curve the paper
draws.

---

## Choices we had to make (and why)

**On the multipliers.** The paper's `3τ_gen` / `8τ_gen` are *presentation
choices, not physics* — just "somewhat longer" and "much longer" training. Even
their base `τ_gen = 3×10⁶/W` is a rounded fit whose values never land on their
logarithmic checkpoint grid, so they too must have snapped or interpolated. We
use **c ∈ {1, 2, 3}**, the largest set for which all three widths give a boundary
inside our tested N range. At `c = 4` the W=256 boundary already exceeds N=4000
and is honestly reported as `above_range` rather than extrapolated.

**On ε.** The paper uses the strict `f_mem = 0`. On our grid that criterion
degenerates: at `c·τ_gen` the only exactly-zero cells sit at the largest N, so a
strict boundary cannot be located for most widths. We therefore use **ε = 1 %**
and the script always prints the strict `ε = 0` result as a sensitivity check.

**Censoring.** `f_mem` increases monotonically with τ, so when `c·τ_gen` exceeds
a run's τ_max the last measured value is a **lower bound**. If it already exceeds
ε the cell is provably memorizing and the boundary stays valid; otherwise the
cell is flagged `censored` and is never used to *create* a boundary.

---

## Result

At `τ_gen` the boundary is **nearly flat** (≈ 1300–1700 channels, essentially
independent of model size) — the same qualitative statement the paper makes for
its own widths. Train longer and the boundary **rises steeply with model size**:
at `3τ_gen` a 15 M-parameter model needs ≈ 3800 channels to stay safe, versus
≈ 1800 for a 1 M-parameter model.

| c | W=64 (0.96 M) | W=128 (3.81 M) | W=256 (15.23 M) |
| --: | --: | --: | --: |
| 1 | 1,288 | 1,712 | 1,626 |
| 2 | 1,538 | 2,464 | 2,756 |
| 3 | 1,805 | 2,538 | 3,802 |

---

## Running it

Requires the locally generated file
`../model_size_effect/results/wsize_fcd_fmem.csv` covering all 18 configs (see
[the model-size instructions](../model_size_effect/README.md)).

```bash
conda activate Mem_Gen
cd DDIM_Evaluation/phase_diagram

# defaults: tol=0.5, eps=0.01, c = 1 2 3
python paper_phase_diagram.py

# more/other boundary curves (out-of-range ones are reported, not drawn)
python paper_phase_diagram.py --multipliers 1 2 3 4

# looser safety criterion → four complete curves
python paper_phase_diagram.py --eps 0.02 --multipliers 1 2 3 4

# pin the collapse constant instead of fitting it
python paper_phase_diagram.py --const 4e6

# stricter/looser quality criterion for τ_gen
python paper_phase_diagram.py --tol 0.3
```

### Outputs

| File | Contents |
| ---- | -------- |
| `figures/phase_diagram_paper.png/.pdf` | the phase diagram |
| `figures/taugen_collapse.png/.pdf` | evidence that `W·τ_gen ≈ const` |
| `results/paper_taugen.csv` | τ_gen per width + the collapse test |
| `results/paper_boundary.csv` | every boundary point with its `status` |
| `results/paper_fmem_at_tau.csv` | the `f_mem(W, N, c·τ_gen)` values behind it |

`status` is one of `interp` (crossing bracketed and interpolated), `grid`
(crossing lands on a tested N), `censored`, `above_range`, `below_range`. Only
`interp` and `grid` points are plotted.

---

## Superseded scripts

`build_phase_diagram_table.py`, `plot_phase_diagram.py` and
`plot_phase_diagram_paper.py` are an **earlier attempt** that estimated
`τ_gen` per *(W, N)* cell rather than per width. Because τ_gen grows with N in
our data, those boundaries came out inverted and needed a fixed-absolute-τ
workaround. They are kept for reference only — **use `paper_phase_diagram.py`.**
