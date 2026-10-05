# Reproducibility

Every stochastic step in this repository is seeded, and the seeds are fixed
constants in `memgen/config.py` rather than command-line defaults that drift.

| Stage | Seed | Constant |
| --- | --- | --- |
| master shuffle, train/test split | 0 | `SPLIT_SEED` |
| model initialisation, batch order | 0 | `DEFAULT_SEED`, `--seed` |
| denoising-loss probe noise | 12345 | `EVAL_NOISE_SEED` |
| channel generation | 0 | `GENERATION_SEED` |
| FCD test folds | 0 | `FOLD_SEED` |
| `f_mem` bootstrap | 42 | `BOOTSTRAP_SEED` |

## What is guaranteed

On one machine, with a fixed GPU model, driver and PyTorch version, two runs
that share a seed produce **bit-identical** results: the same training subset,
the same weights at every checkpoint, the same generated channels, and
therefore the same numbers in every CSV and every figure.

Seeding alone is not enough to get this. cuDNN selects convolution kernels by
autotuning, and some of those kernels accumulate in nondeterministic order.
Training the same configuration twice with the same seed, differing only in
whether deterministic kernels are enforced:

```text
default (autotuning)                 deterministic kernels
tau=   0   identical,  0.000e+00     tau=   0   identical,  0.000e+00
tau= 100   differs,    1.177e-02     tau= 100   identical,  0.000e+00
tau= 200   differs,    2.966e-02     tau= 200   identical,  0.000e+00
tau= 300   differs,    6.557e-02     tau= 300   identical,  0.000e+00
```

The divergence is not a rounding artefact: it grows with `tau`, because each
step feeds the next. Since the results of this study are statements about
*where along `tau`* the generalization window opens and closes, runs have to be
reproducible along that axis. `memgen train` therefore sets

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

by default. The cost is about 5 % in wall-clock time (41.5 s to 43.4 s on a
`W = 256`, `N = 200`, 2000-step run). Pass `--non-deterministic` to trade the
guarantee back for the speed; the choice is recorded in each run's
`config.json`.

## What is not guaranteed

- **Across machines.** A different GPU model, CUDA version or PyTorch build may
  change floating-point reduction order. Curves stay statistically identical,
  individual digits may not.
- **CPU versus GPU.** Not bit-comparable.
- **Across package versions.** Upgrading PyTorch can change kernel selection
  even in deterministic mode. `environment.yml` pins the versions used for the
  paper.

## Datasets

The ray-traced datasets come from `channel_generation.py` in
[GenAI_Channel_Modeling](https://github.com/Telefonica-Scientific-Research/GenAI_Channel_Modeling),
which seeds NumPy and TensorFlow from a `--base_seed` argument. Ray tracing is
itself deterministic given a scene, a UE position list and that seed.

The DICHASUS dataset is measured over the air. It cannot be regenerated; it is
downloaded from the campaign and used as distributed.

See [datasets.md](datasets.md) for file names and layouts.

## Checking it yourself

```bash
MEMGEN_RUN_ROOT=/tmp/a memgen train --size 200 --width 64 --max-tau 300
MEMGEN_RUN_ROOT=/tmp/b memgen train --size 200 --width 64 --max-tau 300
python - <<'PY'
import torch
from pathlib import Path
a = Path("/tmp/a/sionna_3p5ghz/sionna_3p5ghz_N200_W64_B200/checkpoints")
b = Path("/tmp/b/sionna_3p5ghz/sionna_3p5ghz_N200_W64_B200/checkpoints")
for tau in (100, 200, 300):
    sa = torch.load(a / f"checkpoint_tau_{tau}.pt", map_location="cpu")["model_state_dict"]
    sb = torch.load(b / f"checkpoint_tau_{tau}.pt", map_location="cpu")["model_state_dict"]
    print(tau, all(torch.equal(sa[k], sb[k]) for k in sa))
PY
```
