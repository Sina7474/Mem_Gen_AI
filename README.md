<h1 align="center">Mem_Gen_AI</h1>

<p align="center">
  <b>Learn Before Memorizing: Generalization Windows in Diffusion-Based Wireless Channel Synthesis</b>
</p>

<p align="center">
  <a href="#installation"><img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg"></a>
  <a href="https://pytorch.org"><img alt="pytorch" src="https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

---

A diffusion model trained on site-specific MIMO channels first learns the
channel distribution and only later starts reproducing its individual training
channels. Between the two there is a **generalization window**: a band of
training times in which the generated channels are statistically faithful and
still private. This repository contains the code that locates that window,
measures how it scales with the dataset size and the model capacity, and checks
what it is worth for two downstream receivers.

Two numbers describe every checkpoint:

| Metric | Meaning | Behaviour |
| --- | --- | --- |
| `FCD` | Frechet Channel Distance between generated and held-out real channels | falls, then saturates on the real-real floor |
| `f_mem` | fraction of generated channels that collapse onto a single training channel (`d1/d2 < kappa`) | stays near zero, then rises |

The window opens at `tau_gen`, where `FCD` reaches its floor, and closes at
`tau_mem`, where `f_mem` lifts off.

## Highlights

- **One trainer, one evaluator.** The dataset-size sweep, the capacity sweep,
  the batch-size and full-batch controls and the two replication datasets are
  arguments, not separate scripts.
- **Self-consistent metrics.** `FCD` and `f_mem` are computed from the same
  generated channels in the same beamspace feature space.
- **Downstream read-out.** CSI compression and beam alignment consume the
  generated channels through a leakage-safe split, so their gain curves can be
  overlaid directly on the window.
- **Code only.** No datasets, checkpoints, generated channels or figures are
  committed; everything is regenerated from the commands below.

## Repository layout

```text
memgen/
├── config.py          paths, hyper-parameters, dataset registry, run naming
├── beamspace.py       array DFT codebooks, normalisation, feature map
├── datasets.py        loading, master shuffle, nested subsets, held-out split
├── model.py           unconditional U-Net denoiser and the DDIM wrapper
├── train.py           tau-indexed trainer with weight EMA and checkpoint grid
├── sampling.py        checkpoint discovery and cached generation
├── metrics.py         FCD, f_mem, bootstrap CIs, generalization window
├── evaluate.py        one sweep covering every ablation in the paper
├── phase.py           logistic fit of the phase boundary N_c(W)
├── figures.py         paper figures, redrawn from the CSV tables
├── cli.py             the `memgen` command-line interface
└── downstream/
    ├── crnet.py            CRNet autoencoder for CSI compression
    ├── csi_compression.py  reference / augmented / full-real comparison
    ├── beamforming.py      learned probing beams and beam synthesis
    └── beam_alignment.py   learned probing-beam experiment
```

## Installation

```bash
git clone https://github.com/Sina7474/Mem_Gen_AI.git
cd Mem_Gen_AI
conda env create -f environment.yml
conda activate memgen
```

Or, into an existing environment:

```bash
pip install -e .
```

Datasets, checkpoints and results live outside the repository. The defaults are
`./data`, `./runs` and `./results`; override them with `MEMGEN_DATA_ROOT`,
`MEMGEN_RUN_ROOT` and `MEMGEN_RESULT_ROOT`. See
[docs/datasets.md](docs/datasets.md) for the expected files.

## Quick start

```bash
# 1. Train a generator (checkpoints land on a logarithmic grid of tau)
memgen train --dataset sionna_3p5ghz --size 1000

# 2. Measure fidelity and memorization at every checkpoint
memgen evaluate dataset-size --sizes 200 500 1000 2000 4000

# 3. Draw the two panels of the main result
memgen figure fidelity results/tables/sionna_3p5ghz_dataset-size.csv
memgen figure collapse  results/tables/sionna_3p5ghz_dataset-size.csv
```

`memgen --help` lists every subcommand; `memgen <command> --help` documents its
arguments.

## Commands

| Command | Purpose |
| --- | --- |
| `memgen train` | train one DDIM; `--size`, `--width`, `--batch-size`, `--full-batch` select the ablation arm |
| `memgen evaluate` | `FCD` and `f_mem` versus `tau` for the `dataset-size`, `model-size`, `batch-size` or `full-batch` sweep |
| `memgen phase` | fit the generalization-memorization boundary `N_c(W)` |
| `memgen figure` | redraw `loss`, `fidelity`, `collapse`, `phase` or `downstream` figures |
| `memgen csi` | CSI-compression downstream experiment |
| `memgen beam` | beam-alignment downstream experiment |

## Reproducing the paper

[docs/reproducing.md](docs/reproducing.md) maps each figure of the manuscript to
the exact commands that produce it. The short version:

```bash
# Dataset-size scaling and the tau/N collapse
for n in 100 200 500 1000 2000 4000; do memgen train --size $n; done
memgen evaluate dataset-size
memgen figure fidelity results/tables/sionna_3p5ghz_dataset-size.csv
memgen figure collapse  results/tables/sionna_3p5ghz_dataset-size.csv

# Model capacity and the phase diagram
for w in 64 128 256; do for n in 200 1000; do memgen train --size $n --width $w; done; done
memgen evaluate model-size
memgen phase results/tables/sionna_3p5ghz_model-size.csv --eps 0.10
memgen figure phase results/tables/phase_boundary.csv

# Downstream utility
memgen csi  --sizes 200 500 1000
memgen beam --dataset sionna_28ghz --sizes 100 500
```

## Datasets

| Key | Description | Channel shape |
| --- | --- | --- |
| `sionna_3p5ghz` | Sionna RT ray tracing, 3.5 GHz, LoS and NLoS | 4 x 32 |
| `sionna_28ghz` | Sionna RT ray tracing, 28 GHz, LoS only | 4 x 32 |
| `dichasus_1p272ghz` | DICHASUS indoor measurements, 1.272 GHz | 4 x 8 |

All three share one master shuffle per dataset, so training subsets are nested
across `N` and every run is evaluated against the same held-out channels.

## Citation

```bibtex
@article{beyraghi2026memgen,
  title   = {Learn Before Memorizing: Generalization Windows in
             Diffusion-Based Wireless Channel Synthesis},
  author  = {Beyraghi, Sina and Sadeghian, Masoud and Lozano, Angel and
             Almasan, Paul and Geraci, Giovanni},
  year    = {2026}
}
```

