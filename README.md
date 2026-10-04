# Mem_Gen_AI

Official code release for the conference paper **“Is Your Wireless Generative
Model Learning or Memorizing?”** and its extended journal version **“Learn
Before Memorizing: Generalization Windows in Diffusion-Based Wireless Channel
Synthesis.”**

The repository studies when an unconditional DDIM learns the distribution of
site-specific MIMO channels and when it begins reproducing individual training
samples. It contains the common training pipeline, fidelity and memorization
metrics, capacity/data ablations, and the CSI-compression and beam-alignment
downstream evaluations used across the two manuscripts.

## Release scope

This is intentionally a **code-and-documentation-only** repository. It does not
contain datasets, model checkpoints, generated channels, logs, numerical
results, or paper figures. Every experiment writes those artifacts to paths
ignored by Git. See [Data and checkpoints](docs/DATA_AND_CHECKPOINTS.md) for the
expected local layout.

## Repository layout

```text
Code/DDIM_FMM/                         DDIM training, inference, and sampling
DDIM_Evaluation/                       Fidelity/memorization metrics and plots
  dataset_size_effect/                 Dataset-size scaling
  model_size_effect/                   Model-capacity scaling
  batch_size_effect/                   Batch-size control
  fullbatch_effect/                    Full-batch control
  measured_dataset_effect/             Measured-channel replication
  dataset_size_effect_28GHz_LoS/       28 GHz replication
  phase_diagram/                       Journal phase-diagram analysis
Downstream_Tasks/CSI_Compression/      CRNet downstream evaluation
Downstream_Tasks/Beam_alignment/       DL-GF downstream evaluation
docs/                                  Inputs, provenance, and paper/code map
scripts/                               Release validation and smoke tests
```

The original directory structure is retained so that imports and experiment
commands remain close to the research code used for the papers.

## Environment

The experiments were developed with Python 3.10 and PyTorch using CUDA 12.1.
Create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate mem-gen-ai
```

Run the code from the repository root unless a command explicitly changes into
an experiment directory.

## Quick validation

The smoke test uses synthetic arrays only and requires neither data nor model
weights:

```bash
python scripts/smoke_test.py
python scripts/verify_release.py
```

## Main workflows

### 1. Train the DDIM

Place the external channel files under `dataset/` as described in
[Data and checkpoints](docs/DATA_AND_CHECKPOINTS.md), then run, for example:

```bash
cd Code/DDIM_FMM
python train_DDIM_tau_ema.py 200 --max_tau 200000 --incremental --batch_size 200
```

The training scripts support the dataset-size, model-width, full-batch,
measured-channel, and 28 GHz configurations used in the journal study.

### 2. Compute fidelity and memorization

The primary 3.5 GHz dataset-size evaluation is:

```bash
cd DDIM_Evaluation/dataset_size_effect
python compute_dsize_fcd_fmem.py --sizes 100 200 500 1000 2000 4000
python plot_dsize_fcd_fmem.py
python plot_dsize_fmem_collapse.py
```

Related experiment directories expose analogous `compute_*.py` and `plot_*.py`
entry points for model capacity, batch size, full-batch training, measured data,
28 GHz data, and the phase diagram.

### 3. CSI compression

```bash
cd Downstream_Tasks/CSI_Compression
python shared_reference/run_shared_reference.py
python shared_reference/plot_shared_reference_combined.py
```

The same reference channels are used to train the DDIM and CRNet in the shared-
reference experiment. Each augmented CRNet uses a total of 5000 channels.

### 4. Beam alignment

```bash
cd Downstream_Tasks/Beam_alignment
python tau_window_28GHz_LoS/run_experiment.py
python tau_window_28GHz_LoS/plot_results.py
```

See [Paper-to-code map](docs/PAPER_CODE_MAP.md) for the scripts associated with
the conference and journal result families.

## Reproducibility notes

- The code expects datasets and checkpoints to be supplied locally; neither is
  downloaded automatically.
- Paths are resolved relative to the repository layout.
- Generated artifacts are excluded by `.gitignore`.
- `SOURCE_MANIFEST.tsv` records the origin and checksum of every copied research
  source file.
- The public-release changes are limited to path portability, documentation,
  licensing, and validation support. Scientific definitions and experiment
  parameters were not intentionally changed.

## Authors

Sina Beyraghi, Masoud Sadeghian, Firdous Bin Ismail, Angel Lozano, Paul Almasan,
and Giovanni Geraci.

## License and third-party code

This repository is distributed under the GNU General Public License v3.0
because the journal beam-alignment pipeline adapts GPL-3.0-licensed DL-GF code.
The DDIM release and CRNet-derived portions retain their upstream MIT notices.
See [Third-party notices](THIRD_PARTY_NOTICES.md) and `LICENSES/`.

