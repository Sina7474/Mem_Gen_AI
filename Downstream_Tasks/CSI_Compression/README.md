# CSI-compression downstream task

This directory evaluates DDIM-generated channels as augmentation data for CRNet
CSI compression.

The paper-facing experiment is the shared-reference design under
`shared_reference/`: the DDIM and CRNet use the same `N` reference channels,
and each augmented CRNet receives `5000-N` synthetic channels. Eight DDIM
checkpoints are evaluated for each reference-set size, with four independent
CRNet training runs per configuration.

From this directory, run:

```bash
python shared_reference/run_shared_reference.py
python shared_reference/plot_shared_reference_combined.py
```

The scripts expect the 3.5 GHz datasets and DDIM runs documented in
[`../../docs/DATA_AND_CHECKPOINTS.md`](../../docs/DATA_AND_CHECKPOINTS.md).
Generated results and figures are ignored by Git.

