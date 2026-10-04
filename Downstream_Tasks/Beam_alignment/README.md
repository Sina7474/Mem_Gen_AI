# Beam-alignment downstream task

This directory evaluates generated channels as augmentation data for the
DL-GF beam-alignment model used in the journal appendix.

The paper-facing 28 GHz generalization-window workflow is under
`tau_window_28GHz_LoS/`:

```bash
python tau_window_28GHz_LoS/run_experiment.py
python tau_window_28GHz_LoS/compute_window_metrics.py
python tau_window_28GHz_LoS/plot_results.py
```

The scripts expect the 28 GHz dataset and DDIM runs documented in
[`../../docs/DATA_AND_CHECKPOINTS.md`](../../docs/DATA_AND_CHECKPOINTS.md).
Generated samples, checkpoints, metrics, logs, and figures are ignored by Git.

Parts of the implementation adapt the GPL-3.0-licensed DL-GF project. See the
repository-level `THIRD_PARTY_NOTICES.md` and `LICENSES/` directory.

