# 28 GHz LoS beam-alignment tau-window pipeline

This directory implements `../28GHz_LoS/Beam_Alignment_Window_Generation.md` without changing the legacy beam-alignment or DDIM files.

## Environment and working directory

```bash
conda activate Mem_Gen
cd Downstream_Tasks/Beam_alignment
```

## Optional preflight

```bash
python tau_window_28GHz_LoS/compute_window_metrics.py --dry_run
python tau_window_28GHz_LoS/prepare_synthetic.py --dry_run
python tau_window_28GHz_LoS/run_experiment.py --dry_run
```

## Full workflow

```bash
python tau_window_28GHz_LoS/compute_window_metrics.py --device cuda
python tau_window_28GHz_LoS/prepare_synthetic.py
python tau_window_28GHz_LoS/run_experiment.py --device cuda --epochs 1000
python tau_window_28GHz_LoS/plot_results.py
```

Use `--device auto` instead of `--device cuda` if the command must fall back to CPU.

The training runner is resumable. Repeating the identical command validates and skips completed configurations and restarts incomplete configurations. Use `--force` only to replace new isolated artifacts after an intentional configuration change or integrity failure.

## Add only N=500 after N=100 and N=1000 are complete

These commands append the `N=500` experiment while preserving all completed
`N=100` and `N=1000` rows, checkpoints, logs, summaries, and figures:

```bash
python tau_window_28GHz_LoS/compute_window_metrics.py --sizes 500 --device cuda
python tau_window_28GHz_LoS/prepare_synthetic.py --sizes 500
python tau_window_28GHz_LoS/run_experiment.py --sizes 500 --device cuda --epochs 1000 --dry_run
python tau_window_28GHz_LoS/run_experiment.py --sizes 500 --device cuda --epochs 1000
python tau_window_28GHz_LoS/plot_results.py --sizes 500
```

For `N=500`, every augmented training set contains exactly 500 real channels
and 4,500 synthetic channels.

## Outputs

All generated data, metrics, model checkpoints, logs, and figures are written below:

```text
../28GHz_LoS/tau_window_kappa1_4_fmem20/
```

The four final PNG/PDF plots appear in its `figures/` directory. The run-level table is `metrics/run_results.csv`, and the best-checkpoint classifications are saved in `metrics/optimum_summary.csv`.
