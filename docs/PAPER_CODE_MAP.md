# Paper-to-code map

This map identifies the source entry points for the result families used by the
conference paper and the extended journal version. Numerical outputs and figures
are not stored in the repository.

| Result family | Coverage | Training/source | Evaluation | Plotting |
|---|---|---|---|---|
| Training/test denoising loss versus update index | Journal | `Code/DDIM_FMM/train_DDIM_tau_ema.py` | `DDIM_Evaluation/compute_loss_errorbars.py` | `DDIM_Evaluation/plot_tau_train_test_loss.py` |
| FCD and memorization versus dataset size | Both | `Code/DDIM_FMM/train_DDIM_tau_ema.py` | `DDIM_Evaluation/dataset_size_effect/compute_dsize_fcd_fmem.py` | `DDIM_Evaluation/dataset_size_effect/plot_dsize_fcd_fmem.py` |
| Normalized memorization collapse versus `tau/N` | Both | Same as above | Same as above | `DDIM_Evaluation/dataset_size_effect/plot_dsize_fmem_collapse.py` |
| Nearest-neighbor-threshold robustness | Journal; summarized in conference text | Same as above | `DDIM_Evaluation/dataset_size_effect/compute_dsize_fmem_multi_k.py` | `DDIM_Evaluation/dataset_size_effect/plot_dsize_fmem_multi_k.py` |
| Model-capacity dependence | Both | `Code/DDIM_FMM/train_DDIM_tau_ema.py` | `DDIM_Evaluation/model_size_effect/compute_wsize_fcd_fmem.py` | `DDIM_Evaluation/model_size_effect/plot_wsize_all_NW.py`, `plot_wsize_fmem_norm_with_inset.py` |
| Batch-size control | Journal | `Code/DDIM_FMM/train_DDIM_tau_ema.py` | `DDIM_Evaluation/batch_size_effect/compute_bs_fcd_fmem.py` | `DDIM_Evaluation/batch_size_effect/plot_bs_fcd_fmem.py` |
| Full-batch control | Journal | `Code/DDIM_FMM/train_DDIM_tau_ema.py` | `DDIM_Evaluation/fullbatch_effect/compute_fullbatch_fcd_fmem.py` | `DDIM_Evaluation/fullbatch_effect/plot_fullbatch_fcd_fmem.py`, `plot_fullbatch_fmem_collapse.py` |
| Generalization–memorization phase diagram | Journal | Model- and dataset-size runs above | `DDIM_Evaluation/phase_diagram/phase_boundary_logistic.py` | `DDIM_Evaluation/phase_diagram/replot_without_3gen.py` |
| Measured-channel replication | Journal appendix | `Code/DDIM_FMM/train_DDIM_tau_ema_measured.py` | `DDIM_Evaluation/measured_dataset_effect/compute_measured_fcd_fmem.py` | `DDIM_Evaluation/measured_dataset_effect/plot_measured_fcd_fmem.py`, `plot_measured_fmem_collapse.py` |
| 28 GHz replication | Journal appendix | `Code/DDIM_FMM/train_DDIM_tau_ema_28GHz_LoS.py` | `DDIM_Evaluation/dataset_size_effect_28GHz_LoS/compute_dsize_fmem_28GHz_LoS.py` | `DDIM_Evaluation/dataset_size_effect_28GHz_LoS/plot_dsize_fcd_fmem_28GHz_only.py`, `plot_dsize_fmem_collapse_28GHz_only.py` |
| CSI-compression augmentation | Both | `Downstream_Tasks/CSI_Compression/shared_reference/run_shared_reference.py` | Same entry point | `Downstream_Tasks/CSI_Compression/shared_reference/plot_shared_reference_combined.py` |
| Beam-alignment augmentation | Journal appendix | `Downstream_Tasks/Beam_alignment/tau_window_28GHz_LoS/run_experiment.py` | `compute_window_metrics.py` | `plot_results.py` |

The scripts expect external data/checkpoints and write ignored local results.
See [Data and checkpoints](DATA_AND_CHECKPOINTS.md).

