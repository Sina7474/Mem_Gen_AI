# Data and checkpoints

No dataset, checkpoint, generated channel, result table, or figure is included
in this repository.

## Expected dataset layout

Place external inputs under the repository-local `dataset/` directory:

```text
dataset/
├── Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz
├── Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz
├── Final_Single_Scene_Channel_Sionna_V1_28GHz_LoS_UPA.npz
├── 28GHz_Channel_UE_positions_full_LoS_iso_115_8759_51.546036612508296_-0.17853666925844522.npz
└── Measured_Dataset/
    └── dichasus_0c5x_downlink_1272MHz.npz
```

The 3.5 GHz and 28 GHz ray-tracing files are read by the DDIM, evaluation, and
downstream scripts. The measured file is the DICHASUS 0c5x downlink dataset at
1.272 GHz. Users are responsible for obtaining these datasets under their
respective access and redistribution terms.

## Expected checkpoint layout

Training scripts create their run directories below `Code/DDIM_FMM/`, including:

```text
Code/DDIM_FMM/
├── logs/
├── logs_ema/
└── logs_ema_28GHz_LoS/
```

Evaluation and downstream scripts reuse those directories. Typical checkpoints
follow the pattern:

```text
<run>/checkpoints/checkpoint_tau_<TAU>.pth
```

Generated samples are cached below each run, normally under `generated_ema/`.
All these locations are ignored by Git.

## Result locations

The scripts create local `results/`, `figures/`, `metrics/`, or experiment-
specific output directories. These outputs are intentionally excluded from the
repository. They can be regenerated after the external datasets and checkpoints
have been placed in the expected layout.

