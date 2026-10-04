# Datasets

No channel data is committed to this repository. Place the files below under
`MEMGEN_DATA_ROOT` (default `./data`) and every command in the README will find
them.

```text
data/
├── Final_Single_Scene_Channel_Sionna_V1_3_5GHz_LoS.npz
├── Final_Single_Scene_Channel_Sionna_V1_3_5GHz_NLoS.npz
├── 28GHz_Channel_UE_positions_full_LoS_iso_115_8759_51.546036612508296_-0.17853666925844522.npz
└── Measured_Dataset/
    └── dichasus_0c5x_downlink_1272MHz.npz
```

## Sionna RT scenes (`sionna_3p5ghz`, `sionna_28ghz`)

Ray-traced channels for a single urban scene, generated with
[Sionna RT](https://nvlabs.github.io/sionna/). Each file holds one array,

```python
np.load(path)["combined_array"]      # (users, Nr, 1, Nt, 1, subcarriers, 4)
```

whose last axis stores the complex channel coefficient followed by the UE
position `(x, y, z)`. The loader takes subcarrier 128 at 3.5 GHz and the single
stored subcarrier at 28 GHz, giving `4 x 32` MIMO channels between a `2 x 2` UE
array and an `8 x 4` base-station array.

The 3.5 GHz scene is split into equally sized LoS and NLoS halves, so a
training set of size `N` always contains `N/2` channels of each kind. The
28 GHz scene is LoS only.

## DICHASUS measurements (`dichasus_1p272ghz`)

Indoor measurements from the
[DICHASUS](https://dichasus.inue.uni-stuttgart.de/) campaign at 1.272 GHz,
converted to

```python
np.load(path)["h_downlink"]    # (users, 2, 1, 32) real/imaginary
np.load(path)["ue_locations"]  # (users, 3)
```

The 32 base-station ports are re-indexed onto the physical `4 x 8` antenna grid
before the spatial DFT (see `DICHASUS_ANTENNA_MAP` in `memgen/datasets.py`).
The UE has a single antenna, so both array transforms reduce to uniform linear
arrays and the model input is `2 x 4 x 8`.

## Splits

For each dataset one permutation is drawn with `SPLIT_SEED = 0` and cached in
`runs/<dataset>/shared_indices.npz`. It is reused by every `N` and every width,
which has two consequences:

- training sets are **nested**: the channels used at `N = 200` are a subset of
  those used at `N = 1000`;
- the **held-out split** is the final 10 % of the shuffle and is disjoint from
  every training set, so it is a valid reference for the Frechet Channel
  Distance and a leakage-free test set for the downstream tasks.

Delete the cached file only if you intend to invalidate every existing run.

## Artefacts written at run time

```text
runs/<dataset>/<dataset>_N<N>_W<W>_B<B>/
├── checkpoints/checkpoint_tau_<tau>.pt   model + EMA weights + optimiser state
├── samples/ema_tau<tau>_seed0.npz        cached generated channels
├── train.npy, test.npy                   antenna-domain splits
├── train_coords.npy, test_coords.npy     UE positions
├── loss_curve.csv                        train/test denoising loss versus tau
└── config.json                           full run configuration
```

Generation is the only expensive part of evaluation, so samples are cached on
first use and reused by every metric and by both downstream tasks.
