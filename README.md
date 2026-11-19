# FIDN-2: Wildfire Final Burnt-Area Prediction

This repository provides Matched DenseNet FIDN and Residual U-Net FIDN for training and evaluating models that predict the final burnt area of individual wildfires.

[![Python 3.9.13](https://img.shields.io/badge/Python-3.9.13-3776AB)](https://www.python.org/downloads/release/python-3913/)
[![TensorFlow 2.9.1](https://img.shields.io/badge/TensorFlow-2.9.1-FF6F00)](https://pypi.org/project/tensorflow/2.9.1/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

## Introduction

Estimating where a wildfire will ultimately burn from its earliest observations can support wildfire assessment and planning while the fire is still active. The task is spatial as well as quantitative. Burnt-area extent depends on interactions among terrain, fuels, and weather, including changes in vegetation, wind, and precipitation across a fire's surroundings.

Pang et al. (2025) introduced Fire-Image-DenseNet (FIDN) to predict final burnt area directly from observations of the first three days after ignition and environmental variables. One DenseNet encoder processes the fire observations, and six parallel encoders process the environmental inputs. The model fuses features from all seven branches and decodes them into a final burnt-area map.

This repository implements Matched DenseNet FIDN, a retrained baseline that retains the original DenseNet encoders and decoder blocks, and Residual U-Net FIDN, which retains the seven-branch design and feature fusion. The latter uses residual encoders and decoder blocks, with U-Net-style multiscale skip connections from the fire-observation encoder to the decoder. Both architectures use the same data partition, preprocessing, training protocol, and evaluation rules.

## Project Structure

```text
FIDN-2-public/
├── train.py                         # Train one selected architecture
├── evaluate.py                      # Evaluate a saved full-model H5
├── configs/
│   ├── train.json                   # Training settings
│   └── evaluate.json                # Evaluation settings
├── FIDN/
│   ├── dataset.py                   # Data loading, preprocessing, and augmentation
│   ├── models/                      # DenseNet and Residual U-Net model builders
│   ├── evaluation_metrics.py        # Binary spatial and boundary metrics
│   ├── raw_probability_metrics.py   # Continuous-map metrics
│   ├── visualization.py             # Per-event JPG outputs
│   ├── losses.py                    # Training loss and metrics
│   ├── protocol.py                  # Shared data and preprocessing protocol
│   └── run_utils.py                 # Configuration, run directories, and provenance
├── docs/
│   ├── protocol.md                  # Experiment protocol
│   └── source_manifest.json         # Source provenance
├── env/
│   └── requirements.txt             # Pinned Python dependencies
├── LICENSE
├── THIRD_PARTY_NOTICES.md
└── licenses/
    └── KERAS_APPLICATIONS_MIT.txt
```

## Code Metadata

| Item | Verified environment |
| --- | --- |
| Language | Python 3.9.13 |
| Model framework | TensorFlow 2.9.1 and Keras 2.9.0 |
| Array processing | NumPy 1.23.1 |
| Training platform | Linux with an NVIDIA A100-PCIE-40GB GPU |

Both model architectures are implemented in Python using TensorFlow and Keras.

## Dependencies

Use Python 3.9.13 and install the pinned packages from the project root. GPU training requires a compatible TensorFlow GPU runtime configured separately. [Dependency file](env/requirements.txt)

```bash
python -m pip install -r env/requirements.txt
```

## Access to Datasets

Pang et al. (2025) provide the processed dataset through the original FIDN project. The released NumPy array contains 304 wildfire events, each with 15 raster channels on a common 512 × 512 grid. The first 14 channels are model inputs, and the final channel is the binary final burnt-area label. The matched study uses the first 303 events and excludes the final event from 2020. [Original FIDN dataset access and download](https://github.com/DL-WG/FIDN#access-to-datasets)

| Subset | Event count | Zero-based array rows |
| --- | ---: | --- |
| Training | 243 | `0:243` |
| Validation | 30 | `243:273` |
| Test | 30 | `273:303` |

Slices are half-open. The table below numbers channels from 1 to 15, corresponding to Python indices 0 to 14. Original resolution refers to source data. Both models use the full 512 × 512 fire-observation grid and the central 128 × 128 crop of the environmental channels, without resampling.

| Channel | Description | Source | Original resolution | Source link |
| ---: | --- | --- | --- | --- |
| 1 | Burnt area – day 0 | MODIS/VIIRS-derived fire records | ≈1 km¹ | [Fire data catalogue][fire-data] |
| 2 | Burnt area – day 1 | MODIS/VIIRS-derived fire records | ≈1 km¹ | [Fire data catalogue][fire-data] |
| 3 | Burnt area – day 2 | MODIS/VIIRS-derived fire records | ≈1 km¹ | [Fire data catalogue][fire-data] |
| 4 | Above-ground biomass | ORNL DAAC | 300 m | [Biomass carbon density][biomass-data] |
| 5 | Below-ground biomass | ORNL DAAC | 300 m | [Biomass carbon density][biomass-data] |
| 6 | Slope | Conservation Science Partners (CSP) | 270 m | [CSP terrain catalogue][terrain-data] |
| 7 | Tree cover | Copernicus PROBA-V | 100 m | [Land cover][land-cover-data] |
| 8 | Grass cover | Copernicus PROBA-V | 100 m | [Land cover][land-cover-data] |
| 9 | Bare-ground cover | Copernicus PROBA-V | 100 m | [Land cover][land-cover-data] |
| 10 | Snow cover | Copernicus PROBA-V | 100 m | [Land cover][land-cover-data] |
| 11 | Water cover | Copernicus PROBA-V | 100 m | [Land cover][land-cover-data] |
| 12 | Wind u-component (monthly mean at 10 m) | ECMWF ERA5 | 27.83 km | [ERA5 monthly aggregates][era5-data] |
| 13 | Wind v-component (monthly mean at 10 m) | ECMWF ERA5 | 27.83 km | [ERA5 monthly aggregates][era5-data] |
| 14 | Precipitation (monthly total of rain and snow) | ECMWF ERA5 | 27.83 km | [ERA5 monthly aggregates][era5-data] |
| 15 | Final burnt area | MODIS/VIIRS-derived fire records | ≈1 km¹ | [Fire data catalogue][fire-data] |

¹ The original FIDN README reports approximately 1 km for the fire channels. The linked ORNL DAAC Global Fire Atlas catalogue describes its underlying 500 m MODIS product.

[fire-data]: https://data.nasa.gov/dataset/global-fire-atlas-with-characteristics-of-individual-fires-2003-2016-69d65
[biomass-data]: https://developers.google.com/earth-engine/datasets/catalog/NASA_ORNL_biomass_carbon_density_v1
[terrain-data]: https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_mTPI
[land-cover-data]: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V-C3_Global
[era5-data]: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY

## Usage

Both scripts expect the processed `float64` NumPy array with shape `(304, 512, 512, 15)`. Place it at `data/fire_data.npy` to use the default `data_path`, or point each configuration to your copy. Relative paths resolve from the project root.

### Training

Edit the training configuration before running the script. `train.py` reads it automatically. [Training configuration](configs/train.json)

| Setting | Purpose |
| --- | --- |
| `model` | Select `matched_densenet` or `residual_unet`. |
| `seed`, `epochs`, `batch_size` | Set the seed, maximum epoch count, and batch size. |
| `optimizer`, `learning_rate`, `loss` | Set the learning rate with the implemented `adam` optimizer and `binary_crossentropy` loss. |
| `data_path`, `save_path` | Set the input array and training output directory. |
| `use_early_stopping`, `early_stopping_patience` | Enable or disable early stopping and set its patience. |

```bash
python train.py
```

The manuscript used batch size **10** and seeds **42, 123, and 2024** for each architecture. The supplied configuration defaults to batch size **16**, seed **42**, and `residual_unet`. Training allocates about 28.5 GiB of host memory for the augmented array alone. [Matched experiment protocol](docs/protocol.md)

A successful run writes one full-model H5, `history.csv`, and `training_run.json` to a new directory under `save_path` (default `./models`). With early stopping enabled, it saves the checkpoint with the lowest validation BCE. With early stopping disabled, it saves the model from the last epoch. The H5 name records the selected epoch.

### Evaluation

Edit the evaluation configuration before running the script. `evaluate.py` reads it automatically. Point `data_path` to the dataset used for training, set `model_path` to the path of a **specific existing full-model `.h5` file**, choose `save_path`, and set `save_predictions` to enable or disable JPG output. [Evaluation configuration](configs/evaluate.json)

```bash
python evaluate.py
```

Keep the H5 beside its completed `training_run.json` so evaluation can verify the dataset and checkpoint hashes and reuse the training normalization. For a historical H5 without that record, the optional `legacy` object must supply verified `model`, `seed`, `dataset_sha256`, `h5_sha256`, and `preprocessing` set to `fidn_fixed_split_train_minmax_v1`. Use this option for checkpoints with verified fixed-split and training-only normalization provenance.

Each evaluation scores one model on the fixed 30 test events. MSE, PSNR, and SSIM use raw sigmoid probabilities. CSI, POD, FAR, signed area bias, absolute area bias, and mean boundary distance use the burnt-area mask at `p >= 0.5`. [Metric definitions](docs/protocol.md#evaluation)

A successful run writes the following files under `save_path` (default `./results`).

- `metrics_per_event.csv` contains metrics and dataset indices for all 30 test events.
- `metrics_summary.csv` contains macro-averages and counts of defined and undefined metric values.
- `evaluation_run.json` records configuration, preprocessing, hashes, and run status.
- With `save_predictions=true`, `predictions/0.jpg` through `predictions/29.jpg` show the day-2 observation, final observation, and predicted probability.
- With `save_predictions=true`, `features/0_info.jpg` through `features/29_info.jpg` show all 14 input channels and the final burnt-area label.

Both scripts create independent run directories named `{model}_seed{seed}_{YYYYMMDD}_{HHMMSS}` using the Asia/Shanghai start time. A numeric suffix handles same-second collisions.

<a name="associated-research"></a>

## Associated Research & Funding

*A matched comparison of DenseNet and Residual U-Net architectures for direct wildfire final burnt-area prediction*

| Author | Affiliation |
| --- | --- |
| Xin Song (宋新) | School of Public Administration, Southwest Jiaotong University, Chengdu, China |
| Fei Long (龙飞) | Department of Fire Protection Engineering, Southwest Jiaotong University, Chengdu, China |
| Tao Li (李涛), corresponding author | Department of Fire Protection Engineering, Southwest Jiaotong University, Chengdu, China |

Correspondence to Tao Li at <taoli@swjtu.edu.cn>.

The research associated with this repository was supported by the National Social Science Fund of China (Grant No. 24BZZ031).

## Citation & Attribution

**Associated manuscript.** Xin Song, Fei Long, and Tao Li. *A matched comparison of DenseNet and Residual U-Net architectures for direct wildfire final burnt-area prediction*. Manuscript in preparation for submission.

**Original FIDN.** Pang et al. (2025). *Fire-Image-DenseNet (FIDN) for predicting wildfire burnt area using remote sensing data*. Computers & Geosciences, 195, 105783. [DOI 10.1016/j.cageo.2024.105783](https://doi.org/10.1016/j.cageo.2024.105783)

The original FIDN code is attributed to Bo Pang, Copyright 2022 IRP Project. This repository also retains attribution to the respective Keras Applications contributors for the adapted DenseNet building blocks. [Original FIDN source](https://github.com/DL-WG/FIDN) · [Third-party notices](THIRD_PARTY_NOTICES.md)

## License & Contact

This repository is distributed under [Apache-2.0](LICENSE). The adapted Keras Applications DenseNet components carry MIT terms. [Third-party notices](THIRD_PARTY_NOTICES.md) · [Keras Applications MIT license](licenses/KERAS_APPLICATIONS_MIT.txt)

**Tao Li**, corresponding author  
Email <taoli@swjtu.edu.cn>
