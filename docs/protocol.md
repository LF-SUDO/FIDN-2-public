# Matched experiment protocol

This protocol accompanies the manuscript *A matched comparison of DenseNet and Residual U-Net architectures for direct wildfire final burnt-area prediction* by Xin Song, Fei Long and Tao Li. [Associated research](../README.md#associated-research)

## Sampling and preprocessing

The source array has 304 × 512 × 512 × 15 float64 elements. The final event is excluded to retain the study's 2012–2019 cohort. Index ranges are zero-based and half-open.

| Role | Source rows | Events | Samples used |
| --- | ---: | ---: | ---: |
| Training | `0:243` | 243 | 972 |
| Validation | `243:273` | 30 | 30 |
| Test | `273:303` | 30 | 30 |

Channels 0–2 are the daily fire observations, channels 3–13 are the environmental predictors, and channel 14 is the binary final burnt-area target. Both models accept 14 channels in channels-last layout and output a 512 × 512 × 1 sigmoid probability map. The fire branch uses the full 512 × 512 grid. The six environmental branch groups use the center 128 × 128 crop without resampling, with 1, 1, 1, 5, 2 and 1 channels for above-ground biomass, below-ground biomass, slope, land cover, wind and precipitation, respectively.

Fit separate min/max pairs on raw training events only. Use those values for all three splits. Near-constant channels bypass scaling, and values beyond the training range are retained. Keep the fire/target channels unscaled. Training blocks are stacked in the order 0°, 90°, 180°, 270°. All channels and labels rotate together using the historical pixel-grid rotation. The wind components retain their numerical values under that rotation, with no component exchange or sign changes.

## Model architectures and training

| Model | Encoder and decoder design | Expected parameters |
| --- | --- | ---: |
| `matched_densenet` | Seven DenseNet encoders, followed by feature fusion and six upsampling stages with dense convolutional blocks | 16,876,041 |
| `residual_unet` | Seven residual encoders, followed by feature fusion and six residual upsampling stages. Five skip connections bring fire-branch features into the decoder. | 33,593,497 |

The manuscript experiments trained each architecture from scratch with seeds 42, 123 and 2024. They used BCE, Adam with an initial learning rate of 0.001, batch size 10, up to 100 epochs, and array shuffling each epoch. Early stopping monitored validation BCE with patience 10 and restored the best weights. The test evaluations used a full-model H5 saved at minimum validation BCE. For the three seeds in order, the recorded historical best and stop epochs are 28/38, 21/31 and 30/40 for DenseNet, and 17/27, 25/35 and 21/31 for Residual U-Net.

The current training JSON accepts a nonnegative integer seed and positive integers for batch size and maximum epochs. Its defaults are seed 42, batch size 16, up to 100 epochs, and early stopping enabled with patience 10. With early stopping disabled, the script saves the last epoch after all configured epochs finish. Python and TensorFlow receive the configured seed before network construction, and NumPy receives that seed modulo `2**32`. [Current training configuration](../configs/train.json)

## Evaluation

Single-model evaluation uses the 30 events at raw rows 273–302. It verifies the dataset and H5 against the completed training record and reuses the recorded training extrema. An explicitly configured historical H5 without a manifest uses training-only extrema under the declared preprocessing protocol. The script verifies that the default Keras prediction adapter uses inference mode and checks the weights, including BatchNorm state, before and after prediction. Inference uses automatic device placement. SSIM and PSNR run on CPU.

| Metric | Rule |
| --- | --- |
| MSE | Direct float32 mean squared error on the unscaled sigmoid probability map |
| SSIM and PSNR | TensorFlow metrics on the unscaled probabilities with `max_val=1`. SSIM uses TensorFlow's default filter and constants. |
| CSI, POD and FAR | Threshold predictions at `p >= 0.5`. Compute `TP/(TP+FP+FN)`, `TP/(TP+FN)` and `FP/(TP+FP)`, respectively. |
| Signed and absolute area bias | Signed bias is 100 times the difference in predicted and observed burnt pixel counts divided by the observed burnt pixel count. Absolute bias is its absolute value. |
| Mean boundary distance | Extract the inner boundaries of the two binary masks. Average the mean nearest-boundary distance in each direction, in pixels. |

Undefined ratios and distances with a missing boundary remain NaN. Each event has equal weight in the macro average of defined values. The summary records defined and undefined event counts and preserves positive infinite PSNR for exact matches. The formal evaluation reports these nine metrics. RRMSE belongs to the historical training logs.

The recorded SHA-256 of the matched-study dataset is `ebd788aeb294570faf263ec117c4d9817c6eb54f216556b47eb38304afb7fa68`. The shared preprocessing identifier is `fidn_fixed_split_train_minmax_v1`. Training and evaluation data hashes must agree. The fixed array shape, channel semantics and split apply to each dataset.

### Manuscript analysis

The manuscript compared the architectures on 30 paired test events after averaging matched-seed differences within each event. The research used two-sided Wilcoxon signed-rank tests with Holm adjustment across eight metrics, percentile intervals from 100,000 event-level bootstrap draws, and a separate exact sign-test sensitivity analysis with Holm adjustment. Signed area bias was descriptive. These analyses describe the manuscript methodology. The public scripts provide training and single-model evaluation with per-event metrics and macro averages.

## Source provenance

The public sources draw on FIDN-main-2 for the matched DenseNet and preprocessing, and on FIDN-2 for the Residual U-Net, evaluation and visualization. The three DenseNet training records name an earlier revision, `ede76bda5c42a07e2e7036759ea1e05dd21e830c`. Their recorded behavior is consistent with working tree changes. The verified upstream reference revisions identify source content available for comparison. The precise source bytes executed for each historical run remain unverified.

The manifest inventories current public-project files, excluding the local audit directory and the manifest itself. `current_sha256` hashes the exact bytes of a current file. Where an upstream source could be verified, `original_sha256` hashes the recorded upstream working copy and `git_blob_sha256` hashes the bytes stored at the specified reference revision. Those two hashes can differ because of line endings. `previous_candidate_sha256` and `source_sha256`, where present, are historical public-project records. Entries with `upstream_status` set to `unknown` have no independently verified upstream source file. [Source manifest](source_manifest.json)
