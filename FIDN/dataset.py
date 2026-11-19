# Adapted for the paper reproduction package. Removed unused legacy helpers.
__author__ = "Bo Pang"
__copyright__ = "Copyright 2022, IRP Project"
__credits__ = ["Bo Pang"]
__license__ = "Apache 2.0"
__version__ = "1.0"

"""
This file stores functions related to the processing of data sets
"""

import numpy as np


RAW_EVENT_COUNT = 304
TRAIN_EVENT_COUNT = 243
VALIDATION_EVENT_COUNT = 30
TEST_EVENT_COUNT = 30
MODELLING_EVENT_COUNT = (
    TRAIN_EVENT_COUNT + VALIDATION_EVENT_COUNT + TEST_EVENT_COUNT
)
VALIDATION_START = TRAIN_EVENT_COUNT
TEST_START = TRAIN_EVENT_COUNT + VALIDATION_EVENT_COUNT
AUXILIARY_CHANNELS = tuple(range(3, 14))


def create_shifted_frames(data):
    """
    Helper function for splitting a single piece
    of data into source and target

    Args:
        data: Individual data to be split

    Returns:
        Source data x, Target data y

    """
    x = data[..., :-1]
    y = data[..., -1:]
    return x, y


def load_dataset(fpath, mmap_mode="r"):
    """
    Read the raw dataset from a NumPy file.

    The default read-only memory mapping avoids loading the complete source
    file into RAM and guarantees that preprocessing cannot change it.
    Auxiliary channels are deliberately not normalised here because their
    statistics must be fitted after the modelling dataset has been split.
    The source file must contain exactly 304 events; unexpected counts raise
    ValueError instead of silently changing the matched experiment.

    Args:
        fpath: Path of the data set in the file system
        mmap_mode: NumPy memory-map mode. ``"r"`` is read-only.

    Returns:
        Raw data set
    """
    dataset = np.load(fpath, mmap_mode=mmap_mode)
    if dataset.ndim == 0 or dataset.shape[0] != RAW_EVENT_COUNT:
        raise ValueError(
            f"Expected a raw FIDN file with exactly {RAW_EVENT_COUNT} events; "
            f"received shape {dataset.shape}. Check the dataset explicitly "
            "before applying the matched protocol."
        )
    return dataset


def select_modelling_dataset(dataset):
    """Select the 303 events used by the published modelling protocol."""
    if dataset.shape[0] not in (MODELLING_EVENT_COUNT, RAW_EVENT_COUNT):
        raise ValueError(
            f"Expected {RAW_EVENT_COUNT} raw events or an already selected "
            f"{MODELLING_EVENT_COUNT}-event view; received {dataset.shape[0]}."
        )
    return dataset[:MODELLING_EVENT_COUNT]


def fit_auxiliary_normalization(train_dataset):
    """Fit per-channel min/max statistics on the raw training subset only."""
    if train_dataset.shape[-1] <= max(AUXILIARY_CHANNELS):
        raise ValueError(
            "The FIDN dataset must contain auxiliary channels 3 through 13."
        )

    return {
        channel: (
            float(np.min(train_dataset[..., channel])),
            float(np.max(train_dataset[..., channel])),
        )
        for channel in AUXILIARY_CHANNELS
    }


def apply_auxiliary_normalization(dataset, normalization_stats,
                                  scale_range=None):
    """Apply training-fitted min/max statistics to auxiliary channels.

    Values outside the extrema observed in training are intentionally not
    clipped.  A channel that is constant in training is left unchanged, which
    preserves the historical safe handling of a near-zero denominator.
    """
    if not np.issubdtype(dataset.dtype, np.floating):
        raise TypeError("Auxiliary normalization requires floating-point data.")

    if scale_range is None:
        scale_range = [0, 1]
    scale_min = min(scale_range)
    scale_max = max(scale_range)

    for channel in AUXILIARY_CHANNELS:
        data_min, data_max = normalization_stats[channel]
        if np.isclose(data_max, data_min):
            continue

        channel_data = dataset[..., channel]
        channel_data -= data_min
        channel_data /= data_max - data_min
        channel_data *= scale_max - scale_min
        channel_data += scale_min

    return dataset


def split_dataset(dataset, normalize_auxiliary=True):
    """
    Apply the fixed published split to the 303-event modelling dataset.

    The split is chronological and never shuffled: events 0:243 are used for
    training, 243:273 for validation, and 273:303 for testing.  By default,
    auxiliary-channel statistics are fitted on the raw training view and then
    applied to one copy of each subset. The source dataset remains unchanged.

    Args:
        dataset: Original full dataset
        normalize_auxiliary: Whether to perform training-fitted normalization

    Returns:
        Training set, validation set, test set
    """
    dataset = select_modelling_dataset(dataset)
    train_dataset = dataset[:VALIDATION_START]
    val_dataset = dataset[VALIDATION_START:TEST_START]
    test_dataset = dataset[TEST_START:MODELLING_EVENT_COUNT]

    if normalize_auxiliary:
        normalization_stats = fit_auxiliary_normalization(train_dataset)
        train_dataset = np.array(train_dataset, copy=True, order="C")
        val_dataset = np.array(val_dataset, copy=True, order="C")
        test_dataset = np.array(test_dataset, copy=True, order="C")
        apply_auxiliary_normalization(train_dataset, normalization_stats)
        apply_auxiliary_normalization(val_dataset, normalization_stats)
        apply_auxiliary_normalization(test_dataset, normalization_stats)

    return train_dataset, val_dataset, test_dataset


def augment_training_dataset(train_dataset, normalization_stats=None):
    """Return original, 90, 180 and 270 degree training samples.

    A single output buffer is allocated so the real float64 dataset does not
    incur the large temporary arrays produced by repeated concatenation.
    """
    event_count = train_dataset.shape[0]
    augmented_shape = (event_count * 4, *train_dataset.shape[1:])
    augmented_dataset = np.empty(augmented_shape, dtype=train_dataset.dtype)

    original_block = augmented_dataset[:event_count]
    np.copyto(original_block, train_dataset)
    if normalization_stats is not None:
        apply_auxiliary_normalization(original_block, normalization_stats)

    for block_index, rotation in enumerate((1, 2, 3), start=1):
        block_start = block_index * event_count
        block_stop = block_start + event_count
        block = augmented_dataset[block_start:block_stop]
        np.copyto(
            block,
            np.rot90(original_block, k=rotation, axes=(1, 2)),
        )

    return augmented_dataset


def setup_dataset(data_path):
    """
    Read the raw dataset, select and split the 303 modelling events, fit
    auxiliary normalization on training only, and augment all training events.

    Args:
        data_path: Path of the data set in the file system

    Returns:
        raw 303-event dataset view, train_dataset, val_dataset, test_dataset,
        x_train, y_train, x_val, y_val, x_test, y_test

    """
    raw_dataset = load_dataset(data_path)
    dataset = select_modelling_dataset(raw_dataset)
    raw_train, raw_val, raw_test = split_dataset(
        dataset, normalize_auxiliary=False
    )
    normalization_stats = fit_auxiliary_normalization(raw_train)

    train_dataset = augment_training_dataset(raw_train, normalization_stats)
    val_dataset = np.array(raw_val, copy=True, order="C")
    test_dataset = np.array(raw_test, copy=True, order="C")
    apply_auxiliary_normalization(val_dataset, normalization_stats)
    apply_auxiliary_normalization(test_dataset, normalization_stats)

    x_train, y_train = create_shifted_frames(train_dataset)
    x_val, y_val = create_shifted_frames(val_dataset)
    x_test, y_test = create_shifted_frames(test_dataset)

    return dataset, train_dataset, val_dataset, test_dataset, x_train, y_train, x_val, y_val, x_test, y_test
