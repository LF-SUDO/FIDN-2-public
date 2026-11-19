"""Evaluation-only image metrics on unscaled sigmoid probabilities.

Training/serialized metrics in losses.py deliberately retain their historical
definitions. TensorFlow's max_val=1.0 is the data_range=1.0 convention. Keep
the original tf.image.ssim defaults (11, 1.5, 0.01, 0.03) and channels-last
layout; switching to skimage defaults would change more than normalization.
"""

import numpy as np
import tensorflow as tf


def validate_probability_pair(y_true, y_prob):
    """Validate an aligned event or batch without clipping or rescaling.

    This is an eager evaluation helper, not a compiled training metric.
    Validate the original arrays before any dtype conversion.
    """
    target, probability = np.asarray(y_true), np.asarray(y_prob)
    if target.shape != probability.shape:
        raise ValueError(f"Target/prediction shape mismatch: {target.shape} / {probability.shape}")
    if target.ndim not in (3, 4) or target.shape[-1] != 1:
        raise ValueError("Expected [H,W,1] or [N,H,W,1] single-channel images")
    if target.size == 0:
        raise ValueError("Empty target/prediction")
    if not np.issubdtype(probability.dtype, np.floating):
        raise ValueError("Prediction must contain floating-point sigmoid probabilities")
    if not np.isfinite(target).all() or not np.isfinite(probability).all():
        raise ValueError("Non-finite target/prediction; no automatic repair is permitted")
    if not np.isin(target, (0, 1)).all():
        raise ValueError(f"Target must be binary 0/1; found {np.unique(target)}")
    minimum, maximum = float(probability.min()), float(probability.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"Prediction outside [0,1]: [{minimum}, {maximum}]; no clipping")
    return {
        "prediction_min": minimum,
        "prediction_max": maximum,
        "prediction_below_zero": int(np.count_nonzero(probability < 0)),
        "prediction_above_one": int(np.count_nonzero(probability > 1)),
        "target_unique_values": np.unique(target).tolist(),
    }


def ssim_raw(y_true, y_prob):
    """SSIM on raw probabilities, with the original TensorFlow defaults."""
    validate_probability_pair(y_true, y_prob)
    # Float conversion is identical to the original TensorFlow metric kernel;
    # it does not change the probability scale (also avoids uint8 scaling).
    return tf.reduce_mean(tf.image.ssim(
        tf.cast(y_true, tf.float32), tf.cast(y_prob, tf.float32), max_val=1.0
    ))


def psnr_raw(y_true, y_prob):
    """PSNR on raw probabilities with a fixed data range of 1.0."""
    validate_probability_pair(y_true, y_prob)
    return tf.reduce_mean(tf.image.psnr(
        tf.cast(y_true, tf.float32), tf.cast(y_prob, tf.float32), max_val=1.0
    ))


def mse_raw(y_true, y_prob):
    """Use precisely the existing evaluation MSE reduction, without scaling."""
    validate_probability_pair(y_true, y_prob)
    return np.mean(tf.keras.metrics.mean_squared_error(
        np.asarray(y_true, dtype=np.float32), y_prob
    ).numpy())
