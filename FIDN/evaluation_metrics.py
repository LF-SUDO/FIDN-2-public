"""Spatial metrics for final burned-area event predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt


PREDICTION_THRESHOLD = 0.5

EVENT_METRIC_COLUMNS = (
    "event",
    "SSIM",
    "PSNR",
    "MSE",
    "RRMSE",
    "CSI",
    "POD",
    "FAR",
    "area_bias_percent",
    "mean_boundary_distance_pixels",
)

# Keep the original metrics.csv interface available for existing comparison
# scripts while also writing the canonical field names requested above.
LEGACY_METRIC_ALIASES = {
    "index": "event",
    "ssim": "SSIM",
    "psnr": "PSNR",
    "mse": "MSE",
    "rrmse": "RRMSE",
}

METRICS_CSV_COLUMNS = (
    *LEGACY_METRIC_ALIASES,
    *EVENT_METRIC_COLUMNS,
)

SUMMARY_STATISTICS = (
    "mean",
    "std",
    "median",
    "defined_count",
    "undefined_count",
)


def _as_spatial_array(values, name):
    """Return a single two-dimensional event image.

    Evaluation passes either ``(height, width)``, ``(height, width, 1)``,
    or a one-item model batch with shape ``(1, height, width, 1)``.
    Singleton batch/channel axes are removed explicitly so that a genuine
    spatial axis of length one is not accidentally squeezed away.
    """
    array = np.asarray(values)

    if array.ndim == 4:
        if array.shape[0] != 1 or array.shape[-1] != 1:
            raise ValueError(
                f"{name} must contain one event and one channel; "
                f"received shape {array.shape}"
            )
        array = array[0, ..., 0]
    elif array.ndim == 3:
        if array.shape[-1] == 1:
            array = array[..., 0]
        elif array.shape[0] == 1:
            array = array[0]
        else:
            raise ValueError(
                f"{name} must have a singleton batch or channel axis; "
                f"received shape {array.shape}"
            )

    if array.ndim != 2:
        raise ValueError(
            f"{name} must resolve to a two-dimensional mask; "
            f"received shape {array.shape}"
        )
    return array


def _binary_masks(y_true, y_pred):
    true_values = _as_spatial_array(y_true, "y_true")
    pred_values = _as_spatial_array(y_pred, "y_pred")
    if true_values.shape != pred_values.shape:
        raise ValueError(
            "y_true and y_pred must have the same spatial shape; "
            f"received {true_values.shape} and {pred_values.shape}"
        )
    if not np.all(np.isfinite(true_values)):
        raise ValueError("y_true contains non-finite pixel values")
    if not np.all(np.isfinite(pred_values)):
        raise ValueError("y_pred contains non-finite pixel values")

    # The target is stored as binary 0/1 values, sometimes in a floating
    # dtype. The same explicit cutoff converts it safely to a boolean mask.
    true_binary = true_values >= PREDICTION_THRESHOLD
    pred_binary = pred_values >= PREDICTION_THRESHOLD
    return true_binary, pred_binary


def _confusion_counts_from_masks(true_binary, pred_binary):
    return {
        "tp": int(np.count_nonzero(true_binary & pred_binary)),
        "fp": int(np.count_nonzero(~true_binary & pred_binary)),
        "fn": int(np.count_nonzero(true_binary & ~pred_binary)),
        "tn": int(np.count_nonzero(~true_binary & ~pred_binary)),
    }


def compute_confusion_counts(y_true, y_pred):
    """Compute pixel-level TP, FP, FN, and TN after fixed thresholding."""
    true_binary, pred_binary = _binary_masks(y_true, y_pred)
    return _confusion_counts_from_masks(true_binary, pred_binary)


def extract_boundary(mask):
    """Extract the one-pixel inner boundary of a two-dimensional mask."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be two-dimensional; received {mask.shape}")
    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    return mask ^ eroded


def mean_boundary_distance(true_binary, pred_binary):
    """Return symmetric mean nearest-boundary distance in pixels.

    If either mask has no boundary, the distance is mathematically undefined
    and ``np.nan`` is returned. Euclidean distance transforms avoid an
    all-pairs boundary-point calculation.
    """
    true_boundary = extract_boundary(true_binary)
    pred_boundary = extract_boundary(pred_binary)
    if not np.any(true_boundary) or not np.any(pred_boundary):
        return np.nan

    distance_to_true = distance_transform_edt(~true_boundary)
    distance_to_pred = distance_transform_edt(~pred_boundary)
    pred_to_true = float(np.mean(distance_to_true[pred_boundary]))
    true_to_pred = float(np.mean(distance_to_pred[true_boundary]))
    return 0.5 * (pred_to_true + true_to_pred)


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def compute_spatial_metrics(y_true, y_pred):
    """Compute event-level burned-area metrics using threshold 0.5.

    Undefined ratios and empty-boundary distances are represented by
    ``np.nan`` rather than artificial zero/one values.
    """
    true_binary, pred_binary = _binary_masks(y_true, y_pred)
    counts = _confusion_counts_from_masks(true_binary, pred_binary)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    csi = _safe_ratio(tp, tp + fp + fn)
    pod = _safe_ratio(tp, tp + fn)
    far = _safe_ratio(fp, tp + fp)

    n_true = tp + fn
    n_pred = tp + fp
    area_bias = _safe_ratio(n_pred - n_true, n_true)
    area_bias_percent = (
        np.nan if np.isnan(area_bias) else 100.0 * area_bias
    )

    return {
        "CSI": csi,
        "POD": pod,
        "FAR": far,
        "area_bias_percent": area_bias_percent,
        "mean_boundary_distance_pixels": mean_boundary_distance(
            true_binary, pred_binary
        ),
    }


def _finite_summary(values):
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    undefined_count = int(numeric.size - finite.size)
    if finite.size == 0:
        mean = std = median = np.nan
    else:
        mean = float(np.mean(finite))
        std = float(np.std(finite))
        median = float(np.median(finite))
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "defined_count": int(finite.size),
        "undefined_count": undefined_count,
    }


def summarize_event_metrics(metrics_df):
    """Build a NaN/non-finite-safe summary for every event-level metric."""
    metric_columns = [
        column for column in metrics_df.columns if column != "event"
    ]
    event_values = {
        column: metrics_df[column] for column in metric_columns
    }
    if "area_bias_percent" in metrics_df:
        event_values["mean_absolute_area_bias_percent"] = pd.to_numeric(
            metrics_df["area_bias_percent"], errors="coerce"
        ).abs()

    summary = {
        metric: _finite_summary(values)
        for metric, values in event_values.items()
    }
    summary_df = pd.DataFrame(summary).reindex(SUMMARY_STATISTICS)
    summary_df.index.name = "statistic"
    return summary_df


def save_evaluation_metrics(metrics, result_path):
    """Write per-event and aggregate metric CSV files.

    ``metrics.csv`` begins with the original lowercase fields for backwards
    compatibility, followed by the canonical fields. The summary uses only
    canonical fields so aliases are never treated as separate metrics.
    """
    canonical_df = pd.DataFrame(metrics, columns=EVENT_METRIC_COLUMNS)
    summary_df = summarize_event_metrics(canonical_df)
    metrics_df = canonical_df.assign(
        **{
            legacy_name: canonical_df[canonical_name]
            for legacy_name, canonical_name in LEGACY_METRIC_ALIASES.items()
        }
    ).loc[:, list(METRICS_CSV_COLUMNS)]
    result_path = Path(result_path)
    metrics_df.to_csv(result_path / "metrics.csv", index=False)
    summary_df.to_csv(result_path / "metrics_summary.csv")
    return metrics_df.set_index("index"), summary_df
