"""Evaluate one complete H5 model using configs/evaluate.json.

Formal per-event metrics and inference path adapted from FIDN-2 canonical run2.
"""

import argparse
import ast
import hashlib
import importlib
import inspect
from importlib.metadata import version
import json
import logging
import math
from pathlib import Path
import os
import platform
import random
import subprocess
import sys
import textwrap

from FIDN.models import EXPECTED_PARAMS
from FIDN.protocol import PREPROCESSING, SPLIT_RANGES, digest_file, validate_raw
from FIDN.run_utils import (ROOT, create_run_directory, now, project_path, read_config,
                            require, source_hashes, validate_identity, write_csv, write_json)

EVENTS = tuple(range(30))
METRICS = {
    "ssim_raw": "ssim_raw", "psnr_raw": "psnr_raw", "mse_raw": "mse_raw",
    "csi_audit": "csi", "pod_audit": "pod", "far_audit": "far",
    "signed_area_bias_audit": "signed_area_bias",
    "absolute_area_bias_audit": "absolute_area_bias",
    "boundary_distance_audit": "boundary_distance",
}
FLAGS = {"TF_ENABLE_AUTO_MIXED_PRECISION": "0",
         "TF_ENABLE_ONEDNN_OPTS": "0", "TF_DETERMINISTIC_OPS": "1"}
LOG = logging.getLogger("single_model_evaluation")
METRIC_DEFINITIONS = {
    "continuous": "Unscaled float32 sigmoid probabilities; direct float32 MSE; TensorFlow SSIM/PSNR on CPU, max_val=1",
    "ssim": "TensorFlow defaults: filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03",
    "binary_threshold": 0.5,
    "spatial": "CSI, POD, FAR; signed/absolute area bias percent; symmetric mean nearest-boundary distance in pixels",
    "undefined": "Zero-denominator ratios and unavailable boundaries are NaN; perfect PSNR is +Infinity",
    "macro": "Equal event weights; exclude nonfinite values except retain +Infinity for PSNR; record defined counts",
}


def array_hash(array, np):
    """Section 4.3 content hash: dtype + shape + C-order bytes."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": array.dtype.str,
                              "shape": list(array.shape)}, sort_keys=True).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def validate_probability(probability, np):
    # Check the ORIGINAL dtype/range; never cast, clip, or repair bad predictions.
    require(probability.shape in ((512, 512), (512, 512, 1)),
            f"Invalid probability shape: {probability.shape}")
    require(probability.dtype == np.float32, f"Expected float32, got {probability.dtype}")
    require(np.isfinite(probability).all(), "Probability contains NaN/Inf; stopping")
    minimum, maximum = float(probability.min()), float(probability.max())
    require(minimum >= 0 and maximum <= 1,
            f"Probability outside [0,1]: [{minimum}, {maximum}]; stopping without clipping")
    return {"min": minimum, "max": maximum, "finite": True,
            "near_threshold_1e_6": int(np.count_nonzero(abs(probability - 0.5) <= 1e-6)),
            "near_threshold_1e_4": int(np.count_nonzero(abs(probability - 0.5) <= 1e-4)),
            "near_threshold_1e_3": int(np.count_nonzero(abs(probability - 0.5) <= 1e-3))}


def runtime_environment(manifest):
    require("tensorflow" not in sys.modules, "Start this standalone script in a fresh process")
    for name, expected in FLAGS.items():
        require(os.environ.get(name, expected) == expected, f"Conflicting {name}; expected {expected}")
        os.environ[name] = expected
    for name in ("TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE",
                 "TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING"):
        require(os.environ.get(name, "0") == "0", f"Conflicting AMP override: {name}")
    require(not os.environ.get("TF_XLA_FLAGS"), "Unset TF_XLA_FLAGS to preserve the verified default")
    require(not os.environ.get("NVIDIA_TF32_OVERRIDE"), "Unset NVIDIA_TF32_OVERRIDE")
    require(os.environ.get("TF_CUDNN_DETERMINISTIC") in (None, "1"),
            "TF_CUDNN_DETERMINISTIC conflicts with deterministic execution")
    modules = {n: importlib.import_module(n) for n in
               ("numpy", "tensorflow", "keras", "scipy", "pandas", "h5py")}
    tf = modules["tensorflow"]
    environment = {"python": sys.version, "platform": platform.platform(),
                   **{n: m.__version__ for n, m in modules.items()},
                   "flags": {k: os.environ.get(k) for k in (
                       *FLAGS, "TF_CUDNN_DETERMINISTIC", "CUDA_VISIBLE_DEVICES",
                       "TF_XLA_FLAGS", "NVIDIA_TF32_OVERRIDE",
                       "TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE",
                       "TF_ENABLE_AUTO_MIXED_PRECISION_LOSS_SCALING")}}
    manifest["environment"] = environment
    for name, version in {"tensorflow": "2.9.1", "keras": "2.9.0", "numpy": "1.23.1",
                          "scipy": "1.9.0", "h5py": "3.14.0"}.items():
        require(environment[name] == version, f"Expected {name} {version}, got {environment[name]}")
    require(sys.version_info[:2] == (3, 9), "Use the verified Python 3.9 environment")
    gpus = tf.config.list_physical_devices("GPU")
    environment["visible_gpus"] = [{"name": g.name,
                                    "details": tf.config.experimental.get_device_details(g)} for g in gpus]
    environment["tensorflow_build"] = tf.sysconfig.get_build_info()
    # Preserve default TF32 behavior: Section 4.3 did not explicitly toggle it.
    environment["tf32_enabled"] = tf.config.experimental.tensor_float_32_execution_enabled()
    tf.keras.mixed_precision.set_global_policy("float32")
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})
    tf.config.optimizer.set_jit(False)
    tf.config.set_soft_device_placement(True)
    logical_gpus = tf.config.list_logical_devices("GPU")
    LOG.info("TensorFlow soft device placement: %s", tf.config.get_soft_device_placement())
    LOG.info("Visible GPUs: %s", tf.config.get_visible_devices("GPU"))
    LOG.info("Logical GPUs: %s", logical_gpus)
    environment.update({"amp_enabled": False, "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name,
                        "optimizer_options": tf.config.optimizer.get_experimental_options(),
                        "xla_jit": tf.config.optimizer.get_jit(), "execution_device": "automatic",
                        "soft_device_placement": tf.config.get_soft_device_placement(),
                        "logical_gpus": [{"name": g.name, "device_type": g.device_type} for g in logical_gpus],
                        "onednn_enabled": False, "deterministic_ops": True})
    try:
        environment["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        environment["nvidia_smi_error"] = str(exc)
    for name in ("dataset", "losses", "evaluation_metrics", "raw_probability_metrics"):
        modules[name] = importlib.import_module("FIDN." + name)
    return modules

def validate_predict_step(model, tf):
    """Keep the original Keras predict adapter and verify its training=False call."""
    require(model.predict_step.__func__ is tf.keras.Model.predict_step,
            "Unexpected custom predict_step; refusing a changed inference path")
    source = textwrap.dedent(inspect.getsource(model.predict_step))
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "self"]
    require(len(calls) == 1 and any(k.arg == "training" and isinstance(k.value, ast.Constant)
                                   and k.value.value is False for k in calls[0].keywords),
            "Keras predict_step must explicitly call self(x, training=False)")
    return source


def weights_hash(model, np):
    digest = hashlib.sha256()
    for variable in model.weights:  # includes BatchNorm moving statistics
        digest.update(array_hash(variable.numpy(), np).encode())
    return digest.hexdigest()


def prepare_event(test, event, stats, ds, np):
    sample = np.array(test[event:event + 1], copy=True, order="C")
    untouched = sample[..., [0, 1, 2, 14]].copy()
    ds.apply_auxiliary_normalization(sample, stats)
    require(np.array_equal(untouched, sample[..., [0, 1, 2, 14]]),
            "Fire observations/target changed during preprocessing")
    x, y = ds.create_shifted_frames(sample)
    x, y = np.ascontiguousarray(x[0]), np.ascontiguousarray(y[0, ..., 0])
    require(x.shape == (512, 512, 14) and x.dtype == np.float64, "Unexpected input shape/dtype")
    require(np.isfinite(x).all() and np.isin(y, (0, 1)).all(), "Invalid input/binary target")
    return x, y


def compute_metrics(target, probability, modules):
    np, tf = modules["numpy"], modules["tensorflow"]
    raw, spatial = modules["raw_probability_metrics"], modules["evaluation_metrics"]
    y = target.astype(np.float32)[None, ..., None]
    p = probability[None, ..., None]
    raw.validate_probability_pair(y, p)
    # Single-channel direct float32 mean square; no normalization anywhere.
    mse = float(np.mean((y - p) ** 2, dtype=np.float32))
    # Pin continuous metric execution device separately from GPU inference.
    with tf.device("/CPU:0"):
        ssim = float(raw.ssim_raw(y, p).numpy())
        psnr = float(raw.psnr_raw(y, p).numpy())
    require(math.isfinite(ssim) and math.isfinite(mse), "Non-finite continuous metric")
    require(math.isfinite(psnr) or (psnr == math.inf and mse == 0), "Invalid PSNR")
    result = spatial.compute_spatial_metrics(target, probability)
    return {"ssim_raw": ssim, "psnr_raw": psnr, "mse_raw": mse,
            "csi_audit": result["CSI"], "pod_audit": result["POD"], "far_audit": result["FAR"],
            "signed_area_bias_audit": result["area_bias_percent"],
            "absolute_area_bias_audit": abs(result["area_bias_percent"]),
            "boundary_distance_audit": result["mean_boundary_distance_pixels"]}


def macro(values, include_infinity=False):
    defined = [v for v in values if (not math.isnan(v) if include_infinity else math.isfinite(v))]
    return (sum(defined) / len(defined) if defined else math.nan), len(defined)


def read_identity(config):
    model_path = project_path(config["model_path"])
    require(model_path.suffix.lower() == ".h5" and model_path.is_file(),
            f"model_path must name an existing specific .h5 file: {model_path}")
    model_hash = digest_file(model_path)
    record_path = model_path.parent / "training_run.json"
    if record_path.is_file():
        require("legacy" not in config, "Remove legacy overrides when training_run.json exists")
        meta = json.loads(record_path.read_text(encoding="utf-8"))
        require(meta.get("status") == "complete", "Training record is not complete")
        name = meta.get("h5_file", meta.get("best_h5"))
        expected = meta.get("h5_sha256", meta.get("best_h5_sha256"))
        require(name == model_path.name and expected == model_hash,
                "H5 filename/SHA-256 differs from training_run.json")
        require(meta.get("split_ranges") == SPLIT_RANGES, "Training split differs from the fixed protocol")
        require(meta.get("rotation_degrees") == [0, 90, 180, 270], "Training rotation protocol differs")
        if "preprocessing" in meta:
            require(meta["preprocessing"] == PREPROCESSING, "Unsupported preprocessing protocol")
        else:
            # Compatibility with the prior public train.py manifest.
            dataset_source = ROOT / "FIDN/dataset.py"
            known_hashes = {digest_file(dataset_source), hashlib.sha256(
                dataset_source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}
            require(meta.get("source_sha256", {}).get("FIDN/dataset.py") in known_hashes,
                    "Historical training record has no verified preprocessing source")
        require("normalization_statistics" in meta, "Training record is missing normalization_statistics")
        identity = dict(meta)
        identity["training_record"] = {"path": str(record_path), "sha256": digest_file(record_path)}
    else:
        legacy = config.get("legacy")
        require(isinstance(legacy, dict),
                "Missing training_run.json. For a historical H5, explicitly provide legacy model, seed, "
                "dataset_sha256, h5_sha256 and preprocessing in configs/evaluate.json. See README.")
        required = {"model", "seed", "dataset_sha256", "h5_sha256", "preprocessing"}
        require(set(legacy) == required, f"legacy must contain exactly {sorted(required)}")
        require(legacy["preprocessing"] == PREPROCESSING, "Unsupported legacy preprocessing")
        require(legacy["h5_sha256"] == model_hash, "Legacy H5 SHA-256 mismatch")
        identity = dict(legacy)
        identity["training_record"] = None
    validate_identity(identity.get("model"), identity.get("seed"))
    expected_data = identity.get("dataset_sha256")
    require(isinstance(expected_data, str) and len(expected_data) == 64
            and all(c in "0123456789abcdef" for c in expected_data), "Invalid dataset SHA-256 in record")
    identity.update(model_path=str(model_path), h5_sha256=model_hash)
    return identity


def normalization_for_evaluation(identity, raw, ds):
    if identity["training_record"] is None:
        # Explicit legacy compatibility choice, never use validation/test extrema.
        return ds.fit_auxiliary_normalization(raw[:243])
    recorded = identity["normalization_statistics"]
    require(isinstance(recorded, dict) and set(recorded) == {str(i) for i in range(3, 14)},
            "Normalization must contain exactly channels 3 through 13")
    stats = {}
    for channel in range(3, 14):
        pair = recorded[str(channel)]
        require(isinstance(pair, (list, tuple)) and len(pair) == 2
                and all(type(v) in (int, float) and math.isfinite(v) for v in pair)
                and pair[0] <= pair[1], f"Invalid normalization extrema for channel {channel}")
        stats[channel] = tuple(pair)
    return stats


def summarize(rows, model, seed):
    require([r["event"] for r in rows] == list(EVENTS), "Incomplete or duplicate test events")
    require(all(r["model"] == model and r["seed"] == seed for r in rows), "Mixed model identities")
    record = {"model": model, "seed": seed, "n_events": 30}
    for metric, name in METRICS.items():
        mean, count = macro([r[metric] for r in rows], metric == "psnr_raw")
        record["mean_" + name] = mean
        record["defined_events_" + name] = count
        record["undefined_events_" + name] = 30 - count
    return record


def evaluate_events(model, test, stats, identity, output, config, modules):
    np, ds = modules["numpy"], modules["dataset"]
    rows = []
    for event in EVENTS:
        x, y = prepare_event(test, event, stats, ds, np)
        batch = np.asarray(model.predict(np.expand_dims(x, axis=0), verbose=0))
        require(batch.shape == (1, 512, 512, 1), f"Wrong prediction shape: {batch.shape}")
        probability = batch[0, ..., 0]
        validate_probability(probability, np)
        metrics = compute_metrics(y, probability, modules)
        rows.append({"model": identity["model"],
                     "architecture": "densenet" if identity["model"] == "matched_densenet" else identity["model"],
                     "seed": identity["seed"], "event": event,
                     "original_dataset_index": 273 + event, **metrics})
        if config["save_predictions"]:
            from FIDN.visualization import save_event_jpgs
            # Copy raw event and normalize the display exactly as the original evaluator.
            display = np.array(test[event:event + 1], copy=True)
            ds.apply_auxiliary_normalization(display, stats)
            save_event_jpgs(display[0], probability, output, event)
        LOG.info("%s seed %d event %d", identity["model"], identity["seed"], event)
    return rows


def run(config):
    started = now()
    identity = read_identity(config)
    output = create_run_directory(project_path(config["save_path"]), identity["model"], identity["seed"], started)
    manifest = {
        "schema_version": 2, "status": "incomplete", "started_at": started.isoformat(),
        "config": config, "model": identity["model"], "seed": identity["seed"],
        "model_path": identity["model_path"], "h5_sha256": identity["h5_sha256"],
        "training_record": identity["training_record"], "metric_definitions": METRIC_DEFINITIONS,
        "preprocessing": PREPROCESSING, "test_indices": [273, 303],
        "source_sha256": source_hashes(), "outputs": [],
    }
    report = output / "evaluation_run.json"
    write_json(report, manifest)
    try:
        data = project_path(config["data_path"])
        require(data.is_file(), f"Dataset file does not exist: {data}")
        data_hash = digest_file(data)
        manifest.update(data_path=str(data), dataset_sha256=data_hash)
        require(data_hash == identity["dataset_sha256"], "Dataset SHA-256 differs from training/legacy record")
        import h5py
        with h5py.File(identity["model_path"], "r") as handle:
            require("model_config" in handle.attrs and "model_weights" in handle, "Expected a complete full-model H5")
            graph = json.loads(handle.attrs["model_config"])
            require(graph["config"]["layers"][-1]["config"].get("activation") == "sigmoid", "Expected sigmoid output")
        modules = runtime_environment(manifest)
        if config["save_predictions"]:
            manifest.setdefault("environment", {}).update(
                matplotlib=version("matplotlib"), pillow=version("Pillow"))
        np, tf, ds = (modules[name] for name in ("numpy", "tensorflow", "dataset"))
        raw = np.load(data, mmap_mode="r", allow_pickle=False)
        validate_raw(raw)
        stats = normalization_for_evaluation(identity, raw, ds)
        manifest["normalization_statistics"] = stats
        manifest["normalization_source"] = "training_run.json" if identity["training_record"] else "explicit legacy training-only fit"
        _, _, test = ds.split_dataset(raw, normalize_auxiliary=False)
        tf.keras.backend.clear_session()
        random.seed(identity["seed"])
        np.random.seed(identity["seed"] % (2 ** 32))
        tf.random.set_seed(identity["seed"])
        custom_objects = {name: getattr(modules["losses"], name) for name in (
            "ssim_metrics", "psnr_metrics", "custom_mean_squared_error", "relative_root_mean_squared_error")}
        model = tf.keras.models.load_model(identity["model_path"], compile=False, custom_objects=custom_objects)
        require(tuple(model.input_shape) == (None, 512, 512, 14), "Invalid model input")
        require(tuple(model.output_shape) == (None, 512, 512, 1), "Invalid model output")
        require(model.count_params() == EXPECTED_PARAMS[identity["model"]][0], "Model parameter count differs from architecture")
        require(all(layer.dtype_policy.compute_dtype == "float32" for layer in model.submodules
                    if hasattr(layer, "dtype_policy")), "Non-float32 checkpoint policy")
        require(modules["evaluation_metrics"].PREDICTION_THRESHOLD == 0.5, "Spatial threshold changed")
        manifest["predict_step_source"] = validate_predict_step(model, tf)
        before = weights_hash(model, np)
        rows = evaluate_events(model, test, stats, identity, output, config, modules)
        require(weights_hash(model, np) == before, "Inference modified model weights or BatchNorm state")
        manifest["weights_sha256"] = before
        require(digest_file(data) == data_hash, "Dataset changed during evaluation")
        require(digest_file(identity["model_path"]) == identity["h5_sha256"], "H5 changed during evaluation")
        if identity["training_record"]:
            record = identity["training_record"]
            require(digest_file(record["path"]) == record["sha256"], "Training record changed during evaluation")
        summary = summarize(rows, identity["model"], identity["seed"])
        write_csv(output / "metrics_per_event.csv", rows)
        write_csv(output / "metrics_summary.csv", [summary])
        manifest["outputs"] = [{"path": p.relative_to(output).as_posix(), "sha256": digest_file(p), "bytes": p.stat().st_size}
                               for p in sorted(output.rglob("*")) if p.is_file() and p != report]
        manifest.update(status="complete", completed_events=len(rows),
                        jpg_count=sum(p["path"].endswith(".jpg") for p in manifest["outputs"]))
        print(f"Evaluation saved: {output}")
        return output
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_at"] = now().isoformat()
        write_json(report, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(read_config(ROOT / "configs/evaluate.json", "evaluate"))


if __name__ == "__main__":
    main()


