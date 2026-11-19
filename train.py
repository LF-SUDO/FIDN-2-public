"""Train one FIDN model using configs/train.json.

Adapted from FIDN-main-2 and FIDN-2. Original FIDN implementation by
Bo Pang, Copyright 2022 IRP Project, Apache-2.0.
"""

import argparse
from importlib.metadata import version
import os
import platform
import random
import sys

from FIDN.models import build_model
from FIDN.protocol import PREPROCESSING, SPLIT_RANGES, digest_file, validate_raw
from FIDN.run_utils import (ROOT, create_run_directory, now, project_path, read_config,
                            require, source_hashes, write_json)


def seed_libraries(seed, np, tf):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    tf.random.set_seed(seed)


def selection_callbacks(keras, output, checkpoint, config):
    callbacks = [keras.callbacks.CSVLogger(str(output / "history.csv"))]
    if not config["use_early_stopping"]:
        return callbacks
    return callbacks + [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", mode="min",
            save_best_only=True, save_weights_only=False,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=config["early_stopping_patience"],
            restore_best_weights=True, verbose=1,
        ),
    ]


def checkpoint_filename(config, selected_epoch):
    stem = (f"{config['model']}_seed{config['seed']}_maxep{config['epochs']}"
            f"_bs{config['batch_size']}")
    if config["use_early_stopping"]:
        return f"{stem}_es{config['early_stopping_patience']}_bestep{selected_epoch}.h5"
    return f"{stem}_es0_lastep{selected_epoch}.h5"


def run(config):
    started = now()
    output = create_run_directory(project_path(config["save_path"]), config["model"], config["seed"], started)
    report = output / "training_run.json"
    manifest = {
        "schema_version": 2, "status": "incomplete", "started_at": started.isoformat(),
        "config": config, "model": config["model"], "seed": config["seed"],
        "numpy_seed": config["seed"] % (2 ** 32),
        "preprocessing": PREPROCESSING, "split_ranges": SPLIT_RANGES,
        "rotation_degrees": [0, 90, 180, 270], "source_sha256": source_hashes(),
        "shuffle": True, "monitor": "val_loss", "mode": "min",
        "restore_best_weights": config["use_early_stopping"],
    }
    write_json(report, manifest)
    try:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        import keras
        import numpy as np
        import tensorflow as tf
        from FIDN import dataset, losses

        require((sys.version_info[:2], tf.__version__, keras.__version__) == ((3, 9), "2.9.1", "2.9.0"),
                "Use Python 3.9, TensorFlow 2.9.1 and Keras 2.9.0")
        require(np.__version__ == "1.23.1" and tf.keras.mixed_precision.global_policy().name == "float32",
                "Use NumPy 1.23.1 and the float32 Keras policy")
        manifest["environment"] = {
            "python": platform.python_version(), "tensorflow": tf.__version__,
            "keras": keras.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "h5py": version("h5py"),
            "device_names": [tf.config.experimental.get_device_details(g).get("device_name", g.name)
                             for g in tf.config.list_physical_devices("GPU")],
            "tensorflow_build": dict(tf.sysconfig.get_build_info()),
            "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
            "flags": {k: os.environ.get(k) for k in (
                "TF_DETERMINISTIC_OPS", "TF_ENABLE_AUTO_MIXED_PRECISION",
                "TF_CUDNN_DETERMINISTIC", "TF_ENABLE_ONEDNN_OPTS", "NVIDIA_TF32_OVERRIDE")},
        }
        source = project_path(config["data_path"])
        require(source.is_file(), f"Dataset file does not exist: {source}")
        initial_stat = source.stat()
        raw = dataset.load_dataset(source)
        validate_raw(raw)
        manifest.update(data_path=str(source), dataset_sha256=digest_file(source),
                        shape=list(raw.shape), dtype=raw.dtype.name)

        # Preserve the historical preprocessing initialization and model seed reset.
        seed_libraries(42, np, tf)
        manifest["normalization_statistics"] = dataset.fit_auxiliary_normalization(raw[:243])
        write_json(report, manifest)
        _, _, _, _, x_train, y_train, x_val, y_val, _, _ = dataset.setup_dataset(source)
        require(len(x_train) == 972 and len(x_val) == 30, "Unexpected training or validation count")

        seed_libraries(config["seed"], np, tf)
        model = build_model(config["model"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
            loss=config["loss"], metrics=[
                losses.ssim_metrics, losses.psnr_metrics,
                losses.custom_mean_squared_error, losses.relative_root_mean_squared_error,
            ],
        )
        temporary_best = output / "best_checkpoint.tmp.h5"
        history = model.fit(
            x_train, y_train, batch_size=config["batch_size"], epochs=config["epochs"],
            validation_data=(x_val, y_val), shuffle=True,
            callbacks=selection_callbacks(tf.keras, output, temporary_best, config),
        )
        selected_epoch, values = finalize_checkpoint(model, history, config, temporary_best, output, np)
        final_stat = source.stat()
        require((initial_stat.st_size, initial_stat.st_mtime_ns) == (final_stat.st_size, final_stat.st_mtime_ns),
                "Input dataset changed during training")
        selected = output / checkpoint_filename(config, selected_epoch)
        best_index = min(range(len(values)), key=values.__getitem__)
        manifest.update(
            completed_epochs=len(values), best_epoch=best_index + 1,
            best_val_loss=float(values[best_index]), last_val_loss=float(values[-1]),
            selected_epoch=selected_epoch, selected_val_loss=float(values[selected_epoch - 1]),
            selection="best_val_loss" if config["use_early_stopping"] else "last_epoch",
            h5_file=selected.name, h5_sha256=digest_file(selected),
            history_file="history.csv", history_sha256=digest_file(output / "history.csv"),
            status="complete",
        )
        print(f"Saved model: {selected}")
        return output
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_at"] = now().isoformat()
        write_json(report, manifest)


def finalize_checkpoint(model, history, config, temporary_best, output, np):
    values = history.history.get("val_loss", [])
    require(len(values) > 0 and np.isfinite(values).all(), "Training did not yield finite validation losses")
    require(len(values) <= config["epochs"], "Unexpected completed epoch count")
    selected_epoch = min(range(len(values)), key=values.__getitem__) + 1
    if config["use_early_stopping"]:
        require(temporary_best.is_file(), "Best checkpoint was not saved")
        saved = temporary_best
    else:
        require(len(values) == config["epochs"], "Training stopped before the configured last epoch")
        selected_epoch = len(values)
        saved = output / "last_checkpoint.tmp.h5"
        model.save(str(saved), save_format="h5")
    destination = output / checkpoint_filename(config, selected_epoch)
    require(not destination.exists(), f"Checkpoint already exists: {destination}")
    saved.rename(destination)
    return selected_epoch, values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    run(read_config(ROOT / "configs/train.json", "train"))


if __name__ == "__main__":
    main()
