"""Configuration, run directories and provenance for local experiments."""

import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from FIDN.models import MODEL_NAMES
from FIDN.protocol import digest_file

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def now():
    return datetime.now(LOCAL_TIMEZONE)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def project_path(value):
    require(isinstance(value, str) and bool(value.strip()), "Expected a nonempty path string")
    path = Path(value)
    return (ROOT / path).resolve()


def nonnegative_integer(value, name):
    require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")


def validate_identity(model, seed):
    require(model in MODEL_NAMES, f"model must be one of {MODEL_NAMES}")
    nonnegative_integer(seed, "seed")


def read_config(path, kind):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(config, dict), "Configuration must be a JSON object")
    train_fields = {
        "model", "seed", "epochs", "batch_size", "optimizer", "learning_rate", "loss",
        "data_path", "save_path", "use_early_stopping", "early_stopping_patience",
    }
    evaluation_fields = {"data_path", "model_path", "save_path", "save_predictions"}
    fields = train_fields if kind == "train" else evaluation_fields
    optional = set() if kind == "train" else {"legacy"}
    require(fields <= config.keys(), f"Missing configuration fields: {sorted(fields - config.keys())}")
    require(config.keys() <= fields | optional, f"Unknown configuration fields: {sorted(config.keys() - fields - optional)}")
    for key in ("data_path", "save_path"):
        project_path(config[key])
    if kind == "train":
        validate_identity(config["model"], config["seed"])
        for key in ("epochs", "batch_size", "early_stopping_patience"):
            nonnegative_integer(config[key], key)
        require(config["epochs"] > 0 and config["batch_size"] > 0, "epochs and batch_size must be positive")
        require(type(config["use_early_stopping"]) is bool, "use_early_stopping must be boolean")
        require(config["optimizer"] == "adam", "Supported optimizer: adam")
        require(config["loss"] == "binary_crossentropy", "Supported loss: binary_crossentropy")
        rate = config["learning_rate"]
        require(type(rate) in (int, float) and math.isfinite(rate) and rate > 0,
                "learning_rate must be finite and positive")
    else:
        project_path(config["model_path"])
        require(type(config["save_predictions"]) is bool, "save_predictions must be boolean")
    return config


def create_run_directory(base, model, seed, started):
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{model}_seed{seed}_{started:%Y%m%d_%H%M%S}"
    index = 0
    while True:
        suffix = f"_{index}" if index else ""
        folder = base / (stem + suffix)
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            index += 1


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, columns=None):
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    sources = [ROOT / "train.py", ROOT / "evaluate.py", *sorted((ROOT / "FIDN").rglob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): digest_file(p) for p in sources}
