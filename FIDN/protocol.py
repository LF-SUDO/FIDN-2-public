"""Fixed scientific data protocol shared by training and evaluation."""

import hashlib
from pathlib import Path

SHAPE = (304, 512, 512, 15)
SPLIT_RANGES = [[0, 243], [243, 273], [273, 303]]
PREPROCESSING = "fidn_fixed_split_train_minmax_v1"
ORIGINAL_DATA_SHA256 = "ebd788aeb294570faf263ec117c4d9817c6eb54f216556b47eb38304afb7fa68"
PAPER_SEEDS = (42, 123, 2024)
PAPER_BATCH_SIZE = 10


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw(raw):
    if raw.shape != SHAPE or raw.dtype.name != "float64":
        raise ValueError(f"Expected float64 {SHAPE}, got {raw.dtype} {raw.shape}")
