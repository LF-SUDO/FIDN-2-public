"""The two architectures in the matched comparison."""

from importlib import import_module

MODEL_NAMES = ("matched_densenet", "residual_unet")
EXPECTED_PARAMS = {
    "matched_densenet": (16876041, 16714465, 161576),
    "residual_unet": (33593497, None, None),
}


def build_model(name):
    """Build the selected architecture on a single channels-last input."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}")
    from tensorflow import keras

    if keras.backend.image_data_format() != "channels_last":
        raise ValueError("This study uses channels-last inputs")
    module = import_module(f"FIDN.models.{name}")
    builder = getattr(module, f"build_{name}")
    model = builder(keras.layers.Input(shape=(512, 512, 14)))
    if model.count_params() != EXPECTED_PARAMS[name][0]:
        raise ValueError(f"Unexpected parameter count for {name}")
    return model
