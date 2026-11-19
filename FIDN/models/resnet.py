"""ResNet18-based fire-image encoder used by FIDN."""

from keras import backend
from keras.engine import training
from keras.layers import VersionAwareLayers


layers = VersionAwareLayers()


def BasicBlock(x, filters, stride=1, name="basic_block"):
    """Build a ResNet18 BasicBlock for an NHWC feature tensor."""
    if stride not in (1, 2):
        raise ValueError("BasicBlock stride must be 1 or 2.")

    shortcut = x
    input_channels = backend.int_shape(shortcut)[-1]
    if input_channels is None:
        raise ValueError("BasicBlock requires a statically known channel count.")

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=f"{name}_bn1"
    )(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=f"{name}_bn2"
    )(x)

    if stride != 1 or input_channels != filters:
        shortcut = layers.Conv2D(
            filters,
            kernel_size=1,
            strides=stride,
            use_bias=False,
            name=f"{name}_shortcut_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(
            axis=3, epsilon=1.001e-5, name=f"{name}_shortcut_bn"
        )(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(x)


def _make_stage(x, filters, blocks, first_stride, name):
    """Build one ResNet stage from sequential BasicBlocks."""
    if blocks < 1:
        raise ValueError("A ResNet stage must contain at least one BasicBlock.")

    x = BasicBlock(
        x,
        filters=filters,
        stride=first_stride,
        name=f"{name}_block1",
    )
    for block_index in range(2, blocks + 1):
        x = BasicBlock(
            x,
            filters=filters,
            stride=1,
            name=f"{name}_block{block_index}",
        )
    return x


def ResidualEncoder128(input_channels=1, name="residual_encoder128"):
    """Construct the shared residual encoder for 128x128 FIDN inputs.

    The encoder uses the ResNet18 stem followed by stages 1 through 3. Its
    output interface matches the original DenseNet Encoder-128:
    ``(batch, 8, 8, 256)`` in NHWC layout.
    """
    if not isinstance(input_channels, int) or isinstance(input_channels, bool):
        raise ValueError("ResidualEncoder128 input_channels must be an integer.")
    if input_channels < 1:
        raise ValueError("ResidualEncoder128 input_channels must be positive.")

    img_input = layers.Input(
        shape=(128, 128, input_channels), name=f"{name}_input"
    )

    # ResNet stem: 128 -> 64 -> 32.
    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name=f"{name}_stem_pad"
    )(img_input)
    x = layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        use_bias=False,
        name=f"{name}_stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=f"{name}_stem_bn"
    )(x)
    x = layers.Activation("relu", name=f"{name}_stem_relu")(x)
    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name=f"{name}_stem_pool_pad"
    )(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, name=f"{name}_stem_pool"
    )(x)

    # ResNet18 stages 1-3 preserve the original 8x8x256 interface.
    x = _make_stage(
        x, filters=64, blocks=2, first_stride=1, name=f"{name}_stage1"
    )
    x = _make_stage(
        x, filters=128, blocks=2, first_stride=2, name=f"{name}_stage2"
    )
    x = _make_stage(
        x, filters=256, blocks=2, first_stride=2, name=f"{name}_stage3"
    )

    return training.Model(img_input, x, name=name)


def ResNet18Encoder(input_shape=(512, 512, 3), name="resnet18_encoder",
                    return_skips=False):
    """Construct the FIDN ResNet18 Encoder-512.

    The encoder is intentionally limited to the full-resolution fire-image
    branch. Its output interface matches the original DenseNet encoder:
    ``(batch, 8, 8, 352)`` in NHWC layout.

    By default the model returns only the latent, preserving the original
    interface. With ``return_skips=True`` its outputs are ordered as
    ``[latent, skip_16, skip_32, skip_64, skip_128, skip_256]``. The skips
    come from stage4/3/2/1 block2 outputs and the stem ReLU, respectively;
    exposing them adds no layers or weights.
    """
    img_input = layers.Input(shape=input_shape, name=f"{name}_input")
    actual_input_shape = backend.int_shape(img_input)[1:]
    if actual_input_shape != (512, 512, 3):
        raise ValueError(
            "ResNet18Encoder expects input_shape=(512, 512, 3); "
            f"received {actual_input_shape}."
        )

    # ResNet stem: 512 -> 256 -> 128.
    x = layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name=f"{name}_stem_pad"
    )(img_input)
    x = layers.Conv2D(
        64,
        kernel_size=7,
        strides=2,
        use_bias=False,
        name=f"{name}_stem_conv",
    )(x)
    x = layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=f"{name}_stem_bn"
    )(x)
    x = layers.Activation("relu", name=f"{name}_stem_relu")(x)
    skip_256 = x
    x = layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name=f"{name}_stem_pool_pad"
    )(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, name=f"{name}_stem_pool"
    )(x)

    # ResNet18 stages use [2, 2, 2, 2] BasicBlocks.
    x = _make_stage(
        x, filters=64, blocks=2, first_stride=1, name=f"{name}_stage1"
    )
    skip_128 = x
    x = _make_stage(
        x, filters=128, blocks=2, first_stride=2, name=f"{name}_stage2"
    )
    skip_64 = x
    x = _make_stage(
        x, filters=256, blocks=2, first_stride=2, name=f"{name}_stage3"
    )
    skip_32 = x
    x = _make_stage(
        x, filters=512, blocks=2, first_stride=2, name=f"{name}_stage4"
    )
    skip_16 = x

    # FIDN output adapter: 16x16x512 -> 8x8x352.
    x = layers.MaxPooling2D(
        pool_size=2, strides=2, name=f"{name}_output_pool"
    )(x)
    x = layers.Conv2D(
        352,
        kernel_size=1,
        use_bias=False,
        name=f"{name}_output_projection",
    )(x)
    x = layers.BatchNormalization(
        axis=3, epsilon=1.001e-5, name=f"{name}_output_bn"
    )(x)
    x = layers.Activation("relu", name=f"{name}_output_relu")(x)

    if return_skips:
        return training.Model(
            img_input, [x, skip_16, skip_32, skip_64, skip_128, skip_256],
            name=name,
        )
    return training.Model(img_input, x, name=name)
