# Adapted for the paper reproduction package. Extracted the published Residual U-Net variant.
__author__ = "Bo Pang"
__copyright__ = "Copyright 2022, IRP Project"
__credits__ = ["Bo Pang"]
__license__ = "Apache 2.0"
__version__ = "1.0"

from tensorflow import keras
from keras.api._v2 import keras
from keras.layers import VersionAwareLayers

from FIDN.models.resnet import BasicBlock, ResidualEncoder128, ResNet18Encoder

layers = VersionAwareLayers()

def build_residual_unet(inp):
    """Build the paper model with five fire-encoder skip connections."""
    border_size = int((inp.shape[2] - 128) / 2)
    other_inp = inp[:, border_size:-border_size, border_size:-border_size, ...]

    # Build FIDN encoder for different layers
    dse_fire = ResNet18Encoder(
        input_shape=(*inp.shape[-3:-1], 3),
        name='dse_fire',
        return_skips=True,
    )
    dse_biomass_above = ResidualEncoder128(
        input_channels=1,
        name='dse_biomass_above'
    )
    dse_biomass_under = ResidualEncoder128(
        input_channels=1,
        name='dse_biomass_under'
    )
    dse_slope = ResidualEncoder128(
        input_channels=1,
        name='dse_slope'
    )
    dse_tree_grass_bare_snow_water = ResidualEncoder128(
        input_channels=5,
        name='dse_tree_grass_bare_snow_water'
    )
    dse_wind = ResidualEncoder128(
        input_channels=2,
        name='dse_wind'
    )
    dse_precipitation = ResidualEncoder128(
        input_channels=1,
        name='dse_precipitation'
    )

    feature_fire, *decoder_skips = dse_fire(inp[..., 0:3])
    feature_biomass_above = dse_biomass_above(other_inp[..., 3:4])
    feature_biomass_under = dse_biomass_under(other_inp[..., 4:5])
    feature_slope = dse_slope(other_inp[..., 5:6])
    feature_tree_grass_bare_snow_water = dse_tree_grass_bare_snow_water(
        other_inp[..., 6:11])
    feature_wind = dse_wind(other_inp[..., 11:13])
    feature_precipitation = dse_precipitation(other_inp[..., 13])

    # Concat all of the feature maps
    x = layers.Concatenate(
        axis=-1, name='bottleneck_fusion'
    )([
        feature_fire, feature_biomass_above, feature_biomass_under,
        feature_slope, feature_tree_grass_bare_snow_water,
        feature_wind, feature_precipitation,
    ])

    # Construct FIDN Decoder
    for stage, filters in enumerate((256, 128, 64, 32, 16, 4), start=1):
        x = layers.Conv2DTranspose(
            filters, kernel_size=2, strides=(2, 2),
            name=f'decoder_{stage}_conv2dtranspose',
        )(x)
        if stage <= len(decoder_skips):
            x = layers.Concatenate(
                axis=-1, name=f'decoder_{stage}_fire_skip_concat',
            )([x, decoder_skips[stage - 1]])
        x = BasicBlock(
            x, filters=filters, stride=1,
            name=f'decoder_{stage}_residual_block',
        )
    x = layers.Conv2D(
        filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same"
    )(x)

    model = keras.models.Model(inp, x)
    return model
