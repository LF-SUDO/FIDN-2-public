"""JPG plots adapted from FIDN-2/FIDN/evaluate.py by Bo Pang.

Copyright 2022 IRP Project. Apache-2.0. Display arrays are training-normalized
copies. Predictions retain their original 512 by 512 probability raster.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

CHANNEL_TITLES = (
    "burned area day 0", "burned area day 1", "burned area day 2",
    "biomass above ground", "biomass under ground", "land slope",
    "tree density", "grass density", "bare density", "snow", "water",
    "wind direction u", "wind direction v", "rain + snow", "final burned area",
)


def feature_figure(normalized_event):
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    for channel, ax in enumerate(axes.flat):
        image = normalized_event[..., channel]
        if 2 < channel < 14:
            image = image[192:320, 192:320]
        artist = ax.imshow(image, cmap="viridis")
        divider = make_axes_locatable(ax)
        color_axis = divider.append_axes("right", size="7%", pad="2%")
        fig.colorbar(artist, cax=color_axis, orientation="vertical")
        ax.set_title(CHANNEL_TITLES[channel])
    fig.suptitle("Environmental channels use training min/max normalization; near-constant channels retain source values",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def prediction_figure(normalized_event, probability):
    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    images = (normalized_event[..., 2], normalized_event[..., 14], probability)
    for ax, image, title in zip(axes, images, ("Burned area day 2", "Final burned area", "Predict")):
        ax.imshow(np.squeeze(image), cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    return fig


def save_event_jpgs(normalized_event, probability, output, event):
    for folder in ("predictions", "features"):
        (output / folder).mkdir(exist_ok=True)
    for path, builder, arguments in (
        (output / "predictions" / f"{event}.jpg", prediction_figure, (normalized_event, probability)),
        (output / "features" / f"{event}_info.jpg", feature_figure, (normalized_event,)),
    ):
        figure = builder(*arguments)
        try:
            with path.open("xb") as stream:
                figure.savefig(stream, format="jpg", dpi=100)
        finally:
            plt.close(figure)
