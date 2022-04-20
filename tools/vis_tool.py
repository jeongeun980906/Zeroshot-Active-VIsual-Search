from IPython.display import HTML, display
import sys
import imageio
from moviepy.editor import ImageSequenceClip
from typing import Sequence
import numpy as np
import os
from typing import Optional
import ai2thor.server
from typing import Union
from PIL import Image

import matplotlib.pyplot as plt

def plot_frames(event: Union[ai2thor.server.Event, np.ndarray],gridmap : np.ndarray) -> None:
    """Visualize all the frames on an AI2-THOR Event.
    Example:
    plot_frames(controller.last_event)
    """
    if isinstance(event, ai2thor.server.Event):
        third_person_frames = event.third_party_camera_frames
        RGB = event.frame
        DEPTH = event.depth_frame

        rows = 2 
        cols = 2
        fig, axs = plt.subplots(
            nrows=rows, ncols=cols, dpi=150, figsize=(3 * cols, 3 * rows)
        )

        ax = axs[0][0]
        im = ax.imshow(RGB)
        ax.axis("off")
        ax.set_title('RGB')

        ax = axs[0][1]
        im = ax.imshow(DEPTH)
        ax.axis("off")
        ax.set_title('DEPTH')
        fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)

        # add third party camera frames
        ax = axs[1][0]
        ax.set_title("Map View")
        ax.imshow(third_person_frames[0])
        ax.axis("off")

        ax = axs[1][1]
        ax.set_title("Grid Map")
        ax.imshow(gridmap,cmap=plt.cm.gray_r)
        ax.axis("off")
        plt.show()


def show_path(path,gridmap):
    gridmap_rgb = np.expand_dims(gridmap)
    gridmap_rgb = np.repeat(gridmap_rgb,3,0)
    