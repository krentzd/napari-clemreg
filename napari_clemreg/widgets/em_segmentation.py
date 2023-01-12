#!/usr/bin/env python3
# coding: utf-8
from pathlib import Path
from magicgui import magic_factory
from napari.layers import Labels, Image
from typing_extensions import Annotated


# TODO: Change run button to abort button for stoping thread_worker
@magic_factory
def train_model(viewer: "napari.viewer.Viewer",
                source: Image,
                target: Labels,
                output_directory: Annotated[Path, {"mode": "d"}],
                dense_annotations: bool = False,
                epochs: int = 50,
                batch_size: int = 8):
    """ Trains a sparse u-net to do segmentation of the fixed EM image

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        Napari viewer allows addition of layer once thread_worker finished
        executing
    source : napari.layers.Image
        Source image to train the sparse u-net to segment
    target : napari.layers.Labels
        Target labels to train the sparse u-net
    output_directory : str
        Directory to save the trained sparse u-net model
    dense_annotations : napari.layers.Labels
        ?
    epochs : int
        Number of epochs to train the model
    batch_size : int
        Number of samples that will be propagated through the network.
    Returns
    -------
        Trained u-net model
    """
    from .sparse_unet.model import SparseUnet

    # from napari.qt import thread_worker
    #
    # pbar = widgets.ProgressBar()
    # pbar.range = (0, 0)  # unknown duration
    # train_model.insert(0, pbar)
    model = SparseUnet(shape=(512, 512, 1))
    # @thread_worker
    # def _train_model(model, source, target, output_directory, epochs, batch_size, dense_annotations):
    model.train(source,
                target,
                out_dir=str(output_directory),
                epochs=epochs,
                batch_size=batch_size,
                dense=dense_annotations)
    # _train_model(model,
    #              source,
    #              target,
    #              output_directory,
    #              epochs, batch_size,
    #              dense_annotations)


@magic_factory
def predict_from_model(input: Image,
                       model_path: Path) -> Image:
    """ Use pretrained model to make predictions from input image.

    Parameters
    ----------
    input : napari.layers.Image
        Image to apply model prediction to.
    model_path : Path
        Path to the model.
    Returns
    -------
    napari.layers.Image
        Returns image containing the segmentations produced by the pretrained model.
    """
    import numpy as np
    from .sparse_unet.model import SparseUnet

    model = SparseUnet(shape=(512, 512, 1))
    model.load(str(model_path))
    segmented_image = []  # [model.predict(im, tile_shape=(512,512)) for im in input.data]
    for i, im in enumerate(input.data):
        print(i)
        segmented_image.append(model.predict(im, tile_shape=(512, 512)))

    return Image(np.stack(segmented_image))
