#!/usr/bin/env python3
# coding: utf-8
from magicgui import magic_factory, widgets
import os
import numpy as np
from pathlib import Path
from napari.layers import Labels, Image
from typing_extensions import Annotated

from .sparse_unet.model import SparseUnet

# TODO: Change run button to abort button for stoping thread_worker
@magic_factory
def train_model(viewer: "napari.viewer.Viewer",
                source: Image,
                target: Labels,
                output_directory: Annotated[Path, {"mode": "d"}],
                dense_annotations: bool=False,
                epochs: int=50,
                batch_size: int=8):

    # from napari.qt import thread_worker
    #
    # pbar = widgets.ProgressBar()
    # pbar.range = (0, 0)  # unknown duration
    # train_model.insert(0, pbar)
    model = SparseUnet(shape=(512,512,1))
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
    model = SparseUnet(shape=(512,512,1))
    model.load(str(model_path))
    segmented_image = [] #[model.predict(im, tile_shape=(512,512)) for im in input.data]
    for i, im in enumerate(input.data):
        print(i)
        segmented_image.append(model.predict(im, tile_shape=(512,512)))

    return Image(np.stack(segmented_image))
