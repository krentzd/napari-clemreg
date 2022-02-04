#!/usr/bin/env python3
# coding: utf-8
from magicgui import magic_factory
import os
from pathlib import Path
from napari.layers import Labels, Image
from napari.qt import thread_worker

from .sparse_unet.model import SparseUnet

@magic_factory
def train_model(viewer: "napari.viewer.Viewer",
                source: Image,
                target: Labels,
                output_directory: Path,
                dense_annotations: bool=False,
                epochs: int=50,
                batch_size: int=16): # -> Image:

    _train_model(source, target, output_directory, epochs, batch_size, dense_annotations)

    @thread_worker
    def _train_model(source, target, output_directory, epochs, batch_size, dense_annotations):
        model = SparseUnet(shape=(512,512,1))
        model.train(source,
                    target,
                    out_dir=output_directory,
                    epochs=epochs,
                    batch_size=batch_size,
                    dense=dense_annotations)

@magic_factory
def predict_from_model(input: Image,
                       model_path: Path):
    pass

    # model.predict(source.data, tile_shape=(512,512))

    # return Image(cropped_images)
