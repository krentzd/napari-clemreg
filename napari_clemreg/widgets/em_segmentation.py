#!/usr/bin/env python3
# coding: utf-8
from magicgui import magic_factory

from .sparse_unet.model import SparseUnet

from napari.layers import Labels, Image

@magic_factory
def train_model(source: Image,
                target: Labels): # -> Image:

    # train_generator = SparseDataGenerator(source, target, crop_shape=(512,512))
    # print('Len', len(train_generator))

    model = SparseUnet(shape=(512,512,1))

    model.train(source,
                target,
                out_dir='output_dir',
                epochs=10,
                batch_size=16)

    model.predict(source.data, tile_shape=(512,512))

    # return Image(cropped_images)
