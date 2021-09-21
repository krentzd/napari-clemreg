#!/usr/bin/env python3
# coding: utf-8
from magicgui import magic_factory
from napari.types import LabelsData, ImageData
import numpy as np

@magic_factory
def change_layer_type(input: ImageData) -> LabelsData:
    return input
    
# Widget to load unet model
# Choose layer which was used for sparse segmentation
# Train button
# Run predictions on chosen layer from unet model
