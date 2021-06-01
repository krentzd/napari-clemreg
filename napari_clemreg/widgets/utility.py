#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from magicgui import magic_factory, widgets
from scipy import ndimage
from napari.layers import Image, Shapes
from napari.qt import thread_worker
import time
import math

@magic_factory
def crop_volume(input: Image,
                crop_area: Shapes,
                z_max: int) -> Image:
    "Cropping in xy determined by shape with lowest z and z determined by largest z"
    # TODO: Ensure cropping is within shape of input
    z_min = math.floor(crop_area.data[0][0][0])
    # z_max = math.floor(crop_area.data[-1][0][0])
    x_min = math.floor(crop_area.data[0][0][1])
    x_max = math.floor(crop_area.data[0][2][1])
    y_min = math.floor(crop_area.data[0][0][2])
    y_max = math.floor(crop_area.data[0][1][2])
    print(z_min, z_max, x_min, x_max, y_min, y_max)
    return Image(input.data[z_min:z_max, x_min:x_max, y_min:y_max],
                 name=input.name + '_crop')
