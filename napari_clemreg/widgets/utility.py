#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from magicgui import magic_factory, widgets
from scipy import ndimage
from napari.layers import Image, Shapes
import time
import math
from skimage import draw
from typing_extensions import Annotated

def on_init(widget):
    """Initializes widget layout and updates widget layout according to user input."""

    def change_z_max(input_image: Image):
        if len(input_image.data.shape) == 3:
            widget.z_max.max = input_image.data.shape[0]
            widget.z_max.value = input_image.data.shape[0]
        elif len(input_image.data.shape) == 4:
            widget.z_max.max = input_image.data.shape[1]
            widget.z_max.value = input_image.data.shape[1]

    def change_z_min(z_max_val: int):
        widget.z_min.max = z_max_val

    def change_z_max_from_z_min(z_min_val: int):
        widget.z_max.min = z_min_val

    widget.z_max.changed.connect(change_z_min)
    widget.input.changed.connect(change_z_max)
    widget.z_min.changed.connect(change_z_max_from_z_min)

@magic_factory(widget_init=on_init, layout='vertical', call_button="Mask")
def mask_roi(input: Image,
             crop_mask: Shapes,
             z_min: Annotated[int, {"min": 0, "max": 10, "step": 1}]=0,
             z_max: Annotated[int, {"min": 10, "max": 100, "step": 1}]=10) -> Image: #Annotated[slice, {"start": 0, "stop": 10, "step": 1}]
    """
    Take crop_mask and mask input
    """

    # Need to add support for multi-channel images
    # Squeeze array
    # If 4 dimensions, assume dimension with smallest size is colour channel
    # Reshape stack to be [channel, z, x, y]

    print(input.data.shape)

    if crop_mask.data[0].shape[-1] > 3:
        crop_mask.data = [mask[:,-3:] for mask in crop_mask.data]

    assert len(crop_mask.data) == 1, 'Crop mask must contain one shape'

    top_idx = crop_mask.shape_type.index('polygon') if 'polygon' in crop_mask.shape_type else crop_mask.shape_type.index('rectangle')
    top_z = z_min

    bot_z = z_max

    input_arr = input.data #np.squeeze(input.data)
    print(input_arr.shape)

    temp_idx = 2 if len(input_arr.shape) == 4 else 1

    print(input_arr.shape[temp_idx:])

    binary_mask = draw.polygon2mask(input_arr.shape[temp_idx:], crop_mask.data[top_idx][:,1:])
    top_vol = [np.zeros(input_arr.shape[temp_idx:])] * top_z
    binary_mask_vol = [binary_mask] * (bot_z - top_z)
    bot_vol = [np.zeros(input_arr.shape[temp_idx:])] * (input_arr.shape[temp_idx - 1] - bot_z)

    binary_mask_full_vol = np.stack([top_vol + binary_mask_vol + bot_vol])
    if len(input_arr.shape) == 4:
        binary_mask_full_vol = np.stack([binary_mask_full_vol] * input_arr.shape[0])
        binary_mask_full_vol = np.squeeze(binary_mask_full_vol)
    else:
        binary_mask_full_vol = np.squeeze(binary_mask_full_vol)

    assert binary_mask_full_vol.shape == input_arr.shape, "Mask and image volume don't match!"

    masked_input = input_arr.data * binary_mask_full_vol

    if not masked_input.shape == input.data.shape:
        print('Reshaped array')
        masked_input.reshape(input.data.shape)

    return Image(masked_input,
                 name=input.name + '_masked')
