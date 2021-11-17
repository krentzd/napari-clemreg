#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from magicgui import magic_factory, widgets
from scipy import ndimage
from napari.layers import Image, Shapes
import time
import math
from skimage import draw

@magic_factory
def mask_roi(input: Image,
             crop_mask: Shapes) -> Image:
    """
    Take crop_mask and mask input
    """

    if crop_mask.data[0].shape[-1] > 3:
        crop_mask.data = [mask[:,-3:] for mask in crop_mask.data]

    assert len(crop_mask.data) <= 2, 'Crop mask must contain no more than two shapes'
    assert 'polygon' or 'rectangle' in crop_mask.shape_type, 'Crop mask must contain one polygon or rectangle'

    top_idx = crop_mask.shape_type.index('polygon') if 'polygon' in crop_mask.shape_type else crop_mask.shape_type.index('rectangle')
    top_z = int(crop_mask.data[top_idx][0][0])

    if len(crop_mask.data) == 2:
        bot_idx = crop_mask.shape_type.index('ellipse')
        bot_z = int(crop_mask.data[bot_idx][0][0])
        assert top_z < bot_z, 'Ellipse shape must mark last z value'
    else:
        bot_z = input.data.shape[0]
        assert top_z < bot_z, 'Shape must be contained within image'

    binary_mask = draw.polygon2mask(input.data.shape[1:], crop_mask.data[top_idx][:,1:])
    top_vol = [np.zeros(input.data.shape[1:])] * top_z
    binary_mask_vol = [binary_mask] * (bot_z - top_z)
    bot_vol = [np.zeros(input.data.shape[1:])] * (input.data.shape[0] - bot_z)

    binary_mask_full_vol = np.stack([top_vol + binary_mask_vol + bot_vol])

    masked_input = np.squeeze(input.data * binary_mask_full_vol)

    return Image(masked_input,
                 name=input.name + '_masked')
