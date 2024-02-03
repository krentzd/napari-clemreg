#!/usr/bin/env python3
# coding: utf-8
import time
import numpy as np
from napari.layers import Image, Shapes, Labels
from skimage import draw
from typing_extensions import Annotated
from ..clemreg.data_preprocessing import _make_isotropic

def mask_area(x, y):
    """ Calculates the area of the mask

    Parameters
    ----------
    x : int
        The x dimension of the mask
    y : int
        The y dimension of the mask

    Returns
    -------
    area : float
        The area of the mask based on its x and y dimensions
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def mask_roi(input_arr: np.ndarray,
             crop_mask: Shapes,
             z_min: Annotated[int, {"min": 0, "max": 10, "step": 1}] = 0,
             z_max: Annotated[int, {"min": 10, "max": 100,
                                    "step": 1}] = 10) -> Image:  # Annotated[slice, {"start": 0, "stop": 10, "step": 1}]
    """ Take crop_mask and mask input

    Parameters
    ----------
    input : napari.layers.Image
        Image to apply crop to
    crop_mask : napari.layers.Shapes
        Mask to be used to crop the image
    z_min : int
        Minimum z slice to apply masking to
    z_max : int
        Maximum z slice to apply masking to

    Returns
    -------
    masked_input : napari.layers.Image
        Masked image of the original inputted image
    """

    # Need to add support for multi-channel images
    # Squeeze array
    # If 4 dimensions, assume dimension with smallest size is colour channel
    # Reshape stack to be [channel, z, x, y]

    print(f'Masking with {crop_mask.name} between {z_min} and {z_max}...')
    start_time = time.time()

    if crop_mask.data[0].shape[-1] > 3:
        crop_mask.data = [mask[:, -3:] for mask in crop_mask.data]

    assert len(crop_mask.data) == 1, 'Crop mask must contain one shape'

    top_idx = crop_mask.shape_type.index(
        'polygon') if 'polygon' in crop_mask.shape_type else crop_mask.shape_type.index('rectangle')
    top_z = z_min
    bot_z = z_max

    temp_idx = 2 if len(input_arr.shape) == 4 else 1

    binary_mask = draw.polygon2mask(input_arr.shape[temp_idx:], crop_mask.data[top_idx][:, 1:])
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
    masked_input = masked_input.astype(int)

    if not masked_input.shape == input_arr.shape:
        print('Reshaped array')
        masked_input.reshape(input_arr.shape)

    print(f'Finished masking after {time.time() - start_time}s!')

    return masked_input
