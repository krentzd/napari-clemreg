#!/usr/bin/env python3
# coding: utf-8
import time
from dataclasses import dataclass

import numpy as np
from skimage import draw

from ._arrays import as_array


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


@dataclass(frozen=True)
class MaskRoi:
    """A polygon ROI applied over a range of z slices.

    polygon : (N, 2) array of (y, x) vertices. Wider arrays, such as
        napari's (z, y, x) shape vertices, are accepted and only the last
        two columns are used.
    z_min, z_max : the z slices to keep, as ``z_min <= z < z_max``.
    """
    polygon: np.ndarray
    z_min: int = 0
    z_max: int = 10


def _polygon_from(crop_mask):
    """(N, 2) (y, x) vertices from an array, or from a Shapes-like object."""
    if isinstance(crop_mask, np.ndarray):
        return crop_mask[:, -2:]
    # Shapes-like (backwards compatibility): one shape, with
    # vertices in `.data`.
    assert len(crop_mask.data) == 1, 'Crop mask must contain one shape'
    return np.asarray(crop_mask.data[0])[:, -2:]


def mask_roi(input_arr: np.ndarray,
             crop_mask,
             z_min: int = 0,
             z_max: int = 10) -> np.ndarray:
    """ Zero everything outside a polygon, and outside a range of z slices.

    Parameters
    ----------
    input_arr : np.ndarray
        Volume to mask, (z, y, x) or (c, z, y, x). An object with the
        volume in `.data` is also accepted (backwards compatibility).
    crop_mask : np.ndarray or napari Shapes-like
        The polygon's vertices, (N, 2) as (y, x); only the last two
        columns of wider arrays are used. For backwards compatibility, an
        object with a single shape's vertices in `.data` (e.g. a napari
        Shapes layer) is also accepted.
    z_min : int
        First z slice to keep.
    z_max : int
        Keep z slices below this one.

    Returns
    -------
    np.ndarray
        The masked volume, as integers.
    """
    print(f'Masking with {getattr(crop_mask, "name", "polygon")} between {z_min} and {z_max}...')
    start_time = time.time()

    polygon = _polygon_from(crop_mask)
    volume = as_array(input_arr)

    top_z = z_min
    bot_z = z_max

    temp_idx = 2 if len(volume.shape) == 4 else 1

    binary_mask = draw.polygon2mask(volume.shape[temp_idx:], polygon)

    top_vol = [np.zeros(volume.shape[temp_idx:])] * top_z
    binary_mask_vol = [binary_mask] * (bot_z - top_z)
    bot_vol = [np.zeros(volume.shape[temp_idx:])] * (volume.shape[temp_idx - 1] - bot_z)

    binary_mask_full_vol = np.stack([top_vol + binary_mask_vol + bot_vol])

    if len(volume.shape) == 4:
        binary_mask_full_vol = np.stack([binary_mask_full_vol] * volume.shape[0])
        binary_mask_full_vol = np.squeeze(binary_mask_full_vol)
    else:
        binary_mask_full_vol = np.squeeze(binary_mask_full_vol)

    assert binary_mask_full_vol.shape == volume.shape, "Mask and image volume don't match!"

    masked_input = volume * binary_mask_full_vol
    masked_input = masked_input.astype(int)

    print(f'Finished masking after {time.time() - start_time}s!')

    return masked_input
