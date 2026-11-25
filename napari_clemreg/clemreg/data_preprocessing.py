#!/usr/bin/env python3
# coding: utf-8
from napari.layers import Image
from ._arrays import as_array
from ._napari_compat import get_linked_layers
from scipy import ndimage
import numpy as np
import time

def _to_float(value, default=1):
    """A pixel size from metadata as a float.

    Metadata values are file content, so they're parsed as numbers only
    (a plain decimal such as '0.13', or a fraction such as '1/3'), never
    evaluated. Anything else falls back to `default`, as missing
    metadata does.
    """
    from fractions import Fraction

    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return float(Fraction(value.strip()))
    except (ValueError, ZeroDivisionError):
        print(f'Could not parse pixel size {value!r} from metadata; using {default}')
        return default


def get_pixelsize(metadata: dict):
    """ Parse pixel sizes from image metadata

    Checks for bioio-derived pixel sizes first (see _reader.py, which
    puts a `physical_pixel_sizes` key -- a PhysicalPixelSizes(Z, Y, X)
    in microns, possibly with None components if the file's format/
    metadata doesn't have them -- in every freshly-loaded layer's
    metadata). Falls back to the older, ImageJ-specific TIFF-tag
    parsing below for metadata dicts that predate it, e.g.
    sample_data.py's hand-populated dicts (kept as-is until real
    Zenodo sample data with correct embedded metadata replaces it --
    confirmed directly that one of the two current sample files has no
    pixel size metadata embedded at all, so there's nothing for either
    approach to extract there).

    Parameters
    ----------
    metadata : dict
        Metadata of user inputted image
    Returns
    -------
        Pixel size
    """
    pixel_sizes = metadata.get('physical_pixel_sizes')
    if pixel_sizes is not None and None not in (pixel_sizes.X, pixel_sizes.Y, pixel_sizes.Z):
        return (pixel_sizes.X, pixel_sizes.Y, pixel_sizes.Z, 'micron')

    try:
        x_pxlsz = 1 / metadata['XResolution']
        y_pxlsz = 1 / metadata['YResolution']
    # If no metadata, set pixelsize to 1: no effect on output image
    except KeyError:
        x_pxlsz = 1
        y_pxlsz = 1
        print('XResolution and YResolution not recorded in metadata')

    try:
        # Parse ImageJ Metadata to get z pixelsize
        ij_metadata = metadata['ImageDescription'].split('\n')
        ij_metadata = [i for i in ij_metadata if i not in '=']
        ij_dict = dict((k, v) for k, v in (i.rsplit('=') for i in ij_metadata))

        z_pxlsz = ij_dict['spacing']
        unit = ij_dict['unit']
    except (KeyError, ValueError) as error:
        z_pxlsz = 1
        unit = 'micron'
        print('ImageJ metdata not recorded in metadata')

    return (_to_float(x_pxlsz), _to_float(y_pxlsz), _to_float(z_pxlsz), unit)


def _rescale(im_arr: np.ndarray, factors: tuple, order: int = 0):
    """ Resample an array by exact per-axis scale factors.

    Output voxel `o` samples input position `o / factor`, so input index
    0 stays at output index 0 and the scale is exactly `factor`: the
    convention _rescale_affine_matrix() and napari's layer `scale` use.
    ndimage.zoom() can't be used for this. It aligns the first and last
    voxels, so its real scale is (n_out - 1) / (n_in - 1), about 3% off
    in z for a 28-slice FM stack (#98). Its grid_mode=True aligns voxel
    edges instead, which gets the ratio right but shifts everything by
    (factor - 1) / 2 voxels. The output shape matches zoom()'s.

    Parameters
    ----------
    im_arr : np.ndarray
        Input array
    factors : tuple
        Output voxels per input voxel, one per axis
    order : int
        Order of resampling
    Returns
    -------
        Resampled image array
    """
    factors = np.asarray(factors, dtype=float)
    output_shape = tuple(int(round(n * f)) for n, f in zip(im_arr.shape, factors))
    return ndimage.affine_transform(im_arr, 1 / factors, output_shape=output_shape,
                                    order=order, mode='nearest')


def _zoom_values(xy, z, xy_ref, z_ref):
    """
    ?
    Parameters
    ----------
    xy : int
        ?
    z : int
        ?
    xy_ref : int
        ?
    z_ref : int
        ?
    Returns
    -------
    ?
    """
    xy_zoom = xy / xy_ref
    z_zoom = z / z_ref

    return xy_zoom, z_zoom


def _make_isotropic_v1(image: Image,
                       z_zoom_value: float):
    """
    ?

    Parameters
    ----------
    image : Image
        ?
    Returns
    -------
    ?
    """

    # Inplace operation
    if z_zoom_value == None:
        moving_xy_pixelsize, __, moving_z_pixelsize, __ = get_pixelsize(image.metadata)
        z_zoom = moving_z_pixelsize / moving_xy_pixelsize
    else:
        z_zoom = z_zoom_value

    print(f'Interpolating {image.name} with zoom_value={z_zoom}...')
    start_time = time.time()

    image.data = _rescale(image.data, (z_zoom, 1, 1), order=3)

    print(f'Finished interpolating after {time.time() - start_time}s!')

    return z_zoom

def _make_isotropic(im_arr: np.ndarray, pxlsz_lm: tuple, pxlsz_em: tuple, inverse: bool=False, ref_frame: str='LM', order=0):
    """ Return isotropic images based on pixelsizes

    Parameters
    ----------
    im_arr : np.ndarray
        Input image array
    pxlsz_lm : tuple
        LM image pixelsizes
    pxlsz_em : tuple
        EM image pixelsizes
    inverse : bool
        True returns isotropic resampling and False returns inverse
    ref_frame : str
        Denotes reference frame
    order : int
        Denotes order of resampling
    Returns
    -------
        Resampled image array
    """

    assert ref_frame in ['LM', 'EM'], 'Allowed ref_frame: EM or LM'
    z_lm, xy_lm = pxlsz_lm
    z_em, xy_em = pxlsz_em

    if ref_frame == 'LM':
        zoom_vals = (z_lm / z_em, xy_lm / z_em, xy_lm / z_em)
    elif ref_frame == 'EM':
        zoom_vals = (1, xy_em / z_em, xy_em / z_em)

    if inverse:
        zoom_vals = tuple(1 / x for x in zoom_vals)
    return _rescale(im_arr, zoom_vals, order=order)

def resample_to_pixelsize(im_arr: np.ndarray, working_pxlsz: float, target_pxlsz: tuple, order: int = 0):
    """ Resample a dense array from a uniform working voxel size onto an
    arbitrary (possibly anisotropic) target pixel size.

    Parameters
    ----------
    im_arr : np.ndarray
        Input array, isotropic at `working_pxlsz` in all three axes (e.g.
        a warp_image_volume_from_list() output, which is always isotropic
        at the EM z-pixel-size).
    working_pxlsz : float
        The (uniform) voxel size of `im_arr`.
    target_pxlsz : tuple
        Desired (z, xy) voxel size of the output.
    order : int
        Order of resampling.
    Returns
    -------
        Resampled image array
    """
    zoom_vals = (working_pxlsz / target_pxlsz[0],
                working_pxlsz / target_pxlsz[1],
                working_pxlsz / target_pxlsz[1])
    return _rescale(im_arr, zoom_vals, order=order)

def return_isotropic_image_list(input_image: Image,
                                pxlsz_lm: tuple,
                                pxlsz_em: tuple,
                                **kwargs):
    """ Returns list of isotropic images based on pixelsizes for all linked layers

    Parameters
    ----------
    im_arr : np.ndarray
        Input image array
    pxlsz_lm : tuple
        LM image pixelsizes
    pxlsz_em : tuple
        EM image pixelsizes
    """
    # Inplace operation
    image_iso_list = []
    # print(f'Resampling {input_image.name}')
    if len(get_linked_layers(input_image)) > 0:
        images = get_linked_layers(input_image)
        images.add(input_image)
    else:
        images = [input_image]

    for image in images:
        print(f'Resampling {image.name}')
        image_iso = _make_isotropic(as_array(image),
                                    pxlsz_lm,
                                    pxlsz_em,
                                    inverse=kwargs.get('inverse', False),
                                    ref_frame=kwargs.get('ref_frame', 'LM'),
                                    order=kwargs.get('order', 0))

        # A plain copy.deepcopy(image) here (as this used to do) crashes
        # on a real, viewer-attached layer -- confirmed directly: it
        # carries live Qt/vispy canvas state (down to a QFont, several
        # frames deep through overlay callbacks) that can't be pickled.
        # Only .data (replaced below anyway), .name, .colormap and
        # .blending are ever read from these downstream (confirmed by
        # checking every call site) -- so build a fresh, un-attached
        # Image layer with just those instead of deep-copying the whole
        # live object.
        target_image_iso = Image(image_iso,
                                 name=image.name,
                                 colormap=image.colormap,
                                 blending=image.blending)
        image_iso_list.append(target_image_iso)

    return image_iso_list

def make_isotropic(input_image: Image,
                   pxlsz_lm: tuple,
                   pxlsz_em: tuple,
                   **kwargs):
    """ Inplace change of isotropic images based on pixelsizes for all linked layers

    Parameters
    ----------
    im_arr : np.ndarray
        Input image array
    pxlsz_lm : tuple
        LM image pixelsizes
    pxlsz_em : tuple
        EM image pixelsizes
    """
    # Inplace operation
    print(f'Resampling {input_image.name}')
    if len(get_linked_layers(input_image)) > 0:
        images = get_linked_layers(input_image)
        images.add(input_image)
        for image in images:
            image.data = _make_isotropic(as_array(image),
                                               pxlsz_lm,
                                               pxlsz_em,
                                               inverse=kwargs.get('inverse', False),
                                               ref_frame=kwargs.get('ref_frame', 'LM'),
                                               order=kwargs.get('order', 0))
    else:
        input_image.data = _make_isotropic(as_array(input_image),
                                           pxlsz_lm,
                                           pxlsz_em,
                                           inverse=kwargs.get('inverse', False),
                                           ref_frame=kwargs.get('ref_frame', 'LM'),
                                           order=kwargs.get('order', 0))
    # return resampled_image
