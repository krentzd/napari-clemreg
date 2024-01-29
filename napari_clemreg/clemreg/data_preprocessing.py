#!/usr/bin/env python3
# coding: utf-8
from napari.layers import Image
from napari.layers.utils._link_layers import get_linked_layers
from scipy import ndimage
import numpy as np
import time
import copy

def get_pixelsize(metadata: dict):
    """ Parse pixel sizes from image metadata

    Parameters
    ----------
    metadata : dict
        Metadata of user inputted image
    Returns
    -------
        Pixel size
    """

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

    return (eval(x_pxlsz) if isinstance(x_pxlsz, str) else x_pxlsz,
            eval(y_pxlsz) if isinstance(y_pxlsz, str) else y_pxlsz,
            eval(z_pxlsz) if isinstance(z_pxlsz, str) else z_pxlsz,
            unit)


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

    image.data = ndimage.zoom(image.data, (z_zoom, 1, 1))

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
    return ndimage.zoom(im_arr, zoom_vals, order=order)

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
        for image in images:
            print(f'Resampling {image.name}')
            target_image_iso = copy.deepcopy(image)
            image_iso = _make_isotropic(image.data,
                                        pxlsz_lm,
                                        pxlsz_em,
                                        inverse=kwargs.get('inverse', False),
                                        ref_frame=kwargs.get('ref_frame', 'LM'),
                                        order=kwargs.get('order', 0))

            target_image_iso.data = image_iso
            image_iso_list.append(target_image_iso)
    else:
        target_image_iso = copy.deepcopy(input_image)
        image_iso = _make_isotropic(input_image.data,
                                    pxlsz_lm,
                                    pxlsz_em,
                                    inverse=kwargs.get('inverse', False),
                                    ref_frame=kwargs.get('ref_frame', 'LM'),
                                    order=kwargs.get('order', 0))
        target_image_iso.data = image_iso
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
            image.data = _make_isotropic(image.data,
                                               pxlsz_lm,
                                               pxlsz_em,
                                               inverse=kwargs.get('inverse', False),
                                               ref_frame=kwargs.get('ref_frame', 'LM'),
                                               order=kwargs.get('order', 0))
    else:
        input_image.data = _make_isotropic(input_image.data,
                                           pxlsz_lm,
                                           pxlsz_em,
                                           inverse=kwargs.get('inverse', False),
                                           ref_frame=kwargs.get('ref_frame', 'LM'),
                                           order=kwargs.get('order', 0))
    # return resampled_image
