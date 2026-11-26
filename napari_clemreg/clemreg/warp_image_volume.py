#!/usr/bin/env python3
# coding: utf-8
import itertools
import math
from napari.layers import Points, Image
from napari.types import PointsData, ImageData
from scipy import ndimage
from ._napari_compat import get_linked_layers
import numpy as np


def _fit_tps(from_points, to_points):
    """A thin plate spline mapping `from_points` exactly onto `to_points`.

    Call the result with an (n, 3) array of positions. A point listed more
    than once is used once, with its targets averaged; repeated points
    would otherwise make the spline wrong without any warning.
    """
    from scipy.interpolate import RBFInterpolator

    from_points, index = np.unique(np.asarray(from_points, dtype=float), axis=0, return_inverse=True)
    index = index.ravel()
    to_points = np.asarray(to_points, dtype=float)
    targets = np.zeros((len(from_points), to_points.shape[1]))
    np.add.at(targets, index, to_points)
    targets /= np.bincount(index)[:, np.newaxis]
    return RBFInterpolator(from_points, targets, kernel='thin_plate_spline', degree=1)


def _coarse_axis(size, approximate_grid):
    """Evenly spaced positions from the first to the last voxel along an
    axis, at most `approximate_grid` voxels apart."""
    steps = math.ceil((size - 1) / approximate_grid) + 1
    return np.linspace(0, size - 1, steps)


def _coarse_warp(tps, output_shape, approximate_grid):
    """The spline evaluated on a grid over the whole output, as an array
    of shape (3, *grid shape): where each grid position samples the
    moving image."""
    axes = [_coarse_axis(size, approximate_grid) for size in output_shape]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    warp = tps(grid.reshape(-1, 3)).reshape(grid.shape)
    return np.moveaxis(warp, -1, 0)


def _sampling_coordinates(coarse_warp, output_shape, region):
    """Where each voxel of `region` (a tuple of slices) samples the moving
    image: the coarse warp, linearly interpolated."""
    grid_index = []
    for size, steps, axis_slice in zip(output_shape, coarse_warp.shape[1:], region):
        to_grid = (steps - 1) / (size - 1) if size > 1 else 0
        grid_index.append(np.arange(axis_slice.start, axis_slice.stop) * to_grid)
    grid_index = np.meshgrid(*grid_index, indexing='ij')
    return np.stack([ndimage.map_coordinates(axis_warp, grid_index, order=1, mode='nearest')
                     for axis_warp in coarse_warp])


def _regions(output_shape, sub_division_factor):
    """Split the output into `sub_division_factor` chunks along each axis."""
    chunks = [math.ceil(size / sub_division_factor) for size in output_shape]
    starts = [range(0, size, chunk) for size, chunk in zip(output_shape, chunks)]
    for start in itertools.product(*starts):
        yield tuple(slice(s, min(s + chunk, size)) for s, chunk, size in zip(start, chunks, output_shape))


def _warp_image_volume_affine(image,
                              matrix,
                              output_shape,
                              interpolation_order):
    """ Wraps scipy.ndimage.affine_transform()

    Parameters
    ----------
    image : Image
        ?
    matrix : ?
        ?
    output_shape : ?
        ?
    interpolation_order : ?
        ?
    Returns
    -------
        ?
    """
    # image = exposure.rescale_intensity(image, out_range='uint8')
    inv_mat = np.linalg.inv(matrix)
    img_wrp = ndimage.affine_transform(input=image,
                                       matrix=inv_mat,
                                       output_shape=output_shape,
                                       order=interpolation_order)

    return img_wrp


def _rescale_affine_matrix(matrix, raw_pxlsz, working_pxlsz, target_pxlsz):
    """ Recompose a registration matrix so it maps directly from a target
    output grid to the raw (un-resampled) moving image's own native grid.

    `matrix` (moving -> fixed) is computed by point_cloud_registration()
    in the internal isotropic-at-EM-z-pixel-size "working grid" shared by
    both point clouds. Naively applying it means resampling the moving
    image up onto that working grid first, warping there, then resampling
    the result back down to whatever output resolution is actually
    wanted -- two extra dense resamples for no benefit. Folding both
    resamples' scale factors directly into the matrix lets
    _warp_image_volume_affine() warp straight from the raw moving image
    to the target grid in a single interpolation pass.

    Parameters
    ----------
    matrix : np.ndarray
        4x4 homogeneous registration matrix, moving->fixed, in the
        working grid's voxel-index coordinates.
    raw_pxlsz : tuple
        (z, xy) voxel size of the raw (un-resampled) moving image.
    working_pxlsz : float
        Voxel size of the working grid `matrix` was computed in (uniform
        across all three axes).
    target_pxlsz : tuple
        (z, xy) voxel size of the desired output grid.
    Returns
    -------
        4x4 homogeneous matrix mapping target-grid output indices
        directly to raw-moving-image indices.
    """
    scale_out_inv = np.diag([working_pxlsz / target_pxlsz[0],
                             working_pxlsz / target_pxlsz[1],
                             working_pxlsz / target_pxlsz[1],
                             1.0])
    scale_in_inv = np.diag([raw_pxlsz[0] / working_pxlsz,
                            raw_pxlsz[1] / working_pxlsz,
                            raw_pxlsz[1] / working_pxlsz,
                            1.0])
    return scale_out_inv @ matrix @ scale_in_inv


def _warp_image_volume(moving_image: Image,
                       output_shape: tuple,
                       moving_points: PointsData,
                       transformed_points: PointsData,
                       interpolation_order: int = 1,
                       approximate_grid: int = 1,
                       sub_division_factor: int = 1):
    """Warp `moving_image` onto an `output_shape` grid with a thin plate
    spline (the non-rigid BCPD result).

    The spline maps `transformed_points` (on the output grid) onto
    `moving_points` (in the moving image), so each output voxel knows
    where to sample the moving image. It's evaluated `approximate_grid`
    voxels apart and interpolated in between. The output is filled in
    `sub_division_factor`³ chunks to limit memory.

    Returns (warped image, layer kwargs); for a multichannel image
    (channels first), the kwargs are a list with one dict per channel.
    """
    assert len(moving_points) == len(transformed_points), 'Moving and transformed points must be of same length.'

    data = np.asarray(moving_image.data)
    multichannel = data.ndim == len(output_shape) + 1
    channels = data if multichannel else data[np.newaxis]

    coarse_warp = _coarse_warp(_fit_tps(transformed_points, moving_points), output_shape, approximate_grid)
    warped = np.empty((len(channels), *output_shape))
    for region in _regions(output_shape, sub_division_factor):
        coordinates = _sampling_coordinates(coarse_warp, output_shape, region)
        for warped_channel, channel in zip(warped, channels):
            warped_channel[region] = ndimage.map_coordinates(channel, coordinates, order=interpolation_order)

    if multichannel:
        return np.squeeze(warped), [dict(name=f'{moving_image.name}_warped_ch_{c}') for c in range(len(channels))]
    return warped[0], dict(name=moving_image.name + '_warped')


def warp_image_volume_from_list(
        moving_image_list: list,
        output_shape: tuple,
        transform_type: str,
        moving_points: Points,
        transformed_points: Points,
        interpolation_order: int=1,
        approximate_grid: int=1,
        sub_division_factor: int=1
):
    if transform_type == 'BCPD':
        warping_args = {'output_shape': output_shape,
                        'moving_points': moving_points.data,
                        'transformed_points': transformed_points.data,
                        'interpolation_order': interpolation_order,
                        'approximate_grid': approximate_grid,
                        'sub_division_factor': sub_division_factor}

    elif transform_type == 'Affine CPD' or transform_type == 'Rigid CPD':
        affine_matrix = transformed_points.affine.affine_matrix
        warping_args = {'matrix': affine_matrix,
                        'output_shape': output_shape,
                        'interpolation_order': interpolation_order}

    img_wrp_list = []
    for image in moving_image_list:
        print(f'Warping {image.name} with {transform_type}...')
        if transform_type == 'Affine CPD' or transform_type == 'Rigid CPD':
            img_wrp = _warp_image_volume_affine(image=image.data, **warping_args)
        elif transform_type == 'BCPD':
            img_wrp, __ = _warp_image_volume(moving_image=image, **warping_args)

        kwargs = dict(
            name=image.name + '_warped',
            colormap=image.colormap,
            blending=image.blending
        )
        img_wrp_list.append(Image(img_wrp, **kwargs))

    return img_wrp_list

def warp_image_volume(
        moving_image: Image,
        output_shape: tuple,
        transform_type: str,
        moving_points: Points,
        transformed_points: Points,
        interpolation_order: int = 1,
        approximate_grid: int = 1,
        sub_division_factor: int = 1
):
    """
    ?

    Parameters
    ----------
    moving_image : ?
        ?
    fixed_image : ?
        ?
    transform_type : ?
        ?
    moving_points : ?
        ?
    transformed_points : ?
        ?
    interpolation_order : ?
        ?
    approximate_grid : ?
        ?
    sub_division_factor : ?
        ?

    Returns
    -------
        ?
    """

    if transform_type == 'BCPD':
        warping_args = {'output_shape': output_shape,
                        'moving_points': moving_points.data,
                        'transformed_points': transformed_points.data,
                        'interpolation_order': interpolation_order,
                        'approximate_grid': approximate_grid,
                        'sub_division_factor': sub_division_factor}

    elif transform_type == 'Affine CPD' or transform_type == 'Rigid CPD':
        affine_matrix = transformed_points.affine.affine_matrix
        warping_args = {'matrix': affine_matrix,
                        'output_shape': output_shape,
                        'interpolation_order': interpolation_order}

    # Get linked layers
    if len(get_linked_layers(moving_image)) > 0:
        img_wrp_list = []
        images = get_linked_layers(moving_image)
        images.add(moving_image)
        # Include actual image in set --> only one layer added to viewer
        for image in images:
            if transform_type == 'Affine CPD' or transform_type == 'Rigid CPD':
                img_wrp = _warp_image_volume_affine(image=image.data, **warping_args)
            elif transform_type == 'BCPD':
                img_wrp, __ = _warp_image_volume(moving_image=image, **warping_args)

            kwargs = dict(
                name=image.name + '_warped',
                colormap=image.colormap,
                blending=image.blending
            )
            img_wrp_list.append((img_wrp, kwargs))
        return img_wrp_list

    else:
        if transform_type == 'Affine CPD' or transform_type == 'Rigid CPD':
            img_wrp = _warp_image_volume_affine(image=moving_image.data, **warping_args)
        elif transform_type == 'BCPD':
            img_wrp, __ = _warp_image_volume(moving_image=moving_image, **warping_args)

        kwargs = dict(
            name=moving_image.name + '_warped'
        )
        return img_wrp, kwargs

def warp_image_volume_deprecated(
        moving_image: Image,
        fixed_image: ImageData,
        transform_type: str,
        moving_points: Points,
        transformed_points: Points,
        interpolation_order: int = 1,
        approximate_grid: int = 1,
        sub_division_factor: int = 1
):
    """
    ?

    Parameters
    ----------
    moving_image : ?
        ?
    fixed_image : ?
        ?
    transform_type : ?
        ?
    moving_points : ?
        ?
    transformed_points : ?
        ?
    interpolation_order : ?
        ?
    approximate_grid : ?
        ?
    sub_division_factor : ?
        ?

    Returns
    -------
        ?
    """

    if transform_type == 'BCPD':
        return _warp_image_volume(moving_image=moving_image,
                                  fixed_image=fixed_image.data,
                                  moving_points=moving_points.data,
                                  transformed_points=transformed_points.data,
                                  interpolation_order=interpolation_order,
                                  approximate_grid=approximate_grid,
                                  sub_division_factor=sub_division_factor)

    elif transform_type == 'Affine CPD' or transform_type == 'Rigid CPD':
        affine_matrix = transformed_points.affine.affine_matrix

        # Get linked layers
        if len(get_linked_layers(moving_image)) > 0:
            img_wrp_list = []
            images = get_linked_layers(moving_image)
            images.add(moving_image)
            # Include actual image in set --> only one layer added to viewer
            for image in images:
                img_wrp = _warp_image_volume_affine(image=image.data,
                                                    matrix=affine_matrix,
                                                    output_shape=fixed_image.data.shape,
                                                    interpolation_order=interpolation_order)
                kwargs = dict(
                    name=image.name + '_warped',
                    colormap=image.colormap,
                    blending=image.blending
                )
                img_wrp_list.append((img_wrp, kwargs))
            return img_wrp_list

        else:
            img_wrp = _warp_image_volume_affine(image=moving_image.data,
                                                matrix=affine_matrix,
                                                output_shape=fixed_image.data.shape,
                                                interpolation_order=interpolation_order)
            kwargs = dict(
                name=moving_image.name + '_warped'
            )
            return img_wrp, kwargs
