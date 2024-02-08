#!/usr/bin/env python3
# coding: utf-8
# Adapted from: https://github.com/zpincus/celltool/blob/master/celltool/numerics/image_warp.py
import math
from napari.layers import Points, Image
from napari.types import PointsData, ImageData
from scipy import ndimage
from napari.layers.utils._link_layers import get_linked_layers
from skimage import exposure
import numpy as np

def _make_inverse_warp(from_points, to_points, output_region, approximate_grid):
    """
    ?

    Parameters
    ----------
    from_points : ?
        ?
    to_points : ?
        ?
    output_region : ?
        ?
    approximate_grid : ?
        ?

    Returns
    -------
    ?

    """
    x_min, y_min, z_min, x_max, y_max, z_max = output_region
    if approximate_grid is None: approximate_grid = 1
    x_steps = (x_max - x_min) // approximate_grid
    y_steps = (y_max - y_min) // approximate_grid
    z_steps = (z_max - z_min) // approximate_grid

    x, y, z = np.mgrid[x_min:x_max:x_steps * 1j, y_min:y_max:y_steps * 1j, z_min:z_max:z_steps * 1j]
    transform = _make_warp(to_points, from_points, x, y, z)

    if approximate_grid != 1:
        # linearly interpolate the zoomed transform grid
        new_x, new_y, new_z = np.mgrid[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
        x_fracs, x_indices = np.modf((x_steps - 1) * (new_x - x_min) / float(x_max - x_min))
        y_fracs, y_indices = np.modf((y_steps - 1) * (new_y - y_min) / float(y_max - y_min))
        z_fracs, z_indices = np.modf((z_steps - 1) * (new_z - z_min) / float(z_max - z_min))

        x_indices = x_indices.astype(int)
        y_indices = y_indices.astype(int)
        z_indices = z_indices.astype(int)

        x1 = 1 - x_fracs
        y1 = 1 - y_fracs
        z1 = 1 - z_fracs

        ix1 = (x_indices + 1).clip(0, x_steps - 1)
        iy1 = (y_indices + 1).clip(0, y_steps - 1)
        iz1 = (z_indices + 1).clip(0, z_steps - 1)

        transform_x = _trilinear_interpolation(0, transform, x1, y1, z1, x_fracs, y_fracs, z_fracs, x_indices,
                                               y_indices, z_indices, ix1, iy1, iz1)
        transform_y = _trilinear_interpolation(1, transform, x1, y1, z1, x_fracs, y_fracs, z_fracs, x_indices,
                                               y_indices, z_indices, ix1, iy1, iz1)
        transform_z = _trilinear_interpolation(2, transform, x1, y1, z1, x_fracs, y_fracs, z_fracs, x_indices,
                                               y_indices, z_indices, ix1, iy1, iz1)

        transform = [transform_x, transform_y, transform_z]
    return transform


def _trilinear_interpolation(d, t, x0, y0, z0, x1, y1, z1, ix0, iy0, iz0, ix1, iy1, iz1):
    """
    ?

    Parameters
    ----------
    d : ?
        ?
    t : ?
        ?
    x0 : ?
        ?
    y0 : ?
        ?
    z0 : ?
        ?
    x1 : ?
        ?
    y1 : ?
        ?
    z1 : ?
        ?
    ix0 : ?
        ?
    iy0 : ?
        ?
    iz0 : ?
        ?
    ix1 : ?
        ?
    iy1 : ?
        ?
    iz1 : ?
        ?
    Returns
    -------
    ?
    """
    t000 = t[d][(ix0, iy0, iz0)]
    t001 = t[d][(ix0, iy0, iz1)]
    t010 = t[d][(ix0, iy1, iz0)]
    t100 = t[d][(ix1, iy0, iz0)]
    t011 = t[d][(ix0, iy1, iz1)]
    t101 = t[d][(ix1, iy0, iz1)]
    t110 = t[d][(ix1, iy1, iz0)]
    t111 = t[d][(ix1, iy1, iz1)]

    return t000 * x0 * y0 * z0 + t001 * x0 * y0 * z1 + t010 * x0 * y1 * z0 + t100 * x1 * y0 * z0 + t011 * x0 * y1 * z1 + t101 * x1 * y0 * z1 + t110 * x1 * y1 * z0 + t111 * x1 * y1 * z1


def _U(x):
    """
    ?

    Parameters
    ----------
    x : ?
        ?
    Returns
    -------
        ?
    """
    _small = 1e-100
    return (x ** 2) * np.where(x < _small, 0, np.log(x))


def _interpoint_distances(points):
    """
    ?

    Parameters
    ----------
    points : Points
        ?
    Returns
    -------
        ?
    """
    xd = np.subtract.outer(points[:, 0], points[:, 0])
    yd = np.subtract.outer(points[:, 1], points[:, 1])
    zd = np.subtract.outer(points[:, 2], points[:, 2])
    return np.sqrt(xd ** 2 + yd ** 2 + zd ** 2)


def _make_L_matrix(points):
    """
    ?

    Parameters
    ----------
    points : ?
        ?
    Returns
    -------
    ?
    """
    n = len(points)
    K = _U(_interpoint_distances(points))
    P = np.ones((n, 4))
    P[:, 1:] = points
    O = np.zeros((4, 4))
    L = np.block([[K, P], [P.transpose(), O]])
    return L


def _calculate_f(coeffs, points, x, y, z):
    """
    ?

    Parameters
    ----------
    coeffs : ?
        ?
    points : ?
        ?
    x : ?
        ?
    y : ?
        ?
    z : ?
        ?
    Returns
    -------
    ?
    """
    w = coeffs[:-3]
    a1, ax, ay, az = coeffs[-4:]
    summation = np.zeros(x.shape)
    for wi, Pi in zip(w, points):
        summation += wi * _U(np.sqrt((x - Pi[0]) ** 2 + (y - Pi[1]) ** 2 + (z - Pi[2]) ** 2))
    return a1 + ax * x + ay * y + az * z + summation


def _make_warp(from_points, to_points, x_vals, y_vals, z_vals):
    """
    ?

    Parameters
    ----------
    from_points : ?
        ?
    to_points : ?
        ?
    x_vals : ?
        ?
    y_vals : ?
        ?
    z_vals : ?
        ?

    Returns
    -------
        ?
    """
    from_points, to_points = np.asarray(from_points), np.asarray(to_points)
    err = np.seterr(divide='ignore')
    L = _make_L_matrix(from_points)
    V = np.resize(to_points, (len(to_points) + 4, 3))
    V[-3:, :] = 0
    # TODO: benchmark speed of numpy and scipy implementations of pinv
    # TODO: if piecewise non-linear transform only compute pseudoinverse once!
    L_pseudo_inverse = np.linalg.pinv(L)  # L increases with number of control points!
    coeffs = np.dot(L_pseudo_inverse, V)
    x_warp = _calculate_f(coeffs[:, 0], from_points, x_vals, y_vals, z_vals)
    y_warp = _calculate_f(coeffs[:, 1], from_points, x_vals, y_vals, z_vals)
    z_warp = _calculate_f(coeffs[:, 2], from_points, x_vals, y_vals, z_vals)
    np.seterr(**err)
    return [x_warp, y_warp, z_warp]


def _warp_images(from_points, to_points, image, output_region, interpolation_order=5, approximate_grid=10):
    """
    ?

    Parameters
    ----------
    from_points : ?
        ?
    to_points : ?
        ?
    image : Image
        ?
    output_region : ?
        ?
    interpolation_order : ?
        ?
    approximate_grid : ?
        ?

    Returns
    -------
    ?
    """
    transform = _make_inverse_warp(from_points, to_points, output_region, approximate_grid)
    return ndimage.map_coordinates(np.asarray(image), transform, order=interpolation_order)


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


def _warp_image_volume(moving_image: Image,
                       output_shape: tuple,
                       moving_points: PointsData,
                       transformed_points: PointsData,
                       interpolation_order: int = 1,
                       approximate_grid: int = 1,
                       sub_division_factor: int = 1):
    """
    ?

    Parameters
    ----------
    moving_image : Image
        ?
    fixed_image : ImageData
        ?
    moving_points : PointsData
        ?
    transformed_points : PointsData
        ?
    interpolation_order : int
        ?
    approximate_grid : int
        ?
    sub_division_factor : ?
        ?

    Returns
    -------
    ?

    """
    assert len(moving_points) == len(transformed_points), 'Moving and transformed points must be of same length.'

    x_chunk = math.ceil(output_shape[1] / sub_division_factor)
    y_chunk = math.ceil(output_shape[2] / sub_division_factor)
    z_chunk = math.ceil(output_shape[0] / sub_division_factor)

    if len(moving_image.data.shape) == 1 + len(output_shape):
        warped_images = []
        kwargs_list = []

        for c in range(moving_image.data.shape[0]):
            warped_image = np.empty(output_shape)

            for x in range(math.ceil(output_shape[1] / x_chunk)):
                for y in range(math.ceil(output_shape[2] / y_chunk)):
                    for z in range(math.ceil(output_shape[0] / z_chunk)):
                        output_region = (z * z_chunk,
                                         x * x_chunk,
                                         y * y_chunk,
                                         min((z * z_chunk + z_chunk), output_shape[0]),
                                         min((x * x_chunk + x_chunk), output_shape[1]),
                                         min((y * y_chunk + y_chunk), output_shape[2]))

                        z_min, x_min, y_min, z_max, x_max, y_max = output_region

                        warped_region = _warp_images(from_points=moving_points,
                                                     to_points=transformed_points,
                                                     image=moving_image.data[c],
                                                     output_region=output_region,
                                                     interpolation_order=interpolation_order,
                                                     approximate_grid=approximate_grid)

                        # Warping function returns images padded by one in each dimension
                        warped_image[z_min:z_max, x_min:x_max, y_min:y_max] = warped_region[:-1, :-1, :-1]

            warped_images.append(warped_image)
            kwargs_list.append(dict(name=moving_image.name + '_warped_ch_' + str(c)))

        return (np.squeeze(np.stack(warped_images)), kwargs_list)

    else:
        warped_image = np.empty(output_shape)

        for x in range(math.ceil(output_shape[1] / x_chunk)):
            for y in range(math.ceil(output_shape[2] / y_chunk)):
                for z in range(math.ceil(output_shape[0] / z_chunk)):
                    output_region = (z * z_chunk,
                                     x * x_chunk,
                                     y * y_chunk,
                                     min((z * z_chunk + z_chunk), output_shape[0]),
                                     min((x * x_chunk + x_chunk), output_shape[1]),
                                     min((y * y_chunk + y_chunk), output_shape[2]))

                    z_min, x_min, y_min, z_max, x_max, y_max = output_region

                    warped_region = _warp_images(from_points=moving_points,
                                                 to_points=transformed_points,
                                                 image=moving_image.data,
                                                 output_region=output_region,
                                                 interpolation_order=interpolation_order,
                                                 approximate_grid=approximate_grid)

                    # Warping function returns images padded by one in each dimension
                    warped_image[z_min:z_max, x_min:x_max, y_min:y_max] = warped_region[:-1, :-1, :-1]

                # Return list of Images and add all at once
        kwargs = dict(
            name=moving_image.name + '_warped'
        )
        return warped_image, kwargs

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
