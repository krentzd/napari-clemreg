#!/usr/bin/env python3
# coding: utf-8
import json
import os.path
import napari
import numpy as np
import pint
from magicgui import magic_factory, widgets
from napari.layers import Image, Shapes, Labels, Points
from napari.utils.notifications import show_error

from pathlib import Path
from ..clemreg.empanada_segmentation import empanada_segmentation
from ..clemreg.point_cloud_registration import point_cloud_registration
from ..clemreg.point_cloud_sampling import point_cloud_sampling
from ..clemreg.warp_image_volume import warp_image_volume
from ..clemreg.data_preprocessing import make_isotropic, _make_isotropic
from napari.qt.threading import thread_worker
from napari.layers.utils._link_layers import link_layers

"""
Moving segmentation
"""
def run_moving_segmentation(Moving_Image,
                            Mask_ROI,
                            z_min,
                            z_max,
                            log_sigma,
                            log_threshold,
                            filter_segmentation,
                            filter_size_lower,
                            filter_size_upper,
):

    from ..clemreg.log_segmentation import log_segmentation, filter_binary_segmentation
    from ..clemreg.mask_roi import mask_roi

    seg_volume = log_segmentation(input=Moving_Image,
                                  sigma=log_sigma,
                                  threshold=log_threshold)

    if filter_segmentation:
        seg_volume = filter_binary_segmentation(input=seg_volume,
                                                percentile=(filter_size_lower, filter_size_upper))

    if len(set(seg_volume.data.ravel())) <= 1:
        return 'No segmentation'

    if Mask_ROI is not None:
        # Convert Mask_ROI to new space
        z_min = int(z_min * (pxlsz_moving[0] / pxlsz_fixed[0]))
        z_max = min(int(z_max * (pxlsz_moving[0] / pxlsz_fixed[0])), seg_volume.data.shape[0])

        m_z = np.expand_dims(Mask_ROI.data[0][:,0] * (pxlsz_moving[0] / pxlsz_fixed[0]), axis=1)
        m_xy = Mask_ROI.data[0][:,1:] * (pxlsz_moving[1] / pxlsz_fixed[0])

        Mask_ROI.data = np.hstack((m_z, m_xy))

        seg_volume_mask = mask_roi(input=seg_volume,
                                   crop_mask=Mask_ROI,
                                   z_min=z_min,
                                   z_max=z_max)
    else:
        seg_volume_mask = seg_volume

    return seg_volume_mask

"""
Fixed segmentation
"""

def run_fixed_segmentation(Fixed_Image,
                           em_seg_axis

):
    from ..clemreg.empanada_segmentation import empanada_segmentation

    seg_volume = empanada_segmentation(input=Fixed_Image.data,
                                       axis_prediction=em_seg_axis)

    if len(set(seg_volume.ravel())) <= 1:
        return 'No segmentation'

    return seg_volume

"""
Point cloud sampling
"""
def run_point_cloud_sampling(Moving_Segmentation,
                             Fixed_Segmentation,
                             moving_image_pixelsize_xy,
                             moving_image_pixelsize_z,
                             fixed_image_pixelsize_xy,
                             fixed_image_pixelsize_z,
                             point_cloud_sampling_frequency,
                             voxel_size,
                             point_cloud_sigma
):
    import pint
    from ..clemreg.data_preprocessing import _make_isotropic
    from ..clemreg.point_cloud_sampling import point_cloud_sampling

    ureg = pint.UnitRegistry()

    pxlsz_moving = (moving_image_pixelsize_z.to_preferred([ureg.nanometers]).magnitude, moving_image_pixelsize_xy.to_preferred([ureg.nanometers]).magnitude)
    pxlsz_fixed = (fixed_image_pixelsize_z.to_preferred([ureg.nanometers]).magnitude, fixed_image_pixelsize_xy.to_preferred([ureg.nanometers]).magnitude)

    moving_seg = _make_isotropic(Moving_Segmentation.data > 0,
                                 pxlsz_lm=pxlsz_moving,
                                 pxlsz_em=pxlsz_fixed)
    # Check for EM or LM volume
    # Need to check for isotropic EM volume
    if pxlsz_fixed[0] != pxlsz_fixed[1]:
        fixed_seg = _make_isotropic(Fixed_Segmentation.data,
                                    pxlsz_lm=pxlsz_moving,
                                    pxlsz_em=pxlsz_fixed,
                                    ref_frame='EM')
    else:
        fixed_seg = Fixed_Segmentation.data

    moving_seg_kwargs = dict(
        name=Moving_Segmentation.name,
    )
    fixed_seg_kwargs = dict(
        name=Fixed_Segmentation.name,
    )

    point_freq = point_cloud_sampling_frequency / 100
    moving_point_cloud = point_cloud_sampling(input=Labels(moving_seg, **moving_seg_kwargs),
                                              every_k_points=1 // point_freq,
                                              voxel_size=voxel_size,
                                              sigma=point_cloud_sigma)

    fixed_point_cloud = point_cloud_sampling(input=Labels(fixed_seg, **fixed_seg_kwargs),
                                             every_k_points=1 // point_freq,
                                             voxel_size=voxel_size,
                                             sigma=point_cloud_sigma)

    moving_points_kwargs = dict(
        name='Moving_point_cloud',
        face_color='red',
        edge_color='black',
        size=5,
        metadata={'pxlsz': pxlsz_moving}
    )

    fixed_points_kwargs = dict(
        name='Fixed_point_cloud',
        face_color='blue',
        edge_color='black',
        size=5,
        metadata={'pxlsz': pxlsz_fixed, 'output_shape': fixed_seg.shape}
    )

    return Points(moving_point_cloud, **moving_points_kwargs), Points(fixed_point_cloud, **fixed_points_kwargs)

"""
Point cloud registration
"""
def point_cloud_registration():
    moving, fixed, transformed, kwargs = point_cloud_registration(moving_input_points.data, fixed_input_points.data,
                                                                  algorithm=registration_algorithm,
                                                                  max_iterations=registration_max_iterations)

"""
Image warping
"""

def image_warping(
    moving_points,
    fixed_points,
    output_shape
):
    import numpy as np
    from ..clemreg.data_preprocessing import make_isotropic
    from ..clemreg.log_segmentation import log_segmentation, filter_binary_segmentation
    from ..clemreg.mask_roi import mask_roi, mask_area

    warp_outputs = warp_image_volume(moving_image=moving_input_image,
                                     output_shape=output_shape,
                                     transform_type=registration_algorithm,
                                     moving_points=Points(moving),
                                     transformed_points=transformed,
                                     interpolation_order=warping_interpolation_order,
                                     approximate_grid=warping_approximate_grid,
                                     sub_division_factor=warping_sub_division_factor)

    if pxlsz_fixed[0] != pxlsz_fixed[1]:
        if not isinstance(warp_outputs, list):
            warp_outputs = [warp_outputs]
        warp_outputs_temp = []
        for warp_output in warp_outputs:
            warp_outputs_temp.append((_make_isotropic(warp_output[0],
                                                      (pxlsz_fixed[0], pxlsz_fixed[0]),
                                                      pxlsz_fixed,
                                                      inverse=True,
                                                      ref_frame='EM'), warp_output[1]))
        warp_outputs = warp_outputs_temp

    return warp_outputs

"""
Helper functions
"""

def _add_data(return_value):
    # pbar.hide()

    # if return_value == 'No segmentation':
    #     show_error('WARNING: No mitochondria in Fixed Image or Moving Image')
    #     return
    #
    # if isinstance(return_value, list):
    #     layers = []
    #     for image_data in return_value:
    #         data, kwargs = image_data
    #         viewer.add_image(data, **kwargs)
    #         layers.append(viewer.layers[kwargs['name']])
    #     link_layers(layers)
    # else:
    data, layer_type, kwargs = return_value

    if layer_type == 'image':
        viewer.add_image(data, **kwargs)

    elif layer_type == 'labels':
        viewer.add_labels(data, **kwargs)


    # def _add_data(return_value):
    #     if isinstance(return_value, str):
    #         show_error('WARNING: No mitochondria in Fixed Image')
    #         return
    #
    #     viewer.add_labels(return_value.data.astype(np.int64),
    #                       name="Moving_Segmentation")

def _yield_segmentation(yield_value):

    image = yield_value[0]
    image_type = yield_value[1]

    if image_type == 'lm':
        viewer.add_labels(np.asarray(image.data, dtype=np.uint32), name=image_type)
    else:
        viewer.add_labels(image, name=image_type)

def _yield_point_clouds(yield_value):
    points, kwargs = yield_value[0], yield_value[1]
    viewer.add_points(points.data, **kwargs)

def _create_json_file(path_to_json):
    dictionary = {
        "registration_algorithm": registration_algorithm,
        "em_seg_axis": em_seg_axis,
        "log_sigma": log_sigma,
        "log_threshold": log_threshold,
        "custom_z_zoom": custom_z_zoom,
        "z_zoom_value": z_zoom_value,
        "filter_segmentation": filter_segmentation,
        "filter_size": filter_size,
        "point_cloud_sampling_frequency": point_cloud_sampling_frequency,
        "point_cloud_sigma": point_cloud_sigma,
        "registration_voxel_size": registration_voxel_size,
        "registration_max_iterations": registration_max_iterations,
        "warping_interpolation_order": warping_interpolation_order,
        "warping_approximate_grid": warping_approximate_grid,
        "warping_sub_division_factor": warping_sub_division_factor
    }

    json_object = json.dumps(dictionary, indent=4)

    if path_to_json == '':
        path_to_json = 'parameters.json'

    with open(path_to_json, "w") as outfile:
        outfile.write(json_object)



    if Moving_Image is None or Fixed_Image is None:
        show_error("WARNING: You have not inputted both a fixed and moving image")
        return

    if len(Moving_Image.data.shape) != 3:
        show_error("WARNING: Your moving_image must be 3D, you're current input has a shape of {}".format(
            Moving_Image.data.shape))
        return
    elif len(Moving_Image.data.shape) == 3 and (Moving_Image.data.shape[2] == 3 or Fixed_Image.data.shape[2] == 4):
        show_error("WARNING: YOUR moving_image is RGB, your input must be grayscale and 3D")
        return

    if len(Fixed_Image.data.shape) != 3:
        show_error("WARNING: Your Fixed_Image must be 3D, you're current input has a shape of {}".format(
            Moving_Image.data.shape))
        return
    elif len(Fixed_Image.data.shape) == 3 and (Fixed_Image.data.shape[2] == 3 or Fixed_Image.data.shape[2] == 4):
        show_error("WARNING: YOUR fixed_image is RGB, your input must be grayscale and 3D")
        return

    if Mask_ROI is not None:
        if len(Mask_ROI.data) != 1:
            show_error("WARNING: You must input only 1 Mask ROI, you have inputted {}.".format(len(Mask_ROI.data)))
            return
        if mask_area(Mask_ROI.data[0][:, 1], Mask_ROI.data[0][:, 2]) > Moving_Image.data.shape[1] * \
                Moving_Image.data.shape[2]:
            show_error("WARNING: Your mask size exceeds the size of the image.")
            return

    if save_json and not params_from_json:
        _create_json_file(path_to_json=save_json_path)
