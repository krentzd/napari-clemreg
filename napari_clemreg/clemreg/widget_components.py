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
                            filter_size_upper
):

    from ..clemreg.log_segmentation import log_segmentation, filter_binary_segmentation
    from ..clemreg.mask_roi import mask_roi

    seg_volume = log_segmentation(input=Moving_Image,
                                  sigma=log_sigma,
                                  threshold=log_threshold)

    if filter_segmentation:
        seg_volume = filter_binary_segmentation(input=seg_volume,
                                                percentile=(filter_size_lower, filter_size_upper))

    if len(set(seg_volume.ravel())) <= 1:
        return 'No segmentation'

    if Mask_ROI is not None:
        seg_volume = mask_roi(input_arr=seg_volume,
                              crop_mask=Mask_ROI,
                              z_min=z_min,
                              z_max=z_max)

    return seg_volume

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
def run_point_cloud_registration_and_warping(Moving_Points,
                                             Fixed_Points,
                                             Moving_Image,
                                             Fixed_Image,
                                             registration_algorithm,
                                             registration_max_iterations,
                                             warping_interpolation_order,
                                             warping_approximate_grid,
                                             warping_sub_division_factor,
                                             registration_direction,
                                             benchmarking_mode: bool=False,
                                             **reg_kwargs
):
    from ..clemreg.point_cloud_registration import point_cloud_registration
    from ..clemreg.data_preprocessing import return_isotropic_image_list, _make_isotropic
    from ..clemreg.warp_image_volume import warp_image_volume_from_list

    if registration_direction == u'EM \u2192 FM':
        Fixed_Points, Moving_Points = Moving_Points, Fixed_Points
        Fixed_Image, Moving_Image = Moving_Image, Fixed_Image

    point_cloud_reg_return_vals = point_cloud_registration(moving=Moving_Points.data,
                                                           fixed=Fixed_Points.data,
                                                           algorithm=registration_algorithm,
                                                           max_iterations=registration_max_iterations,
                                                           benchmarking_mode=benchmarking_mode,
                                                           **reg_kwargs)
    if benchmarking_mode:
        moving, fixed, transformed, kwargs, elapsed = point_cloud_reg_return_vals
    else:
        moving, fixed, transformed, kwargs = point_cloud_reg_return_vals

    if registration_algorithm == 'Affine CPD' or registration_algorithm == 'Rigid CPD':
        transformed = Points(moving, **kwargs)
    # Make images isotropic for linked layers
    moving_image_list = return_isotropic_image_list(input_image=Moving_Image,
                                                    pxlsz_lm=Moving_Points.metadata['pxlsz'],
                                                    pxlsz_em=Fixed_Points.metadata['pxlsz'])
    print('Returned isotropic images')
    warp_outputs = warp_image_volume_from_list(moving_image_list=moving_image_list,
                                               output_shape=Fixed_Points.metadata['output_shape'],
                                               transform_type=registration_algorithm,
                                               moving_points=Points(moving),
                                               transformed_points=transformed,
                                               interpolation_order=warping_interpolation_order,
                                               approximate_grid=warping_approximate_grid,
                                               sub_division_factor=warping_sub_division_factor)
    print('Finished warping images')
    if Fixed_Points.metadata['pxlsz'][0] != Fixed_Points.metadata['pxlsz'][1]:
        src_pxlsz = (Fixed_Points.metadata['pxlsz'][0], Fixed_Points.metadata['pxlsz'][0])
        for warp_output in warp_outputs:
            warp_output.data = _make_isotropic(warp_output.data,
                                               src_pxlsz,
                                               Fixed_Points.metadata['pxlsz'],
                                               inverse=True,
                                               ref_frame='EM')
    if benchmarking_mode:
        return warp_outputs, transformed, elapsed
    else:
        return warp_outputs, transformed

"""
Helper functions
"""
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

def load_from_json():
    if params_from_json and load_json_file.is_file():
        f = open(str(load_json_file))

        data = json.load(f)
        try:
            registration_algorithm = data["registration_algorithm"]
            em_seg_axis = data["em_seg_axis"]
            log_sigma = data["log_sigma"]
            log_threshold = data["log_threshold"]
            custom_z_zoom = data["custom_z_zoom"],
            z_zoom_value = ["z_zoom_value"],
            filter_segmentation = ["filter_segmentation"],
            filter_size = ["filter_size"],
            point_cloud_sampling_frequency = data["point_cloud_sampling_frequency"]
            point_cloud_sigma = data["point_cloud_sigma"]
            registration_voxel_size = data["registration_voxel_size"]
            registration_max_iterations = data["registration_max_iterations"]
            warping_interpolation_order = data["warping_interpolation_order"]
            warping_approximate_grid = data["warping_approximate_grid"]
            warping_sub_division_factor = data["warping_sub_division_factor"]
        except KeyError:
            show_error("JSON file missing required param")
            return
    elif params_from_json and not load_json_file.is_file():
        show_error("Load from JSON selected but no JSON file selected or file path isn't real")
        return
