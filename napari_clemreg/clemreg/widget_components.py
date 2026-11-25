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
from ..clemreg.point_cloud_registration import point_cloud_registration
from ..clemreg.point_cloud_sampling import point_cloud_sampling
from ..clemreg.warp_image_volume import warp_image_volume
from ..clemreg._arrays import as_array
from ..clemreg.data_preprocessing import make_isotropic, _make_isotropic
from napari.qt.threading import thread_worker

"""
Moving segmentation
"""
# The EM Segmentation Backend dropdown's labels (on_init_specs.py), mapped
# to the core's backend names.
EM_BACKENDS = {
    'MitoNet (Segment-Flow)': 'segment-flow',
    'MitoNet (empanada-dl)': 'empanada',
}


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
    """Adapter: FM segmentation from napari layers.

    Raises NoSegmentationError (from the core) if nothing is segmented.
    """
    from ..clemreg.log_segmentation import segment_fm
    from ..clemreg.mask_roi import MaskRoi

    roi = None
    if Mask_ROI is not None:
        assert len(Mask_ROI.data) == 1, 'Crop mask must contain one shape'
        roi = MaskRoi(polygon=np.asarray(Mask_ROI.data[0]), z_min=z_min, z_max=z_max)
    size_filter = (filter_size_lower, filter_size_upper) if filter_segmentation else None

    return segment_fm(as_array(Moving_Image),
                      sigma=log_sigma,
                      threshold=log_threshold,
                      size_filter=size_filter,
                      roi=roi)

"""
Fixed segmentation
"""
def em_backend_ready(em_segmentation_backend):
    """Whether the chosen EM Segmentation Backend can run; if not, show an
    error saying what to do. Cheap (it doesn't import the backend), so
    widgets call it on the GUI thread before starting a worker."""
    from ..clemreg.em_segmentation import check_backend_installed
    from ..clemreg.exceptions import BackendNotInstalledError

    try:
        check_backend_installed(EM_BACKENDS[em_segmentation_backend])
    except BackendNotInstalledError as e:
        show_error(
            f"{em_segmentation_backend} needs the optional `empanada` extra "
            f"(missing: {' and '.join(e.missing)}). That extra can't currently be installed "
            "on Python 3.11 or newer (https://github.com/martlj/napari-clemreg/issues/5), "
            "so choose MitoNet (Segment-Flow) as the EM Segmentation Backend instead."
        )
        return False
    return True


def run_fixed_segmentation(Fixed_Image,
                           em_segmentation_backend='MitoNet (Segment-Flow)'
):
    """Adapter: EM segmentation from a napari layer.

    Raises NoSegmentationError (from the core) if nothing is segmented.
    """
    from napari.utils.notifications import show_info
    from ..clemreg.em_segmentation import segment_em

    backend = EM_BACKENDS[em_segmentation_backend]
    if backend == 'segment-flow':
        # Segment-Flow shells out to Nextflow and can run for minutes with
        # no other GUI feedback (it prints progress to the terminal, but
        # that's easy to miss if you're only watching the napari window) --
        # show_info() is thread-safe to call from this background worker.
        show_info(
            'Running EM segmentation via MitoNet (Segment-Flow)... this '
            'can take a while, especially on first run. Progress is printed '
            'to the terminal.'
        )
    seg_volume = segment_em(as_array(Fixed_Image), backend=backend)
    if backend == 'segment-flow':
        show_info('EM segmentation via Segment-Flow finished.')
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

    moving_seg = _make_isotropic(as_array(Moving_Segmentation) > 0,
                                 pxlsz_lm=pxlsz_moving,
                                 pxlsz_em=pxlsz_fixed)
    # Need to check for isotropic EM volume
    if pxlsz_fixed[0] != pxlsz_fixed[1]:
        fixed_seg = _make_isotropic(as_array(Fixed_Segmentation),
                                    pxlsz_lm=pxlsz_moving,
                                    pxlsz_em=pxlsz_fixed,
                                    ref_frame='EM')
    else:
        fixed_seg = as_array(Fixed_Segmentation)

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
        # napari renamed Points' edge_color constructor arg to
        # border_color (confirmed directly: napari 0.9.1's Points.__init__
        # has no edge_color parameter at all, migration shim included --
        # this is a real, previously-unexercised API break, not a typo).
        border_color='black',
        size=5,
        # Both clouds record their working grid's shape: it's the warp's
        # output shape for whichever is fixed (EM → FM makes the FM fixed).
        metadata={'pxlsz': pxlsz_moving, 'output_shape': moving_seg.shape}
    )

    fixed_points_kwargs = dict(
        name='Fixed_point_cloud',
        face_color='blue',
        border_color='black',
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
                                             warping_output_resolution='Native LM resolution',
                                             benchmarking_mode: bool=False,
                                             **reg_kwargs
):
    from ..clemreg.point_cloud_registration import point_cloud_registration
    from ..clemreg.data_preprocessing import return_isotropic_image_list, resample_to_pixelsize
    from ..clemreg.warp_image_volume import (
        warp_image_volume_from_list, _warp_image_volume_affine, _rescale_affine_matrix,
    )
    from ..clemreg._napari_compat import get_linked_layers

    # Before any swap, Moving is the FM and Fixed the EM. Both point clouds
    # are in the same working grid, isotropic at the EM z-pixel-size (see
    # run_point_cloud_sampling), whichever direction is registered.
    fm_pxlsz, em_pxlsz = Moving_Points.metadata['pxlsz'], Fixed_Points.metadata['pxlsz']
    working_pxlsz = em_pxlsz[0]
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
    else:
        transformed = Points(transformed)
    mobie = _print_mobie_affine(transformed, registration_algorithm, working_pxlsz,
                                Moving_Points, Fixed_Points, Moving_Image, Fixed_Image)
    # The output's pixel size follows the option's name in both directions.
    target_pxlsz = em_pxlsz if warping_output_resolution == 'EM pixel grid (legacy)' else fm_pxlsz

    # Relative to the fixed image's pixels, so the output overlays it.
    scale = (target_pxlsz[0] / Fixed_Points.metadata['pxlsz'][0],
            target_pxlsz[1] / Fixed_Points.metadata['pxlsz'][1],
            target_pxlsz[1] / Fixed_Points.metadata['pxlsz'][1])

    if (warping_output_resolution != 'EM pixel grid (legacy)'
            and (registration_algorithm == 'Affine CPD' or registration_algorithm == 'Rigid CPD')):
        # Affine/Rigid CPD's transform is a real matrix, so instead of
        # resampling the moving image up onto the working grid, warping
        # it there, and resampling the result back down to the target
        # resolution (three dense resamples total), fold both resampling
        # steps' scale factors directly into the matrix and warp straight
        # from the raw moving image to the target grid in one pass.
        matrix_combined = _rescale_affine_matrix(transformed.affine.affine_matrix,
                                                 raw_pxlsz=Moving_Points.metadata['pxlsz'],
                                                 working_pxlsz=working_pxlsz,
                                                 target_pxlsz=target_pxlsz)
        extent = tuple(s * working_pxlsz for s in Fixed_Points.metadata['output_shape'])
        coarse_output_shape = (round(extent[0] / target_pxlsz[0]),
                               round(extent[1] / target_pxlsz[1]),
                               round(extent[2] / target_pxlsz[1]))

        if len(get_linked_layers(Moving_Image)) > 0:
            images = get_linked_layers(Moving_Image)
            images.add(Moving_Image)
        else:
            images = [Moving_Image]

        warp_outputs = []
        for image in images:
            print(f'Warping {image.name} with {registration_algorithm} (direct-to-native)...')
            img_wrp = _warp_image_volume_affine(image=as_array(image),
                                                matrix=matrix_combined,
                                                output_shape=coarse_output_shape,
                                                interpolation_order=warping_interpolation_order)
            warp_outputs.append(Image(img_wrp,
                                      name=image.name + '_warped',
                                      colormap=image.colormap,
                                      blending=image.blending,
                                      scale=scale))
        print('Finished warping images')
    else:
        # Make images isotropic for linked layers
        moving_image_list = return_isotropic_image_list(input_image=Moving_Image,
                                                        pxlsz_lm=Moving_Points.metadata['pxlsz'],
                                                        pxlsz_em=(working_pxlsz, working_pxlsz))
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

        for warp_output in warp_outputs:
            warp_output.data = resample_to_pixelsize(warp_output.data, working_pxlsz, target_pxlsz)
            warp_output.scale = scale

    if mobie is not None:
        for warp_output in warp_outputs:
            warp_output.metadata['mobie_affine'] = mobie

    if benchmarking_mode:
        return warp_outputs, transformed, elapsed
    else:
        return warp_outputs, transformed


def _layer_origin_um(layer, pxlsz_nm):
    """Where `layer`'s first voxel lies, in µm (z, y, x): the origin recorded
    in its metadata (e.g. by the sample loaders, for a crop of a file), plus
    napari's `translate`. `pxlsz_nm` is its (z, xy) pixel size in nm."""
    pixel_size_um = np.array([pxlsz_nm[0], pxlsz_nm[1], pxlsz_nm[1]]) / 1000
    translate_voxels = np.asarray(layer.translate[-3:]) / np.asarray(layer.scale[-3:])
    return np.asarray(layer.metadata.get('origin_um', (0, 0, 0))) + translate_voxels * pixel_size_um


def _print_mobie_affine(transformed, registration_algorithm, working_pxlsz_nm,
                        Moving_Points, Fixed_Points, Moving_Image, Fixed_Image):
    """Print a Rigid or Affine CPD registration as a MoBIE affine (µm), and
    return its 12 numbers. BCPD is non-rigid, so it has none.
    `working_pxlsz_nm` is the working grid's voxel size."""
    from ..clemreg.mobie import format_mobie_affine, mobie_affine

    if registration_algorithm not in ('Rigid CPD', 'Affine CPD'):
        return None
    parameters = mobie_affine(transformed.affine.affine_matrix,
                              working_pixel_size_um=working_pxlsz_nm / 1000,
                              moving_origin_um=_layer_origin_um(Moving_Image, Moving_Points.metadata['pxlsz']),
                              fixed_origin_um=_layer_origin_um(Fixed_Image, Fixed_Points.metadata['pxlsz']))
    print(f'\nMoBIE affine (µm), {Moving_Image.name} onto {Fixed_Image.name}. In MoBIE, right-click > '
          f'Transform > Registration - Enter Transformation, select {Moving_Image.name} as the moving '
          f'image, and paste:\n{format_mobie_affine(parameters)}\n')
    return parameters


"""
Helper functions
"""
def _create_json_file(path_to_json):
    dictionary = {
        "registration_algorithm": registration_algorithm,
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
