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
from napari.qt.threading import GeneratorWorker
from ..clemreg.on_init_specs import specs

# Use as worker.join workaround --> Launch registration thread_worker from here
class RegistrationThreadJoiner:
    def __init__(self, worker_function, init_kwargs, returned, yielded):
        self.moving_ready = False
        self.fixed_ready = False
        self.worker_function = worker_function
        self.init_kwargs = init_kwargs
        self.returned = returned
        self.yielded = yielded

    def set_moving_kwargs(self, kwargs):
        self.moving_kwargs = kwargs

    def set_fixed_kwargs(self, kwargs):
        self.fixed_kwargs = kwargs

    def finished_fixed(self):
        self.fixed_ready = True
        if self.moving_ready and self.fixed_ready:
            self.launch_worker()

    def finished_moving(self):
        self.moving_ready = True
        if self.moving_ready and self.fixed_ready:
            self.launch_worker()

    def launch_worker(self):
        print('Launching registration and warping...')

        worker = self.worker_function(**{**self.init_kwargs, **self.moving_kwargs,**self.fixed_kwargs})
        worker.returned.connect(self.returned)
        if isinstance(worker, GeneratorWorker):
            worker.yielded.connect(self.yielded)
        worker.start()

def on_init(widget):
    """ Initializes widget layout and updates widget layout according to user input.

    Parameters
    ----------
    widget : magicgui.widgets.Widget
        The parent widget of the plugin.
    """
    from ..clemreg.data_preprocessing import get_pixelsize

    standard_settings = ['widget_header', 'Moving_Image', 'Fixed_Image', 'Mask_ROI', 'advanced']
    advanced_settings = ['em_seg_header',
                         'em_seg_axis',
                         'log_header',
                         'log_sigma',
                         'log_threshold',
                         'filter_segmentation',
                         'point_cloud_header',
                         'point_cloud_sampling_frequency',
                         'point_cloud_sigma',
                         'registration_header',
                         'registration_voxel_size',
                         'registration_max_iterations',
                         'warping_header',
                         'warping_interpolation_order',
                         'warping_approximate_grid',
                         'warping_sub_division_factor',
                         'save_json',
                         'visualise_intermediate_results']

    json_settings = ['load_json_file']
    filter_segmentation_settings = ['filter_size_lower', 'filter_size_upper']
    save_json_settings = ['save_json_path']

    for x in standard_settings:
        setattr(getattr(widget, x), 'visible', True)
    for x in advanced_settings + ['z_min', 'z_max'] + json_settings + filter_segmentation_settings + save_json_settings:
        setattr(getattr(widget, x), 'visible', False)

    def toggle_transform_widget(advanced: bool):
        if advanced:
            for x in advanced_settings + standard_settings:
                setattr(getattr(widget, x), 'visible', True)
            for x in json_settings:
                setattr(getattr(widget, x), 'visible', False)

            if widget.params_from_json.value:
                widget.params_from_json.value = False

        else:
            for x in standard_settings:
                setattr(getattr(widget, x), 'visible', True)
            for x in advanced_settings:
                setattr(getattr(widget, x), 'visible', False)

    def toggle_json_widget(load_json: bool):
        if load_json:
            for x in json_settings:
                setattr(getattr(widget, x), 'visible', True)
            if widget.advanced.value:
                widget.advanced.value = False
        else:
            for x in json_settings:
                setattr(getattr(widget, x), 'visible', False)

    def toggle_filter_segmentation(filter_segmentation: bool):
        if filter_segmentation:
            for x in filter_segmentation_settings:
                setattr(getattr(widget, x), 'visible', True)
        else:
            for x in filter_segmentation_settings:
                setattr(getattr(widget, x), 'visible', False)

    def toggle_save_json(save_json: bool):
        if save_json:
            for x in save_json_settings:
                setattr(getattr(widget, x), 'visible', True)
        else:
            for x in save_json_settings:
                setattr(getattr(widget, x), 'visible', False)

    def change_z_max(input_image: Image):
        if len(input_image.data.shape) == 3:
            widget.z_max.max = input_image.data.shape[0]
            widget.z_max.value = input_image.data.shape[0]
        elif len(input_image.data.shape) == 4:
            widget.z_max.max = input_image.data.shape[1]
            widget.z_max.value = input_image.data.shape[1]

    def change_z_min(z_max_val: int):
        widget.z_min.max = z_max_val

    def change_z_max_from_z_min(z_min_val: int):
        widget.z_max.min = z_min_val

    # TODO: z_min and z_max only shown and not hidden if no layer chosen
    def reveal_z_min_and_z_max():
        if len(widget.Mask_ROI.choices) > 0:
            for x in ['z_min', 'z_max']:
                setattr(getattr(widget, x), 'visible', True)
        else:
            for x in ['z_min', 'z_max']:
                setattr(getattr(widget, x), 'visible', False)

    def change_moving_pixelsize(input_image: Image):
        moving_xy_pixelsize, __, moving_z_pixelsize, unit = get_pixelsize(input_image.metadata)

        if unit in ['nanometer', 'nm', 'um', 'micron', 'micrometer']:
            if unit == 'um' or unit == 'micron':
                unit = 'micrometer'
            elif unit == 'nm':
                unit = 'nanometer'
        else:
            unit = 'nanometer'

        widget.moving_image_pixelsize_xy.value = str(moving_xy_pixelsize) + str(unit)
        widget.moving_image_pixelsize_z.value = str(moving_z_pixelsize) + str(unit)

    def change_fixed_pixelsize(input_image: Image):
        fixed_xy_pixelsize, __, fixed_z_pixelsize, unit = get_pixelsize(input_image.metadata)

        if unit in ['nanometer', 'nm', 'um', 'micron', 'micrometer']:
            if unit == 'um' or unit == 'micron':
                unit = 'micrometer'
            elif unit == 'nm':
                unit = 'nanometer'
        else:
            unit = 'nanometer'

        widget.fixed_image_pixelsize_xy.value = str(fixed_xy_pixelsize) + str(unit)
        widget.fixed_image_pixelsize_z.value = str(fixed_z_pixelsize) + str(unit)

    widget.z_max.changed.connect(change_z_min)
    widget.Moving_Image.changed.connect(change_z_max)
    widget.Moving_Image.changed.connect(change_moving_pixelsize)
    widget.Fixed_Image.changed.connect(change_fixed_pixelsize)
    widget.z_min.changed.connect(change_z_max_from_z_min)
    widget.Mask_ROI.changed.connect(reveal_z_min_and_z_max)
    widget.advanced.changed.connect(toggle_transform_widget)
    widget.params_from_json.changed.connect(toggle_json_widget)
    widget.filter_segmentation.changed.connect(toggle_filter_segmentation)
    widget.save_json.changed.connect(toggle_save_json)

@magic_factory(widget_init=on_init, layout='vertical', call_button='Register',
               widget_header={'widget_type': 'Label',
                              'label': f'<h1 text-align="left">CLEM-Reg</h1>'},

               z_min=specs['z_min'],
               z_max=specs['z_max'],
               registration_algorithm=specs['registration_algorithm'],
               params_from_json=specs['params_from_json'],
               load_json_file=specs['load_json_file'],
               advanced=specs['advanced'],

               em_seg_header={'widget_type': 'Label',
                              'label': f'<h3 text-align="left">MitoNet Segmentation Parameters</h3>'},

               em_seg_axis=specs['em_seg_axis'],

               log_header={'widget_type': 'Label',
                           'label': f'<h3 text-align="left">LoG Segmentation Parameters</h3>'},

               log_sigma=specs['log_sigma'],
               log_threshold=specs['log_threshold'],
               filter_segmentation=specs['filter_segmentation'],
               filter_size_lower=specs['filter_size_lower'],
               filter_size_upper=specs['filter_size_upper'],

               point_cloud_header={'widget_type': 'Label',
                                   'label': f'<h3 text-align="left">Point Cloud Sampling</h3>'},
               point_cloud_sampling_frequency=specs['point_cloud_sampling_frequency'],
               point_cloud_sigma=specs['point_cloud_sigma'],

               registration_header={'widget_type': 'Label',
                                    'label': f'<h3 text-align="left">Point Cloud Registration</h3>'},
               registration_voxel_size=specs['registration_voxel_size'],
               registration_max_iterations=specs['registration_max_iterations'],

               warping_header={'widget_type': 'Label',
                               'label': f'<h3 text-align="left">Image Warping</h3>'},
               warping_interpolation_order=specs['warping_interpolation_order'],
               warping_approximate_grid=specs['warping_approximate_grid'],
               warping_sub_division_factor=specs['warping_sub_division_factor'],
               save_json=specs['save_json'],
               save_json_path=specs['save_json_path'],
               visualise_intermediate_results=specs['visualise_intermediate_results'],
               moving_image_pixelsize_xy=specs['moving_image_pixelsize_xy'],
               moving_image_pixelsize_z=specs['moving_image_pixelsize_z'],
               fixed_image_pixelsize_xy=specs['fixed_image_pixelsize_xy'],
               fixed_image_pixelsize_z=specs['fixed_image_pixelsize_z'],
               registration_direction=specs['registration_direction'],
               Moving_Image=specs['Moving_Image'],
               Fixed_Image=specs['Fixed_Image']
               )
def make_run_registration(
        viewer: 'napari.viewer.Viewer',
        widget_header,
        Moving_Image: Image,

        moving_image_pixelsize_xy,
        moving_image_pixelsize_z,
        Mask_ROI: Shapes,
        z_min,
        z_max,

        Fixed_Image: Image,
        fixed_image_pixelsize_xy,
        fixed_image_pixelsize_z,

        registration_algorithm,
        params_from_json,
        load_json_file,
        advanced,

        em_seg_header,
        em_seg_axis,

        log_header,
        log_sigma,
        log_threshold,
        filter_segmentation,
        filter_size_lower,
        filter_size_upper,

        point_cloud_header,
        point_cloud_sampling_frequency,
        registration_voxel_size,
        point_cloud_sigma,

        registration_header,
        registration_max_iterations,

        warping_header,
        warping_interpolation_order,
        warping_approximate_grid,
        warping_sub_division_factor,

        save_json,
        save_json_path,
        visualise_intermediate_results,

        registration_direction
        ) -> Image:
    """Run CLEM-Reg end-to-end

    Parameters
    ----------
    save_json
    viewer
    widget_header
    Moving_Image
    Fixed_Image
    Mask_ROI
    z_min
    z_max
    registration_algorithm
    advanced
    white_space_0
    em_seg_header
    em_seg_axis
    white_space_1
    log_header
    log_sigma
    log_threshold
    zoom_value
    filter_size
    white_space_2
    point_cloud_header
    point_cloud_sampling_frequency
    point_cloud_sigma
    white_space_3
    registration_header
    registration_voxel_size
    registration_max_iterations
    white_space_4
    warping_header
    warping_interpolation_order
    warping_approximate_grid
    warping_sub_division_factor

    Returns
    -------

    """
    from pathlib import Path
    from ..clemreg.empanada_segmentation import empanada_segmentation
    from ..clemreg.log_segmentation import log_segmentation, filter_binary_segmentation
    from ..clemreg.mask_roi import mask_roi, mask_area
    from ..clemreg.point_cloud_registration import point_cloud_registration
    from ..clemreg.point_cloud_sampling import point_cloud_sampling
    from ..clemreg.warp_image_volume import warp_image_volume
    from ..clemreg.data_preprocessing import make_isotropic, _make_isotropic
    from napari.qt.threading import thread_worker
    from napari.layers.utils._link_layers import link_layers

    def _add_data(return_value):
        if isinstance(return_value, str):
            show_error('WARNING: No mitochondria in Moving Image')
            return
        if isinstance(return_value, list):
            layers = []
            for image_layer in return_value:
                print(f'Adding {image_layer.name} to viewer...')
                viewer.add_layer(image_layer)
                layers.append(viewer.layers[image_layer.name])
            link_layers(layers)
        else:
            print(f'Adding {return_value.name} to viewer...')
            viewer.add_layer(return_value)

    def _yield_segmentation(yield_value):
        viewer.add_layer(yield_value)

    def _yield_point_clouds(yield_value):
        points, kwargs = yield_value[0], yield_value[1]
        viewer.add_points(points.data, **kwargs)

    @thread_worker
    def _run_moving_thread(**kwargs):
        from ..clemreg.widget_components import run_moving_segmentation

        seg_volume_mask = run_moving_segmentation(**kwargs)
        seg_volume_mask = Labels(seg_volume_mask.astype(np.uint32),
                                 name='FM_segmentation',
                                 metadata=Moving_Image.metadata)

        if visualise_intermediate_results:
            yield seg_volume_mask

        return {'Moving_Segmentation': seg_volume_mask}

    @thread_worker
    def _run_fixed_thread(**kwargs):
        from ..clemreg.widget_components import run_fixed_segmentation
        #Increasing levels of CLAHE

        seg_volume = run_fixed_segmentation(**kwargs)
        seg_volume = Labels(seg_volume.astype(np.int64),
                            name='EM_segmentation',
                            metadata=Fixed_Image.metadata)

        if visualise_intermediate_results:
            yield seg_volume

        return {'Fixed_Segmentation': seg_volume}

    @thread_worker
    def _run_registration_thread(**kwargs):
        from ..clemreg.widget_components import run_point_cloud_sampling
        from ..clemreg.widget_components import run_point_cloud_registration_and_warping

        point_cloud_keys = ['Moving_Segmentation',
                            'Fixed_Segmentation',
                            'moving_image_pixelsize_xy',
                            'moving_image_pixelsize_z',
                            'fixed_image_pixelsize_xy',
                            'fixed_image_pixelsize_z',
                            'point_cloud_sampling_frequency',
                            'voxel_size',
                            'point_cloud_sigma']

        point_cloud_kwargs = dict((k, kwargs[k]) for k in point_cloud_keys if k in kwargs)
        moving_points, fixed_points = run_point_cloud_sampling(**point_cloud_kwargs)

        if visualise_intermediate_results:
            yield (moving_points.data, {'name': 'moving_points', 'face_color': 'red'})
            yield (fixed_points.data, {'name': 'fixed_points', 'face_color': 'blue'})

        reg_and_warping_keys = ['Moving_Image',
                                'Fixed_Image',
                                'registration_algorithm',
                                'registration_max_iterations',
                                'warping_interpolation_order',
                                'warping_approximate_grid',
                                'warping_sub_division_factor',
                                'registration_direction']

        reg_and_warping_kwargs = dict((k, kwargs[k]) for k in reg_and_warping_keys if k in kwargs)
        point_cloud_return_kwargs = dict(Moving_Points=moving_points, Fixed_Points=fixed_points)
        point_cloud_reg_and_warping_kwargs = {**point_cloud_return_kwargs, **reg_and_warping_kwargs}
        warp_outputs, transformed = run_point_cloud_registration_and_warping(**point_cloud_reg_and_warping_kwargs)

        if visualise_intermediate_results:
            yield (transformed, {'name': 'transformed_points', 'face_color': 'yellow'})

        return warp_outputs

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

    registration_thread_kwargs = dict(
        moving_image_pixelsize_xy=moving_image_pixelsize_xy,
        moving_image_pixelsize_z=moving_image_pixelsize_z,
        fixed_image_pixelsize_xy=fixed_image_pixelsize_xy,
        fixed_image_pixelsize_z=fixed_image_pixelsize_z,
        point_cloud_sampling_frequency=point_cloud_sampling_frequency,
        voxel_size=registration_voxel_size,
        point_cloud_sigma=point_cloud_sigma,
        Moving_Image=Moving_Image,
        Fixed_Image=Fixed_Image,
        registration_algorithm=registration_algorithm,
        registration_max_iterations=registration_max_iterations,
        warping_interpolation_order=warping_interpolation_order,
        warping_approximate_grid=warping_approximate_grid,
        warping_sub_division_factor=warping_sub_division_factor,
        registration_direction=registration_direction
    )
    joiner = RegistrationThreadJoiner(worker_function=_run_registration_thread,
                                      init_kwargs=registration_thread_kwargs,
                                      returned=_add_data,
                                      yielded=_yield_point_clouds)

    def _class_setter_moving(x):
        joiner.set_moving_kwargs(x)

    def _class_setter_fixed(x):
        joiner.set_fixed_kwargs(x)

    def _finished_moving_emitter():
        joiner.finished_moving()

    def _finished_fixed_emitter():
        joiner.finished_fixed()

    worker_moving = _run_moving_thread(Moving_Image=Moving_Image,
                                       Mask_ROI=Mask_ROI,
                                       z_min=z_min,
                                       z_max=z_max,
                                       log_sigma=log_sigma,
                                       log_threshold=log_threshold,
                                       filter_segmentation=filter_segmentation,
                                       filter_size_lower=filter_size_lower,
                                       filter_size_upper=filter_size_upper)
    worker_moving.returned.connect(_class_setter_moving)
    worker_moving.finished.connect(_finished_moving_emitter)
    worker_moving.yielded.connect(_yield_segmentation)
    worker_moving.start()

    worker_fixed = _run_fixed_thread(Fixed_Image=Fixed_Image,
                                     em_seg_axis=em_seg_axis)
    worker_fixed.returned.connect(_class_setter_fixed)
    worker_fixed.finished.connect(_finished_fixed_emitter)
    worker_fixed.yielded.connect(_yield_segmentation)
    worker_fixed.start()
