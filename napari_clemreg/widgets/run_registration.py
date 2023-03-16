#!/usr/bin/env python3
# coding: utf-8
import json
import warnings
import napari
from magicgui import magic_factory
from napari.layers import Image, Shapes, Labels, Points
from napari.utils.notifications import show_error


# Use as worker.join workaround --> Launch registration thread_worker from here
class RegistrationThreadJoiner:
    def __init__(self, worker_function):
        self.moving_ready = False
        self.fixed_ready = False
        self.fixed_points = None
        self.moving_points = None
        self.worker_function = worker_function

    def set_fixed_points(self, points):
        self.fixed_points = points

    def set_moving_points(self, points):
        self.moving_points = points

    def finished_fixed(self):
        self.fixed_ready = True
        if self.moving_ready and self.fixed_ready:
            self.launch_worker()

    def finished_moving(self):
        self.moving_ready = True
        if self.moving_ready and self.fixed_ready:
            self.launch_worker()

    def launch_worker(self):
        self.worker_function(self.moving_points, self.fixed_points)


def on_init(widget):
    """ Initializes widget layout and updates widget layout according to user input.

    Parameters
    ----------
    widget : magicgui.widgets.Widget
        The parent widget of the plugin.
    """
    standard_settings = ['widget_header', 'Moving_Image', 'Fixed_Image', 'Mask_ROI', 'advanced']
    advanced_settings = ['em_seg_header',
                         'em_seg_axis',
                         'log_header',
                         'log_sigma',
                         'log_threshold',
                         'point_cloud_header',
                         'point_cloud_sampling_frequency',
                         'point_cloud_sigma',
                         'registration_header',
                         'registration_voxel_size',
                         'registration_every_k_points',
                         'registration_max_iterations',
                         'warping_header',
                         'warping_interpolation_order',
                         'warping_approximate_grid',
                         'warping_sub_division_factor',
                         'save_json',
                         'visualise_intermediate_results'
                         ]
    json_settings = ['load_json_file']

    for x in standard_settings:
        setattr(getattr(widget, x), 'visible', True)
    for x in advanced_settings + ['z_min', 'z_max']:
        setattr(getattr(widget, x), 'visible', False)
    for x in json_settings:
        setattr(getattr(widget, x), 'visible', True)

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
            widget.params_from_json.value = True
            toggle_json_widget(True)

    def toggle_json_widget(load_json: bool):

        if load_json:
            for x in json_settings:
                setattr(getattr(widget, x), 'visible', True)
            if widget.advanced.value:
                widget.advanced.value = False
        else:
            widget.advanced.value = True
            toggle_transform_widget(True)

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

    widget.z_max.changed.connect(change_z_min)
    widget.Moving_Image.changed.connect(change_z_max)
    widget.z_min.changed.connect(change_z_max_from_z_min)
    widget.Mask_ROI.changed.connect(reveal_z_min_and_z_max)
    widget.advanced.changed.connect(toggle_transform_widget)
    widget.params_from_json.changed.connect(toggle_json_widget)


@magic_factory(widget_init=on_init, layout='vertical', call_button='Register',
               widget_header={'widget_type': 'Label',
                              'label': f'<h1 text-align="left">CLEM-Reg</h1>'},

               z_min={'widget_type': 'SpinBox',
                      'label': 'Minimum z value for masking',
                      "min": 0, "max": 10, "step": 1,
                      'value': 0},
               z_max={'widget_type': 'SpinBox',
                      'label': 'Maximum z value for masking',
                      "min": 0, "max": 10, "step": 1,
                      'value': 0},
               registration_algorithm={'label': 'Registration Algorithm',
                                       'widget_type': 'ComboBox',
                                       'choices': ["BCPD", "Rigid CPD", "Affine CPD"],
                                       'value': 'Rigid CPD',
                                       'tooltip': 'Speed: Rigid CPD > Affine CPD > BCPD'},
               advanced={'text': 'Parameters custom',
                         'widget_type': 'CheckBox',
                         'value': False},

               em_seg_header={'widget_type': 'Label',
                              'label': f'<h3 text-align="left">MitoNet Segmentation Parameters</h3>'},
               em_seg_axis={'text': 'Prediction Across Three Axis',
                            'widget_type': 'CheckBox',
                            'value': False},

               log_header={'widget_type': 'Label',
                           'label': f'<h3 text-align="left">LoG Segmentation Parameters</h3>'},
               log_sigma={'label': 'Sigma',
                          'widget_type': 'FloatSpinBox',
                          'min': 0.5, 'max': 20, 'step': 0.5,
                          'value': 3},
               log_threshold={'label': 'Threshold',
                              'widget_type': 'FloatSpinBox',
                              'min': 0, 'max': 20, 'step': 0.1,
                              'value': 1.2},

               point_cloud_header={'widget_type': 'Label',
                                   'label': f'<h3 text-align="left">Point Cloud Sampling</h3>'},
               point_cloud_sampling_frequency={'label': 'Sampling Frequency',
                                               'widget_type': 'SpinBox',
                                               'min': 1, 'max': 100, 'step': 1,
                                               'value': 5},
               point_cloud_sigma={'label': 'Sigma',
                                  'widget_type': 'FloatSpinBox',
                                  'min': 0, 'max': 10, 'step': 0.1,
                                  'value': 1.0},

               registration_header={'widget_type': 'Label',
                                    'label': f'<h3 text-align="left">Point Cloud Registration</h3>'},
               registration_voxel_size={'label': 'Voxel Size',
                                        'widget_type': 'SpinBox',
                                        'min': 1, 'max': 1000, 'step': 1,
                                        'value': 5},
               registration_every_k_points={'label': 'Subsampling',
                                            'widget_type': 'SpinBox',
                                            'min': 1, 'max': 1000, 'step': 1,
                                            'value': 1},
               registration_max_iterations={'label': 'Maximum Iterations',
                                            'widget_type': 'SpinBox',
                                            'min': 1, 'max': 1000, 'step': 1,
                                            'value': 50},

               warping_header={'widget_type': 'Label',
                               'label': f'<h3 text-align="left">Image Warping</h3>'},
               warping_interpolation_order={'label': 'Interpolation Order',
                                            'widget_type': 'SpinBox',
                                            'min': 0, 'max': 5, 'step': 1,
                                            'value': 1},
               warping_approximate_grid={'label': 'Approximate Grid',
                                         'widget_type': 'SpinBox',
                                         'min': 1, 'max': 10, 'step': 1,
                                         'value': 5},
               warping_sub_division_factor={'label': 'Sub-division Factor',
                                            'widget_type': 'SpinBox',
                                            'min': 1, 'max': 10, 'step': 1,
                                            'value': 1},
               save_json={'label': 'Save parameters',
                          'widget_type': 'CheckBox',
                          'value': False},
               params_from_json={'label': 'Parameters from JSON',
                                 'widget_type': 'CheckBox',
                                 'value': True},
               load_json_file={'label': 'Select Parameter File',
                               'widget_type': 'FileEdit',
                               'mode': 'r',
                               'filter': '*.json'},
               visualise_intermediate_results={'label': 'Visualise Intermediate Results',
                                               'widget_type': 'CheckBox',
                                               'value': True
                                               }
               )
def make_run_registration(
        viewer: 'napari.viewer.Viewer',
        widget_header,
        Moving_Image: Image,
        Fixed_Image: Image,
        Mask_ROI: Shapes,
        z_min,
        z_max,
        registration_algorithm,
        params_from_json,
        advanced,

        em_seg_header,
        em_seg_axis,

        log_header,
        log_sigma,
        log_threshold,

        point_cloud_header,
        point_cloud_sampling_frequency,
        point_cloud_sigma,

        registration_header,
        registration_voxel_size,
        registration_every_k_points,
        registration_max_iterations,

        warping_header,
        warping_interpolation_order,
        warping_approximate_grid,
        warping_sub_division_factor,

        save_json,
        visualise_intermediate_results,

        load_json_file) -> Image:
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
    white_space_2
    point_cloud_header
    point_cloud_sampling_frequency
    point_cloud_sigma
    white_space_3
    registration_header
    registration_voxel_size
    registration_every_k_points
    registration_max_iterations
    white_space_4
    warping_header
    warping_interpolation_order
    warping_approximate_grid
    warping_sub_division_factor

    Returns
    -------

    """
    from ..clemreg.empanada_segmentation import empanada_segmentation
    from ..clemreg.log_segmentation import log_segmentation
    from ..clemreg.mask_roi import mask_roi, mask_area
    from ..clemreg.point_cloud_registration import point_cloud_registration
    from ..clemreg.point_cloud_sampling import point_cloud_sampling
    from ..clemreg.warp_image_volume import warp_image_volume
    from ..clemreg.data_preprocessing import make_isotropic
    from napari.qt.threading import thread_worker
    from napari.layers.utils._link_layers import link_layers

    if params_from_json and load_json_file.is_file():
        f = open(str(load_json_file))

        data = json.load(f)
        try:
            registration_algorithm = data["registration_algorithm"]
            em_seg_axis = data["em_seg_axis"]
            log_sigma = data["log_sigma"]
            log_threshold = data["log_threshold"]
            point_cloud_sampling_frequency = data["point_cloud_sampling_frequency"]
            point_cloud_sigma = data["point_cloud_sigma"]
            registration_voxel_size = data["registration_voxel_size"]
            registration_every_k_points = data["registration_every_k_points"]
            registration_max_iterations = data["registration_max_iterations"]
            warping_interpolation_order = data["warping_interpolation_order"]
            warping_approximate_grid = data["warping_approximate_grid"]
            warping_sub_division_factor = data["warping_sub_division_factor"]
        except KeyError:
            warnings.warn("JSON file missing required param")
            return
    elif params_from_json and not load_json_file.is_file():
        warnings.warn("Load from JSON selected but no JSON file selected or file path isn't real")
        return

    @thread_worker
    def _run_moving_thread():
        # Inplace operation, metadata extraction only works if TIFF file
        z_zoom = make_isotropic(input_image=Moving_Image)

        seg_volume = log_segmentation(input=Moving_Image,
                                      sigma=log_sigma,
                                      threshold=log_threshold)

        if len(set(seg_volume.data.ravel())) <= 1:
            return 'No segmentation'

        if Mask_ROI is not None:
            seg_volume_mask = mask_roi(input=seg_volume,
                                       crop_mask=Mask_ROI,
                                       z_min=int(z_min * z_zoom),
                                       z_max=int(z_max * z_zoom))
        else:
            seg_volume_mask = seg_volume

        if visualise_intermediate_results:
            yield seg_volume_mask, 'lm'

        point_cloud = point_cloud_sampling(input=seg_volume_mask,
                                           sampling_frequency=point_cloud_sampling_frequency / 100,
                                           sigma=point_cloud_sigma)
        return point_cloud

    @thread_worker
    def _run_fixed_thread():
        seg_volume = empanada_segmentation(input=Fixed_Image.data,
                                           axis_prediction=em_seg_axis)

        if len(set(seg_volume.ravel())) <= 1:
            return 'No segmentation'

        if visualise_intermediate_results:
            yield seg_volume, 'em'

        point_cloud = point_cloud_sampling(input=Labels(seg_volume),
                                           sampling_frequency=point_cloud_sampling_frequency / 100,
                                           sigma=point_cloud_sigma)
        return point_cloud

    def _add_data(return_value):
        if return_value == 'No segmentation':
            show_error('WARNING: No mitochondria in Fixed Image or Moving Image')
            return

        if isinstance(return_value, list):
            layers = []
            for image_data in return_value:
                data, kwargs = image_data
                viewer.add_image(data, **kwargs)
                layers.append(viewer.layers[kwargs['name']])
            link_layers(layers)
        else:
            data, kwargs = return_value
            viewer.add_image(data, **kwargs)

    def _yield_segmentation(yield_value):

        image = yield_value[0]
        image_type = yield_value[1]

        if image_type == 'lm':
            viewer.add_labels(image.data, name=image_type)
        else:
            viewer.add_labels(image, name=image_type)

    def _yield_point_clouds(yield_value):
        mp = yield_value[0]
        fp = yield_value[1]

        viewer.add_points(mp.data,
                          name='moving_points',
                          face_color='red')
        viewer.add_points(fp.data,
                          name='fixed_points',
                          face_color='blue')

    def _create_json_file():
        dictionary = {
            "registration_algorithm": registration_algorithm,
            "em_seg_axis": em_seg_axis,
            "log_sigma": log_sigma,
            "log_threshold": log_threshold,
            "point_cloud_sampling_frequency": point_cloud_sampling_frequency,
            "point_cloud_sigma": point_cloud_sigma,
            "registration_voxel_size": registration_voxel_size,
            "registration_every_k_points": registration_every_k_points,
            "registration_max_iterations": registration_max_iterations,
            "warping_interpolation_order": warping_interpolation_order,
            "warping_approximate_grid": warping_approximate_grid,
            "warping_sub_division_factor": warping_sub_division_factor
        }

        json_object = json.dumps(dictionary, indent=4)

        with open("sample.json", "w") as outfile:
            outfile.write(json_object)

    @thread_worker(connect={"returned": _add_data, "yielded": _yield_point_clouds})
    def _run_registration_thread(moving_points, fixed_points):
        if moving_points == 'No segmentation' or fixed_points == 'No segmentation':
            return 'No segmentation'

        if visualise_intermediate_results:
            yield moving_points, fixed_points

        moving, fixed, transformed, kwargs = point_cloud_registration(moving_points.data, fixed_points.data,
                                                                      algorithm=registration_algorithm,
                                                                      voxel_size=registration_voxel_size,
                                                                      every_k_points=registration_every_k_points,
                                                                      max_iterations=registration_max_iterations)

        if registration_algorithm == 'Affine CPD' or registration_algorithm == 'Rigid CPD':
            transformed = Points(moving, **kwargs)

        return warp_image_volume(moving_image=Moving_Image,
                                 fixed_image=Fixed_Image.data,
                                 transform_type=registration_algorithm,
                                 moving_points=Points(moving),
                                 transformed_points=transformed,
                                 interpolation_order=warping_interpolation_order,
                                 approximate_grid=warping_approximate_grid,
                                 sub_division_factor=warping_sub_division_factor)

    if Moving_Image is None or Fixed_Image is None:
        warnings.warn("WARNING: You have not inputted both a fixed and moving image")
        return

    if len(Moving_Image.data.shape) != 3:
        warnings.warn("WARNING: Your moving_image must be 3D, you're current input has a shape of {}".format(
            Moving_Image.data.shape))
        return
    elif len(Moving_Image.data.shape) == 3 and Moving_Image.data.shape[2] == 3:
        warnings.warn("WARNING: YOUR moving_image is RGB, your input must be grayscale and 3D")
        return

    if len(Fixed_Image.data.shape) != 3:
        warnings.warn("WARNING: Your Fixed_Image must be 3D, you're current input has a shape of {}".format(
            Moving_Image.data.shape))
        return
    elif len(Fixed_Image.data.shape) == 3 and Fixed_Image.data.shape[2] == 3:
        warnings.warn("WARNING: YOUR fixed_image is RGB, your input must be grayscale and 3D")
        return

    if Mask_ROI is not None:
        if len(Mask_ROI.data) != 1:
            warnings.warn("WARNING: You must input only 1 Mask ROI, you have inputted {}.".format(len(Mask_ROI.data)))
            return
        if mask_area(Mask_ROI.data[0][:, 1], Mask_ROI.data[0][:, 2]) > Moving_Image.data.shape[1] * \
                Moving_Image.data.shape[2]:
            warnings.warn("WARNING: Your mask size exceeds the size of the image.")
            return

    if save_json and not params_from_json:
        _create_json_file()

    joiner = RegistrationThreadJoiner(worker_function=_run_registration_thread)

    def _class_setter_moving(x):
        joiner.set_moving_points(x)

    def _class_setter_fixed(x):
        joiner.set_fixed_points(x)

    def _finished_moving_emitter():
        joiner.finished_moving()

    def _finished_fixed_emitter():
        joiner.finished_fixed()

    worker_moving = _run_moving_thread()
    worker_moving.returned.connect(_class_setter_moving)
    worker_moving.finished.connect(_finished_moving_emitter)
    worker_moving.yielded.connect(_yield_segmentation)
    worker_moving.start()

    worker_fixed = _run_fixed_thread()
    worker_fixed.returned.connect(_class_setter_fixed)
    worker_fixed.finished.connect(_finished_fixed_emitter)
    worker_fixed.yielded.connect(_yield_segmentation)
    worker_fixed.start()
