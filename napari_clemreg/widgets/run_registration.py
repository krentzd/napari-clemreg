#!/usr/bin/env python3
# coding: utf-8
from napari.layers import Image, Shapes, Labels
from magicgui import magic_factory, widgets
from napari.qt.threading import thread_worker
import time
from ..clemreg.log_segmentation import log_segmentation
from ..clemreg.point_cloud_sampling import point_cloud_sampling
from ..clemreg.mask_roi import mask_roi
from ..clemreg.empanada_segmentation import empanada_segmentation

def on_init(widget):
    """Initializes widget layout and updates widget layout according to user input."""
    standard_settings = ['widget_header', 'Moving_Image', 'Fixed_Image', 'Mask_ROI', 'advanced']
    advanced_settings = ['log_header',
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
                         'white_space_1',
                         'white_space_2',
                         'white_space_3',
                         'white_space_4']

    for x in standard_settings:
        setattr(getattr(widget, x), 'visible', True)
    for x in advanced_settings + ['z_min', 'z_max']:
        setattr(getattr(widget, x), 'visible', False)

    def toggle_transform_widget(advanced: bool):
        if advanced == True:
            for x in advanced_settings + standard_settings:
                setattr(getattr(widget, x), 'visible', True)

        else:
            for x in standard_settings:
                setattr(getattr(widget, x), 'visible', True)
            for x in advanced_settings:
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

    widget.z_max.changed.connect(change_z_min)
    widget.Moving_Image.changed.connect(change_z_max)
    widget.z_min.changed.connect(change_z_max_from_z_min)
    widget.Mask_ROI.changed.connect(reveal_z_min_and_z_max)

    widget.advanced.changed.connect(toggle_transform_widget)

class PointCloudRegistration:
    def __init__(self):
        self.moving_ready = False
        self.fixed_ready = False
        self.fixed_points = None
        self.moving_points = None

    def set_fixed_points(self, points):
        self.fixed_points = points
        self.moving_ready = True
        print(self.moving_ready, self.fixed_ready)
        if self.moving_ready and self.fixed_ready:
            self.register_point_clouds()

    def set_moving_points(self, points):
        self.moving_points = points
        self.fixed_ready = True
        print(self.moving_ready, self.fixed_ready)
        if self.moving_ready and self.fixed_ready:
            self.register_point_clouds()

    def moving_thread_finished(self):
        self.moving_ready = True
        print(self.moving_ready, self.fixed_ready)
        if self.moving_ready and self.fixed_ready:
            self.register_point_clouds()

    def fixed_thread_finished(self):
        self.fixed_ready = True
        print(self.moving_ready, self.fixed_ready)

        if self.moving_ready and self.fixed_ready:
            self.register_point_clouds()

    def configure_worker(self):
        self.worker = self.register_point_clouds()

    # @thread_worker
    def register_point_clouds(self):
        print('Both threads finished! Starting registration')


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
                                       'widget_type':'ComboBox',
                                       'choices': ["BCPD", "Rigid CPD", "Affine CPD"],
                                       'value': 'Rigid CPD',
                                       'tooltip': 'Speed: Rigid CPD > Affine CPD > BCPD'},
               advanced={'text': 'Show advanced parameters',
                          'widget_type': 'CheckBox',
                          'value': False},

               white_space_1={'widget_type': 'Label', 'label': ' '},
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

               white_space_2={'widget_type': 'Label', 'label': ' '},
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

               white_space_3={'widget_type': 'Label', 'label': ' '},
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

               white_space_4={'widget_type': 'Label', 'label': ' '},
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
                                            'value': 1}
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
    advanced,

    white_space_1,
    log_header,
    log_sigma,
    log_threshold,

    white_space_2,
    point_cloud_header,
    point_cloud_sampling_frequency,
    point_cloud_sigma,

    white_space_3,
    registration_header,
    registration_voxel_size,
    registration_every_k_points,
    registration_max_iterations,

    white_space_4,
    warping_header,
    warping_interpolation_order,
    warping_approximate_grid,
    warping_sub_division_factor) -> Image:
    """Run CLEM-Reg end-to-end"""

    @thread_worker
    def _run_moving_thread():
        # seg_volume = log_segmentation(input=Moving_Image,
        #                               sigma=log_sigma,
        #                               threshold=log_threshold)
        # print('Mask_ROI:', Mask_ROI)
        # if Mask_ROI is not None:
        #     seg_volume_mask = mask_roi(input=seg_volume,
        #                                crop_mask=Mask_ROI, z_min=z_min, z_max=z_max)
        # else:
        #     seg_volume_mask = seg_volume
        #
        # point_cloud = point_cloud_sampling(input=seg_volume_mask,
        #                                    sampling_frequency=point_cloud_sampling_frequency / 100,
        #                                    sigma=point_cloud_sigma)
        # return point_cloud
        time.sleep(10)
        print('Moving thread')
        return 10

    @thread_worker
    def _run_fixed_thread():
        # seg_volume = empanada_segmentation(input=Fixed_Image.data)
        # point_cloud = point_cloud_sampling(input=Labels(seg_volume),
        #                                    sampling_frequency=point_cloud_sampling_frequency / 100,
        #                                    sigma=point_cloud_sigma)
        # return point_cloud
        print('Target thread')
        return 11

    registration = PointCloudRegistration()

    def _class_setter_moving(x):
        registration.set_moving_points(x)

    def _class_setter_fixed(x):
        registration.set_fixed_points(x)

    worker_moving = _run_moving_thread()
    worker_moving.returned.connect(_class_setter_moving)
    # worker_moving.finished.connect(registration.moving_thread_finished)
    worker_moving.start()

    worker_fixed = _run_fixed_thread()
    worker_fixed.returned.connect(_class_setter_fixed)
    # worker_fixed.finished.connect(registration.fixed_thread_finished)
    worker_fixed.start()

    # Need to implement equivalent function to join in thread_worker
    # worker_target.join()
    # worker_moving.join()

    # print('Waited for both threads!')
    # worker_log_segmentation = log_segmentation(input=Moving_Image,
    #                                            sigma=log_sigma,
    #                                            threshold=log_threshold)
    #
    # worker_log_segmentation.returned.connect(_add_labels_data)
    # worker_log_segmentation.start()
    #
    # worker_point_cloud_sampling = point_cloud_sampling(input=viewer.layers[Moving_Image.name + '_seg'].data,
    #                                                    sampling_frequency=point_cloud_sampling_frequency / 100,
    #                                                    sigma=point_cloud_sigma)
    #
    # worker_point_cloud_sampling.returned.connect(_add_points_data)
    # worker_log_segmentation.finished.connect(worker_point_cloud_sampling.start)

    # @thread_worker(connect={"returned": _add_data})
    # def _run_registration(input: Labels,
    #                                percentile: int=95):
    #
    #
    # _run_registration(input=input,
    #                            percentile=percentile)
