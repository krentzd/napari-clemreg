#!/usr/bin/env python3
# coding: utf-8
from napari.layers import Image, Shapes
from magicgui import magic_factory, widgets
from typing_extensions import Annotated

def on_init(widget):
    """Initializes widget layout and updates widget layout according to user input."""
    standard_settings = ['widget_header', 'moving_image', 'fixed_image', 'mask_roi', 'advanced']
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
    for x in advanced_settings:
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

    widget.advanced.changed.connect(toggle_transform_widget)


@magic_factory(widget_init=on_init, layout='vertical', call_button='Register',
               widget_header={'widget_type': 'Label',
                              'label': f'<h1 text-align="center">CLEM-Reg</h1>'},
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
                           'label': f'<h3 text-align="center">LoG Segmentation Parameters</h3>'},
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
                                   'label': f'<h3 text-align="center">Point Cloud Sampling</h3>'},
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
                                    'label': f'<h3 text-align="center">Point Cloud Registration</h3>'},
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
                               'label': f'<h3 text-align="center">Image Warping</h3>'},
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
    moving_image: Image,
    fixed_image: Image,
    mask_roi: Shapes,
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

    def _add_data(return_value, self=make_clean_binary_segmentation):
        print('Adding new layer to viewer...')
        data, kwargs = return_value
        viewer.add_labels(data, **kwargs)
        self.pop(0).hide()  # remove the progress bar
        print('Done!')

    # @thread_worker(connect={"returned": _add_data})
    # def _run_registration(input: Labels,
    #                                percentile: int=95):
    #
    #
    # _run_registration(input=input,
    #                            percentile=percentile)
