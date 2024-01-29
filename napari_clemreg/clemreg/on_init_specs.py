#!/usr/bin/env python3
# coding: utf-8

specs = {
    'z_min':{'widget_type': 'SpinBox',
           'label': 'Minimum z value for masking',
           "min": 0, "max": 10, "step": 1,
           'value': 0},

    'z_max':{'widget_type': 'SpinBox',
           'label': 'Maximum z value for masking',
           "min": 0, "max": 10, "step": 1,
           'value': 0},

    'registration_algorithm':{'label': 'Registration Algorithm',
                            'widget_type': 'ComboBox',
                            'choices': ["BCPD", "Rigid CPD"],
                            'choices': ["BCPD", "Rigid CPD", "Affine CPD"],
                            'value': 'Rigid CPD',
                            'tooltip': 'Speed: Rigid CPD > Affine CPD > BCPD'},

    'params_from_json':{'label': 'Parameters from JSON',
                      'widget_type': 'CheckBox',
                      'value': False},

    'load_json_file':{'label': 'Select Parameter File',
                    'widget_type': 'FileEdit',
                    'mode': 'r',
                    'filter': '*.json'},

    'advanced':{'text': 'Parameters custom',
              'widget_type': 'CheckBox',
              'value': False},

    'em_seg_axis':{'text': 'Prediction Across Three Axis',
                 'widget_type': 'CheckBox',
                 'value': False},

    'log_sigma':{'label': 'Sigma',
               'widget_type': 'FloatSpinBox',
               'min': 0.5, 'max': 20, 'step': 0.5,
               'value': 3},

    'log_threshold':{'label': 'Threshold',
                   'widget_type': 'FloatSpinBox',
                   'min': 0, 'max': 20, 'step': 0.1,
                   'value': 1.2},

    'filter_segmentation':{'text': 'Apply size filter to segmentation',
                     'widget_type': 'CheckBox',
                     'value': False},

    'filter_size_lower':{'label': 'Lower filter threshold',
                    'widget_type': 'SpinBox',
                    'min': 0, 'max': 100, 'step': 1,
                    'value': 5},

    'filter_size_upper':{'label': 'Upper filter threshold',
                    'widget_type': 'SpinBox',
                    'min': 0, 'max': 100, 'step': 1,
                    'value': 95},

    'point_cloud_sampling_frequency':{'label': 'Sampling Frequency',
                                    'widget_type': 'SpinBox',
                                    'min': 1, 'max': 100, 'step': 1,
                                    'value': 3},

    'point_cloud_sigma':{'label': 'Sigma',
                       'widget_type': 'FloatSpinBox',
                       'min': 0, 'max': 10, 'step': 0.1,
                       'value': 1.0},

    'registration_voxel_size':{'label': 'Voxel Size',
                             'widget_type': 'SpinBox',
                             'min': 1, 'max': 1000, 'step': 1,
                             'value': 15},

    'registration_max_iterations':{'label': 'Maximum Iterations',
                                 'widget_type': 'SpinBox',
                                 'min': 1, 'max': 1000, 'step': 1,
                                 'value': 50},

    'warping_interpolation_order':{'label': 'Interpolation Order',
                                 'widget_type': 'SpinBox',
                                 'min': 0, 'max': 5, 'step': 1,
                                 'value': 1},

    'warping_approximate_grid':{'label': 'Approximate Grid',
                              'widget_type': 'SpinBox',
                              'min': 1, 'max': 10, 'step': 1,
                              'value': 5},

    'warping_sub_division_factor':{'label': 'Sub-division Factor',
                                 'widget_type': 'SpinBox',
                                 'min': 1, 'max': 10, 'step': 1,
                                 'value': 1},

    'save_json':{'label': 'Save parameters',
               'widget_type': 'CheckBox',
               'value': False},

    'save_json_path':{'label': 'Path to save parameters',
                   'widget_type': 'FileEdit',
                   'mode': 'w',
                   'filter': '*.json'},

    'visualise_intermediate_results':{'label': 'Visualise Intermediate Results',
                                    'widget_type': 'CheckBox',
                                    'value': True
                                    },

    'moving_image_pixelsize_xy':{'label': 'FM Pixel size (xy)',
                                    'widget_type': 'QuantityEdit',
                                    'value': '0 nanometer'
                                    },

    'moving_image_pixelsize_z':{'label': 'FM Pixel size (z)',
                                    'widget_type': 'QuantityEdit',
                                    'value': '0 nanometer'
                                    },

    'fixed_image_pixelsize_xy':{'label': 'EM Pixel size (xy)',
                                    'widget_type': 'QuantityEdit',
                                    'value': '0 nanometer'
                                    },

    'fixed_image_pixelsize_z':{'label': 'EM Pixel size (z)',
                                    'widget_type': 'QuantityEdit',
                                    'value': '0 nanometer'
                                    },

    'registration_direction':{'label': 'Registration direction',
                                    'widget_type': 'RadioButtons',
                                    'choices': [u'FM \u2192 EM', u'EM \u2192 FM'],
                                    'value': u'FM \u2192 EM'
                                    },

    'Moving_Image':{'label': 'Fluorescence Microscopy Image (FM)'},

    'Fixed_Image':{'label': 'Electron Microscopy Image (EM)'},

    'Moving_Segmentation':{'label': 'Fluorescence Microscopy (FM) Segmentation'},

    'Fixed_Segmentation':{'label': 'Electron Microscopy (EM) Segmentation'},

    'Moving_Points':{'label': 'Fluorescence Microscopy (FM) Point Cloud'},

    'Fixed_Points':{'label': 'Electron Microscopy (EM) Point Cloud'}
}
