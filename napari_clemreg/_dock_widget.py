#!/usr/bin/env python
# -*- coding: utf-8 -*-
from napari_plugin_engine import napari_hook_implementation
from .widgets.clean_binary_segmentation import make_clean_binary_segmentation
from .widgets.log_segmentation import make_log_segmentation
from .widgets.data_preprocessing import make_data_preprocessing
from .widgets.point_cloud_registration import make_point_cloud_registration
from .widgets.point_cloud_sampling import make_point_cloud_sampling
from .widgets.warp_image_volume import make_image_warping
from .widgets.convert_em_segmentation import change_layer_type
from .widgets.utility import crop_volume

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [(crop_volume, {"name": "A1. Crop volume"}),
            (make_data_preprocessing, {"name": "A2. Preprocess data"}),
            (make_log_segmentation, {"name": "B. Segment FM data"}),
            (change_layer_type, {"name": "C. Load EM segmentation"}),
            (make_clean_binary_segmentation, {"name": "D. Clean segmentations"}),
            (make_point_cloud_sampling, {"name": "E. Sample point clouds"}),
            (make_point_cloud_registration, {"name": "F. Register point clouds"}),
            (make_image_warping, {"name": "G. Warp image volume"}),]
