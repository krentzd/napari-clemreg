#!/usr/bin/env python
# -*- coding: utf-8 -*-
from napari_plugin_engine import napari_hook_implementation
from .widgets.clean_binary_segmentation import make_clean_binary_segmentation
from .widgets.log_segmentation import make_log_segmentation
from .widgets.data_preprocessing import make_data_preprocessing
from .widgets.point_cloud_registration import make_point_cloud_registration
from .widgets.point_cloud_sampling import make_point_cloud_sampling
from .widgets.warp_image_volume import make_image_warping
from .widgets.em_segmentation import train_model
from .widgets.utility import mask_roi

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [(mask_roi, {"name": "A1. Mask ROI"}),
            (make_data_preprocessing, {"name": "A2. Preprocess data"}),
            (make_log_segmentation, {"name": "B. Segment FM data"}),
            (train_model, {"name": "C. Train EM segmentation model"}),
            (make_clean_binary_segmentation, {"name": "D. Clean segmentations"}),
            (make_point_cloud_sampling, {"name": "E. Sample point clouds"}),
            (make_point_cloud_registration, {"name": "F. Register point clouds"}),
            (make_image_warping, {"name": "G. Warp image volume"}),]
