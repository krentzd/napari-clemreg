#!/usr/bin/env python
# -*- coding: utf-8 -*-
from napari_plugin_engine import napari_hook_implementation
from .widgets.clean_binary_segmentation import make_clean_binary_segmentation
from .widgets.log_segmentation import make_log_segmentation
from .widgets.data_preprocessing import make_data_preprocessing
from .widgets.point_cloud_registration import make_point_cloud_registration
from .widgets.point_cloud_sampling import make_point_cloud_sampling
from .widgets.warp_image_volume import make_image_warping
from .widgets.em_segmentation import train_model, predict_from_model
from .widgets.utility import mask_roi

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [(mask_roi, {"name": "Mask ROI"}),
            (make_data_preprocessing, {"name": "Preprocess data"}),
            (make_log_segmentation, {"name": "Segment FM data"}),
            (train_model, {"name": "Train EM segmentation model"}),
            (predict_from_model, {"name": "Segment EM data"}),
            (make_clean_binary_segmentation, {"name": "Clean segmentations"}),
            (make_point_cloud_sampling, {"name": "Sample point clouds"}),
            (make_point_cloud_registration, {"name": "Register point clouds"}),
            (make_image_warping, {"name": "Warp image volume"}),]
