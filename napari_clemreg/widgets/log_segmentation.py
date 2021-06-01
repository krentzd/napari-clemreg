#!/usr/bin/env python3
# coding: utf-8
from scipy.ndimage import gaussian_filter1d
import numpy as np
from skimage import exposure
from napari.qt.threading import thread_worker
from magicgui import magic_factory, widgets
import napari
from napari.types import LabelsData
from napari.layers import Image
from skimage import feature
from typing_extensions import Annotated


@magic_factory
def make_log_segmentation(
    viewer: "napari.viewer.Viewer",
    input: Image,
    sigma: Annotated[float, {"min": 0.5, "max": 20, "step": 0.5}]=3,
    threshold: Annotated[float, {"min": 0, "max": 20, "step": 0.1}]=1.2
):
    from napari.qt import thread_worker

    pbar = widgets.ProgressBar()
    pbar.range = (0, 0)  # unknown duration
    make_log_segmentation.insert(0, pbar)  # add progress bar to the top of widget

    def _min_max_scaling(data):
        n = data - np.min(data)
        d = np.max(data) - np.min(data)

        return n / d

    def _diff_of_gauss(img, sigma_1=2.5, sigma_2=4):
        gauss_img_0_e = gaussian_filter1d(img, sigma_1, axis=0)
        gauss_img_1_e = gaussian_filter1d(gauss_img_0_e, sigma_1, axis=1)
        gauss_img_2_e = gaussian_filter1d(gauss_img_1_e, sigma_1, axis=2)

        gauss_img_0_i = gaussian_filter1d(img, sigma_2, axis=0)
        gauss_img_1_i = gaussian_filter1d(gauss_img_0_i, sigma_2, axis=1)
        gauss_img_2_i = gaussian_filter1d(gauss_img_1_i, sigma_2, axis=2)

        diff_of_gauss = gauss_img_2_e - gauss_img_2_i

        return diff_of_gauss

    def _slice_adaptive_thresholding(img, thresh):
        thresh_img = []
        for i in range(img.shape[0]):
            slice = exposure.rescale_intensity(img[i], out_range='uint8')
            slice_thresh = np.sum(slice) / (slice.shape[0] * slice.shape[1]) * thresh
            slice[slice < slice_thresh] = 0
            slice[slice >= slice_thresh] = 1
            thresh_img.append(slice)

        return np.asarray(thresh_img)

    # this function will be called after we return
    def _add_data(return_value, self=make_log_segmentation):
        data, kwargs = return_value
        viewer.add_labels(data, **kwargs)
        self.pop(0).hide()  # remove the progress bar

    @thread_worker(connect={"returned": _add_data})
    def _log_segmentation(input: Image,
                         sigma: float=3,
                         threshold: float=1.2):

        volume = _min_max_scaling(input.data)
        sigma_2 = sigma * 1.6
        log_iso_volume = _diff_of_gauss(volume, sigma, sigma_2)
        seg_volume = _slice_adaptive_thresholding(log_iso_volume, threshold)

        kwargs = dict(
            name=input.name + '_seg'
        )
        return (seg_volume, kwargs)

    # start the thread
    _log_segmentation(input=input,
                      sigma=sigma,
                      threshold=threshold)
