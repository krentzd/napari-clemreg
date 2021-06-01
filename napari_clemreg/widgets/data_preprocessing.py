#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from magicgui import magic_factory, widgets
from scipy import ndimage
from napari.layers import Image
from napari.qt import thread_worker
import time

@magic_factory
def make_data_preprocessing(
    viewer: "napari.viewer.Viewer",
    input: Image,
    input_xy_pixelsize: float,
    input_z_pixelsize: float,
    reference_xy_pixelsize: float,
    reference_z_pixelsize: float):
    """Generates widget for adjusting resolution of input and reference images

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        Napari viewer allows addition of layer once thread_worker finished
        executing
    input : napari.layers.Image
        Input image volume of which resolution will be adjusted
    input_xy_pixelsize : float
        Pixelsize in x-y-plane of input image volume
    input_z_pixelsize : float
        Pixelsize along z-axis of input image volume
    reference_xy_pixelsize : float
        Pixelsize in x-y-plane of reference image volume
    reference_z_pixelsize : float
        Pixelsize along z-axis of reference image volume
    """

    pbar = widgets.ProgressBar()
    pbar.range = (0, 0)  # unknown duration
    make_data_preprocessing.insert(0, pbar)  # add progress bar to the top of widget

    def _add_data(return_value, self=make_data_preprocessing):
        print('Adding new layer to viewer...')
        data, kwargs = return_value
        viewer.add_image(data, **kwargs)
        self.pop(0).hide()  # remove the progress bar
        print('Done!')

    def _zoom_values(xy, z, xy_ref, z_ref):
        xy_zoom = xy / xy_ref
        z_zoom = z / z_ref

        return xy_zoom, z_zoom

    @thread_worker(connect={"returned": _add_data})
    def _preprocess(input: Image,
                    input_xy_pixelsize: float,
                    input_z_pixelsize: float,
                    reference_xy_pixelsize: float,
                    reference_z_pixelsize: float):
        """Wraps scipy.ndimage.zoom to adjust pixelsize of input image

        Parameters
        ----------
        input : napari.layers.Image
            Input image volume of which resolution will be adjusted
        input_xy_pixelsize : float
            Pixelsize in x-y-plane of input image volume
        input_z_pixelsize : float
            Pixelsize along z-axis of input image volume
        reference_xy_pixelsize : float
            Pixelsize in x-y-plane of reference image volume
        reference_z_pixelsize : float
            Pixelsize along z-axis of reference image volume

        Returns
        -------
        output : numpy.ndarray
            Zoomed input array
        kwargs : dict
            A dictionary of parameters for adding output volume to
            napari viewer
        """
        start = time.time()

        xy_zoom, z_zoom = _zoom_values(input_xy_pixelsize,
                                       input_z_pixelsize,
                                       reference_xy_pixelsize,
                                       reference_z_pixelsize)
        print(f'Zooming by {xy_zoom} in x-y-plane and {z_zoom} along z-axis')

        output = ndimage.zoom(input.data, (z_zoom, xy_zoom, xy_zoom))

        elapsed = time.time() - start
        print('Finished execution after {elapsed} seconds.')

        kwargs = dict(
            name=input.name + '_preprocessed'
        )
        return (output, kwargs)

    _preprocess(input=input,
                input_xy_pixelsize=input_xy_pixelsize,
                input_z_pixelsize=input_z_pixelsize,
                reference_xy_pixelsize=reference_xy_pixelsize,
                reference_z_pixelsize=reference_z_pixelsize)
