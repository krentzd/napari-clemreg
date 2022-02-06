#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from magicgui import magic_factory, widgets
from scipy import ndimage
from napari.layers import Image
from napari.qt import thread_worker
import time
from typing_extensions import Annotated

def get_pixelsize(metadata: dict):
    """ Parse pixelsizes from image metadata"""

    try:
        x_pxlsz = 1 / metadata['XResolution']
        y_pxlsz = 1 / metadata['YResolution']
    except KeyError:
        x_pxlsz = 0
        y_pxlsz = 0
        print('XResolution and YResolution not recorded in metadata')

    try:
        # Parse ImageJ Metadata to get z pixelsize
        ij_metadata = metadata['ImageDescription'].split('\n')
        ij_metadata = [i for i in ij_metadata if i not in '=']
        ij_dict = dict((k,v) for k,v in (i.rsplit('=') for i in ij_metadata))

        z_pxlsz = ij_dict['spacing']
        unit = ij_dict['unit']
    except KeyError:
        z_pxlsz = 0
        unit = 'micron'
        print('ImageJ metdata not recorded in metadata')

    return (eval(x_pxlsz) if isinstance(x_pxlsz, str) else x_pxlsz,
           eval(y_pxlsz) if isinstance(y_pxlsz, str) else y_pxlsz,
           eval(z_pxlsz) if isinstance(z_pxlsz, str) else z_pxlsz,
           unit)

def on_init(widget):
    """Initializes widget layout and updates widget layout according to user input."""

    def change_moving_pixelsize(input_image: Image):
        x_pxlsz, y_pxlsz, z_pxlsz, unit = get_pixelsize(input_image.metadata)
        print(x_pxlsz, y_pxlsz, z_pxlsz, unit)
        if widget.unit.value == 'micron' and unit == 'micron':
            widget.moving_xy_pixelsize.value = x_pxlsz
            widget.moving_z_pixelsize.value = z_pxlsz

        elif widget.unit.value == 'nm' and unit == 'micron':
            widget.moving_xy_pixelsize.value = x_pxlsz * 1000
            widget.moving_z_pixelsize.value = z_pxlsz * 1000
            widget.moving_xy_pixelsize.max = x_pxlsz * 1000
            widget.moving_z_pixelsize.max = z_pxlsz * 1000

        elif widget.unit.value == 'nm' and unit == 'nm':
            widget.moving_xy_pixelsize.value = x_pxlsz
            widget.moving_z_pixelsize.value = z_pxlsz

        elif widget.unit.value == 'micron' and unit == 'nm':
            widget.moving_xy_pixelsize.value = x_pxlsz / 1000
            widget.moving_z_pixelsize.value = z_pxlsz / 1000

    def change_fixed_pixelsize(input_image: Image):
        x_pxlsz, y_pxlsz, z_pxlsz, unit = get_pixelsize(input_image.metadata)
        # widget.fixed_xy_pixelsize.value = x_pxlsz
        # widget.fixed_z_pixelsize.value = z_pxlsz

        if widget.unit.value == 'micron' and unit == 'micron':
            widget.fixed_xy_pixelsize.value = x_pxlsz
            widget.fixed_z_pixelsize.value = z_pxlsz

        elif widget.unit.value == 'nm' and unit == 'micron':
            widget.fixed_xy_pixelsize.value = x_pxlsz * 1000
            widget.fixed_z_pixelsize.value = z_pxlsz * 1000
            widget.fixed_xy_pixelsize.max = x_pxlsz * 1000
            widget.fixed_z_pixelsize.max = z_pxlsz * 1000

        elif widget.unit.value == 'nm' and unit == 'nm':
            widget.fixed_xy_pixelsize.value = x_pxlsz
            widget.fixed_z_pixelsize.value = z_pxlsz

        elif widget.unit.value == 'micron' and unit == 'nm':
            widget.fixed_xy_pixelsize.value = x_pxlsz / 1000
            widget.fixed_z_pixelsize.value = z_pxlsz / 1000

    def adjust_values_to_unit(unit: str):
        if unit == 'nm':
            for x in ['moving_xy_pixelsize', 'moving_z_pixelsize', 'fixed_xy_pixelsize', 'fixed_z_pixelsize']:
                current_value = getattr(getattr(widget, x), 'value')
                setattr(getattr(widget, x), 'max', current_value * 1000)
                setattr(getattr(widget, x), 'value', current_value * 1000)
        elif unit == 'micron':
            for x in ['moving_xy_pixelsize', 'moving_z_pixelsize', 'fixed_xy_pixelsize', 'fixed_z_pixelsize']:
                current_value = getattr(getattr(widget, x), 'value')
                setattr(getattr(widget, x), 'value', current_value / 1000)

    widget.moving.changed.connect(change_moving_pixelsize)
    widget.fixed.changed.connect(change_fixed_pixelsize)
    widget.unit.changed.connect(adjust_values_to_unit)

@magic_factory(widget_init=on_init, layout='vertical', call_button="Preprocess")
def make_data_preprocessing(
    viewer: "napari.viewer.Viewer",
    moving: Image,
    fixed: Image,
    unit: Annotated[str, {"choices": ["nm", "micron"]}],
    moving_xy_pixelsize: float,
    moving_z_pixelsize: float,
    fixed_xy_pixelsize: float,
    fixed_z_pixelsize: float):
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

        input_arr = input.data
        if len(input_arr.shape) == 4:
            output = [ndimage.zoom(input_arr[c], (z_zoom, xy_zoom, xy_zoom)) for c in range(4)]
            output = np.squeeze(np.stack(output))
        else:
            output = ndimage.zoom(input_arr, (z_zoom, xy_zoom, xy_zoom))

        elapsed = time.time() - start
        print('Finished execution after {elapsed} seconds.')

        kwargs = dict(
            name=input.name + '_preprocessed'
        )
        return (output, kwargs)

    _preprocess(input=moving,
                input_xy_pixelsize=moving_xy_pixelsize,
                input_z_pixelsize=moving_z_pixelsize,
                reference_xy_pixelsize=fixed_xy_pixelsize,
                reference_z_pixelsize=fixed_z_pixelsize)
