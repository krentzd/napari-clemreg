import napari
from magicgui import magic_factory
from napari.layers import Image, Shapes, Labels
from napari.utils.notifications import show_error
from napari.qt.threading import thread_worker

from ..clemreg.on_init_specs import specs
from ..clemreg._qt_layout import group_into_collapsible, wrap_in_scroll_area

def on_init(widget):
    from ..clemreg.data_preprocessing import get_pixelsize

    filter_segmentation_settings = ['filter_size_lower', 'filter_size_upper']

    for x in ['z_min', 'z_max'] + filter_segmentation_settings:
        setattr(getattr(widget, x), 'visible', False)

    # Grouped into its own collapsible section for consistency with the
    # other widgets in this plugin, even though this one is small enough
    # that the old flat show/hide wasn't causing the lag/height problems
    # run_registration.py had -- see the comment there for the full story.
    group_into_collapsible(widget, 'Size Filter',
                           ['filter_segmentation', 'filter_size_lower', 'filter_size_upper'])

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

    # Qt widgets don't auto-shrink once they've been made larger -- hiding
    # child widgets recomputes the container's sizeHint correctly, but
    # nothing makes the actual widget follow it back down without an
    # explicit adjustSize() call (confirmed directly on the equivalent bug
    # in run_registration.py's "Parameters custom" toggle).
    def _shrink_to_fit():
        widget.native.adjustSize()

    def reveal_z_min_and_z_max():
        if len(widget.Mask_ROI.choices) > 0:
            for x in ['z_min', 'z_max']:
                setattr(getattr(widget, x), 'visible', True)
        else:
            for x in ['z_min', 'z_max']:
                setattr(getattr(widget, x), 'visible', False)
        _shrink_to_fit()

    def toggle_filter_segmentation(filter_segmentation: bool):
        if filter_segmentation:
            for x in filter_segmentation_settings:
                setattr(getattr(widget, x), 'visible', True)
        else:
            for x in filter_segmentation_settings:
                setattr(getattr(widget, x), 'visible', False)
        _shrink_to_fit()

    widget.z_max.changed.connect(change_z_min)
    widget.Moving_Image.changed.connect(change_z_max)
    widget.z_min.changed.connect(change_z_max_from_z_min)
    widget.Mask_ROI.changed.connect(reveal_z_min_and_z_max)
    widget.filter_segmentation.changed.connect(toggle_filter_segmentation)
    # Change state of call button once clicked and allow clicking when function finishes running

@magic_factory(widget_init=on_init, layout='vertical', call_button='Segment',
               widget_header={'widget_type': 'Label',
                              'label': f'<h2 text-align="left">Fluorescence Microscopy Segmentation</h2>'},

               Moving_Image=specs['Moving_Image'],

               z_min=specs['z_min'],
               z_max=specs['z_max'],

               log_sigma=specs['log_sigma'],
               log_threshold=specs['log_threshold'],

               filter_segmentation=specs['filter_segmentation'],
               filter_size_lower=specs['filter_size_lower'],
               filter_size_upper=specs['filter_size_upper']
               )
def moving_segmentation_widget(viewer: 'napari.viewer.Viewer',
                               widget_header,
                               Moving_Image: Image,

                               Mask_ROI: Shapes,
                               z_min,
                               z_max,

                               log_sigma,
                               log_threshold,

                               filter_segmentation,
                               filter_size_lower,
                               filter_size_upper,
                               ):
    """
    This function performs segmentation of the mitochondria
    of the inputted Light Microscopy Image. This is performed
    using adaptive thresholding.

    One can apply a mask to the LM image which restrict thresholding
    to a user defined region.

    One Can also define the layers in the Z stack to be segmented.

    Parameters
    ----------
    viewer : 'Viewer'
        napari viewer
    widget_header : str
        Heading of the widget
    Moving_Image :
        The LM image to be segmented
    Mask_ROI :
        Shapes Layer of the region to be segmented
    z_min : int
        Min Z layer to be segmented in z stack
    z_max : int
        Max Z layer to be segmented in z stack
    log_sigma : float
        Sigma value for 1D gaussian filter to be applied oto image before segmentation
    log_threshold : int
        Threshold value to apply to image


    Returns
    -------
        Thresholding of the inputted light microscopy image using
        adaptive thresholding.
    """
    import numpy as np
    from ..clemreg.widget_components import run_moving_segmentation
    from ..clemreg.mask_roi import mask_area
    
    def _add_data(return_value):
        viewer.add_layer(return_value)

    def _errored(exc):
        # Includes NoSegmentationError when nothing is found (#50).
        show_error(f'FM segmentation failed: {exc}')

    # Handlers passed at creation (not connected afterwards), so superqt
    # doesn't also re-raise the error; the worker starts itself.
    @thread_worker(connect={'returned': _add_data, 'errored': _errored})
    def _run_segmentation_thread(**kwargs):
        seg_volume_mask = run_moving_segmentation(**kwargs)
        seg_volume_mask = Labels(seg_volume_mask.astype(np.uint32),
                                 name='FM_segmentation',
                                 metadata=Moving_Image.metadata)

        return seg_volume_mask

    if Moving_Image is None:
        show_error("WARNING: You have not inputted both a fixed and moving image")
        return

    if len(Moving_Image.data.shape) != 3:
        show_error("WARNING: Your moving_image must be 3D, you're current input has a shape of {}".format(
            Moving_Image.data.shape))
        return
    elif len(Moving_Image.data.shape) == 3 and (Moving_Image.data.shape[2] == 3 or Moving_Image.data.shape[2] == 4):
        show_error("WARNING: YOUR moving_image is RGB, your input must be grayscale and 3D")
        return

    if Mask_ROI is not None:
        if len(Mask_ROI.data) != 1:
            show_error("WARNING: You must input only 1 Mask ROI, you have inputted {}.".format(len(Mask_ROI.data)))
            return
        if mask_area(Mask_ROI.data[0][:, 1], Mask_ROI.data[0][:, 2]) > Moving_Image.data.shape[1] * \
                Moving_Image.data.shape[2]:
            show_error("WARNING: Your mask size exceeds the size of the image.")
            return

    _run_segmentation_thread(Moving_Image=Moving_Image,
                             Mask_ROI=Mask_ROI,
                             z_min=z_min,
                             z_max=z_max,
                             log_sigma=log_sigma,
                             log_threshold=log_threshold,
                             filter_segmentation=filter_segmentation,
                             filter_size_lower=filter_size_lower,
                             filter_size_upper=filter_size_upper)


def moving_segmentation_dock_widget(napari_viewer: 'napari.viewer.Viewer' = None):
    """The actual napari-docked widget -- wraps moving_segmentation_widget()
    in a QScrollArea for a bounded height/width, matching AIoD's own
    napari plugin's approach (verified against its real source).

    napari_viewer defaults to None and is otherwise unused -- confirmed
    in napari's own source that viewer-injection-by-parameter-name only
    applies to class-based widgets, not plain functions like this one
    (see the longer note in run_registration.make_run_registration_widget).
    """
    # magic_factory's __call__ treats kwargs as widget-option overrides,
    # not runtime values, so call with no args and let its own
    # Viewer-typed-parameter auto-injection resolve the current viewer.
    gui = moving_segmentation_widget()
    return wrap_in_scroll_area(gui)
