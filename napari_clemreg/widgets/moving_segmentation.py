import napari
from magicgui import magic_factory
from napari.layers import Image, Shapes
from napari.utils.notifications import show_error
from napari.qt.threading import thread_worker

def on_init(widget):
    from ..clemreg.data_preprocessing import get_pixelsize

    custom_z_zom_settings = ['z_zoom_value']
    filter_segmentation_settings = ['filter_size']

    standard_settings = ['Moving_Image',
                         'moving_image_pixelsize_xy',
                         'moving_image_pixelsize_z',
                         'Mask_ROI',
                         'log_sigma',
                         'log_threshold',
                         'custom_z_zoom',
                         'filter_segmentation']
    advanced_settings = ['z_min',
                         'z_max',
                         'z_zoom_value',
                         'filter_size']

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

    def reveal_z_min_and_z_max():
        if len(widget.Mask_ROI.choices) > 0:
            for x in ['z_min', 'z_max']:
                setattr(getattr(widget, x), 'visible', True)
        else:
            for x in ['z_min', 'z_max']:
                setattr(getattr(widget, x), 'visible', False)

    def change_moving_pixelsize(input_image: Image):
        moving_image_pixelsize_xy, __, moving_image_pixelsize_z, unit = get_pixelsize(input_image.metadata)

        if unit in ['nanometer', 'nm', 'um', 'micron', 'micrometer']:
            if unit == 'um' or unit == 'micron':
                unit = 'micrometer'
            elif unit == 'nm':
                unit = 'nanometer'
        else:
            unit = 'nanometer'

        widget.moving_image_pixelsize_xy.value = str(moving_image_pixelsize_xy) + str(unit)
        widget.moving_image_pixelsize_z.value = str(moving_image_pixelsize_z) + str(unit)

    def toggle_custom_z_zoom(custom_z_zoom: bool):
        if custom_z_zoom:
            for x in custom_z_zom_settings:
                setattr(getattr(widget, x), 'visible', True)
        else:
            for x in custom_z_zom_settings:
                setattr(getattr(widget, x), 'visible', False)

    def toggle_filter_segmentation(filter_segmentation: bool):
        if filter_segmentation:
            for x in filter_segmentation_settings:
                setattr(getattr(widget, x), 'visible', True)
        else:
            for x in filter_segmentation_settings:
                setattr(getattr(widget, x), 'visible', False)

    widget.z_max.changed.connect(change_z_min)
    widget.Moving_Image.changed.connect(change_z_max)
    widget.z_min.changed.connect(change_z_max_from_z_min)
    widget.Mask_ROI.changed.connect(reveal_z_min_and_z_max)
    widget.Moving_Image.changed.connect(change_moving_pixelsize)
    widget.custom_z_zoom.changed.connect(toggle_custom_z_zoom)
    widget.filter_segmentation.changed.connect(toggle_filter_segmentation)


@magic_factory(widget_init=on_init,
               layout='vertical',
               call_button='Segment',
               widget_header={'widget_type': 'Label',
                              'label': f'<h2 text-align="left">Fluorescence Microscopy Segmentation</h2>'},
               log_sigma={'label': 'Sigma',
                          'widget_type': 'FloatSpinBox',
                          'min': 0.5, 'max': 20, 'step': 0.5,
                          'value': 3},
               log_threshold={'label': 'Threshold',
                              'widget_type': 'FloatSpinBox',
                              'min': 0, 'max': 20, 'step': 0.1,
                              'value': 1.2},
               z_min={'widget_type': 'SpinBox',
                      'label': 'Minimum z value for masking',
                      "min": 0, "max": 10, "step": 1,
                      'value': 0},
               z_max={'widget_type': 'SpinBox',
                      'label': 'Maximum z value for masking',
                      "min": 0, "max": 10, "step": 1,
                      'value': 0},
               Moving_Image={'label': 'Fluorescence Microscopy Image (FM)'},
               moving_image_pixelsize_xy={'label': 'Pixel size (xy)',
                                               'widget_type': 'QuantityEdit',
                                               'value': '0 nanometer'
                                               },
               moving_image_pixelsize_z={'label': 'Pixel size (z)',
                                               'widget_type': 'QuantityEdit',
                                               'value': '0 nanometer'
                                               },
               z_zoom_value={'label': 'Z interpolation factor',
                           'widget_type': 'FloatSpinBox',
                           'min': 0, 'step': 0.01,
                           'value': 1},
               custom_z_zoom={'text': 'Custom z interpolation factor',
                              'widget_type': 'CheckBox',
                              'value': False},
               filter_segmentation={'text': 'Apply size filter to segmentation',
                                'widget_type': 'CheckBox',
                                'value': False},
               filter_size={'label': 'Filter threshold (as percentile of size)',
                                'widget_type': 'SpinBox',
                                'min': 0, 'max': 100, 'step': 1,
                                'value': 50},
               )
def moving_segmentation_widget(viewer: 'napari.viewer.Viewer',
                               widget_header,
                               Moving_Image: Image,

                               moving_image_pixelsize_xy,
                               moving_image_pixelsize_z,

                               Mask_ROI: Shapes,
                               z_min,
                               z_max,

                               log_sigma,
                               log_threshold,
                               custom_z_zoom,
                               z_zoom_value,
                               filter_segmentation,
                               filter_size,
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
    from ..clemreg.data_preprocessing import make_isotropic
    from ..clemreg.log_segmentation import log_segmentation
    from ..clemreg.mask_roi import mask_roi, mask_area

    @thread_worker
    def _run_moving_thread():
        print('Starting LoG segmentation...')
        if not custom_z_zoom:
            # Need to verify units are the same in xy and z
            z_zoom_value = moving_image_pixelsize_z.magnitude / moving_image_pixelsize_xy.magnitude

        z_zoom = make_isotropic(input_image=Moving_Image, z_zoom_value=z_zoom_value if custom_z_zoom else None)

        seg_volume = log_segmentation(input=Moving_Image,
                                      sigma=log_sigma,
                                      threshold=log_threshold)

        if filter_segmentation:
            seg_volume = filter_binary_segmentation(input=seg_volume,
                                                    percentile=filter_size)

        if len(set(seg_volume.data.ravel())) <= 1:
            return 'No segmentation'

        if Mask_ROI is not None:
            print('Applying Mask ROI...')
            seg_volume_mask = mask_roi(input=seg_volume,
                                       crop_mask=Mask_ROI,
                                       z_min=int(z_min * z_zoom),
                                       z_max=int(z_max * z_zoom))
        else:
            seg_volume_mask = seg_volume

        return seg_volume_mask

    def _add_data(return_value):
        if isinstance(return_value, str):
            show_error('WARNING: No mitochondria in Fixed Image')
            return

        viewer.add_labels(return_value.data.astype(np.int64),
                          name="Moving_Segmentation")

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

    worker_moving = _run_moving_thread()
    worker_moving.returned.connect(_add_data)
    worker_moving.start()
