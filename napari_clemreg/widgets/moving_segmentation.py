import napari
from magicgui import magic_factory
from napari.layers import Image, Shapes
from napari.utils.notifications import show_error
from napari.qt.threading import thread_worker


def on_init(widget):
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

    widget.z_max.changed.connect(change_z_min)
    widget.Moving_Image.changed.connect(change_z_max)
    widget.z_min.changed.connect(change_z_max_from_z_min)


@magic_factory(widget_init=on_init,
               layout='vertical',
               call_button='Segment',
               widget_header={'widget_type': 'Label',
                              'label': f'<h2 text-align="left">Moving Segmentation</h2>'},
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
               )
def moving_segmentation_widget(viewer: 'napari.viewer.Viewer',
                               widget_header,
                               Moving_Image: Image,
                               Mask_ROI: Shapes,
                               z_min,
                               z_max,
                               log_sigma,
                               log_threshold,
                               ):
    from ..clemreg.data_preprocessing import make_isotropic
    from ..clemreg.log_segmentation import log_segmentation
    from ..clemreg.mask_roi import mask_roi
    @thread_worker
    def _run_moving_thread():
        z_zoom = make_isotropic(input_image=Moving_Image)

        seg_volume = log_segmentation(input=Moving_Image,
                                      sigma=log_sigma,
                                      threshold=log_threshold)

        if len(set(seg_volume.data.ravel())) <= 1:
            return 'No segmentation'

        if Mask_ROI is not None:
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

        viewer.add_labels(return_value.data,
                          name="Moving_Segmentation")

    worker_moving = _run_moving_thread()
    worker_moving.returned.connect(_add_data)
    worker_moving.start()
