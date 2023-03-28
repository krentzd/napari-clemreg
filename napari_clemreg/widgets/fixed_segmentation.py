import napari
from magicgui import magic_factory
from napari.layers import Image
from napari.utils.notifications import show_error
from napari.qt.threading import thread_worker


@magic_factory(layout='vertical',
               call_button='Segment',
               widget_header={'widget_type': 'Label',
                              'label': f'<h2 text-align="left">MitoNet Image Segmentation</h2>'},
               em_seg_axis={'text': 'Prediction Across Three Axis',
                            'widget_type': 'CheckBox',
                            'value': False},
               )
def fixed_segmentation_widget(viewer: 'napari.viewer.Viewer',
                              widget_header,
                              Fixed_Image: Image,
                              em_seg_axis: bool
                              ):
    """
    This widget takes an EM image as input and performs
    segmentation of the mitochondria on it using the
    Empanada MitoNet model.

    Parameters
    ----------
    viewer :
        napari viewer
    widget_header : str
        Heading of the widget
    Fixed_Image :
        The EM Image
    em_seg_axis :
        Option to run segmentation across three axis

    Returns
    -------
    napari Labels layer containing the segmentation of mitochondria
    produced by the MitoNet model.

    """
    from ..clemreg.empanada_segmentation import empanada_segmentation

    @thread_worker
    def _run_fixed_thread():
        seg_volume = empanada_segmentation(input=Fixed_Image.data,
                                           axis_prediction=em_seg_axis)

        if len(set(seg_volume.ravel())) <= 1:
            return 'No segmentation'

        return seg_volume

    def _add_data(return_value):
        if isinstance(return_value, str):
            show_error('WARNING: No mitochondria in Fixed Image')
            return

        viewer.add_labels(return_value,
                          name="Fixed_Segmentation")

    worker_fixed = _run_fixed_thread()
    worker_fixed.returned.connect(_add_data)
    worker_fixed.start()
