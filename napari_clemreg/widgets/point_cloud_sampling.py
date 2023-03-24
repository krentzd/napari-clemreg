import napari
from magicgui import magic_factory
from napari.layers import Labels
from napari.utils.notifications import show_error
from napari.qt.threading import thread_worker
from ..clemreg.point_cloud_sampling import point_cloud_sampling


@magic_factory(layout='vertical',
               call_button='Sample',
               widget_header={'widget_type': 'Label',
                              'label': f'<h2 text-align="left">Point Cloud Sampling</h2>'},
               point_cloud_sampling_frequency={'label': 'Sampling Frequency',
                                               'widget_type': 'SpinBox',
                                               'min': 1, 'max': 100, 'step': 1,
                                               'value': 5},
               point_cloud_sigma={'label': 'Sigma',
                                  'widget_type': 'FloatSpinBox',
                                  'min': 0, 'max': 10, 'step': 0.1,
                                  'value': 1.0},
               )
def point_cloud_sampling_widget(viewer: 'napari.viewer.Viewer',
                                widget_header,
                                Moving_Segmentation: Labels,
                                Fixed_Segmentation: Labels,
                                point_cloud_sampling_frequency,
                                point_cloud_sigma,
                                ):
    @thread_worker
    def _run_point_cloud_sampling_thread():
        moving_point_cloud = point_cloud_sampling(input=Moving_Segmentation,
                                                  sampling_frequency=point_cloud_sampling_frequency / 100,
                                                  sigma=point_cloud_sigma)

        fixed_point_cloud = point_cloud_sampling(input=Fixed_Segmentation,
                                                 sampling_frequency=point_cloud_sampling_frequency / 100,
                                                 sigma=point_cloud_sigma)

        return moving_point_cloud, fixed_point_cloud

    def _add_data(return_value):
        mp = return_value[0]
        fp = return_value[1]

        viewer.add_points(mp.data,
                          name='moving_points',
                          face_color='red')
        viewer.add_points(fp.data,
                          name='fixed_points',
                          face_color='blue')

    worker_pc_sampling = _run_point_cloud_sampling_thread()
    worker_pc_sampling.returned.connect(_add_data)
    worker_pc_sampling.start()
