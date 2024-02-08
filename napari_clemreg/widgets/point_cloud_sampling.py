import napari
from magicgui import magic_factory
from napari.layers import Labels, Image, Layer
from napari.qt.threading import thread_worker
from ..clemreg.on_init_specs import specs

def on_init(widget):
    from ..clemreg.data_preprocessing import get_pixelsize

    def change_moving_pixelsize(input_label: Labels):
        moving_image_pixelsize_xy, __, moving_image_pixelsize_z, unit = get_pixelsize(input_label.metadata)

        if unit in ['nanometer', 'nm', 'um', 'micron', 'micrometer']:
            if unit == 'um' or unit == 'micron':
                unit = 'micrometer'
            elif unit == 'nm':
                unit = 'nanometer'
        else:
            unit = 'nanometer'

        widget.moving_image_pixelsize_xy.value = str(moving_image_pixelsize_xy) + str(unit)
        widget.moving_image_pixelsize_z.value = str(moving_image_pixelsize_z) + str(unit)

    def change_fixed_pixelsize(input_label: Labels):
        fixed_image_pixelsize_xy, __, fixed_image_pixelsize_z, unit = get_pixelsize(input_label.metadata)

        if unit in ['nanometer', 'nm', 'um', 'micron', 'micrometer']:
            if unit == 'um' or unit == 'micron':
                unit = 'micrometer'
            elif unit == 'nm':
                unit = 'nanometer'
        else:
            unit = 'nanometer'

        widget.fixed_image_pixelsize_xy.value = str(fixed_image_pixelsize_xy) + str(unit)
        widget.fixed_image_pixelsize_z.value = str(fixed_image_pixelsize_z) + str(unit)

    widget.Moving_Segmentation.changed.connect(change_moving_pixelsize)
    widget.Fixed_Segmentation.changed.connect(change_fixed_pixelsize)


@magic_factory(widget_init=on_init, layout='vertical', call_button='Sample',
               widget_header={'widget_type': 'Label',
                              'label': f'<h2 text-align="left">Point Cloud Sampling</h2>'},

                Moving_Segmentation=specs['Moving_Segmentation'],
                moving_image_pixelsize_xy=specs['moving_image_pixelsize_xy'],
                moving_image_pixelsize_z=specs['moving_image_pixelsize_z'],

                Fixed_Segmentation=specs['Fixed_Segmentation'],
                fixed_image_pixelsize_xy=specs['fixed_image_pixelsize_xy'],
                fixed_image_pixelsize_z=specs['fixed_image_pixelsize_z'],

                point_cloud_sampling_frequency=specs['point_cloud_sampling_frequency'],
                voxel_size=specs['registration_voxel_size'],
                point_cloud_sigma=specs['point_cloud_sigma']
               )
def point_cloud_sampling_widget(viewer: 'napari.viewer.Viewer',
                                widget_header,

                                Moving_Segmentation: Labels,
                                moving_image_pixelsize_xy,
                                moving_image_pixelsize_z,

                                Fixed_Segmentation: Labels,
                                fixed_image_pixelsize_xy,
                                fixed_image_pixelsize_z,

                                point_cloud_sampling_frequency,
                                voxel_size,
                                point_cloud_sigma,
                                ):
    """
    This function point samples the segmentation of produced
    by the Fixed and Moving segmentation widgets.

    Parameters
    ----------
    viewer : 'napari.viewer.Viewer'
        napari Viewer
    widget_header : str
        The widget header
    Moving_Segmentation
        The moving light microscopy image
    Fixed_Segmentation
        The fixed electron microscopy image
    point_cloud_sampling_frequency : int
        Frequency of cloud sampling
    point_cloud_sigma : float
        ?

    Returns
    -------
        Two Points layers sampling the inputted moving
        and fixed image
    """
    from ..clemreg.widget_components import run_point_cloud_sampling

    @thread_worker
    def _run_point_cloud_sampling_thread(**kwargs):

        moving_point_cloud, fixed_point_cloud = run_point_cloud_sampling(**kwargs)

        return moving_point_cloud, fixed_point_cloud

    def _add_data(return_value):
        moving_points, fixed_points = return_value
        viewer.add_layer(moving_points)
        viewer.add_layer(fixed_points)

    worker_pc_sampling = _run_point_cloud_sampling_thread(Moving_Segmentation=Moving_Segmentation,
                                                          Fixed_Segmentation=Fixed_Segmentation,
                                                          moving_image_pixelsize_xy=moving_image_pixelsize_xy,
                                                          moving_image_pixelsize_z=moving_image_pixelsize_z,
                                                          fixed_image_pixelsize_xy=fixed_image_pixelsize_xy,
                                                          fixed_image_pixelsize_z=fixed_image_pixelsize_z,
                                                          point_cloud_sampling_frequency=point_cloud_sampling_frequency,
                                                          voxel_size=voxel_size,
                                                          point_cloud_sigma=point_cloud_sigma)
    worker_pc_sampling.returned.connect(_add_data)
    worker_pc_sampling.start()
