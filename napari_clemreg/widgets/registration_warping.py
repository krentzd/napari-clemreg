import napari
from magicgui import magic_factory
from napari.layers import Points, Image
from napari.qt.threading import thread_worker
from napari.layers.utils._link_layers import link_layers
from napari.utils.notifications import show_error
import pint

from ..clemreg.on_init_specs import specs
from ..clemreg._qt_layout import wrap_in_scroll_area


def on_init(widget):

    # Only point clouds from Point Cloud Sampling carry 'pxlsz'. Any other
    # Points layer can be selected too (napari auto-selects new layers), so
    # skip those rather than raise KeyError.
    def change_moving_pixelsize(input_points: Points):
        if input_points is None or 'pxlsz' not in input_points.metadata:
            return
        moving_image_pixelsize_z, moving_image_pixelsize_xy = input_points.metadata['pxlsz']

        widget.moving_image_pixelsize_xy.value = str(moving_image_pixelsize_xy) + 'nanometer'
        widget.moving_image_pixelsize_z.value = str(moving_image_pixelsize_z) + 'nanometer'

    def change_fixed_pixelsize(input_points: Points):
        if input_points is None or 'pxlsz' not in input_points.metadata:
            return
        fixed_image_pixelsize_z, fixed_image_pixelsize_xy = input_points.metadata['pxlsz']

        widget.fixed_image_pixelsize_xy.value = str(fixed_image_pixelsize_xy) + 'nanometer'
        widget.fixed_image_pixelsize_z.value = str(fixed_image_pixelsize_z) + 'nanometer'

    widget.Moving_Points.changed.connect(change_moving_pixelsize)
    widget.Fixed_Points.changed.connect(change_fixed_pixelsize)

    # A layer already selected at construction time never fires `changed`,
    # so seed the pixel sizes once from what's shown.
    change_moving_pixelsize(widget.Moving_Points.value)
    change_fixed_pixelsize(widget.Fixed_Points.value)


@magic_factory(widget_init=on_init, layout='vertical',
               call_button='Register',
               widget_header={'widget_type': 'Label',
                              'label': f'<h2 text-align="left">Point Cloud Registration</h2>'},
               widget_header_2={'widget_type': 'Label',
                            'label': f'<h2 text-align="middle">and Image Warping</h2>'},

               Moving_Image=specs['Moving_Image'],
               moving_image_pixelsize_xy=specs['moving_image_pixelsize_xy'],
               moving_image_pixelsize_z=specs['moving_image_pixelsize_z'],

               Fixed_Image=specs['Fixed_Image'],
               fixed_image_pixelsize_xy=specs['fixed_image_pixelsize_xy'],
               fixed_image_pixelsize_z=specs['fixed_image_pixelsize_z'],

               Moving_Points=specs['Moving_Points'],
               Fixed_Points=specs['Fixed_Points'],

               registration_algorithm=specs['registration_algorithm'],

               registration_header={'widget_type': 'Label',
                                    'label': f'<h3 text-align="left">Point Cloud Registration</h3>'},

               registration_max_iterations=specs['registration_max_iterations'],

               warping_header={'widget_type': 'Label',
                               'label': f'<h3 text-align="left">Image Warping</h3>'},

               warping_interpolation_order=specs['warping_interpolation_order'],
               warping_approximate_grid=specs['warping_approximate_grid'],
               warping_sub_division_factor=specs['warping_sub_division_factor'],
               warping_output_resolution=specs['warping_output_resolution'],

               registration_direction=specs['registration_direction'],
               )
def registration_warping_widget(viewer: 'napari.viewer.Viewer',
                                widget_header,
                                widget_header_2,

                                Moving_Image: Image,
                                moving_image_pixelsize_xy,
                                moving_image_pixelsize_z,

                                Fixed_Image: Image,
                                fixed_image_pixelsize_xy,
                                fixed_image_pixelsize_z,

                                Moving_Points: Points,
                                Fixed_Points: Points,

                                registration_header,
                                registration_algorithm,
                                registration_max_iterations,

                                warping_header,
                                warping_interpolation_order,
                                warping_approximate_grid,
                                warping_sub_division_factor,
                                warping_output_resolution,

                                registration_direction
                                ):
    """
    This widget registers the moving and fixed points and then uses
    these registered points to warp the light microscopy moving image
    in the process aligning the fixed and moving image.

    Parameters
    ----------
    viewer : 'napari.viewer.Viewer'
        napari Viewer
    widget_header : str
        Widget heading
    Moving_Image : Image
        The moving image to be warped
    Fixed_Image : Image
        The fixed image to reference the moving warping to.
    Moving_Points : Points
        The sampled moving points of the moving light microscopy image.
    Fixed_Points : Points
        The sampled fixed points of the fixed electron microscopy image
    registration_header : str
        The registration heading
    registration_algorithm : 'magicgui.widgets.ComboBox'
        The algorithm to do the registration of the moving and fixed points.
    registration_max_iterations : int
        Maximum number of CPD iterations
    warping_header : str
        Warping headings
    warping_interpolation_order : int
        ?
    warping_approximate_grid : int
        ?
    warping_sub_division_factor : int
        ?
    warping_output_resolution : str
        Pixel grid to save the warped output at -- native FM resolution
        (default) or EM's own pixel grid (legacy behaviour).

    Returns
    -------
        napari Image layer containing the warping of the moving image to the
        fixed image.
    """
    ureg = pint.UnitRegistry()

    pxlsz_moving = (moving_image_pixelsize_z.to_preferred([ureg.nanometers]).magnitude, moving_image_pixelsize_xy.to_preferred([ureg.nanometers]).magnitude)
    pxlsz_fixed = (fixed_image_pixelsize_z.to_preferred([ureg.nanometers]).magnitude, fixed_image_pixelsize_xy.to_preferred([ureg.nanometers]).magnitude)

    Moving_Points.metadata['pxlsz'] = pxlsz_moving
    Fixed_Points.metadata['pxlsz'] = pxlsz_fixed

    @thread_worker
    def _registration_thread(**kwargs):
        from ..clemreg.widget_components import run_point_cloud_registration_and_warping
        warp_outputs, transformed = run_point_cloud_registration_and_warping(**kwargs)

        return warp_outputs, transformed

    def _add_data(return_value):
        image_layers, points_layer = return_value
        viewer.add_layer(points_layer)

        layers = []
        for image_layer in image_layers:
            viewer.add_layer(image_layer)
            layers.append(viewer.layers[image_layer.name])
        link_layers(layers)

    if Moving_Image is None or Fixed_Image is None:
        show_error("WARNING: You have not inputted both a fixed and moving image")
        return

    if len(Moving_Image.data.shape) != 3:
        show_error("WARNING: Your moving_image must be 3D, you're current input has a shape of {}".format(
            Moving_Image.data.shape))
        return
    elif len(Moving_Image.data.shape) == 3 and (Moving_Image.data.shape[2] == 3 or Moving_Image.data.shape[2] == 4):
        show_error("WARNING: YOUR moving_image is RGB, your input must be grayscale and 3D")
        return

    if len(Fixed_Image.data.shape) != 3:
        show_error("WARNING: Your Fixed_Image must be 3D, you're current input has a shape of {}".format(
            Moving_Image.data.shape))
        return
    elif len(Fixed_Image.data.shape) == 3 and (Fixed_Image.data.shape[2] == 3 or Fixed_Image.data.shape[2] == 4):
        show_error("WARNING: YOUR fixed_image is RGB, your input must be grayscale and 3D")
        return

    worker_registration = _registration_thread(Moving_Points=Moving_Points,
                                               Fixed_Points=Fixed_Points,
                                               Moving_Image=Moving_Image,
                                               Fixed_Image=Fixed_Image,
                                               registration_algorithm=registration_algorithm,
                                               registration_max_iterations=registration_max_iterations,
                                               warping_interpolation_order=warping_interpolation_order,
                                               warping_approximate_grid=warping_approximate_grid,
                                               warping_sub_division_factor=warping_sub_division_factor,
                                               warping_output_resolution=warping_output_resolution,
                                               registration_direction=registration_direction)
    worker_registration.returned.connect(_add_data)
    worker_registration.start()


def registration_warping_dock_widget(napari_viewer: 'napari.viewer.Viewer' = None):
    """The actual napari-docked widget -- wraps registration_warping_widget()
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
    gui = registration_warping_widget()
    return wrap_in_scroll_area(gui)
