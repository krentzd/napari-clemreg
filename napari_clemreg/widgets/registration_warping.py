import napari
from magicgui import magic_factory
from napari.layers import Points, Image
from napari.qt.threading import thread_worker
from napari.layers.utils._link_layers import link_layers
from napari.utils.notifications import show_error

@magic_factory(layout='vertical',
               call_button='Register',
               widget_header={'widget_type': 'Label',
                              'label': f'<h2 text-align="left">Point Cloud Registration</h2>'},
               widget_header_2={'widget_type': 'Label',
                            'label': f'<h2 text-align="middle">and Image Warping</h2>'},
               registration_algorithm={'label': 'Registration Algorithm',
                                       'widget_type': 'ComboBox',
                                       'choices': ["BCPD", "Rigid CPD", "Affine CPD"],
                                       'value': 'Rigid CPD',
                                       'tooltip': 'Speed: Rigid CPD > Affine CPD > BCPD'},

               registration_header={'widget_type': 'Label',
                                    'label': f'<h3 text-align="left">Point Cloud Registration</h3>'},

               registration_max_iterations={'label': 'Maximum Iterations',
                                            'widget_type': 'SpinBox',
                                            'min': 1, 'max': 1000, 'step': 1,
                                            'value': 50},

               warping_header={'widget_type': 'Label',
                               'label': f'<h3 text-align="left">Image Warping</h3>'},
               warping_interpolation_order={'label': 'Interpolation Order',
                                            'widget_type': 'SpinBox',
                                            'min': 0, 'max': 5, 'step': 1,
                                            'value': 1},
               warping_approximate_grid={'label': 'Approximate Grid',
                                         'widget_type': 'SpinBox',
                                         'min': 1, 'max': 10, 'step': 1,
                                         'value': 5},
               warping_sub_division_factor={'label': 'Sub-division Factor',
                                            'widget_type': 'SpinBox',
                                            'min': 1, 'max': 10, 'step': 1,
                                            'value': 1},

               registration_direction={'label': 'Registration direction',
                                               'widget_type': 'RadioButtons',
                                               'choices': [u'FM \u2192 EM', u'EM \u2192 FM'],
                                               'value': u'FM \u2192 EM'
                                               },

               Moving_Image={'label': 'Fluorescence Microscopy (FM) Image'},
               Fixed_Image={'label': 'Electron Microscopy (EM) Image'},
               Moving_Points={'label': 'Fluorescence Microscopy (FM) Point Cloud'},
               Fixed_Points={'label': 'Electron Microscopy (EM) Point Cloud'},
               )
def registration_warping_widget(viewer: 'napari.viewer.Viewer',
                                widget_header,
                                widget_header_2,

                                Moving_Image: Image,
                                Fixed_Image: Image,
                                Moving_Points: Points,
                                Fixed_Points: Points,

                                registration_header,
                                registration_algorithm,
                                registration_max_iterations,

                                warping_header,
                                warping_interpolation_order,
                                warping_approximate_grid,
                                warping_sub_division_factor,

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

    Returns
    -------
        napari Image layer containing the warping of the moving image to the
        fixed image.
    """

    @thread_worker
    def _registration_thread():

        from ..clemreg.widget_components import run_point_cloud_registration_and_warping

        if registration_direction == u'FM \u2192 EM':
            moving_input_points = Moving_Points
            fixed_input_points = Fixed_Points

        elif registration_direction == u'EM \u2192 FM':
            moving_input_points = Fixed_Points
            fixed_input_points = Moving_Points

        warp_outputs, transformed = run_point_cloud_registration_and_warping(Moving_Points,
                                                                             Fixed_Points,
                                                                             Moving_Image,
                                                                             registration_algorithm,
                                                                             registration_max_iterations,
                                                                             warping_interpolation_order,
                                                                             warping_approximate_grid,
                                                                             warping_sub_division_factor)
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

    worker_registration = _registration_thread()
    worker_registration.returned.connect(_add_data)
    worker_registration.start()
