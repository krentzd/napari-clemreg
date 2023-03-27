import napari
from magicgui import magic_factory
from napari.layers import Points, Image
from napari.qt.threading import thread_worker
from napari.layers.utils._link_layers import link_layers


@magic_factory(layout='vertical',
               call_button='Register',
               widget_header={'widget_type': 'Label',
                              'label': f'<h2 text-align="left">Registration and Warping</h2>'},
               registration_algorithm={'label': 'Registration Algorithm',
                                       'widget_type': 'ComboBox',
                                       'choices': ["BCPD", "Rigid CPD", "Affine CPD"],
                                       'value': 'Rigid CPD',
                                       'tooltip': 'Speed: Rigid CPD > Affine CPD > BCPD'},

               registration_header={'widget_type': 'Label',
                                    'label': f'<h3 text-align="left">Point Cloud Registration</h3>'},
               registration_voxel_size={'label': 'Voxel Size',
                                        'widget_type': 'SpinBox',
                                        'min': 1, 'max': 1000, 'step': 1,
                                        'value': 5},
               registration_every_k_points={'label': 'Subsampling',
                                            'widget_type': 'SpinBox',
                                            'min': 1, 'max': 1000, 'step': 1,
                                            'value': 1},
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
               )
def registration_warping_widget(viewer: 'napari.viewer.Viewer',
                                widget_header,
                                Moving_Image: Image,
                                Fixed_Image: Image,
                                Moving_Points: Points,
                                Fixed_points: Points,

                                registration_header,
                                registration_algorithm,
                                registration_voxel_size,
                                registration_every_k_points,
                                registration_max_iterations,

                                warping_header,
                                warping_interpolation_order,
                                warping_approximate_grid,
                                warping_sub_division_factor
                                ):
    from ..clemreg.point_cloud_registration import point_cloud_registration
    from ..clemreg.warp_image_volume import warp_image_volume

    @thread_worker
    def _registration_thread():
        moving, fixed, transformed, kwargs = point_cloud_registration(Moving_Points.data,
                                                                      Fixed_points.data,
                                                                      algorithm=registration_algorithm,
                                                                      voxel_size=registration_voxel_size,
                                                                      every_k_points=registration_every_k_points,
                                                                      max_iterations=registration_max_iterations)

        if registration_algorithm == 'Affine CPD' or registration_algorithm == 'Rigid CPD':
            transformed = Points(moving, **kwargs)
        else:
            transformed = Points(transformed)

        return warp_image_volume(moving_image=Moving_Image,
                                 fixed_image=Fixed_Image.data,
                                 transform_type=registration_algorithm,
                                 moving_points=moving,
                                 transformed_points=transformed,
                                 interpolation_order=warping_interpolation_order,
                                 approximate_grid=warping_approximate_grid,
                                 sub_division_factor=warping_sub_division_factor)

    def _add_data(return_value):
        if isinstance(return_value, list):
            layers = []
            for image_data in return_value:
                data, kwargs = image_data
                viewer.add_image(data, **kwargs)
                layers.append(viewer.layers[kwargs['name']])
            link_layers(layers)
        else:
            data, kwargs = return_value
            viewer.add_image(data, **kwargs)

    worker_registration = _registration_thread()
    worker_registration.returned.connect(_add_data)
    worker_registration.start()
