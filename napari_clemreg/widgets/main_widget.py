#!/usr/bin/env python3
# coding: utf-8
import json
import sys
import os.path
import napari
if ".." not in sys.path:
    sys.path.insert(0,"..")
from napari_clemreg.clemreg.widget_components import _create_json_file
import numpy as np
import pint
from magicgui import magic_factory, widgets
from napari.layers import Image, Shapes, Labels, Points
from napari.utils.notifications import show_error
from napari.qt.threading import GeneratorWorker
from ..clemreg.on_init_specs import specs

class CLEMRegWidget:
    def __init__(
        self,
        viewer: Viewer,

        Moving_Image: Image,
        Fixed_Image: Image,

        moving_image_pixelsize_xy: float,
        moving_image_pixelsize_z: float,
        fixed_image_pixelsize_xy: float,
        fixed_image_pixelsize_z: float,

        Mask_ROI: Shapes,
        z_min: int,
        z_max: int,

        registration_algorithm: str,
        registration_max_iterations: int,
        params_from_json: bool,
        load_json_file: str,
        advanced: bool,

        log_sigma: float,
        log_threshold: float,
        filter_segmentation: bool,
        filter_size_lower: float,
        filter_size_upper: float,

        point_cloud_sampling_frequency: float,
        registration_voxel_size: int,
        point_cloud_sigma: float,

        warping_interpolation_order: int,
        warping_approximate_grid: int,
        warping_sub_division_factor: int,

        registration_direction: str
    ):
        self.viewer = viewer


        # ---------------- Define threads ----------------

        @thread_worker
        def _run_moving_thread(
            self,
            **kwargs
        ):
            from ..clemreg.widget_components import run_moving_segmentation

            labels = run_moving_segmentation(**kwargs)
            labels = Labels(
                labels.astype(np.uint16),
                name='FM_segmentation',
                metadata=Moving_Image.metadata
            )

            return dict(Moving_Segmentation=labels)

        @thread_worker
        def _run_fixed_thread(
            self,
            **kwargs
        ):
            from ..clemreg.widget_components import run_fixed_segmentation

            labels = run_fixed_segmentation(**kwargs)
            labels = Labels(
                labels.astype(np.uint16),
                name='EM_segmentation',
                metadata=Fixed_Image.metadata
            )

            return dict(Fixed_Segmentation=labels)

        @thread_worker
        def _run_registration_thread(
            self,
            **kwargs
        ):
            from ..clemreg.widget_components import run_point_cloud_sampling
            from ..clemreg.widget_components import run_point_cloud_registration_and_warping

            point_cloud_args = ['Moving_Segmentation',
                                'Fixed_Segmentation',
                                'moving_image_pixelsize_xy',
                                'moving_image_pixelsize_z',
                                'fixed_image_pixelsize_xy',
                                'fixed_image_pixelsize_z',
                                'point_cloud_sampling_frequency',
                                'voxel_size',
                                'point_cloud_sigma']

            point_cloud_kwargs = dict((k, kwargs[k]) for k in point_cloud_keys if k in kwargs)
            moving_points, fixed_points = run_point_cloud_sampling(**point_cloud_kwargs)

            reg_and_warping_args = ['Moving_Image',
                                    'Fixed_Image',
                                    'registration_algorithm',
                                    'registration_max_iterations',
                                    'warping_interpolation_order',
                                    'warping_approximate_grid',
                                    'warping_sub_division_factor',
                                    'registration_direction']

            reg_and_warping_kwargs = dict((k, kwargs[k]) for k in reg_and_warping_keys if k in kwargs)
            point_cloud_return_kwargs = dict(Moving_Points=moving_points, Fixed_Points=fixed_points)
            point_cloud_reg_and_warping_kwargs = {**point_cloud_return_kwargs, **reg_and_warping_kwargs}
            warp_outputs, transformed = run_point_cloud_registration_and_warping(**point_cloud_reg_and_warping_kwargs)

            # This needs to be written to Zarr file
            return warp_outputs

        def launch_segmentation(
            self
        ):
            registration_thread_kwargs = dict(
                moving_image_pixelsize_xy=moving_image_pixelsize_xy,
                moving_image_pixelsize_z=moving_image_pixelsize_z,
                fixed_image_pixelsize_xy=fixed_image_pixelsize_xy,
                fixed_image_pixelsize_z=fixed_image_pixelsize_z,
                point_cloud_sampling_frequency=point_cloud_sampling_frequency,
                voxel_size=registration_voxel_size,
                point_cloud_sigma=point_cloud_sigma,
                Moving_Image=Moving_Image,
                Fixed_Image=Fixed_Image,
                registration_algorithm=registration_algorithm,
                registration_max_iterations=registration_max_iterations,
                warping_interpolation_order=warping_interpolation_order,
                warping_approximate_grid=warping_approximate_grid,
                warping_sub_division_factor=warping_sub_division_factor,
                registration_direction=registration_direction
            )
            joiner = RegistrationThreadJoiner(worker_function=_run_registration_thread,
                                              init_kwargs=registration_thread_kwargs,
                                              returned=_add_data,
                                              yielded=_yield_point_clouds)

            def _class_setter_moving(x):
                joiner.set_moving_kwargs(x)

            def _class_setter_fixed(x):
                joiner.set_fixed_kwargs(x)

            def _finished_moving_emitter():
                joiner.finished_moving()

            def _finished_fixed_emitter():
                joiner.finished_fixed()

            worker_moving = _run_moving_thread(Moving_Image=Moving_Image,
                                               Mask_ROI=Mask_ROI,
                                               z_min=z_min,
                                               z_max=z_max,
                                               log_sigma=log_sigma,
                                               log_threshold=log_threshold,
                                               filter_segmentation=filter_segmentation,
                                               filter_size_lower=filter_size_lower,
                                               filter_size_upper=filter_size_upper)
            worker_moving.returned.connect(_class_setter_moving)
            worker_moving.finished.connect(_finished_moving_emitter)
            worker_moving.yielded.connect(_yield_segmentation)
            worker_moving.start()

            worker_fixed = _run_fixed_thread(Fixed_Image=Fixed_Image,
                                             em_seg_axis=em_seg_axis)
            worker_fixed.returned.connect(_class_setter_fixed)
            worker_fixed.finished.connect(_finished_fixed_emitter)
            worker_fixed.yielded.connect(_yield_segmentation)
            worker_fixed.start()


def main_widget():


    def on_init(widget):
        pass

    @magic_factory(
        widget_init=on_init,
        layout='vertical',
        call_button='Register',
        widget_header=dict(
            widget_type='Label',
            label=f'<h1 text-align="left">CLEM-Reg</h1>'),

        project_dir=dict(
            widget_type='FileEdit',
            value='',
            label='Directory',
            mode='d',
            tooltip='Specify a project directory.'
        )



    )
    def widget_factory(
        viewer: 'napari.viewer.Viewer',
        widget_header,

        project_dir,
    )
