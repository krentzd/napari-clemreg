#!/usr/bin/env python3
# coding: utf-8
from probreg import cpd, bcpd, callbacks
import numpy as np
import open3d as o3
import transforms3d as t3d
import time
from napari.types import PointsData, ImageData
from magicgui import magic_factory, widgets
import napari
from typing_extensions import Annotated
from math import cos, sin
import math

# TODO: Account for piecewise maxi iterations
class RegistrationProgressCallback(object):
    def __init__(self, maxiter):
        self.counter = 0
        self.maxiter = maxiter

    def __call__(self, *args):
        self.counter += 1
        print('{}/{}'.format(self.counter, self.maxiter))

def _make_matrix_from_rigid_params(rot, trans, s):
    "Create homogenous transformation matrix from rigid parameters"

    T_a = np.array([[1., 0., 0., trans[0]],
                    [0., 1., 0., trans[1]],
                    [0., 0., 1., trans[2]],
                    [0., 0., 0., 1.]])

    # Probreg defines rigid transform update as: S * np.dot(points, R.T) + T
    T_rot = np.vstack((np.hstack((rot, [[0.], [0.], [0.]])), [0., 0., 0., 1.]))

    T_s = np.array([[s, 0., 0., 0.],
                    [0., s, 0., 0.],
                    [0., 0., s, 0.],
                    [0., 0., 0., 1.]])

    return T_a @ T_rot @ T_s

def prepare_source_and_target_nonrigid_3d(source_array,
                                          target_array,
                                          voxel_size=5,
                                          every_k_points=2):
    source = o3.geometry.PointCloud()
    target = o3.geometry.PointCloud()
    source.points = o3.utility.Vector3dVector(source_array)
    target.points = o3.utility.Vector3dVector(target_array)
    source = source.uniform_down_sample(every_k_points=every_k_points)
    target = target.uniform_down_sample(every_k_points=every_k_points)
    source = source.voxel_down_sample(voxel_size=voxel_size)
    target = target.voxel_down_sample(voxel_size=voxel_size)
    return source, target

def on_init(widget):
    """Initializes widget layout amd updates widget layout according to user input."""

    for x in ['moving', 'fixed', 'algorithm', 'visualise', 'max_iterations', 'voxel_size', 'every_k_points']:
        setattr(getattr(widget, x), 'visible', True)
    for x in ['fixed_image', 'sub_division_factor_x', 'sub_division_factor_y', 'sub_division_factor_z']:
        setattr(getattr(widget, x), 'visible', False)

    def toggle_registration_widget(event):
        # if event.value == "BCPD":
        #     for x in ['voxel_size', 'every_k_points']:
        #         setattr(getattr(widget, x), 'visible', True)
        #     for x in ['fixed_image', 'sub_division_factor']:
        #         setattr(getattr(widget, x), 'visible', False)

        if event.value == "Piecewise BCPD":
            for x in ['fixed_image', 'voxel_size', 'every_k_points', 'max_iterations', 'sub_division_factor_x', 'sub_division_factor_y', 'sub_division_factor_z']:
                setattr(getattr(widget, x), 'visible', True)

        else:
            for x in ['moving', 'fixed', 'algorithm', 'visualise', 'max_iterations', 'voxel_size', 'every_k_points']:
                setattr(getattr(widget, x), 'visible', True)
            for x in ['fixed_image', 'sub_division_factor_x', 'sub_division_factor_y', 'sub_division_factor_z']:
                setattr(getattr(widget, x), 'visible', False)

    widget.algorithm.changed.connect(toggle_registration_widget)

@magic_factory(widget_init=on_init, layout='vertical', call_button="Register")
def make_point_cloud_registration(
    viewer: "napari.viewer.Viewer",
    algorithm: Annotated[str, {"choices": ["BCPD", "Rigid CPD", "Affine CPD", "RANSAC", "Piecewise BCPD"]}], # TODO: Make piecewise option boolean
    moving: PointsData,
    fixed: PointsData,
    fixed_image: ImageData,
    sub_division_factor_x: Annotated[int, {"min": 1, "max": 10, "step": 1}] = 1,
    sub_division_factor_y: Annotated[int, {"min": 1, "max": 10, "step": 1}] = 1,
    sub_division_factor_z: Annotated[int, {"min": 1, "max": 10, "step": 1}] = 1,
    voxel_size: Annotated[int, {"min": 1, "max": 1000, "step": 1}] = 5,
    every_k_points: Annotated[int, {"min": 1, "max": 1000, "step": 1}] = 1,
    max_iterations: Annotated[int, {"min": 1, "max": 1000, "step": 1}] = 50,
    visualise: bool=False):

    from napari.qt import thread_worker

    pbar = widgets.ProgressBar()
    pbar.range = (0, 0)  # unknown duration
    make_point_cloud_registration.insert(0, pbar)  # add progress bar to the top of widget

    # this function will be called after we return
    def _add_data(return_value, self=make_point_cloud_registration):
        moving, fixed, transformed, kwargs = return_value
        viewer.add_points(moving,
                          name='moving_points',
                          size=5,
                          face_color='red')
        viewer.add_points(fixed,
                          name='fixed_points',
                          size=5,
                          face_color='green')
        viewer.add_points(moving, **kwargs)
        viewer.add_points(transformed,
                          name='transformed_points_probreg',
                          size=5,
                          face_color='yellow')
        self.pop(0).hide()  # remove the progress bar

    @thread_worker(connect={"returned": _add_data})
    def _point_cloud_registration(moving: PointsData,
                                  fixed: PointsData,
                                  algorithm: str='BCPD',
                                  voxel_size: int=5,
                                  every_k_points: int=1,
                                  max_iterations: int=50,
                                  visualise: bool=False):
        start = time.time()
        source, target = prepare_source_and_target_nonrigid_3d(moving,
                                                               fixed,
                                                               voxel_size=voxel_size,
                                                               every_k_points=every_k_points)
        cbs = []
        cbs.append(RegistrationProgressCallback(max_iterations))
        if visualise:
            cbs.append(callbacks.Open3dVisualizerCallback(source, target))

        if algorithm == 'BCPD':
            tf_param = bcpd.registration_bcpd(source,
                                              target,
                                              maxiter=max_iterations,
                                              callbacks=cbs)

        elif algorithm == 'Rigid CPD':
            tf_param, __, __ = cpd.registration_cpd(source,
                                                    target,
                                                    tf_type_name='rigid',
                                                    maxiter=max_iterations,
                                                    callbacks=cbs)

        elif algorithm == 'Affine CPD':
            tf_param, __, __ = cpd.registration_cpd(source,
                                                    target,
                                                    tf_type_name='affine',
                                                    maxiter=max_iterations,
                                                    callbacks=cbs)
        elif algorithm == 'RANSAC':
            raise NotImplementedError

        elif algorithm == 'Piecewise BCPD':
            # TODO: Reinstantiate RegistrationProgressCallback!
            source_out = []
            transformed_out = []

            x_chunk = math.ceil(fixed_image.shape[1] / sub_division_factor_x)
            y_chunk = math.ceil(fixed_image.shape[2] / sub_division_factor_y)
            z_chunk = math.ceil(fixed_image.shape[0] / sub_division_factor_z)

            for x in range(math.ceil(fixed_image.shape[1] / x_chunk)):
              for y in range(math.ceil(fixed_image.shape[2] / y_chunk)):
                for z in range(math.ceil(fixed_image.shape[0] / z_chunk)):
                    output_region = (z * z_chunk,
                                     x * x_chunk,
                                     y * y_chunk,
                                     min((z * z_chunk + z_chunk), fixed_image.shape[0]),
                                     min((x * x_chunk + x_chunk), fixed_image.shape[1]),
                                     min((y * y_chunk + y_chunk), fixed_image.shape[2]))
                    z_min, x_min, y_min, z_max, x_max, y_max = output_region

                    bbox = o3.geometry.AxisAlignedBoundingBox([z_min, x_min, y_min],
                                                              [z_max, x_max, y_max])
                    z_pad, x_pad, y_pad = 50, 50, 50
                    z_min_pad = min(0, z_min - z_pad)
                    x_min_pad = min(0, x_min - x_pad)
                    y_min_pad = min(0, y_min - y_pad)
                    z_max_pad = min(fixed_image.shape[0], z_max + z_pad)
                    x_max_pad = min(fixed_image.shape[1], x_max + x_pad)
                    y_max_pad = min(fixed_image.shape[2], y_max + y_pad)
                    bbox_pad = o3.geometry.AxisAlignedBoundingBox([z_min_pad, x_min_pad, y_min_pad],
                                                                  [z_max_pad, x_max_pad, y_max_pad])
                    source_crop = source.crop(bbox)
                    target_crop = target.crop(bbox_pad)

                    source_out.append(np.asarray(source_crop.points))
                    # TODO: Ensure continuity across chunks
                    if source_crop.has_points() and target_crop.has_points():
                        print(np.asarray(source_crop.points).shape, np.asarray(target_crop.points).shape)
                        tf_param = bcpd.registration_bcpd(source_crop,
                                                          target_crop,
                                                          maxiter=max_iterations,
                                                          callbacks=cbs)

                        transformed_out.append(tf_param._transform(np.asarray(source_crop.points)))
                    else:
                        transformed_out.append(np.asarray(source_crop.points))

            kwargs = dict(name='transformed_points',
                          face_color='blue',
                          size=5)

            return (np.vstack(source_out),
                    np.asarray(target.points),
                    np.vstack(transformed_out),
                    kwargs)

        elapsed = time.time() - start
        print("time: ", elapsed)

        if algorithm == 'BCPD':
            print("result: ", np.rad2deg(t3d.euler.mat2euler(tf_param.rigid_trans.rot)),
                  tf_param.rigid_trans.scale, tf_param.rigid_trans.t, tf_param.v)
            kwargs = dict(name='transformed_points',
                          face_color='blue',
                          size=5)

        elif algorithm == 'Rigid CPD':
            print("result: ", tf_param.rot,
                  tf_param.scale, tf_param.t)

            mat = _make_matrix_from_rigid_params(tf_param.rot,
                                                 tf_param.t,
                                                 tf_param.scale)

            kwargs = dict(name='transformed_points',
                          face_color='blue',
                          affine=mat,
                          size=5)

        elif algorithm == 'Affine CPD':
            print("result: ", tf_param.b, tf_param.t)
            mat, off = tf_param.b, tf_param.t

            off = np.expand_dims(off, axis=0).T
            mat = np.vstack((np.hstack((mat, off)), np.array([0, 0, 0, 1])))

            kwargs = dict(name='transformed_points',
                          face_color='blue',
                          affine=mat,
                          size=0.5) # Point sizes don't display correctly

        return (np.asarray(source.points),
                np.asarray(target.points),
                tf_param._transform(source.points),
                kwargs)

    _point_cloud_registration(moving=moving,
                              fixed=fixed,
                              algorithm=algorithm,
                              voxel_size=voxel_size,
                              every_k_points=every_k_points,
                              max_iterations=max_iterations,
                              visualise=visualise)
