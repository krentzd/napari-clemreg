#!/usr/bin/env python3
# coding: utf-8
from probreg import cpd, bcpd, callbacks
import numpy as np
import open3d as o3
import transforms3d as t3d
import time
from napari.types import PointsData, ImageData
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

def _add_data(return_value):
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

def point_cloud_registration(moving: PointsData,
                             fixed: PointsData,
                             algorithm: str='Rigid CPD',
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
