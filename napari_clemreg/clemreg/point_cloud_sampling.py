#!/usr/bin/env python3
# coding: utf-8
import time
from napari.layers import Labels, Points
from skimage import feature
import numpy as np
import open3d as o3


def point_cloud_sampling(input: Labels,
                         every_k_points: float = 5,
                         voxel_size: int = 5,
                         sigma: float = 1.0,
                         filter_kwargs = {'nb_neighbors': 5, 'std_ratio': 1.0},
                         edge_color: str = 'black',
                         face_color: str = 'red',
                         point_size: int = 5):
    """
    ?

    Parameters
    ----------
    edge_color : str
        Colour to be displayed in napari viewer
    input : napari.layers.Labels
        Binary labels from which points are sampled from the outside
    sampling_frequency : float
        Frequency of cloud sampling
    sigma : float
        Sigma of Laplacian of Gaussian
    face_color : str
        Face colour to be displayed in napari viewer
    point_size : int
        Point size of points in point cloud to be displayed in napari viewer
    Returns
    -------
    Points layer containing points sampled from edges of binary segmentations with sepcified kwargs
    """
    print(f'Sampling point cloud from {input.name} with voxel_size={voxel_size} and sampling_frequency={1 / every_k_points}...')
    start_time = time.time()

    canny_img_list = []
    for img in (input.data > 0).astype(np.bool):
        canny_img_list.append(feature.canny(img, sigma=sigma))

    canny_img = np.stack(canny_img_list)

    points_idx = np.where(canny_img > 0)
    points = [[points_idx[0][i], points_idx[1][i], points_idx[2][i]] for i in range(points_idx[0].shape[0])]

    point_cloud = o3.geometry.PointCloud()
    point_cloud.points = o3.utility.Vector3dVector(points)
    point_cloud = point_cloud.uniform_down_sample(every_k_points=int(every_k_points))
    point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    # Add point filtering steps and stretching
    point_cloud, __ = point_cloud.remove_statistical_outlier(**filter_kwargs)

    print(f'Finished point cloud sampling after {time.time() - start_time}s!')

    return np.asarray(point_cloud.points)
