#!/usr/bin/env python3
# coding: utf-8
from magicgui import magic_factory, widgets
import numpy as np
from skimage import feature
from napari.layers import Labels
import napari
from typing_extensions import Annotated

@magic_factory
def make_point_cloud_sampling(
    viewer: "napari.viewer.Viewer",
    input: Labels,
    sampling_frequency: Annotated[int, {"min": 1, "max": 100, "step": 1}]=5,
    sigma: Annotated[float, {"min": 0, "max": 10, "step": 0.1}]=1.0,
    face_color: Annotated[str, {"choices": ["red", "green", "blue", "yellow"]}]='red',
    point_size: Annotated[int, {"min": 1, "max": 20, "step": 1}]=5
):
    from napari.qt import thread_worker
    pbar = widgets.ProgressBar()
    pbar.range = (0, 0)  # unknown duration
    make_point_cloud_sampling.insert(0, pbar)  # add progress bar to the top of widget

    # this function will be called after we return
    def _add_data(return_value, self=make_point_cloud_sampling):
        data, kwargs = return_value
        viewer.add_points(data, **kwargs)
        self.pop(0).hide()  # remove the progress bar

    @thread_worker(connect={"returned": _add_data})
    def _point_cloud_sampling(input: Labels,
                              sampling_frequency: float=0.01,
                              sigma: float=1.0,
                              edge_color: str='black',
                              face_color: str='red',
                              point_size: int=5):
        point_lst = []
        for z in range(input.data.shape[0]):
            img = (input.data[z] > 0).astype('uint8') * 255
            img = feature.canny(img, sigma=sigma)
            points = np.where(img == 1)
            # Just keep values at every n-th position
            for i in range(len(points[0])):
                if np.random.rand() < sampling_frequency:
                    point_lst.append([z, points[0][i], points[1][i]])

        kwargs = dict(
            edge_color=edge_color,
            face_color=face_color,
            size=point_size,
            name=input.name + '_points'
        )

        return (np.asarray(point_lst), kwargs)

    _point_cloud_sampling(input=input,
                          sampling_frequency=sampling_frequency / 100,
                          sigma=sigma,
                          face_color=face_color,
                          point_size=point_size)
