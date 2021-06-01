#!/usr/bin/env python3
# coding: utf-8
import numpy as np
# from napari.qt.threading import thread_worker
from magicgui import magic_factory, widgets
import napari
from napari.layers import Labels
from typing_extensions import Annotated
import cc3d
from napari.qt import thread_worker
import time

@magic_factory
def make_clean_binary_segmentation(
    viewer: "napari.viewer.Viewer",
    input: Labels,
    percentile: Annotated[int, {"min": 0, "max": 100, "step": 1}]=95):
    """Generates widget for binary segmentation cleaning function

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        Napari viewer allows addition of layer once thread_worker finished
        executing
    input : napari.layers.Labels
        Input binary segmentation obtained from a napari Labels layer
    percentile : int
        Specifies threshold for filtering individual objects based on size
        determined as the number of True pixels
    """

    pbar = widgets.ProgressBar()
    # Unknown duration
    pbar.range = (0, 0)
    # Add progress bar to the top of widget
    make_clean_binary_segmentation.insert(0, pbar)

    def _add_data(return_value, self=make_clean_binary_segmentation):
        print('Adding new layer to viewer...')
        data, kwargs = return_value
        viewer.add_labels(data, **kwargs)
        self.pop(0).hide()  # remove the progress bar
        print('Done!')

    @thread_worker(connect={"returned": _add_data})
    def _clean_binary_segmentation(input: Labels,
                                   percentile: int=95):
        """Filters binary segmentation based on size

        Parameters
        ----------
        input : napari.layers.Labels
            Binary segmentation volume
        percentile : int
            Specifies threshold for filtering individual objects based on size
            determined as the number of True pixels. Indiviudal objects are
            found with 3D connected components.

        Returns
        -------
        clean_binary_volume : numpy.ndarray
            A numpy array of the filtered binary volume
        kwargs : dict
            A dictionary of parameters for adding filtered binary volume to
            napari viewer
        """
        start = time.time()

        # Find connected components
        labels_out = cc3d.connected_components(input.data)
        print(f'Identified {labels_out.max()} connected components.')

        # Count occurences of each connected component
        num_of_occurences = np.bincount(labels_out.flatten())
        threshold = np.percentile(num_of_occurences, percentile)
        print(f'Objects with size below {threshold} will be removed.')

        elements_below_thresh = num_of_occurences < threshold
        idx_below_thresh = np.where(elements_below_thresh)[0]
        print(f'Removing {len(idx_below_thresh)} objects...')

        # Creates boolean array of components that will be removed
        below_thresh_binary_volume = np.isin(labels_out, idx_below_thresh)
        # Essentially binary subtraction of filtered from input array
        clean_binary_volume = np.logical_xor(input.data, below_thresh_binary_volume)

        elapsed = time.time() - start
        print(f'Finished execution after {elapsed} seconds.')

        kwargs = dict(
            name=input.name + '_cleaned'
        )
        return (clean_binary_volume, kwargs)

    # start the thread
    _clean_binary_segmentation(input=input,
                               percentile=percentile)
