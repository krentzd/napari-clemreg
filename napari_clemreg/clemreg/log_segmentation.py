#!/usr/bin/env python3
# coding: utf-8
from scipy.ndimage import gaussian_filter1d
import numpy as np
import cc3d
from skimage import feature, exposure
from napari.layers import Image
from napari.qt.threading import thread_worker
import time
from napari.layers import Labels


def _min_max_scaling(data):
    """
    ?

    Parameters
    ----------
    data : ?
        ?
    Returns
    -------
        ?
    """
    n = data - np.min(data)
    d = np.max(data) - np.min(data)

    return n / d


def _diff_of_gauss(img, sigma_1=2.5, sigma_2=4):
    """ Calculates difference of gaussian of an inputted image
    with two sigma values.

    Sigma_1 is 2.5 and sigma_2 is 4 which is 1.6 times bigger than sigma_1.
    This ratio was originally proposed by Marr and Hildreth (1980) [1]_ and is commonly
    used when approximating the inverted Laplacian of Gaussian, which is used
    in edge and blob detection.

    Parameters
    ----------
    img : napari.layers.Image
        Image to apply difference of gaussian to
    sigma_1 : float
        float Sigma of the first Gaussian filter
    sigma_2 : float
        float Sigma of the second Gaussian filter
    Returns
    -------
    diff_of_gauss : Image
        Difference of gaussian of image
    """
    gauss_img_0_e = gaussian_filter1d(img, sigma_1, axis=0)
    gauss_img_1_e = gaussian_filter1d(gauss_img_0_e, sigma_1, axis=1)
    gauss_img_2_e = gaussian_filter1d(gauss_img_1_e, sigma_1, axis=2)

    gauss_img_0_i = gaussian_filter1d(img, sigma_2, axis=0)
    gauss_img_1_i = gaussian_filter1d(gauss_img_0_i, sigma_2, axis=1)
    gauss_img_2_i = gaussian_filter1d(gauss_img_1_i, sigma_2, axis=2)

    diff_of_gauss = gauss_img_2_e - gauss_img_2_i

    return diff_of_gauss


def _slice_adaptive_thresholding(img, thresh):
    """ Apply adaptive thresholding to the user inputted image stack
    based on the threshold value.

    Parameters
    ----------
    img : napari.layers.Image
        Image to apply adaptive thresholding to.
    thresh : float
        Threshold value to be applied to Image.
    Returns
    -------
    thresh_img : np.array
        Segmented image
    """
    thresh_img = []
    for i in range(img.shape[0]):
        slice = exposure.rescale_intensity(img[i], out_range='uint8')
        slice_thresh = np.sum(slice) / (slice.shape[0] * slice.shape[1]) * thresh
        slice[slice < slice_thresh] = 0
        slice[slice >= slice_thresh] = 1
        thresh_img.append(slice)

    return np.asarray(thresh_img)


# @thread_worker
def log_segmentation(input: Image,
                     sigma: float = 3,
                     threshold: float = 1.2):
    """ Apply log segmentation to user input.

    Parameters
    ----------
    input : napari.layers.Image
        Image to be segmented as napari Image layer
    sigma : float
        Sigma value for 1D gaussian filter to be applied oto image before segmentation
    threshold : float
        Threshold value to apply to image
    Returns
    -------
    Labels : napari.layers.Labels
        Labels of the segmented moving image
    """
    print(f'Segmenting {input.name} with sigma={sigma} and threshold={threshold}...')
    start_time = time.time()

    volume = _min_max_scaling(input.data)
    sigma_2 = sigma * 1.6
    log_iso_volume = _diff_of_gauss(volume, sigma, sigma_2)
    seg_volume = _slice_adaptive_thresholding(log_iso_volume, threshold)

    kwargs = dict(
        name=input.name + '_seg'
    )

    print(f'Finished segmenting after {time.time() - start_time}s!')

    return Labels(seg_volume, **kwargs)


def filter_binary_segmentation(input: Labels,
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
        name=input.name + '_seg'
    )
    return Labels(clean_binary_volume, **kwargs)
