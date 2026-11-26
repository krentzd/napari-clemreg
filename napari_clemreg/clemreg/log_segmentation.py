#!/usr/bin/env python3
# coding: utf-8
from scipy.ndimage import gaussian_filter1d
import cc3d
from skimage import feature, exposure, filters
import time
import numpy as np
from tqdm import tqdm

from ._arrays import as_array
from .exceptions import NoSegmentationError

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

def _diff_of_gauss_2d(img, sigma_1=2.5, sigma_2=4):
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

    gauss_img_0_i = gaussian_filter1d(img, sigma_2, axis=0)
    gauss_img_1_i = gaussian_filter1d(gauss_img_0_i, sigma_2, axis=1)

    diff_of_gauss = gauss_img_1_e - gauss_img_1_i

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
    for i in tqdm(range(img.shape[0]), desc='Segmenting image..'):
        slice = exposure.rescale_intensity(img[i], out_range='uint8')

        slice_thresh = np.sum(slice) / (slice.shape[0] * slice.shape[1]) * thresh
        slice[slice < slice_thresh] = 0
        slice[slice >= slice_thresh] = 1
        thresh_img.append(slice)

        # slice_thresh_lower = np.sum(slice) / (slice.shape[0] * slice.shape[1]) * thresh[0]
        # slice_thresh_upper = np.sum(slice) / (slice.shape[0] * slice.shape[1]) * thresh[1]
        # slice[slice < slice_thresh_lower] = 0
        # slice[slice >= slice_thresh_upper] = 0
        # thresh_img.append(slice > 0)

    return np.asarray(thresh_img)

def log_segmentation(input: np.ndarray,
                     sigma: float = 3,
                     threshold: float = 1.2):
    """ Apply log segmentation to user input.

    Parameters
    ----------
    input : np.ndarray
        Volume to segment. For backwards compatibility, an object with the
        volume in `.data` (e.g. a napari Image layer) is also accepted.
    sigma : float
        Sigma value for 1D gaussian filter to be applied oto image before segmentation
    threshold : float
        Threshold value to apply to image
    Returns
    -------
    np.ndarray
        Binary segmentation of the volume
    """
    print(f'Segmenting {getattr(input, "name", "volume")} with sigma={sigma} and threshold={threshold}...')
    start_time = time.time()

    # img = np.clip(input.data, 0, np.percentile(input.data, 0.95))

    volume = _min_max_scaling(as_array(input))
    sigma_2 = sigma * 1.6
    log_iso_volume = _diff_of_gauss(volume, sigma, sigma_2)
    seg_volume = _slice_adaptive_thresholding(log_iso_volume, threshold)

    print(f'Finished segmenting after {time.time() - start_time}s!')

    return seg_volume

def filter_binary_segmentation(input: np.ndarray,
                               percentile: tuple=(5,95)):
    """Keep the objects (3D connected components) of a binary volume whose
    size, in voxels, is within the (lower, upper) percentiles of all the
    objects' sizes. Returns the filtered binary volume."""
    labels = cc3d.connected_components(input)
    sizes = np.bincount(labels.ravel())[1:]  # Leave out the background.
    if sizes.size == 0:
        return labels > 0
    lower, upper = np.percentile(sizes, percentile)
    keep = np.flatnonzero((sizes >= lower) & (sizes <= upper)) + 1
    print(f'Keeping {keep.size} of {sizes.size} objects, with {lower:g} to {upper:g} voxels.')
    return np.isin(labels, keep)


def segment_fm(volume: np.ndarray,
               sigma: float = 3,
               threshold: float = 1.2,
               size_filter: tuple = None,
               roi=None) -> np.ndarray:
    """Segment mitochondria in an FM volume (napari-free core API).

    Parameters
    ----------
    volume : np.ndarray
        The FM volume, (z, y, x).
    sigma, threshold : float
        Laplacian-of-Gaussian sigma and threshold.
    size_filter : tuple of (lower, upper) percentiles, optional
        Remove objects whose volume falls outside these percentiles.
    roi : mask_roi.MaskRoi, optional
        Keep only the segmentation inside this polygon and z range.

    Returns
    -------
    np.ndarray
        The segmentation.

    Raises
    ------
    NoSegmentationError
        If nothing is segmented (checked before the ROI is applied).
    """
    from .mask_roi import mask_roi

    seg_volume = log_segmentation(volume, sigma=sigma, threshold=threshold)
    if size_filter is not None:
        seg_volume = filter_binary_segmentation(input=seg_volume, percentile=size_filter)
    if np.all(seg_volume == seg_volume.flat[0]):
        raise NoSegmentationError('No mitochondria found in the FM image')
    if roi is not None:
        seg_volume = mask_roi(seg_volume, roi.polygon, z_min=roi.z_min, z_max=roi.z_max)
    return seg_volume
