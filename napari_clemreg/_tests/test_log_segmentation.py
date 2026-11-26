"""Characterisation tests.

Unlike the deterministic unit tests, these pin what the pipeline does
*today* on a fixed synthetic input, with tolerance, as the regression
oracle for future dependency bumps (numpy/scipy/scikit-image) and the
napari 0.6.6 upgrade. Some drift across dependency versions is expected;
the tolerance bands below are where "acceptable drift" is encoded --
if a bump moves these counts outside the band, that's a deliberate
signal to look closer, not necessarily a bug.
"""
import cc3d
import numpy as np

from napari_clemreg.clemreg.log_segmentation import log_segmentation


class _FakeImage:
    """Minimal duck-typed stand-in for a napari Image layer."""

    def __init__(self, data, name='test'):
        self.data = data
        self.name = name


def test_log_segmentation_detects_expected_number_of_blobs(synthetic_blob_volume):
    seg = log_segmentation(_FakeImage(synthetic_blob_volume), sigma=3, threshold=1.2)

    n_components = cc3d.connected_components(seg.astype(np.uint8)).max()

    # 5 true blobs (SYNTHETIC_BLOB_CENTERS in conftest.py) plus a small
    # number of noise-induced fragments is expected and current baseline
    # is 9; allow some slack either side for dependency-driven drift.
    assert 5 <= n_components <= 13


def test_log_segmentation_foreground_voxel_count_within_tolerance(synthetic_blob_volume):
    seg = log_segmentation(_FakeImage(synthetic_blob_volume), sigma=3, threshold=1.2)

    voxel_count = int(seg.sum())

    # Baseline is 27281 foreground voxels; allow +/-25% for
    # dependency-driven numerical drift.
    assert 20000 <= voxel_count <= 35000


def _cubes(sides):
    """A mask of separate cubes with these side lengths, and each one's slice."""
    volume = np.zeros((6, 6, 8 * len(sides)), dtype=bool)
    where = []
    for i, side in enumerate(sides):
        cube = np.s_[:side, :side, 8 * i:8 * i + side]
        volume[cube] = True
        where.append(cube)
    return volume, where


def test_the_size_filter_keeps_the_objects_between_the_size_percentiles():
    """Sizes 1, 8, 27, 64, 125: the 10th and 90th percentiles are 3.8 and
    100.6 voxels, so the smallest and largest cube go. The background
    isn't an object."""
    from napari_clemreg.clemreg.log_segmentation import filter_binary_segmentation

    volume, cubes = _cubes([1, 2, 3, 4, 5])
    kept = filter_binary_segmentation(volume, percentile=(10, 90))

    assert [bool(kept[cube].all()) for cube in cubes] == [False, True, True, True, False]
    assert kept.sum() == 8 + 27 + 64


def test_the_widest_size_filter_keeps_every_object():
    from napari_clemreg.clemreg.log_segmentation import filter_binary_segmentation

    volume, _ = _cubes([1, 2, 3])
    np.testing.assert_array_equal(filter_binary_segmentation(volume, percentile=(0, 100)), volume)
