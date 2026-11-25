"""End-to-end characterisation test.

Exercises the full segment -> sample -> register -> warp pipeline on a
synthetic dataset with a *known* ground-truth transform between the two
modalities, and asserts the warped moving segmentation overlaps the fixed
segmentation above a Dice threshold. This is the acceptance-gate test the
plan calls for before the Python 3.11 / napari 0.6.6 dependency bumps and
the core/widget split (§6) -- a large numeric drift in any of the four
pipeline stages should show up here even if each stage's own tests still
pass individually.

Uses the affine warp path (_warp_image_volume_affine), not BCPD/TPS, since
the BCPD warp has a known pre-existing bug (#9) unrelated to what this
test is checking.
"""
import numpy as np
import pytest
from scipy import ndimage

from napari_clemreg.clemreg.point_cloud_sampling import point_cloud_sampling
from napari_clemreg.clemreg.point_cloud_registration import point_cloud_registration
from napari_clemreg.clemreg.warp_image_volume import _warp_image_volume_affine

from .conftest import make_blob_volume, SYNTHETIC_BLOB_SHAPE, SYNTHETIC_BLOB_CENTERS, SYNTHETIC_BLOB_RADIUS


class _FakeLabels:
    """Minimal duck-typed stand-in for a napari Labels layer."""

    def __init__(self, data, name='labels'):
        self.data = data
        self.name = name


def _apply_affine(matrix, point):
    z, y, x = point
    return tuple((matrix @ np.array([z, y, x, 1.0]))[:3])


def _dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())


def test_full_pipeline_recovers_known_transform_with_high_dice():
    # Ground-truth FM(moving) -> EM(fixed) rigid transform: small rotation
    # about the z axis plus a few-voxel translation.
    theta = np.deg2rad(8)
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ])
    translation = np.array([1.0, 2.0, -1.5])
    fm_to_em = np.eye(4)
    fm_to_em[:3, :3] = rotation
    fm_to_em[:3, 3] = translation
    em_to_fm = np.linalg.inv(fm_to_em)

    # Independently generate the EM segmentation from the fixed blob
    # centres, and the FM segmentation from those same structures mapped
    # backwards through the ground-truth transform -- so the two volumes
    # are only related via `fm_to_em`, not via image resampling.
    em_centers = SYNTHETIC_BLOB_CENTERS
    fm_centers = [_apply_affine(em_to_fm, c) for c in em_centers]

    em_volume = make_blob_volume(
        SYNTHETIC_BLOB_SHAPE, em_centers, SYNTHETIC_BLOB_RADIUS, rng=np.random.default_rng(0)
    )
    fm_volume = make_blob_volume(
        SYNTHETIC_BLOB_SHAPE, fm_centers, SYNTHETIC_BLOB_RADIUS, rng=np.random.default_rng(1)
    )
    em_binary = (em_volume > 0.3).astype(np.uint8)
    fm_binary = (fm_volume > 0.3).astype(np.uint8)

    em_points = point_cloud_sampling(_FakeLabels(em_binary), every_k_points=2, voxel_size=1, sigma=1.0)
    fm_points = point_cloud_sampling(_FakeLabels(fm_binary), every_k_points=2, voxel_size=1, sigma=1.0)

    _, _, _, kwargs = point_cloud_registration(
        fm_points, em_points, algorithm='Rigid CPD', max_iterations=100
    )
    recovered = kwargs['affine']

    # Loose sanity check: recovered transform is in the right ballpark.
    # Point-cloud sampling (canny edges + voxel downsampling) means CPD
    # won't recover the exact matrix -- Dice below is the real signal.
    np.testing.assert_allclose(recovered, fm_to_em, atol=0.5)

    warped_fm = _warp_image_volume_affine(
        fm_binary, recovered, output_shape=SYNTHETIC_BLOB_SHAPE, interpolation_order=0
    )
    dice = _dice(warped_fm, em_binary)
    assert dice > 0.85


# Balls at known physical positions (µm, z y x), drawn on an isotropic EM
# grid and on an FM grid three times coarser in z, so the two volumes are
# related by the identity in physical space.
PHYSICAL_CENTRES = [(0.4, 0.6, 0.6), (0.4, 1.8, 0.7), (1.0, 0.7, 1.7), (1.05, 1.7, 1.8), (0.7, 1.2, 1.2)]
PHYSICAL_RADIUS = 0.3


def _balls(shape, pixel_size):
    grid = np.meshgrid(*[np.arange(n) * p for n, p in zip(shape, pixel_size)], indexing='ij')
    volume = np.zeros(shape, dtype=np.uint8)
    for centre in PHYSICAL_CENTRES:
        volume[sum((g - c) ** 2 for g, c in zip(grid, centre)) <= PHYSICAL_RADIUS ** 2] = 1
    return volume


# Through the adapters, because the bug is in how they resample between
# grids: a z stretch moves the balls high in the stack the most.
@pytest.mark.parametrize('algorithm, tolerance', [('Rigid CPD', 0.01), ('Affine CPD', 0.01), ('BCPD', 0.025)])
def test_native_resolution_warp_puts_fm_objects_where_they_are(algorithm, tolerance):
    import pint
    from napari.layers import Image, Labels
    from napari_clemreg.clemreg import widget_components as wc

    em = _balls((36, 60, 60), (0.04, 0.04, 0.04))
    fm = _balls((12, 60, 60), (0.12, 0.04, 0.04))
    ureg = pint.UnitRegistry()
    nm = lambda value: ureg.Quantity(value, 'nanometer')

    moving_points, fixed_points = wc.run_point_cloud_sampling(
        Labels(fm), Labels(em), nm(40), nm(120), nm(40), nm(40),
        point_cloud_sampling_frequency=30, voxel_size=1, point_cloud_sigma=1.0)
    warped, __ = wc.run_point_cloud_registration_and_warping(
        moving_points, fixed_points, Image(fm.astype(np.float32), name='FM'),
        Image(em.astype(np.float32), name='EM'), algorithm, registration_direction='FM → EM',
        warping_output_resolution='Native LM resolution', registration_max_iterations=50,
        warping_interpolation_order=1, warping_approximate_grid=5, warping_sub_division_factor=1)

    # The warped layer's scale is relative to EM's 40 nm pixels.
    mask = warped[0].data > 0.5
    labels, n = ndimage.label(mask)
    centroids = np.array(ndimage.center_of_mass(mask, labels, range(1, n + 1)))
    centroids = centroids * np.asarray(warped[0].scale) * 0.04
    assert n == len(PHYSICAL_CENTRES)
    errors = [np.linalg.norm(np.asarray(PHYSICAL_CENTRES) - c, axis=1).min() for c in centroids]
    assert max(errors) < tolerance


@pytest.mark.parametrize('algorithm', ['Rigid CPD', 'Affine CPD', 'BCPD'])
def test_transformed_points_overlay_the_fixed_points(algorithm):
    """Where napari draws the transformed points (data through the layer's
    affine) is on the fixed point cloud. The FM is drawn 0.24 µm (6 voxels)
    off in y, so drawing the moving points unchanged would fail."""
    import pint
    from napari.layers import Image, Labels
    from napari_clemreg.clemreg import widget_components as wc
    from scipy.spatial import cKDTree

    em = _balls((36, 60, 60), (0.04, 0.04, 0.04))
    fm = _balls((12, 60, 60), (0.12, 0.04, 0.04))
    shifted = np.roll(fm, -6, axis=1)
    assert np.array_equal(np.roll(shifted, 6, axis=1)[:, 6:], fm[:, 6:]) and not fm[:, :6].any()  # no ball wraps
    fm = shifted
    nm = lambda value: pint.UnitRegistry().Quantity(value, 'nanometer')
    moving_points, fixed_points = wc.run_point_cloud_sampling(
        Labels(fm), Labels(em), nm(40), nm(120), nm(40), nm(40),
        point_cloud_sampling_frequency=30, voxel_size=1, point_cloud_sigma=1.0)
    __, transformed = wc.run_point_cloud_registration_and_warping(
        moving_points, fixed_points, Image(fm.astype(np.float32)), Image(em.astype(np.float32)),
        algorithm, registration_direction='FM → EM', registration_max_iterations=50,
        warping_interpolation_order=1, warping_approximate_grid=5, warping_sub_division_factor=1)

    drawn = transformed.affine(transformed.data)
    centre = lambda points: np.mean(points, axis=0)
    assert np.linalg.norm(centre(moving_points.data) - centre(fixed_points.data)) > 4  # 40 nm voxels
    assert np.linalg.norm(centre(drawn) - centre(fixed_points.data)) < 1
    assert np.median(cKDTree(fixed_points.data).query(drawn)[0]) < 1.5
