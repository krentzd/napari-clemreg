import numpy as np
import pytest
from napari.layers import Image
from scipy import ndimage

from napari_clemreg.clemreg import warp_image_volume
from napari_clemreg.clemreg.exceptions import ClemregError
from napari_clemreg.clemreg.warp_image_volume import (
    _fit_tps,
    _rescale_affine_matrix,
    _warp_image_volume,
    _warp_image_volume_affine,
)

# A shear plus a shift: moving = fixed @ MATRIX.T + OFFSET.
MATRIX = np.array([[1.0, 0.1, 0.0],
                   [0.0, 1.0, 0.15],
                   [0.05, 0.0, 1.0]])
OFFSET = np.array([4.0, 6.0, 5.0])


def test_fit_tps_identity_transform_is_unchanged(rng):
    """Regression test for #9: the affine part of the solve was wrong."""
    points = rng.uniform(0, 10, size=(6, 3))
    query = rng.uniform(0, 10, size=(5, 3))

    np.testing.assert_allclose(_fit_tps(points, points)(query), query, atol=1e-6)


def test_fit_tps_recovers_a_known_affine_transform(rng):
    """Points related by an affine map give that map everywhere, not
    only at the points (#9)."""
    from_points = rng.uniform(0, 10, size=(6, 3))
    to_points = from_points @ MATRIX.T + OFFSET
    query = rng.uniform(0, 10, size=(5, 3))

    np.testing.assert_allclose(_fit_tps(from_points, to_points)(query), query @ MATRIX.T + OFFSET, atol=1e-6)


def test_fit_tps_merges_duplicate_points(rng):
    """A point listed twice is used once, with its targets averaged."""
    from_points = rng.uniform(0, 10, size=(10, 3))
    to_points = from_points @ MATRIX.T + OFFSET
    from_points = np.vstack([from_points, from_points[:2]])
    to_points = np.vstack([to_points, to_points[:2] + [[0, 0, 0], [2, 0, 0]]])

    tps = _fit_tps(from_points, to_points)

    np.testing.assert_allclose(tps(from_points[:1]), to_points[:1], atol=1e-6)
    np.testing.assert_allclose(tps(from_points[1:2]), to_points[1:2] + [[1, 0, 0]], atol=1e-6)


@pytest.mark.parametrize('points', [
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),          # too few
    np.array([[2, 0, 0], [2, 5, 0], [2, 0, 5], [2, 5, 5]]),  # all in one plane
])
def test_fit_tps_says_when_points_cannot_define_a_3d_warp(points):
    with pytest.raises(ClemregError, match='plane'):
        _fit_tps(points, points)


def test_warp_evaluates_the_spline_one_chunk_at_a_time(rng, monkeypatch):
    """Sub-division Factor limits memory: the spline is never evaluated
    at many more positions than one chunk holds."""
    output_shape = (20, 20, 20)
    moving_image, moving, transformed, expected = _affine_case(rng, output_shape)
    sizes = []

    def recording_fit(*args):
        tps = _fit_tps(*args)
        return lambda positions: sizes.append(len(positions)) or tps(positions)

    monkeypatch.setattr(warp_image_volume, '_fit_tps', recording_fit)
    warped, _ = _warp_image_volume(Image(moving_image), output_shape, moving, transformed,
                                   approximate_grid=1, sub_division_factor=4)

    assert max(sizes) <= 6 ** 3
    np.testing.assert_allclose(warped, expected, atol=1e-6)


def _affine_case(rng, output_shape, n_points=40):
    """A moving image, and point clouds related by the affine map above."""
    moving_image = rng.uniform(0, 1, size=(30, 60, 60))
    transformed = rng.uniform(0, np.array(output_shape) - 1, size=(n_points, 3))
    moving = transformed @ MATRIX.T + OFFSET
    expected = ndimage.affine_transform(moving_image, MATRIX, OFFSET, output_shape=output_shape, order=1)
    return moving_image, moving, transformed, expected


@pytest.mark.parametrize('output_shape, approximate_grid, sub_division_factor', [
    ((12, 40, 40), 5, 1),
    ((12, 40, 40), 1, 1),    # every voxel: crashed before
    ((9, 40, 40), 5, 1),     # thinner than two grid steps: warped as one slice before
    ((20, 30, 30), 10, 10),  # chunks smaller than the grid: crashed before
    ((20, 30, 30), 3, 4),    # chunks that don't divide the volume evenly
])
def test_warp_reproduces_an_affine_transform(rng, output_shape, approximate_grid, sub_division_factor):
    """With points related by an affine map, the warp is that map, for any
    grid spacing and chunking."""
    moving_image, moving, transformed, expected = _affine_case(rng, output_shape)

    warped, kwargs = _warp_image_volume(Image(moving_image, name='FM'), output_shape, moving, transformed,
                                        interpolation_order=1, approximate_grid=approximate_grid,
                                        sub_division_factor=sub_division_factor)

    assert warped.shape == output_shape
    assert kwargs['name'] == 'FM_warped'
    np.testing.assert_allclose(warped, expected, atol=1e-6)


def test_warp_moves_a_landmark_to_its_target(rng):
    """A non-linear warp brings a bright spot in the moving image to where
    the points say it belongs."""
    output_shape = (20, 40, 40)
    transformed = rng.uniform(0, np.array(output_shape) - 1, size=(60, 3))
    bend = 3 * np.sin(transformed / np.array(output_shape) * np.pi)
    moving = transformed + bend
    target = transformed[0]
    moving_image = np.zeros((30, 50, 50))
    moving_image[tuple(np.round(moving[0]).astype(int))] = 1
    moving_image = ndimage.gaussian_filter(moving_image, 2)

    warped, _ = _warp_image_volume(Image(moving_image), output_shape, moving, transformed,
                                   interpolation_order=1, approximate_grid=2)

    centre = np.array(ndimage.center_of_mass(warped * (warped > warped.max() / 2)))
    assert np.linalg.norm(centre - (target + np.round(moving[0]) - moving[0])) < 0.5


def test_warp_multichannel_matches_each_channel(rng):
    output_shape = (12, 40, 40)
    moving_image, moving, transformed, _ = _affine_case(rng, output_shape)
    channels = np.stack([moving_image, 2 * moving_image])

    warped, kwargs = _warp_image_volume(Image(channels, name='FM'), output_shape, moving, transformed)
    first, _ = _warp_image_volume(Image(moving_image), output_shape, moving, transformed)

    assert warped.shape == (2, *output_shape)
    assert [k['name'] for k in kwargs] == ['FM_warped_ch_0', 'FM_warped_ch_1']
    np.testing.assert_allclose(warped[0], first)
    np.testing.assert_allclose(warped[1], 2 * first)


def test_rescale_affine_matrix_is_unchanged_when_grids_all_match():
    # Raw, working and target pixel sizes all equal -> folding in the
    # (identity) resample scale factors should leave the matrix untouched.
    matrix = np.array([
        [0.9, -0.1, 0.0, 1.0],
        [0.1, 0.9, 0.0, -2.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    result = _rescale_affine_matrix(matrix, raw_pxlsz=(10.0, 10.0), working_pxlsz=10.0, target_pxlsz=(10.0, 10.0))
    np.testing.assert_allclose(result, matrix)


def test_rescale_affine_matrix_folds_in_pixel_size_ratios():
    # Identity registration (moving and fixed frames coincide in the
    # working grid) between a moving image at (40, 20) nm/px and a
    # 10nm working grid, targeting the moving image's own native
    # resolution as output -- with no actual registration transform,
    # reading raw data at its own resolution and writing it out at that
    # same resolution should require no rescaling at all.
    identity = np.eye(4)
    result = _rescale_affine_matrix(identity, raw_pxlsz=(40.0, 20.0), working_pxlsz=10.0, target_pxlsz=(40.0, 20.0))
    np.testing.assert_allclose(result, identity)


def test_direct_affine_warp_matches_resample_then_warp_then_downsample():
    """Cross-validates the two code paths in
    run_point_cloud_registration_and_warping(): warping directly from raw
    data via a rescaled matrix should agree with the old approach of
    resampling up to the working grid, warping there, then resampling
    the result back down -- for a pure-translation transform and an
    integer pixel-size ratio, order=0 interpolation should make the two
    agree exactly.
    """
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 255, (8, 8, 8), dtype=np.uint8).astype(float)

    raw_pxlsz = (20.0, 20.0)
    working_pxlsz = 10.0
    target_pxlsz = raw_pxlsz
    # A small translation, expressed in working-grid (10nm) voxel units.
    matrix = np.eye(4)
    matrix[:3, 3] = [4.0, -2.0, 6.0]

    # Old path: upsample raw -> working grid, warp there (to a fixed
    # working-grid output shape), then downsample back to target_pxlsz.
    zoom_up = working_pxlsz / np.array([raw_pxlsz[0], raw_pxlsz[1], raw_pxlsz[1]])
    from scipy import ndimage
    raw_on_working_grid = ndimage.zoom(raw, zoom_up, order=0)
    working_output_shape = raw_on_working_grid.shape
    warped_working = _warp_image_volume_affine(raw_on_working_grid, matrix,
                                               output_shape=working_output_shape,
                                               interpolation_order=0)
    zoom_down = working_pxlsz / np.array([target_pxlsz[0], target_pxlsz[1], target_pxlsz[1]])
    old_result = ndimage.zoom(warped_working, zoom_down, order=0)

    # New path: fold the resample steps into the matrix and warp raw
    # data directly to the target-grid output shape in one pass.
    extent = tuple(s * working_pxlsz for s in working_output_shape)
    coarse_output_shape = tuple(round(e / t) for e, t in zip(extent, (target_pxlsz[0], target_pxlsz[1], target_pxlsz[1])))
    matrix_combined = _rescale_affine_matrix(matrix, raw_pxlsz=raw_pxlsz, working_pxlsz=working_pxlsz,
                                             target_pxlsz=target_pxlsz)
    new_result = _warp_image_volume_affine(raw, matrix_combined, output_shape=coarse_output_shape,
                                           interpolation_order=0)

    assert new_result.shape == old_result.shape
    np.testing.assert_allclose(new_result, old_result)
