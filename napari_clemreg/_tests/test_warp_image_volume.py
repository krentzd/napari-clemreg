import numpy as np

from napari_clemreg.clemreg.warp_image_volume import (
    _U,
    _make_L_matrix,
    _calculate_f,
    _make_warp,
    _rescale_affine_matrix,
    _warp_image_volume_affine,
)


def test_U_at_zero_is_zero():
    assert _U(np.array(0.0)) == 0.0


def test_U_matches_r_squared_log_r():
    x = np.array([1.0, 2.0, np.e])
    expected = (x**2) * np.log(x)
    np.testing.assert_allclose(_U(x), expected)


def test_make_L_matrix_shape_and_blocks(rng):
    points = rng.uniform(0, 10, size=(5, 3))
    n = len(points)
    L = _make_L_matrix(points)

    assert L.shape == (n + 4, n + 4)
    # K block (top-left) has zero diagonal: U(0) == 0 for coincident points
    np.testing.assert_allclose(np.diag(L[:n, :n]), 0.0)
    # K block is symmetric (interpoint distances are symmetric)
    np.testing.assert_allclose(L[:n, :n], L[:n, :n].T)
    # P block (top-right): column of ones followed by the points themselves
    np.testing.assert_allclose(L[:n, n], 1.0)
    np.testing.assert_allclose(L[:n, n + 1:], points)
    # O block (bottom-right) is zero
    np.testing.assert_allclose(L[n:, n:], 0.0)


def test_calculate_f_reduces_to_affine_when_weights_are_zero(rng):
    """With all point-weights zero, f(x,y,z) = a1 + ax*x + ay*y + az*z."""
    points = rng.uniform(0, 10, size=(6, 3))
    a1, ax, ay, az = 1.0, 2.0, -3.0, 0.5
    coeffs = np.array([0.0] * len(points) + [a1, ax, ay, az])

    x = rng.uniform(0, 10, size=(4,))
    y = rng.uniform(0, 10, size=(4,))
    z = rng.uniform(0, 10, size=(4,))

    result = _calculate_f(coeffs, points, x, y, z)
    expected = a1 + ax * x + ay * y + az * z
    np.testing.assert_allclose(result, expected)


def test_make_warp_identity_transform_is_unchanged(rng):
    """Identity transform => image unchanged (within interpolation tolerance).

    When from_points == to_points, _make_warp should be the identity
    everywhere.

    Regression test for a bug (#9) where the TPS right-hand side zeroed
    only 3 of the 4 rows needed to satisfy the affine-orthogonality
    constraint (`V[-3:, :] = 0` instead of `V[-4:, :] = 0`), corrupting
    the affine part of every solve.
    """
    points = rng.uniform(0, 10, size=(6, 3))
    query = rng.uniform(0, 10, size=(5,))
    x_vals, y_vals, z_vals = query, query[::-1].copy(), query * 0.5

    x_warp, y_warp, z_warp = _make_warp(points, points, x_vals, y_vals, z_vals)
    actual = np.stack([x_warp, y_warp, z_warp], axis=1)
    expected = np.stack([x_vals, y_vals, z_vals], axis=1)

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_make_warp_recovers_a_known_affine_transform(rng):
    """When from_points/to_points are related by an affine map, the TPS
    solution should have zero non-affine (bending) component and recover
    that same affine map exactly at arbitrary query points, not just at
    the control points themselves -- a stronger check than the identity
    invariant above, and another regression test for #9.
    """
    from_points = rng.uniform(0, 10, size=(6, 3))
    matrix = np.array([
        [1.2, 0.1, -0.05],
        [0.0, 0.9, 0.2],
        [0.1, 0.0, 1.1],
    ])
    offset = np.array([2.0, -1.0, 0.5])
    to_points = from_points @ matrix.T + offset

    query = rng.uniform(0, 10, size=(5,))
    x_vals, y_vals, z_vals = query, query[::-1].copy(), query * 0.5

    x_warp, y_warp, z_warp = _make_warp(from_points, to_points, x_vals, y_vals, z_vals)
    actual = np.stack([x_warp, y_warp, z_warp], axis=1)
    expected = np.stack([x_vals, y_vals, z_vals], axis=1) @ matrix.T + offset

    np.testing.assert_allclose(actual, expected, atol=1e-6)


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
