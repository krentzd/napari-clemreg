import numpy as np

from napari_clemreg.clemreg.point_cloud_registration import (
    _make_matrix_from_rigid_params,
    point_cloud_registration,
)


def test_make_matrix_from_rigid_params_translation_only():
    identity_rot = np.eye(3)
    trans = [1.0, 2.0, 3.0]
    mat = _make_matrix_from_rigid_params(identity_rot, trans, s=1.0)

    point = np.array([0.0, 0.0, 0.0, 1.0])
    result = mat @ point
    np.testing.assert_allclose(result[:3], trans)


def test_make_matrix_from_rigid_params_rotation_and_scale():
    # 90 degree rotation about z: (x, y, z) -> (-y, x, z)
    rot_90_z = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    scale = 2.0
    mat = _make_matrix_from_rigid_params(rot_90_z, trans=[0.0, 0.0, 0.0], s=scale)

    point = np.array([1.0, 0.0, 0.0, 1.0])
    result = mat @ point
    np.testing.assert_allclose(result[:3], [0.0, 2.0, 0.0], atol=1e-12)


def test_make_matrix_from_rigid_params_is_homogeneous():
    mat = _make_matrix_from_rigid_params(np.eye(3), [1.0, 2.0, 3.0], s=1.0)
    assert mat.shape == (4, 4)
    np.testing.assert_allclose(mat[3], [0.0, 0.0, 0.0, 1.0])


# Characterisation tests: apply a *known*
# transform to a synthetic point cloud and assert CPD recovers it. This is
# the cleanest available oracle for the registration step -- no golden file
# needed, and it exercises the real open3d/probreg dependency chain that's
# highest-risk under the planned numpy/open3d bump (§4).

def test_rigid_cpd_recovers_known_rigid_transform(rng):
    moving = rng.uniform(0, 20, size=(40, 3))

    theta = np.deg2rad(30)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = np.array([5.0, -3.0, 2.0])
    fixed = (moving @ rotation.T) + translation

    _, _, transformed, kwargs = point_cloud_registration(
        moving, fixed, algorithm='Rigid CPD', max_iterations=50
    )

    np.testing.assert_allclose(transformed, fixed, atol=1e-6)
    expected_affine = _make_matrix_from_rigid_params(rotation, translation, s=1.0)
    np.testing.assert_allclose(kwargs['affine'], expected_affine, atol=1e-6)


def test_affine_cpd_recovers_known_affine_transform(rng):
    moving = rng.uniform(0, 20, size=(40, 3))

    matrix = np.array([
        [1.2, 0.15, -0.05],
        [0.0, 0.9, 0.1],
        [0.05, 0.0, 1.1],
    ])
    offset = np.array([2.0, -1.0, 0.5])
    fixed = moving @ matrix.T + offset

    _, _, transformed, kwargs = point_cloud_registration(
        moving, fixed, algorithm='Affine CPD', max_iterations=50
    )

    np.testing.assert_allclose(transformed, fixed, atol=1e-6)
    expected_affine = np.eye(4)
    expected_affine[:3, :3] = matrix
    expected_affine[:3, 3] = offset
    np.testing.assert_allclose(kwargs['affine'], expected_affine, atol=1e-6)
