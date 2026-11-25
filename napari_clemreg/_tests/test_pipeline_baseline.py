"""Pipeline output baseline: refactoring must not change the outputs.

Runs the pipeline through the four `widget_components.run_*` adapters on a
small synthetic dataset, and compares every output with values saved in
`data/pipeline_baseline.npz`.

The pipeline is deterministic (checked across separate processes), so
the tolerances are tight. They are not zero, because floating-point
results can differ very slightly between platforms and BLAS builds.

Point clouds are compared as *sets* of points (rows sorted before
comparing). open3d returns the same points in a different order on
Linux and macOS, while every image, matrix and segmentation matches
exactly, so the order isn't part of the result.

It covers:
- FM segmentation (LoG), with and without the size filter and a Mask ROI
- point cloud sampling, with FM and EM at different, anisotropic pixel sizes
- registration and warping for Rigid CPD, Affine CPD and BCPD
- the "EM pixel grid (legacy)" output option
- two linked FM channels (compared by name, because linked layers come
  back as a set with no fixed order)

EM segmentation isn't covered: it needs MitoNet. The EM -> FM registration
direction isn't in the baseline because it always crashes (#68); a strict
xfail below pins that. Once #68 is fixed, add an EM -> FM case to
REGISTRATION_CASES and regenerate.

To regenerate the saved values after a change that is *meant* to alter
results, run this module directly and say why in the PR:

    python -m napari_clemreg._tests.test_pipeline_baseline
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pint
import pytest
from napari.layers import Image, Labels, Shapes

BASELINE = Path(__file__).parent / 'data' / 'pipeline_baseline.npz'

SHAPE = (16, 48, 48)
CENTERS = [(6, 12, 12), (6, 36, 12), (6, 12, 36), (6, 36, 36), (10, 24, 24)]
FM_OFFSET = (0, 2, -1)
RADIUS = 4

FM_PIXEL_SIZE = {'xy': (40, 'nanometer'), 'z': (80, 'nanometer')}
EM_PIXEL_SIZE = {'xy': (20, 'nanometer'), 'z': (40, 'nanometer')}

SAMPLING = dict(point_cloud_sampling_frequency=30, voxel_size=1, point_cloud_sigma=1.0)
WARPING = dict(registration_max_iterations=30, warping_interpolation_order=1,
               warping_approximate_grid=5, warping_sub_division_factor=1)

FM_TO_EM = 'FM → EM'
EM_TO_FM = 'EM → FM'

# (name, algorithm, output resolution, direction, linked second FM channel)
REGISTRATION_CASES = [
    ('rigid', 'Rigid CPD', 'Native LM resolution', FM_TO_EM, False),
    ('affine', 'Affine CPD', 'Native LM resolution', FM_TO_EM, False),
    ('bcpd', 'BCPD', 'Native LM resolution', FM_TO_EM, False),
    ('rigid_em_grid', 'Rigid CPD', 'EM pixel grid (legacy)', FM_TO_EM, False),
    ('rigid_linked', 'Rigid CPD', 'Native LM resolution', FM_TO_EM, True),
]


def _blobs(centers, seed):
    from napari_clemreg._tests.conftest import make_blob_volume

    return make_blob_volume(SHAPE, centers, RADIUS, rng=np.random.default_rng(seed)).astype(np.float32)


def _quantity(ureg, value_unit):
    return ureg.Quantity(*value_unit)


def compute_outputs():
    """Run the pipeline scenarios and return {key: array}."""
    from napari_clemreg.clemreg import widget_components as wc
    from napari_clemreg.clemreg._napari_compat import link_layers

    ureg = pint.UnitRegistry()
    fm_xy, fm_z = _quantity(ureg, FM_PIXEL_SIZE['xy']), _quantity(ureg, FM_PIXEL_SIZE['z'])
    em_xy, em_z = _quantity(ureg, EM_PIXEL_SIZE['xy']), _quantity(ureg, EM_PIXEL_SIZE['z'])

    em = _blobs(CENTERS, seed=0)
    fm_centers = [tuple(c + o for c, o in zip(center, FM_OFFSET)) for center in CENTERS]
    fm = _blobs(fm_centers, seed=1) * 1000
    outputs = {}

    # FM segmentation (LoG), three ways.
    segmentation_cases = {
        'fm_seg': dict(Mask_ROI=None, filter_segmentation=False),
        'fm_seg_filtered': dict(Mask_ROI=None, filter_segmentation=True),
        'fm_seg_roi': dict(Mask_ROI=Shapes([np.array([[0, 4, 4], [0, 4, 30], [0, 30, 30], [0, 30, 4]])],
                                           shape_type='polygon'),
                           filter_segmentation=False),
    }
    for key, case in segmentation_cases.items():
        seg = wc.run_moving_segmentation(Image(fm, name='FM'), z_min=2, z_max=14, log_sigma=3.0,
                                         log_threshold=1.2, filter_size_lower=5, filter_size_upper=95,
                                         **case)
        outputs[key] = np.asarray(seg)

    # Point cloud sampling from binary segmentations of the two volumes.
    moving_points, fixed_points = wc.run_point_cloud_sampling(
        Labels((fm > 300).astype(np.uint8)), Labels((em > 0.3).astype(np.uint8)),
        fm_xy, fm_z, em_xy, em_z, **SAMPLING)
    outputs['moving_points'] = moving_points.data
    outputs['fixed_points'] = fixed_points.data

    for key, algorithm, resolution, direction, linked in REGISTRATION_CASES:
        moving_image = Image(fm, name='FM')
        fixed_image = Image(em, name='EM')
        if linked:
            second = Image(fm[:, ::-1, :].copy(), name='FM_second')
            link_layers([moving_image, second])
        warped, transformed = wc.run_point_cloud_registration_and_warping(
            moving_points, fixed_points, moving_image, fixed_image, algorithm, registration_direction=direction,
            warping_output_resolution=resolution, **WARPING)
        for layer in warped:
            outputs[f'{key}/{layer.name}/data'] = np.asarray(layer.data)
            outputs[f'{key}/{layer.name}/scale'] = np.asarray(layer.scale)
        outputs[f'{key}/transformed/data'] = np.asarray(transformed.data)
        outputs[f'{key}/transformed/affine'] = np.asarray(transformed.affine.affine_matrix)
    return outputs


@pytest.fixture(scope='module')
def outputs():
    return compute_outputs()


@pytest.fixture(scope='module')
def baseline():
    with np.load(BASELINE) as data:
        return dict(data)


def test_same_outputs_as_baseline(outputs, baseline):
    assert sorted(outputs) == sorted(baseline), 'set of outputs changed'


def _is_point_cloud(key):
    return key.endswith('_points') or key.endswith('/transformed/data')


def _sorted_rows(points):
    # Order by rounded coordinates, so tiny floating-point differences
    # can't swap two neighbouring points; then compare the full values.
    order = np.lexsort(np.round(points, 6).T[::-1])
    return points[order]


@pytest.mark.parametrize('key', sorted(np.load(BASELINE).files) if BASELINE.exists() else [])
def test_output_matches_baseline(outputs, baseline, key):
    actual, expected = outputs[key], baseline[key]
    if _is_point_cloud(key):
        actual, expected = _sorted_rows(actual), _sorted_rows(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if np.issubdtype(expected.dtype, np.integer) or expected.dtype == bool:
        np.testing.assert_array_equal(actual, expected)
    else:
        scale = max(float(np.abs(expected).max()), 1.0) if expected.size else 1.0
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5 * scale)


@pytest.mark.xfail(strict=True, raises=KeyError, reason="#68: EM -> FM direction crashes with KeyError: 'output_shape'")
def test_em_to_fm_direction_runs():
    from napari_clemreg.clemreg import widget_components as wc

    ureg = pint.UnitRegistry()
    em = _blobs(CENTERS, seed=0)
    fm = _blobs([tuple(c + o for c, o in zip(center, FM_OFFSET)) for center in CENTERS], seed=1) * 1000
    moving_points, fixed_points = wc.run_point_cloud_sampling(
        Labels((fm > 300).astype(np.uint8)), Labels((em > 0.3).astype(np.uint8)),
        _quantity(ureg, FM_PIXEL_SIZE['xy']), _quantity(ureg, FM_PIXEL_SIZE['z']),
        _quantity(ureg, EM_PIXEL_SIZE['xy']), _quantity(ureg, EM_PIXEL_SIZE['z']), **SAMPLING)

    warped, __ = wc.run_point_cloud_registration_and_warping(
        moving_points, fixed_points, Image(fm, name='FM'), Image(em, name='EM'), 'Rigid CPD',
        registration_direction=EM_TO_FM, **WARPING)

    assert warped[0].name == 'EM_warped'


if __name__ == '__main__':
    BASELINE.parent.mkdir(exist_ok=True)
    result = compute_outputs()
    np.savez_compressed(BASELINE, **result)
    print(f'Wrote {len(result)} arrays to {BASELINE} ({BASELINE.stat().st_size / 1024:.0f} KiB)')
