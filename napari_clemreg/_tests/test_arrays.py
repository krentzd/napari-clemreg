"""Layer data reaches the core as numpy, computed safely from dask.

The streamed OME-Zarr sample gives dask layers. The pipeline computes
them in worker threads while napari may be slicing the same arrays, so
they must be computed with `callbacks=()`: dask's default swaps the
process-wide callback set, racing napari's slicing cache (napari-dev
skill, reference.md).
"""
import dask.array as da
import numpy as np
import pint
import pytest
from dask.callbacks import Callback
from napari.layers import Image, Labels, Points

import napari_clemreg.clemreg.em_segmentation as em_segmentation
import napari_clemreg.clemreg.log_segmentation as log_segmentation
from napari_clemreg.clemreg._arrays import as_array
from napari_clemreg.clemreg.data_preprocessing import return_isotropic_image_list
from napari_clemreg.clemreg.widget_components import (
    run_fixed_segmentation,
    run_moving_segmentation,
    run_point_cloud_registration_and_warping,
    run_point_cloud_sampling,
)


class _GlobalCallbackSpy(Callback):
    """Registered globally (as napari's slicing cache is) while active."""

    def __init__(self):
        super().__init__()
        self.started = 0

    def _start(self, dsk):
        self.started += 1


def test_as_array_computes_dask_without_global_callbacks():
    spy = _GlobalCallbackSpy()
    with spy:
        result = as_array(da.ones((4, 4), chunks=2))

    assert isinstance(result, np.ndarray)
    assert spy.started == 0


def test_as_array_reads_layer_data():
    data = np.arange(8).reshape(2, 4)
    np.testing.assert_array_equal(as_array(Image(da.from_array(data))), data)


@pytest.fixture
def received(monkeypatch):
    """Record the type of volume each core segmentation function gets."""
    types = {}

    def fake_segment_fm(volume, **kwargs):
        types['fm'] = type(volume)
        return np.ones_like(volume)

    def fake_segment_em(volume, backend):
        types['em'] = type(volume)
        return np.ones_like(volume)

    monkeypatch.setattr(log_segmentation, 'segment_fm', fake_segment_fm)
    monkeypatch.setattr(em_segmentation, 'segment_em', fake_segment_em)
    return types


def _dask_image():
    return Image(da.zeros((4, 8, 8), dtype=np.uint16, chunks=2))


def test_fm_segmentation_adapter_passes_numpy(received):
    run_moving_segmentation(_dask_image(), None, 0, 0, 3, 1.2, False, 1, 99)
    assert received['fm'] is np.ndarray


def test_em_segmentation_adapter_passes_numpy(received):
    run_fixed_segmentation(_dask_image(), em_segmentation_backend='MitoNet (empanada-dl)')
    assert received['em'] is np.ndarray


def test_isotropic_resampling_computes_dask_layers_without_global_callbacks():
    """The BCPD warp path reads the moving image (and linked layers) here."""
    layer = _dask_image()  # napari computes the contrast range here
    spy = _GlobalCallbackSpy()
    with spy:
        [image] = return_isotropic_image_list(layer, pxlsz_lm=(20, 10), pxlsz_em=(10, 10))

    assert isinstance(image.data, np.ndarray)
    assert spy.started == 0


def test_point_cloud_sampling_computes_dask_layers_without_global_callbacks():
    labels = np.zeros((8, 16, 16), dtype=np.uint8)
    labels[2:6, 4:12, 4:12] = 1
    moving, fixed = Labels(da.from_array(labels)), Labels(da.from_array(labels))
    um = pint.UnitRegistry().micrometer
    spy = _GlobalCallbackSpy()
    with spy:
        run_point_cloud_sampling(moving, fixed, 0.1 * um, 0.2 * um, 0.1 * um, 0.1 * um,
                                 point_cloud_sampling_frequency=100, voxel_size=1, point_cloud_sigma=1.0)

    assert spy.started == 0


def test_affine_warp_computes_dask_layers_without_global_callbacks():
    """The default (Rigid CPD, native resolution) warp path."""
    cloud = np.random.default_rng(0).uniform(0, 8, size=(30, 3))
    moving = Points(cloud, metadata={'pxlsz': (100, 100)})
    fixed = Points(cloud + 0.5, metadata={'pxlsz': (100, 100), 'output_shape': (4, 8, 8)})
    image = _dask_image()
    spy = _GlobalCallbackSpy()
    with spy:
        run_point_cloud_registration_and_warping(
            moving, fixed, image, image, 'Rigid CPD', registration_max_iterations=10,
            warping_interpolation_order=1, warping_approximate_grid=5, warping_sub_division_factor=1,
            registration_direction='FM → EM')

    assert spy.started == 0
