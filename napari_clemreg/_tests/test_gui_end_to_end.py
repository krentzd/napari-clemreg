"""End-to-end run of Run Registration on the real sample data (slow).

Loads the benchmark sample (from Zenodo, and streamed as OME-Zarr from the
BioImage Archive) and the precomputed EM mask, then uses
the "Run this step" buttons to run FM segmentation, point cloud
sampling, and registration and warping, with the real pipeline. EM
segmentation is skipped by using the precomputed mask, so no MitoNet,
Segment-Flow or GPU is needed.

Takes about 2-3 minutes per sample, peaking at about 8 GB of memory. The first
run downloads about 600 MB of Zenodo data (cached by pooch afterwards);
the OME-Zarr sample is streamed again on every run.
Marked slow: run with `pytest --run-slow`. The weekly `slow-tests` CI
workflow does this.
"""
import numpy as np
import pytest
from napari.components import ViewerModel
from napari.layers import Layer

from napari_clemreg.clemreg.sample_data import (
    make_em_mask_sample_data,
    make_ome_zarr_sample_data,
    make_sample_data,
)
from napari_clemreg.widgets.run_registration import make_run_registration

STEP_TIMEOUT_MS = 15 * 60 * 1000


@pytest.fixture
def viewer(qtbot, monkeypatch):
    # napari.qt.threading binds current_viewer when it's first imported
    # (by the widget modules, at collection), so it keeps the real one,
    # which returns None here. Only the widgets see this ViewerModel.
    import napari.viewer

    model = ViewerModel()
    monkeypatch.setattr(napari.viewer, 'current_viewer', lambda: model)
    return model


@pytest.mark.slow
@pytest.mark.parametrize('make_benchmark', [make_sample_data, make_ome_zarr_sample_data],
                         ids=['zenodo', 'ome-zarr'])
def test_run_registration_step_by_step_on_sample_data(qtbot, viewer, make_benchmark):
    for data, kwargs, *layer_type in make_benchmark() + make_em_mask_sample_data():
        viewer.add_layer(Layer.create(data, kwargs, layer_type[0] if layer_type else 'image'))
    em = viewer.layers['EM']

    gui = make_run_registration()
    gui.Moving_Image.value = viewer.layers['FM_Mitotracker']
    gui.Fixed_Image.value = em

    gui.run_fm_segmentation_button.clicked.emit()
    qtbot.waitUntil(lambda: 'FM_segmentation' in viewer.layers, timeout=STEP_TIMEOUT_MS)
    assert gui.Moving_Segmentation.value is viewer.layers['FM_segmentation']

    gui.Fixed_Segmentation.value = viewer.layers['EM_mask']
    gui.run_point_cloud_sampling_button.clicked.emit()
    qtbot.waitUntil(lambda: 'Fixed_point_cloud' in viewer.layers, timeout=STEP_TIMEOUT_MS)
    assert len(viewer.layers['Moving_point_cloud'].data) > 1000
    assert len(viewer.layers['Fixed_point_cloud'].data) > 1000

    gui.run_registration_and_warping_button.clicked.emit()
    qtbot.waitUntil(lambda: 'FM_Mitotracker_warped' in viewer.layers, timeout=STEP_TIMEOUT_MS)
    warped = viewer.layers['FM_Mitotracker_warped']

    assert np.asarray(warped.data).max() > 0
    # Default "Native LM resolution" output: fewer pixels than EM, with a
    # scale that places it over the whole EM volume (in EM pixel units).
    assert warped.data.size < em.data.size
    extent = np.array(warped.data.shape) * np.array(warped.scale)
    np.testing.assert_allclose(extent, np.array(em.data.shape) * np.array(em.scale), rtol=0.05)
