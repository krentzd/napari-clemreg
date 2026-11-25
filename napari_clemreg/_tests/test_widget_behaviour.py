"""Behaviour tests for the widgets: run them and check what they do.

test_dock_widget.py only checks that each widget builds. Almost every
widget bug fixed so far (#23, #25, #27, #29, and the viewer-injection
crash fixed in 9305d4b), plus #50, #53 and the open #47, is in code that only
runs once a widget is *called* or a button is *clicked*. These tests
cover that code.

They don't need a real napari window, so they avoid the local
offscreen-OpenGL segfault described in test_dock_widget.py:

- `viewer` is a `napari.components.ViewerModel` (layers, no canvas),
  installed as `napari.viewer.current_viewer()`. magicgui's napari
  integration uses that to fill Layer dropdowns and inject the viewer.
- `pipeline` swaps the four `widget_components.run_*` functions for fast
  stubs that record their keyword arguments. The widgets import these at
  call time, so patching the module attributes reaches all of them.
- `errors` captures `show_error` calls, which is how the widgets report
  problems to users.

Tests marked `xfail(strict=True)` pin a known, open bug to its issue.
When the bug is fixed the test starts passing, strict mode fails it, and
the marker needs removing.
"""
import json

import numpy as np
import pytest
from napari.components import ViewerModel
from napari.layers import Image, Labels, Points

import napari_clemreg.clemreg.widget_components as widget_components
from napari_clemreg.clemreg import em_segmentation
from napari_clemreg.clemreg.exceptions import NoSegmentationError
from napari_clemreg.widgets import run_registration
from napari_clemreg.widgets.run_registration import make_run_registration, make_run_registration_widget

SHAPE = (8, 32, 32)
TIMEOUT_MS = 5000


def _labels_array():
    labels = np.zeros(SHAPE, dtype=np.int64)
    labels[2:6, 8:20, 8:20] = 1
    return labels


def _points(name, **kwargs):
    return Points(np.random.default_rng(0).uniform(0, 30, size=(20, 3)), name=name, **kwargs)


# Rigid and Affine CPD return the moving points with the registration as
# the layer's affine.
REGISTRATION = np.array([[1.0, 0, 0, 2], [0, 0, -1, 30], [0, 1, 0, -5], [0, 0, 0, 1]])


class Pipeline:
    """Stand-ins for the widget_components.run_* functions.

    `calls[name]` is the list of kwargs each stub was called with. Set
    `fixed_result` / `moving_result` to change what the segmentation
    stubs return (an exception instance, such as NoSegmentationError, is
    raised instead, as the real adapters do), or `fixed_error` /
    `sampling_error` to make EM segmentation or point cloud sampling
    raise.
    """

    def __init__(self):
        self.calls = {name: [] for name in (
            'run_fixed_segmentation', 'run_moving_segmentation',
            'run_point_cloud_sampling', 'run_point_cloud_registration_and_warping')}
        self.fixed_result = _labels_array()
        self.moving_result = _labels_array()
        self.fixed_error = None
        self.sampling_error = None

    def run_fixed_segmentation(self, **kwargs):
        self.calls['run_fixed_segmentation'].append(kwargs)
        if self.fixed_error is not None:
            raise self.fixed_error
        if isinstance(self.fixed_result, Exception):
            raise self.fixed_result
        return self.fixed_result

    def run_moving_segmentation(self, **kwargs):
        self.calls['run_moving_segmentation'].append(kwargs)
        if isinstance(self.moving_result, Exception):
            raise self.moving_result
        return self.moving_result

    def run_point_cloud_sampling(self, **kwargs):
        self.calls['run_point_cloud_sampling'].append(kwargs)
        if self.sampling_error is not None:
            raise self.sampling_error
        # The real function's layer names; run_registration.py relies on
        # them to auto-set and highlight the next step's inputs.
        return _points('Moving_point_cloud'), _points('Fixed_point_cloud')

    def run_point_cloud_registration_and_warping(self, **kwargs):
        self.calls['run_point_cloud_registration_and_warping'].append(kwargs)
        warped = Image(np.zeros(SHAPE), name=kwargs['Moving_Image'].name + '_warped')
        return [warped], _points('transformed_points', affine=REGISTRATION)


@pytest.fixture
def viewer(qtbot, monkeypatch):
    import napari.viewer

    model = ViewerModel()
    monkeypatch.setattr(napari.viewer, 'current_viewer', lambda: model)
    return model


def _wait_for_workers(qtbot):
    """Wait until every thread_worker, including queued ones, has finished."""
    from qtpy.QtCore import QThreadPool

    pool = QThreadPool.globalInstance()
    qtbot.waitUntil(lambda: pool.waitForDone(50), timeout=TIMEOUT_MS)


@pytest.fixture(autouse=True)
def _no_real_em_backends(monkeypatch):
    """Fail loudly if any test reaches a real EM segmentation backend.

    A worker that outlives its test would otherwise find the real
    functions once the stubs are removed, and could launch a real
    Segment-Flow (Nextflow) run on a machine that has it installed.
    """
    from napari_clemreg.clemreg import segment_flow_segmentation

    def refuse(*args, **kwargs):
        raise AssertionError('a widget test reached the real Segment-Flow backend')

    monkeypatch.setattr(segment_flow_segmentation, 'segment_flow_em_segmentation', refuse)


@pytest.fixture
def pipeline(monkeypatch, qtbot):
    stubs = Pipeline()
    for name in stubs.calls:
        monkeypatch.setattr(widget_components, name, getattr(stubs, name))
    yield stubs
    # Before the stubs are removed: a test can finish (e.g. once one
    # segmentation step has reported its error) while other workers are
    # still queued or running. Left running, they'd call the real
    # pipeline, or crash once Qt objects are deleted.
    _wait_for_workers(qtbot)


@pytest.fixture
def errors(monkeypatch):
    messages = []
    for module in (run_registration, widget_components):
        monkeypatch.setattr(module, 'show_error', messages.append)
    return messages


@pytest.fixture
def empanada_installed(monkeypatch):
    """Make the MitoNet (empanada-dl) backend's dependency check pass,
    so tests can select it without torch or empanada-dl installed. The
    pipeline stubs never import them."""
    monkeypatch.setattr(em_segmentation, '_EMPANADA_PACKAGES', {})


@pytest.fixture
def images(viewer):
    """An FM and an EM image in the viewer, added before any widget is
    built so its Layer dropdowns pick them up."""
    rng = np.random.default_rng(0)
    fm = viewer.add_image(rng.random(SHAPE), name='FM')
    em = viewer.add_image(rng.random(SHAPE), name='EM')
    return fm, em


@pytest.fixture
def segmentations(viewer):
    fm_seg = viewer.add_labels(_labels_array(), name='FM_seg_existing')
    em_seg = viewer.add_labels(_labels_array(), name='EM_seg_existing')
    return fm_seg, em_seg


def _wait_for_layer(qtbot, viewer, name):
    qtbot.waitUntil(lambda: name in viewer.layers, timeout=TIMEOUT_MS)


def _is_highlighted(field):
    return field.native.styleSheet() != ''


# --- Docked widgets follow the layer list -----------------------------------

def test_docked_widget_picks_up_layers_added_later(qtbot, viewer):
    """napari refreshes a docked widget's layer dropdowns, when layers are
    added, removed or reordered, by calling the docked object's own
    reset_choices (QtMainWindow.add_dock_widget). napari.yaml's factory
    returns a QScrollArea around the magicgui widget, which must forward
    it, or the dropdowns never update. Checks a Labels dropdown, since
    Run Registration refreshes its image dropdowns itself.
    test_real_viewer.py checks the same through napari itself."""
    dock = make_run_registration_widget()
    layer = viewer.add_labels(np.zeros(SHAPE, dtype=np.int64), name='added_later')
    assert layer not in dock.gui.Fixed_Segmentation.choices

    dock.reset_choices(None)  # what napari calls, with the layer event

    assert layer in dock.gui.Fixed_Segmentation.choices
    assert dock.widget() is dock.gui.native


def _assert_reports_missing_empanada(qtbot, errors, pipeline):
    """The error says what to do, and nothing was started."""
    _wait_for_workers(qtbot)
    assert len(errors) == 1
    assert '`empanada` extra' in errors[0]
    assert 'MitoNet (Segment-Flow)' in errors[0]
    assert all(calls == [] for calls in pipeline.calls.values())


# --- Run Registration: the Register button -----------------------------------

def _assert_also_reraised(raised, exc_type):
    """Register connects its segmentation workers' `errored` handlers
    after creating them, not through thread_worker(connect=...), so
    superqt also re-raises the exception in the main thread. In napari,
    that means the user sees the error twice: once from show_error and
    once as napari's own traceback notification. Pin that behaviour here,
    so a change to it is noticed.
    """
    assert [type(exc) for __, exc, __ in raised] == [exc_type]


def _run_registration(images):
    fm, em = images
    gui = make_run_registration()
    gui.Moving_Image.value = fm
    gui.Fixed_Image.value = em
    return gui


def test_run_registration_does_not_preselect_one_image_for_both(qtbot, viewer):
    """Both dropdowns default to the first image layer, which put the EM
    sample in as the FM image too (with the EM's pixel sizes)."""
    em = viewer.add_image(np.zeros(SHAPE), name='EM')
    fm = viewer.add_image(np.zeros(SHAPE), name='FM')
    gui = make_run_registration()

    assert gui.Fixed_Image.value is em
    assert gui.Moving_Image.value is fm


def test_run_registration_separates_images_added_after_it_was_built(qtbot, viewer):
    """The usual order: dock the widget, then open the sample."""
    gui = make_run_registration()
    em = viewer.add_image(np.zeros(SHAPE), name='EM')
    fm = viewer.add_image(np.zeros(SHAPE), name='FM', metadata={'XResolution': 4, 'YResolution': 4})

    assert gui.Fixed_Image.value is em
    assert gui.Moving_Image.value is fm
    assert gui.moving_image_pixelsize_xy.value.to('micrometer').magnitude == pytest.approx(0.25)

    # Once separated, it stops: later layers don't override the user.
    gui.Moving_Image.value = em
    viewer.add_labels(np.zeros(SHAPE, dtype=np.int64), name='EM_segmentation')
    assert gui.Moving_Image.value is em


def test_run_registration_keeps_an_explicit_choice(qtbot, viewer):
    """Picking the same layer twice is the user's call; the pixel sizes
    must still match what the dropdown shows."""
    em = viewer.add_image(np.zeros(SHAPE), name='EM', metadata={'XResolution': 50, 'YResolution': 50})
    viewer.add_image(np.zeros(SHAPE), name='FM', metadata={'XResolution': 4, 'YResolution': 4})
    gui = make_run_registration()

    gui.Moving_Image.value = em
    viewer.add_labels(np.zeros(SHAPE, dtype=np.int64), name='EM_segmentation')

    assert gui.Moving_Image.value is em
    assert gui.moving_image_pixelsize_xy.value.to('micrometer').magnitude == pytest.approx(0.02)


def test_register_runs_every_step_and_adds_warped_image(qtbot, viewer, pipeline, images):
    gui = _run_registration(images)
    gui.registration_voxel_size.value = 42
    gui.warping_output_resolution.value = 'EM pixel grid (legacy)'

    gui()

    _wait_for_layer(qtbot, viewer, 'FM_warped')
    assert all(len(calls) == 1 for calls in pipeline.calls.values())
    assert pipeline.calls['run_point_cloud_sampling'][0]['voxel_size'] == 42
    warp_kwargs = pipeline.calls['run_point_cloud_registration_and_warping'][0]
    assert warp_kwargs['warping_output_resolution'] == 'EM pixel grid (legacy)'


def test_register_shows_the_transformed_points_where_they_were_registered(qtbot, viewer, pipeline, images):
    """The layer Register adds keeps the registration, so the transformed
    points overlay the fixed points."""
    gui = _run_registration(images)

    gui()

    _wait_for_layer(qtbot, viewer, 'transformed_points')
    np.testing.assert_allclose(viewer.layers['transformed_points'].affine.affine_matrix, REGISTRATION)


def test_register_reports_em_failure_and_does_not_register(qtbot, viewer, pipeline, errors, images):
    """#23: a failed segmentation used to leave Register silently hung."""
    pipeline.fixed_error = ValueError('boom')
    gui = _run_registration(images)

    with qtbot.capture_exceptions() as raised:
        gui()
        qtbot.waitUntil(lambda: any('EM segmentation failed' in m for m in errors), timeout=TIMEOUT_MS)
        qtbot.wait(200)

    assert pipeline.calls['run_point_cloud_sampling'] == []
    _assert_also_reraised(raised, ValueError)


def test_register_reports_segment_flow_failure(qtbot, viewer, pipeline, errors, images):
    """#53: Segment-Flow's errors used to subclass RuntimeError, which
    superqt's generator workers swallow, so Register silently hung."""
    from napari_clemreg.clemreg.segment_flow_segmentation import SegmentFlowNotAvailable

    pipeline.fixed_error = SegmentFlowNotAvailable('Nextflow not found')
    gui = _run_registration(images)

    with qtbot.capture_exceptions() as raised:
        gui()
        qtbot.waitUntil(lambda: any('Nextflow not found' in m for m in errors), timeout=TIMEOUT_MS)
        qtbot.wait(200)

    assert pipeline.calls['run_point_cloud_sampling'] == []
    _assert_also_reraised(raised, SegmentFlowNotAvailable)


def test_register_reports_runtime_error_from_segmentation(qtbot, viewer, pipeline, errors, images):
    """#53: any other RuntimeError (from numpy, torch, ...) must be
    reported too, not swallowed."""
    from napari_clemreg.clemreg.exceptions import ClemregError

    pipeline.fixed_error = RuntimeError('CUDA out of memory')
    gui = _run_registration(images)

    with qtbot.capture_exceptions() as raised:
        gui()
        qtbot.waitUntil(lambda: any('CUDA out of memory' in m for m in errors), timeout=TIMEOUT_MS)
        qtbot.wait(200)

    assert pipeline.calls['run_point_cloud_sampling'] == []
    _assert_also_reraised(raised, ClemregError)


def test_register_surfaces_runtime_error_from_registration(qtbot, viewer, pipeline, images):
    """#53: a RuntimeError in the registration worker (e.g. from
    open3d) must reach the user rather than end the run silently."""
    from napari_clemreg.clemreg.exceptions import ClemregError

    pipeline.sampling_error = RuntimeError('open3d failed')
    gui = _run_registration(images)

    with qtbot.capture_exceptions() as raised:
        gui()
        qtbot.waitUntil(lambda: bool(raised), timeout=TIMEOUT_MS)

    assert [type(exc) for __, exc, __ in raised] == [ClemregError]
    assert 'open3d failed' in str(raised[0][1])
    assert 'FM_warped' not in viewer.layers


def test_register_reports_no_segmentation(qtbot, viewer, pipeline, errors, images):
    """#25: no segmentation used to crash with an unrelated AttributeError."""
    pipeline.moving_result = NoSegmentationError('No mitochondria found in the FM image')
    gui = _run_registration(images)

    with qtbot.capture_exceptions() as raised:
        gui()
        qtbot.waitUntil(lambda: any('No mitochondria found' in m for m in errors), timeout=TIMEOUT_MS)

    assert pipeline.calls['run_point_cloud_sampling'] == []
    _assert_also_reraised(raised, NoSegmentationError)


def test_register_reports_missing_empanada_before_starting(qtbot, viewer, pipeline, errors, images,
                                                          empanada_missing):
    gui = _run_registration(images)
    gui.em_segmentation_backend.value = 'MitoNet (empanada-dl)'

    gui()

    _assert_reports_missing_empanada(qtbot, errors, pipeline)


# --- Run Registration: the "Run this step" buttons ---------------------------

def test_em_step_auto_sets_and_highlights_next_input(qtbot, viewer, pipeline, images):
    gui = _run_registration(images)

    gui.run_em_segmentation_button.clicked.emit()

    _wait_for_layer(qtbot, viewer, 'EM_segmentation')
    qtbot.waitUntil(lambda: gui.run_em_segmentation_button.native.isEnabled(), timeout=TIMEOUT_MS)
    assert gui.Fixed_Segmentation.value is viewer.layers['EM_segmentation']
    assert _is_highlighted(gui.Fixed_Segmentation)


def test_fm_step_auto_sets_and_highlights_next_input(qtbot, viewer, pipeline, images):
    gui = _run_registration(images)

    gui.run_fm_segmentation_button.clicked.emit()

    _wait_for_layer(qtbot, viewer, 'FM_segmentation')
    assert gui.Moving_Segmentation.value is viewer.layers['FM_segmentation']
    assert _is_highlighted(gui.Moving_Segmentation)


def test_point_cloud_step_runs_from_existing_segmentations(qtbot, viewer, pipeline, images, segmentations):
    fm_seg, em_seg = segmentations
    gui = _run_registration(images)
    gui.Moving_Segmentation.value = fm_seg
    gui.Fixed_Segmentation.value = em_seg

    gui.run_point_cloud_sampling_button.clicked.emit()

    _wait_for_layer(qtbot, viewer, 'Fixed_point_cloud')
    (kwargs,) = pipeline.calls['run_point_cloud_sampling']
    assert kwargs['Moving_Segmentation'] is fm_seg
    assert kwargs['Fixed_Segmentation'] is em_seg
    assert gui.Moving_Points.value is viewer.layers['Moving_point_cloud']
    assert _is_highlighted(gui.Fixed_Points)


def test_registration_step_adds_warped_image(qtbot, viewer, pipeline, images):
    gui = _run_registration(images)
    gui.Moving_Points.value = viewer.add_layer(_points('Moving_point_cloud'))
    gui.Fixed_Points.value = viewer.add_layer(_points('Fixed_point_cloud'))

    gui.run_registration_and_warping_button.clicked.emit()

    _wait_for_layer(qtbot, viewer, 'FM_warped')


def test_step_button_reports_error_and_is_re_enabled(qtbot, viewer, pipeline, errors, images):
    pipeline.fixed_error = RuntimeError('boom')
    gui = _run_registration(images)

    gui.run_em_segmentation_button.clicked.emit()

    qtbot.waitUntil(lambda: any('EM segmentation failed' in m for m in errors), timeout=TIMEOUT_MS)
    assert gui.run_em_segmentation_button.native.isEnabled()


def test_em_step_reports_missing_empanada(qtbot, viewer, pipeline, errors, images, empanada_missing):
    gui = _run_registration(images)
    gui.em_segmentation_backend.value = 'MitoNet (empanada-dl)'

    gui.run_em_segmentation_button.clicked.emit()

    _assert_reports_missing_empanada(qtbot, errors, pipeline)
    assert gui.run_em_segmentation_button.native.isEnabled()


# --- Known, open GUI bugs -----------------------------------------------------

@pytest.mark.xfail(strict=True, raises=NameError, reason='#47: Save parameters crashes Register')
def test_register_saves_parameters_to_json(qtbot, viewer, pipeline, images, tmp_path):
    path = tmp_path / 'params.json'
    gui = _run_registration(images)
    gui.save_json.value = True
    gui.save_json_path.value = path

    gui()

    assert json.loads(path.read_text())['log_sigma'] == gui.log_sigma.value


@pytest.mark.xfail(strict=True, reason='#47: Parameters from JSON never reads the file')
def test_register_loads_parameters_from_json(qtbot, viewer, pipeline, images, tmp_path):
    path = tmp_path / 'params.json'
    path.write_text(json.dumps({'log_sigma': 7.0}))
    gui = _run_registration(images)
    gui.params_from_json.value = True
    gui.load_json_file.value = path

    gui()

    qtbot.waitUntil(lambda: pipeline.calls['run_moving_segmentation'], timeout=TIMEOUT_MS)
    assert pipeline.calls['run_moving_segmentation'][0]['log_sigma'] == 7.0


def test_three_axis_option_removed(qtbot):
    # #48: Prediction Across Three Axis was removed rather than hidden.
    gui = make_run_registration()

    assert 'em_seg_axis' not in [w.name for w in gui]


@pytest.mark.xfail(strict=True, reason='#49: Voxel Size sits under Point Cloud Registration')
def test_voxel_size_is_in_point_cloud_sampling_section(qtbot):
    from superqt import QCollapsible

    gui = make_run_registration()
    voxel_size = gui.registration_voxel_size.native
    (section,) = [s for s in gui.native.findChildren(QCollapsible) if s.isAncestorOf(voxel_size)]

    assert section.text() == 'Point Cloud Sampling'

