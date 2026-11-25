"""Smoke tests against a real napari Viewer, run on Linux CI only.

The other widget tests use a `ViewerModel` and never go through napari's
own plugin machinery. These open each dock widget the way napari does
(npe2 manifest -> `add_plugin_dock_widget`), which is where bugs like
the viewer-injection crash fixed in 9305d4b live.

A real Viewer segfaults under `QT_QPA_PLATFORM=offscreen` on some
machines (see test_dock_widget.py), and a segfault can't be skipped at
runtime, so these tests are opt-in: they only run when
`CLEMREG_REAL_VIEWER=1` is set. CI sets it on ubuntu-latest, where
pytest-xvfb provides a real X server and GL context.
"""
import os

import numpy as np
import pytest
import yaml

pytestmark = pytest.mark.skipif(
    os.environ.get('CLEMREG_REAL_VIEWER') != '1',
    reason='needs a real Viewer; set CLEMREG_REAL_VIEWER=1 (done on Linux CI)',
)


def _widget_display_names():
    from importlib.resources import files

    manifest = yaml.safe_load(files('napari_clemreg').joinpath('napari.yaml').read_text())
    return [w['display_name'] for w in manifest['contributions']['widgets']]


def _magicgui(dock_content):
    """The magicgui widget inside one of this plugin's scroll-area wrappers."""
    return dock_content.gui


@pytest.mark.parametrize('display_name', _widget_display_names())
def test_dock_widget_opens_in_real_viewer(make_napari_viewer, display_name):
    viewer = make_napari_viewer()

    dock, content = viewer.window.add_plugin_dock_widget('napari-clemreg', display_name)

    assert dock is not None
    assert _magicgui(content) is not None


def test_run_registration_sees_viewer_layers(make_napari_viewer):
    """The Layer dropdowns are filled from the viewer napari docks the
    widget into, including layers added after the widget opened."""
    viewer = make_napari_viewer()
    em = viewer.add_image(np.zeros((4, 16, 16)), name='EM')
    __, content = viewer.window.add_plugin_dock_widget('napari-clemreg', 'Run registration')
    gui = _magicgui(content)

    # No manual gui.reset_choices(): napari has to refresh the dropdowns
    # itself. It only does that if the docked widget has reset_choices,
    # which the QScrollArea wrapper lost until it forwarded it.
    fm = viewer.add_image(np.zeros((4, 16, 16)), name='FM')

    assert em in gui.Fixed_Image.choices
    assert fm in gui.Moving_Image.choices
