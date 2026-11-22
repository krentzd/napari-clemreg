"""Widget / integration smoke tests (modernisation plan §0.1.3).

These build the magicgui widgets and the npe2 manifest, but stop short of
`napari.viewer.Viewer()` (the `make_napari_viewer` fixture from napari's
own pytest plugin): on this machine, constructing a real Viewer under
`QT_QPA_PLATFORM=offscreen` segfaults inside vispy's OpenGL canvas setup
(napari/_vispy/canvas.py -> get_max_texture_sizes), which is an
environment/driver limitation, not something napari-clemreg controls.
CI (tox's `pytest-xvfb` on ubuntu-latest) has a real X server and GL
context, so a `make_napari_viewer`-based test belongs there -- this file
covers what's reliable locally: that each `magic_factory` widget builds
(exercises `on_init_specs.py` and every `on_init` callback) and that the
npe2 manifest + reader hook are wired correctly.
"""
from importlib.resources import files

import numpy as np
import pytest
import tifffile
from qtpy.QtWidgets import QApplication, QScrollArea

from napari_clemreg._reader import napari_get_reader
from napari_clemreg.widgets.run_registration import make_run_registration, make_run_registration_widget


@pytest.fixture
def qapp():
    """A QApplication, needed to construct any Qt/magicgui widget."""
    return QApplication.instance() or QApplication([])


def test_widget_factory_builds(qapp):
    assert make_run_registration() is not None


def test_dock_widget_factory_builds_scroll_area(qapp):
    """This is what napari.yaml actually registers -- confirm it builds
    and is scroll-bounded (see clemreg/_qt_layout.py), not just its
    underlying FunctionGui.

    Called with zero arguments, exactly how napari itself calls it
    (confirmed directly in napari's own source,
    _qnpe2._get_widget_viewer_param: viewer-injection-by-parameter-name
    only applies to class-based widgets, not plain functions like this
    wrapper -- "For magicgui type widget contributions, Viewer
    injection is done by magicgui.register_type instead", that code's
    own comment). A first version of this wrapper required
    `napari_viewer` with no default and crashed in the real app with
    exactly this call pattern, despite `factory(None)` passing here --
    this test calls it the same way napari does.
    """
    widget = make_run_registration_widget()
    assert isinstance(widget, QScrollArea)
    assert widget.widget() is not None


def test_pixelsize_quantity_edit_fields_build(qapp):
    """The pint-quantity pixel-size fields are the item the modernisation
    plan (§2.2) flags as most likely to break on a magicgui upgrade.
    """
    widget = make_run_registration()
    for name in [
        'moving_image_pixelsize_xy',
        'moving_image_pixelsize_z',
        'fixed_image_pixelsize_xy',
        'fixed_image_pixelsize_z',
    ]:
        field = getattr(widget, name)
        assert field.value.units == 'nanometer'


def test_advanced_fields_grouped_into_collapsed_sections(qapp):
    """The fields that used to sit behind one "Parameters custom"
    checkbox are now split into several QCollapsible sections (see
    clemreg/_qt_layout.py) -- confirm the reparenting didn't break field
    access, and that sections start collapsed (matching the old
    default-hidden state) rather than dumping every field's row into
    view at once.
    """
    from superqt import QCollapsible

    widget = make_run_registration()

    # Field values must still be readable/writable via the normal
    # magicgui API after being visually moved into a QCollapsible.
    widget.em_segmentation_backend.value = 'MitoNet (empanada-dl)'
    assert widget.em_segmentation_backend.value == 'MitoNet (empanada-dl)'
    widget.registration_voxel_size.value = 42
    assert widget.registration_voxel_size.value == 42

    sections = [
        w for w in widget.native.findChildren(QCollapsible)
    ]
    assert len(sections) == 5
    assert all(not section.isExpanded() for section in sections)


def test_npe2_manifest_loads():
    from npe2 import PluginManifest

    manifest = PluginManifest.from_file(str(files('napari_clemreg') / 'napari.yaml'))

    assert manifest.name == 'napari-clemreg'
    widget_commands = {w.command for w in manifest.contributions.widgets}
    declared_commands = {c.id for c in manifest.contributions.commands}
    # Every widget must reference a command that's actually declared.
    assert widget_commands <= declared_commands


def test_reader_recognises_tiff_and_rejects_other_extensions():
    assert napari_get_reader('somefile.tif') is not None
    assert napari_get_reader('somefile.tiff') is not None
    assert napari_get_reader('somefile.png') is None


def test_reader_returns_valid_layer_data_tuple(tmp_path):
    array = np.random.default_rng(0).integers(0, 255, size=(5, 10, 10), dtype=np.uint8)
    path = tmp_path / 'test.tif'
    tifffile.imwrite(path, array)

    reader = napari_get_reader(str(path))
    layer_data = reader(str(path))

    assert isinstance(layer_data, list)
    data, layer_kwargs, layer_type = layer_data[0]
    np.testing.assert_array_equal(data, array)
    assert layer_type == 'image'
    assert 'metadata' in layer_kwargs
