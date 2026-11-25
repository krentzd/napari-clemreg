"""Widget / integration smoke tests.

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
    """The pint-quantity pixel-size fields are the most likely to break on a
    magicgui upgrade."""
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


def _section(widget, title):
    from superqt import QCollapsible

    (section,) = [s for s in widget.native.findChildren(QCollapsible) if s.text() == title]
    return section


def _row(widget, field_name):
    """A field's label-and-control row (its parent, once grouped)."""
    return getattr(widget, field_name).native.parent()


def test_image_warping_sits_inside_point_cloud_registration(qapp):
    """Its settings are used by the Point Cloud Registration step's button,
    so they sit inside that section, above the button."""
    widget = make_run_registration()
    registration = _section(widget, 'Point Cloud Registration')
    warping = _section(widget, 'Image Warping')
    layout = registration.content().layout()

    button = widget.run_registration_and_warping_button.native
    (button_index,) = [i for i in range(layout.count())
                       if layout.itemAt(i).widget() in (button, button.parent())]

    assert registration.isAncestorOf(warping)
    assert layout.indexOf(warping) < button_index


_BCPD_ONLY = ['warping_approximate_grid', 'warping_sub_division_factor']
_ALL_ALGORITHMS = ['warping_interpolation_order', 'warping_output_resolution']


@pytest.mark.parametrize('algorithm, bcpd_rows_shown', [
    ('Rigid CPD', False), ('Affine CPD', False), ('BCPD', True)])
def test_bcpd_only_warping_settings_follow_the_algorithm(qapp, algorithm, bcpd_rows_shown):
    widget = make_run_registration()
    widget.registration_algorithm.value = 'BCPD'  # a real change for every case
    widget.registration_algorithm.value = algorithm

    for name in _BCPD_ONLY:
        assert _row(widget, name).isVisibleTo(widget.native) is bcpd_rows_shown
    for name in _ALL_ALGORITHMS:
        assert _row(widget, name).isVisibleTo(widget.native)


def test_bcpd_only_settings_start_hidden(qapp):
    widget = make_run_registration()  # Rigid CPD by default

    assert not any(_row(widget, name).isVisibleTo(widget.native) for name in _BCPD_ONLY)


def test_showing_bcpd_settings_in_an_open_section_does_not_clip_them(qapp):
    """An expanded QCollapsible fixes its height when it opens, and only
    recomputes it when a child resizes, not when a row is shown."""
    widget = make_run_registration()
    registration = _section(widget, 'Point Cloud Registration')
    warping = _section(widget, 'Image Warping')
    warping.expand(animate=False)
    registration.expand(animate=False)

    widget.registration_algorithm.value = 'BCPD'
    QApplication.processEvents()

    for section in (warping, registration):
        content = section.content()
        assert content.maximumHeight() >= content.sizeHint().height()
