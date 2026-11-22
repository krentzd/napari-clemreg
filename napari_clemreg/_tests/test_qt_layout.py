import numpy as np
from qtpy.QtWidgets import QApplication

from napari_clemreg.clemreg._qt_layout import mark_auto_set, clear_highlight_on_user_select


def _labels_field():
    """A Labels-typed field with a *static* choices tuple (set at
    construction, not the live get_layers(viewer) callable napari
    registers for this type) -- there's no real napari.Viewer in this
    test environment for that callable to find layers in (a real
    Viewer() segfaults here under headless/offscreen Qt on this
    machine), so the field's `_default_choices` is pinned directly to
    exactly what mark_auto_set()'s own reset_choices() call needs to
    find.
    """
    import napari
    from magicgui.widgets import create_widget

    viewer = napari.viewer.ViewerModel()
    layer_a = viewer.add_labels(np.zeros((4, 4, 4), dtype=np.uint8), name='layer_a')
    layer_b = viewer.add_labels(np.zeros((4, 4, 4), dtype=np.uint8), name='layer_b')

    field = create_widget(annotation=napari.layers.Labels, label='test',
                          options={'choices': (layer_a, layer_b)})
    return field, layer_a, layer_b


def test_mark_auto_set_sets_value_and_highlight():
    QApplication.instance() or QApplication([])
    field, layer_a, __ = _labels_field()

    mark_auto_set(field, layer_a)

    assert field.value is layer_a
    assert field.native.styleSheet() != ''


def test_mark_auto_set_refreshes_choices_before_setting_value():
    """Caught live while testing the per-section 'Run this step' buttons:
    CategoricalWidget.value's setter raises ValueError for a value not
    already in `.choices`, with no fallback (confirmed directly) -- and
    `value` here is typically a layer that was only just added to the
    viewer, so it can't be relied on to already be in `.choices` without
    mark_auto_set() refreshing them itself first.
    """
    import napari
    from magicgui.widgets import create_widget

    QApplication.instance() or QApplication([])
    viewer = napari.viewer.ViewerModel()
    # Choices start empty -- the new layer is added to `viewer` only
    # *after* the field is constructed, exactly like a pipeline step's
    # output.
    field = create_widget(annotation=napari.layers.Labels, label='test', options={'choices': ()})
    new_layer = viewer.add_labels(np.zeros((4, 4, 4), dtype=np.uint8), name='new_layer')

    # Simulate what a real, viewer-connected field's choices callable
    # would find, since there's no real napari.Viewer in this
    # environment to resolve get_layers(viewer) against.
    field._default_choices = (new_layer,)

    mark_auto_set(field, new_layer)

    assert field.value is new_layer


def test_programmatic_reassignment_does_not_clear_highlight():
    """mark_auto_set() itself does `field.value = ...` -- if the
    highlight-clearing handler were wired to magicgui's own `.changed`
    signal (fired identically for a programmatic assignment and genuine
    user interaction) instead of the underlying Qt widget's `activated`
    signal (fired only for real user interaction), it would immediately
    undo its own highlight the moment mark_auto_set() applied it.
    """
    QApplication.instance() or QApplication([])
    field, layer_a, __ = _labels_field()
    clear_highlight_on_user_select(field)

    mark_auto_set(field, layer_a)
    assert field.native.styleSheet() != ''

    field.value = layer_a
    assert field.native.styleSheet() != ''


def test_genuine_user_interaction_clears_highlight():
    QApplication.instance() or QApplication([])
    field, layer_a, __ = _labels_field()
    clear_highlight_on_user_select(field)

    mark_auto_set(field, layer_a)
    assert field.native.styleSheet() != ''

    field.native.activated.emit(1)
    assert field.native.styleSheet() == ''


def test_scroll_area_is_wide_enough_for_its_widget(qtbot):
    """QScrollArea's own size hint caps its width, so napari docked Run
    Registration narrower than its form, squeezing it and cutting off the
    unit fields. At its minimum width, under napari's stylesheet and with a
    vertical scrollbar, the form should get the width it asks for."""
    from napari.qt import get_stylesheet
    from napari_clemreg.clemreg._qt_layout import wrap_in_scroll_area
    from napari_clemreg.widgets.run_registration import make_run_registration

    scroll = wrap_in_scroll_area(make_run_registration())
    qtbot.addWidget(scroll)
    scroll.setStyleSheet(get_stylesheet('dark'))
    scroll.show()
    scroll.resize(scroll.minimumSizeHint().width(), 300)
    qtbot.waitUntil(scroll.verticalScrollBar().isVisible)

    assert scroll.viewport().width() >= scroll.widget().sizeHint().width()
