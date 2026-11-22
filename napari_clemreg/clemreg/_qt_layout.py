"""Shared Qt layout helpers for restructuring magicgui-built widgets to
match how Crick's AI-on-Demand napari plugin (aiod_napari) actually lays
out its own forms -- verified against its real source, not guessed: a
QScrollArea-bounded panel (so the dock panel can never grow past the
screen, no matter how much content it holds) and settings grouped into
independently-collapsible sections (so expanding one group only lays out
that group's own handful of widgets, not the whole form at once) rather
than one flat form with a single all-or-nothing "advanced" checkbox.

These helpers reparent a magicgui FunctionGui's *existing* field widgets
after construction, rather than rebuilding the form from scratch -- so a
widget module keeps using ``@magic_factory`` for parameter declaration,
type-based binding and the call button exactly as before; only the visual
grouping changes.
"""
from __future__ import annotations

from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import QApplication, QScrollArea, QWidget
from superqt import QCollapsible


def extract_field_widget(gui, field_name: str) -> QWidget:
    """Return the Qt widget to move for one magicgui field: either the
    dedicated (label + control) row magicgui builds for a labeled field,
    or the field's own native widget if it has no separate label row
    (e.g. a CheckBox using inline ``text=`` instead of ``label=``).

    Confirmed directly: for a `layout='vertical'` Container, a labeled
    field's native widget is parented under its own per-field row QWidget
    (containing a QLabel plus the control); an inline-text field's native
    widget is parented directly under the Container's own native widget.
    Moving the row (not just the control) keeps the label attached.
    """
    field = getattr(gui, field_name)
    native = field.native
    parent = native.parent()
    return native if parent is gui.native else parent


def group_into_collapsible(gui, title: str, field_names: list, expanded: bool = False) -> QCollapsible:
    """Move a set of a FunctionGui's fields into one collapsible section.

    Reparents each named field's row widget out of the FunctionGui's own
    flat layout and into a new `QCollapsible`, which is then appended to
    the FunctionGui's layout in their place. The underlying magicgui
    `Widget` objects are untouched (still registered on `gui`, still
    readable via `.value`, still collected into the call button's
    arguments) -- only their visual parent changes.
    """
    section = QCollapsible(title)
    for name in field_names:
        section.addWidget(extract_field_widget(gui, name))
    if expanded:
        section.expand(animate=False)
    else:
        section.collapse(animate=False)
    gui.native.layout().addWidget(section)
    return section


# Explicit dark text colour -- confirmed live: napari's dark theme's
# default (white) text is unreadable against this light amber background
# without overriding it here too.
_AUTO_SET_STYLESHEET = 'background-color: #ffe8a3; color: #000000;'


def mark_auto_set(field_widget, value) -> None:
    """Set a layer-typed field's value as the result of running a
    pipeline step (rather than a deliberate user choice), and highlight
    it amber to signal that -- distinct from show_error()'s red, which
    already means "something's wrong" elsewhere in this plugin; this
    just means "review if this isn't what you wanted".

    Explicitly refreshes the field's own choices first: a magicgui
    CategoricalWidget's `.value` setter raises ValueError for a value
    not already in `.choices` (confirmed directly, no fallback), and
    `value` here is typically a layer that was only just added to the
    viewer -- relying on some other, external mechanism to have already
    refreshed choices by this point (e.g. napari's dock-widget-level
    layers.events wiring, whose own timing/ordering relative to this
    call isn't guaranteed) is fragile; reset_choices() here is cheap and
    makes this call self-sufficient regardless.
    """
    field_widget.reset_choices()
    field_widget.value = value
    field_widget.native.setStyleSheet(_AUTO_SET_STYLESHEET)


def clear_highlight_on_user_select(field_widget) -> None:
    """Revert a field's auto-set highlight the moment the user actually
    interacts with it. Connected to the underlying Qt widget's
    `activated` signal, which fires only on genuine user interaction --
    unlike magicgui's own `.changed`, which fires identically for a
    programmatic `.value = ...` assignment (mark_auto_set's own) and
    would immediately undo its own highlight if wired there instead.
    """
    field_widget.native.activated.connect(lambda *_: field_widget.native.setStyleSheet(''))


class MagicguiScrollArea(QScrollArea):
    """A QScrollArea holding a magicgui widget, docked in its place.

    napari keeps a docked widget's layer dropdowns in sync with the layer
    list only if the docked object itself has a ``reset_choices`` method
    (``QtMainWindow.add_dock_widget`` connects the viewer's layer
    inserted/removed/reordered events to it). A plain QScrollArea has
    none, so wrapping a magicgui widget in one silently stopped its
    dropdowns updating when layers were added. This forwards it.
    """

    def __init__(self, gui):
        super().__init__()
        # Also keeps the magicgui wrapper alive as long as its dock.
        self.gui = gui
        self.setWidget(gui.native)

    def reset_choices(self, *_):
        self.gui.reset_choices()

    def minimumSizeHint(self):
        # QScrollArea's own hint caps its width, so napari would dock the
        # form narrower than it is and squeeze it. Computed on each call, so
        # it reflects napari's stylesheet, which is applied after building.
        hint = super().minimumSizeHint()
        width = (self.widget().sizeHint().width()
                 + self.verticalScrollBar().sizeHint().width()
                 + 2 * self.frameWidth())
        return QSize(max(hint.width(), width), hint.height())


def wrap_in_scroll_area(gui, max_height_fraction: float = 0.85) -> MagicguiScrollArea:
    """Wrap the magicgui widget `gui` in a scroll area, so its dock panel
    never grows taller than the screen. Takes the magicgui widget, not its
    `.native` QWidget, so the scroll area can forward `reset_choices`."""
    scroll = MagicguiScrollArea(gui)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    screen = QApplication.primaryScreen()
    if screen is not None:
        available_height = screen.availableGeometry().height()
        scroll.setMaximumHeight(int(available_height * max_height_fraction))

    return scroll
