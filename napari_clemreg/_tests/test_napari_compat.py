"""Smoke test for the napari private-API isolation layer. If napari ever moves or removes get_linked_layers (no public
equivalent exists), this is the one place expected to break, so it gets
its own explicit test rather than relying on it being exercised
incidentally elsewhere.
"""
import inspect

from napari_clemreg.clemreg._napari_compat import link_layers, get_linked_layers


def test_link_layers_and_get_linked_layers_are_importable_and_callable():
    assert callable(link_layers)
    assert callable(get_linked_layers)
    # Both take a single layer/iterable-of-layers argument.
    assert len(inspect.signature(get_linked_layers).parameters) >= 1
