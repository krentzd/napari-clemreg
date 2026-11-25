"""Isolation layer for napari internals without a stable public API.

`link_layers` has a public alias (`napari.experimental.link_layers`), used
here instead of the private import. `get_linked_layers` has no public
equivalent, so it stays a private import -- kept behind this module so
there's a single place to patch if napari moves or removes it.
"""
from napari.experimental import link_layers
from napari.layers.utils._link_layers import get_linked_layers

__all__ = ["link_layers", "get_linked_layers"]
