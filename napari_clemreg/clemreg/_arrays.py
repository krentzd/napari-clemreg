"""Helpers for core functions that take arrays but used to take napari layers."""
import dask.array as da
import numpy as np


def as_array(obj):
    """Return `obj` (an array, or anything with a `.data` array such as a
    napari layer) as a numpy array.

    Layers are accepted for backwards compatibility. Dask arrays are
    computed with `callbacks=()`: the default swaps dask's process-wide
    callbacks, which races napari's slicing cache when this runs in a
    worker thread.
    """
    if isinstance(obj, np.ndarray):
        return obj
    data = getattr(obj, 'data', obj)
    if isinstance(data, da.Array):
        return data.compute(callbacks=())
    return np.asarray(data)
