"""EM segmentation (napari-free core API).

Both backends run the same MitoNet model: through Crick's AI-on-Demand
Segment-Flow pipeline (the default, run by Nextflow in its own
environment), or in-process via `empanada-dl` (the `empanada` extra,
currently blocked on Python 3.11, #5). Each backend's module is only
imported when it's used, so neither is needed unless it's chosen.
"""
import importlib.util
import logging

import numpy as np

from .exceptions import BackendNotInstalledError, ClemregError, NoSegmentationError

logger = logging.getLogger(__name__)

BACKENDS = ('segment-flow', 'empanada')

# What the empanada backend imports in this process, and the package each
# comes from: {module: package}. Segment-Flow checks its own prerequisites
# (Nextflow, Conda, aiod_utils) when it runs.
_EMPANADA_PACKAGES = {'torch': 'torch', 'empanada': 'empanada-dl'}


def _is_installed(module: str) -> bool:
    """Whether `module` can be imported, without importing it. A plain
    directory of that name on sys.path (a namespace package) doesn't count."""
    spec = importlib.util.find_spec(module)
    return spec is not None and spec.origin is not None


def check_backend_installed(backend: str) -> None:
    """Raise BackendNotInstalledError if the backend's Python packages are
    missing. Doesn't import them (torch is slow to import)."""
    if backend != 'empanada':
        return
    missing = [package for module, package in _EMPANADA_PACKAGES.items() if not _is_installed(module)]
    if missing:
        raise BackendNotInstalledError(
            f"The {backend!r} EM segmentation backend needs the optional `empanada` extra "
            f"(missing: {' and '.join(missing)}), which can't currently be installed on "
            "Python 3.11 or newer (https://github.com/martlj/napari-clemreg/issues/5). "
            "Use backend='segment-flow' instead.",
            missing=missing,
        )


def segment_em(volume: np.ndarray, backend: str = 'segment-flow') -> np.ndarray:
    """Segment mitochondria in an EM volume with MitoNet.

    Parameters
    ----------
    volume : np.ndarray
        The EM volume, (z, y, x).
    backend : {'segment-flow', 'empanada'}
        How to run MitoNet.

    Returns
    -------
    np.ndarray
        The instance labels.

    Raises
    ------
    NoSegmentationError
        If nothing is segmented.
    BackendNotInstalledError
        If the backend's optional dependencies aren't installed.
    ClemregError
        If the backend is unknown, or fails (Segment-Flow's own errors
        subclass ClemregError).
    """
    check_backend_installed(backend)
    if backend == 'segment-flow':
        from .segment_flow_segmentation import segment_flow_em_segmentation

        logger.info('Running EM segmentation via MitoNet (Segment-Flow); progress is printed to the terminal.')
        seg_volume = segment_flow_em_segmentation(volume)
        logger.info('EM segmentation via Segment-Flow finished.')
    elif backend == 'empanada':
        from .empanada_segmentation import empanada_segmentation

        seg_volume = empanada_segmentation(input=volume, axis_prediction=False)
    else:
        raise ClemregError(f'Unknown EM segmentation backend {backend!r}; expected one of {BACKENDS}')

    if np.all(seg_volume == seg_volume.flat[0]):
        raise NoSegmentationError('No mitochondria found in the EM image')
    return seg_volume
