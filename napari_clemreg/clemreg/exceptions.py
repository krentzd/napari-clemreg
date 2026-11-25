"""Exceptions raised by the CLEM-Reg pipeline.

Every pipeline error subclasses `ClemregError`, which subclasses
`Exception` directly and never `RuntimeError`. superqt's generator
workers (used by the Run Registration widget) treat a `RuntimeError`
as "the worker object was deleted" and swallow it without emitting
`errored` or `finished`, which left Register silently hung (#53).

See docs/design/package-split.md (decision D3). This module moves to
the `clemreg` core package in phase 2 of the split.
"""


class ClemregError(Exception):
    """Base class for errors raised by the CLEM-Reg pipeline."""


class NoSegmentationError(ClemregError):
    """Raised when segmentation finds nothing (every voxel has the same label).

    Replaces the 'No segmentation' string the adapters used to return,
    which callers had to remember to check for (#25, #50).
    """


class BackendNotInstalledError(ClemregError):
    """Raised when the chosen EM segmentation backend's optional packages
    (e.g. torch and empanada-dl, the `empanada` extra) aren't installed.
    `missing` lists them.
    """

    def __init__(self, message: str, missing=()):
        super().__init__(message)
        self.missing = list(missing)
