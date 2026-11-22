from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("napari-clemreg")
except PackageNotFoundError:
    __version__ = "unknown"

from .widgets.run_registration import make_run_registration

__all__ = (
    "make_run_registration",
)
