try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._dock_widget import napari_experimental_provide_dock_widget
from ._reader import napari_get_reader

# Set napari-clemreg reader as default
from napari.settings import get_settings
settings = get_settings()
settings.plugins.extension2reader = {'.tif': 'napari-clemreg', '.tiff': 'napari-clemreg'}
