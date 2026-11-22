"""What a built wheel contains.

setup.cfg uses `packages = find:`, which skips folders without an
`__init__.py`. `widgets/` had none, so a normal (non-editable) install
couldn't import the plugin at all. The tests import the source tree, so
nothing else catches this.
"""
from pathlib import Path

from setuptools import find_packages

REPO = Path(__file__).resolve().parents[2]


def test_every_folder_with_code_or_config_is_packaged():
    packages = set(find_packages(where=REPO))
    folders = [
        folder for folder in (REPO / "napari_clemreg").rglob("*")
        if folder.is_dir() and folder.name != "__pycache__"
        and (any(folder.glob("*.py")) or any(folder.glob("*.yaml")))
    ]
    missing = [str(folder.relative_to(REPO)) for folder in folders
               if ".".join(folder.relative_to(REPO).parts) not in packages]
    assert missing == []
