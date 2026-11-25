"""The core modules must not import napari or Qt (package split boundary).

The modules in CORE_MODULES are meant to become a napari-free `clemreg`
package. Until then they live in
`napari_clemreg/clemreg/`, and `napari_clemreg/__init__.py` imports the
widgets, so importing any of them normally pulls napari in regardless.
This test therefore imports them the way the future package will be
imported: in a fresh subprocess, under a bare stand-in for the
`napari_clemreg` package (its `__init__` isn't run), with the napari and
Qt packages blocked.

Modules still listed in NOT_YET_NAPARI_FREE are expected to fail
(strict xfail). Remove each one from the list in the PR that makes it
napari-free; the list must be empty by the end of phase 1 (#58).
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[1]

CORE_MODULES = [
    'exceptions',
    '_arrays',
    'log_segmentation',
    'mask_roi',
    'em_segmentation',
    'segment_flow_segmentation',
    'empanada_segmentation',
    'sample_data',
    'data_preprocessing',
    'point_cloud_sampling',
    'point_cloud_registration',
    'warp_image_volume',
]

NOT_YET_NAPARI_FREE = {
    'data_preprocessing',
    'point_cloud_sampling',
    'point_cloud_registration',
    'warp_image_volume',
}

BLOCKED = ('napari', 'magicgui', 'qtpy', 'superqt', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'vispy')

_PROBE = r'''
import importlib, importlib.abc, json, sys, types

BLOCKED = {blocked!r}

class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError(f'blocked import of {{name}}')
        return None

sys.meta_path.insert(0, _Blocker())

# Stand-ins for the packages, so their __init__ files (which import the
# widgets, and so napari) don't run.
for name, path in (('napari_clemreg', {package_dir!r}), ('napari_clemreg.clemreg', {package_dir!r} + '/clemreg')):
    stub = types.ModuleType(name)
    stub.__path__ = [path]
    sys.modules[name] = stub

results = {{}}
for module in {modules!r}:
    try:
        importlib.import_module('napari_clemreg.clemreg.' + module)
        results[module] = 'ok'
    except ImportError as exc:
        results[module] = ('blocked: ' if 'blocked import' in str(exc) else 'missing: ') + str(exc)
print(json.dumps(results))
'''


@pytest.fixture(scope='module')
def import_results():
    probe = _PROBE.format(blocked=BLOCKED, package_dir=str(PACKAGE_DIR), modules=CORE_MODULES)
    completed = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize('module', [
    pytest.param(m, marks=pytest.mark.xfail(strict=True, reason='not napari-free yet (#58)'))
    if m in NOT_YET_NAPARI_FREE else m
    for m in CORE_MODULES
])
def test_core_module_imports_without_napari_or_qt(import_results, module):
    result = import_results[module]
    if result.startswith('missing: '):
        # A heavy optional dependency (e.g. empanada-dl) isn't installed.
        # That's not a boundary violation.
        pytest.skip(result)
    assert result == 'ok', result
