"""The EM segmentation core's backend dispatch, without running a backend."""
import sys

import numpy as np
import pytest

from napari_clemreg.clemreg import em_segmentation
from napari_clemreg.clemreg.em_segmentation import check_backend_installed, segment_em
from napari_clemreg.clemreg.exceptions import BackendNotInstalledError, ClemregError


def test_missing_empanada_says_what_to_do(empanada_missing):
    with pytest.raises(BackendNotInstalledError) as excinfo:
        check_backend_installed('empanada')

    message = str(excinfo.value)
    assert '`empanada` extra' in message
    assert 'missing: _clemreg_no_such_module' in message
    assert "backend='segment-flow'" in message
    assert excinfo.value.missing == ['_clemreg_no_such_module']
    assert isinstance(excinfo.value, ClemregError)


def test_missing_empanada_lists_every_missing_package(monkeypatch):
    monkeypatch.setattr(em_segmentation, '_EMPANADA_PACKAGES', {'_clemreg_missing_a': '_clemreg_missing_a',
                                                               '_clemreg_missing_b': '_clemreg_missing_b'})

    with pytest.raises(BackendNotInstalledError, match='missing: _clemreg_missing_a and _clemreg_missing_b'):
        check_backend_installed('empanada')


def test_missing_modules_are_reported_by_package_name(monkeypatch):
    """The `empanada` module comes from the `empanada-dl` package."""
    monkeypatch.setattr(em_segmentation, '_EMPANADA_PACKAGES', {'_clemreg_no_such_module': 'no-such-package'})

    with pytest.raises(BackendNotInstalledError, match='missing: no-such-package'):
        check_backend_installed('empanada')


def test_a_plain_directory_does_not_count_as_installed(monkeypatch, tmp_path):
    """A folder named `empanada` in the working directory is importable as
    a namespace package, but isn't empanada-dl."""
    (tmp_path / '_clemreg_namespace_only').mkdir()
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(em_segmentation, '_EMPANADA_PACKAGES', {'_clemreg_namespace_only': '_clemreg_namespace_only'})

    with pytest.raises(BackendNotInstalledError):
        check_backend_installed('empanada')


def test_segment_em_checks_before_importing_the_backend(empanada_missing, monkeypatch):
    """The empanada-dl backend imports torch at module level; segment_em
    must report the missing extra instead of a bare ModuleNotFoundError."""
    monkeypatch.delitem(sys.modules, 'napari_clemreg.clemreg.empanada_segmentation', raising=False)

    with pytest.raises(BackendNotInstalledError):
        segment_em(np.zeros((2, 4, 4)), backend='empanada')

    assert 'napari_clemreg.clemreg.empanada_segmentation' not in sys.modules


def test_segment_flow_has_no_python_module_check(empanada_missing):
    """Segment-Flow checks its own prerequisites when it runs."""
    check_backend_installed('segment-flow')


def test_unknown_backend_is_still_a_clemreg_error():
    with pytest.raises(ClemregError, match='Unknown EM segmentation backend'):
        segment_em(np.zeros((2, 4, 4)), backend='nope')
