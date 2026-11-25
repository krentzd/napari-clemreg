import os

# torch (pip wheel, bundled libiomp5) and napari's conda-forge dependency
# stack (libomp) can load two OpenMP runtimes in the same process and
# segfault on import on macOS. Harmless on Linux CI, where this is a no-op.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow/network-heavy tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def empanada_missing(monkeypatch):
    """Make the MitoNet (empanada-dl) backend look uninstalled, whatever
    this environment has."""
    from napari_clemreg.clemreg import em_segmentation

    monkeypatch.setattr(em_segmentation, '_EMPANADA_PACKAGES', {'_clemreg_no_such_module': '_clemreg_no_such_module'})


@pytest.fixture
def seed():
    """A fixed seed for tests that need reproducible randomness."""
    value = 0
    np.random.seed(value)
    return value


@pytest.fixture
def rng():
    """A seeded numpy Generator for tests that need reproducible randomness."""
    return np.random.default_rng(0)


def make_blob_volume(shape, centers, radius, rng, noise_sigma=0.02):
    """Synthetic volume with Gaussian blobs at known centres, for
    characterisation tests. Generated (seeded) rather than committed as
    binary test data.
    """
    vol = np.zeros(shape, dtype=np.float64)
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
    for cz, cy, cx in centers:
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        vol += np.exp(-dist2 / (2 * radius**2))
    vol = gaussian_filter(vol, sigma=0.5)
    vol += rng.normal(0, noise_sigma, size=shape)
    return np.clip(vol, 0, None)


# Fixed synthetic dataset shared by characterisation tests: 5 well-separated
# blobs in a small volume. Centres/radius/seed are pinned so the resulting
# segmentation/registration counts are a stable regression oracle.
SYNTHETIC_BLOB_SHAPE = (40, 60, 60)
SYNTHETIC_BLOB_CENTERS = [(10, 15, 15), (10, 45, 15), (10, 15, 45), (10, 45, 45), (30, 30, 30)]
SYNTHETIC_BLOB_RADIUS = 4


@pytest.fixture
def synthetic_blob_volume():
    """A reproducible synthetic volume with SYNTHETIC_BLOB_CENTERS blobs."""
    return make_blob_volume(
        SYNTHETIC_BLOB_SHAPE,
        SYNTHETIC_BLOB_CENTERS,
        SYNTHETIC_BLOB_RADIUS,
        rng=np.random.default_rng(0),
    )
