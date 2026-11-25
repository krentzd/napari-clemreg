"""Sample-data caching and the bundled EM mask sample data.

The fast tests below just check the pooch registry is well-formed and
the bundled mask loads -- no network. The Zenodo end-to-end test
actually downloads/caches the real ~275MB + ~337MB files, so it's
marked slow and skipped unless `pytest --run-slow` is passed.
"""
import re
from pathlib import Path

import numpy as np
import pytest

from napari_clemreg.clemreg.sample_data import POOCH, make_em_mask_sample_data, make_sample_data

SHA256_HEX = re.compile(r"^sha256:[0-9a-f]{64}$")


def test_pooch_registry_is_well_formed():
    assert POOCH.base_url.startswith("https://")
    assert len(POOCH.registry) == 2
    for fname, hash_str in POOCH.registry.items():
        assert fname.endswith(".tif")
        assert SHA256_HEX.match(hash_str), f"{fname} hash doesn't look like sha256: {hash_str!r}"


def test_make_em_mask_sample_data():
    result = make_em_mask_sample_data()

    assert len(result) == 1
    mask, kwargs, layer_type = result[0]
    assert layer_type == 'labels'
    assert kwargs['name'] == 'EM_mask'
    assert mask.shape == (106, 1750, 1484)
    assert np.issubdtype(mask.dtype, np.integer)
    # Same instance-mask characteristics confirmed by direct inspection:
    # background (0) plus 9 distinct instance labels.
    assert len(np.unique(mask)) == 10


@pytest.mark.slow
def test_make_sample_data_returns_expected_layers():
    result = make_sample_data()

    assert len(result) == 5
    names = [kwargs["name"] for _, kwargs in result]
    assert names == ["EM", "FM_TGN46", "FM_Lysotracker", "FM_Mitotracker", "FM_Hoechst"]

    em_data, em_kwargs = result[0]
    assert em_data.shape == (106, 1750, 1484)
    assert em_data.dtype.name == "uint8"
    # The EM layer is the whole file, so plugins that work on files can find it
    # (aiod_napari reads metadata["path"] when a layer has no source path).
    assert Path(em_kwargs["metadata"]["path"]).name == "em_20nm_z_40_145.tif"
    assert Path(em_kwargs["metadata"]["path"]).is_file()

    # Where each layer starts in its file, for the MoBIE affine: the FM
    # layers skip the first 7 of 28 slices (0.13 µm apart).
    assert em_kwargs["metadata"]["origin_um"] == (0.0, 0.0, 0.0)
    for data, kwargs in result[1:]:
        assert data.ndim == 3
        assert kwargs["metadata"]["origin_um"] == pytest.approx((7 * 0.13, 0.0, 0.0))
        # FM layers are channel slices of a hyperstack: its path would be wrong.
        assert "path" not in kwargs["metadata"]
