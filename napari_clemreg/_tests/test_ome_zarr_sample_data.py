"""The OME-Zarr variant of the benchmark sample data, streamed from the
BioImage Archive (S-BIAD4355).

The fast tests use small local stores laid out like the archive's (NGFF
0.4, Zarr v2, gzip, "/" chunk keys, `"filters": []`, "micron" units, one
image per FM channel), so they need no network. Each voxel holds its own
z index, so the tests can check which slices were kept. The slow test
streams the real data and compares it with the Zenodo sample.
"""
import functools
import json
import threading
import time
import warnings
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

import aiohttp
import dask.array as da
import numcodecs
import numpy as np
import pytest
import zarr

from napari_clemreg.clemreg import sample_data
from napari_clemreg.clemreg.data_preprocessing import get_pixelsize
from napari_clemreg.clemreg.exceptions import ClemregError
from napari_clemreg.clemreg.sample_data import OME_ZARR_URLS, make_ome_zarr_sample_data

FM_NAMES = ["FM_TGN46", "FM_Lysotracker", "FM_Mitotracker", "FM_Hoechst"]
# Distinct per axis, so a z/y/x mix-up shows.
EM_S2_SCALE = (0.03, 0.02, 0.01)
FM_S0_SCALE = (0.13, 0.035, 0.034)


def _write_ome_zarr(path, levels, dtype, unit, axes):
    """Write an NGFF 0.4 image. `levels` maps each level path to
    (shape, z chunk depth, scale); every voxel's value is its z index."""
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    datasets = []
    for name, (shape, z_chunk, scale) in levels.items():
        z = np.arange(shape[0]).reshape(-1, 1, 1)
        group.create_array(name, data=np.broadcast_to(z, shape).astype(dtype), chunks=(z_chunk, *shape[1:]),
                           chunk_key_encoding={"name": "v2", "separator": "/"},
                           compressors=numcodecs.GZip(-1))
        # As in the archive's copy; zarr warns about it.
        zarray = Path(path, name, ".zarray")
        zarray.write_text(json.dumps({**json.loads(zarray.read_text()), "filters": []}))
        datasets.append({"path": name, "coordinateTransformations": [{"type": "scale", "scale": list(scale)}]})
    group.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"type": "space", "name": axis, "unit": unit} for axis in axes],
        "datasets": datasets,
    }]
    return str(path)


@pytest.fixture
def local_urls(tmp_path):
    """Make local stores with the sample's layout: EM levels s0-s2 (only s2
    is read; it needs 146 slices for the crop) and 28-slice FM channels
    stored big-endian, with the archive's z chunk depths (56 and 13)."""
    def make(unit="micron", axes="zyx"):
        root = tmp_path / f"{unit}-{axes}"
        urls = {"em": _write_ome_zarr(root / "em.ome.zarr", {
            "s0": ((4, 4, 4), 4, (0.0075, 0.005, 0.0025)),
            "s1": ((4, 4, 4), 4, (0.015, 0.01, 0.005)),
            "s2": ((150, 4, 4), 56, EM_S2_SCALE),
        }, np.uint8, unit, axes)}
        for channel in ("golgi", "lyso", "mito", "dapi"):
            urls[channel] = _write_ome_zarr(root / f"{channel}.ome.zarr", {
                "s0": ((28, 4, 4), 13, FM_S0_SCALE),
            }, ">u2", unit, axes)
        return urls
    return make


class _FailingChunksHandler(SimpleHTTPRequestHandler):
    """Serves the metadata files; chunk requests hang for the server's
    `hang` seconds, then fail with a server error."""

    def send_head(self):
        if Path(self.path).name.startswith((".z", "zarr.json")):
            return super().send_head()
        time.sleep(self.server.hang)
        self.send_error(503)
        return None

    def log_message(self, *args):
        pass


@pytest.fixture
def served_urls(local_urls):
    """Serve local stores over HTTP, failing chunk requests (see
    _FailingChunksHandler). Returns a function taking `hang`."""
    servers = []

    def serve(hang=0):
        urls = local_urls()
        root = Path(urls["em"]).parent
        server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_FailingChunksHandler, directory=str(root)))
        server.hang = hang
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        base = f"http://127.0.0.1:{server.server_port}"
        return {key: f"{base}/{Path(url).name}" for key, url in urls.items()}

    yield serve
    for server in servers:
        server.shutdown()
        server.server_close()


def _layers(urls):
    return {kwargs["name"]: (data, kwargs) for data, kwargs in make_ome_zarr_sample_data(urls)}


def test_ome_zarr_urls_point_at_the_bioimage_archive_study():
    assert set(OME_ZARR_URLS) == {"em", "golgi", "lyso", "mito", "dapi"}
    for url in OME_ZARR_URLS.values():
        assert url.startswith("https://livingobjects.ebi.ac.uk/bioimaging-integrator-data/S-BIAD4355/")
        assert url.endswith(".ome.zarr")


def test_ome_zarr_sample_has_the_zenodo_samples_layers(local_urls):
    """Same names, order and display settings as `make_sample_data`, so the
    widgets treat both variants alike."""
    result = make_ome_zarr_sample_data(local_urls())

    assert [kwargs["name"] for _, kwargs in result] == ["EM", *FM_NAMES]
    assert [kwargs.get("colormap") for _, kwargs in result] == [None, "green", "magenta", "cyan", "blue"]
    assert all(kwargs["blending"] == "additive" for _, kwargs in result[1:])


def test_ome_zarr_sample_reads_no_pixels_when_opened(local_urls, monkeypatch):
    urls = local_urls()
    read = []
    original_get = zarr.storage.LocalStore.get

    async def recording_get(self, key, *args, **kwargs):
        read.append(key)
        return await original_get(self, key, *args, **kwargs)

    monkeypatch.setattr(zarr.storage.LocalStore, "get", recording_get)
    layers = _layers(urls)

    assert all(isinstance(data, da.Array) for data, _ in layers.values())
    assert read, "the spy saw no reads at all"
    assert all(Path(key).name.startswith((".z", "zarr.json")) for key in read), read


def test_ome_zarr_sample_em_is_the_zenodo_crop_of_level_s2(local_urls):
    """The Zenodo EM file is level s2 (20 nm), slices 40-145, which the
    bundled EM mask is aligned to. The crop spans the archive's 56-slice
    chunks."""
    urls = local_urls()
    em, kwargs = _layers(urls)["EM"]

    assert em.shape == (106, 4, 4)
    assert em.dtype == np.uint8
    np.testing.assert_array_equal(np.asarray(em)[:, 0, 0], np.arange(40, 146))
    assert kwargs["metadata"]["ome_zarr"] == {"url": urls["em"], "level": "s2", "z_slice": [40, 146]}


def test_ome_zarr_sample_fm_drops_the_first_7_slices_in_native_byte_order(local_urls):
    """As the Zenodo loader does (`fm[7:]`). The archive stores `>u2`; the
    layers are native uint16, as the Zenodo TIFF gives."""
    layers = _layers(local_urls())
    for name in FM_NAMES:
        data, _ = layers[name]
        assert data.shape == (21, 4, 4)
        assert data.dtype == np.dtype(np.uint16)
        assert data.dtype.isnative
        np.testing.assert_array_equal(np.asarray(data)[:, 0, 0], np.arange(7, 28))


def test_ome_zarr_sample_pixel_sizes_come_from_the_file(local_urls):
    """`get_pixelsize`, which every pipeline step uses, reads them from the
    layer metadata, in µm and (x, y, z) order."""
    layers = _layers(local_urls())

    assert get_pixelsize(layers["EM"][1]["metadata"]) == (*EM_S2_SCALE[::-1], "micron")
    for name in FM_NAMES:
        assert get_pixelsize(layers[name][1]["metadata"]) == (*FM_S0_SCALE[::-1], "micron")


def test_ome_zarr_sample_accepts_micrometer_units(local_urls):
    """NGFF's own name for the unit, in case the archive's metadata is fixed."""
    _, kwargs = _layers(local_urls(unit="micrometer"))["EM"]
    assert get_pixelsize(kwargs["metadata"])[:3] == EM_S2_SCALE[::-1]


def test_ome_zarr_sample_refuses_other_units(local_urls):
    with pytest.raises(ClemregError, match="nanometer"):
        make_ome_zarr_sample_data(local_urls(unit="nanometer"))


def test_ome_zarr_sample_refuses_other_axes(local_urls):
    with pytest.raises(ClemregError, match=r"\(z, y, x\)"):
        make_ome_zarr_sample_data(local_urls(axes="cyx"))


def test_ome_zarr_sample_opens_without_warnings(local_urls):
    """The archive's `"filters": []` makes zarr warn on every open."""
    urls = local_urls()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_ome_zarr_sample_data(urls)


def test_ome_zarr_sample_reports_a_store_it_cannot_open(local_urls, tmp_path):
    urls = {**local_urls(), "em": str(tmp_path / "gone.ome.zarr")}
    with pytest.raises(ClemregError, match="gone.ome.zarr"):
        make_ome_zarr_sample_data(urls)


def test_ome_zarr_sample_reports_a_server_it_cannot_reach(local_urls):
    """Port 9 on localhost refuses connections, as a server that's down does."""
    urls = {**local_urls(), "em": "http://127.0.0.1:9/em.ome.zarr"}
    with pytest.raises(ClemregError, match="internet connection"):
        make_ome_zarr_sample_data(urls)


def test_ome_zarr_sample_reports_a_server_error_as_unreachable(served_urls):
    """Not as missing data: the chunks may well be there."""
    with pytest.raises(ClemregError, match="internet connection"):
        make_ome_zarr_sample_data(served_urls())


def test_ome_zarr_sample_times_out_on_a_server_that_stops_answering(served_urls, monkeypatch):
    """The timeout reaches aiohttp: without it, napari froze for minutes."""
    monkeypatch.setattr(sample_data, "_HTTP_TIMEOUT", aiohttp.ClientTimeout(sock_read=0.5))
    urls = served_urls(hang=10)

    start = time.monotonic()
    with pytest.raises(ClemregError, match="internet connection"):
        make_ome_zarr_sample_data(urls)
    assert time.monotonic() - start < 5


def test_ome_zarr_sample_reports_a_missing_level(local_urls, monkeypatch):
    monkeypatch.setattr(sample_data, "_FM_LEVEL", "s9")
    with pytest.raises(ClemregError, match="no resolution level 's9'"):
        make_ome_zarr_sample_data(local_urls())


def test_ome_zarr_sample_reports_missing_image_data(local_urls):
    """zarr reads a missing chunk as zeros, which would give black layers."""
    urls = local_urls()
    for path in Path(urls["mito"], "s0").rglob("*"):
        if path.is_file() and not path.name.startswith(".z"):
            path.unlink()
    with pytest.raises(ClemregError, match="no image data"):
        make_ome_zarr_sample_data(urls)


def test_manifest_offers_both_benchmark_samples():
    from npe2 import PluginManifest

    manifest = PluginManifest.from_file(str(files("napari_clemreg") / "napari.yaml"))
    commands = {c.id: c.python_name for c in manifest.contributions.commands}
    samples = {s.key: s.command for s in manifest.contributions.sample_data}

    assert commands[samples["benchmark_dataset.1"]].endswith(":make_sample_data")
    assert commands[samples["benchmark_dataset_ome_zarr.1"]] == (
        "napari_clemreg.clemreg.sample_data:make_ome_zarr_sample_data")


@pytest.mark.slow
def test_streamed_ome_zarr_sample_matches_the_zenodo_sample():
    """Streams about 350 MB, and downloads the Zenodo files if they aren't
    cached. Checked 2026-11-20: the FM channels are identical; the EM is the
    same grid, but the Zenodo file's contrast is stretched (about
    1.22 x OME-Zarr - 65), so it's compared by correlation."""
    from napari_clemreg.clemreg.sample_data import make_sample_data

    zenodo = {kwargs["name"]: (data, kwargs) for data, kwargs in make_sample_data()}
    streamed = _layers(OME_ZARR_URLS)

    for name in FM_NAMES:
        np.testing.assert_array_equal(np.asarray(streamed[name][0]), zenodo[name][0])
        np.testing.assert_allclose(get_pixelsize(streamed[name][1]["metadata"])[:3],
                                   get_pixelsize(zenodo[name][1]["metadata"])[:3], rtol=1e-6)

    em = np.asarray(streamed["EM"][0])
    assert em.shape == zenodo["EM"][0].shape
    assert np.corrcoef(em[::10].ravel(), zenodo["EM"][0][::10].ravel())[0, 1] > 0.99
    np.testing.assert_allclose(get_pixelsize(streamed["EM"][1]["metadata"])[:3],
                               get_pixelsize(zenodo["EM"][1]["metadata"])[:3])
