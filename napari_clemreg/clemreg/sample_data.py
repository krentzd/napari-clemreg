from __future__ import annotations

import importlib.resources
import warnings

import aiohttp
import dask.array as da
import fsspec
import pooch
import tifffile
import zarr
from bioio import PhysicalPixelSizes

from .exceptions import ClemregError

# Cached under the OS-appropriate app-cache directory (~/.cache/napari-clemreg
# on Linux, ~/Library/Caches/napari-clemreg on macOS, %LOCALAPPDATA%\napari-clemreg
# on Windows) so these ~275MB/~337MB files are downloaded once, not on every
# call. Hashes pin the exact archived Zenodo record 7936982 files, verified on
# fetch so a corrupted/interrupted download is caught rather than silently used.
POOCH = pooch.create(
    path=pooch.os_cache("napari-clemreg"),
    base_url="https://zenodo.org/record/7936982/files/",
    registry={
        "em_20nm_z_40_145.tif": "sha256:805df382bee3c31a58b44e670d59eb2b299fe54a507151d4aff2f095d19c0f2a",
        "EM04468_2_63x_pos8T_LM_raw.tif": "sha256:3afd3730fcafebf4355a494e36c04c338dad2cf84f0d970c4e5f653435c51131",
    },
)

# The FM channels, in the Zenodo hyperstack's channel order:
# (key in OME_ZARR_URLS, layer name, colormap).
_FM_CHANNELS = [
    ("golgi", "FM_TGN46", "green"),
    ("lyso", "FM_Lysotracker", "magenta"),
    ("mito", "FM_Mitotracker", "cyan"),
    ("dapi", "FM_Hoechst", "blue"),
]


def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    print('Loading EM...')
    em_path = POOCH.fetch('em_20nm_z_40_145.tif', progressbar=True)
    em = tifffile.imread(em_path)
    print('Loading FM...')
    fm = tifffile.imread(POOCH.fetch('EM04468_2_63x_pos8T_LM_raw.tif', progressbar=True))

    # tifffile already parses this ImageJ hyperstack's axes as (Z, C, Y, X) --
    # confirmed directly (raw shape (28, 4, 1226, 1226), matching the
    # ImageDescription's slices=28/channels=4). No axis reordering is needed;
    # index channel on axis 1 and the z-range on axis 0.

    fm_metadata = {'ImageDescription': 'ImageJ=1.53t\nimages=112\nchannels=4\nslices=28\nhyperstack=true\nmode=grayscale\nunit=micron\nspacing=0.13\nloop=false\nmin=0.0\nmax=65535.0\n',
                   'XResolution': 28.349506,
                   'YResolution': 28.349506}
    # 'path': the EM layer is the whole file, unmodified, so record where it is.
    # Plugins that run on files rather than layer data (aiod_napari, which
    # sends images to Segment-Flow) find it here, as they would for a layer
    # opened from disk (its sample loader sets the same key for this file).
    # The FM layers are single channels sliced out of a 4-channel hyperstack,
    # so the file's path would be wrong for them; they get none.
    em_metadata = {'ImageDescription': '\nunit=micron\nspacing=0.02\n',
                   'XResolution': 50,
                   'YResolution': 50,
                   'path': em_path}

    fm_layers = [(fm[7:, channel], {'blending': 'additive', 'colormap': colormap, 'name': name, 'metadata': fm_metadata})
                 for channel, (_, name, colormap) in enumerate(_FM_CHANNELS)]
    return [(em, {'name': 'EM', 'metadata': em_metadata}), *fm_layers]


# The same benchmark data as OME-Zarr (NGFF 0.4, Zarr v2) on the BioImage
# Archive's S3 store, study S-BIAD4355: the whole FIB-SEM pyramid (5 nm at
# s0) and one image per FM channel, with pixel sizes in the metadata.
OME_ZARR_BASE_URL = "https://livingobjects.ebi.ac.uk/bioimaging-integrator-data/S-BIAD4355/"
OME_ZARR_URLS = {
    "em": OME_ZARR_BASE_URL + "87faf390-a7b4-4010-b716-057a89806842/3996bf7c-c174-475d-8504-cd571ea14b2b.ome.zarr",
    "golgi": OME_ZARR_BASE_URL + "c52de4e0-45b8-4424-a747-4ae79bcc8d25/5e8cbd5c-47fa-46f8-84d1-a90d4a056457.ome.zarr",
    "lyso": OME_ZARR_BASE_URL + "56d837b9-5a28-4c7f-9e24-ffb96a3f563b/ef1f5683-503d-4fac-9cb5-d2f36dd2d155.ome.zarr",
    "mito": OME_ZARR_BASE_URL + "d763e8fe-f6bc-4ef2-b99a-b318d5f2ea7c/50a92a4a-a381-423c-8ba3-a174ef58705e.ome.zarr",
    "dapi": OME_ZARR_BASE_URL + "378737ac-ec0a-4dd6-9cae-4b86caaaa48d/dac0d3d2-40c1-4b71-bda6-0fea44fa0ec1.ome.zarr",
}
# Level and z range matching the Zenodo files, so the bundled EM mask lines
# up and both variants give the same layer shapes: the Zenodo EM file is
# level s2 (20 nm), slices 40-145. Its contrast is stretched relative to
# the archive's copy, so the EM pixels differ; the FM pixels are identical.
_EM_LEVEL, _EM_Z = "s2", slice(40, 146)
_FM_LEVEL, _FM_Z = "s0", slice(7, None)
_MICROMETRE_UNITS = {"micron", "micrometer"}
# Without these, a host that drops packets freezes napari for aiohttp's
# default 5 minutes.
_HTTP_TIMEOUT = aiohttp.ClientTimeout(sock_connect=10, sock_read=60)


def _pixel_sizes(url, attrs, level):
    """Check an NGFF 0.4 (z, y, x) image's metadata, and return `level`'s
    pixel sizes in µm."""
    if "multiscales" not in attrs:
        raise ClemregError(f"{url} isn't an OME-Zarr 0.4 image (it has no 'multiscales' metadata).")
    multiscale = attrs["multiscales"][0]
    axes = [axis["name"] for axis in multiscale["axes"]]
    if axes != ["z", "y", "x"]:
        raise ClemregError(f"{url}: expected (z, y, x) axes, got {axes}.")
    units = {axis.get("unit") for axis in multiscale["axes"]}
    if not units <= _MICROMETRE_UNITS:
        raise ClemregError(f"{url}: expected pixel sizes in micrometres, got units {sorted(map(str, units))}.")
    datasets = {dataset["path"]: dataset for dataset in multiscale["datasets"]}
    if level not in datasets:
        raise ClemregError(f"{url} has no resolution level {level!r}.")
    transforms = {t["type"]: t for t in datasets[level]["coordinateTransformations"]}
    z_size, y_size, x_size = transforms["scale"]["scale"]
    return PhysicalPixelSizes(Z=z_size, Y=y_size, X=x_size)


def _open_ome_zarr_level(url, level, z):
    """Open one level of an NGFF 0.4 (z, y, x) image, cropped to `z`, as a
    lazy dask array in native byte order. Returns it with its pixel sizes
    in µm. Reads no pixels."""
    is_remote = url.startswith(("http://", "https://"))
    storage_options = {"client_kwargs": {"timeout": _HTTP_TIMEOUT}} if is_remote else None
    try:
        with warnings.catch_warnings():
            # The archive's copy stores "filters": [] (see ROADMAP).
            warnings.filterwarnings("ignore", message="Found an empty list of filters")
            group = zarr.open_group(url, mode="r", storage_options=storage_options)
            pixel_sizes = _pixel_sizes(url, group.attrs, level)
            array = group[level]
        # zarr reads a missing chunk as zeros, which would give black layers,
        # so check one exists. `strict` raises server errors instead of
        # reporting them as missing.
        first_chunk = array.metadata.encode_chunk_key((z.start // array.chunks[0], 0, 0))
        fs, path = fsspec.core.url_to_fs(url, **(storage_options or {}))
        has_data = fs.exists(f"{path}/{level}/{first_chunk}", strict=True)
    except (OSError, aiohttp.ClientError, zarr.errors.GroupNotFoundError) as err:
        raise ClemregError(
            f"Couldn't open the OME-Zarr sample at {url}: {err}. Check your internet connection, "
            "or use the Zenodo copy of the sample (Sample Benchmark Data)."
        ) from err
    if not has_data:
        raise ClemregError(f"{url} has metadata but no image data ({level}/{first_chunk} is missing).")

    data = da.from_zarr(array)[z]
    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder("="))
    return data, pixel_sizes


def make_ome_zarr_sample_data(urls=None):
    """The benchmark sample streamed from the BioImage Archive's OME-Zarr
    copy instead of downloaded from Zenodo.

    Same layers as `make_sample_data`, as lazy dask arrays with pixel sizes
    from the files. napari fetches the chunks it displays (and keeps them in
    its cache); pipeline steps fetch what they read every time.

    `urls` replaces `OME_ZARR_URLS`, e.g. with local copies.
    """
    urls = OME_ZARR_URLS if urls is None else urls

    def layer(key, level, z, **kwargs):
        data, pixel_sizes = _open_ome_zarr_level(urls[key], level, z)
        # No 'path' key, unlike the Zenodo EM layer: there's no local file.
        metadata = {
            "physical_pixel_sizes": pixel_sizes,
            "ome_zarr": {"url": urls[key], "level": level, "z_slice": [z.start, z.stop]},
        }
        return data, {**kwargs, "metadata": metadata}

    em = layer("em", _EM_LEVEL, _EM_Z, name="EM")
    fm = [layer(key, _FM_LEVEL, _FM_Z, name=name, colormap=colormap, blending="additive")
          for key, name, colormap in _FM_CHANNELS]
    return [em, *fm]


def make_em_mask_sample_data():
    """A precomputed EM segmentation mask (mitochondria instances) matching
    the EM volume in ``make_sample_data``, for exercising the pipeline from
    Point Cloud Sampling onward without a GPU or the bundled MitoNet/empanada
    widget (see issue #12). Bundled directly in the package (~1MB) rather
    than fetched, since it's small and was already sitting unused in
    notebooks/data/em_mask.tif.
    """
    path = importlib.resources.files('napari_clemreg.clemreg.data') / 'em_mask.tif'
    mask = tifffile.imread(path)
    return [(mask, {'name': 'EM_mask'}, 'labels')]
