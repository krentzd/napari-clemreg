# Changelog

All notable changes to napari-clemreg. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Changes since the version published in Nature Methods ([`v1.0.0-nature-methods`](https://github.com/krentzd/napari-clemreg/tree/v1.0.0-nature-methods), PyPI 0.2.1).

### Added
- **EM segmentation through AI OnDemand's Segment-Flow**, now the default **EM Segmentation Backend**. It runs MitoNet in its own environment through Nextflow, so it works on Python 3.11 and newer. It's installed with the `segment-flow` extra. The `AIOD_NXF_REPO` and `AIOD_NXF_PROFILE` environment variables select a local Segment-Flow checkout and a Nextflow profile.
- **Run this step** buttons in every section of **Run registration**, to run one step at a time or start partway through from existing layers. Each step's output fills the next step's input, highlighted in amber.
- **OME-Zarr sample data, streamed from the BioImage Archive** (S-BIAD4355), alongside the Zenodo sample. It stores nothing on disk.
- **EM Mask** sample data: a precomputed EM segmentation of the sample, for starting without EM segmentation.
- Sample data and MitoNet weights are cached instead of downloaded on every run.
- The TIFF reader reads pixel sizes with `bioio`, so they're found in TIFFs not written by ImageJ.
- **Register** prints its total run time in the terminal.
- A Docker image with napari and everything the default EM segmentation needs, and a pixi environment for running from source.
- Rigid and Affine CPD registrations print the transform for [MoBIE](https://imagej.net/plugins/mobie) (`Transform > Registration - Enter Transformation`), in micrometres and including known offsets such as the sample data's crops. It's also stored in the warped layers' metadata (`mobie_affine`).

### Changed
- **Requires Python 3.11 or newer** (previously 3.7+), and supports current napari releases.
- The warped FM image keeps the FM pixel size by default and is placed over the EM image with the layer's scale. **Output Pixel Size → EM pixel grid (legacy)** gives the previous output.
- **Image Warping** settings sit inside **Point Cloud Registration**, whose button also warps the image. **Approximate Grid** and **Sub-division Factor** only appear for BCPD.
- `empanada-dl` and `torch` moved to the optional `empanada` extra. That extra can't currently be installed on Python 3.11 or newer.
- The non-rigid (BCPD) warp uses SciPy's thin plate spline. It's faster and uses less memory (on the sample data about 3 minutes and 9 GB, down from over 10 minutes and 14 GB), and it's more exact, so BCPD results differ slightly from earlier versions'.
- New dependencies: `zarr >= 3.1`, `dask`, `fsspec[http]` and `aiohttp`, for the streamed sample. They can't be installed alongside plugins that require `zarr < 3`.

### Removed
- The four numbered widgets for running single steps (*1) EM Segmentation* to *4) Point Cloud Registration & Image Warping*). **Run registration**'s **Run this step** buttons replace them.
- The **Prediction Across Three Axis** option.

### Fixed
- **Apply size filter to segmentation** kept the wrong objects: it compared each object's label with the objects' sizes, and counted the background as an object. It now keeps the objects whose volume is within the chosen percentiles of all the objects' volumes.
- EM segmentation through Segment-Flow could return the mask of an earlier run, e.g. of the sample data after segmenting another image: every run's mask lands in the same cache folder, and the one that sorted last by name was read. Each run now names its own mask, reads that one, and deletes it afterwards. Masks left by earlier versions in `~/.nextflow/aiod/aiod_cache/empanada/` can be deleted.
- EM segmentation through Segment-Flow on the `local` profile ran several copies of the model at once, which can fill an ordinary GPU's memory. It now runs them one at a time, as Segment-Flow's own single-GPU profile does.
- **Save parameters** made **Register** fail, and **Parameters from JSON** never read its file. Register, and the Point Cloud Registration step's **Run this step**, now save the settings to the chosen file and say where, and choosing a parameter file fills in the widget's settings from it. Settings from older versions that no longer exist are ignored, and a file that can't be used leaves the settings unchanged.
- **Registration direction → EM → FM** crashed with `KeyError: 'output_shape'`. It now warps the EM onto the FM: with **Native LM resolution** onto the FM image's own grid, with **EM pixel grid** at the EM's pixel size over the FM's extent.
- With **Visualise Intermediate Results** on, **Register** showed Rigid and Affine CPD's `transformed_points` where the moving points were, instead of on the fixed points: it dropped the layer's affine, which holds the registration.
- The non-rigid (BCPD) warp's affine part was computed incorrectly.
- The BCPD warp crashed with **Approximate Grid** 1. It was also wrong, or crashed, where the volume or a chunk (with a high **Sub-division Factor**) was thinner than two grid steps.
- FM volumes were stretched by a few percent in z when resampled, which shifted warped objects high in the stack.
- **Register** hung silently when a segmentation step failed. It now reports the error.
- Clear errors when no mitochondria are found, or when the chosen EM backend isn't installed.
- Layer dropdowns stayed out of date after layers were added, removed or renamed.
- **Run registration** preselected the same layer as both the FM and the EM image.
- The sample data's FM channels displayed as stripes.
- Crashes when warping layers shown in the viewer, and with current napari's Points layers.
- Pixel sizes from ImageJ metadata are parsed as numbers instead of being evaluated as code.
- Panels are wide enough for their forms, and don't stay enlarged after hiding settings.
