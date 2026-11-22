

<h1 align="center">
napari-clemreg

</h1>

### An automated point cloud based registration algorithm for correlative light and volume electron microscopy
<p align="center">
    <a href="https://www.nature.com/articles/s41592-025-02794-0"><img alt="Paper" src="https://img.shields.io/badge/paper-Nat._Methods-orange"></a>
    <a href="https://github.com/martlj/napari-clemreg/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/martlj/napari-clemreg/actions/workflows/test.yml/badge.svg"></a>
    <a href="https://pypi.org/project/napari-clemreg"><img alt="PyPI" src="https://img.shields.io/pypi/v/napari-clemreg.svg?color=green"></a><a href="https://pypistats.org/packages/napari-clemreg"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/napari-clemreg"></a>
    <a href="https://github.com/krentzd/napari-clemreg/"><img alt="github" src="https://img.shields.io/github/stars/krentzd/napari-clemreg?style=social"></a>
    <a href="https://github.com/krentzd/napari-clemreg/"><img alt="github" src="https://img.shields.io/github/forks/krentzd/napari-clemreg?style=social"></a>
</p>

## Overview
CLEM-Reg fully automates the registration step for vCLEM datasets by first segmenting mitochondria in both image modalities, sampling point
clouds from these segmentations and registering them. Once registered, the point cloud alignment is used to warp the fluorescence microscopy onto the 
volume electron microscopy. 
![width=200](docs%2Fimages%2Fclemreg_algorithm.png)

### EM segmentation (MitoNet): current status

CLEM-Reg segments mitochondria in the EM volume with the [MitoNet](https://github.com/volume-em/empanada) model. On this fork's `modernisation` branch it can run MitoNet in two ways, chosen with the **EM Segmentation Backend** dropdown:

| Backend | Status |
|---|---|
| `MitoNet (Segment-Flow)` | **Default.** Runs MitoNet through Crick's [AI-on-Demand (AIoD)](https://franciscrickinstitute.github.io/aiod_docs/) Segment-Flow pipeline, in its own isolated environment. Works on Python 3.11+. Needs Nextflow, Conda and a JDK ([setup below](#em-segmentation-with-mitonet)). |
| `MitoNet (empanada-dl)` | Runs MitoNet in-process via `empanada-dl`, as the published release does. `empanada-dl` is unmaintained and currently can't be installed on Python 3.11 ([#5](https://github.com/martlj/napari-clemreg/issues/5)), so on this branch the option is there but won't run unless you install that dependency yourself. |

The release on PyPI (`0.2.1`) predates this work. It only has the in-process `empanada-dl` backend, and its install instructions below use Python 3.9.


## Installation
### Local Installation (PyPI release)

These steps install the published PyPI release (`0.2.1`), which runs MitoNet in-process via `empanada-dl`. For the Segment-Flow default and the rest of the modernisation work, use the [development installation](#development-installation-this-fork) instead.

To install `napari-clemreg` it is recommended to create a fresh [conda] environment with Python 3.9:

```
conda create -n clemreg_env python=3.9
```
Now we must activate the conda environment.

``` 
conda activate clemreg_env
```

Next, install `napari` with the following command via [pip]: 

```
pip install "napari[all]"
```

Then, `napari-clemreg` can be installed with:
```
pip install napari-clemreg
```

Finally, to run napari run the following.
```
napari
```

When installing `napari-clemreg` on a Windows machine, the following error might appear:
```
ImportError: DLL load failed while importing pybind: A dynamic link library (DLL) initialization routine failed.
```
This is `open3d` (a dependency), which needs both the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) and real GPU-backed OpenGL 4.1+ support — it won't import in a virtual machine or remote desktop without GPU passthrough (confirmed against [isl-org/Open3D#4468](https://github.com/isl-org/Open3D/issues/4468) and [#5328](https://github.com/isl-org/Open3D/issues/5328); this is also why this project's CI only tests Linux and macOS, not Windows).

[![Watch the video](docs%2Fimages%2Fclemreg_installation_thumbnail.png)](https://youtu.be/ZN68q9OU59s)

### Development Installation (this fork)

This fork (`martlj/napari-clemreg`) is being actively modernised; see [docs/napari-clemreg-modernisation-plan.md](docs/napari-clemreg-modernisation-plan.md) for the full plan. The work is on the `modernisation` branch, which is the fork's default branch, so a plain clone gets it:
```
git clone https://github.com/martlj/napari-clemreg.git
cd napari-clemreg
```
If you cloned before 2026-09-23, your checkout is probably on `main`, which doesn't have this work. Switch branches rather than cloning again, because cloning into an existing checkout creates a confusing nested copy:
```
git fetch origin
git checkout modernisation
```

This branch requires **Python 3.11+**. The base install no longer includes `empanada-dl`, so it resolves on Python 3.11. Install it with the `segment-flow` extra to get the default EM segmentation backend (it also needs Nextflow, Conda and a JDK; see [EM segmentation with MitoNet](#em-segmentation-with-mitonet)).

Using [pixi](https://pixi.sh/) (recommended on Apple-silicon Macs and on Linux with glibc 2.31 or newer, the platforms `pixi.toml` supports: one command sets up everything, including Nextflow and a JDK for the Segment-Flow backend):

```
pixi run napari
```

The first run creates the environment in `.pixi/` from `pixi.toml` and `pixi.lock`, with napari-clemreg installed from your checkout in editable mode, so code changes appear the next time you start napari. Segment-Flow still needs `conda` on `PATH` (your own Miniconda or Miniforge) to build the model's environment. Run the tests with `pixi run -e test test`.

Or using [`uv`](https://docs.astral.sh/uv/) (fast, and installs the Python version for you if needed). `uv pip install` and `uv run` both auto-detect the local `.venv`, so **don't `source .venv/bin/activate`**. If a conda environment is already active (e.g. `base` auto-activates in many setups), a plain `source activate` can silently lose to conda's own `PATH` handling and run the wrong `napari` entirely:

```
uv venv --python 3.11
uv pip install -e ".[segment-flow]" pyqt6
uv run napari
```

Or with plain `pip` in a conda environment:

```
conda create -n clemreg_dev python=3.11
conda activate clemreg_dev
pip install -e ".[segment-flow]" pyqt6
napari
```

`pyqt6` is the Qt binding napari needs to open a window. Leave out `[segment-flow]` if you only want to run the steps after EM segmentation, for example from the [precomputed EM mask sample](#em-segmentation-with-mitonet). The `[empanada]` extra (in-process MitoNet) is still defined, but it won't resolve on Python 3.11 until [#5](https://github.com/martlj/napari-clemreg/issues/5) is fixed.

### Docker Container
If you would like to run `napari-clemreg` in a docker container instead of installing it as above, please follow the instructions in our [Docker guide](docker_guide.md)

[![Watch the video](docs%2Fimages%2Fclemreg_docker_installation_guide.png)](https://youtu.be/2GRB99UlP6g)

## Usage
CLEM-reg is the combination of 5 main steps, electron microscopy segmentation, fluorescence microscopy segmentation,
point cloud sampling, point cloud registration and image warping. These 5 steps 
can be run all at once using the run registration widget.
Alternatively, they can be run individually with the numbered widgets.

![clemreg_widget_options.png](docs%2Fimages%2Fnapari_dropdown.png)

### Run Registration

![registration_labels.png](docs%2Fimages%2Fclemreg_params.png)

> The screenshot above predates the current layout (tracked in [#51](https://github.com/martlj/napari-clemreg/issues/51)). The list below describes the widget as it is now, from top to bottom.

**Inputs**
- **Fluorescence Microscopy Image (FM)**: the layer with the fluorescence (moving) image. It must be a 3D greyscale volume.
- **FM Pixel size (xy)** and **FM Pixel size (z)**: filled in from the image's metadata when you pick a layer. Check them, and correct them if the metadata is missing or wrong.
- **Mask ROI**: an optional Shapes layer with a single shape, used to crop the FM segmentation. When one is selected, **Minimum/Maximum z value for masking** also appear, to limit the mask to a range of slices.
- **Electron Microscopy Image (EM)**: the layer with the electron microscopy (fixed) image. It must be a 3D greyscale volume.
- **EM Pixel size (xy)** and **EM Pixel size (z)**: as for FM.
- **Registration Algorithm**: `Rigid CPD` (default), `Affine CPD` or `BCPD`. Rigid CPD is fastest and BCPD (non-rigid) is slowest.
- **Parameters from JSON**: load parameters from a saved JSON file. Currently broken: the file is not read ([#47](https://github.com/martlj/napari-clemreg/issues/47)).

**Parameter sections.** Each section is collapsed by default. Each has its own **Run this step** button, which runs just that step on whatever layers are in the section's inputs. When a step finishes, its output is filled into the next section's input and highlighted in amber, so you can step through the pipeline or start partway through from layers you already have.

1. **EM Segmentation Parameters**
   - **EM Segmentation Backend**: `MitoNet (Segment-Flow)` (default) or `MitoNet (empanada-dl)`. Both run the same model; see [EM segmentation with MitoNet](#em-segmentation-with-mitonet).
2. **LoG Segmentation Parameters** (FM segmentation)
   - **Sigma**: sigma of the Laplacian of Gaussian filter.
   - **Threshold**: threshold applied to the filtered FM image.
   - **Apply size filter to segmentation**: remove objects whose volume falls outside a percentile range, set by **Lower filter threshold** and **Upper filter threshold** (percentiles, default 5–95).
3. **Point Cloud Sampling**
   - **Fluorescence Microscopy (FM) Segmentation** and **Electron Microscopy (EM) Segmentation**: the inputs for this step. Filled in automatically by the steps above, or pick existing Labels layers.
   - **Sampling Frequency**: roughly the percentage of segmentation-edge points to keep (the default, 3, keeps about 1 in 33). **Higher values give more points**: more detail, but slower and more memory.
   - **Sigma**: sigma of the Canny edge filter used to find segmentation edges.
4. **Point Cloud Registration**
   - **Fluorescence Microscopy (FM) Point Cloud** and **Electron Microscopy (EM) Point Cloud**: the inputs for this step.
   - **Voxel Size**: size of the voxel grid the point clouds are downsampled onto. **Larger values give fewer points**, which is faster and uses less memory. It is applied during point cloud sampling, although it sits in this section ([#49](https://github.com/martlj/napari-clemreg/issues/49)).
   - **Maximum Iterations**: maximum number of registration iterations. If this is too low, the registration may not converge.
5. **Image Warping**
   - **Interpolation Order**: order of the spline interpolation.
   - **Approximate Grid**: controls the resolution of the grid used to approximate the warp. A higher value reduces the step size between coordinates.
   - **Sub-division Factor**: controls the size of the chunks the warp is applied in.
   - **Output Pixel Size**: `Native LM resolution` (default) warps the FM image onto a grid close to its own pixel size and places it over the EM image using the layer's scale. `EM pixel grid (legacy)` resamples it onto the EM image's much finer pixel grid, as earlier versions did.

**Output options**
- **Save parameters**: save the current parameters to a JSON file, for reproducibility. Currently broken: ticking it makes **Register** fail ([#47](https://github.com/martlj/napari-clemreg/issues/47)).
- **Visualise Intermediate Results**: add each step's output (segmentations and point clouds) to the viewer as it finishes. On by default.
- **Registration direction**: `FM → EM` (default) warps the FM image onto the EM image; `EM → FM` does the reverse.

**Register** runs the whole pipeline from the FM and EM images. The terminal prints the total run time when it finishes.

[![Watch the video](docs%2Fimages%2Fclem_reg_tutorial_thumbnail.png)](https://youtu.be/ud3zTLgl8Ks)

### Split Registration
As well as running every step in the `Run registration` widget, you can run the steps independently with four separate widgets, each with its own inputs and outputs:
1. `1) Electron Microscopy (EM) Segmentation`
   - **Input**: EM image
   - **Output**: EM segmentation
2. `2) Fluorescence Microscopy (FM) Segmentation`
   - **Input**: FM image
   - **Output**: FM segmentation
3. `3) Point Cloud Sampling`
   - **Input**: FM segmentation and EM segmentation
   - **Output**: FM point cloud and EM point cloud
4. `4) Point Cloud Registration & Image Warping`
   - **Input**: EM image, FM image, FM point cloud and EM point cloud
   - **Output**: registered FM image(s) and registered FM point cloud

The combined widget's **Run this step** buttons do the same job without switching widgets.

[![Watch the video](docs%2Fimages%2Fclemreg_split_registration_thumbnail.png)](https://youtu.be/cypDti0UUwY)

### EM segmentation with MitoNet

Both the `1) Electron Microscopy (EM) Segmentation` widget and the `Run registration` widget default to running MitoNet through Crick's [AI-on-Demand (AIoD)](https://franciscrickinstitute.github.io/aiod_docs/) Segment-Flow pipeline. This is fully integrated: pick the backend and run. There's no need to install AIoD's separate `aiod_napari` plugin or to pass its output to `Point Cloud Sampling` by hand. The bundled alternative, `MitoNet (empanada-dl)`, is still in the same **EM Segmentation Backend** dropdown, but `empanada-dl` is unmaintained and currently can't be installed on Python 3.11 ([#5](https://github.com/martlj/napari-clemreg/issues/5)). Segment-Flow avoids that problem because the model runs in its own isolated environment.

**Prerequisites for the Segment-Flow backend** (the rest of napari-clemreg works without them):
- [Nextflow](https://www.nextflow.io/) and [Conda](https://docs.conda.io/) on `PATH`. The pixi environment includes Nextflow, but not Conda.
- A JDK, ideally version 17–21 (the pixi environment includes JDK 21). Very recent JDKs (tested: 26) fail with `Unsupported class file major version`. On macOS, `brew install openjdk@21` is enough on its own: it's detected automatically without setting `JAVA_HOME`. On other platforms, set `JAVA_HOME` to a compatible JDK if your system default is newer.
- The `segment-flow` extra, which adds the small, pure-Python packages used to talk to AIoD's model registry (no heavy ML dependencies). The [development installation](#development-installation-this-fork) above already includes it; to add it to an existing install:
  ```
  uv pip install "napari-clemreg[segment-flow]"
  ```
  (or `pip install "napari-clemreg[segment-flow]"`)

The first run on a given machine downloads and builds an isolated Conda environment for the model (cached under `~/.nextflow/aiod` and reused on later runs), so it can take several minutes. Progress is printed to the terminal, not the napari window. The model always runs inside Segment-Flow's own environment, never in napari-clemreg's Python process.

**Want to try the pipeline without a GPU or setting up AIoD?** `File → Open Sample → napari-clemreg → EM Mask (precomputed, no GPU needed)` loads a precomputed EM segmentation that matches the EM volume in the main sample data. You can then start from **Point Cloud Sampling** (in either widget) without running EM segmentation at all.

### Registering Multiple LM Channels
One can register multiple LM channels at once by doing the following.

1. Start by splitting the LM channels into the separate layers by right-clicking on
the layer and then selecting `Split Stack`.
![merged-channel-split-options.png](docs%2Fimages%2Fmerged-channel-split-options.png)
This will result in each of the channels having their own individual layer. 

2. Once this is done we must link all the LM layers together, this is done 
by selecting all the layers which will highlight them in blue, once again right-clicking
on the layer and then selecting `Link Layers.`
![split-channels-link-layers.png](docs%2Fimages%2Fsplit-channels-link-layers.png)

3. When you run CLEM-Reg, select the LM layer that contains mitochondria as the **Fluorescence Microscopy Image (FM)**.

## Datasets
Below are the links to the datasets that were used as part of this study.

**EMPIAR-10819**
- [EM] - https://www.ebi.ac.uk/empiar/EMPIAR-10819/
- [FM] - https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST707

**EMPIAR-11537**
- [EM] - https://www.ebi.ac.uk/empiar/EMPIAR-11537/
- [FM] - https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST1075

Here is a sample dataset which is the binned version of EMPIAR-10819: https://zenodo.org/records/7936982.

The same data is on the BioImage Archive as OME-Zarr, with the FIB-SEM volume at full resolution (5 nm) and pixel sizes in the files: [S-BIAD4355](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4355).

### Sample data in napari
`File → Open Sample → napari-clemreg` offers the benchmark data from either source. Both give the same five layers (`EM` at 20 nm, and four FM channels), so the widgets work the same way with either.

- **Sample Benchmark Data** downloads two TIFFs from Zenodo (about 600 MB) the first time, then loads them from a local cache. Nothing needs the network after that.
- **Sample Benchmark Data (OME-Zarr, streamed from the BioImage Archive)** downloads nothing in advance and stores nothing on disk. Instead:
  - Opening it takes 10–15 seconds, during which napari doesn't respond: it downloads about 250 MB to show the first slices. The archive stores the EM in blocks of 56 slices, so napari also pauses for a few seconds the first time you scroll into each new block. It keeps what it has shown in memory, so going back is instant.
  - Each pipeline step downloads the images it reads, every time it runs, and every new napari session starts again. You need a network connection throughout.
  - The pixel sizes come from the files. The FM channels are identical to the Zenodo ones; the EM is the same volume, but the Zenodo file's contrast was stretched, so segmentation and registration results differ slightly.

## How to cite
```bibtex
@article{krentzel2025clem,
  title={CLEM-Reg: an automated point cloud-based registration algorithm for volume correlative light and electron microscopy},
  author={Krentzel, Daniel and Elphick, Matou{\v{s}} and Domart, Marie-Charlotte and Peddie, Christopher J and Laine, Romain F and Shand, Cameron and Henriques, Ricardo and Collinson, Lucy M and Jones, Martin L},
  journal={Nature Methods},
  pages={1--12},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```
## License

Distributed under the terms of the [MIT] licence,
"napari-clemreg" is free and open source software

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/krentzd/napari-clemreg/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[conda]: https://docs.conda.io/en/latest/

[//]: # (This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.)
