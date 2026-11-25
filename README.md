

<h1 align="center">
napari-clemreg

</h1>

### An automated point cloud based registration algorithm for correlative light and volume electron microscopy
<p align="center">
    <a href="https://www.nature.com/articles/s41592-025-02794-0"><img alt="Paper" src="https://img.shields.io/badge/paper-Nat._Methods-orange"></a>
    <a href="https://pypi.org/project/napari-clemreg"><img alt="PyPI" src="https://img.shields.io/pypi/v/napari-clemreg.svg?color=green"></a><a href="https://pypistats.org/packages/napari-clemreg"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/napari-clemreg"></a>
    <a href="https://github.com/krentzd/napari-clemreg/"><img alt="github" src="https://img.shields.io/github/stars/krentzd/napari-clemreg?style=social"></a>
    <a href="https://github.com/krentzd/napari-clemreg/"><img alt="github" src="https://img.shields.io/github/forks/krentzd/napari-clemreg?style=social"></a>
</p>

## Overview
CLEM-Reg fully automates the registration step for vCLEM datasets by first segmenting mitochondria in both image modalities, sampling point
clouds from these segmentations and registering them. Once registered, the point cloud alignment is used to warp the fluorescence microscopy onto the 
volume electron microscopy. 
![width=200](docs%2Fimages%2Fclemreg_algorithm.png)

The code as published in Nature Methods is tagged [`v1.0.0-nature-methods`](https://github.com/krentzd/napari-clemreg/tree/v1.0.0-nature-methods).


## Installation
### Local Installation

This version of `napari-clemreg` isn't on PyPI yet, so install it from source. First, get the code:

```
git clone -b version-2 https://github.com/krentzd/napari-clemreg.git
cd napari-clemreg
```

#### With conda

Create a [conda] environment with Python 3.11, plus Nextflow and a JDK for EM segmentation:

```
conda create -n clemreg_env -c conda-forge -c bioconda python=3.11 nextflow "openjdk=21"
```
Now activate the environment:

```
conda activate clemreg_env
```

Then install `napari-clemreg` from the cloned code, together with the Qt library napari needs to open a window:

```
pip install -e ".[segment-flow]" pyqt6
```

Finally, to run napari run the following.
```
napari
```

#### With pixi

On Apple-silicon Macs and Linux, [pixi](https://pixi.sh/) sets up everything with one command. It needs a recent pixi: if you get an error about `pixi.toml`, run `pixi self-update` first.

```
pixi run napari
```

#### EM segmentation

EM segmentation runs MitoNet through [AI OnDemand](https://franciscrickinstitute.github.io/aiod_docs/)'s Segment-Flow pipeline. Both environments above include Nextflow and a JDK, but it also needs [Conda](https://docs.conda.io/) (e.g. Miniconda or Miniforge) on your system. The first run sets up the model's environment, which takes a few minutes.

If you can't install it locally, use the Docker container below.

[//]: # (When installing `napari-clemreg` on a Windows machine, the following error might appear:)

[//]: # (```)

[//]: # (error Microsoft Visual C++ 14.0 is required)

[//]: # (```)

[//]: # (Ensure that [Visual Studios C++ 14.00]&#40;https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16&#41; is installed)


### Docker Container
If you would like to run `napari-clemreg` in a docker container instead of installing it as above, build the image in the clone from above and run it:

```
docker build -t clemreg --target clemreg -f clemreg.dockerfile .
docker run --rm -e DISPLAY=$DISPLAY --net=host clemreg
```

Add `--gpus=all` to use an NVIDIA GPU. Without an X server (e.g. on Mac), build the `clemreg_xpra` target instead, run it with `docker run --rm -p 9876:9876 clemreg_xpra` and open http://localhost:9876 in a browser.

[![Watch the video](docs%2Fimages%2Fclemreg_docker_installation_guide.png)](https://youtu.be/2GRB99UlP6g)

## Usage
CLEM-reg is the combination of 5 main steps, electron microscopy segmentation, fluorescence microscopy segmentation,
point cloud sampling, point cloud registration and image warping. These 5 steps 
can be run all at once using the run registration widget.
Alternatively, they can be run one at a time with the **Run this step** button in each settings section.

To try it out, `File → Open Sample → napari-clemreg` offers a sample dataset, downloaded from Zenodo or streamed as OME-Zarr from the BioImage Archive, and a precomputed EM segmentation.

### Run Registration

1. **Fluorescence Microscopy Image (FM)** - Here you select the layer with the fluorescence microscopy image.
2. **FM Pixel Size (xy)** - Here you can set the xy pixel size of your FM image and its corresponding unit.
3. **FM Pixel Size (z)** - Here you can set the z pixel size of your FM image and its corresponding unit.
4. **Mask ROI** - Here you can select a mask layer which will be used to crop the resulting segmentation mask in the FM.
5. **Electron Microscopy (EM)** - Here you can select the layer with the electron microscopy image.
6. **EM Pixel Size (xy)** - Here you can set the xy pixel size of your EM image and its corresponding unit.
7. **EM Pixel Size (z)** - Here you can set the z pixel size of your EM image and its corresponding unit.
8. **Registration Algorithm** - Here you can decide which type of registration algorith will be used for the registration of inputted LM and EM. In terms of speed of each algorithm the following is the generally true, Rigid CPD > Affine CPD > BCPD.
9. **Parameters from JSON** - Here you can select a JSON file containing the parameters for the registration.
10. **EM Segmentation Parameters** - Here are the advanced options for the segmentation of the mitochondria in the EM data.
    1. **EM Segmentation Backend** - How MitoNet is run: through Segment-Flow (default), or in-process with `empanada-dl` (which can't currently be installed on Python 3.11 or newer).
11. **LoG Segmentation Parameters** - Here are the advanced options for the segmentation of the mitochondria in the LM data.
    1. **Sigma** - Sigma value for the Laplacian of Gaussian filter.
    2. **Threshold** - Threshold value for the segmenting the LM data.
    3. **Apply size filter to segmentation** - If you select this, you can then select a lower and upper volume threshold to filter out spurious segmentation.
12. **Point Cloud Sampling** - Here are the advanced options for the point cloud sampling of the segmentations of the LM and EM data.
    1. **Sampling Frequency** - Frequency of point sampling from the fixed and moving segmentation. The greater the value the more points in the point cloud.
    2. **Sigma** - Sigma value for the canny edge filter.
13. **Point Cloud Registration** - Here are the advanced options for the registration of the point clouds of both the LM and EM data.
    1. **Voxel Size** - The size voxel size of each point. Larger values give fewer points and less memory consumption.
    2. **Maximum Iterations** - The number of round of point cloud registration. If too small it won't converge on an opitmal registration.
    3. **Image Warping** - The advanced options for the image warping of the moving images, which this step's **Run this step** button also runs.
        1. **Interpolation Order** - The order of the spline interpolation.
        2. **Output Pixel Size** - `Native LM resolution` (default) keeps the FM pixel size; `EM pixel grid (legacy)` resamples onto the EM grid.
        3. **Aproximate Grid** (BCPD only) - Controls the "resolution" of the grid onto which you're warping. A higher value reduces the step size between coordinates.
        4. **Sub-division Factor** (BCPD only) - Controls the size of the chunk when applying the warping.
14. **Save Parameters** - Here you can select the option to save the advanced options you've selected to a JSON file which can be kept for reproducibility as well as running the registration again.
15. **Visualise Intermediate Results** - Here you can select to view the outputs of each step as they are completed.
16. **Registration direction** - Here you can select which of the modalities will be registered to the other. Either EM to FM or FM to EM.

[![Watch the video](docs%2Fimages%2Fclem_reg_tutorial_thumbnail.png)](https://youtu.be/ud3zTLgl8Ks)

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

3. When you finally go to run CLEM-reg ensure that for the **Fluorescence Microscopy Image (FM)**
you select the LM layer that contains mitochondria.

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
