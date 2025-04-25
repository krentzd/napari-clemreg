

<h1 align="center">
napari-clemreg

</h1>

### An automated point cloud based registration algorithm for correlative light and volume electron microscopy
<p align="center">
    <a href="https://www.biorxiv.org/content/10.1101/2023.05.11.540445v3"><img alt="Paper" src="https://img.shields.io/badge/paper-bioRxiv-%23b62b39"></a>
    <a href="https://pypi.org/project/napari-clemreg"><img alt="PyPI" src="https://img.shields.io/pypi/v/napari-clemreg.svg?color=green"></a><a href="https://pypistats.org/packages/napari-clemreg"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/napari-clemreg"></a>
    <a href="https://github.com/krentzd/napari-clemreg/"><img alt="github" src="https://img.shields.io/github/stars/krentzd/napari-clemreg?style=social"></a>
    <a href="https://github.com/krentzd/napari-clemreg/"><img alt="github" src="https://img.shields.io/github/forks/krentzd/napari-clemreg?style=social"></a>
</p>

## Overview
CLEM-Reg fully automates the registration step for vCLEM datasets by first segmenting mitochondria in both image modalities, sampling point
clouds from these segmentations and registering them. Once registered, the point cloud alignment is used to warp the fluorescence microscopy onto the 
volume electron microscopy. 
![width=200](docs%2Fimages%2Fclemreg_algorithm.png)


## Installation
### Local Installation

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

[//]: # (When installing `napari-clemreg` on a Windows machine, the following error might appear:)

[//]: # (```)

[//]: # (error Microsoft Visual C++ 14.0 is required)

[//]: # (```)

[//]: # (Ensure that [Visual Studios C++ 14.00]&#40;https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16&#41; is installed)

### Docker Container
If you would like to run `napari-clemreg` in a docker container instead of installing it as above, please follow the instructions in our [Docker guide](docker_guide.md)

## Usage
CLEM-reg is the combination of 5 main steps, electron microscopy segmentation, fluorescence microscopy segmentation,
point cloud sampling, point cloud registration and image warping. These 5 steps 
can be run all at once using the run registration widget.
Alternatively, they can be run individually with the numbered widgets.

![clemreg_widget_options.png](docs%2Fimages%2Fnapari_dropdown.png)

### Run Registration

![registration_labels.png](docs%2Fimages%2Fclemreg_params.png)

1. **Fluorescence Microscopy Image (FM)** - Here you select the layer with the fluorescence microscopy image.
2. **FM Pixel Size (xy)** - Here you can set the xy pixel size of your FM image and its corresponding unit.
3. **FM Pixel Size (z)** - Here you can set the z pixel size of your FM image and its corresponding unit.
4. **Mask ROI** - Here you can select a mask layer which will be used to crop the resulting segmentation mask in the FM.
5. **Electron Microscopy (EM)** - Here you can select the layer with the electron microscopy image.
6. **EM Pixel Size (xy)** - Here you can set the xy pixel size of your EM image and its corresponding unit.
7. **EM Pixel Size (z)** - Here you can set the z pixel size of your EM image and its corresponding unit.
8. **Registration Algorithm** - Here you can decide which type of registration algorith will be used for the registration of inputted LM and EM. In terms of speed of each algorithm the following is the generally true, Rigid CPD > Affine CPD > BCPD.
9. **Parameters from JSON** - Here you can select a JSON file containing the parameters for the registration.
10. **Parameters custom** - If you select this, you will be able to edit the default parameters.
11. **MitoNet Segmentation Parameters** - Here are the advanced options for the segmentation of the mitochondria in the EM data.
    1. **Prediction Across Three Axis** - By selecting this option MitoNet will run segmentation across all three axis of the EM volume and then these three predictions will be aggregate.
12. **LoG Segmentation Parameters** - Here are the advanced options for the segmentation of the mitochondria in the LM data.
    1. **Sigma** - Sigma value for the Laplacian of Gaussian filter.
    2. **Threshold** - Threshold value for the segmenting the LM data.
    3. **Apply size filter to segmentation** - If you select this, you can then select a lower and upper volume threshold to filter out spurious segmentation.
13. **Point Cloud Sampling** - Here are the advanced options for the point cloud sampling of the segmentations of the LM and EM data.
    1. **Sampling Frequency** - Frequency of point sampling from the fixed and moving segmentation. The greater the value the more points in the point cloud.
    2. **Voxel Size** - The size voxel size of each point. Smaller the size the less memory consumption.
    3. **Sigma** - Sigma value for the canny edge filter.
14. **Point Cloud Registration** - Here are the advanced options for the registration of the point clouds of both the LM and EM data.
    1. Maximum Iterations - The number of round of point cloud registration. If too small it won't converge on an opitmal registration.
15. **Image Warping** - Here are the advanced options for the image warping of the moving images.
    1. Interpolation Order - The order of the spline interpolation.
    2. Aproximate Grid - Controls the "resolution" of the grid onto which you're warping. A higher value reduces the step size between coordinates.
    3. Sub-division Factor - Controls the size of the chunk when applying the warping.
16. **Save Parameters** - Here you can select the option to save the advanced options you've selected to a JSON file which can be kept for reproducibility as well as running the registration again.
17. **Visualise Intermediate Results** - Here you can select to view the outputs of each step as they are completed.
18. **Registration direction** - Here you can select which of the modalities will be registered to the other. Either EM to FM or FM to EM.

[![Watch the video](https://github.com/krentzd/napari-clemreg/blob/krentzd-patch-1/clem_reg_tutorial_thumbnail.png)](https://youtu.be/ud3zTLgl8Ks)

### Split Registration
As well as being able to run all the steps of CLEM-reg in one widget (the `Run registration` widget),
you are also able to do all these steps independently using the `Split Registration` functionality. 

There are four separate widgets that encapsulate the 5 steps of CLEM-reg each of which have
their own unique input and output:
1. `Electron Micrscopy (EM) Segmentation` 
   - **Input**: EM Image
   - **Output**: EM Segmentation
2. `Fluorescence Microscopy (FM) Segmentation`
   - **Input**: LM Image
   - **Output**: LM Segmentation
3. `Point Cloud Sampling`
   - **Input**: LM Segmentation & EM Segmentation
   - **Output**: LM Point Cloud & LM Point Cloud
4. `Point Cloud Registration & Image Warping`
   - **Input**: EM Image, LM Image, LM Point Cloud & EM Point Cloud
   - **Output**: Registered LM Image, Registered LM Point Cloud

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

3. When you finally go to run CLEM-reg ensure that for the `Moving Image`
you select the LM layer that contains mitochondria.
## Datasets
Below are the links to the datasets that were used as part of this study.

**EMPIAR-10819**
- [EM] - https://www.ebi.ac.uk/empiar/EMPIAR-10819/
- [FM] - https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST707

**EMPIAR-11537**
- [EM] - https://www.ebi.ac.uk/empiar/EMPIAR-11537/
- [FM] - https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BSST1075

Here is a sample dataset which is the binned version of EMPIAR-10819: https://zenodo.org/records/7936982.

## How to cite
```bibtex
@article {Krentzel2023.05.11.540445,
	author = {Krentzel, Daniel and Elphick, Matou{\v s} and Domart, Marie-Charlotte and Peddie, Christopher J. and Laine, Romain F. and Shand, Cameron and Henriques, Ricardo and Collinson, Lucy M. and Jones, Martin L.},
	title = {CLEM-Reg: An automated point cloud based registration algorithm for correlative light and volume electron microscopy},
	elocation-id = {2023.05.11.540445},
	year = {2024},
	doi = {10.1101/2023.05.11.540445},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Correlative light and volume electron microscopy (vCLEM) is a powerful imaging technique that enables the visualisation of fluorescently labelled proteins within their ultrastructural context on a subcellular level. Currently, expert microscopists align vCLEM acquisitions using time-consuming and subjective manual methods. This paper presents CLEM-Reg, an algorithm that automates the 3D alignment of vCLEM datasets by leveraging probabilistic point cloud registration techniques. These point clouds are derived from segmentations of common structures in each modality, created by state-of-the-art open-source methods, with the option to leverage alternative tools from other plugins or platforms. CLEM-Reg drastically reduces the time required to register vCLEM datasets to a few minutes and achieves correlation of fluorescent signal to sub-micron target structures in EM on three newly acquired vCLEM benchmark datasets (fluorescence microscopy combined with FIB-SEM or SBF-SEM). CLEM-Reg was then used to automatically obtain vCLEM overlays to unambiguously identify TGN46-positive transport carriers involved in the trafficking of proteins between the trans-Golgi network and plasma membrane. The datasets are available in the EMPIAR and BioStudies public image archives for reuse in testing and developing multimodal registration algorithms by the wider community. A napari plugin integrating the algorithm is also provided to aid end-user adoption.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2024/12/26/2023.05.11.540445},
	eprint = {https://www.biorxiv.org/content/early/2024/12/26/2023.05.11.540445.full.pdf},
	journal = {bioRxiv}
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
