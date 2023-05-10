# napari-clemrerg
### An automated point cloud based registration algorithm for correlative light and volume electron microscopy

[![License](https://img.shields.io/pypi/l/napari-clemreg.svg?color=green)](https://github.com/krentzd/napari-clemreg/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-clemreg.svg?color=green)](https://pypi.org/project/napari-clemreg)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-clemreg.svg?color=green)](https://python.org)

[//]: # ([![codecov]&#40;https://codecov.io/gh/krentzd/napari-clemreg/branch/master/graph/badge.svg&#41;]&#40;https://codecov.io/gh/krentzd/napari-clemreg&#41;)
[//]: # ([![tests]&#40;https://github.com/krentzd/napari-clemreg/workflows/tests/badge.svg&#41;]&#40;https://github.com/krentzd/napari-clemreg/actions&#41;)

## Installation

To install `napari-clemreg` it is recommended to create a fresh [conda] environment with Python 3.9:

```
conda create -n clemreg_env python=3.9
```
Next, install `napari` with the following command via [pip]: 

```
pip install "napari[all]"
```

Finally, `napari-clemreg` can be installed with:
```
pip install napari-clemreg
```
When installing `napari-clemreg` on a Windows machine, the following error might appear:
```
error Microsoft Visual C++ 14.0 is required
```
Ensure that [Visual Studios C++ 14.00](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16) is installed
## Usage
CLEM-reg is the combination of 5 main steps, MitoNet segmentation, LoG segmentation,
point cloud sampling, point cloud registration and lastly image warping. These 5 steps 
can be run all at once using the run registration widget shown below with the tick next to it.
Alternatively, they can be run individually with the numbered widgets.

![clemreg_widget_options.png](docs%2Fimages%2Fclemreg_widget_options.png)

### Run Registration



![registration_labels.png](docs%2Fimages%2FCLEMreg-fig.png)

1. **Moving Image** - Here you select your light microscopy (LM) data which will
be warped to align with the fixed electron microscopy (EM) image.
2. **Fixed Image** - Here you select your EM data which will
act as the reference point for the LM to be aligned to.
3. **Registration Algorithm** - Here you can decide which type of registration algorith
will be used for the registration of inputted LM and EM. In terms of speed of each algorithm
the following is the generally true, Rigid CPD > Affine CPD > BCPD.
4. **MitoNet Segmentation Parameters** - Here are the advanced options for the segmentation
of the mitochondria in the EM data.
   1. Prediction Across Three Axis - By selecting this option MitoNet will run segmentation
across all three axis of the EM volume and then these three predictions will be aggregate.
5. **LoG Segmentation Parameters** - Here are the advanced options for the segmentation of 
the mitochondria in the LM data.
   1. Sigma - Sigma value for the Laplacian of Gaussian filter.
   2. Threshold - Threshold value for the segmenting the LM data.
6. **Point Cloud Sampling** - Here are the advanced options for the point cloud sampling of the 
segmentations of the LM and EM data.
   1. Sampling Frequency - Frequency of point sampling from the fixed and moving segmentation. The greater the value the more points in the point cloud.
   2. Sigma - Sigma value for the canny edge filter.
7. **Point Cloud Registration** - Here are the advanced options for the registration of the point clouds
of both the LM and EM data.
   1. Voxel Size - The size voxel size of each point. Smaller the size the less memory consumption.
   2. Subsampling - Downsampling of the point clouds to reduce memory consumption. Greater the number the fewer points in the point cloud.
   3. Maximum Iterations - The number of round of point cloud registration. If too small it won't converge on an opitmal registration.
8. **Image Warping** - Here are the advanced options for the image warping of the moving LM images.
   1. Interpolation Order - The order of the spline interpolation.
   2. Aproximate Grid - Controls the "resolution" of the grid onto which you're warping. A higher value reduces the step size between coordinates.
   3. Sub-division Factor - Controls the size of the chunk when applying the warping.
9. **Save Parameters** - Here you can select the option to save the advanced options you've selected
to a JSON file which can be kept for reproducibility as well as running the registration again.
10. **Visualise Intermediate Results** - Here you can select to view the outputs of each step as they
are completed.

### Split Registration
As well as being able to run all the steps of CLEM-reg in one widget (the `Run registration` widget),
you are also able to do all these steps independently using the `Split Registration` functionality. 

There are four separate widgets that encapsulate the 5 steps of CLEM-reg each of which have
their own unique input and output:
1. `MitoNet Segmentation` 
   - **Input**: EM Image
   - **Output**: EM Segmentation
2. `LoG Segmentation`
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

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-clemreg" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

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

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.
