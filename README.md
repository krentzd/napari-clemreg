# napari-clemreg

<!-- [![License](https://img.shields.io/pypi/l/napari-clemreg.svg?color=green)](https://github.com/krentzd/napari-clemreg/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-clemreg.svg?color=green)](https://pypi.org/project/napari-clemreg)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-clemreg.svg?color=green)](https://python.org)
[![tests](https://github.com/krentzd/napari-clemreg/workflows/tests/badge.svg)](https://github.com/krentzd/napari-clemreg/actions)
[![codecov](https://codecov.io/gh/krentzd/napari-clemreg/branch/master/graph/badge.svg)](https://codecov.io/gh/krentzd/napari-clemreg) -->

An automated point-set based registration algorithm for correlative light and electron microscopy (CLEM) 
----------------------------------
## Installation

To install `napari-clemreg` it is recommended to create a fresh [conda] enviornment with Python 3.8:

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
There are two means of running CLEM-reg, one is using the Run Registration which includes one

alternatively you can use the Split Registration
which is the combination of four separate widgets, splitting each step of the CLEM-reg workflow.

![Widget Options](https://github.com/krentzd/napari-clemreg/blob/main/docs/images/clemreg_widget_options.png)
### Run Registration
![Run Registration](https://github.com/krentzd/napari-clemreg/blob/main/docs/images/Full_registration_labels.png)

1. **Moving Image** - Here you select your light microscopy (LM) data which will
be warped to align with the fixed electron microscopy (EM) image.
2. **Fixed Image** - Here you select your EM data which will
act as the reference point for the LM to be aligned to.
3. **Mask ROI** - Here you can select a shapes layer where one can define what regions of
the moving LM data will be segmented and subsequently sampled.
4. **Registration Algorithm** - Here you can decide which type of registration algorith
will be used for the registration of inputted LM and EM. In terms of speed of each algorithm
the following is the generally true, Rigid CPD > Affine CPD > BCPD.
   1. BCPD - (EXPLAIN BCPD)
   2. Rigid CPD - (EXPLAIN RIGID CPD)
   3. Affine CPD - (EXPLAIN AFFINE CPD)
5. **Parameters** - Here one can decide where the parameters of the algorithm should come
from, either from a JSON file of from the napari interface.
   1. Parameters from JSON - If you select this, you'll be prompted to select a JSON file from a file
dialogue.
   2. Parameters custom - If you select this, you'll be shown the advanced options of 
CLEM-reg.
6. **MitoNet Segmentation Parameters** - Here are the advanced options for the segmentation
of the mitochondria in the fixed EM data.
   1. Prediction Across Three Axis - By selecting this option MitoNet will run segmentation
across all three axis of the EM volume and then these three predictions will be aggregate. (MAY TAKE A WHILE)
7. **LoG Segmentation Parameters**  - Here are the advanced options for the segmentation of 
the mitochondria in the fixed LM data.
   1. Sigma - (EXPLAIN SIGMA)
   2. Threshold - (EXPLAIN THRESHOLD)
8. **Point Cloud Sampling** - Here are the advanced options for the point cloud sampling of the 
segmentations of the LM and EM data.
   1. Sampling Frequency - Frequency of cloud sampling
   2. Sigma - (EXPLAIN SIGMA)
9. **Point Cloud Registration** - Here are the advanced options for the registration of the point clouds
of both the LM and EM data.
   1. Interpolation Order - (EXPLAIN INTERPOLATION ORDER)
   2. Subsampling - (EXPLAIN SUBSAMPLING)
   3. Maximum Iterations - (EXPLAIN MAXIMUM ITERATIONS)
10. **Image Warping** - Here are the advanced options for the image warping of the moving LM images.
    1. Interpolation Order - (EXPLAIN INTERPOLATION ORDER)
    2. Aproximate Grid - (EXPLAIN APROXIMATE GRID)
    3. Sub-division Factor - (EXPLAIN SUB-DIVISION FACTOR)
11. **Save Parameters** - Here you can select the option to save the advanced options you've selected
to a JSON file which can be kept for reproducibility as well as running the registration again.
12. **Visualise Intermediate Results** - Here you can select to view the outputs of each step as they
are completed.

### Registering Multiple LM Layers
![afskml](https://github.com/krentzd/napari-clemreg/blob/main/docs/images/multi-channel-lm-merged.png) ![](https://github.com/krentzd/napari-clemreg/blob/main/docs/images/multi-channel-lm-merged.png)

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
