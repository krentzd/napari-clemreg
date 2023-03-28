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
conda create -n clemreg_env python=3.8
```
Next, install `napari` with the following command via [pip]: 

```
pip install "napari[all]"
```
If using Mac M1 please do the following instead:
```
conda install imagecodecs pyqt
pip install napari
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
There are two means of running the CLEM-reg workflow, one is using the Run Registration
also known as the Full Run Registration, alternatively you can use the Split Run Registration
which is the combination of four separate widgets, splitting each step of the CLEM-reg workflow.

![Widget Options](https://github.com/krentzd/napari-clemreg/blob/main/docs/images/clemreg_widget_options.png)
### Full Run Registration
![Run Registration](https://github.com/krentzd/napari-clemreg/blob/main/docs/images/Full_registration_labels.png)

1. **Moving Image** -
2. **Fixed Image**
3. **Mask ROI**
4. **Registration Algorithm**
5. **Parameters**
   1. Parameters from JSON
   2. Parameters custom
6. **MitoNet Segmentation Parameters**
   1. Prediction Across Three Axis
7. **LoG Segmentation Parameters**
   1. Sigma
   2. Threshold
8. **Point Cloud Sampling**
   1. Sampling Frequency
   2. Sigma
9. **Point Cloud Registration**
   1. Interpolation Order
   2. Subsampling
   3. Maximum Iterations
10. **Image Warping**
    1. Interpolation Order
    2. Aproximate Grid
    3. Sub-division Factor
11. **Save Parameters**
12. **Visualise Intermediate Results**

### Split Run Registration


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
