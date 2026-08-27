# clemreg

**Core registration library for [CLEM-Reg](https://github.com/krentzd/napari-clemreg).**

> ⚠️ **Placeholder release.** This distribution currently reserves the `clemreg`
> name on PyPI. It contains no functionality yet. The CLEM-Reg algorithms live
> in the `napari-clemreg` plugin today and are being extracted into this
> standalone, napari-free core package.

## What this will be

`clemreg` is the pure-Python core of CLEM-Reg — correlative light and electron
microscopy (CLEM) image-volume registration based on common segmented structures
and point-cloud registration. It will operate on plain NumPy arrays with no
napari, Qt or GUI dependencies, so it can be used headlessly, scripted, and
tested in isolation:

- fluorescence (FM) and electron microscopy (EM) segmentation
- point-cloud sampling from segmentations
- point-cloud registration (Rigid / Affine / BCPD)
- image-volume warping
- export of results as a [MoBIE](https://mobie.github.io/) project

The napari widget will remain in a separate `napari-clemreg` package that depends
on this core.

## Install

Nothing to use here yet. For the working plugin:

```bash
pip install napari-clemreg
```

## Citation

If you use CLEM-Reg in your research, please cite the CLEM-Reg publication (see
the [main repository](https://github.com/krentzd/napari-clemreg)).

## License

MIT — see [LICENSE](https://github.com/krentzd/napari-clemreg/blob/main/LICENSE).
