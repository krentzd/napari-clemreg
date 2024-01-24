from __future__ import annotations

import numpy as np
from skimage.io import imread

def make_sample_data():
    """Generates an image"""
    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    print('Loading EM...')
    em = imread('https://zenodo.org/record/7936982/files/em_20nm_z_40_145.tif', plugin='tifffile')
    print('Loading FM...')
    fm = imread('https://zenodo.org/record/7936982/files/EM04468_2_63x_pos8T_LM_raw.tif', plugin='tifffile')

    fm = np.moveaxis(fm, -1, 0)

    fm_metadata = {'ImageDescription': 'ImageJ=1.53t\nimages=112\nchannels=4\nslices=28\nhyperstack=true\nmode=grayscale\nunit=micron\nspacing=0.13\nloop=false\nmin=0.0\nmax=65535.0\n',
                   'XResolution': 28.349506,
                   'YResolution': 28.349506}
    em_metadata = {'ImageDescription': '\nunit=micron\nspacing=0.02\n',
                   'XResolution': 50,
                   'YResolution': 50}

    return [(em, {'name': 'EM', 'metadata': em_metadata}),
            (fm[0, 7:], {'blending': 'additive', 'colormap': 'green', 'name': 'FM_TGN46', 'metadata': fm_metadata}),
            (fm[1, 7:], {'blending': 'additive', 'colormap': 'magenta', 'name': 'FM_Lysotracker', 'metadata': fm_metadata}),
            (fm[2, 7:], {'blending': 'additive', 'colormap': 'cyan', 'name': 'FM_Mitotracker', 'metadata': fm_metadata}),
            (fm[3, 7:], {'blending': 'additive', 'colormap': 'blue', 'name': 'FM_Hoechst', 'metadata': fm_metadata})
            ]
