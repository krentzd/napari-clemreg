import tifffile
import numpy as np 
from skimage import metrics, measure
import matplotlib.pyplot as plt
import gc
import math
import napari 
from scipy import ndimage 
import json
from magicgui import magicgui
import os
from skimage.io import imread, imsave
from skimage.measure import label
import torch

from napari.layers import Labels, Points, Image
from napari.types import PointsData

from napari_clemreg.clemreg.widget_components import run_point_cloud_sampling, run_point_cloud_registration_and_warping
from napari_clemreg.clemreg.log_segmentation import log_segmentation
from napari_clemreg.clemreg.sample_data import make_sample_data
from napari_clemreg.clemreg.point_cloud_sampling import point_cloud_sampling
from napari_clemreg.clemreg.point_cloud_registration import point_cloud_registration
from napari_clemreg.clemreg.warp_image_volume import warp_image_volume
from napari.experimental import link_layers

#has_gpu = torch.cuda.is_available()

has_gpu = False # For testing, set to always false

if has_gpu:
    print("Found CUDA, so we'll run the full EM segmentation")
else:
    print("No CUDA found, so we'll use a pre-computed EM segmentation to save time")

em, tgn, lyso, mito, nuc = make_sample_data()  

em_raw = Image(em[0], metadata=em[1], name='EM')
fm_tgn = Image(tgn[0], metadata=tgn[1], name='TGN')
fm_lyso = Image(lyso[0], metadata=lyso[1], name='Lysosomes')
fm_mito = Image(mito[0], metadata=mito[1], name='Mitochondria')
fm_nuc = Image(nuc[0], metadata=nuc[1], name='Nucleus')

_ = link_layers([fm_tgn, fm_lyso, fm_mito, fm_nuc])

if has_gpu:
    pass
else:
    em_mask_file = "data/em_mask.tif"
    em_mask_metadata = {'name': 'EM mask',
                        'metadata': {'ImageDescription': '\nunit=micron\nspacing=0.02\n',
                        'XResolution': 50,
                        'YResolution': 50}}
    em_mask_data = np.uint16(imread(em_mask_file))
    em_mask = Labels(em_mask_data, metadata=em_mask_metadata)

fm_mito_mask_data = log_segmentation(Image(fm_mito.data,metadata=fm_mito.metadata))
fm_mito_mask_metadata = {'name': 'mito_segmentation',
                         'metadata': {'XResolution': fm_mito.metadata['metadata']['XResolution'],
                         'YResolution': fm_mito.metadata['metadata']['YResolution']}}

fm_mito_mask = Labels(fm_mito_mask_data, metadata=fm_mito_mask_metadata)

em_points = point_cloud_sampling(em_mask, voxel_size=15, every_k_points=100//3, sigma=1.0)
fm_points = point_cloud_sampling(fm_mito_mask, voxel_size=15, every_k_points=100//3, sigma=1.0)

moving_points_data, fixed_points_data, tf_moving_points_data, tf_data = point_cloud_registration(moving=fm_points, fixed=em_points)

moving_points_kwargs = dict(
    name='Moving_point_cloud',
    face_color='red',
    edge_color='black',
    size=5,
    metadata={'pxlsz': fm_mito_mask_metadata['metadata']['XResolution']}
)

fixed_points_kwargs = dict(
    name='Fixed_point_cloud',
    face_color='blue',
    edge_color='black',
    size=5,
    metadata={'pxlsz': em_raw.metadata['metadata']['XResolution'], 'output_shape': em_mask.data.shape}
)

moving_points = Points(moving_points_data, **moving_points_kwargs)
fixed_points = Points(fixed_points_data, **fixed_points_kwargs)
transformed_points = Points(moving_points_data, **moving_points_kwargs) # Duplicate of the original data...
transformed_points.affine.affine_matrix = tf_data['affine']             # ...which we apply the transform to

warped_images = warp_image_volume(moving_image=fm_mito,
                                  output_shape=em_raw.data.shape,
                                  transform_type='Rigid CPD',
                                  moving_points=moving_points,
                                  transformed_points=transformed_points)

save_path_root = '/Users/jonesma/Downloads'   # CHANGE THIS TO A LOCATION OF YOUR CHOICE
for warped_image in warped_images:
    save_path = os.path.join(save_path_root, warped_image[1]['name']+'.tif')
    print(f'Saving {save_path}')
    imsave(save_path, warped_image[0], check_contrast=False)