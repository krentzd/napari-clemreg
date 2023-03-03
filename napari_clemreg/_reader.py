#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile

def napari_get_reader(path):
    if isinstance(path, str) and path.endswith((".tif", ".tiff")):
        return tiff_reader

# Adapted from: https://www.thepythoncode.com/article/extracting-image-metadata-in-python
def read_metadata(path):
    from PIL import Image
    from PIL.TiffTags import TAGS
    metadata = {}
    with Image.open(path) as img:
        exifdata = img.getexif()
        for tag_id in exifdata:
            # get the tag name, instead of human unreadable tag id
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            # decode bytes
            try:
                if isinstance(data, bytes):
                    data = data.decode()
                print(f"{tag:25}: {data}")
                metadata[f"{tag}"] = data
            except UnicodeDecodeError:
                print(f"Could not decode {data}")
    return metadata


def get_image_dims(metadata: dict):
    """ Parse image dimesnions from image metadata"""

    try:
        width = metadata['ImageWidth']  # X
        length = metadata['ImageLength']  # Y
    except KeyError:
        width = None
        length = None
        print('ImageWidth and ImageLength not recorded in metadata')

    try:
        # Parse ImageJ Metadata to get z pixelsize
        ij_metadata = metadata['ImageDescription'].split('\n')
        ij_metadata = [i for i in ij_metadata if i not in '=']
        ij_dict = dict((k, v) for k, v in (i.rsplit('=') for i in ij_metadata))

        slices = eval(ij_dict['slices'])
        channels = eval(ij_dict['channels'])
    except KeyError:
        slices = None
        channels = None
        print('ImageJ metdata not recorded in metadata')

    return (channels, slices, width, length)


def to_czxy(img, metadata):
    m_dims = get_image_dims(metadata)
    if not img.shape[1:] == m_dims:
        src_idx = [img.shape.index(d) for d in m_dims[:2]]
        img = np.moveaxis(img, src_idx, [0, 1])
        return img
    elif img.shape[1:] == m_dims:
        return img


def tiff_reader(path: str):
    tiff_image = tifffile.imread(path)
    img_metadata = read_metadata(path)
    if len(tiff_image.shape) > 3:
        tiff_image = to_czxy(np.squeeze(tiff_image), img_metadata)
    layer_kwargs = {"metadata": img_metadata}
    return [(tiff_image, layer_kwargs, "image")]
