#!/usr/bin/env python
# -*- coding: utf-8 -*-
from napari_plugin_engine import napari_hook_implementation
import numpy as np
import tifffile
import os

@napari_hook_implementation
def napari_get_reader(path):
    if isinstance(path, str) and path.endswith((".tif", ".tiff")):
      return tiff_reader

# Adapted from: https://www.thepythoncode.com/article/extracting-image-metadata-in-python
def read_metadata(path):
    from PIL import Image
    from PIL.ExifTags import TAGS
    metadata = {}
    with Image.open(path) as img:
        exifdata = img.getexif()
        for tag_id in exifdata:
            # get the tag name, instead of human unreadable tag id
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            # decode bytes
            if isinstance(data, bytes):
                data = data.decode()
            print(f"{tag:25}: {data}")
            metadata[f"{tag}"] = data
    return metadata

def tiff_reader(path: str):
    tiff_image = tifffile.imread(path)
    img_metadata = read_metadata(path)
    layer_kwargs = {"metadata": img_metadata}
    return [(tiff_image, layer_kwargs, "image")]
