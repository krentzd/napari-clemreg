"""The registration as an affine transform for MoBIE (Fiji).

MoBIE's "Registration - Enter Transformation" takes 12 numbers: the top
three rows of a 4x4 affine matrix, row by row, in (x, y, z) order and
physical units. It maps the moving image onto the fixed image.
"""
import numpy as np


def mobie_affine(matrix, working_pixel_size_um, moving_origin_um=(0, 0, 0), fixed_origin_um=(0, 0, 0)):
    """Convert a moving -> fixed registration matrix to MoBIE's 12 numbers.

    `matrix` is the 4x4 (z, y, x) matrix from point cloud registration, in
    voxels of the working grid, whose voxels are `working_pixel_size_um`
    on every axis. The origins are where each image's first voxel lies in
    µm (z, y, x), e.g. because it was cropped from a larger file.
    """
    matrix = np.asarray(matrix, dtype=float)
    linear = matrix[:3, :3]
    translation = (matrix[:3, 3] * working_pixel_size_um
                   + np.asarray(fixed_origin_um) - linear @ np.asarray(moving_origin_um))
    # Reverse both axes, from (z, y, x) to (x, y, z).
    rows = np.hstack([linear[::-1, ::-1], translation[::-1, None]])
    return rows.ravel().tolist()


def format_mobie_affine(parameters):
    """The 12 numbers in the form MoBIE's text field shows and reads."""
    return "[" + ", ".join(f"{p:.10g}" for p in parameters) + "]"
