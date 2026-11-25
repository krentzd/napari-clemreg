"""Converting the registration matrix to MoBIE's affine format."""
import numpy as np
import pytest
from napari.layers import Image

from napari_clemreg.clemreg.mobie import format_mobie_affine, mobie_affine
from napari_clemreg.clemreg.widget_components import _layer_origin_um


def _as_matrix(parameters):
    return np.vstack([np.reshape(parameters, (3, 4)), [0, 0, 0, 1]])


def test_identity_stays_the_identity():
    assert mobie_affine(np.eye(4), working_pixel_size_um=0.02) == pytest.approx(
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])


def test_maps_the_moving_image_onto_the_fixed_image_in_micrometres():
    """MoBIE's matrix takes a moving-image point in µm (x, y, z), each image
    placed at its origin, to the matching fixed-image point."""
    rng = np.random.default_rng(0)
    matrix = np.eye(4)
    matrix[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0] * 1.1
    matrix[:3, 3] = (3.0, -2.0, 5.0)
    working, moving_origin, fixed_origin = 0.02, np.array([0.8, 0.1, -0.3]), np.array([0.0075, 0.005, 0.0025])

    bdv = _as_matrix(mobie_affine(matrix, working, moving_origin, fixed_origin))

    for voxel in rng.uniform(0, 100, size=(5, 3)):  # (z, y, x) in the moving working grid
        moving_um = voxel * working + moving_origin
        fixed_um = (matrix @ [*voxel, 1])[:3] * working + fixed_origin
        np.testing.assert_allclose((bdv @ [*moving_um[::-1], 1])[:3], fixed_um[::-1], atol=1e-9)


def test_axes_are_in_x_y_z_order():
    matrix = np.eye(4)
    matrix[0, 0] = 2.0   # stretches z
    matrix[2, 3] = 10.0  # moves 10 voxels in x

    parameters = mobie_affine(matrix, working_pixel_size_um=0.5)

    assert parameters[10] == 2.0  # z row, z column
    assert parameters[3] == 5.0   # x row, translation


def test_format_is_what_mobie_shows_and_reads():
    assert format_mobie_affine([1, 0, 0, 0.25, 0, 1, 0, 0, 0, 0, 1, -1.5]) == \
        "[1, 0, 0, 0.25, 0, 1, 0, 0, 0, 0, 1, -1.5]"


@pytest.mark.parametrize("layer_kwargs", [
    {"translate": (2, 0, 0)},                                     # napari units are voxels
    {"scale": (0.12, 0.04, 0.04), "translate": (0.24, 0, 0)},     # napari units are µm
], ids=["voxels", "micrometres"])
def test_layer_origin_combines_recorded_offsets_and_napari_translate(layer_kwargs):
    layer = Image(np.zeros((4, 4, 4)), metadata={"origin_um": (0.0, 0.5, 0.0)}, **layer_kwargs)

    np.testing.assert_allclose(_layer_origin_um(layer, (120, 40)), (0.24, 0.5, 0.0))
