import numpy as np
import pytest

from napari_ndev._rescale_by import RescaleBy

image_4d = np.random.random((1, 1, 10, 10))


@pytest.mark.parametrize(
    "scale_x, scale_y, scale_z, scale_in_z, expected_scale",
    [
        (1.5, 2.5, 3.5, False, (1, 1, 2.5, 1.5)),
        (1.5, 2.5, 3.5, True, (1, 3.5, 2.5, 1.5)),
    ],
)
def test_rescale_by(
    make_napari_viewer, scale_x, scale_y, scale_z, scale_in_z, expected_scale
):
    viewer = make_napari_viewer()
    viewer.add_image(image_4d)
    container = RescaleBy(viewer)

    container._layer_to_scale.value = viewer.layers["image_4d"]
    container._scale_x.value = scale_x
    container._scale_y.value = scale_y
    container._scale_z.value = scale_z
    container._scale_in_z.value = scale_in_z

    container.rescale_by()

    assert np.array_equal(
        container._layer_to_scale.value.scale, expected_scale
    )


def test_inherit_from(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(image_4d, scale=(3.5, 2.5, 1.5))
    container = RescaleBy(viewer)

    container._inherit_from_layer.value = viewer.layers["image_4d"]

    # Check if the scale values have been inherited correctly
    assert container._scale_x.value == 1.5
    assert container._scale_y.value == 2.5
    assert container._scale_z.value == 3.5
