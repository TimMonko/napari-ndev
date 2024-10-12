import os
from pathlib import Path

import numpy as np
import pytest
from bioio import BioImage

from napari_ndev.widgets._utilities_container import UtilitiesContainer

image_2d = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 1, 1], [2, 2, 1, 1]])
shapes_2d = np.array([[0.25, 0.25], [0.25, 2.75], [2.75, 2.75], [2.75, 0.25]])
labels_2d = np.asarray(
    [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
)

image_4d = np.random.random((1, 1, 10, 10))
shapes_4d = [
    np.array([[0, 0, 1, 1], [0, 0, 1, 3], [0, 0, 5, 3], [0, 0, 5, 1]]),
    np.array([[0, 0, 5, 5], [0, 0, 5, 9], [0, 0, 9, 9], [0, 0, 9, 5]]),
]
labels_4d = np.array(
    [
        [
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
            ]
        ]
    ]
)


@pytest.fixture(
    params=[
        (image_2d, shapes_2d, labels_2d, 'YX'),
        (image_4d, shapes_4d, labels_4d, 'TZYX'),
    ]
)
def test_data(request: pytest.FixtureRequest):
    return request.param


@pytest.mark.parametrize('image_layer', [True, False])
def test_save_shapes_as_labels(
    make_napari_viewer, tmp_path: Path, test_data, image_layer: bool
):
    test_image, test_shape, test_labels, test_dims = test_data

    viewer = make_napari_viewer()
    viewer.add_image(test_image)
    viewer.add_shapes(test_shape)
    container = UtilitiesContainer(viewer)

    container._viewer.layers.selection.active = viewer.layers['test_shape']
    container._squeezed_dims = test_dims
    container._save_directory.value = tmp_path
    container._save_name.value = 'test.tiff'

    shapes_as_labels = container.save_shapes_as_labels()

    expected_save_loc = tmp_path / 'ShapesAsLabels' / 'test.tiff'
    assert expected_save_loc.exists()
    assert shapes_as_labels.shape == test_image.shape
    assert np.array_equal(shapes_as_labels, test_labels)
    assert BioImage(expected_save_loc).channel_names == ['Shapes']


def test_save_labels(make_napari_viewer, tmp_path: Path, test_data):
    _, _, test_labels, test_dims = test_data

    viewer = make_napari_viewer()
    viewer.add_labels(
        test_labels
    )  # <- should add a way to specify this is the selected layer in the viewer
    viewer.layers.selection.active = viewer.layers['test_labels']
    container = UtilitiesContainer(viewer)

    container._squeezed_dims = test_dims
    container._save_directory.value = tmp_path
    container._save_name.value = 'test.tiff'

    labels = container.save_labels()

    expected_save_loc = tmp_path / 'Labels' / 'test.tiff'
    assert expected_save_loc.exists()
    assert np.array_equal(labels, test_labels)
    assert BioImage(expected_save_loc).channel_names == ['Labels']


def test_save_ome_tiff(make_napari_viewer, test_data, tmp_path: Path):
    test_image, _, _, _ = test_data
    viewer = make_napari_viewer()
    viewer.add_image(test_image)
    container = UtilitiesContainer(viewer)

    container._concatenate_image_files.value = False
    container._concatenate_image_layers.value = True
    container._viewer.layers.selection.active = viewer.layers['test_image']
    container._channel_names.value = ['0']
    container._save_directory.value = tmp_path
    container._save_name.value = 'test.tiff'

    container.save_ome_tiff()

    expected_save_loc = tmp_path / 'Images' / 'test.tiff'
    assert expected_save_loc.exists()
    assert len(container._img_data.shape) == 5


@pytest.fixture
def test_rgb_image():
    path = os.path.join(
        'src', 'napari_ndev', '_tests', 'resources', 'RGB.tiff'
    )
    img = BioImage(path)
    return path, img


def test_update_metadata_from_file(make_napari_viewer, test_rgb_image):
    viewer = make_napari_viewer()
    container = UtilitiesContainer(viewer)

    path, _ = test_rgb_image
    container._files.value = path
    container.update_metadata_from_file()

    assert container._save_name.value == 'RGB.tiff'
    assert container._img.dims.order == 'TCZYXS'
    assert container._squeezed_dims == 'YX'
    assert container._channel_names.value == "['red', 'green', 'blue']"


def test_update_metadata_from_layer(make_napari_viewer, test_data):
    test_image, _, _, _ = test_data
    viewer = make_napari_viewer()
    viewer.add_image(test_image, scale=(2, 3))
    container = UtilitiesContainer(viewer)

    container._viewer.layers.selection.active = viewer.layers['test_image']
    container.update_metadata_from_layer()

    assert (
        'Tried to update metadata, but could only update scale'
        ' because layer not opened with napari-bioio'
    ) in container._results.value
    assert container._scale_tuple.value == (1, 2, 3)

def test_open_images(make_napari_viewer, test_rgb_image):
    viewer = make_napari_viewer()
    container = UtilitiesContainer(viewer)

    path, _ = test_rgb_image
    container._files.value = path
    container.open_images()

    assert container._img.dims.order == "TCZYXS"
    assert container._squeezed_dims == "YX"
    assert container._channel_names.value == "['red', 'green', 'blue']"
