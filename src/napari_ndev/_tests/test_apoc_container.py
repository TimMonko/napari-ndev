import os
import tempfile

import numpy as np

from napari_ndev import ApocContainer
import pytest

image = np.asarray(
    [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 1, 1],
        [2, 2, 1, 1],
    ]
)

annotation = np.asarray(
    [[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]
)

# test the training on each filter!!!

# def test_update_metadata(make_napari_viewer):
#     """
#     Test the _update_channel_order method of the SegmentImg class.
#     """
#     wdg = SegmentImg(make_napari_viewer())
#     with tempfile.TemporaryDirectory() as tmpdir:
#         # Create dummy image file
#         image_path = os.path.join(tmpdir, "dummy_image.tif")
#         with open(image_path, "w") as f:
#             f.write("dummy image content")
#         wdg._image_directory.value = tmpdir
#         assert wdg._image_channels.choices == ["Channel_0"]


def test_update_channel_order(make_napari_viewer):
    """
    Test the _update_channel_order method of the SegmentImg class.
    """
    viewer = make_napari_viewer()
    # wdg = SegmentImg("dummy_viewer")
    wdg = ApocContainer(viewer)
    wdg._image_channels.choices = ["C0", "C1", "C2", "C3"]
    wdg._image_channels.value = ["C1", "C3"]
    wdg._update_channel_order()
    assert (
        wdg._channel_order_label.value
        == "Selected Channel Order: ['C1', 'C3']"
    )


@pytest.fixture
def dummy_classifier_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_file_path = os.path.join(tmpdir, "dummy_classifier.cl")
        with open(classifier_file_path, "w") as f:
            f.write(
                "OpenCL RandomForestClassifier\n"
                "classifier_class_name = PixelClassifier\n"
                "num_ground_truth_dimensions = 2\n"
                "num_classes = 3\n"
                "max_depth = 5\n"
                "num_trees = 100\n"
                # "positive_class_identifier = 2\n"
            )
        yield classifier_file_path


def test_update_classifier_metadata(make_napari_viewer, dummy_classifier_file):
    viewer = make_napari_viewer()
    wdg = ApocContainer(viewer)

    num_widgets = len(viewer.window._dock_widgets)
    # This automatically calls wdg._update_classifier_metadata() because of
    # wdg ._classifier_file.changed.connect
    wdg._classifier_file.value = dummy_classifier_file

    assert len(viewer.window._dock_widgets) == 1 + num_widgets
    assert wdg._classifier_type.value == "PixelClassifier"
    assert wdg._classifier_channels.value == "Trained on 3 Channels"
    assert wdg._max_depth.value == 5
    assert wdg._num_trees.value == 100
    assert wdg._positive_class_id.value == 2

