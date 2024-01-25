import os
import tempfile

import numpy as np
import pytest

from napari_ndev import CustomApoc, SegmentImg

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


def test_update_channel_order():
    """
    Test the _update_channel_order method of the SegmentImg class.
    """
    wdg = SegmentImg("dummy_viewer")
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
    wdg = SegmentImg(viewer)

    assert len(viewer.window._dock_widgets) == 0
    assert wdg._classifier_type.value == "ObjectSegmenter"

    # This automatically calls wdg._update_classifier_metadata() because of
    # wdg ._classifier_file.changed.connect
    wdg._classifier_file.value = dummy_classifier_file

    assert len(viewer.window._dock_widgets) == 1
    assert wdg._classifier_type.value == "PixelClassifier"
    assert wdg._classifier_channels.value == "Trained on 3 Channels"
    assert wdg._max_depth.value == 5
    assert wdg._num_trees.value == 100
    assert wdg._positive_class_id.value == 2


# Test CustomAPOC
def test_generate_feature_string_button(make_napari_viewer):
    wdg = CustomApoc(make_napari_viewer())
    wdg._original.value = True
    wdg._generate_string_button.clicked.emit()
    assert wdg._feature_string.value != ""


def test_generate_features_none(make_napari_viewer):
    wdg = CustomApoc(make_napari_viewer())
    wdg.generate_feature_string()
    assert wdg._feature_string.value == ""


def test_generate_feature_string_original(make_napari_viewer):
    wdg = CustomApoc(make_napari_viewer())
    wdg._original.value = True
    wdg.generate_feature_string()
    assert "original" in wdg._feature_string.value


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        ("3", "gaussian_blur=3"),
        ("3,4,5", "gaussian_blur=3 gaussian_blur=4 gaussian_blur=5"),
        ("3, 4,   5", "gaussian_blur=3 gaussian_blur=4 gaussian_blur=5"),
    ],
)
def test_generate_feature_string_gaussian_blur(
    make_napari_viewer, input_value, expected_output
):
    wdg = CustomApoc(make_napari_viewer())
    wdg._gaussian_blur.value = input_value
    wdg.generate_feature_string()
    assert wdg._feature_string.value == expected_output
