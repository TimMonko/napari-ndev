import os
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import apoc
import dask.array as da
import numpy as np
import pyclesperanto_prototype as cle
import xarray as xr
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    Label,
    LineEdit,
    PushButton,
    RadioButtons,
    Select,
    SpinBox,
)

if TYPE_CHECKING:
    import napari

PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, da.Array]
MetaArrayLike = Union[ArrayLike, xr.DataArray]
ImageLike = Union[
    PathLike, ArrayLike, MetaArrayLike, List[MetaArrayLike], List[PathLike]
]


class SegmentImg(Container):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
    ):
        super().__init__()
        self
        self._viewer = viewer
        self._image_directory = FileEdit(label="Image Directory", mode="d")
        self._label_directory = FileEdit(label="Label Directory", mode="d")
        self._output_directory = FileEdit(label="Output Directory", mode="d")
        self._classifier_file = FileEdit(
            label="Classifier File (.cl)", mode="r"
        )
        self._classifier_channels = Label(value="Trained on [#] Channels")

        self._classifier_type = RadioButtons(
            label="Classifier Type",
            value="PixelClassifier",
            choices=["ObjectSegmenter", "PixelClassifier"],
        )
        self._max_depth = SpinBox(
            label="Num. of Forests", value=3, max=20, step=1
        )
        self._num_trees = SpinBox(label="Num. of Trees", value=200, step=50)
        self._positive_class_id = SpinBox(
            label="Positive Label ID", value=2, step=1
        )

        self._image_channels = Select(label="Image Channels", choices=[])
        self._channel_order_label = Label(value="Selected Channel Order: []")

        self._PDFS = Enum("PDFS", apoc.PredefinedFeatureSet._member_names_)
        self._predefined_features = ComboBox(
            label="Features", choices=self._PDFS
        )
        self._custom_features = LineEdit(
            label="Custom Feature String", value=None
        )

        self._continue_training = CheckBox(
            label="Continue Training?", value=True
        )
        self._batch_train_button = PushButton(
            label="Train Classifier on Image-Label Pairs"
        )
        self._batch_predict_button = PushButton(
            label="Predict Labels with Classifier"
        )
        self._progress_label = Label(value="Progress: ")

        self.extend(
            [
                self._image_directory,
                self._label_directory,
                self._output_directory,
                self._classifier_file,
                self._classifier_channels,
                self._classifier_type,
                self._max_depth,
                self._num_trees,
                self._positive_class_id,
                self._image_channels,
                self._channel_order_label,
                self._predefined_features,
                self._custom_features,
                self._continue_training,
                self._batch_train_button,
                self._batch_predict_button,
                self._progress_label,
            ]
        )

        self._image_directory.changed.connect(self._update_metadata)
        self._image_channels.changed.connect(self._update_channel_order)
        self._classifier_file.changed.connect(self._update_classifier_metadata)
        self._batch_train_button.clicked.connect(self.batch_train)
        self._batch_predict_button.clicked.connect(self.batch_predict)

    def _update_metadata(self):
        files = os.listdir(self._image_directory.value)
        img = AICSImage(self._image_directory.value / files[0])
        self._image_channels.choices = img.channel_names

    def _update_channel_order(self):
        self._channel_order_label.value = "Selected Channel Order: " + str(
            self._image_channels.value
        )

    def _update_classifier_metadata(self):
        with open(self._classifier_file.value) as file:
            content = file.read()

        # Ignore rest of function if file contents are empty
        if not content.strip():
            print("empty file")
            self._classifier_channels.value = "New Classifier"
            return

        pattern_classifier = re.search(
            pattern=r"classifier_class_name\s*=\s*([^\n]+)", string=content
        )
        self._classifier_type.value = (
            pattern_classifier.group(1) if pattern_classifier else None
        )

        pattern_classes = re.search(
            pattern=r"num_classes\s*=\s*(\d+)", string=content
        )
        self._classifier_channels.value = (
            "Trained on " + pattern_classes.group(1) + " Channels"
            if pattern_classes
            else None
        )

        pattern_max_depth = re.search(r"max_depth\s*=\s*(\d+)", content)
        self._max_depth.value = (
            pattern_max_depth.group(1) if pattern_max_depth else None
        )

        pattern_num_trees = re.search(r"num_trees\s*=\s*(\d+)", content)
        self._num_trees.value = (
            pattern_num_trees.group(1) if pattern_num_trees else None
        )

        pattern_class_id = re.search(
            r"positive_class_identifier\s*=\s*(\d+)", content
        )
        self._positive_class_id.value = (
            pattern_class_id.group(1) if pattern_class_id else 2
        )

    def batch_train(self):
        image_files = os.listdir(self._image_directory.value)
        label_files = os.listdir(self._label_directory.value)

        if not self._continue_training:
            apoc.erase_classifier(self._classifier_file.value)

        if self._classifier_type.value == "PixelClassifier":
            custom_classifier = apoc.PixelClassifier(
                opencl_filename=self._classifier_file.value,
                max_depth=self._max_depth.value,
                num_ensembles=self._num_trees.value,
            )

        if self._classifier_type.value == "ObjectSegmenter":
            custom_classifier = apoc.ObjectSegmenter(
                opencl_filename=self._classifier_file.value,
                positive_class_identifier=self._positive_class_id.value,
                max_depth=self._max_depth.value,
                num_ensembles=self._num_trees.value,
            )

        if self._predefined_features.value == 1:
            feature_set = self._custom_features.value
        else:
            feature_set = apoc.PredefinedFeatureSet[
                self._predefined_features.value.name
            ].value

        img = AICSImage(self._image_directory.value / image_files[0])
        channel_index_list = []
        for channel in self._image_channels.value:
            channel_index = img.channel_names.index(channel)
            channel_index_list.append(channel_index)

        self._progress_label.value = (
            "Starting Training on " + str(len(image_files)) + " Images"
        )

        for idx, file in enumerate(image_files):

            img = AICSImage(self._image_directory.value / image_files[idx])
            channel_img = img.get_image_data("TCZYX", C=channel_index_list)

            lbl = AICSImage(self._label_directory.value / label_files[idx])
            label = lbl.get_image_data("TCZYX", C=0)

            custom_classifier.train(
                features=feature_set,
                image=np.squeeze(channel_img),
                ground_truth=np.squeeze(label),
                continue_training=True,
            )
            self._progress_label.value = (
                "Image: "
                + str(idx)
                + " of "
                + str(len(image_files))
                + " : "
                + file
            )

        self._progress_label.value = (
            "Training Completed on " + str(len(image_files)) + " Images"
        )

    def batch_predict(self):
        image_files = os.listdir(self._image_directory.value)

        if self._classifier_type.value == "PixelClassifier":
            custom_classifier = apoc.PixelClassifier(
                opencl_filename=self._classifier_file.value,
            )

        if self._classifier_type.value == "ObjectSegmenter":
            custom_classifier = apoc.ObjectSegmenter(
                opencl_filename=self._classifier_file.value,
            )

        img = AICSImage(self._image_directory.value / image_files[0])
        channel_index_list = []
        for channel in self._image_channels.value:
            channel_index = img.channel_names.index(channel)
            channel_index_list.append(channel_index)

        img_dims = "".join(
            [
                d
                for d in img.dims.order
                if d != "C" and img.dims._dims_shape[d] > 1
            ]
        )

        self._progress_label.value = (
            "Starting Prediction on " + str(len(image_files)) + " Images"
        )

        for idx, file in enumerate(image_files):
            img = AICSImage(self._image_directory.value / image_files[idx])
            channel_img = img.get_image_data("TCZYX", C=channel_index_list)

            result = custom_classifier.predict(image=np.squeeze(channel_img))

            OmeTiffWriter.save(
                data=cle.pull(result).astype(np.int32),
                uri=self._output_directory.value / image_files[idx],
                dim_order=img_dims,
                channel_names=["Labels"],
                physical_pixel_sizes=img.physical_pixel_sizes,
            )

            self._progress_label.value = (
                "Image: "
                + str(idx)
                + " of "
                + str(len(image_files))
                + " : "
                + file
            )

        self._progress_label.value = (
            "Prediction Completed on " + str(len(image_files)) + " Images"
        )
