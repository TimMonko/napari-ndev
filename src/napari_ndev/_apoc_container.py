import os
import re
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import apoc
import dask.array as da
import numpy as np
import pandas as pd
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
    ProgressBar,
    PushButton,
    RadioButtons,
    Select,
    SpinBox,
    Table,
    TextEdit,
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
        # viewer = napari_viewer
    ):
        super().__init__()
        self
        self._viewer = viewer
        self._image_directory = FileEdit(label="Image Directory", mode="d")
        self._label_directory = FileEdit(label="Label Directory", mode="d")
        self._output_directory = FileEdit(label="Output Directory", mode="d")
        self._classifier_file = FileEdit(
            label="Classifier File (.cl)",
            mode="r",
            tooltip="Create a .txt file and rename it to .cl ending.",
        )
        self._classifier_channels = Label(value="Trained on [#] Channels")

        self._classifier_type_mapping = {
            "PixelClassifier": apoc.PixelClassifier,
            "ObjectSegmenter": apoc.ObjectSegmenter,
        }

        self._classifier_type = RadioButtons(
            label="Classifier Type",
            value="ObjectSegmenter",
            choices=["ObjectSegmenter", "PixelClassifier"],
        )
        self._max_depth = SpinBox(
            label="Num. of Forests",
            value=2,
            max=20,
            step=1,
            tooltip="Increases training time for each forest",
        )
        self._num_trees = SpinBox(
            label="Num. of Trees",
            value=100,
            step=50,
            tooltip="Increases computational requirements.",
        )
        self._positive_class_id = SpinBox(
            label="Object Label ID",
            value=2,
            step=1,
            tooltip="Only used with ObjectSegmenter, otherwise ignored.",
        )

        self._image_channels = Select(
            label="Image Channels",
            choices=[],
            tooltip=(
                "Channel order should be same for training and prediction."
            ),
        )
        self._channel_order_label = Label(value="Select an Image Channel!")

        self._PDFS = Enum("PDFS", apoc.PredefinedFeatureSet._member_names_)
        self._predefined_features = ComboBox(
            label="Features",
            choices=self._PDFS,
            nullable=True,
            value=None,
            tooltip="All featuresets except 'custom' are premade",
        )
        self._custom_features = LineEdit(
            label="Custom Feature String",
            tooltip=(
                "A string in the form of " "'filter1=radius1 filter2=radius2'."
            ),
        )
        self._open_custom_feature_generator = PushButton(
            label="Open Custom Feature Generator Widget"
        )

        self._continue_training = CheckBox(
            label="Continue Training?",
            value=True,
            tooltip=(
                "Continue training only matters if classifier already exists."
            ),
        )
        self._batch_train_button = PushButton(
            label="Train Classifier on Image-Label Pairs"
        )
        self._batch_predict_button = PushButton(
            label="Predict Labels with Classifier"
        )
        self._progress_bar = ProgressBar(label="Progress:")

        self.extend(
            [
                self._classifier_file,
                self._classifier_channels,
                self._classifier_type,
                self._positive_class_id,
                self._max_depth,
                self._num_trees,
                self._predefined_features,
                self._custom_features,
                self._open_custom_feature_generator,
                self._image_directory,
                self._image_channels,
                self._channel_order_label,
                self._label_directory,
                self._continue_training,
                self._batch_train_button,
                self._output_directory,
                self._batch_predict_button,
                self._progress_bar,
            ]
        )

        self._image_directory.changed.connect(self._update_metadata)
        self._image_channels.changed.connect(self._update_channel_order)
        self._classifier_file.changed.connect(self._update_classifier_metadata)
        self._open_custom_feature_generator.clicked.connect(
            self._custom_apoc_widget
        )
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

    def _set_value_from_pattern(self, pattern, content):
        match = re.search(pattern, content)
        return match.group(1) if match else None

    def _process_classifier_metadata(self, content):
        self._classifier_type.value = self._set_value_from_pattern(
            r"classifier_class_name\s*=\s*([^\n]+)", content
        )
        self._classifier_channels.value = (
            "Trained on "
            + self._set_value_from_pattern(r"num_classes\s*=\s*(\d+)", content)
            + " Channels"
        )
        self._max_depth.value = self._set_value_from_pattern(
            r"max_depth\s*=\s*(\d+)", content
        )
        self._num_trees.value = self._set_value_from_pattern(
            r"num_trees\s*=\s*(\d+)", content
        )
        self._positive_class_id.value = (
            self._set_value_from_pattern(
                r"positive_class_identifier\s*=\s*(\d+)", content
            )
            or 2
        )

    def _update_classifier_metadata(self):
        with open(self._classifier_file.value) as file:
            content = file.read()

        # Ignore rest of function if file contents are empty
        if not content.strip():
            self._classifier_channels.value = "New Classifier"
            return

        self._process_classifier_metadata(content)

        if self._classifier_type.value in self._classifier_type_mapping:
            classifier_class = self._classifier_type_mapping[
                self._classifier_type.value
            ]
            custom_classifier = classifier_class(
                opencl_filename=self._classifier_file.value
            )
        else:
            custom_classifier = None

        self._classifier_statistics_table(custom_classifier)

    def _classifier_statistics_table(self, custom_classifier):
        table, _ = custom_classifier.statistics()

        trans_table = {"filter_name": [], "radius": []}

        for value in table.keys():
            filter_name, radius = (
                value.split("=") if "=" in value else (value, 0)
            )
            trans_table["filter_name"].append(filter_name)
            trans_table["radius"].append(int(radius))

        for i in range(len(next(iter(table.values())))):
            trans_table[str(i)] = [round(table[key][i], 2) for key in table]

        table_df = pd.DataFrame.from_dict(trans_table)

        self._viewer.window.add_dock_widget(
            Table(value=table_df),
            name=os.path.basename(self._classifier_file.value),
        )

    def batch_train(self):
        image_files = os.listdir(self._image_directory.value)
        label_files = os.listdir(self._label_directory.value)

        self._progress_bar.label = f"Training on {len(image_files)} Images"
        self._progress_bar.value = 0
        self._progress_bar.max = len(image_files)

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

        if self._predefined_features.value.value == 1:
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

            print(f"Training Image: {idx+1} of {len(image_files)} : {file}")
            self._progress_bar.value = idx + 1

        self._classifier_statistics_table(custom_classifier)
        self._progress_bar.label = f"Trained on {len(image_files)} Images"

    def batch_predict(self):
        image_files = os.listdir(self._image_directory.value)

        self._progress_bar.label = f"Predicting {len(image_files)} Images"
        self._progress_bar.value = 0
        self._progress_bar.max = len(image_files)

        if self._classifier_type.value in self._classifier_type_mapping:
            classifier_class = self._classifier_type_mapping[
                self._classifier_type.value
            ]
            custom_classifier = classifier_class(
                opencl_filename=self._classifier_file.value
            )
        else:
            custom_classifier = None

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

            print(f"Predicting Image: {idx+1} of {len(image_files)} : {file}")
            self._progress_bar.value = idx + 1

        self._progress_bar.label = f"Predicted {len(image_files)} Images"

    def _custom_apoc_widget(self):
        # self._viewer.window.add_dock_widget(CustomApoc(self._viewer))
        self._viewer.window.add_plugin_dock_widget(
            plugin_name="napari-ndev", widget_name="Custom APOC Feature Set"
        )


class CustomApoc(Container):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
    ):
        super().__init__()
        self
        self._viewer = viewer

        self._original = CheckBox(label="Keep Original Image")
        self._gaussian_blur = LineEdit(label="Gaussian Blur")
        self._DoG = LineEdit(label="Difference of Gauss.")
        self._LoG = LineEdit(label="Laplacian of Gauss.")
        self._SoG = LineEdit(label="Sobel of Gauss.")
        self._sHoG = LineEdit(label="Small Hessian of Gauss.")
        self._lHoG = LineEdit(label="Large Hessian of Gauss.")
        self._median = LineEdit(label="Median")
        self._tophat = LineEdit(label="Top Hat")

        self._generate_string_button = PushButton(
            label="Generate Feature String"
        )
        self._feature_string = TextEdit(label="Custom Feature String")

        self.extend(
            [
                self._original,
                self._gaussian_blur,
                self._DoG,
                self._LoG,
                self._SoG,
                self._sHoG,
                self._lHoG,
                self._median,
                self._tophat,
                self._generate_string_button,
                self._feature_string,
            ]
        )

        self._generate_string_button.clicked.connect(
            self.generate_feature_string
        )

    def generate_feature_string(self):
        def process_feature(prefix, input_str):
            return [
                prefix + num.strip()
                for num in input_str.split(",")
                if num.strip()
            ]

        feature_list = []
        if self._original.value:
            feature_list.append("original")
        feature_list.extend(
            process_feature("gaussian_blur=", self._gaussian_blur.value)
        )
        feature_list.extend(
            process_feature("difference_of_gaussian=", self._DoG.value)
        )
        feature_list.extend(
            process_feature("laplace_box_of_gaussian_blur=", self._LoG.value)
        )
        feature_list.extend(
            process_feature("sobel_of_gaussian_blur=", self._SoG.value)
        )
        feature_list.extend(
            process_feature(
                "small_hessian_eigenvalue_of_gaussian_blur=", self._sHoG.value
            )
        )
        feature_list.extend(
            process_feature(
                "large_hessian_of_eigenvalue_of_gaussian_blur=",
                self._lHoG.value,
            )
        )
        feature_list.extend(
            process_feature("median_sphere=", self._median.value)
        )
        feature_list.extend(
            process_feature("top_hat_sphere=", self._tophat.value)
        )

        self._feature_string.value = " ".join(feature_list)
