import os
import re
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
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
    create_widget,
)
from pyclesperanto_prototype import set_wait_for_kernel_finish
from qtpy.QtWidgets import QTabWidget

from napari_ndev import helpers

if TYPE_CHECKING:
    import napari

# Lazy Imports:
# from aicsimageio import AICSImage
# from aicsimageio.writers import OmeTiffWriter
# from napari import layers
# import apoc


class ApocContainer(Container):
    """
    Container class for managing the ApocContainer widget in napari.

    Parameters:
    -----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.

    Attributes:
    -----------
    _viewer : napari.viewer.Viewer
        The napari viewer instance.

    _image_directory : FileEdit
        Widget for selecting the image directory.

    _label_directory : FileEdit
        Widget for selecting the label directory.

    _output_directory : FileEdit
        Widget for selecting the output directory.

    _classifier_file : FileEdit
        Widget for selecting the classifier file.

    _classifier_type_mapping : dict
        Mapping of classifier types to their corresponding classes.

    _classifier_type : RadioButtons
        Widget for selecting the classifier type.

    _max_depth : SpinBox
        Widget for selecting the number of forests.

    _num_trees : SpinBox
        Widget for selecting the number of trees.

    _positive_class_id : SpinBox
        Widget for selecting the object label ID.

    _image_channels : Select
        Widget for selecting the image channels.

    _channel_order_label : Label
        Label widget for displaying the selected channel order.

    _PDFS : Enum
        Enum for predefined feature sets.

    _predefined_features : ComboBox
        Widget for selecting the features.

    _custom_features : LineEdit
        Widget for entering custom feature string.

    _open_custom_feature_generator : PushButton
        Button for opening the custom feature generator widget.

    _continue_training : CheckBox
        Checkbox for indicating whether to continue training.

    _batch_train_button : PushButton
        Button for training the classifier on image-label pairs.

    _batch_predict_button : PushButton
        Button for predicting labels with the classifier.

    _progress_bar : ProgressBar
        Progress bar widget.

    _image_layer : Select
        Widget for selecting the image layers.

    _label_layer : Widget
        Widget for selecting the label layers.

    _train_image_button : PushButton
        Button for training the classifier on selected layers using labels.

    _predict_image_layer : PushButton
        Button for predicting using the classifier on selected layers.

    _single_result_label : Label
        Label widget for displaying a single result.

    Methods:
    --------
    _update_metadata_from_file()
        Update the metadata from the selected image directory.

    _update_channel_order()
        Update the channel order label based on the selected image channels.

    _set_value_from_pattern(pattern, content)
        Set the value from a pattern in the content.

    _process_classifier_metadata(content)
        Process the classifier metadata from the content.

    _update_classifier_metadata()
        Update the classifier metadata based on the selected classifier file.

    _classifier_statistics_table(custom_classifier)
        Display the classifier statistics table.

    _get_feature_set()
        Get the selected feature set.

    _get_training_classifier_instance()
        Get the training classifier instance based on the selected classifier
        type.

    _get_channel_image(img, channel_index_list)
        Get the channel image based on the selected channel index list.
    """

    def __init__(
        self,
        viewer: "napari.viewer.Viewer" = None,
        # viewer = napari_viewer
    ):
        super().__init__()

        ##############################
        # Lazy Imports
        ##############################
        import apoc

        # from napari.layers import Image as ImageLayer

        self.apoc = apoc

        ##############################
        # Attributes
        ##############################
        self
        self._viewer = viewer if viewer is not None else None

        ##############################
        # Widgets
        ##############################
        self._classifier_file = FileEdit(
            label="Classifier File (.cl)",
            mode="r",
            tooltip="Create a .txt file and rename it to .cl ending.",
        )

        self._continue_training = CheckBox(
            label="Continue Training?",
            value=True,
            tooltip=(
                "Continue training only matters if classifier already exists."
            ),
        )

        self._classifier_type_mapping = {
            "PixelClassifier": apoc.PixelClassifier,
            "ObjectSegmenter": apoc.ObjectSegmenter,
        }

        self._classifier_type = RadioButtons(
            label="Classifier Type",
            value="ObjectSegmenter",
            choices=["ObjectSegmenter", "PixelClassifier"],
            tooltip="Object Segmenter is used for detecting objects of one "
            "class, including connected components. "
            "Pixel Classifier is used to classify pixel-types.",
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

        ######
        # Batch Container
        ######
        self._image_directory = FileEdit(label="Image Directory", mode="d")
        self._label_directory = FileEdit(label="Label Directory", mode="d")
        self._output_directory = FileEdit(label="Output Directory", mode="d")

        self._image_channels = Select(
            label="Image Channels",
            choices=[],
            tooltip=(
                "Channel order should be same for training and prediction."
            ),
        )
        self._channel_order_label = Label(value="Select an Image Channel!")

        self._batch_train_button = PushButton(label="Train")
        self._batch_predict_button = PushButton(label="Predict")

        self._batch_train_container = Container(
            layout="horizontal",
            # label="Train Classifier on Image-Label Pairs",
        )
        self._batch_train_container.extend(
            [self._label_directory, self._batch_train_button]
        )

        self._batch_predict_container = Container(
            layout="horizontal",
            # label="Predict Labels with Classifier on Images"
        )
        self._batch_predict_container.extend(
            [self._output_directory, self._batch_predict_button]
        )

        self._progress_bar = ProgressBar(label="Progress:")

        self._batch_container = Container(layout="vertical")
        self._batch_container.extend(
            [
                self._image_directory,
                self._image_channels,
                self._channel_order_label,
                self._batch_train_container,
                self._batch_predict_container,
                self._progress_bar,
            ]
        )
        #######
        # Viewer Container
        #######
        self._label_layer = create_widget(
            annotation="napari.layers.Labels", label="Labels"
        )
        self._train_image_button = PushButton(
            label="Train classifier on selected layers using label"
        )
        self._predict_image_layer = PushButton(
            label="Predict using classifier on selected layers"
        )
        self._single_result_label = Label()

        self._viewer_container = Container(layout="vertical")
        self._viewer_container.extend(
            [
                # self._image_layer,
                self._label_layer,
                self._train_image_button,
                self._predict_image_layer,
                self._single_result_label,
            ]
        )

        ######
        # Widget Layout
        ######

        self.extend(
            [
                self._classifier_file,
                self._continue_training,
                self._classifier_type,
                self._positive_class_id,
                self._max_depth,
                self._num_trees,
                self._predefined_features,
                self._custom_features,
                self._open_custom_feature_generator,
            ]
        )

        tabs = QTabWidget()

        tabs.addTab(self._batch_container.native, "Batch")
        tabs.addTab(self._viewer_container.native, "Viewer")
        self.native.layout().addWidget(tabs)

        ##############################
        # Event Handling
        ##############################
        self._image_directory.changed.connect(self._update_metadata_from_file)
        self._image_channels.changed.connect(self._update_channel_order)
        self._classifier_file.changed.connect(self._update_classifier_metadata)
        self._open_custom_feature_generator.clicked.connect(
            self._custom_apoc_widget
        )
        self._batch_train_button.clicked.connect(self.batch_train)
        self._batch_predict_button.clicked.connect(self.batch_predict)
        self._train_image_button.clicked.connect(self.image_train)
        self._predict_image_layer.clicked.connect(self.image_predict)

    def _update_metadata_from_file(self):
        from aicsimageio import AICSImage

        _, files = helpers.get_directory_and_files(self._image_directory.value)
        img = AICSImage(files[0])
        self._image_channels.choices = helpers.get_channel_names(img)

    def _update_channel_order(self):
        self._channel_order_label.value = "Selected Channel Order: " + str(
            self._image_channels.value
        )

    ##############################
    # Classifier Related Functions
    ##############################
    def _set_value_from_pattern(self, pattern, content):
        match = re.search(pattern, content)
        return match.group(1) if match else None

    def _process_classifier_metadata(self, content):
        self._classifier_type.value = self._set_value_from_pattern(
            r"classifier_class_name\s*=\s*([^\n]+)", content
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
        if self._viewer is not None:
            self._viewer.window.add_dock_widget(
                Table(value=table_df),
                name=os.path.basename(self._classifier_file.value),
            )

    def _get_feature_set(self):
        if self._predefined_features.value.value == 1:
            return self._custom_features.value
        else:
            return self.apoc.PredefinedFeatureSet[
                self._predefined_features.value.name
            ].value

    def _get_training_classifier_instance(self):
        if self._classifier_type.value == "PixelClassifier":
            return self.apoc.PixelClassifier(
                opencl_filename=self._classifier_file.value,
                max_depth=self._max_depth.value,
                num_ensembles=self._num_trees.value,
            )

        if self._classifier_type.value == "ObjectSegmenter":
            return self.apoc.ObjectSegmenter(
                opencl_filename=self._classifier_file.value,
                positive_class_identifier=self._positive_class_id.value,
                max_depth=self._max_depth.value,
                num_ensembles=self._num_trees.value,
            )

    ##############################
    # Training and Prediction
    ##############################
    def _get_channel_image(self, img, channel_index_list):
        if "S" in img.dims.order:
            channel_img = img.get_image_data("TSZYX", S=channel_index_list)
        else:
            channel_img = img.get_image_data("TCZYX", C=channel_index_list)
        return channel_img

    def batch_train(self):
        from aicsimageio import AICSImage

        image_directory, image_files = helpers.get_directory_and_files(
            self._image_directory.value
        )
        label_directory, _ = helpers.get_directory_and_files(
            self._label_directory.value
        )
        # missing_files = check_for_missing_files(image_files, label_directory)

        log_loc = self._classifier_file.value.with_suffix(".log.txt")
        logger, handler = helpers.setup_logger(log_loc)

        logger.info(
            f"""
        Classifier: {self._classifier_file.value}
        Channels: {self._image_channels.value}
        Num. Files: {len(image_files)}
        Image Directory: {image_directory}
        Label Directory: {label_directory}
        """
        )

        # https://github.com/clEsperanto/pyclesperanto_prototype/issues/163
        set_wait_for_kernel_finish(True)

        self._progress_bar.label = f"Training on {len(image_files)} Images"
        self._progress_bar.value = 0
        self._progress_bar.max = len(image_files)

        if not self._continue_training:
            self.apoc.erase_classifier(self._classifier_file.value)

        custom_classifier = self._get_training_classifier_instance()
        feature_set = self._get_feature_set()

        channel_index_list = [
            self._image_channels.choices.index(channel)
            for channel in self._image_channels.value
        ]

        # Iterate over image files, only pulling label files with an identical
        # name to the image file. Ensuring that files match by some other
        # method would be much more complicated, so I'm leaving it up to the
        # user at this point. In addition, the utilities widget saves with
        # the same name, so this should be a non-issue, if staying within the
        # same workflow.
        for idx, image_file in enumerate(image_files):
            if not (label_directory / image_file.name).exists():
                logger.error(f"Label file missing for {image_file.name}")
                self._progress_bar.value = idx + 1
                continue

            logger.info(f"Training Image {idx+1}: {image_file.name}")

            img = AICSImage(image_directory / image_file.name)
            channel_img = self._get_channel_image(img, channel_index_list)

            lbl = AICSImage(label_directory / image_file.name)
            label = lbl.get_image_data("TCZYX", C=0)

            # <- this is where setting up dask processing would be useful

            try:
                custom_classifier.train(
                    features=feature_set,
                    image=np.squeeze(channel_img),
                    ground_truth=np.squeeze(label),
                    continue_training=True,
                )
                self._progress_bar.value = idx + 1
            except Exception as e:
                logger.error(f"Error training {image_file}: {e}")
                self._progress_bar.value = idx + 1
                continue

        self._classifier_statistics_table(custom_classifier)
        self._progress_bar.label = f"Trained on {len(image_files)} Images"
        logger.removeHandler(handler)

    def image_train(self):
        layer_name = self._viewer.layers.selection.active.name

        # layer_name = self._image_layer.value[0].name
        print(f"Training on {layer_name}")
        image_list = [image.data for image in self._viewer.layers.selection]
        # image_list = [image.data for image in self._image_layer.value]
        image_stack = np.stack(image_list, axis=0)
        label = self._label_layer.value.data

        # https://github.com/clEsperanto/pyclesperanto_prototype/issues/163
        set_wait_for_kernel_finish(True)

        if not self._continue_training:
            self.apoc.erase_classifier(self._classifier_file.value)

        custom_classifier = self._get_training_classifier_instance()
        feature_set = self._get_feature_set()

        custom_classifier.train(
            features=feature_set,
            image=np.squeeze(image_stack),
            ground_truth=np.squeeze(label),
            continue_training=True,
        )

        self._single_result_label.value = f"Trained on {layer_name}"

    def _get_prediction_classifier_instance(self):
        if self._classifier_type.value in self._classifier_type_mapping:
            classifier_class = self._classifier_type_mapping[
                self._classifier_type.value
            ]
            return classifier_class(
                opencl_filename=self._classifier_file.value
            )
        else:
            return None

    def batch_predict(self):
        from aicsimageio import AICSImage
        from aicsimageio.writers import OmeTiffWriter

        image_directory, image_files = helpers.get_directory_and_files(
            dir=self._image_directory.value,
        )

        log_loc = self._output_directory.value / "log.txt"
        logger, handler = helpers.setup_logger(log_loc)

        logger.info(
            f"""
        Classifier: {self._classifier_file.value}
        Channels: {self._image_channels.value}
        Num. Files: {len(image_files)}
        Image Directory: {image_directory}
        Output Directory: {self._output_directory.value}"""
        )

        # https://github.com/clEsperanto/pyclesperanto_prototype/issues/163
        set_wait_for_kernel_finish(True)

        self._progress_bar.label = f"Predicting {len(image_files)} Images"
        self._progress_bar.value = 0
        self._progress_bar.max = len(image_files)

        custom_classifier = self._get_prediction_classifier_instance()

        channel_index_list = [
            self._image_channels.choices.index(channel)
            for channel in self._image_channels.value
        ]

        for idx, file in enumerate(image_files):
            logger.info(f"Predicting Image {idx+1}: {file.name}")

            img = AICSImage(file)
            channel_img = self._get_channel_image(img, channel_index_list)
            squeezed_dim_order = helpers.get_squeezed_dim_order(img)

            # <- this is where setting up dask processing would be useful

            try:
                result = custom_classifier.predict(
                    image=np.squeeze(channel_img)
                )
            except Exception as e:
                logger.error(f"Error predicting {file}: {e}")
                self._progress_bar.value = idx + 1
                continue

            save_data = np.asarray(result)
            if save_data.max() > 65535:
                save_data = save_data.astype(np.int32)
            else:
                save_data = save_data.astype(np.int16)

            OmeTiffWriter.save(
                data=save_data,
                uri=self._output_directory.value / (file.stem + ".tiff"),
                dim_order=squeezed_dim_order,
                channel_names=["Labels"],
                physical_pixel_sizes=img.physical_pixel_sizes,
            )
            del result

            self._progress_bar.value = idx + 1

        self._progress_bar.label = f"Predicted {len(image_files)} Images"
        logger.removeHandler(handler)

    def image_predict(self):
        layer_name = self._viewer.layers.selection.active.name
        print(f"Predicting {layer_name}")
        # https://github.com/clEsperanto/pyclesperanto_prototype/issues/163
        set_wait_for_kernel_finish(True)

        image_list = [image.data for image in self._viewer.layers.selection]
        image_stack = np.stack(image_list, axis=0)
        scale = self._viewer.layers.selection.active.scale

        custom_classifier = self._get_prediction_classifier_instance()

        result = custom_classifier.predict(image=np.squeeze(image_stack))

        # sometimes, input layers may have shape with 1s, like (1,1,10,10)
        # however, we are squeezing the input, so the reuslt will have shape
        # (10,10), and therefore scale needs to accomodate dropped axes
        result_dims = result.ndim
        if len(scale) > result_dims:
            scale = scale[-result_dims:]

        self._viewer.add_labels(result, scale=scale)

        self._single_result_label.value = f"Predicted {layer_name}"

        return result

    def _custom_apoc_widget(self):
        if self._viewer is not None:
            self._viewer.window.add_plugin_dock_widget(
                plugin_name="napari-ndev",
                widget_name="Custom APOC Feature Set",
            )
        else:
            return
