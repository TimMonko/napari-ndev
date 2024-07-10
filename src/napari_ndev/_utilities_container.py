import ast
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import numpy as np
from magicgui.widgets import (
    CheckBox,
    Container,
    FileEdit,
    Label,
    LineEdit,
    PushButton,
    TextEdit,
    TupleEdit,
    create_widget,
)

from napari_ndev import helpers

if TYPE_CHECKING:
    import napari
    from aicsimageio import AICSImage
    from napari.layers import Image as ImageLayer


class UtilitiesContainer(Container):
    """
    A container class that provides utility functions for working with napari
    images and layers.

    Parameters:
    - viewer: napari.viewer.Viewer
        The napari viewer instance.

    Attributes:
    - _viewer: napari.viewer.Viewer
        The napari viewer instance.
    - _img_data: numpy.ndarray or None
        The concatenated image data.
    - _image_save_dims: str or None
        The dimension order for saving images.
    - _label_save_dims: str or None
        The dimension order for saving labels.
    - _p_sizes: PhysicalPixelSizes
        The physical pixel sizes for the image.

    Widgets:
    - _files: FileEdit
        Widget for selecting file(s).
    - _open_image_button: PushButton
        Button for opening images.
    - _save_directory: FileEdit
        Widget for selecting the save directory.
    - _save_name: LineEdit
        Widget for entering the file save name.
    - _metadata_from_selected_layer: PushButton
        Button for updating metadata from the selected layer.
    - _dim_order: LineEdit
        Widget for entering the dimension order.
    - _channel_names: LineEdit
        Widget for entering the channel names.
    - _physical_pixel_sizes_z: FloatSpinBox
        Widget for entering the Z pixel size in micrometers.
    - _physical_pixel_sizes_y: FloatSpinBox
        Widget for entering the Y pixel size in micrometers.
    - _physical_pixel_sizes_x: FloatSpinBox
        Widget for entering the X pixel size in micrometers.
    - _image_layer: Select
        Widget for selecting the image layer.
    - _concatenate_image_files: CheckBox
        Checkbox for concatenating image files.
    - _concatenate_image_layers: CheckBox
        Checkbox for concatenating image layers.
    - _save_image_button: PushButton
        Button for saving images.
    - _labels_layer: Widget
        Widget for working with labels layer.
    - _save_labels_button: PushButton
        Button for saving labels.
    - _shapes_layer: Widget
        Widget for working with shapes layer.
    - _save_shapes_button: PushButton
        Button for saving shapes as labels.
    - _results: TextEdit
        Widget for displaying information.

    Methods:
    - _update_metadata(img)
        Update the metadata based on the given image.
    - update_metadata_from_file()
        Update the metadata from the selected file.
    - update_metadata_from_layer()
        Update the metadata from the selected layer.
    - open_images()
        Open the selected images in the napari viewer.
    - concatenate_images(concatenate_files, files, concatenate_layers, layers)
        Concatenate the image data based on the selected options.
    - p_sizes()
        Get the physical pixel sizes.
    - _get_save_loc(parent)
        Get the save location based on the parent directory.
    - _common_save_logic(data, uri, dim_order, channel_names, layer)
        Common logic for saving data as OME-TIFF.
    - save_ome_tiff()
        Save the concatenated image data as OME-TIFF.
    - save_labels()
        Save the labels data.
    - save_shapes_as_labels()
        Save the shapes data as labels.
    """

    def __init__(
        self,
        viewer: "napari.viewer.Viewer" = None,
    ):
        super().__init__()

        ##############################
        # Attributes
        ##############################
        self
        self._viewer = viewer if viewer is not None else None
        self._img_data = None
        self._image_save_dims = None
        self._label_save_dims = None
        self._p_sizes = None
        self._squeezed_dims = None

        ##############################
        # Widgets
        ##############################

        self._file_metadata_update = PushButton(label="File")
        self._layer_metadata_update = PushButton(label="Selected Layer")
        self._metadata_container = Container(
            layout="horizontal", label="Update Metadata from"
        )
        self._metadata_container.append(self._layer_metadata_update)
        self._metadata_container.append(self._file_metadata_update)

        self._files = FileEdit(
            label="File(s)",
            mode="rm",
            tooltip="Select file(s) to load.",
        )
        self._open_image_button = PushButton(label="Open File(s)")
        self._open_image_update_metadata = CheckBox(
            value=True, label="Update Metadata on File Selection"
        )
        self._open_image_container = Container(layout="horizontal")
        self._open_image_container.append(self._open_image_button)
        self._open_image_container.append(self._open_image_update_metadata)

        self._save_directory = FileEdit(
            label="Save Directory",
            mode="d",
            tooltip="Directory where images will be saved.",
        )
        self._save_name = LineEdit(
            label="File Save Name",
            tooltip="Name of saved file. Helpful to include a"
            ".ome/.tif/.tiff extension.",
        )

        self._dim_order = Label(
            label="Dimension Order: ",
            tooltip="Sanity check for available dimensions.",
        )
        self._scenes = Label(
            label="Number of Scenes: ",
        )
        self._info_container = Container(layout="horizontal")
        self._info_container.append(self._dim_order)
        self._info_container.append(self._scenes)

        self._channel_names = LineEdit(
            label="Channel Name(s)",
            tooltip="Enter channel names as a list. If left blank or the "
            "channel names are not the proper length, then default channel "
            "names will be used.",
        )

        self._scale_tuple = TupleEdit(
            value=(0.0000, 1.0000, 1.0000),
            label="Scale, ZYX",
            tooltip="Pixel size, usually in μm",
            options={"step": 0.0001},
        )
        self._scale_layers = PushButton(
            label="Scale Layer(s)",
            tooltip="Scale the selected layer(s) based on the given scale.",
        )
        self._scale_container = Container(
            layout="horizontal",
            label="Scale Selected",
        )
        self._scale_container.append(self._scale_layers)

        self._scenes_to_extract = LineEdit(
            # label="Scenes to Extract",
            tooltip="Enter the scenes to extract as a list. If left blank "
            "then all scenes will be extracted.",
        )
        self._extract_scenes = PushButton(
            label="Extract and Save Scenes",
            tooltip="Extract scenes from a single selected file.",
        )
        self._scene_container = Container(
            layout="horizontal",
            label="Extract Scenes",
            tooltip="Must be in list index format. Ex: [0, 1, 2] or [5:10]",
        )
        self._scene_container.append(self._scenes_to_extract)
        self._scene_container.append(self._extract_scenes)

        self._concatenate_image_files = CheckBox(
            value=True,
            label="Concatenate Files",
            tooltip="Concatenate files in the selected directory. Removes "
            "blank channels.",
        )
        self._concatenate_image_layers = CheckBox(
            label="Concatenate Image Layers",
            tooltip="Concatenate image layers in the viewer. Removes empty.",
        )
        self._concatenate_container = Container(
            layout="horizontal",
            label="Image Save Options",
        )
        self._concatenate_container.append(self._concatenate_image_files)
        self._concatenate_container.append(self._concatenate_image_layers)

        self._save_image_button = PushButton(
            label="Save Images",
            tooltip="Save the concatenated image data as OME-TIFF.",
        )

        self._labels_layer = create_widget(
            annotation="napari.layers.Labels", label="Labels"
        )
        self._save_labels_button = PushButton(
            label="Save Labels", tooltip="Save the labels data as OME-TIFF."
        )

        self._shapes_layer = create_widget(
            annotation="napari.layers.Shapes", label="Shapes"
        )
        self._save_shapes_button = PushButton(
            label="Save Shapes as Labels",
            tooltip="Save the shapes data as labels (OME-TIFF) according to "
            "selected image layer dimensions.",
        )
        self._save_container = Container(
            layout="horizontal",
            label="Save Selected Layers",
        )
        self._save_container.append(self._save_image_button)
        self._save_container.append(self._save_labels_button)
        self._save_container.append(self._save_shapes_button)

        self._results = TextEdit(label="Info")

        # Container Widget Order
        self.extend(
            [
                self._save_directory,
                self._files,
                self._open_image_container,
                self._save_name,
                self._metadata_container,
                self._info_container,
                self._channel_names,
                self._scale_tuple,
                self._scale_container,
                self._scene_container,
                self._concatenate_container,
                self._save_container,
                self._results,
            ]
        )

        ##############################
        # Event Handling
        ##############################
        self._files.changed.connect(self.update_metadata_from_file)
        self._open_image_button.clicked.connect(self.open_images)
        self._layer_metadata_update.clicked.connect(
            self.update_metadata_from_layer
        )
        self._file_metadata_update.clicked.connect(
            self.update_metadata_from_file
        )
        self._scale_layers.clicked.connect(self.rescale_by)
        self._extract_scenes.clicked.connect(self.save_scenes_ome_tiff)
        self._save_image_button.clicked.connect(self.save_ome_tiff)
        self._save_labels_button.clicked.connect(self.save_labels)
        self._save_shapes_button.clicked.connect(self.save_shapes_as_labels)
        self._results._on_value_change()

    def _update_metadata(self, img: "AICSImage"):
        self._dim_order.value = img.dims.order

        self._squeezed_dims = helpers.get_squeezed_dim_order(img)
        self._channel_names.value = helpers.get_channel_names(img)

        self._scale_tuple.value = (
            img.physical_pixel_sizes.Z or 0,
            img.physical_pixel_sizes.Y or 1,
            img.physical_pixel_sizes.X or 1,
        )

    def update_metadata_from_file(self):
        from aicsimageio import AICSImage

        self._save_name.value = str(self._files.value[0].stem + ".tiff")

        if self._open_image_update_metadata.value:
            img = AICSImage(self._files.value[0])
            self._img = img
            self._update_metadata(img)
            self._scenes.value = len(img.scenes)

    def update_metadata_from_layer(self):
        selected_layer = self._viewer.layers.selection.active
        try:
            self._update_metadata(selected_layer.metadata["aicsimage"])
        except AttributeError:
            self._results.value = (
                "Tried to update metadata, but no layer selected."
            )
        except KeyError:
            scale = selected_layer.scale
            self._scale_tuple.value = (
                scale[-3] if len(scale) >= 3 else 0.0,
                scale[-2],
                scale[-1],
            )
            self._results.value = (
                "Tried to update metadata, but could only update scale"
                " because layer not opened with aicsimageio"
            )

    def open_images(self):
        self._viewer.open(self._files.value, plugin="napari-aicsimageio")

    def rescale_by(self):
        layers = self._viewer.layers.selection
        scale_tup = self._scale_tuple.value
        for layer in layers:
            scale_len = len(layer.scale)
            layer.scale = scale_tup[1:3] if scale_len == 2 else scale_tup
        # from napari.layers import Image as ImageLayer
        # from napari.layers import Labels as LabelsLayer

        # layers = self._viewer.layers.selection
        # scale_tup = self._scale_tuple.value
        # for layer in layers:
        #     if isinstance(layer, (ImageLayer, LabelsLayer)):
        #         scale_len = len(layer.scale)
        #         layer.scale = scale_tup[1:3] if scale_len == 2 else scale_tup
        #     else:
        #         continue

    def concatenate_images(
        self,
        concatenate_files: bool,
        files: List[Union[str, Path]],
        concatenate_layers: bool,
        layers: List["ImageLayer"],
    ):
        from aicsimageio import AICSImage

        array_list = []
        if concatenate_files:
            for file in files:
                img = AICSImage(file)
                if "S" in img.dims.order:
                    img_data = img.get_image_data("TSZYX")
                else:
                    img_data = img.data

                # iterate over all channels and only keep if not blank
                for idx in range(img_data.shape[1]):
                    array = img_data[:, [idx], :, :, :]
                    if array.max() > 0:
                        array_list.append(array)

        # <- fix if RGB image is the layer data
        if concatenate_layers:
            for layer in layers:
                layer_data = layer.data
                # convert to 5D array for compatability with image dims
                while len(layer_data.shape) < 5:
                    layer_data = np.expand_dims(layer_data, axis=0)
                array_list.append(layer_data)

        return np.concatenate(array_list, axis=1)

    @property
    def p_sizes(self):
        from aicsimageio.types import PhysicalPixelSizes

        return PhysicalPixelSizes(
            self._scale_tuple.value[0],
            self._scale_tuple.value[1],
            self._scale_tuple.value[2],
        )

    def _get_save_loc(self, parent):
        save_directory = self._save_directory.value / parent
        save_directory.mkdir(parents=False, exist_ok=True)
        return save_directory / self._save_name.value

    def _common_save_logic(
        self,
        data: np.ndarray,
        uri: Path,
        dim_order: str,
        channel_names: List[str],
        layer: str,
    ) -> None:
        from aicsimageio.writers import OmeTiffWriter

        # AICSImage does not allow saving labels as np.int64
        # napari generates labels differently depending on the OS
        # so we need to convert to np.int32 in case np.int64 generated
        # see: https://github.com/napari/napari/issues/5545
        # This is a failsafe
        if data.dtype == np.int64:
            data = data.astype(np.int32)

        try:
            OmeTiffWriter.save(
                data=data,
                uri=uri,
                dim_order=dim_order or None,
                channel_names=channel_names or None,
                physical_pixel_sizes=self.p_sizes,
            )
            self._results.value = f"Saved {layer}: " + str(
                self._save_name.value
            )
        # if ValueError is raised, save with default channel names
        except ValueError as e:
            OmeTiffWriter.save(
                data=data,
                uri=uri,
                dim_order=dim_order,
                physical_pixel_sizes=self.p_sizes,
            )
            self._results.value = (
                "ValueError: "
                + str(e)
                + "\nSo, saved with default channel names: \n"
                + str(self._save_name.value)
            )
        return

    def save_scenes_ome_tiff(self) -> None:
        from aicsimageio import AICSImage

        img = AICSImage(self._files.value[0])
        scenes = self._scenes_to_extract.value
        scenes_list = ast.literal_eval(scenes) if scenes else None
        save_directory = self._save_directory.value / "Images"
        save_directory.mkdir(parents=False, exist_ok=True)
        for scene in scenes_list:
            img.set_scene(scene)

            img_save_name = (
                f'{self._save_name.value.split(".")[0]}'
                f"_scene_{img.current_scene}.ome.tiff"
            )
            img_save_loc = save_directory / img_save_name

            # get channel names from widget if truthy
            cnames = self._channel_names.value
            channel_names = ast.literal_eval(cnames) if cnames else None

            self._common_save_logic(
                data=img.data,
                uri=img_save_loc,
                dim_order="TCZYX",
                channel_names=channel_names,
                layer=f"Scene: {img.current_scene}",
            )
        return

    def save_ome_tiff(self) -> None:
        self._img_data = self.concatenate_images(
            self._concatenate_image_files.value,
            self._files.value,
            self._concatenate_image_layers.value,
            list(self._viewer.layers.selection),
        )
        img_save_loc = self._get_save_loc("Images")
        # get channel names from widget if truthy
        cnames = self._channel_names.value
        channel_names = ast.literal_eval(cnames) if cnames else None

        self._common_save_logic(
            data=self._img_data,
            uri=img_save_loc,
            dim_order="TCZYX",
            channel_names=channel_names,
            layer="Image",
        )
        return self._img_data

    def save_labels(self) -> None:
        label_data = self._viewer.layers.selection.active.data

        if label_data.max() > 65535:
            label_data = label_data.astype(np.int32)
        else:
            label_data = label_data.astype(np.int16)

        label_save_loc = self._get_save_loc("Labels")

        self._common_save_logic(
            data=label_data,
            uri=label_save_loc,
            dim_order=self._squeezed_dims,
            channel_names=["Labels"],
            layer="Labels",
        )
        return label_data

    def save_shapes_as_labels(self) -> None:
        from napari.layers import Image as ImageLayer

        # inherit shape from selected image layer or else a default
        image_layers = [
            x for x in self._viewer.layers if isinstance(x, ImageLayer)
        ]
        label_dim = image_layers[0].data.shape

        # drop last axis if represents RGB image
        label_dim = label_dim[:-1] if label_dim[-1] == 3 else label_dim

        shapes = self._viewer.layers.selection.active
        shapes_as_labels = shapes.to_labels(labels_shape=label_dim)
        shapes_as_labels = shapes_as_labels.astype(np.int16)

        shapes_save_loc = self._get_save_loc("ShapesAsLabels")

        self._common_save_logic(
            data=shapes_as_labels,
            uri=shapes_save_loc,
            dim_order=self._squeezed_dims,
            channel_names=["Shapes"],
            layer="Shapes",
        )

        return shapes_as_labels
