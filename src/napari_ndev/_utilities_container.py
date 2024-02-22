import ast
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import dask.array as da
import numpy as np
import xarray as xr
from aicsimageio import AICSImage
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeTiffWriter
from magicgui.widgets import (
    CheckBox,
    Container,
    FileEdit,
    FloatSpinBox,
    LineEdit,
    PushButton,
    Select,
    TextEdit,
    create_widget,
)
from napari import layers

from napari_ndev import helpers

if TYPE_CHECKING:
    import napari

PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, da.Array]
MetaArrayLike = Union[ArrayLike, xr.DataArray]
ImageLike = Union[
    PathLike, ArrayLike, MetaArrayLike, List[MetaArrayLike], List[PathLike]
]


class UtilitiesContainer(Container):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
    ):
        super().__init__()
        ##############################
        # Attributes
        ##############################
        self
        self._viewer = viewer
        self._img_data = None
        self._image_save_dims = None
        self._label_save_dims = None
        self._p_sizes = None
        ##############################
        # Widgets
        ##############################
        self._files = FileEdit(label="File(s)", mode="rm")

        self._open_image_button = PushButton(label="Open Images")

        self._save_directory = FileEdit(label="Save Directory", mode="d")
        self._save_name = LineEdit(label="File Save Name")

        self._metadata_from_selected_layer = PushButton(
            label="Update Metadata from Selected Layer"
        )

        self._dim_order = LineEdit(label="Dimension Order")
        self._channel_names = LineEdit(label="Channel Name(s)")

        self._physical_pixel_sizes_z = FloatSpinBox(
            value=1, step=0.00000001, label="Z Pixel Size, um"
        )
        self._physical_pixel_sizes_y = FloatSpinBox(
            value=1, step=0.00000001, label="Y Pixel Size, um"
        )
        self._physical_pixel_sizes_x = FloatSpinBox(
            value=1, step=0.00000001, label="X Pixel Size, um"
        )

        # Use a function for layer inputs so that it is constantly updated
        # when the dependency changes
        def current_layers(_):
            return [
                x for x in self._viewer.layers if isinstance(x, layers.Image)
            ]

        self._image_layer = Select(
            choices=current_layers, nullable=False, label="Images"
        )  # use no value and allow user to unhighlight layers

        self._concatenate_image_files = CheckBox(label="Concatenate Files")
        self._concatenate_image_layers = CheckBox(
            label="Concatenate Image Layers"
        )

        self._save_image_button = PushButton(label="Save Images")

        self._labels_layer = create_widget(
            annotation="napari.layers.Labels", label="Labels"
        )
        self._save_labels_button = PushButton(label="Save Labels")

        self._shapes_layer = create_widget(
            annotation="napari.layers.Shapes", label="Shapes"
        )
        self._save_shapes_button = PushButton(label="Save Shapes as Labels")

        self._results = TextEdit(label="Info")

        # Container Widget Order
        self.extend(
            [
                self._files,
                self._open_image_button,
                self._dim_order,
                self._channel_names,
                self._physical_pixel_sizes_z,
                self._physical_pixel_sizes_y,
                self._physical_pixel_sizes_x,
                self._image_layer,
                self._metadata_from_selected_layer,
                self._concatenate_image_files,
                self._concatenate_image_layers,
                self._save_directory,
                self._save_name,
                self._save_image_button,
                self._labels_layer,
                self._save_labels_button,
                self._shapes_layer,
                self._save_shapes_button,
                self._results,
            ]
        )
        ##############################
        # Event Handling
        ##############################
        self._files.changed.connect(self.update_metadata_from_file)
        self._open_image_button.clicked.connect(self.open_images)
        self._metadata_from_selected_layer.clicked.connect(
            self.update_metadata_from_layer
        )

        self._save_image_button.clicked.connect(self.save_ome_tiff)
        self._save_labels_button.clicked.connect(self.save_labels)
        self._save_shapes_button.clicked.connect(self.save_shapes_as_labels)
        self._results._on_value_change()

    def _update_metadata(self, img):
        self._dim_order.value = img.dims.order

        self._squeezed_dims = helpers.get_squeezed_dim_order(img)
        self._channel_names.value = helpers.get_channel_names(img)

        self._physical_pixel_sizes_z.value = img.physical_pixel_sizes.Z or 0
        self._physical_pixel_sizes_y.value = img.physical_pixel_sizes.Y
        self._physical_pixel_sizes_x.value = img.physical_pixel_sizes.X

    def update_metadata_from_file(self):
        img = AICSImage(self._files.value[0])
        self._img = img
        self._update_metadata(img)
        self._save_name.value = str(self._files.value[0].stem + ".tif")

    def update_metadata_from_layer(self):
        try:
            img = self._image_layer.value[0].metadata["aicsimage"]
            self._img = img
            self._update_metadata(img)
        except KeyError as e:
            self._results.value = "KeyError: " + str(e)

    def open_images(self):
        self._viewer.open(self._files.value, plugin="napari-aicsimageio")

    def concatenate_images(
        self,
        concatenate_files: bool,
        files: List[PathLike],
        concatenate_layers: bool,
        layers: List[layers.Image],
    ):
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
        return PhysicalPixelSizes(
            self._physical_pixel_sizes_z.value,
            self._physical_pixel_sizes_y.value,
            self._physical_pixel_sizes_x.value,
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

    def save_ome_tiff(self) -> None:
        self._img_data = self.concatenate_images(
            self._concatenate_image_files.value,
            self._files.value,
            self._concatenate_image_layers.value,
            self._image_layer.value,
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
        label_data = self._labels_layer.value.data

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
        # inherit shape from selected image layer or else a default
        if self._image_layer.value:
            label_dim = self._image_layer.value[0].data.shape
        else:
            label_dim = self._image_layer.choices[0].data.shape

        # drop last axis if represents RGB image
        label_dim = label_dim[:-1] if label_dim[-1] == 3 else label_dim

        shapes = self._shapes_layer.value
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
