import ast
import os
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
    PushButton,
    Select,
    TextEdit,
    create_widget,
)
from napari import layers
from napari.viewer import current_viewer

if TYPE_CHECKING:
    import napari

PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, da.Array]
MetaArrayLike = Union[ArrayLike, xr.DataArray]
ImageLike = Union[
    PathLike, ArrayLike, MetaArrayLike, List[MetaArrayLike], List[PathLike]
]


class MetaImg(Container):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
    ):
        super().__init__()
        self
        self._viewer = viewer
        self._img_data = None
        self._layer_save_dims = None
        self._p_sizes = None

        self._files = FileEdit(label="File(s)", mode="rm")

        self._open_image_button = PushButton(label="Open Images")

        self._save_directory = FileEdit(label="Save Directory", mode="d")
        self._save_name = create_widget(
            annotation=str, value=None, label="File Save Name"
        )

        self._metadata_from_selected_layer = PushButton(
            label="Update Metadata from Selected Layer"
        )

        self._dim_order = create_widget(
            annotation=str, label="Dimension Order"
        )
        self._channel_names = create_widget(
            annotation=str, label="Channel Name(s)"
        )

        self._physical_pixel_sizes_z = FloatSpinBox(
            value=1, step=0.00000001, label="Z Pixel Size, um"
        )
        self._physical_pixel_sizes_y = FloatSpinBox(
            value=1, step=0.00000001, label="Y Pixel Size, um"
        )
        self._physical_pixel_sizes_x = FloatSpinBox(
            value=1, step=0.00000001, label="X Pixel Size, um"
        )

        def current_layers(_):
            return [
                x
                for x in current_viewer().layers
                if isinstance(x, layers.Image)
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

        # Callbacks
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
        self._layer_save_dims = "".join(
            [
                d
                for d in img.dims.order
                if d != "C" and img.dims._dims_shape[d] > 1
            ]
        )
        self._channel_names.value = img.channel_names

        if img.physical_pixel_sizes.Z is not None:
            self._physical_pixel_sizes_z.value = img.physical_pixel_sizes.Z
        elif img.physical_pixel_sizes.Z is None:
            self._physical_pixel_sizes_z.value = 0
        self._physical_pixel_sizes_y.value = img.physical_pixel_sizes.Y
        self._physical_pixel_sizes_x.value = img.physical_pixel_sizes.X

        self._save_name.value = str(
            os.path.splitext(os.path.basename(self._files.value[0]))[0]
            + ".tif"
        )

    def update_metadata_from_file(self):
        print(self._files.value[0])
        img = AICSImage(self._files.value[0])
        self._img = img
        self._update_metadata(img)

    def update_metadata_from_layer(self):
        img = self._image_layer.value[0].metadata["aicsimage"]
        self._img = img
        self._update_metadata(img)

    def open_images(self):
        self._viewer.open(self._files.value, plugin="napari-aicsimageio")

    def concatenate_images(self):
        array_list = []
        if self._concatenate_image_files.value:
            for file in self._files.value:
                img = AICSImage(file)

                for idx, _ in enumerate(img.channel_names):
                    array = img.data[:, [idx], :, :, :]

                    if array.max() > 0:
                        array_list.append(array)

        if self._concatenate_image_layers.value:
            for layer in self._image_layer.value:
                print(layer.data.shape)
                layer_reshape = layer.data[np.newaxis, np.newaxis, np.newaxis]
                array_list.append(layer_reshape)

        self._img_data = np.concatenate(array_list, axis=1)

    def _get_p_sizes(self):
        self._p_sizes = PhysicalPixelSizes(
            self._physical_pixel_sizes_z.value,
            self._physical_pixel_sizes_y.value,
            self._physical_pixel_sizes_x.value,
        )

    def _get_save_loc(self, parent):
        save_directory = self._save_directory.value / parent
        save_directory.mkdir(parents=False, exist_ok=True)
        save_loc = save_directory / self._save_name.value
        return save_loc

    def save_ome_tiff(self):
        self.concatenate_images()
        self._get_p_sizes()
        img_save_loc = self._get_save_loc("Images")

        try:
            OmeTiffWriter.save(
                data=self._img_data,
                uri=img_save_loc,
                dim_order=self._dim_order.value,
                channel_names=ast.literal_eval(self._channel_names.value),
                physical_pixel_sizes=self._p_sizes,
            )
            # print("saved:", self._save_name.value)
            self._results.value = "Saved image: " + str(self._save_name.value)

        except ValueError as e:
            OmeTiffWriter.save(
                data=self._img_data,
                uri=img_save_loc,
                dim_order=self._dim_order.value,
                physical_pixel_sizes=self._p_sizes,
            )
            self._results.value = (
                "ValueError: "
                + str(e)
                + "\nSo, saved with default file names: \n"
                + str(self._save_name.value)
            )

    def save_labels(self):
        lbl = self._labels_layer.value.data
        self._get_p_sizes()
        lbl_save_loc = self._get_save_loc("Labels")

        OmeTiffWriter.save(
            data=lbl,
            uri=lbl_save_loc,
            dim_order=self._layer_save_dims,
            channel_names=["Labels"],
            physical_pixel_sizes=self._p_sizes,
        )
        self._results.value = "Saved labels: " + str(self._save_name.value)

    def save_shapes_as_labels(self):
        shapes = self._shapes_layer.value

        # inherit shape from selected image layer or else a default
        if self._image_layer.value:
            label_dim = self._image_layer.value[0].data.shape
            print("image layer:", self._image_layer.value[0])
        else:
            label_dim = self._image_layer.choices[0].data.shape

        shapes_as_labels = shapes.to_labels(labels_shape=label_dim)

        self._get_p_sizes()
        shapes_save_loc = self._get_save_loc("Shapes")

        OmeTiffWriter.save(
            data=shapes_as_labels,
            uri=shapes_save_loc,
            dim_order=self._layer_save_dims,
            channel_names=["Shapes"],
            physical_pixel_sizes=self._p_sizes,
        )

        try:
            self._results.value = (
                "Saved shapes as labels: "
                + str(self._save_name.value)
                + "\nwith dimensions inherited from: "
                + str(self._image_layer.value[0])
            )
        except IndexError:
            self._results.value = (
                "Saved shapes as labels: "
                + str(self._save_name.value)
                + "\nWarning: no image layer selected "
                + "so inherited dimensions from: "
                + str(self._image_layer.choices[0])
            )
