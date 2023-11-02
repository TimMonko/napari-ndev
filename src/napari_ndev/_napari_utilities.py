import os
import pathlib
import string

import numpy as np
from aicsimageio.writers import OmeTiffWriter
from magicgui import magic_factory
from napari import layers
from napari.types import LayerDataTuple


def init_rescale_by(rescale_by):
    @rescale_by.inherit_from.changed.connect
    @rescale_by.scale_in_z.changed.connect
    def _inherit_from():
        print("changed")
        if rescale_by.scale_in_z.value is False:
            rescale_by.scale_y.value = rescale_by.inherit_from.value.scale[0]
            rescale_by.scale_x.value = rescale_by.inherit_from.value.scale[1]
            print(rescale_by.scale_x.value)
        if rescale_by.scale_in_z.value is True:
            rescale_by.scale_z.value = rescale_by.inherit_from.value.scale[0]
            rescale_by.scale_y.value = rescale_by.inherit_from.value.scale[1]
            rescale_by.scale_x.value = rescale_by.inherit_from.value.scale[2]


@magic_factory(
    widget_init=init_rescale_by,
    scale_x=dict(widget_type="FloatSpinBox", step=0.00000001),
    scale_y=dict(widget_type="FloatSpinBox", step=0.00000001),
    scale_z=dict(widget_type="FloatSpinBox", step=0.00000001),
)
def rescale_by(
    layer: layers.Layer,
    inherit_from: layers.Layer = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
    scale_in_z: bool = False,
) -> LayerDataTuple:
    """
    Rescale an image by a given set of scale factors.

    Parameters
    ----------
    layer : Layer
        selected layer type will be returned in LayerDataTuple
    scale_x : float, optional
        factor by which to scale the image along the x axis.
        The default is 1.0.
    scale_y : float, optional
        factor by which to scale the image along the y dimension.
        The default is 1.0.
    scale_z : float, optional
        factor by which to scale the image along the z dimension.
        The default is 1.0.
    scale_in_z : bool = False
        if True, then scaling is additionally done to Z access
        (not possible with 2D layer inputs)

    Returns
    -------
    LayerDataTuple overwriting original layer with new scale
    """

    if scale_in_z is False:
        scale_factors = np.asarray([scale_y, scale_x])
    elif scale_in_z is True:
        scale_factors = np.asarray([scale_z, scale_y, scale_x])

    return (layer.data, {"name": layer.name, "scale": scale_factors})


@magic_factory(
    auto_call=False,
    result_widget=True,
    call_button="Save Layers to Output Folders",
    annotation_type=dict(
        widget_type="RadioButtons", choices=["Labels", "Shapes"]
    ),
    file_directory=dict(
        widget_type="FileEdit", mode="d", label="File Directory"
    ),
    output_folder_prefix=dict(widget_type="LineEdit", label="Output Folder"),
    save_suffix=dict(widget_type="LineEdit", label="Save Suffix"),
)
def annotation_saver(
    image: layers.Image,
    labels: layers.Labels,
    shapes: layers.Shapes,
    annotation_type: str = "Labels",
    file_directory=pathlib.Path(),
    output_folder_prefix="Annotated",
    save_suffix=".tif",
):
    """Annotation Saver

    Used for annotating images and saving images of interest into a
    folder for the image and a folder for the labels. The GUI allows
    selecting of the intended label layer and image layer, as well as
    the prefix for the output folders. These output folders can already
    exist: mkdir(exist_ok=True), so should be used to save multiple
    images within the same folder group. Images are saved as ome.tif
    with aicsimageio.OmeTiffWriter

    Parameters
    ----------
    image : napari.layers.layers.Image
        Image layer to save
    labels : napari.layers.layers.Labels
        Labels layer (annotation) to save
    file_directory : pathlib.Path
        Top-level folder where separate image and labels folders are present
    output_folder_prefix : str
        Prefix to _images or _labels folder created in `file_directory`,
        by default "Annotated"
    save_suffix : str
        File ending can be changed if needed for interoperability, but still
        saves as an OME-TIFF with aicsimageio.OmeTiffWriter,
        by default ".tif"

    Returns
    -------
    str
        Message of "saved" and the respective image name
    """

    def _format_filename(char_string):
        """convert strings to valid file names"""
        valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
        filename = "".join(char for char in char_string if char in valid_chars)
        filename = filename.replace(" ", "_")
        return filename

    def _save_path(folder_suffix, save_suffix_str):
        """Create save directories and return the path to save a file"""
        folder_name = str(output_folder_prefix + folder_suffix)
        save_directory = file_directory / folder_name
        save_directory.mkdir(parents=False, exist_ok=True)

        # image_name_base = os.path.splitext(os.path.basename(image.name))[0]
        # image_name = str(image_name_base + save_suffix_str)
        image_name = str(image.name + save_suffix_str)
        save_name = _format_filename(image_name)
        save_path = save_directory / save_name
        return save_path

    """save image"""
    img_path = _save_path("_images", save_suffix)
    # TO DO, make possible without aicsimageio, use KeyError for call
    img = image.metadata["aicsimage"]
    OmeTiffWriter.save(
        data=img.data,
        uri=img_path,
        dim_order=img.dims.order,
        channel_names=img.channel_names,
        physical_pixel_sizes=img.physical_pixel_sizes,
    )

    """save label"""
    lbl_path = _save_path("_labels", save_suffix)
    # lbl_dims = _get_img_dims(img)

    # if type(labels) is layers.Shapes:
    if annotation_type == "Shapes":
        lbl = shapes.to_labels(labels_shape=image.data.shape)
    # elif type(labels) is layers.Labels:
    elif annotation_type == "Labels":
        lbl = labels.data

    if len(np.squeeze(img.data).shape) > len(lbl.shape):
        lbl_dims = "".join(
            [
                d
                for d in img.dims.order
                if d != "C" and img.dims._dims_shape[d] > 1
            ]
        )
    elif len(np.squeeze(img.data).shape) == len(lbl.shape):
        lbl_dims = img.dims.order

    lbl = lbl.astype(np.int32)
    OmeTiffWriter.save(
        data=lbl,
        uri=lbl_path,
        dim_order=lbl_dims,
        channel_names=["Labels"],
        physical_pixel_sizes=img.physical_pixel_sizes,
    )

    return "Saved: " + os.path.splitext(os.path.basename(image.name))[0]
