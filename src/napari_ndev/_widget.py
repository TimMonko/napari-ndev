"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import pathlib
from typing import TYPE_CHECKING

from magicgui import magic_factory
from napari import layers

if TYPE_CHECKING:
    pass


@magic_factory(
    auto_call=False,
    result_widget=True,
    call_button="Create Output Folders",
    file_directory=dict(
        widget_type="FileEdit", mode="d", label="File Directory"
    ),
    output_folder_prefix=dict(widget_type="LineEdit", label="Output Folder"),
)
def batch_annotator(
    labels: layers.Labels,
    image: layers.Image,
    file_directory=pathlib.Path(),
    output_folder_prefix="Annotated",
):
    def saver(image_type, folder_suffix, save_suffix=".tif"):
        save_folder = file_directory / str(
            output_folder_prefix + folder_suffix
        )
        save_folder.mkdir(parents=False, exist_ok=True)
        save_path = save_folder / str(image.name + save_suffix)
        image_type.save(path=str(save_path))

    saver(image, "_images", ".tif")
    saver(labels, "_labels", ".tif")
    return
