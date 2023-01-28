"""
neural development (nDev) widget collection
"""
import pathlib
import string
from typing import TYPE_CHECKING

from magicgui import magic_factory
from napari import layers

if TYPE_CHECKING:
    pass


@magic_factory(
    auto_call=False,
    result_widget=True,
    call_button="Save Layers to Output Folders",
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
    """Batch Annotation

    Used for annotating images and saving images of interest into a folder
      for the image and a folder for the labels.
    The GUI allows selecting of the intended label layer and image layer,
      as well as the prefix for the output folders.
    This output folder can already exist: mkdir(exist_ok=True),
      so should be used to save multiple images within the same folder group.
    The images are saved as .tif files though this can be adjusted in
      the saver function, and could be added as a GUI element.
    napari.layers. is event connected, so labels and image selection
      will update as new labels and images are added.
    """

    def saver(image_type, folder_suffix, save_suffix=".tif"):
        save_folder = file_directory / str(
            output_folder_prefix + folder_suffix
        )
        save_folder.mkdir(parents=False, exist_ok=True)

        # convert AICSImageIO (and others, potentially)
        # tile opened image layer names.
        # Otherwise these are invalid file names
        def format_filename(char_string):
            valid_chars = "-_.() {}{}".format(
                string.ascii_letters, string.digits
            )  # valid character list
            filename = "".join(
                char for char in char_string if char in valid_chars
            )  # join only valid characters from input string
            filename = filename.replace(
                " ", "_"
            )  # replace spaces with underscores
            return filename

        save_name = format_filename(
            str(image.name + folder_suffix + save_suffix)
        )
        save_path = save_folder / save_name  # path-like
        image_type.save(path=str(save_path))

    saver(image, "_images", ".tif")
    saver(labels, "_labels", ".tif")
    return "success"
