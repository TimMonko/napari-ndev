"""
neural development (nDev) widget collection
"""
import pathlib
import string
from typing import TYPE_CHECKING

from aicsimageio import AICSImage
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
    save_suffix=dict(widget_type="LineEdit", label="Save Suffix"),
)
def batch_annotator(
    labels: layers.Labels,
    image: layers.Image,
    file_directory=pathlib.Path(),
    output_folder_prefix="Annotated",
    save_suffix=".ome.tiff",
):
    """Batch Annotation

    Used for annotating images and saving images of interest into a folder
    for the image and a folder for the labels. The GUI allows selecting of the
    intended label layer and image layer, as well as the prefix for the output
    folders. This output folder can already exist: mkdir(exist_ok=True), so
    should be used to save multiple images within the same folder group. The
    images are saved as .tif files though this can be adjusted in the saver
    function, and could be added as a GUI element.

    napari.layers. is event connected, so labels and image selection will
    update as new labels and images are added.
    """

    def format_filename(char_string):
        """convert strings to valid file names"""
        valid_chars = f"-_.() {string.ascii_letters}{string.digits}"
        filename = "".join(char for char in char_string if char in valid_chars)
        filename = filename.replace(" ", "_")
        return filename

    def saver(image_type, folder_suffix, save_suffix_str):
        """selects layers to save into a specified folder
        uses aicsimageio to save, because this has better control over metadata
        """
        folder_name = str(output_folder_prefix + folder_suffix)
        save_directory = file_directory / folder_name
        save_directory.mkdir(parents=False, exist_ok=True)

        image_name = str(image.name + save_suffix_str)
        save_name = format_filename(image_name)
        save_path = save_directory / save_name

        """use the current scene of the original aicsimage object, unless
        object does not exist (such as label layer), then just use layer data
        """
        try:
            img = image_type.metadata["aicsimage"].xarray_data
        except KeyError:
            img = image_type.data
        AICSImage(img).save(uri=save_path)
        # image_type.save(path=str(save_path)) #napari layer save
        # If naive defaults of AICSImage.save are insufficient:
        # OmeTiffWriter.save(data = image_type.data, uri = str(save_path))

    saver(image, "_images", save_suffix)
    saver(labels, "_labels", save_suffix)
    return "Saved Successfully"
