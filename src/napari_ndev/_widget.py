"""
neural development (nDev) widget collection
"""
import os
import pathlib
import string
from enum import Enum
from typing import TYPE_CHECKING

import apoc
import dask.array as da
import numpy as np
import pyclesperanto_prototype as cle
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from magicgui import magic_factory
from magicgui.tqdm import tqdm
from napari import layers

if TYPE_CHECKING:
    pass


def _get_channel_image(img, dims: str, channel: str or int):
    """From an AICSImage object (img), get image data for a particular
    channel whether from the channels name (str) or the channel's index
    (int)

    The index method is useful for labels layer, which have no name but
    do have index = 0
    """
    if isinstance(channel, str):
        channel_index = img.channel_names.index(channel)
    elif isinstance(channel, int):
        channel_index = channel
    channel_img = img.get_image_data(dims, C=channel_index)
    return channel_img


def _get_img_dims(img):
    """Extracts actual dimenions, except for C (color) of an AICSImage
    object (img), that are greater than 1. Ignores C because C is split
    into layers in napari and also because Labels layers have no C.

    Especially useful for saving the dim order of Label layers as
    relevant to the original matching AICSImage file. Because data.shape
    is effectively squeezed by napari, this can recapture the actual
    dims of the original image and make both the image and label layers
    comparable.
    """
    endstring = ""
    for d in img.dims.order:
        if d == "C":
            continue
        if img.dims._dims_shape[d] > 1:
            endstring = endstring + d
    return endstring


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
    image: layers.Image,
    labels: layers.Labels,
    file_directory=pathlib.Path(),
    output_folder_prefix="Annotated",
    save_suffix=".ome.tif",
):
    """Batch Annotation

    Used for annotating images and saving images of interest into a
    folder for the image and a folder for the labels. The GUI allows
    selecting of the intended label layer and image layer, as well as
    the prefix for the output folders. This output folder can already
    exist: mkdir(exist_ok=True), so should be used to save multiple
    images within the same folder group. The images are saved as .tif
    files though this can be adjusted in the saver function, and could
    be added as a GUI element.

    napari.layers. is event connected, so labels and image selection
    will update as new labels and images are added.
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

        image_name = str(image.name + save_suffix_str)
        save_name = _format_filename(image_name)
        save_path = save_directory / save_name
        return save_path

    """save image"""
    img_path = _save_path("_images", save_suffix)
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
    lbl_dims = _get_img_dims(img)
    lbl = labels.data
    lbl = lbl.astype(np.int32)
    OmeTiffWriter.save(
        data=lbl,
        uri=lbl_path,
        dim_order=lbl_dims,
        channel_names=["Labels"],
        physical_pixel_sizes=img.physical_pixel_sizes,
    )

    return "Saved Successfully"


# Predefined feature sets extract from apoc and put in an Enum
PDFS = Enum("PDFS", apoc.PredefinedFeatureSet._member_names_)


def init_training(batch_training):
    @batch_training.image_directory.changed.connect
    def _image_info():
        image_list = os.listdir(batch_training.image_directory.value)
        img = AICSImage(batch_training.image_directory.value / image_list[0])

        img_dims = _get_img_dims(img)
        batch_training.img_dims.value = img_dims
        batch_training.channel_list.choices = img.channel_names


@magic_factory(
    widget_init=init_training,
    auto_call=False,
    call_button="Batch Train",
    result_widget=True,
    image_directory=dict(widget_type="FileEdit", mode="d"),
    label_directory=dict(widget_type="FileEdit", mode="d"),
    predefined_features=dict(widget_type="ComboBox", choices=PDFS),
    channel_list=dict(widget_type="Select", choices=[]),
)
def batch_training(
    image_directory=pathlib.Path(),
    label_directory=pathlib.Path(),
    cl_filename: str = "classifier.cl",
    predefined_features=PDFS(1),
    custom_features: str = None,
    channel_list: str = [],
    img_dims: str = None,
):
    """Batch APOC Training

    Train APOC (Accelerated-Pixel-Object-Classifiers) on a folder of
    images and labels. See documentation here:
    https://github.com/haesleinhuepf/apoc

    Predefined features allow selection of apoc.PredefinedFeatureSets
    https://github.com/haesleinhuepf/apoc/blob/main/demo/feature_stacks.ipynb

    Creates the classifier.cl file in your current directory, which is
    usually where you launch python from.
    """
    image_list = os.listdir(image_directory)

    apoc.erase_classifier(cl_filename)
    custom_classifier = apoc.PixelClassifier(opencl_filename=cl_filename)

    for file in tqdm(image_list, label="progress"):

        image_stack = []
        img = AICSImage(image_directory / file)

        for channels in channel_list:
            ch_img = _get_channel_image(
                img=img, dims=img_dims, channel=channels
            )
            image_stack.append(ch_img)

        dask_stack = da.stack(image_stack, axis=0)

        lbl = AICSImage(label_directory / file)
        labels = _get_channel_image(img=lbl, dims=img_dims, channel=0)

        if predefined_features.value == 1:
            print("custom")
            feature_set = custom_features

        else:
            print("predefined")
            feature_set = apoc.PredefinedFeatureSet[
                predefined_features.name
            ].value

        custom_classifier.train(
            features=feature_set,
            image=dask_stack,
            ground_truth=labels,
            continue_training=True,
        )

    feature_importances = custom_classifier.feature_importances()
    print("success")
    # return pd.Series(feature_importances).plot.bar()
    return feature_importances


def init_predict(batch_predict):
    @batch_predict.image_directory.changed.connect
    def _image_info():
        image_list = os.listdir(batch_predict.image_directory.value)
        img = AICSImage(batch_predict.image_directory.value / image_list[0])

        img_dims = _get_img_dims(img)
        batch_predict.img_dims.value = img_dims
        batch_predict.channel_list.choices = img.channel_names


@magic_factory(
    widget_init=init_predict,
    auto_call=False,
    call_button="Batch Predict",
    image_directory=dict(widget_type="FileEdit", mode="d"),
    result_directory=dict(widget_type="FileEdit", mode="d"),
    classifier_path=dict(widget_type="FileEdit", mode="r"),
    channel_list=dict(widget_type="Select", choices=[]),
)
def batch_predict(
    image_directory=pathlib.Path(),
    result_directory=pathlib.Path(),
    classifier_path=pathlib.Path(),
    channel_list: str = [],
    img_dims: str = None,
):
    """Batch APOC Predict

    Use any APOC (Accelerated-Pixel-Object-Classifiers)-trained
    classifier on a folder of images. See documentation here:
    https://github.com/haesleinhuepf/apoc

    Produces an output folder with results label images.
    """
    image_list = os.listdir(image_directory)
    custom_classifier = apoc.PixelClassifier(opencl_filename=classifier_path)

    for file in tqdm(image_list, label="progress"):
        image_stack = []
        img = AICSImage(image_directory / file)

        for channels in channel_list:
            ch_img = _get_channel_image(
                img=img, dims=img_dims, channel=channels
            )
            image_stack.append(ch_img)

        dask_stack = da.stack(image_stack, axis=0)
        result = custom_classifier.predict(
            image=dask_stack,
        )

        lbl = cle.pull(result)
        lbl = lbl.astype(np.int32)
        OmeTiffWriter.save(
            data=lbl,
            uri=result_directory / file,
            dim_order=img_dims,
            channel_names=["Labels"],
            physical_pixel_sizes=img.physical_pixel_sizes,
        )
        # AICSImage(cle.pull(result)).save(uri=result_directory / file)

    return result
