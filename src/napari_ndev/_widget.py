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
import napari
import numpy as np
import pyclesperanto_prototype as cle
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
        if type(image_type) == napari.layers.labels.labels.Labels:
            img = img.astype(np.int32)
        AICSImage(img).save(uri=save_path)

        # image_type.save(path=str(save_path)) #napari layer save
        # If naive defaults of AICSImage.save are insufficient:
        # OmeTiffWriter.save(data = image_type.data, uri = str(save_path))

    saver(image, "_images", save_suffix)
    saver(labels, "_labels", save_suffix)
    return "Saved Successfully"


channel_nums = [0, 1, 2, 3, 4]
PDFS = Enum("PDFS", apoc.PredefinedFeatureSet._member_names_)


@magic_factory(
    auto_call=False,
    call_button="Batch Train",
    result_widget=True,
    image_directory=dict(widget_type="FileEdit", mode="d"),
    label_directory=dict(widget_type="FileEdit", mode="d"),
    predefined_features=dict(widget_type="ComboBox", choices=PDFS),
    channel_list=dict(widget_type="Select", choices=channel_nums),
)
def batch_training(
    image_directory=pathlib.Path(),
    label_directory=pathlib.Path(),
    cl_filename: str = "classifier.cl",
    predefined_features=PDFS(1),
    custom_features: str = None,
    channel_list: int = 0,
    img_dims: str = "TYX",
    label_dims: str = "ZYX",
):
    image_list = os.listdir(image_directory)

    apoc.erase_classifier(cl_filename)
    custom_classifier = apoc.PixelClassifier(opencl_filename=cl_filename)

    for file in image_list:

        image_stack = []
        img = AICSImage(image_directory / file)

        def channel_image(img, dims: str, channel: str or int):
            if isinstance(channel, str):
                channel_index = img.channel_names.index(channel)
            elif isinstance(channel, int):
                channel_index = channel
            channel_img = img.get_image_data(dims, C=channel_index)
            return channel_img

        for channels in channel_list:
            ch_img = channel_image(img=img, dims=img_dims, channel=channels)
            image_stack.append(ch_img)

        dask_stack = da.stack(image_stack, axis=0)

        lbl = AICSImage(label_directory / file)
        labels = channel_image(img=lbl, dims=label_dims, channel=0)

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


@magic_factory(
    auto_call=False,
    call_button="Batch Predict",
    image_directory=dict(widget_type="FileEdit", mode="d"),
    result_directory=dict(widget_type="FileEdit", mode="d"),
    classifier_path=dict(widget_type="FileEdit", mode="r"),
    channel_list=dict(widget_type="Select", choices=channel_nums),
)
def batch_predict(
    image_directory=pathlib.Path(),
    result_directory=pathlib.Path(),
    classifier_path=pathlib.Path(),
    channel_list: int = 0,
    img_dims: str = "TYX",
):
    image_list = os.listdir(image_directory)
    custom_classifier = apoc.PixelClassifier(opencl_filename=classifier_path)

    for file in image_list:
        # print('started predicting: ', file)
        image_stack = []
        img = AICSImage(image_directory / file)

        def channel_image(img, dims: str, channel: str or int):
            if isinstance(channel, str):
                channel_index = img.channel_names.index(channel)
            elif isinstance(channel, int):
                channel_index = channel
            channel_img = img.get_image_data(dims, C=channel_index)
            return channel_img

        for channels in channel_list:
            ch_img = channel_image(img=img, dims=img_dims, channel=channels)
            image_stack.append(ch_img)

        dask_stack = da.stack(image_stack, axis=0)

        result = custom_classifier.predict(
            image=dask_stack,
        )

        AICSImage(cle.pull(result)).save(uri=result_directory / file)

    return result
