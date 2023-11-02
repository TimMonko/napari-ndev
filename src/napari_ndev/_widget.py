"""
neural development (nDev) widget collection
"""
import os
import pathlib
from enum import Enum
from functools import reduce
from typing import TYPE_CHECKING

import apoc
import dask.array as da
import numpy as np
import pyclesperanto_prototype as cle
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
from magicgui import magic_factory
from magicgui.tqdm import tqdm
from napari_workflows._io_yaml_v1 import load_workflow

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

    dims = "".join(
        [d for d in img.dims.order if d != "C" and img.dims._dims_shape[d] > 1]
    )
    return dims


def init_utilities(batch_utilities):
    @batch_utilities.image_directory.changed.connect
    def _image_info():
        image_list = os.listdir(batch_utilities.image_directory.value)
        img = AICSImage(batch_utilities.image_directory.value / image_list[0])
        batch_utilities.channel_list.choices = img.channel_names
        batch_utilities.X_range.value = slice(0, img.dims.X, 1)
        batch_utilities.Y_range.value = slice(0, img.dims.Y, 1)
        batch_utilities.Z_range.value = slice(0, img.dims.Z, 1)


@magic_factory(
    widget_init=init_utilities,
    auto_call=False,
    call_button="Batch Adjust",
    image_directory=dict(widget_type="FileEdit", mode="d"),
    result_directory=dict(widget_type="FileEdit", mode="d"),
    channel_list=dict(widget_type="Select", choices=[]),
    keep_scenes=dict(widget_type="LineEdit"),
    project_combo=dict(
        widget_type="ComboBox", choices=["No Projection", "np.max", "np.sum"]
    ),
)
def batch_utilities(
    image_directory=pathlib.Path(),
    result_directory=pathlib.Path(),
    channel_list: str = [],
    keep_scenes: str = "",
    # project_bool: bool = False,
    project_combo: str = "No Projection",
    X_range: slice = slice(0, 1, 1),
    Y_range: slice = slice(0, 1, 1),
    Z_range: slice = slice(0, 1, 1),
):
    """Batch Utilities

    Quick adjustments to apply to a batch of images and save the resulting
        images in an output folder. Intended for adjustments either too simple
        (e.g. max projection, saving only certain channels) or not possible
        (e.g. cropping) with napari-workflows / batch-workflow widgets.

    Parameters
    ----------
    image_directory : pathlib.Path
        Directory of files to be processed.
    result_directory : pathlib.Path
        Directory to save output images to. Often created by user in OS GUI.
    channel_list : list
        Channels to process. Extracted from aicsimageio metadata.
    keep_scenes : str, optional
        Comma separated string of scene indexes or scene names.
    project_bool : bool
        If True, then do a maximum project along Z axis
    X_range, Y_range, Z_range : slice
        Dimension of respective axis to crop between. Use step to downsample.

    Returns
    -------
    None
        Processed images are saved in result directory
    """

    image_list = os.listdir(image_directory)

    img = AICSImage(image_directory / image_list[0])

    all_x, all_y, all_z = False, False, False
    if X_range == slice(0, img.dims.X, 1):
        all_x = True
    if Y_range == slice(0, img.dims.Y, 1):
        all_y = True
    if Z_range == slice(0, img.dims.Z, 1):
        all_z = True

    for file in tqdm(image_list, label="file"):
        result_stack = None
        img = AICSImage(image_directory / file)

        # Create kept scene list, if applicable. By using the else statement,
        # will work for single scene images
        if keep_scenes == "":
            scene_list = img.scenes
        else:
            scene_list = keep_scenes.split(",")
            try:  # for converting numeric keep_scenes to a list of indexes
                # convert to np array, subtract 1-index
                # (at least for ZEN/czi naming) and
                # convert back to python list
                scene_list = np.array(scene_list).astype("int") - 1
                scene_list = scene_list.tolist()
            except ValueError:
                pass

        for scene in tqdm(scene_list, label="scene"):
            img.set_scene(scene)

            for idx, channel in enumerate(channel_list):
                ch_img = _get_channel_image(
                    img=img, dims=img.dims.order, channel=channel
                )

                #  T C Z Y X default from aicsimageio
                if all_x is True:
                    X_range = slice(0, img.dims.X, 1)
                if all_y is True:
                    Y_range = slice(0, img.dims.Y, 1)
                if all_z is True:
                    Z_range = slice(0, img.dims.Z, 1)

                ch_img = ch_img[:, :, Z_range, Y_range, X_range]

                # project along the Z axis (2)
                if project_combo == "np.max":
                    ch_img = np.max(ch_img, axis=2, keepdims=True)
                if project_combo == "np.sum":
                    ch_img = np.sum(ch_img, axis=2, keepdims=True)

                # concatenate images, to keep proper dims stack along C (1)
                try:
                    result_stack = np.concatenate(
                        [result_stack, ch_img], axis=1
                    )
                except ValueError:
                    result_stack = ch_img

                if np.max(result_stack) > 65535:
                    result_stack = result_stack.astype(np.float32)

            # save the image
            file_stem = os.path.splitext(os.path.basename(file))[0]
            if len(scene_list) > 1:
                save_name = str(file_stem + "_scene_" + scene + ".tif")
            else:
                save_name = str(file_stem + ".tif")
            save_uri = result_directory / save_name

            OmeTiffWriter.save(
                data=result_stack,
                uri=save_uri,
                dim_order=img.dims.order,
                channel_names=channel_list,
                physical_pixel_sizes=img.physical_pixel_sizes,
            )

            result_stack = None
    return


def init_workflow(batch_workflow):
    @batch_workflow.image_directory.changed.connect
    def _image_info():
        image_list = os.listdir(batch_workflow.image_directory.value)
        img = AICSImage(batch_workflow.image_directory.value / image_list[0])

        img_dims = _get_img_dims(img)
        batch_workflow.img_dims.value = img_dims

        batch_workflow.root_0.choices = img.channel_names
        batch_workflow.root_1.choices = img.channel_names
        batch_workflow.root_2.choices = img.channel_names
        batch_workflow.root_3.choices = img.channel_names
        batch_workflow.root_4.choices = img.channel_names

    @batch_workflow.workflow_path.changed.connect
    def _workflow_info():
        wf = load_workflow(batch_workflow.workflow_path.value)
        batch_workflow.workflow_roots.value = wf.roots()


@magic_factory(
    widget_init=init_workflow,
    auto_call=False,
    call_button="Batch Workflow",
    result_widget=True,
    image_directory=dict(widget_type="FileEdit", mode="d"),
    result_directory=dict(widget_type="FileEdit", mode="d"),
    workflow_path=dict(widget_type="FileEdit", mode="r"),
    workflow_roots=dict(widget_type="Label", label="Roots:"),
    root_0=dict(widget_type="ComboBox", choices=[], nullable=True),
    root_1=dict(widget_type="ComboBox", choices=[], nullable=True),
    root_2=dict(widget_type="ComboBox", choices=[], nullable=True),
    root_3=dict(widget_type="ComboBox", choices=[], nullable=True),
    root_4=dict(widget_type="ComboBox", choices=[], nullable=True),
)
def batch_workflow(
    image_directory=pathlib.Path(),
    result_directory=pathlib.Path(),
    workflow_path=pathlib.Path(),
    workflow_roots: list = None,
    root_0: str = None,
    root_1: str = None,
    root_2: str = None,
    root_3: str = None,
    root_4: str = None,
    img_dims: str = None,
    keep_original_images: bool = True,
):
    """Batch Workflow widget using napari-workflow metadata file

    Load a napari-workflow metadata file and show the original roots.
    Select an image directory to populate the root dropdowns with
    channels from the first image read by aicsimageio. Then, select these
    channels in the dropdown to match the Roots in the proper order.
    If there are less roots than displayed, leave dropdown as
    '-----' which represents python None.

    Parameters
    ----------
    image_directory : pathlib.Path
        Location of image files to process, by default pathlib.Path()
    result_directory : pathlib.Path
        Location to save output images, by default pathlib.Path()
    workflow_path : pathlib.Path
        Location of workflow _metadata_.yaml file, by default pathlib.Path()
    workflow_roots : list, optional
        Roots extracted from , by default None
    root_0, root_1, root_2, root_3, root_4 : str, optional
        _description_, by default None
    img_dims : str, optional
        Can be changed, if necessary, to account for different image
        shapes by default None
    keep_original_images : bool, optional
        Stack original images with result images prior to saving,
        by default True

    Returns
    -------
    None
        Resulting image stacks are saved to result_directory after workflow
            processing
    """
    image_list = os.listdir(image_directory)

    wf = load_workflow(workflow_path)

    roots = []
    roots.append(root_0)
    roots.append(root_1)
    roots.append(root_2)
    roots.append(root_3)
    roots.append(root_4)

    root_list = [i for i in roots if i is not None]

    for file in tqdm(image_list, label="progress"):
        image_stack = []
        img = AICSImage(image_directory / file)

        """for each root selected in the root list, extract the channel image
        and set the workflow root names,
        this will be used later for workflow.get
        """
        for idx, root in enumerate(root_list):
            ch_img = _get_channel_image(img=img, dims=img_dims, channel=root)
            print(ch_img.shape)
            wf.set(name=wf.roots()[idx], func_or_data=ch_img)

            image_stack.append(ch_img)

        # dask_stack = da.stack(image_stack, axis=0)

        result = wf.get(name=wf.leafs())
        result_stack = cle.pull(result)  # adds a new dim at 0th axis, as "C"

        """extract the leaf name corresponding to each root to save into
        channel names
        """
        result_names = reduce(
            lambda res, lst: res + [lst[0] + "_" + lst[1]],
            zip(root_list, wf.leafs()),
            [],
        )

        if keep_original_images is True:
            dask_images = da.stack(image_stack, axis=0)
            dask_result = da.stack(result_stack, axis=0)
            result_stack = da.concatenate([dask_images, dask_result], axis=0)
            result_names = root_list + result_names


        file_stem = os.path.splitext(os.path.basename(file))[0]
        save_name = file_stem + ".tif"

        # print(result_stack.shape)

        save_uri = result_directory / save_name

        # Need to explicitly order CYX if dims are just XY else the stack
        # will be in the order of TZCYX but the saver won't know
        if img_dims == "YX":
            save_dim_order = "CYX"
        else:
            save_dim_order = None

        OmeTiffWriter.save(
            data=result_stack,
            uri=save_uri,
            dim_order=save_dim_order,  # this was previously commented out
            channel_names=result_names,
            physical_pixel_sizes=img.physical_pixel_sizes,
        )

    return


# Predefined feature sets extract from apoc and put in an Enum
PDFS = Enum("PDFS", apoc.PredefinedFeatureSet._member_names_)
cl_types = ["Pixel", "Object"]


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
    cl_directory=dict(widget_type="FileEdit", mode="d"),
    cl_type=dict(widget_type="RadioButtons", choices=cl_types),
    predefined_features=dict(widget_type="ComboBox", choices=PDFS),
    channel_list=dict(widget_type="Select", choices=[]),
    cl_label_id=dict(widget_type="SpinBox", min=1),
)
def batch_training(
    image_directory=pathlib.Path(),
    label_directory=pathlib.Path(),
    cl_directory=pathlib.Path(),
    cl_filename: str = "classifier.cl",
    cl_type: str = cl_types[0],
    cl_forests: int = 2,
    cl_trees: int = 100,
    predefined_features=PDFS(1),
    custom_features: str = None,
    channel_list: str = [],
    cl_label_id: int = 2,
    img_dims: str = None,
):
    """Train APOC (Accelerated-Pixel-Object-Classifiers) on a folder of
    images and labels.

    Parameters
    ----------
    image_directory : pathlib.Path
        Location of images
    label_directory : pathlib.Path
        Location of labels (annotations)
    cl_directory : pathlib.Path
        Location to save apoc classifier ".cl" file
    cl_filename : str, optional
        Filename to save apoc classifier, end with ".cl", by default
        "classifier.cl"
    cl_type : str, optional
        Choose between Pixel or Object Classifier, by default Pixel Classifier
    cl_forests : int, optional
        Number of random forests, by default 2
    cl_trees : int, optional
        Number of trees, by default 100
    predefined_features : str, optional
        Allows selection of `apoc.PredefinedFeatureSets`, by default custom,
        requires input of `custom_features`
    custom_features : str, optional
        Space-separated string of pyclesperanto filters, by default None
    channel_list : list
        Select channels to pass into the classifier to serve as bases for
        features, by default []
    cl_label_id : int, optional
        Label id number if using Object Classifier for `cl_type`,
        by default 2
    img_dims : str, optional
        Image dimensions read from aicsimageio metadata, by default None

    Returns
    -------
    str
        String of feature importances returned in classifier file

    Notes
    -----
    Accelerated pixel object classifier information from:
        https://github.com/haesleinhuepf/apoc
    Predefined features from
    https://github.com/haesleinhuepf/apoc/blob/main/demo/feature_stacks.ipynb
    """
    image_list = os.listdir(image_directory)
    label_list = os.listdir(label_directory)

    cl_path = str(cl_directory / cl_filename)

    apoc.erase_classifier(cl_path)

    if cl_type == "Pixel":
        custom_classifier = apoc.PixelClassifier(
            opencl_filename=cl_path,
            max_depth=cl_forests,
            num_ensembles=cl_trees,
        )

    if cl_type == "Object":
        custom_classifier = apoc.ObjectSegmenter(
            opencl_filename=cl_path,
            positive_class_identifier=cl_label_id,
            max_depth=cl_forests,
            num_ensembles=cl_trees,
        )

    # use an enumerate so that the index of the file can be used to extract
    # the proper label file, in case the image and label files do not have
    # matching names
    # This could be problematic if the images aren't sorted the same way,
    # but should generally be ok
    for idx, file in enumerate(tqdm(image_list, label="progress")):

        image_stack = []
        img = AICSImage(image_directory / file)

        for channels in channel_list:
            ch_img = _get_channel_image(
                img=img, dims=img_dims, channel=channels
            )
            image_stack.append(ch_img)

        dask_stack = da.stack(image_stack, axis=0)

        lbl = AICSImage(label_directory / label_list[idx])
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
    cl_path=dict(widget_type="FileEdit", mode="r"),
    cl_type=dict(widget_type="RadioButtons", choices=cl_types),
    channel_list=dict(widget_type="Select", choices=[]),
)
def batch_predict(
    image_directory=pathlib.Path(),
    result_directory=pathlib.Path(),
    cl_path=pathlib.Path(),
    cl_type: str = cl_types[0],
    channel_list: str = [],
    img_dims: str = None,
):
    """Predict APOC (Accelerated-Pixel-Object-Classifiers) on a folder of
    images and labels.

    Parameters
    ----------
    image_directory : pathlib.Path
        Location of images
    result_directory : pathlib.Path
        Location to save output labels
    cl_path : pathlib.Path
        Location of apoc classifier ".cl" file
    cl_type : str, optional
        Choose between Pixel or Object Classifier, by default Pixel Classifier
    channel_list : list
        Select channels to pass into the classifier to serve as bases for
        features, by default []
    cl_label_id : int, optional
        Label id number if using Object Classifier for `cl_type`, by default 2
    img_dims : str, optional
        Image dimensions read from aicsimageio metadata, by default None

    Returns
    -------
    None
        Predicted labels saved to `result_directory`

    Notes
    -----
    Accelerated pixel object classifier information from:
        https://github.com/haesleinhuepf/apoc
    """
    image_list = os.listdir(image_directory)

    if cl_type == "Pixel":
        custom_classifier = apoc.PixelClassifier(opencl_filename=cl_path)
    if cl_type == "Object":
        custom_classifier = apoc.ObjectSegmenter(opencl_filename=cl_path)

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
    return
