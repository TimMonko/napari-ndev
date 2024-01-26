"""
neural development (nDev) widget collection
"""
import os
import pathlib
from functools import reduce
from typing import TYPE_CHECKING

import dask.array as da
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
