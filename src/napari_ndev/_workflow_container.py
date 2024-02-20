from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import dask.array as da
import numpy as np
import pyclesperanto_prototype as cle
import xarray as xr
from aicsimageio import AICSImage, transforms
from aicsimageio.writers import OmeTiffWriter
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    Label,
    ProgressBar,
    PushButton,
)
from napari_workflows._io_yaml_v1 import load_workflow

from napari_ndev.helpers import get_directory_and_files, setup_logger

if TYPE_CHECKING:
    import napari

PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, da.Array]
MetaArrayLike = Union[ArrayLike, xr.DataArray]
ImageLike = Union[
    PathLike, ArrayLike, MetaArrayLike, List[MetaArrayLike], List[PathLike]
]


class WorkflowContainer(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        ##############################
        # Attributes
        ##############################
        self
        self.viewer = viewer
        self.roots = []
        self._channel_names = []
        self._img_dims = ""

        ##############################
        # Widgets
        ##############################
        self.image_directory = FileEdit(label="Image Directory", mode="d")
        self.result_directory = FileEdit(label="Result Directory", mode="d")

        self.workflow_file = FileEdit(
            label="Workflow File",
            filter="*.yaml",
            tooltip="Select a workflow file to load",
        )
        self._keep_original_images = CheckBox(
            label="Keep Original Images",
            value=False,
        )
        self.batch_button = PushButton(label="Batch Workflow")

        self._progress_bar = ProgressBar(label="Progress:")
        self._workflow_roots = Label(label="Workflow Roots:")

        self.extend(
            [
                self.image_directory,
                self.result_directory,
                self.workflow_file,
                self._keep_original_images,
                self.batch_button,
                self._progress_bar,
                self._workflow_roots,
            ]
        )
        ##############################
        # Event Handling
        ##############################
        self.image_directory.changed.connect(self._get_image_info)
        # <- currently this is not triggering the update of the roots
        self.viewer.layers.events.inserted.connect(self._update_root_choices)
        self.viewer.layers.events.removed.connect(self._update_root_choices)
        self.workflow_file.changed.connect(self._get_workflow_info)
        self.batch_button.clicked.connect(self.batch_workflow)

    # Get Channel names and image dimensions without C
    def _get_image_info(self):
        self.image_dir, self.image_files = get_directory_and_files(
            self.image_directory.value
        )
        img = AICSImage(self.image_files[0])
        self._channel_names = img.channel_names
        self._update_root_choices()

        self._img_dims = "".join(
            [
                dim
                for dim in img.dims.order
                if dim != "C" and img.dims._dims_shape[dim] > 1
            ]
        )
        return self._img_dims

    def _update_root_choices(self):
        for root in self.roots:
            root.choices = self._channel_names + self.viewer.layers

    def _update_roots(self):
        for root in self.roots:
            self.remove(root)
        self.roots.clear()

        for idx, root in enumerate(self.workflow.roots()):
            root = ComboBox(
                label=f"Root {idx}: {root}",
                choices=self._channel_names,
                nullable=True,
                value=None,
            )
            self.roots.append(root)
            self.append(root)
        return

    def _get_workflow_info(self):
        self.workflow = load_workflow(self.workflow_file.value)
        self._workflow_roots.value = self.workflow.roots()
        # self._update_roots(len(self.workflow.roots()))
        self._update_roots()
        return

    def batch_workflow(self):
        result_dir = self.result_directory.value
        image_files = self.image_files
        # image_dir = self.image_dir
        # image_dims = self._img_dims
        workflow = self.workflow

        root_list = [root.value for root in self.roots]
        roots = [root for root in root_list if root is not None]

        img = AICSImage(image_files[0])
        root_index_list = [img.channel_names.index(root) for root in roots]

        # Setting up Logging File
        log_loc = result_dir / "workflow.log.txt"
        logger, handler = setup_logger(log_loc)
        logger.info(
            f"""
        Image Directory: {self.image_directory.value}
        Result Directory: {result_dir}
        Workflow File: {self.workflow_file.value}
        Roots: {roots}
        """
        )

        self._progress_bar.label = f"Workflow on {len(image_files)} images"
        self._progress_bar.value = 0
        self._progress_bar.max = len(image_files)

        for idx_file, image_file in enumerate(image_files):
            logger.info(f"Processing {idx_file+1}: {image_file.name}")
            img = AICSImage(image_file)

            # get indexes of channel names, in case not all images have
            # the same channel names, the index should be in the same order
            root_stack = []
            for idx, root_index in enumerate(root_index_list):
                # <- RGB prep here
                root_img = img.get_image_data("TCZYX", C=root_index)
                # squeeze the root image for workflow bc workflow expects
                root_squeeze = np.squeeze(root_img)
                # set the root image to the index of the root in the workflow
                workflow.set(
                    name=workflow.roots()[idx], func_or_data=root_squeeze
                )
                root_stack.append(root_img)  # stacks the TCZYX images

            leaf_names = workflow.leafs()
            result = workflow.get(name=leaf_names)

            result_stack = cle.pull(
                result
            )  # cle.pull stacks the results on the 0th axis as "C"
            # transform result_stack to TCZYX
            result_stack = transforms.reshape_data(
                data=result_stack,
                given_dims="C" + self._img_dims,
                return_dims="TCZYX",
            )

            # <- should I add a check for the result_stack to be a dask array?
            # <- should this be done using dask or numpy?
            if self._keep_original_images.value:
                dask_images = da.concatenate(root_stack, axis=1)  # along "C"
                result_stack = da.concatenate(
                    [dask_images, result_stack], axis=1
                )
                result_names = root_list + leaf_names
            else:
                result_names = leaf_names

            OmeTiffWriter.save(
                data=result_stack,
                uri=result_dir / (image_file.stem + ".tif"),
                dim_order="TCZYX",
                channel_names=result_names,
                physical_pixel_sizes=img.physical_pixel_sizes,
            )

            self._progress_bar.value = idx_file + 1

        logger.removeHandler(handler)
        return

    # Add single workflow here:
