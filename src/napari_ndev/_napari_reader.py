from __future__ import annotations

import contextlib
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from bioio import BioImage
from bioio_base.dimensions import DimensionNames
from bioio_base.exceptions import UnsupportedFileFormatError
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

if TYPE_CHECKING:
    import xarray as xr

    from napari.types import LayerData, PathLike, ReaderFunction

logger = logging.getLogger(__name__)

SCENE_LABEL_DELIMITER = " :: "
BIOIO_CHOICES = "BioIO Scene Management"
CLEAR_LAYERS_ON_SELECT = "Clear All Layers on New Scene Selection"
UNPACK_CHANNELS_TO_LAYERS = "Unpack Channels as Layers"
DONT_MERGE_MOSAICS = "Don't Merge Mosaics"

def napari_get_reader(
    path: PathLike,
    in_memory: bool | None = None,
    open_first_scene_only: bool = False,
) -> ReaderFunction | None:
    """
    Get the appropriate reader function for a single given path.

    Parameters
    ----------
    path : PathLike
        Path to the file to be read
    in_memory : bool, optional
        Whether to read the file in memory, by default None
    open_first_scene_only : bool, optional
        Whether to ignore multi-scene files and just open the first scene,
        by default False


    Returns
    -------
    ReaderFunction
        The reader function for the given path

    """
    if isinstance(path, list):
        logger.info("Bioio: Expected a single path, got a list of paths.")
        return None

    try:
        plugin = BioImage.determine_plugin(path)
        reader = plugin.metadata.get_reader()
        # return napari_reader_function(path, reader, in_memory)
        return partial(
            napari_reader_function,
            reader=reader,
            in_memory=in_memory,
            open_first_scene_only=open_first_scene_only
        )
    except UnsupportedFileFormatError:
        logger.warning("Bioio: Unsupported file format")
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Bioio: Error reading file")
        logger.warning(e)
        return None

def napari_reader_function(
    path: PathLike,
    reader: Callable,
    in_memory: bool | None = None,
    open_first_scene_only: bool = False,
    layer_type: str = 'image'
) -> list[LayerData] | None:
    """
    Read a file using the given reader function.

    Parameters
    ----------
    path : PathLike
        Path to the file to be read
    reader : None
        Bioio Reader function to be used to read the file, by default None.
    in_memory : bool, optional
        Whether to read the file in memory, by default None.
    layer_type : str, optional
        Type of layer to be created in napari, by default 'image'.
    open_first_scene_only : bool, optional
        Whether to ignore multi-scene files and just open the first scene,
        by default False.

    Returns
    -------
    list
        List containing image data, metadata, and layer type

    """
    if isinstance(path, list):
        logger.info("Bioio: Expected a single path, got a list of paths.")
        return None

    in_memory = _determine_in_memory(path) if in_memory is None else in_memory
    logger.info('Bioio: Reading in-memory: %s', in_memory)

    img = BioImage(path, reader=reader)

    if len(img.scenes) > 1 and not open_first_scene_only:
        _get_scenes(path=path, img=img, in_memory=in_memory)
        return [(None,)]

    # TODO: why should I return the squeezed data and not the full data
    # is it because napari squeezes it anyway?
    img_data = _get_image_data(img, in_memory=in_memory)
    img_meta = _get_napari_metadata(path, img_data, img)

    return [(img_data.data, img_meta, layer_type)]

def _determine_in_memory(
    path: PathLike,
    max_mem_bytes: int = 4e9,
    max_mem_percent: int = 0.3
) -> bool:
    """Determine whether to read the file in memory."""
    from bioio_base.io import pathlike_to_fs
    from psutil import virtual_memory

    fs, path = pathlike_to_fs(path)
    filesize = fs.size(path)
    available_mem = virtual_memory().available
    return (
        filesize <= max_mem_bytes
        and filesize / available_mem <= max_mem_percent
    )

def _get_image_data(
    img: BioImage,
    in_memory: bool | None = None
) -> xr.DataArray:
    """
    Get the image data from the BioImage object.

    If the image has a mosaic, the data will be returned as a single
    xarray DataArray with the mosaic dimensions removed.

    Parameters
    ----------
    img : BioImage
        BioImage object to get the data from
    in_memory : bool, optional
        Whether to read the data in memory, by default None

    Returns
    -------
    xr.DataArray
        The image data as an xarray DataArray

    """
    if DimensionNames.MosaicTile in img.reader.dims.order:
        try:
            if in_memory:
                return img.reader.mosaic_xarray_data.squeeze()

            return img.reader.mosaic_xarray_dask_data.squeeze()

        except NotImplementedError:
            logger.warning(
                "Bioio: Mosaic tile switching not supported for this reader"
            )
            return None

    if in_memory:
        return img.reader.xarray_data.squeeze()

    return img.reader.xarray_dask_data.squeeze()

def _get_napari_metadata(
    path: PathLike,
    img_data: xr.DataArray,
    img: BioImage
) -> dict:
    """
    Get the metadata for the image.

    Parameters
    ----------
    path : PathLike
        Path to the file
    img_data : xr.DataArray
        Image data as an xarray DataArray
    img : BioImage
        BioImage object

    Returns
    -------
    dict
        Dictionary containing the image metadata for napari

    """
    meta = {}
    scene = img.current_scene
    scene_index = img.current_scene_index
    single_no_scene = len(img.scenes) == 1 and img.current_scene == "Image:0"
    channel_dim = DimensionNames.Channel

    if channel_dim in img_data.dims:
        # use filename if single scene and no scene name available
        if single_no_scene:
            channels_with_scene_index = [
                f'{Path(path).stem}{SCENE_LABEL_DELIMITER}{C}'
                for C in img_data.coords[channel_dim].data.tolist()
            ]
        else:
            channels_with_scene_index = [
                f'{scene_index}{SCENE_LABEL_DELIMITER}'
                f'{scene}{SCENE_LABEL_DELIMITER}{C}'
                for C in img_data.coords[channel_dim].data.tolist()
            ]
        meta['name'] = channels_with_scene_index
        meta['channel_axis'] = img_data.dims.index(channel_dim)

    # not multi-chnanel, use current scene as image name
    else:
        if single_no_scene:
            meta['name'] = Path(path).stem
        else:
            meta['name'] = img.reader.current_scene

    # Handle if RGB
    if DimensionNames.Samples in img.reader.dims.order:
        meta['rgb'] = True

    # Handle scales
    scale = [
        getattr(img.physical_pixel_sizes, dim)
        for dim in img_data.dims
        if dim in {DimensionNames.SpatialX, DimensionNames.SpatialY, DimensionNames.SpatialZ}
        and getattr(img.physical_pixel_sizes, dim) is not None
    ]

    if scale:
        meta['scale'] = tuple(scale)

    # get all other metadata
    img_meta = {'bioimage': img, 'raw_image_metadata': img.metadata}

    with contextlib.suppress(NotImplementedError):
        img_meta['metadata'] = img.ome_metadata

    meta['metadata'] = img_meta
    return meta

def _widget_is_checked(widget_name: str) -> bool:
    import napari

    # Get napari viewer from current process
    viewer = napari.current_viewer()

    # Get scene management widget
    scene_manager_choices_widget = viewer.window._dock_widgets[BIOIO_CHOICES]
    for child in scene_manager_choices_widget.widget().children():
        if isinstance(child, QCheckBox) and child.text() == widget_name:
                return child.isChecked()

    return False


# Function to handle multi-scene files.
def _get_scenes(path: PathLike, img: BioImage, in_memory: bool) -> None:
    import napari

    # Get napari viewer from current process
    viewer = napari.current_viewer()

    # Add a checkbox widget if not present
    if BIOIO_CHOICES not in viewer.window._dock_widgets:
        # Create a checkbox widget to set "Clear On Scene Select" or not
        scene_clear_checkbox = QCheckBox(CLEAR_LAYERS_ON_SELECT)
        scene_clear_checkbox.setChecked(False)

        # Create a checkbox widget to set "Unpack Channels" or not
        channel_unpack_checkbox = QCheckBox(UNPACK_CHANNELS_TO_LAYERS)
        channel_unpack_checkbox.setChecked(False)

        # Create a checkbox widget to set "Mosaic Merge" or not
        dont_merge_mosaics_checkbox = QCheckBox(DONT_MERGE_MOSAICS)
        dont_merge_mosaics_checkbox.setChecked(False)

        # Add all scene management state to a single box
        scene_manager_group = QGroupBox()
        scene_manager_group_layout = QVBoxLayout()
        scene_manager_group_layout.addWidget(scene_clear_checkbox)
        scene_manager_group_layout.addWidget(channel_unpack_checkbox)
        scene_manager_group_layout.addWidget(dont_merge_mosaics_checkbox)
        scene_manager_group.setLayout(scene_manager_group_layout)
        scene_manager_group.setFixedHeight(100)

        viewer.window.add_dock_widget(
            scene_manager_group,
            area="right",
            name=BIOIO_CHOICES,
        )

    # Create the list widget and populate with the ids & scenes in the file
    list_widget = QListWidget()
    for i, scene in enumerate(img.scenes):
        list_widget.addItem(f"{i}{SCENE_LABEL_DELIMITER}{scene}")

    # Add this files scenes widget to viewer
    viewer.window.add_dock_widget(
        list_widget,
        area="right",
        name=f"{Path(path).name}{SCENE_LABEL_DELIMITER}Scenes",
    )

    # Function to create image layer from a scene selected in the list widget
    def open_scene(item: QListWidgetItem) -> None:
        scene_text = item.text()

        # Use scene indexes to cover for duplicate names
        scene_index = int(scene_text.split(SCENE_LABEL_DELIMITER)[0])

        # Update scene on image and get data
        img.set_scene(scene_index)
        # check whether to mosaic merge or not
        if _widget_is_checked(DONT_MERGE_MOSAICS):
            data = _get_image_data(
                img=img, in_memory=in_memory, reconstruct_mosaic=False
            )
        else:
            data = _get_image_data(img=img, in_memory=in_memory)

        # Get metadata and add to image
        meta = _get_napari_metadata("", data, img)

        # Optionally clear layers
        if _widget_is_checked(CLEAR_LAYERS_ON_SELECT):
            viewer.layers.clear()

        # Optionally remove channel axis
        if not _widget_is_checked(UNPACK_CHANNELS_TO_LAYERS):
            meta["name"] = scene_text
            meta.pop("channel_axis", None)

        viewer.add_image(data.data, **meta)

    list_widget.currentItemChanged.connect(open_scene)  # type: ignore
