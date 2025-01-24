from __future__ import annotations

import importlib
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from bioio_base.exceptions import UnsupportedFileFormatError
from qtpy.QtWidgets import (
    QListWidget,
    QListWidgetItem,
)

from napari_ndev import get_settings, nImage

if TYPE_CHECKING:

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
    settings = get_settings()
    open_first_scene_only = settings.SCENE_HANDLING == "View First Scene Only"

    if isinstance(path, list):
        logger.info("Bioio: Expected a single path, got a list of paths.")
        return None

    try:
        # TODO: Test this if else functionality.
        from bioio import plugin_feasibility_report as pfr
        fr = pfr(path)
        if settings.PREFERRED_READER in fr and fr[settings.PREFERRED_READER].supported:
            reader_module = importlib.import_module(
                settings.PREFERRED_READER.replace('-', '_')
            )
            reader = reader_module.Reader
        else:
            plugin = nImage.determine_plugin(path)
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

    img = nImage(path, reader=reader)
    in_memory = img._determine_in_memory(path) if in_memory is None else in_memory
    logger.info('Bioio: Reading in-memory: %s', in_memory)

    if len(img.scenes) > 1 and not open_first_scene_only:
        _get_scenes(path=path, img=img, in_memory=in_memory)
        return [(None,)]

    # TODO: why should I return the squeezed data and not the full data
    # is it because napari squeezes it anyway?
    img_data = img.get_napari_image_data(in_memory=in_memory)
    img_meta = img.get_napari_metadata(path)

    return [(img_data.data, img_meta, layer_type)]

# Function to handle multi-scene files.
def _get_scenes(path: PathLike, img: nImage, in_memory: bool) -> None:
    import napari

    # Get napari viewer from current process
    viewer = napari.current_viewer()
    settings = get_settings()

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
        # if _widget_is_checked(DONT_MERGE_MOSAICS):
        #     data = img.get_napari_image_data(in_memory=in_memory, reconstruct_mosaic=False)
        # else:
        data = img.get_napari_image_data(in_memory=in_memory)

        # Get metadata and add to image
        meta = img.get_napari_metadata("")

        # Optionally clear layers
        if settings.CLEAR_LAYERS_ON_NEW_SCENE:
            viewer.layers.clear()

        # Optionally remove channel axis
        if not settings.UNPACK_CHANNELS_AS_LAYERS:
            # If not unpacking channels, remove channel axis from metadata
            meta["name"] = scene_text
            meta.pop("channel_axis", None)

        viewer.add_image(data.data, **meta)

    list_widget.currentItemChanged.connect(open_scene)  # type: ignore
