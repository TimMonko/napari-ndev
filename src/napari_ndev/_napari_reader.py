from __future__ import annotations

import importlib
import logging
from functools import partial
from typing import TYPE_CHECKING, Callable

from bioio_base.exceptions import UnsupportedFileFormatError
from magicgui.widgets import Container, Select

from napari_ndev import get_settings, nImage

if TYPE_CHECKING:
    import napari
    from napari.types import LayerData, PathLike, ReaderFunction

logger = logging.getLogger(__name__)

DELIMITER = " :: "

def napari_get_reader(
    path: PathLike,
    in_memory: bool | None = None,
    open_first_scene_only: bool | None = None,
    open_all_scenes: bool | None = None,
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
    open_all_scenes : bool, optional
        Whether to open all scenes in a multi-scene file, by default False
        Ignored if open_first_scene_only is True


    Returns
    -------
    ReaderFunction
        The reader function for the given path

    """
    settings = get_settings()
    if open_first_scene_only is None:
        open_first_scene_only = settings.SCENE_HANDLING == "View First Scene Only"
    if open_all_scenes is None:
        open_all_scenes = settings.SCENE_HANDLING == "View All Scenes"

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
            open_first_scene_only=open_first_scene_only,
            open_all_scenes=open_all_scenes,
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
    open_all_scenes: bool = False,
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
    open_all_scenes : bool, optional
        Whether to open all scenes in a multi-scene file, by default False.
        Ignored if open_first_scene_only is True.

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
    # TODO: Guess layer type here (check channel names for labels?)
    logger.info('Bioio: Reading in-memory: %s', in_memory)


    # open first scene only
    if len(img.scenes) == 1 or open_first_scene_only:
        img_data = img.get_napari_image_data(in_memory=in_memory)
        img_meta = img.get_napari_metadata(path)
        return [(img_data.data, img_meta, layer_type)]

    # TODO: USE settings for open first or all scenes to set the nubmer of iterations of a for loop
    # check napari reader settings stuff
    # open all scenes as layers
    if len(img.scenes) > 1 and open_all_scenes:
        layer_list = []
        for scene in img.scenes:
            img.set_scene(scene)
            img_data = img.get_napari_image_data(in_memory=in_memory)
            img_meta = img.get_napari_metadata(path)
            layer_list.append((img_data.data, img_meta, layer_type))
        return layer_list

    # open scene widget
    if len(img.scenes) > 1 and not open_all_scenes:
        _open_scene_container(path=path, img=img, in_memory=in_memory)
        return [(None,)]

    logger.warning("Bioio: Error reading file")
    return [(None,)]

def _open_scene_container(path: PathLike, img: nImage, in_memory: bool) -> None:
    from pathlib import Path

    import napari

    viewer = napari.current_viewer()
    viewer.window.add_dock_widget(
        nImageSceneWidget(viewer, path, img, in_memory),
        area='right',
        name=f'{Path(path).stem}{DELIMITER}Scenes',
    )
class nImageSceneWidget(Container):
    """
    Widget to select a scene from a multi-scene file.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.
    path : PathLike
        Path to the file.
    img : nImage
        The nImage instance.
    in_memory : bool
        Whether the image should be added in memory.

    Attributes
    ----------
    viewer : napari.viewer.Viewer
        The napari viewer instance.
    path : PathLike
        Path to the file.
    img : nImage
        The nImage instance.
    in_memory : bool
        Whether the image should be added in memory.
    settings : Settings
        The settings instance.
    scenes : list
        List of scenes in the image.
    _scene_list_widget : magicgui.widgets.Select
        Widget to select a scene from a multi-scene file.

    Methods
    -------
    open_scene
        Opens the selected scene(s) in the viewer.

    """

    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        path: PathLike,
        img: nImage,
        in_memory: bool,
    ):
        """
        Initialize the nImageSceneWidget.

        Parameters
        ----------
        viewer : napari.viewer.Viewer
            The napari viewer instance.
        path : PathLike
            Path to the file.
        img : nImage
            The nImage instance.
        in_memory : bool
            Whether the image should be added in memory.

        """
        super().__init__(labels=False)
        self.max_height = 200
        self.viewer = viewer
        self.path = path
        self.img = img
        self.in_memory = in_memory
        self.settings = get_settings()
        self.scenes = [
            f'{idx}{DELIMITER}{scene}'
            for idx, scene in enumerate(self.img.scenes)
        ]

        self._init_widgets()
        self._connect_events()

    def _init_widgets(self):

        self._scene_list_widget = Select(
            value = None,
            nullable = True,
            choices = self.scenes,
        )
        self.append(self._scene_list_widget)

    def _connect_events(self):
        self._scene_list_widget.changed.connect(self.open_scene)

    def open_scene(self) -> None:
        """Open the selected scene(s) in the viewer."""
        if self.settings.CLEAR_LAYERS_ON_NEW_SCENE:
            self.viewer.layers.clear()

        for scene in self._scene_list_widget.value:
            if scene is None:
                continue
            # Use scene indexes to cover for duplicate names
            scene_index = int(scene.split(DELIMITER)[0])
            self.img.set_scene(scene_index)
            img_data = self.img.get_napari_image_data(in_memory=self.in_memory)
            img_meta = self.img.get_napari_metadata()

            self.viewer.add_image(img_data.data, **img_meta)
