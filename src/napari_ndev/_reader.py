from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Union

from bioio import BioImage
from bioio_base.exceptions import UnsupportedFileFormatError

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import xarray as xr
    from napari.types import LayerData, PathLike, ReaderFunction

logger = logging.getLogger(__name__)


def reader_function(
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
    img = BioImage(path, reader=reader)

    if len(img.scenes) > 1 and not open_first_scene_only:
        _get_scenes(path=path, img=img, in_memory=in_memory)
        return [(None,)]

    img_data = _get_full_image_data(img, in_memory=in_memory)
    img_meta = _get_img_metadata(path, img_data, img)
    return [(img.data, img_meta, layer_type)]


def napari_get_reader(
    path: PathLike, 
    in_memory: bool | None = None
) -> ReaderFunction | None:
    """
    Get the appropriate reader function for a single given path.

    Parameters
    ----------
    path : PathLike
        Path to the file to be read
    in_memory : bool, optional
        Whether to read the file in memory, by default None

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
        return reader_function(path, reader, in_memory)
    except UnsupportedFileFormatError:
        logger.warning("Bioio: Unsupported file format")
        return None
    except Exception as e:
        logger.warning("Bioio: Error reading file")
        logger.warning(e)
        return None

def _get_full_image_data(
    img: BioImage,
    in_memory: bool | None = None
) -> xr.DataArray:
    if 'M' in img.reader.dims.order:
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

# def _get_meta(
#     path: PathLike,
#     img_data: xr.DataArray,
#     img: BioImage
# ) -> dict:
#     meta = {}
#     if 'C' in data.dims:
        
