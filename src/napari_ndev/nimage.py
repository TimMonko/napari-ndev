"""Additional functionality for BioImage objects to be used in napari-ndev."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import xarray as xr
from bioio import BioImage
from bioio_base.dimensions import DimensionNames
from bioio_base.reader import Reader
from bioio_base.types import ImageLike

from napari.types import PathLike

logger = logging.getLogger(__name__)

LABEL_DELIMITER = " :: "

class nImage(BioImage):
    """
    An nImage is a BioImage with additional functionality for napari-ndev.

    Parameters
    ----------
    image : ImageLike
        Image to be loaded.
    reader : Reader, optional
        Reader to be used to load the image. If not provided, a reader will be
        determined based on the image type.

    Attributes
    ----------
    See BioImage for inherited attributes.

    Methods
    -------
    get_napari_image_data(in_memory=None)
        Get the image data as a xarray, optionally loading it into memory.


    """

    def __init__(
        self,
        image: ImageLike,
        reader: Reader | None = None
    ) -> None:
        """Initialize an nImage with an image, and optionally a reader."""
        super().__init__(image, reader)
        self.napari_data = None
        self.napari_metadata = {}
        self.path = image if isinstance(image, (str, Path)) else None

    def _determine_in_memory(
        self,
        path=None,
        max_in_mem_bytes: int = 4e9,
        max_in_mem_percent: int = 0.3
    ) -> bool:
        """
        Determine whether the image should be loaded into memory or not.

        If the image is smaller than the maximum filesize or percentage of the
        available memory, this will determine to load image in memory.
        Otherwise, suggest to load as a dask array.

        Parameters
        ----------
        path : str or Path
            Path to the image file.
        max_in_mem_bytes : int
            Maximum number of bytes that can be loaded into memory.
            Default is 4 GB (4e9 bytes)
        max_in_mem_percent : float
            Maximum percentage of memory that can be loaded into memory.
            Default is 30% of available memory (0.3)

        Returns
        -------
        bool
            True if image should be loaded in memory, False otherwise.

        """
        from bioio_base.io import pathlike_to_fs
        from psutil import virtual_memory

        if path is None:
            path = self.path

        fs, path = pathlike_to_fs(path)
        filesize = fs.size(path)
        available_mem = virtual_memory().available
        return (
            filesize <= max_in_mem_bytes
            and filesize < max_in_mem_percent * available_mem
        )

    def get_napari_image_data(self, in_memory: bool | None = None) -> xr.DataArray:
        """
        Get the image data as a xarray DataArray.

        From BioImage documentation:
        If you do not want the image pre-stitched together, you can use the base reader
        by either instantiating the reader independently or using the `.reader` property.

        Parameters
        ----------
        in_memory : bool, optional
            Whether to load the image in memory or not.
            If None, will determine whether to load in memory based on the image size.

        Returns
        -------
        xr.DataArray
            Image data as a xarray DataArray.

        """
        if in_memory is None:
            in_memory = self._determine_in_memory()

        if DimensionNames.MosaicTile in self.reader.dims.order:
            try:
                if in_memory:
                    self.napari_data = self.reader.mosaic_xarray_data.squeeze()
                else:
                    self.napari_data = self.reader.mosaic_xarray_dask_data.squeeze()

            except NotImplementedError:
                logger.warning(
                    "Bioio: Mosaic tile switching not supported for this reader"
                )
                return None
        else:
            if in_memory:
                self.napari_data = self.reader.xarray_data.squeeze()
            else:
                self.napari_data = self.reader.xarray_dask_data.squeeze()

        return self.napari_data

    def get_napari_metadata(
        self,
        path: PathLike,
    ) -> dict:
        """
        Get the metadata for the image to be displayed in napari.

        Parameters
        ----------
        path : PathLike
            Path to the image file.
        img_data : xr.DataArray
            Image data as a xarray DataArray.
        img : BioImage
            BioImage object containing the image metadata

        Returns
        -------
        dict
            Metadata for the image to be displayed in napari.

        """
        if self.napari_data is None:
            self.get_napari_image_data()

        meta = {}
        scene = self.current_scene
        scene_index = self.current_scene_index
        single_no_scene = len(self.scenes) == 1 and self.current_scene == "Image:0"
        channel_dim = DimensionNames.Channel

        if channel_dim in self.napari_data.dims:
            # use filename if single scene and no scene name available
            if single_no_scene:
                channels_with_scene_index = [
                    f'{Path(path).stem}{LABEL_DELIMITER}{C}'
                    for C in self.napari_data.coords[channel_dim].data.tolist()
                ]
            else:
                channels_with_scene_index = [
                    f'{scene_index}{LABEL_DELIMITER}'
                    f'{scene}{LABEL_DELIMITER}{C}'
                    for C in self.napari_data.coords[channel_dim].data.tolist()
                ]
            meta['name'] = channels_with_scene_index
            meta['channel_axis'] = self.napari_data.dims.index(channel_dim)

        # not multi-chnanel, use current scene as image name
        else:
            if single_no_scene:
                meta['name'] = Path(path).stem
            else:
                meta['name'] = self.reader.current_scene

        # Handle if RGB
        if DimensionNames.Samples in self.reader.dims.order:
            meta['rgb'] = True

        # Handle scales
        scale = [
            getattr(self.physical_pixel_sizes, dim)
            for dim in self.napari_data.dims
            if dim in {DimensionNames.SpatialX, DimensionNames.SpatialY, DimensionNames.SpatialZ}
            and getattr(self.physical_pixel_sizes, dim) is not None
        ]

        if scale:
            meta['scale'] = tuple(scale)

        # get all other metadata
        img_meta = {'bioimage': self, 'raw_image_metadata': self.metadata}

        with contextlib.suppress(NotImplementedError):
            img_meta['metadata'] = self.ome_metadata

        meta['metadata'] = img_meta
        self.napari_metadata = meta
        return self.napari_metadata
