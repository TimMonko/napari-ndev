"""Additional functionality for BioImage objects to be used in napari-ndev."""

from __future__ import annotations

import contextlib
import importlib
import logging
from pathlib import Path

import xarray as xr
import zarr
from bioio import BioImage
from bioio_base.dimensions import DimensionNames
from bioio_base.reader import Reader
from bioio_base.types import ImageLike

from napari.types import PathLike
from napari_ndev import get_settings

logger = logging.getLogger(__name__)

DELIM = " :: "

def _apply_zarr_compat_patch():
    """Apply zarr compatibility patch for zarr>=3.0 with BioIO."""
    zarr_version = tuple(int(x) for x in zarr.__version__.split('.')[:2])
    if zarr_version >= (3, 0) and not hasattr(zarr.storage, 'FSStore'):
        class FSStoreShim:
            def __new__(cls, *args, **kwargs):
                return zarr.storage.directory.DirectoryStore(*args, **kwargs)
        # patch zarr.storage.FSStore to FFStoreShim
        zarr.storage.FSStore = FSStoreShim

_apply_zarr_compat_patch()

class nImage(BioImage):
    """
    An nImage is a BioImage with additional functionality for napari-ndev.

    Parameters
    ----------
    image : ImageLike
        Image to be loaded. Can be a path to an image file, a numpy array,
        or an xarray DataArray.
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
        """
        Initialize an nImage with an image, and optionally a reader.

        If a reader is not provided, a reader will be determined by bioio.
        However, if the image is supported by bioio-ome-tiff, the reader
        will be set to bioio_ome_tiff.Reader to override the softer decision
        made by bioio.BioImage.determine_plugin().

        Note: The issue here is that bioio.BioImage.determine_plugin() will
        sort by install time and choose the first plugin that supports the
        image. This is not always the desired behavior, because bioio-tifffile
        can take precedence over bioio-ome-tiff, even if the image was saved
        as an OME-TIFF via bioio.writers.OmeTiffWriter (which is the case
        for napari-ndev).
        """
        self.settings = get_settings()

        if reader is None:
            from bioio import plugin_feasibility_report as pfr
            fr = pfr(image)
            if self.settings.PREFERRED_READER in fr and fr[self.settings.PREFERRED_READER].supported:
                reader_module = importlib.import_module(
                    self.settings.PREFERRED_READER.replace('-', '_')
                )
                reader = reader_module.Reader

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
        path: PathLike | None = None,
    ) -> dict:
        """
        Get the metadata for the image to be displayed in napari.

        Parameters
        ----------
        path : PathLike
            Path to the image file.

        Returns
        -------
        dict
            Metadata for the image to be displayed in napari.

        """
        if self.napari_data is None:
            self.get_napari_image_data() # this also sets self.path
        # override the nImage path only if provided
        path = self.path if path is None else path

        # Determine available metadata information
        meta = {}
        scene = self.current_scene
        scene_idx = self.current_scene_index
        path_stem = Path(self.path).stem if self.path is not None else 'unknown path'

        NO_SCENE = len(self.scenes) == 1 and self.current_scene == 'Image:0' # Image:0 is the default scene name, suggesting there is no information
        CHANNEL_DIM = DimensionNames.Channel
        IS_MULTICHANNEL = CHANNEL_DIM in self.napari_data.dims

        # Build metadata under various condition
        # add channel_axis if unpacking channels as layers
        if IS_MULTICHANNEL:
            channel_names = self.napari_data.coords[CHANNEL_DIM].data.tolist()

            if self.settings.UNPACK_CHANNELS_AS_LAYERS:
                meta['channel_axis'] = self.napari_data.dims.index(CHANNEL_DIM)

                if NO_SCENE:
                    meta['name'] = [
                        f'{C}{DELIM}{path_stem}'
                        for C in channel_names
                    ]
                else:
                    meta['name'] = [
                        f'{C}{DELIM}{scene_idx}{DELIM}{scene}{DELIM}{path_stem}'
                        for C in channel_names
                    ]

            if not self.settings.UNPACK_CHANNELS_AS_LAYERS:
                meta['name'] = (
                    f'{DELIM}'.join(channel_names) +
                    f'{DELIM}{scene_idx}{DELIM}{scene}{DELIM}{path_stem}'
                )

        if not IS_MULTICHANNEL:
            if NO_SCENE:
                meta['name'] = path_stem
            else:
                meta['name'] = f'{scene_idx}{DELIM}{scene}{DELIM}{path_stem}'

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
