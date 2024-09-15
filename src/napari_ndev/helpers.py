"""
Helper functions for file handling, image processing, and logging setup.

Functions
---------
get_directory_and_files : Get the directory and files in the specified directory.
get_channel_names : Get the channel names from an AICSImage or BioImage object.
get_squeezed_dim_order : Return a string containing the squeezed dimensions of the given AICSImage or BioImage object.
create_id_string : Create an ID string for the given image.
check_for_missing_files : Check if the given files are missing in the specified directories.
setup_logger : Set up a logger with the specified log location.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from aicsimageio import AICSImage
    from bioio import BioImage

__all__ = [
    'check_for_missing_files',
    'create_id_string',
    'get_channel_names',
    'get_directory_and_files',
    'get_Image',
    'get_squeezed_dim_order',
    'setup_logger',
]

def get_Image(file: str | Path) -> AICSImage | BioImage:
    """
    Read the image file with BioImage or AICSImage.

    Open the image file with BioImage, or if the file is not supported,
    open it with AICSImage.

    Parameters
    ----------
    file : str or Path
        The file path.

    Returns
    -------
    AICSImage or BioImage
        The image object.

    """
    from bioio import BioImage
    from bioio_base.exceptions import UnsupportedFileFormatError

    try:
        img = BioImage(file)
    except UnsupportedFileFormatError:
        from aicsimageio import AICSImage
        img = AICSImage(file)
    return img


def get_directory_and_files(
    dir_path: str | Path | None = None,
    pattern: list[str] | str | None = None,
) -> tuple[Path, list[Path]]:
    """
    Get the directory and files in the specified directory.

    Parameters
    ----------
    dir_path : str or Path or None, optional
        The directory path.
    pattern : list of str or str or None, optional
        The file pattern(s) to match. If a string is provided, it will be treated as a single pattern.
        If a list is provided, each element will be treated as a separate pattern.
        Defaults to ['tif', 'tiff', 'nd2', 'czi', 'lif', 'oib', 'png', 'jpg', 'jpeg', 'bmp', 'gif'].

    Returns
    -------
    tuple of (Path, list of Path)
        A tuple containing the directory path and a list of file paths.

    """
    if pattern is None:
        pattern = [
            'tif',
            'tiff',
            'nd2',
            'czi',
            'lif',
            'oib',
            'png',
            'jpg',
            'jpeg',
            'bmp',
            'gif',
        ]
    if dir_path is None:
        return None, []

    directory = Path(dir_path)

    if dir_path is not None and not directory.exists():
        raise FileNotFoundError(f'Directory {dir_path} does not exist.')

    pattern = [pattern] if isinstance(pattern, str) else pattern
    # add *. to each pattern if it doesn't already have either
    pattern_glob = []
    for pat in pattern:
        if '.' not in pat:
            pat = f'*.{pat}'
        if '*' not in pat:
            pat = f'*{pat}'
        pattern_glob.append(pat)

    files = []
    for p_glob in pattern_glob:
        for file in directory.glob(p_glob):
            files.append(file)
    return directory, files


def get_channel_names(img: AICSImage | BioImage) -> list[str]:
    """
    Get the channel names from an AICSImage object.

    If the image has a dimension order that includes "S" (it is RGB),
    return the default channel names ["red", "green", "blue"].
    Otherwise, return the channel names from the image.

    Parameters
    ----------
    img : AICSImage or BioImage
        The AICSImage object.

    Returns
    -------
    list of str
        The channel names.

    """
    if 'S' in img.dims.order:
        return ['red', 'green', 'blue']
    return img.channel_names


def get_squeezed_dim_order(
    img: AICSImage | BioImage,
    skip_dims: list[str] | str | None = None,
) -> str:
    """
    Return a string containing the squeezed dimensions of the given AICSImage object.

    Parameters
    ----------
    img : AICSImage or BioImage
        The AICSImage object.
    skip_dims : list of str or str or None, optional
        Dimensions to skip. Defaults to ["C", "S"].

    Returns
    -------
    str
        A string containing the squeezed dimensions.

    """
    if skip_dims is None:
        skip_dims = ['C', 'S']
    return ''.join(
        {k: v for k, v in img.dims.items() if v > 1 and k not in skip_dims}
    )


def create_id_string(img: BioImage | AICSImage, identifier: str) -> str:
    """
    Create an ID string for the given image.

    Parameters
    ----------
    img : BioImage or AICSImage
        The image object.
    identifier : str
        The identifier string.

    Returns
    -------
    str
        The ID string.

    Examples
    --------
    >>> create_id_string(img, 'test')
    'test__0__Scene:0'

    """
    scene_idx = img.current_scene_index
    # scene = img.current_scene
    # instead use ome_metadata.name because this gets saved with OmeTiffWriter
    try:
        if img.ome_metadata.images[scene_idx].name is None:
            scene = img.current_scene
        else:
            scene = img.ome_metadata.images[scene_idx].name
    except NotImplementedError:
        scene = img.current_scene  # not useful with OmeTiffReader, atm
    id_string = f'{identifier}__{scene_idx}__{scene}'
    return id_string


def check_for_missing_files(
    files: list[Path] | list[str], *directories: Path | str
) -> list[tuple]:
    """
    Check if the given files are missing in the specified directories.

    Parameters
    ----------
    files : list of Path or list of str
        List of files to check.
    directories : tuple of Path or str
        Tuple of directories to search for the files.

    Returns
    -------
    list of tuple
        List of tuples containing the missing files and their corresponding directories.

    """
    missing_files = []
    for file in files:
        for directory in directories:
            if isinstance(directory, str):
                directory = Path(directory)
            if isinstance(file, str):
                file = Path(file)

            file_loc = directory / file.name
            if not file_loc.exists():
                missing_files.append((file.name, directory.name))

    return missing_files


def setup_logger(log_loc=Union[str, Path]):
    """
    Set up a logger with the specified log location.

    Parameters
    ----------
    log_loc : str or Path
        The path to the log file.

    Returns
    -------
    logger : logging.Logger
        The logger object.
    handler : logging.FileHandler
        The file handler object.

    """
    logger = logging.getLogger(__name__ + str(time.time()))
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_loc)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, handler
