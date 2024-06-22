import logging
import time
from pathlib import Path
from typing import List, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from aicsimageio import AICSImage

__all__ = [
    "check_for_missing_files",
    "get_channel_names",
    "get_directory_and_files",
    "get_squeezed_dim_order",
    "setup_logger",
]


def get_directory_and_files(
    dir: Union[str, Path],
    pattern: Union[List[str], str] = [
        "tif",
        "tiff",
        "nd2",
        "czi",
        "lif",
        "oib",
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "gif",
    ],
) -> Tuple[Path, List[Path]]:
    """
    Get the directory and files in the specified directory.

    Args:
        dir (Union[str, Path]): The directory path.
        pattern (Union[List[str], str], optional): The file pattern(s) to match
            If a string is provided, it will be treated as a single pattern.
            If a list is provided, each element will be treated as a separate
            pattern.
            Defaults to ['tif', 'tiff', 'nd2', 'czi', 'lif',
            'oib', 'png', 'jpg', 'jpeg', 'bmp', 'gif']

    Returns:
        Tuple[Path, List[Path]]: A tuple containing the directory path and a
        list of file paths.
    """
    directory = Path(dir)

    pattern = [pattern] if isinstance(pattern, str) else pattern
    pattern_glob = [f"*.{pat}" for pat in pattern]

    files = []
    for p_glob in pattern_glob:
        for file in directory.glob(p_glob):
            files.append(file)
    return directory, files


def get_channel_names(img: "AICSImage") -> List[str]:
    """
    Get the channel names from an AICSImage object.

    If the image has a dimension order that includes "S" (it is RGB),
    return the default channel names ["red", "green", "blue"].
    Otherwise, return the channel names from the image.

    Parameters:
        img (AICSImage): The AICSImage object.

    Returns:
        List[str]: The channel names.
    """
    if "S" in img.dims.order:
        return ["red", "green", "blue"]
    else:
        return img.channel_names


def get_squeezed_dim_order(
    img: "AICSImage", skip_dims: Union[List[str], str] = ["C", "S"]
) -> str:
    """
    Returns a string containing the squeezed dimensions of the given AICSImage
    object.

    Parameters:
        img (AICSImage): The AICSImage object.
        skip_dims (Union[List[str], str], optional): Dimensions to skip.
            Defaults to ["C", "S"].

    Returns:
        str: A string containing the squeezed dimensions.
    """
    return "".join(
        {k: v for k, v in img.dims.items() if v > 1 and k not in skip_dims}
    )


def check_for_missing_files(
    files: List[Path], *directories: Path
) -> List[tuple]:
    """
    Check if the given files are missing in the specified directories.

    Args:
        files (List[Path]): List of files to check.
        directories (Tuple[Path]): Tuple of directories to search for the
        files.

    Returns:
        List[tuple]: List of tuples containing the missing files and their
        corresponding directories.
    """
    missing_files = []
    for file in files:
        for directory in directories:
            file_loc = directory / file.name
            if not file_loc.exists():
                print(f"{file.name} missing in {directory.name}")
                missing_files.append((file.name, directory.name))

    return missing_files


def setup_logger(log_loc=Union[str, Path]):
    """
    Set up a logger with the specified log location.

    Parameters:
    log_loc (PathLike): The path to the log file.

    Returns:
    logger (logging.Logger): The logger object.
    handler (logging.FileHandler): The file handler object.
    """
    logger = logging.getLogger(__name__ + str(time.time()))
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_loc)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, handler
