import logging
import time
from pathlib import Path
from typing import List, Tuple, Union

PathLike = Union[str, Path]


def get_directory_and_files(
    dir: PathLike, pattern: str = "*"
) -> Tuple[Path, List[Path]]:
    """
    Get the directory and files in the specified directory.

    Args:
        dir (PathLike): The directory path.
        pattern (str, optional): The file pattern to match. Defaults to '*'.

    Returns:
        Tuple[Path, List[Path]]: A tuple containing the directory path and a
        list of file paths.
    """
    directory = Path(dir)
    files = list(directory.glob(pattern))
    return directory, files


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


def setup_logger(log_loc=PathLike):
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
