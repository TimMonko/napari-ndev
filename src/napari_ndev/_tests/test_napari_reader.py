from __future__ import annotations

from pathlib import Path
from typing import Any

import dask.array as da

# import npe2
import numpy as np
import pytest

from napari_ndev._napari_reader import napari_get_reader

###############################################################################

RGB_TIFF = "RGB.tiff" # has two scense
MULTISCENE_CZI = r"0T-4C-0Z-7pos.czi"
# PNG_FILE = "example.png"
# GIF_FILE = "example.gif"
OME_TIFF = "cells3d2ch.tiff"

###############################################################################

@pytest.mark.parametrize(
    ("in_memory", "expected_dtype"),
    [
        (True, np.ndarray),
        (False, da.core.Array),
    ],
)
@pytest.mark.parametrize(
    ("filename", "expected_shape", "expected_meta"),
    [
        (
            RGB_TIFF,
            (1440, 1920, 3),
            {
                'name': 'Image:0',
                'scale': (264.5833333333333, 264.5833333333333),
                'rgb': True,
            }
        ),
    ],
)
def test_reader(
    resources_dir: Path,
    filename: str,
    in_memory: bool,
    expected_shape: tuple[int, ...],
    expected_dtype,
    expected_meta: dict[str, Any],
    # npe2pm: TestPluginManager,
) -> None:
    # Resolve filename to filepath
    if isinstance(filename, str):
        path = str(resources_dir / filename)

    # Get reader
    reader = napari_get_reader(path, in_memory=in_memory, open_first_scene_only=True)

    # Check callable
    assert callable(reader)

    # Get data
    layer_data = reader(path)

    # We only return one layer
    if layer_data is not None:
        data, meta, _ = layer_data[0]

        # Check layer data
        assert isinstance(data, expected_dtype)
        assert data.shape == expected_shape

        # Check meta
        meta.pop('metadata', None)
        assert meta == expected_meta


@pytest.mark.parametrize(
    ("in_memory", "expected_dtype"),
    [
        (True, np.ndarray),
        (False, da.core.Array),
    ],
)
@pytest.mark.parametrize(
    ("filename", "expected_shape"),
    [
        (RGB_TIFF, (120, 160, 3)),
        (MULTISCENE_CZI, (32, 32)),
    ],
)
def test_for_multiscene_widget(
    make_napari_viewer,
    resources_dir: Path,
    filename: str,
    in_memory: bool,
    expected_dtype,
    expected_shape: tuple[int, ...],
) -> None:
    # Make a viewer
    viewer = make_napari_viewer()
    assert len(viewer.layers) == 0
    assert len(viewer.window._dock_widgets) == 0

    # Resolve filename to filepath
    if isinstance(filename, str):
        path = str(resources_dir / filename)

    # Get reader
    reader = napari_get_reader(path, in_memory)

    if reader is not None:
        # Call reader on path
        reader(path)

        if len(viewer.window._dock_widgets) != 0:
            # Get the second scene
            viewer.window._dock_widgets[f"{filename} :: Scenes"].widget().setCurrentRow(
                1
            )
            data = viewer.layers[0].data
            assert isinstance(data, expected_dtype)
            assert data.shape == expected_shape
        else:
            data, _, _ = reader(path)[0]
            assert isinstance(data, expected_dtype)
            assert data.shape == expected_shape

def test_napari_get_reader_multi_path(resources_dir: Path) -> None:
    # Get reader
    reader = napari_get_reader(
        [str(resources_dir / RGB_TIFF), str(resources_dir / MULTISCENE_CZI)],
        in_memory=True,
    )

    # Check callable
    assert reader is None

def test_napari_get_reader_ome_override(resources_dir: Path) -> None:
    reader = napari_get_reader(
        str(resources_dir / OME_TIFF),
    )

    assert callable(reader)

def test_napari_get_reader_unsupported(resources_dir: Path) -> None:
    reader = napari_get_reader(
        str(resources_dir / "measure_props_Labels.csv"),
    )

    assert reader is None
