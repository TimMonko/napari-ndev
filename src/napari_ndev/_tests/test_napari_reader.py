from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da

# import npe2
import numpy as np
import pytest

from napari_ndev._napari_reader import napari_get_reader

if TYPE_CHECKING:
    from npe2._pytest_plugin import TestPluginManager

###############################################################################

RGB_TIFF = "RGB.tiff" # has two scense
# PNG_FILE = "example.png"
# GIF_FILE = "example.gif"
# OME_TIFF = "pipeline-4.ome.tiff"

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
    expected_dtype: type,
    expected_meta: dict[str, Any],
    npe2pm: TestPluginManager,
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

        # # confirm that this also works via npe2
        # with npe2pm.tmp_plugin(package='napari-ndev') as plugin:
        #     [via_npe2] = npe2.read([path], stack=False, plugin_name=plugin.name)
        #     assert via_npe2[0].shape == data.shape
