from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.array as da
import npe2
import numpy as np
import pytest

from napari_ndev._napari_reader import napari_get_reader

if TYPE_CHECKING:
    from npe2._pytest_plugin import TestPluginManager

###############################################################################

PNG_FILE = "example.png"
GIF_FILE = "example.gif"
OME_TIFF = "pipeline-4.ome.tiff"

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
        # (PNG_FILE, (800, 537, 4), {"name": Path(PNG_FILE).stem, "rgb": True}),
        # (GIF_FILE, (72, 268, 268, 3), {"name": Path(GIF_FILE).stem, "rgb": True}),
        (
            OME_TIFF,
            (4, 65, 600, 900),
            {
                "name": [
                    f"{Path(OME_TIFF).stem} :: Bright_2",
                    f"{Path(OME_TIFF).stem} :: EGFP",
                    f"{Path(OME_TIFF).stem} :: CMDRP",
                    f"{Path(OME_TIFF).stem} :: H3342",
                ],
                "channel_axis": 0,
                "scale": (0.29, 0.10833333333333332, 0.10833333333333332),
            },
        ),
    ],
)
def test_reader(
    resources_dir: Path,
    filename: str,
    expected_shape: tuple[int, ...],
    expected_dtype: type,
    expected_meta: dict[str, Any],
    npe2pm: TestPluginManager,
) -> None:
    # Resolve filename to filepath
    if isinstance(filename, str):
        path = str(resources_dir / filename)

    # Get reader
    reader = napari_get_reader(path)

    # Check callable
    assert callable(reader)

    # Get data
    layer_data = reader(path)

    # We only return one layer
    if layer_data is not None:
        data, meta, _ = layer_data[0]  # type: ignore

        # Check layer data
        assert isinstance(data, expected_dtype)  # type: ignore
        assert data.shape == expected_shape  # type: ignore

        # Check meta
        meta.pop("metadata", None)
        assert meta == expected_meta  # type: ignore

        # confirm that this also works via npe2
        with npe2pm.tmp_plugin(package="napari-ndev") as plugin:
            [via_npe2] = npe2.read([path], stack=False, plugin_name=plugin.name)
            assert via_npe2[0].shape == data.shape  # type: ignore
