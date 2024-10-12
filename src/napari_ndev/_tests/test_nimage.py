from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import xarray as xr

from napari_ndev import nImage


@pytest.fixture
def mock_bioimage() -> nImage:
    path = Path("mock_path")
    reader = MagicMock()
    bioimage = nImage(path, reader)
    return bioimage

@pytest.fixture
def _setup_mock_properties(mock_bioimage: nImage):
    # Mock the properties
    type(mock_bioimage).current_scene = PropertyMock(return_value="Image:0")
    type(mock_bioimage).current_scene_index = PropertyMock(return_value=0)
    type(mock_bioimage).scenes = PropertyMock(return_value=["Image:0"])
    type(mock_bioimage).physical_pixel_sizes = PropertyMock(return_value=MagicMock(
        SpatialX=0.1, SpatialY=0.1, SpatialZ=None
    ))

@patch('bioio_base.io.pathlike_to_fs')
@patch('psutil.virtual_memory')
def test_determine_in_memory(mock_virtual_memory, mock_pathlike_to_fs, mock_bioimage: nImage):
    mock_virtual_memory.return_value.available = 8e9  # 8 GB
    mock_pathlike_to_fs.return_value = (MagicMock(), 'mock_path')
    mock_pathlike_to_fs.return_value[0].size.return_value = 2e9  # 2 GB

    assert mock_bioimage._determine_in_memory() is True

    mock_pathlike_to_fs.return_value[0].size.return_value = 5e9  # 5 GB
    assert mock_bioimage._determine_in_memory() is False

def test_get_napari_image_data(mock_bioimage: nImage):
    mock_bioimage.reader.dims.order = []
    mock_bioimage.reader.xarray_data.squeeze.return_value = xr.DataArray([1, 2, 3])
    mock_bioimage.reader.xarray_dask_data.squeeze.return_value = xr.DataArray([1, 2, 3])

    data = mock_bioimage.get_napari_image_data(in_memory=True)
    assert data.equals(xr.DataArray([1, 2, 3]))

    data = mock_bioimage.get_napari_image_data(in_memory=False)
    assert data.equals(xr.DataArray([1, 2, 3]))

@pytest.mark.usefixtures("_setup_mock_properties")
def test_get_napari_metadata(mock_bioimage: nImage):
    mock_bioimage.napari_data = xr.DataArray([1, 2, 3])
    mock_bioimage.reader.dims.order = []

    metadata = mock_bioimage.get_napari_metadata("mock_path")
    assert metadata['name'] == "mock_path"
    # assert metadata['scale'] == (0.1, 0.1) # fix because pulls from nImage.napari_data, not physical_pixel_sizes
    assert 'metadata' in metadata
