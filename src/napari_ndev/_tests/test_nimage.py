from __future__ import annotations

from pathlib import Path

from napari_ndev import nImage

RGB_TIFF = "RGB.tiff"  # has two scenes

def test_nImage_init(resources_dir: Path):
    img = nImage(resources_dir / RGB_TIFF)
    assert img.path == resources_dir / RGB_TIFF
    assert img.reader is not None
    assert img.data.shape == (1, 1, 1, 1440, 1920, 3)
    assert img.napari_data is None

def test_determine_in_memory(resources_dir: Path):
    img = nImage(resources_dir / RGB_TIFF)
    assert img._determine_in_memory() is True

def test_get_napari_image_data(resources_dir: Path):
    img = nImage(resources_dir / RGB_TIFF)
    img.get_napari_image_data()
    assert img.napari_data.shape == (1440, 1920, 3)
    assert img.napari_data.dims == ('Y', 'X', 'S')

def test_get_napari_metadata(resources_dir: Path):
    img = nImage(resources_dir / RGB_TIFF)
    img.get_napari_metadata(path = img.path)
    assert img.napari_metadata['name'] == 'Image:0'
    assert img.napari_metadata['scale'] == (264.5833333333333, 264.5833333333333)
    assert img.napari_metadata['rgb'] is True
