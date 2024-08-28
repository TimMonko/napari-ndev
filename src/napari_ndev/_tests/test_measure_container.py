import pathlib

import pytest

from napari_ndev._measure_container import MeasureContainer


@pytest.mark.parametrize(
    "viewer", [None, pytest.lazy_fixture("make_napari_viewer")]
)
def test_widg_init(viewer):
    wdg = MeasureContainer(viewer)
    assert wdg._image_directory.label == "Image directory"
    assert len(wdg._props_container) == len(wdg._sk_props)
    assert hasattr(wdg._prop, "area")
    assert hasattr(wdg._prop, "intensity_min")
    if viewer is None:
        assert wdg.viewer is None


def test_get_0th_img_from_dir(tmp_path):
    image_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Apoc/Images"
    )
    container = MeasureContainer()
    img = container._get_0th_img_from_dir(image_directory)

    assert img is not None
