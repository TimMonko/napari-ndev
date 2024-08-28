import pathlib

import pytest
from bioio import BioImage

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


def test_update_dim_and_scales():
    image_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Apoc/Images"
    )
    file_name = "SPF-4MM-22 slide 9-S6_Top Slide_TR2_p00_0_A01f00d0.tiff"
    container = MeasureContainer()
    img = BioImage(image_directory / file_name)
    container._update_dim_and_scales(img)

    assert container._scale_tuple.value == (1.0, 0.2634, 0.2634)


# def test_update_choices(tmp_path):
#     # Create an instance of MeasureContainer
#     container = MeasureContainer()

#     # Set up the test directories
#     image_directory = tmp_path / "images"
#     image_directory.mkdir()
#     label_directory = tmp_path / "labels"
#     label_directory.mkdir()
#     region_directory = tmp_path / "regions"
#     region_directory.mkdir()

#     # Create some test files in the directories
#     image_file = image_directory / "image.tif"
#     image_file.touch()
#     label_file = label_directory / "label.tif"
#     label_file.touch()
#     region_file = region_directory / "region.tif"
#     region_file.touch()

#     # Set the directory values in the container
#     container._image_directory.value = str(image_directory)
#     container._label_directory.value = str(label_directory)
#     container._region_directory.value = str(region_directory)

#     # Call the _update_choices method
#     container._update_choices(image_directory, "Intensity")
#     container._update_choices(label_directory, "Labels", update_label=True)
#     container._update_choices(region_directory, "Region")

#     # Check the choices in the label image ComboBox
#     assert container._label_image.choices == ["Intensity: image.tif"]

#     # Check the choices in the intensity images Select widget
#     assert container._intensity_images.choices == [
#         "Intensity: image.tif",
#         "Labels: label.tif",
#         "Region: region.tif",
#     ]
