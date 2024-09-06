import pathlib

from bioio import BioImage

from napari_ndev._measure_container import MeasureContainer


def test_widg_init_no_viewer():
    wdg = MeasureContainer()
    assert wdg._image_directory.label == "Image directory"
    assert wdg._image_directory.value is None
    assert len(wdg._props_container) == len(wdg._sk_props)
    assert hasattr(wdg._prop, "area")
    assert hasattr(wdg._prop, "intensity_min")
    assert wdg.viewer is None


def test_widg_init_with_viewer(make_napari_viewer):
    viewer = make_napari_viewer()
    wdg = MeasureContainer(viewer)
    assert wdg.viewer == viewer


def test_get_0th_img_from_dir(tmp_path):
    image_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Apoc/Images"
    )
    container = MeasureContainer()
    img, id = container._get_0th_img_from_dir(image_directory)

    assert img is not None
    assert isinstance(id, pathlib.Path)


def test_update_dim_and_scales():
    image_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Apoc/Images"
    )
    file_name = "SPF-4MM-22 slide 9-S6_Top Slide_TR2_p00_0_A01f00d0.tiff"
    container = MeasureContainer()
    img = BioImage(image_directory / file_name)
    container._update_dim_and_scales(img)

    assert container._scale_tuple.value == (1.0, 0.2634, 0.2634)


def test_update_choices():
    container = MeasureContainer()
    image_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Workflow/Images"
    )
    label_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Workflow/Labels"
    )
    region_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Workflow/ShapesAsLabels"
    )
    container._image_directory.value = image_directory
    container._label_directory.value = label_directory
    container._region_directory.value = region_directory

    container._update_choices(image_directory, "Intensity")
    container._update_choices(label_directory, "Labels", update_label=True)
    container._update_choices(region_directory, "Region")
    print(container._label_choices)
    print(container._intensity_choices)

    # Check the choices in the label image ComboBox
    assert container._label_image.choices == ("Labels: Labels",)

    # Check the choices in the intensity images Select widget
    assert container._intensity_images.choices == (
        None,  # The default choice
        "Intensity: membrane",
        "Intensity: nuclei",
        "Labels: Labels",
        "Region: Shapes",
    )


def test_batch_measure_label_only(tmp_path):
    container = MeasureContainer()
    label_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Workflow/Labels"
    )
    # make a dummy output folder
    output_folder = tmp_path / "Output"
    output_folder.mkdir()

    container._label_directory.value = label_directory
    container._label_image.value = "Labels: Labels"
    container._output_directory.value = output_folder
    df, df_grouped = container.batch_measure()

    assert output_folder.exists()
    assert (output_folder / "measure_props_Labels.csv").exists()
    assert df is not None
    assert df_grouped is None
    assert list(df.columns) == ["id", "area"]


def test_batch_measure_intensity(tmp_path):
    container = MeasureContainer()
    image_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Workflow/Images"
    )
    label_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Workflow/Labels"
    )
    region_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Workflow/ShapesAsLabels"
    )
    # make a dummy output folder
    output_folder = tmp_path / "Output"
    output_folder.mkdir()

    container._image_directory.value = image_directory
    container._label_directory.value = label_directory
    container._region_directory.value = region_directory
    container._scale_tuple.value = (3, 0.25, 0.25)
    container._prop.intensity_mean.value = True

    container._label_image.value = "Labels: Labels"
    container._intensity_images.value = [
        "Region: Shapes",
        "Intensity: membrane",
        "Intensity: nuclei",
    ]
    container._output_directory.value = output_folder
    df, df_grouped = container.batch_measure()

    assert output_folder.exists()
    assert (output_folder / "measure_props_Labels.csv").exists()
    assert df is not None
    assert df_grouped is None
    assert list(df.columns) == [
        "id",
        "area",
        "intensity_mean-0",
        "intensity_mean-1",
        "intensity_mean-2",
    ]

    # TODO: add a multi-label for the test, and multiple images for grouping by id


def test_batch_measure_groupby(tmp_path):
    container = MeasureContainer()

    label_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Workflow/Labels"
    )
    image_directory = pathlib.Path(
        "src/napari_ndev/_tests/resources/Workflow/Images"
    )

    output_folder = tmp_path / "Output"
    output_folder.mkdir()

    container._label_directory.value = label_directory
    container._label_image.value = "Labels: Labels"
    container._image_directory.value = image_directory
    container._output_directory.value = output_folder
    # container._intensity_images.value = ["Intensity: membrane", "Intensity: nuclei"]
    container._prop.area.value = True
    # container._prop.intensity_mean.value = True
    container._create_grouped.value = True

    df, df_grouped = container.batch_measure()

    assert df is not None
    assert df_grouped is not None
    assert list(df.columns) == ["id", "area"]
    assert list(df_grouped.columns) == [
        "id",
        "area_mean",
        "area_std",
        "id_count",
    ]
