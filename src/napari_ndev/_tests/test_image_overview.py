import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib_scalebar.scalebar import ScaleBar

from napari_ndev import nImage
from napari_ndev.image_overview import ImageOverview, image_overview


@pytest.fixture
def image_and_label_sets():
    img = nImage(
        pathlib.Path(
            r'src/napari_ndev/_tests/resources/Workflow/Images/cells3d2ch.tiff'
        )
    )
    image_data = np.squeeze(img.data)

    image_set1 = {
        'image': [image_data[0], image_data[1]],
        'colormap': ['PiYG', 'viridis'],
        'title': ['Image 1', 'Image 2'],
    }

    lbl = nImage(
        pathlib.Path(
            r'src/napari_ndev/_tests/resources/Workflow/Labels/cells3d2ch.tiff'
        )
    )

    label_set = {
        'image': [None, np.squeeze(lbl.data)],
        'colormap': [None, 'Labels'],
        'title': [None, 'Labels'],
    }

    return image_set1, label_set


def test_image_overview(image_and_label_sets):
    fig = image_overview(image_and_label_sets)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4

    # Check the properties of each subplot
    assert fig.axes[0].get_title() == 'Image 1'
    assert fig.axes[0].get_images()[0].get_cmap().name == 'PiYG'

    assert fig.axes[1].get_title() == 'Image 2'
    assert fig.axes[1].get_images()[0].get_cmap().name == 'viridis'

    # fig. axes[2] should be empty
    assert not fig.axes[2].get_title()
    assert not fig.axes[2].get_images()

    assert fig.axes[3].get_title() == 'Labels'

def test_image_overview_plot_scale(image_and_label_sets):
    fig = image_overview(image_and_label_sets, fig_scale=(5, 6))

    assert isinstance(fig, plt.Figure)
    assert np.array_equal(
        fig.get_size_inches(), np.array([10, 12])
    ) # 2 columns * 5 width, 2 rows * 6 height

def test_image_overview_plot_title(image_and_label_sets):
    test_title = 'Test title'
    fig = image_overview(image_and_label_sets, fig_title=test_title)

    assert isinstance(fig, plt.Figure)
    assert fig._suptitle.get_text() == test_title

def test_image_overview_scalebar_float(image_and_label_sets):
    fig = image_overview(image_and_label_sets, scalebar=0.5)

    assert isinstance(fig, plt.Figure)

    scalebar_list = []
    for ax in fig.axes:
        scalebar = [
            child for child in ax.get_children()
            if isinstance(child, ScaleBar)
        ]
        scalebar_list.append(scalebar)

    assert len(scalebar_list) == 4
    assert len(scalebar_list[0]) == 1
    assert len(scalebar_list[1]) == 1
    assert len(scalebar_list[2]) == 0 # has no ax to add scalebar to
    assert len(scalebar_list[3]) == 1

def test_image_overview_scalebar_dict(image_and_label_sets):
    scalebar_dict = {
        'dx': 0.5,
        'units': 'mm',
        'location': 'upper right',
        'badkey': 'badvalue',
    }
    fig = image_overview(image_and_label_sets, scalebar=scalebar_dict)

    assert isinstance(fig, plt.Figure)

    # get scalebar from the first axis, multi-axes tested in float
    scalebar = [
        child for child in fig.axes[0].get_children()
        if isinstance(child, ScaleBar)
    ]
    assert len(scalebar) == 1


def test_imageoverview_init(image_and_label_sets):
    im = ImageOverview(image_and_label_sets, show=False)
    assert isinstance(im.fig, plt.Figure)


def test_imageoverview_save(image_and_label_sets):
    im = ImageOverview(image_and_label_sets, show=False)
    # test that the figure can be saved with .save()
    save_path = pathlib.Path(r'src\napari_ndev\_tests\resources')
    save_file_path = save_path / 'image_overview.png'

    im.save(str(save_path), 'image_overview.png')
    # assert that it was saved
    assert save_file_path.exists()
    # remove the saved file
    save_file_path.unlink()
