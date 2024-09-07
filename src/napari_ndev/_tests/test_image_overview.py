import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
from bioio import BioImage

from napari_ndev.image_overview import ImageOverview, image_overview


@pytest.fixture
def image_and_label_sets():
    img = BioImage(
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

    lbl = BioImage(
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
