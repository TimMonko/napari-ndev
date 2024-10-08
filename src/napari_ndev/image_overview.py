"""
Function and class to create and manage image overviews with stackview.

It includes a function `image_overview` to generate an overview of images
and a class `ImageOverview` to generate and save image overviews.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import stackview


class ImageOverview:
    """
    A class for generating and saving image overviews.

    Use this class to prevent a memory leak otherwise generated by the
    image_overview() function when show=True. For some reason, preventing
    the memory leak requires the use of a class instead of a function, and
    show=False.
    """

    def __init__(
        self,
        image_sets: list[dict],
        xscale: float = 3,
        yscale: float = 3,
        image_title: str = '',
        show: bool = False,
    ):
        """
        Initialize an ImageOverivew object.

        Parameters
        ----------
        image_sets : list of dict
            A list of dictionaries containing image sets. See
            `napari_ndev.image_overview` for more information.
        xscale : float, optional
            The scale factor for the x-axis. Default is 3.
        yscale : float, optional
            The scale factor for the y-axis. Default is 3.
        image_title : str, optional
            The title of the image overview. Default is an empty string.
        show : bool, optional
            Whether to display the generated overview. Default is False.
            Prevents memory leak when False.

        """
        plt.ioff()
        self.fig = image_overview(image_sets, xscale, yscale, image_title)
        if show:
            plt.show()
        plt.close()

    def save(
        self,
        directory: str | None = None,
        filename: str | None = None,
    ):
        """
        Save the generated image overview with matplotlib.savefig.

        Parameters
        ----------
        directory : str, optional
            The directory to save the image overview. If not provided, the
            current directory will be used.
        filename : str, optional
            The filename of the saved image overview. If not provided, a
            default filename will be used.

        """
        import pathlib

        path_dir = pathlib.Path(directory)
        path_dir.mkdir(parents=True, exist_ok=True)
        filepath = path_dir / filename
        self.fig.savefig(filepath)


def image_overview(
    image_sets: list[dict],
    xscale: float = 3,
    yscale: float = 3,
    plot_title: str = '',
):
    """
    Create an overview of images.

    Parameters
    ----------
    image_sets : list of dict
        A list of dictionaries, each containing an image set. Each image set
        should be a dictionary containing the following keys:
        - image (list): A list of images to display.
        - title (list of str, optional): The title of the image set.
        - colormap (list of str, optional): The colormap to use.
            "labels" will display the image as labels.
        - labels (list of bool, optional): Whether to display labels.
        - **kwargs: Additional keyword arguments to pass to stackview.imshow.
    xscale : float, optional
        The x scale of the overview. Defaults to 3.
    yscale : float, optional
        The y scale of the overview. Defaults to 3.
    plot_title : str, optional
        The title of the plot. Defaults to an empty string.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the image overview.

    """
    # create the subplot grid
    num_rows = len(image_sets)
    num_columns = max([len(image_set['image']) for image_set in image_sets])
    fig, axs = plt.subplots(
        num_rows,
        num_columns,
        figsize=(num_columns * xscale, num_rows * yscale),
    )

    if num_rows == 1:
        axs = [axs]
    if num_columns == 1:
        axs = [[ax] for ax in axs]

    # iterate through the image sets
    for row, image_set in enumerate(image_sets):
        for col, _image in enumerate(image_set['image']):
            # create a dictionary from the col-th values of each key
            image_dict = {key: value[col] for key, value in image_set.items()}

            # turn off the subplot and continue if there is no image
            if image_dict.get('image') is None:
                axs[row][col].axis('off')
                continue

            # create a labels key if it doesn't exist, but does in colormap
            cmap = image_dict.get('colormap')
            if cmap is not None and cmap.lower() == 'labels':
                image_dict['labels'] = True

            stackview.imshow(**image_dict, plot=axs[row][col])

    plt.suptitle(plot_title, fontsize=16)
    plt.tight_layout()
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    return fig
