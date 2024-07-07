import matplotlib.pyplot as plt
import stackview


def image_overview(
    image_sets: list[dict], xscale: float = 3, yscale: float = 3
):
    """
    Create an overview of images.

    Parameters:
        image_sets (list[dict]): A dictionary of image sets. Each image set
            should be a dictionary containing the following keys:
            - image (list): A list of images to display.
            - title (list[str], optional): The title of the image set.
            - colormap (list[str], optional): The colormap to use.
                "labels" will display the image as labels.
            - labels (list[bool], optional): Whether to display labels.
            - **kwargs: Additional keyword arguments to pass to
                    stackview.imshow.
        xscale (float, optional): The x scale of the overview. Defaults to 3.
        yscale (float, optional): The y scale of the overview. Defaults to 3.
    """
    # create the subplot grid
    num_rows = len(image_sets)
    num_columns = max([len(image_set["image"]) for image_set in image_sets])
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
        for col, image in enumerate(image_set["image"]):
            # create a dictionary from the col-th values of each key
            image_dict = {key: value[col] for key, value in image_set.items()}

            # turn off the subplot and continue if there is no image
            if image_dict.get("image") is None:
                axs[row][col].axis("off")
                continue

            # create a labels key if it doesn't exist, but does in colormap
            cmap = image_dict.get("colormap")
            if cmap is not None and cmap.lower() == "labels":
                image_dict["labels"] = True

            stackview.imshow(**image_dict, plot=axs[row][col])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    return fig


class ImageOverview:
    """
    A class for generating and saving image overviews.

    Parameters:
    - image_sets (list[dict]): A dictionary containing image sets. See
        `napari_ndev.image_overview` for more information.
    - xscale (float): The scale factor for the x-axis. Default is 3.
    - yscale (float): The scale factor for the y-axis. Default is 3.
    - show (bool): Whether to display the generated overview. Default is False.
    """

    def __init__(
        self,
        image_sets: list[dict],
        xscale: float = 3,
        yscale: float = 3,
        show: bool = False,
    ):
        plt.ioff()
        self.fig = image_overview(image_sets, xscale, yscale)
        if show:
            plt.show()
        plt.close()

    def save(
        self,
        directory: str = None,
        filename: str = None,
    ):
        """
        Save the generated image overview.

        Parameters:
        - directory (str): The directory to save the image overview.
            If not provided, the current directory will be used.
        - filename (str): The filename of the saved image overview.
            If not provided, a default filename will be used.
        """
        import pathlib

        dir = pathlib.Path(directory)
        dir.mkdir(parents=True, exist_ok=True)
        filepath = dir / filename
        self.fig.savefig(filepath)
