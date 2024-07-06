import matplotlib.pyplot as plt


class ImageOverview:
    def __init__(
        self,
        image_sets: dict,
        xscale: float = 3,
        yscale: float = 3,
        show: bool = False,
    ):
        self.image_sets = image_sets
        self.xscale = xscale
        self.yscale = yscale
        plt.ioff()
        self.fig, self.axs = self._construct_overview()
        if show:
            plt.show()
        plt.close()

    def _construct_overview(self):
        import stackview

        # plt.ioff()
        # create the subplot grid
        num_rows = len(self.image_sets)
        num_columns = max(
            [len(image_set["image"]) for image_set in self.image_sets]
        )
        fig, axs = plt.subplots(
            num_rows,
            num_columns,
            figsize=(num_columns * self.xscale, num_rows * self.yscale),
        )

        if num_rows == 1:
            axs = [axs]
        if num_columns == 1:
            axs = [[ax] for ax in axs]

        # iterate through the image sets
        for row, image_set in enumerate(self.image_sets):
            # print(image_set)
            # for col, image in enumerate(image_set):
            for col in range(len(image_set["image"])):
                # create a dictionary from the col-th values of each key
                image_dict = {
                    key: value[col] for key, value in image_set.items()
                }

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

        return fig, axs

    def save(
        self,
        directory: str = None,
        filename: str = None,
    ):
        import pathlib

        dir = pathlib.Path(directory)
        dir.mkdir(parents=True, exist_ok=True)
        filepath = dir / filename
        self.fig.savefig(filepath)
