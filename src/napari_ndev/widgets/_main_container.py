from __future__ import annotations

from typing import TYPE_CHECKING

from magicclass.widgets import (
    ScrollableContainer,
    TabbedContainer,
)
from magicgui.widgets import (
    Label,
    Image,
)

if TYPE_CHECKING:
    import napari

class MainContainer(ScrollableContainer):
    """
    A widget container to container the primary ndev widgets.

    Parameters
    ----------
    viewer: napari.viewer.Viewer, optional
        The napari viewer instance.

    """

    def __init__(self, viewer: napari.viewer.Viewer = None):
        """
        Initialize the MainContainer.

        Parameters
        ----------
        viewer: napari.viewer.Viewer, optional
            The napari viewer instance.

        """
        super().__init__(labels=False)

        self.min_width = 700 # TODO: remove this hardcoded value
        self._viewer = viewer if viewer is not None else None

        # TODO: get image to display. Works outside of napari
        self._logo = Image(value=r"docs\resources\images\ndev-logo.png")
        self._logo.scale_widget_to_image_size()

        self._link_label = Label(
            value=(
                '<p style="color: white;">' # doesn't appear to do anything
                '<a href="https://timmonko.github.io/napari-ndev"'
                'style="color: white;">'
                'nDev documentation</a></p>'
            )
        )
        self._link_label.native.setOpenExternalLinks(True)

        self._init_widget_containers()
        self._init_layout()

    def _init_widget_containers(self):
        from napari_ndev import (
            ApocContainer,
            MeasureContainer,
            UtilitiesContainer,
            WorkflowContainer,
        )
        """Initialize the widget containers."""
        self._apoc_container = ApocContainer(viewer=self._viewer)
        self._apoc_container.label = "APOC"
        self._measure_container = MeasureContainer(viewer=self._viewer)
        self._measure_container.label = "Measure"
        self._utilities_container = UtilitiesContainer(viewer=self._viewer)
        self._utilities_container.label = "Utilities"
        self._workflow_container = WorkflowContainer(viewer=self._viewer)
        self._workflow_container.label = "Workflow"

        self._tabbed_container = TabbedContainer(
            # labels=["Apoc", "Measure", "Utilities", "Workflow"],
            labels = False,
            layout="horizontal",
            widgets=[
                self._utilities_container,
                self._apoc_container,
                self._workflow_container,
                self._measure_container,
            ],
        )

    def _init_layout(self):
        """Initialize the layout."""
        self.append(self._logo)
        self.append(self._link_label)
        self.append(self._tabbed_container)
        # self.stretch(self._tabbed_container)
