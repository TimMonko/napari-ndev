from __future__ import annotations

from typing import TYPE_CHECKING

from magicclass.widgets import (
    ScrollableContainer,
    TabbedContainer,
)
from magicgui.widgets import (
    Label,
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

        self.min_width = 500 # TODO: remove this hardcoded value
        self._viewer = viewer if viewer is not None else None

        self._init_widget_containers()
        self._init_layout()

    def _init_widget_containers(self):
        self._title = Label(label='nDev')

        from napari_ndev import (
            ApocContainer,
            MeasureContainer,
            UtilitiesContainer,
            WorkflowContainer,
        )
        """Initialize the widget containers."""
        self._apoc_container = ApocContainer(viewer=self._viewer)
        self._measure_container = MeasureContainer(viewer=self._viewer)
        self._utilities_container = UtilitiesContainer(viewer=self._viewer)
        self._workflow_container = WorkflowContainer(viewer=self._viewer)

        self._tabbed_container = TabbedContainer(
            # labels=["Apoc", "Measure", "Utilities", "Workflow"],
            labels = True,
            layout="horizontal",
            widgets=[
                self._apoc_container,
                self._measure_container,
                self._utilities_container,
                self._workflow_container,
            ],
        )

    def _init_layout(self):
        """Initialize the layout."""
        self.append(self._title)
        self.append(self._tabbed_container)
        # self.stretch(self._tabbed_container)
