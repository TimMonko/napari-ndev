from __future__ import annotations

from typing import TYPE_CHECKING

from magicclass.widgets import (
    ScrollableContainer,
    TabbedContainer,
)
from magicgui.widgets import (
    Container,
    Label,
    PushButton,
)

from napari_ndev import __version__

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
        # self._logo = Image(value=r'docs\resources\images\ndev-logo.png')
        # self._logo.scale_widget_to_image_size()
        # _logo_path = r'docs\resources\images\ndev-logo.png'
        _logo_path = r'docs\resources\images\neuralDev-logo.svg'
        self._logo_label = Label(
            value='<h1 style="text-align: center;">'
            f'<img src="{_logo_path}" style="width: 20%; max-width: 50px;">'
            '</h1>'
        )
        self._version_label = Label(value=f'v{__version__}')


        self._docs_link_button = PushButton(
            text='Docs',
            icon='ic:round-menu-book'
        )
        self._bug_report_link_button = PushButton(
            text='Bug Report',
            icon='ic:outline-bug-report',
        )
        self._link_container = Container(
            widgets=[self._docs_link_button, self._bug_report_link_button],
            layout='horizontal',
        )

        self._init_widget_containers()
        self._init_layout()
        self._init_callbacks()

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
        self.append(self._logo_label)
        self.append(self._version_label)
        self.append(self._link_container)
        self.append(self._tabbed_container)
        # self.stretch(self._tabbed_container)

    def _init_callbacks(self):
        """Initialize the widget callbacks."""
        self._docs_link_button.clicked.connect(self._open_docs_link)
        self._bug_report_link_button.clicked.connect(self._open_bug_report_link)

    def _open_docs_link(self):
        """Open the documentation link."""
        import webbrowser
        webbrowser.open('https://timmonko.github.io/napari-ndev')

    def _open_bug_report_link(self):
        """Open the bug report link."""
        import webbrowser
        webbrowser.open('https://github.com/TimMonko/napari-ndev/issues')

if __name__ == "__main__":
    import napari
    viewer = napari.Viewer()
    viewer.window.add_dock_widget(MainContainer(viewer=viewer))
    napari.run()
