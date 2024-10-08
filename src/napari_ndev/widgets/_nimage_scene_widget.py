from __future__ import annotations

from bioio import BioImage
from magicclass.widgets import ScrollableContainer
from magicgui.widgets import (
    CheckBox,
    Select,
)

LABEL_DELIMITER = " || "


class NImageSceneWidget(ScrollableContainer):
    def __init__(self, image: BioImage, image_name: str, **kwargs):
        super().__init__(**kwargs)
        self._image = image
        self._image_name = image_name
        self._scenes = image.scenes

        self._init_widgets()
        self._init_layout()
        self._connect_events()

    def _init_widgets(self):
        self._unpack_channels = CheckBox(
            label="Unpack Channels as Layers",
            value=True
        )
        self._open_in_memory = CheckBox(
            label="Open in memory",
            value=True
        )
        self._scene_select = Select(
            label=self._image_name,
            choices=self._scenes,
            allow_multiple=False,
            tooltip=(
                "Select a scene to display. "
                "If 'Unpack Channels as Layers' is checked, "
                "each channel will be displayed as a separate layer."
            )
        )

    def _init_layout(self):
        self.extend([
            self._unpack_channels,
            self._scene_select,
        ])

    def _connect_events(self):
        self._scene_select.changed.connect(self._on_scene_select)

    def _on_scene_select(self):
        scene = self._scene_select.value
        if scene is None:
            return
        self._image.set_scene(scene)
        # self._image.data

    # def _open_scene(self):
