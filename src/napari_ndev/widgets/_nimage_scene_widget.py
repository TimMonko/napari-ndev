# from __future__ import annotations

# from bioio import BioImage
# from magicclass.widgets import ScrollableContainer
# from magicgui.widgets import (
#     CheckBox,
#     Select,
# )

# LABEL_DELIMITER = " :: "


# class NImageSceneWidget(ScrollableContainer):
#     def __init__(self, img: BioImage, img_name: str, **kwargs):
#         super().__init__(**kwargs)
#         self._image = img
#         self._image_name = img_name
#         self._scenes = img.scenes

#         self._init_widgets()
#         self._init_layout()
#         self._connect_events()

#     def _init_widgets(self):
#         self._unpack_channels = CheckBox(
#             label="Unpack Channels as Layers",
#             value=True
#         )
#         self._open_in_memory = CheckBox(
#             label="Open in memory",
#             value=True
#         )
#         self._scene_select = Select(
#             label=self._image_name,
#             choices=self._scenes,
#             allow_multiple=False,
#             tooltip=(
#                 "Select a scene to display. "
#                 "If 'Unpack Channels as Layers' is checked, "
#                 "each channel will be displayed as a separate layer."
#             )
#         )

#     def _init_layout(self):
#         self.extend([
#             self._unpack_channels,
#             self._scene_select,
#         ])

#     def _connect_events(self):
#         self._scene_select.changed.connect(self._on_scene_select)

#     def _on_scene_select(self):
#         scene = self._scene_select.value
#         if scene is None:
#             return
#         self._image.set_scene(scene)
#         # self._image.data

#         # Function to get Metadata to provide with data
#         # def _get_meta(path: "PathLike", data: xr.DataArray, img: AICSImage) -> Dict[str, Any]:
#         # meta: Dict[str, Any] = {}
#         # if DimensionNames.Channel in data.dims:
#         #     # Construct basic metadata
#         #     # Use filename if single scene and no scene name is available
#         #     if len(img.scenes) == 1 and img.current_scene == "Image:0":
#         #         channels_with_scene_index = [
#         #             f"{Path(path).stem}{SCENE_LABEL_DELIMITER}{channel_name}"
#         #             for channel_name in data.coords[DimensionNames.Channel].data.tolist()
#         #         ]
#         #     else:
#         #         channels_with_scene_index = [
#         #             f"{img.current_scene_index}{SCENE_LABEL_DELIMITER}"
#         #             f"{img.current_scene}{SCENE_LABEL_DELIMITER}{channel_name}"
#         #             for channel_name in data.coords[DimensionNames.Channel].data.tolist()
#         #         ]
#         #     meta["name"] = channels_with_scene_index
#         #     meta["channel_axis"] = data.dims.index(DimensionNames.Channel)

#     # def _open_scene(self):
