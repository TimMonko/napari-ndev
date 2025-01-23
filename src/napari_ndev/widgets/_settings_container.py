from magicgui.widgets import CheckBox, ComboBox, Container

from napari_ndev._settings import get_settings


class SettingsContainer(Container):
    def __init__(self):
        super().__init__()
        self.settings = get_settings()

        self._init_widgets()

    def _init_widgets(self):
        self._scene_handling_combo = ComboBox(
            label='Multi-Scene Handling',
            value = self.settings.SCENE_HANDLING,
            choices=[
                'Open Scene Widget',
                'View All Scenes',
                'View First Scene Only',
            ],
            tooltip='How to handle files with multiple-scenes, by default \n'
            'opens a widget to select the scenes to open. \n'
            'If "View All Scenes" then the scenes will be added as a slider \n'
            'dimension in the viewer.'
        )
        self._clear_on_scene_select_checkbox = CheckBox(
            value=self.settings.CLEAR_LAYERS_ON_NEW_SCENE,
            label='Clear All Layers On New Scene Selection',
            tooltip='Whether to clear the viewer when selecting a new scene.',
        )
        self._unpack_channels_as_layers_checkbox = CheckBox(
            value=self.settings.UNPACK_CHANNELS_AS_LAYERS,
            label='Unpack Channels as Layers',
            tooltip='Whether to unpack channels as layers.',
        )
        self._bioio_settings_container = Container(
            widgets=[
                self._scene_handling_combo,
                self._clear_on_scene_select_checkbox,
                self._unpack_channels_as_layers_checkbox,
            ],
            layout='vertical',
        )

        self.extend([
            self._bioio_settings_container,
        ])

        self._scene_handling_combo.changed.connect(self._update_settings)
        self._clear_on_scene_select_checkbox.changed.connect(self._update_settings)
        self._unpack_channels_as_layers_checkbox.changed.connect(self._update_settings)

    def _update_settings(self):
        widget_to_setting = {
            'SCENE_HANDLING': self._scene_handling_combo,
            'CLEAR_LAYERS_ON_NEW_SCENE': self._clear_on_scene_select_checkbox,
            'UNPACK_CHANNELS_AS_LAYERS': self._unpack_channels_as_layers_checkbox,
        }

        for setting_name, widget in widget_to_setting.items():
            setattr(self.settings, setting_name, widget.value)

        self.settings.save_settings()
