from typing import TYPE_CHECKING

from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSpinBox,
    PushButton,
)

if TYPE_CHECKING:
    import napari


class RescaleBy(Container):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
    ):
        super().__init__()
        self
        self._viewer = viewer
        self._scale_x = FloatSpinBox(
            value=1.0, step=0.00000001, label="Scale X"
        )
        self._scale_y = FloatSpinBox(
            value=1.0,
            step=0.00000001,
            label="Scale Y",
        )
        self._scale_z = FloatSpinBox(
            value=1.0,
            step=0.00000001,
            label="Scale Z",
        )

        def current_layers(_):
            return [x for x in self._viewer.layers]

        self._layer_to_scale = ComboBox(
            choices=current_layers, label="Layer to Scale", nullable=False
        )
        self._inherit_from_layer = ComboBox(
            choices=current_layers,
            label="Inherit Scale From Layer",
            nullable=True,
        )

        self._scale_in_z = CheckBox(label="Scale in Z")
        self._rescale_by_button = PushButton(label="Rescale Layer")

        self.extend(
            [
                self._layer_to_scale,
                self._inherit_from_layer,
                self._scale_z,
                self._scale_y,
                self._scale_x,
                self._scale_in_z,
                self._rescale_by_button,
            ]
        )

        self._inherit_from_layer.changed.connect(self._inherit_from)
        self._rescale_by_button.clicked.connect(self.rescale_by)

    def _inherit_from(self):
        scale = self._inherit_from_layer.value.scale
        self._scale_y.value = scale[-2]
        self._scale_x.value = scale[-1]

        if len(scale) >= 3:
            self._scale_z.value = scale[-3]

    def rescale_by(self):
        if self._scale_in_z.value is False:
            scale_factors = (self._scale_y.value, self._scale_x.value)
        elif self._scale_in_z.value is True:
            scale_factors = (
                self._scale_z.value,
                self._scale_y.value,
                self._scale_x.value,
            )

        self._layer_to_scale.value.scale = scale_factors
