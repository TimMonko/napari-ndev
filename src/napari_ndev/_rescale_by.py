from typing import TYPE_CHECKING

from magicgui.widgets import ComboBox, Container, PushButton, TupleEdit

if TYPE_CHECKING:
    import napari


class RescaleBy(Container):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer" = None,
    ):
        super().__init__()
        ##############################
        # Attributes
        ##############################
        self
        self._viewer = viewer if viewer is not None else None
        ##############################
        # Widgets
        ##############################
        self._scale_tuple = TupleEdit(value=(0.0, 1.0, 1.0), label="Scale ZYX")

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

        self._rescale_by_button = PushButton(label="Rescale Layer")

        self.extend(
            [
                self._scale_tuple,
                self._layer_to_scale,
                self._inherit_from_layer,
                self._rescale_by_button,
            ]
        )
        ##############################
        # Event Handling
        ##############################
        self._inherit_from_layer.changed.connect(self._inherit_from)
        self._rescale_by_button.clicked.connect(self.rescale_by)

    ##############################
    # Methods
    ##############################

    def _inherit_from(self):
        scale = self._inherit_from_layer.value.scale
        # Directly create a new tuple with the desired order
        self._scale_tuple.value = (
            scale[-3] if len(scale) >= 3 else 0.0,
            scale[-2],
            scale[-1],
        )

    def rescale_by(self):
        scale_tup = self._scale_tuple.value
        scale_len = len(self._layer_to_scale.value.scale)

        self._layer_to_scale.value.scale = (
            scale_tup[1:3] if scale_len == 2 else scale_tup
        )
