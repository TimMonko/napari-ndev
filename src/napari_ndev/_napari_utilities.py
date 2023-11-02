import numpy as np
from magicgui import magic_factory
from napari import layers
from napari.types import LayerDataTuple


def init_rescale_by(rescale_by):
    @rescale_by.inherit_from.changed.connect
    @rescale_by.scale_in_z.changed.connect
    def _inherit_from():
        print("changed")
        if rescale_by.scale_in_z.value is False:
            rescale_by.scale_y.value = rescale_by.inherit_from.value.scale[0]
            rescale_by.scale_x.value = rescale_by.inherit_from.value.scale[1]
            print(rescale_by.scale_x.value)
        if rescale_by.scale_in_z.value is True:
            rescale_by.scale_z.value = rescale_by.inherit_from.value.scale[0]
            rescale_by.scale_y.value = rescale_by.inherit_from.value.scale[1]
            rescale_by.scale_x.value = rescale_by.inherit_from.value.scale[2]


@magic_factory(
    widget_init=init_rescale_by,
    scale_x=dict(widget_type="FloatSpinBox", step=0.00000001),
    scale_y=dict(widget_type="FloatSpinBox", step=0.00000001),
    scale_z=dict(widget_type="FloatSpinBox", step=0.00000001),
)
def rescale_by(
    layer: layers.Layer,
    inherit_from: layers.Layer = None,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    scale_z: float = 1.0,
    scale_in_z: bool = False,
) -> LayerDataTuple:
    """
    Rescale an image by a given set of scale factors.

    Parameters
    ----------
    layer : Layer
        selected layer type will be returned in LayerDataTuple
    scale_x : float, optional
        factor by which to scale the image along the x axis.
        The default is 1.0.
    scale_y : float, optional
        factor by which to scale the image along the y dimension.
        The default is 1.0.
    scale_z : float, optional
        factor by which to scale the image along the z dimension.
        The default is 1.0.
    scale_in_z : bool = False
        if True, then scaling is additionally done to Z access
        (not possible with 2D layer inputs)

    Returns
    -------
    LayerDataTuple overwriting original layer with new scale
    """

    if scale_in_z is False:
        scale_factors = np.asarray([scale_y, scale_x])
    elif scale_in_z is True:
        scale_factors = np.asarray([scale_z, scale_y, scale_x])

    return (layer.data, {"name": layer.name, "scale": scale_factors})
