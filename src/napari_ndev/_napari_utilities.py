import numpy as np
from magicgui import magic_factory
from napari.layers import Layer
from napari.types import LayerDataTuple


@magic_factory()
def rescale_by(
    layer: Layer,
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
