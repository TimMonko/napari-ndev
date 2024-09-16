"""
Functions for processing label morphology.

Process labels with various functions using pyclesperanto and scikit-image.
Intended to be compatible with workflow.yaml files, and will be incorporated
into napari-workflows and napari-assistant in the future. Should accept both
OCLArray and other ArrayLike types.

Functions
---------
skeletonize_labels : Create skeletons with label identities from a label image.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyclesperanto_prototype as cle

if TYPE_CHECKING:
    from bioio_base.types import ArrayLike

__all__ = [
    'skeletonize_labels',
    'connect_breaks_between_labels',
    'label_voronoi_based_on_intensity'
]

def skeletonize_labels(label: ArrayLike) -> np.ndarray:
    """
    Create skeletons and maintains label identities from a label image.

    Parameters
    ----------
    label : ArrayLike
        Label image.

    Returns
    -------
    np.ndarray
        Skeletonized label image.

    """
    from skimage.morphology import skeletonize

    skeleton = skeletonize(cle.pull(label))
    return (label * skeleton).astype(np.uint16)

def connect_breaks_between_labels(label: ArrayLike, connect_distance: float) -> ArrayLike: # pragma: no cover
    """
    Connect breaks between labels in a label image.

    Return the input label image with new label identities connecting breaks
    between the original labels. The new labels have the original label
    dimensions, so this is intended to keep the overall morphology the same,
    just with new labels connecting the original labels if under the specified
    distance.

    Parameters
    ----------
    label : ArrayLike
        Label image.
    connect_distance : float
        Maximum distance to connect labels, in pixels.

    Returns
    -------
    ArrayLike
        Label image with new labels connecting breaks between original labels.

    """
    label_dilated = cle.dilate_labels(label, radius=connect_distance/2)
    label_merged = cle.merge_touching_labels(label_dilated)
    # relabel original labels based on the merged labels
    return (label_merged * (label > 0)).astype(np.uint16)

def label_voronoi_based_on_intensity(label: ArrayLike, intensity_image: ArrayLike) -> ArrayLike: # pragma: no cover
    """
    Create a voronoi label masks of labels based on an intensity image.

    Return a label image with Voronoi regions based on the intensity image.
    The intensity image should be the same shape as the label image, and the
    labels will be assigned to the Voronoi regions based on the intensity
    values.

    Parameters
    ----------
    label : ArrayLike
        Label image.
    intensity_image : ArrayLike
        Intensity image.

    Returns
    -------
    ArrayLike
        Label image with Voronoi regions based on the intensity image.

    """
    label_binary = cle.greater_constant(label, constant=0) # binarize
    intensity_blur = cle.gaussian_blur(intensity_image, sigma_x=1, sigma_y=1)
    intensity_peaks = cle.detect_maxima_box(intensity_blur, radius_x=0, radius_y=0)
    select_peaks_on_binary = cle.binary_and(intensity_peaks, label_binary)
    return cle.masked_voronoi_labeling(select_peaks_on_binary, label_binary)
