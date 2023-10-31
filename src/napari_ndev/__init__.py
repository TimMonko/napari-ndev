try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._napari_utilities import annotation_saver, rescale_by
from ._widget import (
    batch_predict,
    batch_training,
    batch_utilities,
    batch_workflow,
)

__all__ = [
    "batch_utilities",
    "annotation_saver",
    "batch_workflow",
    "batch_predict",
    "batch_training",
    "rescale_by",
]
