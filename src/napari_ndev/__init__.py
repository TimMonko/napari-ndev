try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import (
    annotation_saver,
    batch_predict,
    batch_quick_adjustments,
    batch_training,
    batch_workflow,
)

__all__ = [
    "batch_quick_adjustments",
    "annotation_saver",
    "batch_workflow",
    "batch_predict",
    "batch_training",
]
