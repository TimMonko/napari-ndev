try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import (
    batch_annotator,
    batch_predict,
    batch_training,
    batch_workflow,
)

__all__ = [
    "batch_annotator",
    "batch_workflow",
    "batch_training",
    "batch_predict",
]
