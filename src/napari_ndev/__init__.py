try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from . import helpers
from ._apoc_container import ApocContainer
from ._apoc_feature_stack import ApocFeatureStack
from ._plate_mapper import PlateMapper
from ._rescale_by import RescaleBy

# Lazy load in widgets in order to speed up import time
# This package does not typically use the widgets directly, so this should not
# be too confusing for the end user. The goal is to allow directly calling the
# widgets without having to know the underlying package structure.


def __getattr__(name):
    module_map = {
        "UtilitiesContainer": ("._utilities_container", "UtilitiesContainer"),
        "WorkflowContainer": ("._workflow_container", "WorkflowContainer"),
    }

    if name in module_map:
        module_path, class_name = module_map[name]
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "WorkflowContainer",
    "UtilitiesContainer",
    "ApocContainer",
    "ApocFeatureStack",
    "RescaleBy",
    "PlateMapper",
    "helpers",
]
