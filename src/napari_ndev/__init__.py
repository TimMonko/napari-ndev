try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from importlib import import_module


class LazyImport:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        if self.module is None:
            self.module = import_module(self.module_name)
        return getattr(self.module, name)


ApocContainer = LazyImport('._apoc_container')
ApocFeatureStack = LazyImport('._apoc_feature_stack')
PlateMapper = LazyImport('._plate_mapper')
RescaleBy = LazyImport('._rescale_by')
UtilitiesContainer = LazyImport('._utilities_container')
WorkflowContainer = LazyImport('._workflow_container')

__all__ = [
    "WorkflowContainer",
    "UtilitiesContainer",
    "ApocContainer",
    "ApocFeatureStack",
    "RescaleBy",
    "PlateMapper",
]
