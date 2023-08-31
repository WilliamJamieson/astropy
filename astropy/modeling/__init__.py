# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This subpackage provides a framework for representing models and
performing model evaluation and fitting. It supports 1D and 2D models
and fitting with parameter constraints. It has some predefined models
and fitting routines.
"""

from . import fitting, models
from .core import *
from .parameters import *
from .separable import *

_OLD_MODULES = [
    "functional_models",
    "mappings",
    "math_functions",
    "physical_models",
    "polynomial",
    "powerlaws",
    "projections",
    "rotations",
    "spline",
    "tabular",
]


def _patch_module():
    import importlib
    import sys
    from builtins import getattr as old_getattr

    def new_getattr(name):
        if name in _OLD_MODULES:
            return importlib.import_module(f"astropy.modeling.{name}")

        return old_getattr(sys.models[__name__], name)

    sys.modules[__name__].__getattr__ = new_getattr


_patch_module()
