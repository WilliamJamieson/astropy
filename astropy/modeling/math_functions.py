"""
This module is deprecated. Please see `astropy.modeling.models`
"""
from astropy.modeling.models import _math_functions

from .utils import _module_deprecation

__all__, __getattr__ = _module_deprecation(_math_functions, __name__)
