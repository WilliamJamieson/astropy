"""
This module is deprecated. Please see `astropy.modeling.models`
"""
from astropy.modeling.models import _spline

from .utils import _module_deprecation

__all__, __getattr__ = _module_deprecation(_spline, __name__)
