"""
This module is deprecated. Please see `astropy.modeling.models`
"""
from astropy.modeling.models import _projections

from .utils import _module_deprecation_getattr, _module_deprecation_warn

_warn = _module_deprecation_warn(__name__)

__all__ = _projections.__all__
__getattr__ = _module_deprecation_getattr(_projections, __name__)
