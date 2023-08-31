"""
This module is deprecated. Please see `astropy.modeling.models`
"""
from astropy.modeling.models import _physical_models

from .utils import _module_deprecation

__all__, __getattr__ = _module_deprecation(_physical_models, __name__)
