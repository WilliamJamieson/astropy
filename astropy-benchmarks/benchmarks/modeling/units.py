"""Unit-aware evaluation benchmarks for astropy.modeling."""

import astropy.units as u
from astropy.modeling import models

from .common import linspace


class TimeUnitsGaussian1D:
    params = [64, 2048]
    param_names = ["n"]

    def setup(self, n):
        self.x = linspace(n)
        self.xq = self.x * u.nm

        self.model_unitless = models.Gaussian1D(amplitude=1.0, mean=0.0, stddev=1.2)
        self.model_quantity = models.Gaussian1D(
            amplitude=1.0 * u.Jy,
            mean=0.0 * u.nm,
            stddev=1.2 * u.nm,
        )

    def time_unitless_eval(self, n):
        self.model_unitless(self.x)

    def time_quantity_eval(self, n):
        self.model_quantity(self.xq)
