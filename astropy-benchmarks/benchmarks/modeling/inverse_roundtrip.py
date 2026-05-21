"""Inverse-transform benchmarks for astropy.modeling."""

from astropy.modeling import models

from .common import linspace


class TimeInverseRoundTrip1D:
    params = [64, 2048]
    param_names = ["n"]

    def setup(self, n):
        self.x = linspace(n)
        self.model = models.Shift(0.3) | models.Scale(1.05)
        self.y = self.model(self.x)

    def time_forward_only(self, n):
        self.model(self.x)

    def time_inverse_only(self, n):
        self.model.inverse(self.y)

    def time_forward_inverse_roundtrip(self, n):
        self.model.inverse(self.model(self.x))
