"""Evaluation-path benchmarks for astropy.modeling."""

from astropy.modeling import models

from .common import grid, linspace


class TimeEval1D:
    """Benchmark scalar/small/medium 1D model evaluation."""

    params = [1, 64, 2048]
    param_names = ["n"]

    def setup(self, n):
        self.x = linspace(n)
        self.gaussian = models.Gaussian1D(amplitude=1.0, mean=0.0, stddev=1.5)
        self.poly = models.Polynomial1D(degree=3, c0=1.0, c1=0.2, c2=-0.05, c3=0.01)

    def time_gaussian_call(self, n):
        self.gaussian(self.x)

    def time_gaussian_evaluate(self, n):
        self.gaussian.evaluate(
            self.x,
            self.gaussian.amplitude.value,
            self.gaussian.mean.value,
            self.gaussian.stddev.value,
        )

    def time_poly_call(self, n):
        self.poly(self.x)


class TimeEval2D:
    """Benchmark 2D grid model evaluation."""

    params = [64, 192]
    param_names = ["shape"]

    def setup(self, shape):
        self.x, self.y = grid(shape, span=10.0)
        self.gaussian = models.Gaussian2D(
            amplitude=1.0,
            x_mean=0.0,
            y_mean=0.0,
            x_stddev=1.5,
            y_stddev=2.0,
            theta=0.3,
        )
        self.poly = models.Polynomial2D(
            degree=2,
            c0_0=1.0,
            c1_0=0.1,
            c0_1=-0.1,
            c2_0=0.01,
            c1_1=0.02,
            c0_2=-0.015,
        )

    def time_gaussian2d_call(self, shape):
        self.gaussian(self.x, self.y)

    def time_polynomial2d_call(self, shape):
        self.poly(self.x, self.y)
