"""Utility-model and custom-subclass benchmarks for astropy.modeling."""

import numpy as np

from astropy.modeling import Fittable1DModel, Model, Parameter, models

from .common import grid, linspace


class CustomStaticModel(Model):
    n_inputs = 1
    n_outputs = 1
    a = Parameter(default=1.0)
    b = Parameter(default=0.5)

    @staticmethod
    def evaluate(x, a, b):
        return a * np.sin(x) + b


class CustomFittableModel(Fittable1DModel):
    a = Parameter(default=1.0)
    b = Parameter(default=0.5)

    @staticmethod
    def evaluate(x, a, b):
        return a * x + b


class TimeUtilityModels:
    def setup(self):
        self.x = linspace(2048)
        self.x2, self.y2 = grid(96, span=5.0)

        points = np.linspace(-5.0, 5.0, 128)
        lookup = np.sin(points)

        self.identity = models.Identity(1)
        self.mapping = models.Mapping((0, 0))
        self.tabular = models.Tabular1D(points=points, lookup_table=lookup)
        self.poly2d = models.Polynomial2D(degree=2)
        self.shift_scale = models.Shift(0.3) | models.Scale(1.1)

    def time_identity_eval(self):
        self.identity(self.x)

    def time_mapping_eval(self):
        self.mapping(self.x)

    def time_tabular_eval(self):
        self.tabular(self.x)

    def time_polynomial2d_eval(self):
        self.poly2d(self.x2, self.y2)

    def time_shift_scale_eval(self):
        self.shift_scale(self.x)


class TimeCustomModelEvaluation:
    params = [64, 2048]
    param_names = ["n"]

    def setup(self, n):
        self.x = linspace(n)
        self.static_model = CustomStaticModel(a=1.2, b=0.3)
        self.fittable_model = CustomFittableModel(a=0.9, b=-0.1)

    def time_custom_static_model(self, n):
        self.static_model(self.x)

    def time_custom_fittable_model(self, n):
        self.fittable_model(self.x)
