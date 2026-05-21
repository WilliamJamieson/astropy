"""Compound-model construction and evaluation benchmarks."""

from astropy.modeling import models

from .common import grid, linspace


def build_chain(depth):
    model = models.Identity(1)
    for _ in range(depth):
        model = model | models.Shift(0.1) | models.Scale(1.001)
    return model


class TimeCompoundConstruction:
    params = [1, 5, 10, 20]
    param_names = ["depth"]

    def time_construct_chain(self, depth):
        build_chain(depth)


class TimeCompoundEvaluation1D:
    params = [1, 5, 10, 20]
    param_names = ["depth"]

    def setup(self, depth):
        self.model = build_chain(depth)
        self.x = linspace(2048)

    def time_eval_chain(self, depth):
        self.model(self.x)


class TimeCompoundParallelFanout:
    params = [2, 4, 8]
    param_names = ["branches"]

    def setup(self, branches):
        left = models.Identity(1)
        for _ in range(branches - 1):
            left = left & models.Identity(1)
        self.model = left | models.Mapping(tuple(range(branches)))
        xy = grid(96, span=3.0)
        # Build one input vector per branch.
        self.inputs = tuple(xy[0].ravel() for _ in range(branches))

    def time_eval_parallel(self, branches):
        self.model(*self.inputs)
