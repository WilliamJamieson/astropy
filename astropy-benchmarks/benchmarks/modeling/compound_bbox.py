"""Compound bounding-box benchmarks for astropy.modeling."""

from astropy.modeling import bind_compound_bounding_box, models

from .common import linspace


class TimeCompoundBoundingBox:
    """Benchmark selector-based compound bounding-box evaluation."""

    params = [512, 2048]
    param_names = ["n"]

    def setup(self, n):
        self.model = models.Gaussian1D(amplitude=1.0, mean=0.0, stddev=0.4)
        self.x = linspace(n, start=-1.5, stop=1.5)

        # Two selectable domains matching the pattern used in core tests.
        bboxes = {
            0: (-1.0, 0.0),
            1: (0.0, 1.0),
        }
        bind_compound_bounding_box(self.model, bboxes, [("x", False)])

    def time_eval_selector_0(self, n):
        self.model(self.x, with_bounding_box=0)

    def time_eval_selector_1(self, n):
        self.model(self.x, with_bounding_box=1)

    def time_eval_implicit_selector(self, n):
        self.model(self.x, with_bounding_box=True)
