"""Bounding-box benchmarks for astropy.modeling."""

import numpy as np

from astropy.modeling import bind_bounding_box, models

from .common import grid


class TimeBoundingBoxEvaluation:
    """Measure with/without bounding-box evaluation overhead."""

    params = [16.0, 32.0, 64.0]
    param_names = ["span"]

    def setup(self, span):
        self.x, self.y = grid(192, span=span)
        base = models.Gaussian2D(
            amplitude=1.0,
            x_mean=0.0,
            y_mean=0.0,
            x_stddev=2.5,
            y_stddev=2.5,
            theta=0.0,
        )
        # Valid region is fixed while sampled grid expands.
        self.model = bind_bounding_box(base, ((-16.0, 16.0), (-16.0, 16.0)))

    def time_eval_no_bbox(self, span):
        self.model(self.x, self.y)

    def time_eval_with_bbox(self, span):
        self.model(self.x, self.y, with_bounding_box=True, fill_value=np.nan)
