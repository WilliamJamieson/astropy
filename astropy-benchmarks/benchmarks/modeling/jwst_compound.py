"""JWST assign_wcs/assign_mtwcs-style compound model benchmarks.

These benchmarks intentionally construct large compound model graphs with
`Mapping`, `Polynomial2D`, and shift/scale stages, which mirror common
structure patterns in JWST WCS steps.
"""

import copy

import numpy as np

from astropy.modeling import bind_bounding_box, models

from .common import grid


def _poly2d_pair(degree):
    # Two separate 2D distortion polynomials, one for x and one for y.
    poly_x = models.Polynomial2D(
        degree=degree,
        c0_0=0.0,
        c1_0=1.0,
        c0_1=0.02,
        c2_0=1e-5,
        c1_1=2e-5,
        c0_2=-1e-5,
    )
    poly_y = models.Polynomial2D(
        degree=degree,
        c0_0=0.0,
        c1_0=-0.01,
        c0_1=1.0,
        c2_0=-1e-5,
        c1_1=1e-5,
        c0_2=2e-5,
    )
    return poly_x, poly_y


def build_assign_wcs_like_model(stages, degree=2):
    """Build a large 2D compound graph similar to assign_wcs transform chains."""
    model = models.Identity(1) & models.Identity(1)
    for _ in range(stages):
        poly_x, poly_y = _poly2d_pair(degree)
        stage = (
            models.Mapping((0, 1, 0, 1))
            | (poly_x & poly_y)
            | (models.Shift(0.03) & models.Shift(-0.02))
            | (models.Scale(1.0002) & models.Scale(0.9998))
        )
        model = model | stage
    return model


def build_assign_mtwcs_like_model(stages, degree=2):
    """Build a large 3-input/3-output chain inspired by assign_mtwcs usage."""
    model = models.Identity(1) & models.Identity(1) & models.Identity(1)
    for _ in range(stages):
        poly_x, poly_y = _poly2d_pair(degree)
        stage = (
            models.Mapping((0, 1, 0, 1, 2))
            | (poly_x & poly_y & models.Identity(1))
            | (models.Shift(0.02) & models.Shift(-0.015) & models.Shift(0.001))
            | (models.Scale(1.0001) & models.Scale(0.9999) & models.Scale(1.0))
        )
        model = model | stage
    return model


class TimeJWSTAssignWCSCompound:
    """assign_wcs-style compound graphs with optional bounding box."""

    params = ([4, 8, 12], [False, True])
    param_names = ["stages", "use_bounding_box"]

    def setup(self, stages, use_bounding_box):
        self.model = build_assign_wcs_like_model(stages=stages, degree=2)
        if use_bounding_box:
            self.model = bind_bounding_box(self.model, ((-32.0, 32.0), (-32.0, 32.0)))

        self.x, self.y = grid(96, span=48.0)

    def time_evaluate(self, stages, use_bounding_box):
        if use_bounding_box:
            self.model(self.x, self.y, with_bounding_box=True, fill_value=np.nan)
        else:
            self.model(self.x, self.y)

    def time_copy(self, stages, use_bounding_box):
        self.model.copy()

    def time_deepcopy(self, stages, use_bounding_box):
        copy.deepcopy(self.model)


class TimeJWSTAssignMTWCSCompound:
    """assign_mtwcs-style large compound graphs including copy costs."""

    params = [4, 8, 12]
    param_names = ["stages"]

    def setup(self, stages):
        self.model = build_assign_mtwcs_like_model(stages=stages, degree=2)

        self.x, self.y = grid(80, span=24.0)
        self.lam = np.full_like(self.x, 1.0)

    def time_evaluate(self, stages):
        self.model(self.x, self.y, self.lam)

    def time_copy(self, stages):
        self.model.copy()

    def time_deepcopy(self, stages):
        copy.deepcopy(self.model)
