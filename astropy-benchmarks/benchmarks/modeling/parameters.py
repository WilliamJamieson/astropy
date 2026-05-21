"""Parameter and constraint mutation benchmarks for astropy.modeling."""

from astropy.modeling import models

from .common import linspace


class TimeParameterConstraints:
    """Benchmark constraint mutation and post-mutation evaluation."""

    params = [3, 10, 20]
    param_names = ["n_params"]

    def setup(self, n_params):
        self.model = models.Polynomial1D(degree=n_params - 1)
        self.x = linspace(512)
        self.param_names = self.model.param_names

    def time_set_fixed(self, n_params):
        for idx, name in enumerate(self.param_names):
            getattr(self.model, name).fixed = (idx % 2) == 0

    def time_set_bounds(self, n_params):
        for name in self.param_names:
            getattr(self.model, name).bounds = (-1.0, 1.0)

    def time_set_tied(self, n_params):
        # Tie all non-leading parameters to c0 to represent common coupling.
        c0 = self.param_names[0]
        for name in self.param_names[1:]:
            getattr(self.model, name).tied = lambda m, _c0=c0: getattr(m, _c0).value

    def time_eval_after_constraints(self, n_params):
        self.model(self.x)
