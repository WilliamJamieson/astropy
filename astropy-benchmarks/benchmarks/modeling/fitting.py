"""Fitting benchmarks for astropy.modeling fitters."""

from astropy.modeling import fitting, models

from .common import linspace, noisy


class TimeLinearFitting:
    params = ([256, 2048], ["linear", "poly3"], [False, True])
    param_names = ["n", "model_kind", "inplace"]

    def setup(self, n, model_kind, inplace):
        self.x = linspace(n)
        if model_kind == "linear":
            truth = models.Linear1D(slope=2.0, intercept=-0.5)
            init = models.Linear1D(slope=1.0, intercept=0.0)
        else:
            truth = models.Polynomial1D(degree=3, c0=0.5, c1=1.2, c2=-0.2, c3=0.03)
            init = models.Polynomial1D(degree=3)
        self.y = noisy(truth(self.x), sigma=0.02, seed=12345)
        self.model = init
        self.fitter = fitting.LinearLSQFitter()
        self.inplace = inplace

    def time_linear_lsq(self, n, model_kind, inplace):
        self.fitter(self.model, self.x, self.y, inplace=self.inplace)


class TimeNonlinearFitting1D:
    params = ([512, 2048], ["trf", "levmar"], ["unconstrained", "bounded"])
    param_names = ["n", "fitter", "constraints"]

    def setup(self, n, fitter, constraints):
        self.x = linspace(n)
        truth = models.Gaussian1D(amplitude=2.0, mean=0.2, stddev=1.1) + models.Const1D(
            0.1
        )
        self.y = noisy(truth(self.x), sigma=0.03, seed=67890)

        self.model = models.Gaussian1D(
            amplitude=1.5, mean=0.0, stddev=1.8
        ) + models.Const1D(0.0)
        if constraints == "bounded":
            self.model.amplitude_0.bounds = (0.5, 5.0)
            self.model.stddev_0.bounds = (0.2, 5.0)

        if fitter == "trf":
            self.fitter = fitting.TRFLSQFitter()
        else:
            self.fitter = fitting.LevMarLSQFitter()

    def time_nonlinear_fit(self, n, fitter, constraints):
        self.fitter(self.model, self.x, self.y, inplace=False)
