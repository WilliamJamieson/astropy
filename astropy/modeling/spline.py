# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Spline models and fitters."""
# pylint: disable=line-too-long, too-many-lines, too-many-arguments, invalid-name

import warnings

import abc

from astropy.utils.exceptions import (AstropyUserWarning,)
from .core import (Fittable1DModel, Fittable2DModel, ModelDefinitionError)

from .fitting import _FitterMeta


class _Spline(abc.ABC):
    """
    Abstract Class for Splines

    Parameters
    ----------
    optional_inputs: dict
        dictionary of the form
            optional input name: default value of input
    """

    optional_inputs = {}

    @property
    def spline(self):
        return self._spline

    @spline.setter
    def spline(self, value):
        if self._spline is not None:
            warnings.warn("Spline already defined for this model, you are overriding it.",
                          AstropyUserWarning)
        self._spline = value

    def reset_spline(self):
        self._spline = None

    @property
    def bbox(self):
        try:
            return self.bounding_box.bbox
        except NotImplementedError:
            return [None]*(2*self.n_inputs)

    @staticmethod
    def _optional_arg(arg):
        return f'_{arg}'

    def _create_optional_inputs(self):
        for arg in self.optional_inputs:
            attribute = self._optional_arg(arg)
            if hasattr(self, attribute):
                raise ValueError(f'Optional argument {arg} already exists in this class!')
            else:
                setattr(self, attribute, None)

    def _intercept_optional_inputs(self, **kwargs):
        new_kwargs = kwargs
        for arg in self.optional_inputs:
            if (arg in kwargs):
                attribute = self._optional_arg(arg)
                if getattr(self, attribute) is None:
                    setattr(self, attribute, kwargs[arg])
                    del new_kwargs[arg]
                else:
                    raise RuntimeError(f'{arg} has already been set, something has gone wrong!')

        return new_kwargs

    def _get_optional_inputs(self, **kwargs):
        optional_inputs = kwargs
        for arg in self.optional_inputs:
            attribute = self._optional_arg(arg)

            if arg in kwargs:
                # Options passed in
                optional_inputs[arg] = kwargs[arg]
            elif getattr(self, attribute) is not None:
                # No options passed in and Options set
                optional_inputs[arg] = getattr(self, attribute)
                setattr(self, attribute, None)
            else:
                # No options passed in and No options set
                optional_inputs[arg] = self.optional_inputs[arg]

        return optional_inputs


class Spline1D(Fittable1DModel, _Spline):
    """
    One dimensional Spline model

    Parameters
    ----------
    k : int, optional
        Degree of the smoothing spline. Must be 1 <= k <= 5. Default is
        k = 3, a cubic spline.
    ext : int or str, optional
        Controls the extrapolation mode for elements not in the interval
        defined by the knot sequence.

        if ext=0 or ‘extrapolate’, return the extrapolated value.

        if ext=1 or ‘zeros’, return 0

        if ext=2 or ‘raise’, raise a ValueError

        if ext=3 of ‘const’, return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite
        numbers. Disabling may give a performance gain, but may result
        in problems (crashes, non-termination or non-sensical results)
        if the inputs do contain infinities or NaNs. Default is False.

    Notes
    -----
    This is largely a wrapper of the `scipy.interpolate.UnivariateSpline`
    class and its child `scipy.interpolate.LSQUnivariateSpline`. See
    scipy.interpolate for more details.
    """

    optional_inputs = {'nu': 0}

    def __init__(self, k=3, ext=0, check_finite=False, n_models=None,
                 model_set_axis=None, name=None, meta=None, **params):
        self._k = k
        self._ext = ext
        self._check_finite = check_finite
        self._spline = None
        super().__init__(n_models=n_models, model_set_axis=model_set_axis,
                         name=name, meta=meta, **params)

        # Hack to allow an optional model argument
        self._create_optional_inputs()

    def evaluate(self, x, **kwargs):
        """
        Evaluate the model

        Parameters
        ----------
        x : array_like
            A 1-D array of points at which to return the value of the smoothed
            spline or its derivatives. Note: `x` can be unordered but the
            evaluation is more efficient if `x` is (partially) ordered.
        nu : int
            The order of derivative of the spline to compute.
        """

        # Hack to allow an optional model argument
        kwargs = self._get_optional_inputs(**kwargs)

        return self.spline(x, **kwargs)

    def __call__(self, *args, **kwargs):

        # Hack to allow an optional model argument
        kwargs = self._intercept_optional_inputs(**kwargs)

        return super().__call__(*args, **kwargs)

    def fit_spline(self, x, y, w=None, s=None):
        """
        Fit spline using `scipy.interpolate.UnivariateSpline`

        Parameters
        ----------
        x : (N,) array_like
            1-D array of independent input data. Must be increasing;
            must be strictly increasing if s is 0.
        y : (N,) array_like
            1-D array of dependent input data, of the same length as x.
        w : (N,) array_like, optional
            Weights for spline fitting. Must be positive. If None
            (default), weights are all equal.
        s : float or None, optional
            Positive smoothing factor used to choose the number of knots.
            Number of knots will be increased until the smoothing
            condition is satisfied:
            ```
            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
            ```
            If None (default), s = len(w) which should be a good value
            if 1/w[i] is an estimate of the standard deviation of y[i].
            If 0, spline will interpolate through all data points.
        """

        from scipy.interpolate import UnivariateSpline

        self.spline = UnivariateSpline(x, y, w=w, bbox=self.bbox,
                                       k=self._k, s=s, ext=self._ext,
                                       check_finite=self._check_finite)

    def fit_LSQ_spline(self, x, y, t, w=None):
        """
        Fit spline using `scipy.interpolate.LSQUnivariateSpline`

        Parameters
        ----------
        x : (N,) array_like
            1-D array of independent input data. Must be increasing;
            must be strictly increasing if s is 0.
        y : (N,) array_like
            1-D array of dependent input data, of the same length as x.
        t(M,) : array_like
            interior knots of the spline. Must be in ascending order and:
            ```
            bbox[0] < t[0] < ... < t[-1] < bbox[-1]
            ```
        w : (N,) array_like, optional
            Weights for spline fitting. Must be positive. If None
            (default), weights are all equal.
        """

        from scipy.interpolate import LSQUnivariateSpline

        self.spline = LSQUnivariateSpline(x, y, t, w=w, bbox=self.bbox,
                                          k=self._k, ext=self._ext,
                                          check_finite=self._check_finite)


class Spline2D(Fittable2DModel, _Spline):
    """
    Two dimensional Spline model

    Parameters
    ----------
    kx, ky : ints, optional
        Degrees of the bivariate spline. Default is 3.
    eps : float, optional
        A threshold for determining the effective rank of an
        over-determined linear system of equations. eps should have a
        value within the open interval (0, 1), the default is 1e-16.

    Notes
    -----
    This is largely a wrapper of the `scipy.interpolate.SmoothBivariateSpline`
    class and its child `scipy.interpolate.LSQBivariateSpline`. See
    scipy.interpolate for more details.
    """

    optional_inputs = {'dx': 0,
                       'dy': 0}

    def __init__(self, kx=3, ky=3, eps=1e-16, n_models=None,
                 model_set_axis=None, name=None, meta=None, **params):
        self._kx = kx
        self._ky = ky
        self._eps = eps
        self._spline = None
        super().__init__(n_models=n_models, model_set_axis=model_set_axis,
                         name=name, meta=meta, **params)

        # Hack to allow an optional model argument
        self._create_optional_inputs()

    def evaluate(self, x, y, **kwargs):
        """
        Evaluate the model

        Parameters
        ----------
        x, y : array_like
            Input coordinates. The arrays must be sorted to increasing order.
        dx : int
            Order of x-derivative
        dy : int
            Order of y-derivative
        """

        # Hack to allow an optional model argument
        kwargs = self._get_optional_inputs(**kwargs)

        return self.spline(x, y, **kwargs)

    def __call__(self, *args, **kwargs):

        # Hack to allow an optional model argument
        kwargs = self._intercept_optional_inputs(**kwargs)

        return super().__call__(*args, **kwargs)

    def fit_spline(self, x, y, z, w=None, s=None):
        """
        Fit spline using `scipy.interpolate.SmoothBivariateSpline`

        Parameters
        ----------
        x, y, z : array_like
            1-D sequences of data points (order is not important).
        w : array_like, optional
            Positive 1-D sequence of weights, of same length as x, y,
            and z.
        s : float or None, optional
            Positive smoothing factor defined for estimation condition:
            ```
            sum((w[i]*(z[i]-s(x[i], y[i])))**2, axis=0) <= s
            ```
            Default s=len(w) which should be a good value if 1/w[i] is
            an estimate of the standard deviation of z[i].
        """

        from scipy.interpolate import SmoothBivariateSpline

        self.spline = SmoothBivariateSpline(x, y, z, w=w, bbox=self.bbox,
                                            kx=self._kx, ky=self._ky, s=s,
                                            eps=self._eps)

    def fit_LSQ_spline(self, x, y, z, tx, ty, w=None):
        """
        Fit spline using `scipy.interpolate.LSQBivariateSpline`

        Parameters
        ----------
        x, y, z : array_like
            1-D sequences of data points (order is not important).
        tx, ty : array_like
            Strictly ordered 1-D sequences of knots coordinates.
        w : array_like, optional
            Positive 1-D sequence of weights, of same length as x, y and z.
        """

        from scipy.interpolate import LSQBivariateSpline

        self.spline = LSQBivariateSpline(x, y, z, tx, ty, w=w, bbox=self.bbox,
                                         kx=self._kx, ky=self._ky,
                                         eps=self._eps)


class SplineFitter(metaclass=_FitterMeta):
    """
    Spline Fitter
    """

    def __call__(self, model, x, y, z=None, *, w=None, s=None):
        """
        Fit spline

        Parameters
        ----------
        model : FittableModel
            The model to be fit. Must be Spline1D or Spline2D model.
        x, y, z : array_like
            equal length 1-D sequences of data points.
                If 1D spline, x must be increasing; must be strictly
                increasing if s is 0.
                If 1D spline z must be None, otherwise z must be present.
        w : array_like, optional
            Weights for spline fitting. Must be positive. Must have same
            length as x. If None (default), weights are all equal.
        s : float or None, optional
            Positive smoothing factor used to choose the number of knots.
            Number of a knots will be increased until the smoothing
            condition is satisfied:
                For 1-D:
                ```
                sum((w[i] * (y[i]-spl(x[i])))**2, axis=0) <= s
                ```
                For 2-D:
                ```
                sum((w[i]*(z[i]-s(x[i], y[i])))**2, axis=0) <= s
                ```
            If None (default), s = len(w) which should be a good value
            if 1/w[i] is an estimate of the standard deviation of y[i].
            If 0, spline will interpolate through all data points.
        """

        model_copy = model.copy()

        if isinstance(model_copy, Spline1D):
            if z is not None:
                raise ValueError("1D model can only have 2 data points.")

            model_copy.fit_spline(x, y, w=w, s=s)
            return model_copy
        elif isinstance(model_copy, Spline2D):
            if z is None:
                raise ValueError("2D model must have 3 data points.")

            model_copy.fit_spline(x, y, z, w=w, s=s)
            return model_copy
        else:
            raise ModelDefinitionError("Only spline models are compatible with this fitter")


class SplineLSQFitter(metaclass=_FitterMeta):
    """
    Spline LSQ Fitter
    """

    def __call__(self, model, t, x, y, z=None, *, w=None):
        """
        Fit spline using least-squares

        Parameters
        ----------
        model : FittableModel
            The model to be fit. Must be Spline1D or Spline2D model.
        t : array_like or tuple of array_like
            If 1-D, a strictly ordered 1-D sequence of knot positions.
            If 2-D, an xy-tuple of strictly ordered 1-D sequences of
            knots coordinates.
        x, y, z : array_like
            equal length 1-D sequences of data points.
                If 1D spline, x must be increasing; must be strictly
                increasing if s is 0.
                If 1D spline z must be None, otherwise z must be present.
        w : array_like, optional
            Weights for spline fitting. Must be positive. Must have same
            length as x. If None (default), weights are all equal.
        """

        model_copy = model.copy()

        if isinstance(model_copy, Spline1D):
            if z is not None:
                raise ValueError("1D model can only have 2 data points.")

            model_copy.fit_LSQ_spline(x, y, t, w=w)
            return model_copy
        elif isinstance(model_copy, Spline2D):
            if z is None:
                raise ValueError("2D model must have 3 data points.")
            if len(t) != 2:
                raise ValueError("Must have both x and y knots defined")

            model_copy.fit_LSQ_spline(x, y, z, t[0], t[1], w=w)
            return model_copy
        else:
            raise ModelDefinitionError("Only spline models are compatible with this fitter")
