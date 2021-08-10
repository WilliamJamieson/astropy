# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Spline models and fitters."""
# pylint: disable=line-too-long, too-many-lines, too-many-arguments, invalid-name

import warnings

import abc
import numpy as np

from astropy.utils.exceptions import (AstropyUserWarning,)
from .core import (Fittable1DModel, Fittable2DModel,)


class _Spline(abc.ABC):
    """
    Meta class for spline models
    """
    spline_dimension = None
    optional_inputs = {}

    def _init_spline(self, t=None, c=None, k=None):
        self._t = t
        self._c = c
        self._k = k
        self._check_dimension()

        # Hack to allow an optional model argument
        self._create_optional_inputs()

    def _check_dimension(self):
        t_dim = self._get_dimension('_t')
        k_dim = self._get_dimension('_k')

        if (t_dim != 0) and (k_dim != 0) and (t_dim != k_dim):
            raise ValueError("The dimensions for knots and degree do not agree!")

    def _get_dimension(self, var):
        value = getattr(self, var)

        dim = 1
        if value is None:
            return 0
        elif isinstance(value, tuple):
            length = len(value)
            if length > 1:
                dim = length
            else:
                warnings.warn(f"{var} should not be a tuple of length 1",
                              AstropyUserWarning)
                setattr(self, var, value[0])

        if (self.spline_dimension is not None) and dim != self.spline_dimension:
            raise RuntimeError(f'{var} should have dimension {self.spline_dimension}')
        else:
            return dim

    def reset(self):
        self._t = None
        self._c = None
        self._k = None

    @property
    def _has_tck(self):
        return (self._t is not None) and (self._c is not None) and (self._k is not None)

    @property
    def knots(self):
        if self._t is None:
            warnings.warn("The knots have not been defined yet!",
                          AstropyUserWarning)
        return self._t

    @knots.setter
    def knots(self, value):
        if self._has_tck:
            warnings.warn("The knots have already been defined, reseting rest of tck!",
                          AstropyUserWarning)
            self.reset()

        self._t = value
        self._check_dimension()

    @property
    def coeffs(self):
        if self._c is None:
            warnings.warn("The fit coeffs have not been defined yet!",
                          AstropyUserWarning)
        return self._c

    @coeffs.setter
    def coeffs(self, value):
        if self._has_tck:
            warnings.warn("The fit coeffs have already been defined, reseting rest of tck!",
                          AstropyUserWarning)
            self.reset()

        self._c = value

    @property
    def degree(self):
        if self._k is None:
            warnings.warn("The fit degree have not been defined yet!",
                          AstropyUserWarning)
        return self._k

    @degree.setter
    def degree(self, value):
        if self._has_tck:
            warnings.warn("The fit degrees have already been defined, reseting rest of tck!",
                          AstropyUserWarning)
            self.reset()

        self._k = value
        self._check_dimension()

    def _get_tck(self):
        raise NotImplementedError

    def _set_tck(self, value):
        raise NotImplementedError

    @property
    def tck(self):
        if self._has_tck:
            return self._get_tck()
        else:
            raise RuntimeError('tck needs to be defined!')

    @tck.setter
    def tck(self, value):
        self._set_tck(value)

    def _get_spline(self):
        raise NotImplementedError

    @property
    def spline(self):
        return self._get_spline()

    @spline.setter
    def spline(self, value):
        self.tck = value

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
    One dimensional Spline Model

    Parameters
    ----------
    t : array-like, optional
        The knots for the spline.
    c : array-like, optional
        The spline coefficients.
    k : int, optional
        The degree of the spline polynomials. Supported:
            1 <= k <= 5

    Notes
    -----
    The supported version of t, c, k are the tck-tuples used by 1-D
    `scipy.interpolate` models.

    Much of the additional functionality of this model is provided by
    `scipy.interpolate.BSpline` which can be directly accessed via the
    spline property.

    Note that t, c, and k must all be set in order to evaluate this model.
    """

    spline_dimension = 1
    optional_inputs = {'nu': 0}

    def __init__(self, t=None, c=None, k=None, n_models=None,
                 model_set_axis=None, name=None, meta=None, **params):
        self._init_spline(t=t, c=c, k=k)

        super().__init__(n_models=n_models, model_set_axis=model_set_axis,
                         name=name, meta=meta, **params)

    def _get_tck(self):
        return self.knots, self.coeffs, self.degree

    def _set_tck(self, value):
        from scipy.interpolate import BSpline

        if isinstance(value, tuple) and (len(value) == 3):
            self.knots = value[0]
            self.coeffs = value[1]
            self.degree = value[2]
        elif isinstance(value, BSpline):
            self.tck = value.tck
        else:
            raise NotImplementedError('tck 3-tuple and BSpline setting implemented')

    def _get_spline(self):
        from scipy.interpolate import BSpline

        if self._has_tck:
            return BSpline(*self.tck)
        else:
            return BSpline([0, 1, 2, 3], [0, 0], 1)

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

        if 'nu' in kwargs and self._has_tck:
            if kwargs['nu'] > self.degree + 1:
                raise RuntimeError(f'Cannot evaluate a derivative of order higher than {self.degree + 1}')

        return self.spline(x, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        Make model callable to model evaluation

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
        kwargs = self._intercept_optional_inputs(**kwargs)

        return super().__call__(*args, **kwargs)

    def derivative(self, nu=1):
        """
        Create a spline that is a derivative of this one

        Parameters
        ----------
        nu : int, optional
            Derivative order, default is 1.
        """
        if nu <= self.degree:
            spline = self.spline

            derivative = Spline1D()
            derivative.spline = spline.derivative(nu=nu)

            return derivative
        else:
            raise ValueError(f'Must have nu <= {self.degree}')

    def antiderivative(self, nu=1):
        """
        Create a spline that is a derivative of this one

        Parameters
        ----------
        nu : int, optional
            Antiderivative order, default is 1.

        Notes
        -----
        Assumes constant of integration is 0
        """
        if (nu + self.degree) <= 5:
            spline = self.spline

            antiderivative = Spline1D()
            antiderivative.spline = spline.antiderivative(nu=nu)

            return antiderivative
        else:
            raise ValueError(f'Spline can have max degree 5, antiderivative degree will be {nu + self.degree}')

    @staticmethod
    def _sort_xy(x, y, sort=True):
        if sort:
            arg_sort = np.argsort(x)
            return x[arg_sort], y[arg_sort]
        else:
            return x, y

    def fit_spline(self, x, y, w=None, k=3, s=None, t=None):
        """
        Fit spline using `scipy.interpolate.splrep`

        Parameters
        ----------
        x, y : array-like
            The data points defining a curve y = f(x)

        w : array-like, optional
            Strictly positive rank-1 array of weights the same length
            as x and y. The weights are used in computing the weighted
            least-squares spline fit. If the errors in the y values have
            standard-deviation given by the vector d, then w should be
            1/d. Default is ones(len(x)).
        k : int, optional
            The degree of the spline fit. It is recommended to use cubic
            splines. Even values of k should be avoided especially with
            small s values.
                1 <= k <= 5
        s : float, optional
            A smoothing condition. The amount of smoothness is
            determined by satisfying the conditions:
                sum((w * (y - g))**2,axis=0) <= s
            where g(x) is the smoothed interpolation of (x,y). The user
            can use s to control the tradeoff between closeness and
            smoothness of fit. Larger s means more smoothing while
            smaller values of s indicate less smoothing. Recommended
            values of s depend on the weights, w. If the weights
            represent the inverse of the standard-deviation of y, then
            a good s value should be found in the range
                (m-sqrt(2*m),m+sqrt(2*m))
            where m is the number of datapoints in x, y, and w.
            default : s=m-sqrt(2*m) if weights are supplied.
                      s = 0.0 (interpolating) if no weights are supplied.

        t : array_like, optional
            User specified knots. s is ignored if t is passed.
        """

        if (s is not None) and (t is not None):
            warnings.warn("Knots specified so moothing condition will be ignored",
                          AstropyUserWarning)

        xb = self.bbox[0]
        xe = self.bbox[1]

        x, y = self._sort_xy(x, y)

        from scipy.interpolate import (splrep, BSpline)

        self.tck, fp, ier, msg = splrep(x, y, w=w, xb=xb, xe=xe, k=k, s=s, t=t,
                                        full_output=1)

        return fp, ier, msg


class Spline2D(Fittable2DModel, _Spline):
    """
    Two dimensional Spline model

    Parameters
    ----------
    t : tuple(array-like, array-like), optional
        The knots in x and knots in y for the spline
    c : array-like, optional
        The spline coefficients.
    k : tuple(int, int), optional
        The degree of the spline polynomials. Supported:
            1 <= k <= 5

    Notes
    -----
    The supported versions of t, c, k are the tck-tuples used by 2-D
    `scipy.interpolate` models.

    Much of the additional functionality of this model is provided by
    `scipy.interpolate.BivariateSpline` which can be directly accessed
    via the spline property.

    Note that t, c, and k must all be set in order to evaluate this model.
    """

    spline_dimension = 2
    optional_inputs = {'dx': 0,
                       'dy': 0}

    def __init__(self, t=None, c=None, k=None, n_models=None,
                 model_set_axis=None, name=None, meta=None, **params):
        self._init_spline(t=t, c=c, k=k)

        super().__init__(n_models=n_models, model_set_axis=model_set_axis,
                         name=name, meta=meta, **params)

    def _get_tck(self):
        tck = list(self.knots)
        tck.append(self.coeffs)
        tck.extend(list(self.degree))

        return tuple(tck)

    def _set_tck(self, value):
        from scipy.interpolate import BivariateSpline

        if isinstance(value, list) or isinstance(value, tuple):
            if len(value) == 3:
                self.knots = tuple(value[0])
                self.coeffs = value[1]
                self.degree = tuple(value[2])
            elif len(value) == 5:
                self.knots = tuple(value[:2])
                self.coeffs = value[2]
                self.degree = tuple(value[3:])
            else:
                raise ValueError('tck must be of length 3 or 5')
        elif isinstance(value, BivariateSpline):
            self.tck = (value.get_knots(), value.get_coeffs(), value.degrees)
        else:
            raise NotImplementedError('tck-tuple and BivariateSpline setting implemented')

    def _get_spline(self):
        from scipy.interpolate import BivariateSpline

        if self._has_tck:
            return BivariateSpline._from_tck(self.tck)
        else:
            return BivariateSpline._from_tck([[0, 1, 2, 3], [0, 1, 2, 3], [0, 0, 0, 0], [1], [1]])

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

        if self._has_tck:
            if 'dx' in kwargs:
                if kwargs['dx'] > self.degree[0] - 1:
                    raise RuntimeError(f'Cannot evaluate a derivative of order higher than {self.degree[0] - 1}')
            if 'dy' in kwargs:
                if kwargs['dy'] > self.degree[1] - 1:
                    raise RuntimeError(f'Cannot evaluate a derivative of order higher than {self.degree[1] - 1}')

        return self.spline(x, y, **kwargs)

    def __call__(self, *args, **kwargs):
        """
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
        kwargs = self._intercept_optional_inputs(**kwargs)

        return super().__call__(*args, **kwargs)

    def fit_spline(self, x, y, z, w=None, kx=3, ky=3, s=None, tx=None, ty=None):
        """
        Fit spline using `scipy.interpolate.bisplrep`

        Parameters
        ----------
        x, y, z : ndarray
            Rank-1 arrays of data points.
        w : ndarray, optional
            Rank-1 array of weights. By default w=np.ones(len(x)).
        kx, ky :int, optional
            The degrees of the spline.
                1 <= kx, ky <= 5
            Third order, the default (kx=ky=3), is recommended.
        s : float, optional
            A non-negative smoothing factor. If weights correspond to
            the inverse of the standard-deviation of the errors in z,
            then a good s-value should be found in the range
                (m-sqrt(2*m),m+sqrt(2*m))
            where m=len(x).
        tx, ty : ndarray, optional
            Rank-1 arrays of the user knots of the spline. Must be
            specified together and s is ignored when specified.
        """

        if ((tx is None) and (ty is not None)) or ((tx is not None) and (ty is None)):
            raise ValueError('If 1 dimension of knots are specified, both must be specified')

        if (s is not None) and (tx is not None):
            warnings.warn("Knots specified so moothing condition will be ignored",
                          AstropyUserWarning)

        xb = self.bbox[0]
        xe = self.bbox[1]
        yb = self.bbox[2]
        ye = self.bbox[3]

        from scipy.interpolate import bisplrep

        self.tck, fp, ier, msg = bisplrep(x, y, z, w=w, kx=kx, ky=ky,
                                          xb=xb, xe=xe, yb=yb, ye=ye,
                                          s=s, tx=tx, ty=ty, full_output=1)

        return fp, ier, msg
