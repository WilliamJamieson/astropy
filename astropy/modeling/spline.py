# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Spline models and fitters."""
# pylint: disable=line-too-long, too-many-lines, too-many-arguments, invalid-name

import warnings

import abc
import re
import functools
import numpy as np

from astropy.utils.exceptions import (AstropyUserWarning,)
from astropy.utils import isiterable
from .core import (FittableModel, Fittable1DModel, Fittable2DModel,)

from .parameters import Parameter


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
            x = np.array(x)
            y = np.array(y)
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

        from scipy.interpolate import splrep

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


class _NewSpline(FittableModel):
    """Base class for spline models"""

    optional_inputs = {}

    def __init__(self, knots=None, coeffs=None, degree=None, bounds=None,
                 n_models=None, model_set_axis=None, name=None, meta=None):
        self._knot_names = []
        self._coeff_names = []

        super().__init__(
            n_models=n_models, model_set_axis=model_set_axis, name=name,
            meta=meta)

        self._t = None
        self._c = None
        self._user_knots = False
        self._degree = degree

        # Hack to allow an optional model argument
        self._create_optional_inputs()

        if knots is not None:
            self._init_spline(knots, coeffs, bounds)
        elif coeffs is not None:
            raise ValueError("If one passes a coeffs vector one needs to also pass a knots vector!")

    @property
    def param_names(self):
        """
        Coefficient names generated based on the spline's degree and
        number of knots.
        """

        param_names = ()

        if hasattr(self, '_knot_names'):
            param_names += tuple(self._knot_names)

        if hasattr(self, '_coeff_names'):
            param_names += tuple(self._coeff_names)

        return param_names

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

    def evaluate(self, *args, **kwargs):
        """ Extract the optional kwargs passed to call """

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

    def __call__(self, *args, **kwargs):
        """
        Make model callable to model evaluation
        """

        # Hack to allow an optional model argument
        kwargs = self._intercept_optional_inputs(**kwargs)

        return super().__call__(*args, **kwargs)

    def _create_parameter(self, name: str, index: int, attr: str, fixed=False):
        """
        Create a spline parameter linked to an attribute array.

        Parameters
        ----------
        name : str
            Name for the parameter
        index : int
            The index of the parameter in the array
        attr : str
            The name for the attribute array
        fixed : optional, bool
            If the parameter should be fixed or not
        """

        # Hack to allow parameters and attribute array to freely exchange values
        #   _getter forces reading value from attribute array
        #   _setter forces setting value to attribute array

        def _getter(value, model: "_NewSpline", index: int, attr: str):
            return getattr(model, attr)[index]

        def _setter(value, model: "_NewSpline", index: int, attr: str):
            getattr(model, attr)[index] = value
            return value
        getter = functools.partial(_getter, index=index, attr=attr)
        setter = functools.partial(_setter, index=index, attr=attr)

        default = getattr(self, attr)
        param = Parameter(name=name, default=default[index], fixed=fixed,
                          getter=getter, setter=setter)
        # setter/getter wrapper for parameters in this case require the
        # parameter to have a reference back to its parent model
        param.model = self
        param.value = default[index]

        # Add parameter to model
        self.__dict__[name] = param

    def _create_parameters(self, base_name: str, attr: str, fixed=False):
        """
        Create a spline parameters linked to an attribute array for all
        elements in that array

        Parameters
        ----------
        base_name : str
            Base name for the parameters
        attr : str
            The name for the attribute array
        fixed : optional, bool
            If the parameters should be fixed or not

        Returns
        -------
            Tuple of all the names created
        """
        names = []
        for index in range(len(getattr(self, attr))):
            name = f"{base_name}{index}"
            names.append(name)

            self._create_parameter(name, index, attr, fixed)

        return names

    def _init_parameters(self):
        raise NotImplementedError("This needs to be implemented")

    def _init_data(self, knots, coeffs, bounds=None):
        raise NotImplementedError("This needs to be implemented")

    def _init_spline(self, knots, coeffs, bounds=None):
        self._init_data(knots, coeffs, bounds)
        self._init_parameters()

        # initialize the internal parameter data
        self._initialize_parameters((), {})
        self._initialize_slices()


class NewSpline1D(_NewSpline):
    """
    One dimensional Spline Model

    Parameters
    ----------
    knots :  optional
        Define the knots for the spline. Can be 1) the number of interior
        knots for the spline, 2) the array of all knots for the spline, or
        3) If both bounds are defined, the interior knots for the spline
    coeffs : optional
        The array of knot coefficients for the spline
    degree : optional
        The degree of the spline. It must be 1 <= degree <= 5, default is 3.
    bounds : optional
        The upper and lower bounds of the spline.

    Notes
    -----
    Much of the functionality of this model is provided by
    `scipy.interpolate.BSpline` which can be directly accessed via the
    bspline property.

    Fitting for this model is provided by wrappers for:
    `scipy.interpolate.UnivariateSpline`,
    `scipy.interpolate.InterpolatedUnivariateSpline`,
    and `scipy.interpolate.LSQUnivariateSpline`.

    If one fails to define any knots/coefficients, no parameters will
    be added to this model until a fitter is called. This is because
    some of the fitters for splines vary the number of parameters and so
    we cannot define the parameter set until after fitting in these cases.

    Since parameters are not necessarily known at model initialization,
    setting model parameters directly via the model interface has been
    disabled.

    Direct constructors are provided for this model which incorporate the
    fitting to data directly into model construction.

    Knot parameters are declared as "fixed" parameters by default to
    enable the use of other `astropy.modeling` fitters to be used to
    fit this model.
    """
    n_inputs = 1
    n_outputs = 1
    _separable = True

    optional_inputs = {'nu': 0}

    def __init__(self, knots=None, coeffs=None, degree=3, bounds=None,
                 n_models=None, model_set_axis=None, name=None, meta=None):

        super().__init__(
            knots=knots, coeffs=coeffs, degree=degree, bounds=bounds,
            n_models=n_models, model_set_axis=model_set_axis, name=name, meta=meta
        )

    @property
    def t(self):
        if self._t is None:
            return np.concatenate((np.zeros(self._degree + 1), np.ones(self._degree + 1)))
        else:
            return self._t

    @t.setter
    def t(self, value):
        if self._t is None:
            raise ValueError("The model parameters must be initialized before setting knots.")
        elif len(value) == len(self._t):
            self._t = value
        else:
            raise ValueError("There must be exactly as many knots as previously defined.")

    @property
    def t_interior(self):
        return self.t[self.degree + 1: -(self.degree + 1)]

    @property
    def c(self):
        if self._c is None:
            return np.zeros(len(self.t))
        else:
            return self._c

    @c.setter
    def c(self, value):
        if self._c is None:
            raise ValueError("The model parameters must be initialized before setting coeffs.")
        elif len(value) == len(self._c):
            self._c = value
        else:
            raise ValueError("There must be exactly as many coeffs as previously defined.")

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, value):
        if value != self._degree:
            raise ValueError("The value of degree cannot be changed!")

    @property
    def _initialized(self):
        return self._t is not None and self._c is not None

    @property
    def tck(self):
        return (self.t, self.c, self.degree)

    @tck.setter
    def tck(self, value):
        if self._initialized:
            self.t = value[0]
            self.c = value[1]
            self.degree = value[2]
        else:
            self._init_spline(value[0], value[1])

    @property
    def bspline(self):
        from scipy.interpolate import BSpline

        return BSpline(*self.tck)

    @bspline.setter
    def bspline(self, value):
        from scipy.interpolate import BSpline

        if isinstance(value, BSpline):
            self.tck = value.tck
        else:
            self.tck = value

    @property
    def knots(self):
        return [getattr(self, knot) for knot in self._knot_names]

    @property
    def coeffs(self):
        return [getattr(self, coeff) for coeff in self._coeff_names]

    def _init_parameters(self):
        self._knot_names = self._create_parameters("knot", "t", fixed=True)
        self._coeff_names = self._create_parameters("coeff", "c")

    def _init_bounds(self, bounds=None):
        if bounds is None:
            bounds = [None, None]

        if bounds[0] is None:
            lower = np.zeros(self._degree + 1)
        else:
            lower = np.array([bounds[0]] * (self._degree + 1))

        if bounds[1] is None:
            upper = np.ones(self._degree + 1)
        else:
            upper = np.array([bounds[1]] * (self._degree + 1))

        if bounds[0] is not None and bounds[1] is not None:
            self.bounding_box = bounds
            has_bounds = True
        else:
            has_bounds = False

        return has_bounds, lower, upper

    def _init_knots(self, knots, has_bounds, lower, upper):
        if np.issubdtype(type(knots), np.integer):
            self._t = np.concatenate(
                (lower, np.zeros(knots), upper)
            )
        elif isiterable(knots):
            self._user_knots = True
            if has_bounds:
                self._t = np.concatenate(
                    (lower, np.array(knots), upper)
                )
            else:
                if len(knots) < 2*(self._degree + 1):
                    raise ValueError(f"Must have at least {2*(self._degree + 1)} knots.")
                self._t = np.array(knots)
        else:
            raise ValueError(f"Knots: {knots} must be iterable or value")

        # check that knots form a viable spline
        self.bspline

    def _init_coeffs(self, coeffs=None):
        if coeffs is None:
            self._c = np.zeros(len(self._t))
        else:
            self._c = np.array(coeffs)

        # check that coeffs form a viable spline
        self.bspline

    def _init_data(self, knots, coeffs, bounds=None):
        self._init_knots(knots, *self._init_bounds(bounds))
        self._init_coeffs(coeffs)

    def evaluate(self, *args, **kwargs):
        """
        Evaluate the spline.

        Parameters
        ----------
        x :
            (positional) The points where the model is evaluating the spline at
        nu : optional
            (kwarg) The derivative of the spline for evaluation, 0 <= nu <= degree + 1.
            Default: 0.

        Returns
        -------
        Spline values at each x point.
        """
        kwargs = super().evaluate(*args, **kwargs)
        x = args[0]

        if 'nu' in kwargs:
            if kwargs['nu'] > self.degree + 1:
                raise RuntimeError("Cannot evaluate a derivative of "
                                   f"order higher than {self.degree + 1}")

        return self.bspline(x, **kwargs)

    def interpolate_data(self, x, y, w=None, bbox=[None, None]):
        """
        Fit the spline as an interpolating spline to the (x, y) data points.

        Parameters
        ----------
        x, y :
            The data points defining a curve y = f(x)
        w : optional
            Weights for each data points in the interpolation. Note that this must
            be the same shape as x. Default: None, which equally weights
            all data points
        bbox : optional
            The lower and upper bounds for the fit. Default: [None, None]. If a
            bound is None, the bound is assumed to be the smallest/largest x value.

        Notes
        -----
        This is a wrapper of the `scipy.interpolate.InterpolatedUnivariateSpline`
        function.

        It is very dangerous to attempt to use this function after one defines
        the knots/coefficients for the model, as this function may attempt
        to change the number of parameters for this model.
        """
        if self._user_knots:
            warnings.warn("The current user specified knots maybe ignored for interpolating data",
                          AstropyUserWarning)
            self._user_knots = False

        from scipy.interpolate import InterpolatedUnivariateSpline
        spline = InterpolatedUnivariateSpline(x, y, w=w, bbox=bbox, k=self._degree)

        self.tck = spline._eval_args

        return spline

    @classmethod
    def interpolate(cls, x, y, k=3, w=None, bbox=[None, None]):
        """
        Create an interpolating spline model for the given data.

        Parameters
        ----------
        x, y :
            The data points defining a curve y = f(x)
        k : optional, int
            The degree of the spline for interpolation. Default: 3.
        w : optional
            Weights for each data points in the interpolation. Note that this must
            be the same shape as x. Default: None, which equally weights
            all data points
        bbox : optional
            The lower and upper bounds for the fit. Default: [None, None]. If a
            bound is None, the bound is assumed to be the smallest/largest x value.

        Returns
        -------
        A spline model which interpolates the data
        """
        spline = cls(degree=k)
        spline.interpolate_data(x, y, w=w, bbox=bbox)

        return spline

    def smoothing_fit(self, x, y, s=None, w=None, bbox=[None, None]):
        """
        Fit the spline as a smoothing spline to the (x, y) data points.

        Parameters
        ----------
        x, y :
            The data points defining a curve y = f(x)
        s : optional
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
        w : optional
            Weights for each data points in the interpolation. Note that this must
            be the same shape as x. Default: None, which equally weights
            all data points
        bbox : optional
            The lower and upper bounds for the fit. Default: [None, None]. If a
            bound is None, the bound is assumed to be the smallest/largest x value.

        Notes
        -----
        This is a wrapper of the `scipy.interpolate.UnivariateSpline`
        function.

        It is very dangerous to attempt to use this function after one defines
        the knots/coefficients for the model, as this function may attempt
        to change the number of parameters for this model.
        """
        if self._user_knots:
            warnings.warn("The current user specified knots maybe ignored for interpolating data",
                          AstropyUserWarning)
            self._user_knots = False

        if bbox != [None, None]:
            self.bounding_box = bbox

        from scipy.interpolate import UnivariateSpline
        spline = UnivariateSpline(x, y, w=w, bbox=bbox, k=self._degree, s=s)

        self.tck = spline._eval_args

        return spline

    @classmethod
    def smoothing(cls, x, y, k=3, s=None, w=None, bbox=[None, None]):
        """
        Create a smoothing spline model for the given data.

        Parameters
        ----------
        x, y :
            The data points defining a curve y = f(x)
        k : optional, int
            The degree of the spline for interpolation. Default: 3.
        s : optional
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
        w : optional
            Weights for each data points in the interpolation. Note that this must
            be the same shape as x. Default: None, which equally weights
            all data points
        bbox : optional
            The lower and upper bounds for the fit. Default: [None, None]. If a
            bound is None, the bound is assumed to be the smallest/largest x value.

        Returns
        -------
        A spline model which smooth fits the data
        """
        spline = cls(degree=k)
        spline.smoothing_fit(x, y, s=s, w=w, bbox=bbox)

        return spline

    def lsq_fit(self, x, y, t=None, w=None, bbox=[None, None]):
        """
        Fit use least-squares regression to fit spline to the (x, y) data points,
        using the provided knots

        Parameters
        ----------
        x, y :
            The data points defining a curve y = f(x)
        t : optional
            Knots to override (or define if model is empty) the knots of the model
            for use in fitting the model.
        w : optional
            Weights for each data points in the interpolation. Note that this must
            be the same shape as x. Default: None, which equally weights
            all data points
        bbox : optional
            The lower and upper bounds for the fit. Default: [None, None]. If a
            bound is None, the bound is assumed to be the smallest/largest x value.

        Notes
        -----
        This is a wrapper of the `scipy.interpolate.LSQUnivariateSpline`
        function.

        When overriding previously specified knots, the provided knots must
        be interior knots and be the same number of interior knots as the previously
        defined ones.
        """
        if t is not None:
            if self._user_knots:
                warnings.warn("The current user specified knots will be "
                              "overwritten for by knots passed into this function",
                              AstropyUserWarning)
        else:
            if self._user_knots:
                t = self.t_interior
            else:
                raise RuntimeError("No knots have been provided")

        if bbox != [None, None]:
            self.bounding_box = bbox

        from scipy.interpolate import LSQUnivariateSpline
        spline = LSQUnivariateSpline(x, y, t, w=w, bbox=bbox, k=self._degree)

        self.tck = spline._eval_args

        return spline

    @classmethod
    def lsq(cls, x, y, t, k=3, w=None, bbox=[None, None]):
        """
        Create a spline model for the given data using least-squares regression

        Parameters
        ----------
        x, y :
            The data points defining a curve y = f(x)
        t :
            Knots to override (or define if model is empty) the knots of the model
            for use in fitting the model.
        k : optional, int
            The degree of the spline for interpolation. Default: 3.
        w : optional
            Weights for each data points in the interpolation. Note that this must
            be the same shape as x. Default: None, which equally weights
            all data points
        bbox : optional
            The lower and upper bounds for the fit. Default: [None, None]. If a
            bound is None, the bound is assumed to be the smallest/largest x value.

        Returns
        -------
        A spline model which fits the data using least-squares regression.
        """
        spline = cls(degree=k)
        spline.lsq_fit(x, y, t, w=w, bbox=bbox)

        return spline

    def splrep_data(self, x, y, w=None, s=None, task=0, t=None, bbox=[None, None]):
        """
        Alternate interface for using the `scipy.interpolate.splrep` method.

        Parameters
        ----------
        x, y :
            The data points defining a curve y = f(x)
        w : optional
            Weights for each data points in the interpolation. Note that this must
            be the same shape as x. Default: None, which equally weights
            all data points
        s : optional
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
        task : optional
            If task==0 find t and c for a given smoothing factor, s.
            If task==1 find t and c for another value of the smoothing
            factor, s. There must have been a previous call with task=0
            or task=1 for the same set of data (t will be stored an used
            internally) If task=-1 find the weighted least square spline
            for a given set of knots, t. These should be interior knots
            as knots on the ends will be added automatically. Default: 0.
        t : optional
            Knots to override (or define if model is empty) the knots of the model
            for use in fitting the model.
        bbox : optional
            The lower and upper bounds for the fit. Default: [None, None]. If a
            bound is None, the bound is assumed to be the smallest/largest x value.

        Notes
        -----
        This is a wrapper of the `scipy.interpolate.splrep` function.

        It is very dangerous to attempt to use this function after one defines
        the knots/coefficients for the model, as this function may attempt
        to change the number of parameters for this model.
        """
        if t is not None:
            if self._user_knots:
                warnings.warn("The current user specified knots will be "
                              "overwritten for by knots passed into this function",
                              AstropyUserWarning)
        else:
            if self._user_knots:
                t = self.t_interior

        if bbox != [None, None]:
            self.bounding_box = bbox

        from scipy.interpolate import splrep
        self.tck = splrep(x, y, w=w, xb=bbox[0], xe=bbox[1], k=self._degree, s=s, t=t, task=task)

    @classmethod
    def splrep(cls, x, y, k=3, w=None, s=None, task=0, t=None, bbox=[None, None]):
        """
        Create a spline model for the given data using least-squares regression

        Parameters
        ----------
        x, y :
            The data points defining a curve y = f(x)
        k : optional, int
            The degree of the spline for interpolation. Default: 3.
        w : optional
            Weights for each data points in the interpolation. Note that this must
            be the same shape as x. Default: None, which equally weights
            all data points
        s : optional
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
        task : optional
            If task==0 find t and c for a given smoothing factor, s.
            If task==1 find t and c for another value of the smoothing
            factor, s. There must have been a previous call with task=0
            or task=1 for the same set of data (t will be stored an used
            internally) If task=-1 find the weighted least square spline
            for a given set of knots, t. These should be interior knots
            as knots on the ends will be added automatically. Default: 0.
        t : optional
            Knots to override (or define if model is empty) the knots of the model
            for use in fitting the model.
        bbox : optional
            The lower and upper bounds for the fit. Default: [None, None]. If a
            bound is None, the bound is assumed to be the smallest/largest x value.

        Returns
        -------
        A spline model which fits the data using the method selected by
        `scipy.interpolate.splrep`.
        """
        spline = cls(degree=k)
        spline.splrep_data(x, y, w=w, s=s, task=task, t=t, bbox=bbox)

        return spline

    def derivative(self, nu=1):
        """
        Create a spline that is a derivative of this one

        Parameters
        ----------
        nu : int, optional
            Derivative order, default is 1.

        Return
        ------
        A spline model for the derivative
        """
        if nu <= self.degree:
            bspline = self.bspline.derivative(nu=nu)

            derivative = NewSpline1D(degree=bspline.k)
            derivative.bspline = bspline

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

        Returns
        -------
        A spline model for the antiderivative.

        Notes
        -----
        Assumes constant of integration is 0
        """
        if (nu + self.degree) <= 5:
            bspline = self.bspline.antiderivative(nu=nu)

            antiderivative = NewSpline1D(degree=bspline.k)
            antiderivative.bspline = bspline

            return antiderivative
        else:
            raise ValueError("Supported splines can have max degree 5, "
                             f"antiderivative degree will be {nu + self.degree}")
