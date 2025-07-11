# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module contains the fundamental classes used for representing
coordinates in astropy.
"""

import functools
from typing import Any, NamedTuple

import numpy as np

from astropy import units as u
from astropy.units import SpecificTypeQuantity
from astropy.utils.compat import COPY_IF_NEEDED, NUMPY_LT_2_0

from . import formats

__all__ = ["Angle", "Latitude", "Longitude"]


# these are used by the `hms` and `dms` attributes
class hms_tuple(NamedTuple):
    """A named tuple of (hour, minute, second) values."""

    h: float | np.floating[Any]
    """The hour value."""

    m: float | np.floating[Any]
    """The minute value."""

    s: float | np.floating[Any]
    """The second value."""


class dms_tuple(NamedTuple):
    """A named tuple of (degree, minute, second) values."""

    d: float | np.floating[Any]
    """The degree value."""

    m: float | np.floating[Any]
    """The minute value."""

    s: float | np.floating[Any]
    """The second value."""


class signed_dms_tuple(NamedTuple):
    """A named tuple of (sign, degree, minute, second) values."""

    sign: float | np.integer[Any]
    """The sign of the angle, either -1 or +1."""

    d: float | np.floating[Any]
    """The degree value."""

    m: float | np.floating[Any]
    """The minute value."""

    s: float | np.floating[Any]
    """The second value."""


class Angle(SpecificTypeQuantity):
    """
    One or more angular value(s) with units equivalent to radians or degrees.

    An angle can be specified either as an array, scalar, tuple (see
    below), string, `~astropy.units.Quantity` or another
    :class:`~astropy.coordinates.Angle`.

    The input parser is flexible and supports a variety of formats.
    The examples below illustrate common ways of initializing an
    `~astropy.coordinates.Angle` object. First some imports::

      >>> from astropy.coordinates import Angle
      >>> from astropy import units as u

    The angle values can now be provided::

      >>> Angle('10.2345d')
      <Angle 10.2345 deg>
      >>> Angle(['10.2345d', '-20d'])
      <Angle [ 10.2345, -20.    ] deg>
      >>> Angle('1:2:30.43 degrees')
      <Angle 1.04178611 deg>
      >>> Angle('1 2 0 hours')
      <Angle 1.03333333 hourangle>
      >>> Angle(np.arange(1, 8), unit=u.deg)
      <Angle [1., 2., 3., 4., 5., 6., 7.] deg>
      >>> Angle('1°2′3″')
      <Angle 1.03416667 deg>
      >>> Angle('1°2′3″N')
      <Angle 1.03416667 deg>
      >>> Angle('1d2m3.4s')
      <Angle 1.03427778 deg>
      >>> Angle('1d2m3.4sS')
      <Angle -1.03427778 deg>
      >>> Angle('-1h2m3s')
      <Angle -1.03416667 hourangle>
      >>> Angle('-1h2m3sE')
      <Angle -1.03416667 hourangle>
      >>> Angle('-1h2.5m')
      <Angle -1.04166667 hourangle>
      >>> Angle('-1h2.5mW')
      <Angle 1.04166667 hourangle>
      >>> Angle('-1:2.5', unit=u.deg)
      <Angle -1.04166667 deg>
      >>> Angle(10.2345 * u.deg)
      <Angle 10.2345 deg>
      >>> Angle(Angle(10.2345 * u.deg))
      <Angle 10.2345 deg>

    Parameters
    ----------
    angle : `~numpy.array`, scalar, `~astropy.units.Quantity`, `~astropy.coordinates.Angle`
        The angle value. If a tuple, will be interpreted as ``(h, m,
        s)`` or ``(d, m, s)`` depending on ``unit``. If a string, it
        will be interpreted following the rules described above.

        If ``angle`` is a sequence or array of strings, the resulting
        values will be in the given ``unit``, or if `None` is provided,
        the unit will be taken from the first given value.

    unit : unit-like, optional
        The unit of the value specified for the angle.  This may be
        any string that `~astropy.units.Unit` understands, but it is
        better to give an actual unit object.  Must be an angular
        unit.

    dtype : `~numpy.dtype`, optional
        See `~astropy.units.Quantity`.

    copy : bool, optional
        See `~astropy.units.Quantity`.

    Raises
    ------
    `~astropy.units.UnitsError`
        If a unit is not provided or it is not an angular unit.
    """

    _equivalent_unit = u.radian
    _include_easy_conversion_members = True

    def __new__(cls, angle, unit=None, dtype=np.inexact, copy=True, **kwargs):
        if not isinstance(angle, u.Quantity):
            unit = cls._convert_unit_to_angle_unit(unit)

            if isinstance(angle, tuple):
                raise TypeError(
                    "Creating an Angle with a tuple of degrees (or hours), minutes, and seconds "
                    "is no longer supported, as it has ambiguous behavior when the degree "
                    "value is 0. Use another way of creating angles instead (e.g., a less "
                    "ambiguous string like '-0d1m2.3s'). In a future version of astropy, a tuple "
                    "will be interpreted simply as a sequence with the given unit."
                )

            if isinstance(angle, str):
                angle, angle_unit = formats.parse_angle(angle, unit)
                if angle_unit is None:
                    angle_unit = unit

                if isinstance(angle, tuple):
                    if angle_unit == u.hourangle:
                        formats._check_hour_range(angle[0])
                    formats._check_minute_range(angle[1])
                    a = np.abs(angle[0]) + angle[1] / 60.0
                    if len(angle) == 3:
                        formats._check_second_range(angle[2])
                        a += angle[2] / 3600.0

                    angle = np.copysign(a, angle[0])

                if angle_unit is not unit:
                    # Possible conversion to `unit` will be done below.
                    angle = u.Quantity(angle, angle_unit, copy=COPY_IF_NEEDED)

            elif isinstance(angle, np.ndarray):
                if angle.dtype.kind in "SUVO":
                    angle = [cls(x, unit, copy=COPY_IF_NEEDED) for x in angle]

            elif hasattr(angle, "__array__") and (
                not hasattr(angle, "dtype") or angle.dtype.kind not in "SUVO"
            ):
                angle = np.asarray(angle)

            elif np.iterable(angle):
                angle = [cls(x, unit, copy=COPY_IF_NEEDED) for x in angle]

        return super().__new__(cls, angle, unit, dtype=dtype, copy=copy, **kwargs)

    @staticmethod
    @functools.cache
    def _convert_unit_to_angle_unit(unit):
        # using caching to return early when possible (unit comparison is expensive)
        if unit is not None:
            unit = u.Unit(unit)
        return u.hourangle if unit == u.hour else unit

    def _set_unit(self, unit):
        super()._set_unit(self._convert_unit_to_angle_unit(unit))

    @property
    def hour(self):
        """
        The angle's value in hours (read-only property).
        """
        return self.hourangle

    @property
    def hms(self):
        """The angle's value in hours, as a named tuple with ``(h, m, s)`` members."""
        return hms_tuple(*formats._decimal_to_sexagesimal(self.hourangle))

    @property
    def dms(self):
        """The angle's value in degrees, as a ``(d, m, s)`` named tuple."""
        return dms_tuple(*formats._decimal_to_sexagesimal(self.degree))

    @property
    def signed_dms(self):
        """The angle's value in degrees, as a ``(sign, d, m, s)`` named tuple.

        The ``d``, ``m``, ``s`` are thus always positive, and the sign of
        the angle is given by ``sign``.

        This is primarily intended for use with `dms` to generate string
        representations of coordinates that are correct for negative angles.
        """
        return signed_dms_tuple(
            np.sign(self.degree), *formats._decimal_to_sexagesimal(np.abs(self.degree))
        )

    def to_string(
        self,
        unit=None,
        decimal: bool = False,
        sep: str = "fromunit",
        precision: int | None = None,
        alwayssign: bool = False,
        pad: bool = False,
        fields: int = 3,
        format: str | None = None,
    ) -> str:
        """A string representation of the angle.

        Parameters
        ----------
        unit : `~astropy.units.UnitBase`, optional
            Specifies the unit.  Must be an angular unit.  If not
            provided, the unit used to initialize the angle will be
            used.

        decimal : bool, optional
            If `False`, the returned string will be in sexagesimal form
            if possible (for units of degrees or hourangle).  If `True`,
            a decimal representation will be used. In that case, no unit
            will be appended if ``format`` is not explicitly given.

        sep : str, optional
            The separator between numbers in a sexagesimal
            representation.  E.g., if it is ':', the result is
            ``'12:41:11.1241'``. Also accepts 2 or 3 separators. E.g.,
            ``sep='hms'`` would give the result ``'12h41m11.1241s'``, or
            sep='-:' would yield ``'11-21:17.124'``.  Alternatively, the
            special string 'fromunit' means 'dms' if the unit is
            degrees, or 'hms' if the unit is hours.

        precision : int, optional
            The level of decimal precision.  If ``decimal`` is `True`,
            this is the raw precision, otherwise it gives the
            precision of the last place of the sexagesimal
            representation (seconds).  If `None`, or not provided, the
            number of decimal places is determined by the value, and
            will be between 0-8 decimal places as required.

        alwayssign : bool, optional
            If `True`, include the sign no matter what.  If `False`,
            only include the sign if it is negative.

        pad : bool, optional
            If `True`, include leading zeros when needed to ensure a
            fixed number of characters for sexagesimal representation.

        fields : int, optional
            Specifies the number of fields to display when outputting
            sexagesimal notation.  For example:

                - fields == 1: ``'5d'``
                - fields == 2: ``'5d45m'``
                - fields == 3: ``'5d45m32.5s'``

            By default, all fields are displayed.

        format : str, optional
            The format of the result.  If not provided, an unadorned
            string is returned.  Supported values are:

            - 'latex': Return a LaTeX-formatted string

            - 'latex_inline': Return a LaTeX-formatted string which is the
              same as with ``format='latex'`` for |Angle| instances

            - 'unicode': Return a string containing non-ASCII unicode
              characters, such as the degree symbol

        Returns
        -------
        strrepr : str or array
            A string representation of the angle. If the angle is an array, this
            will be an array with a unicode dtype.

        """
        if decimal and sep != "fromunit":
            raise ValueError(
                f"With decimal=True, separator cannot be used (got {sep=!r})"
            )

        if unit is None:
            unit = {"dms": u.degree, "hms": u.hourangle}.get(sep, self.unit)
        else:
            unit = self._convert_unit_to_angle_unit(unit)

        separators = {
            "generic": {u.degree: "dms", u.hourangle: "hms"},
            "latex": {
                u.degree: [r"^\circ", r"{}^\prime", r"{}^{\prime\prime}"],
                u.hourangle: [r"^{\mathrm{h}}", r"^{\mathrm{m}}", r"^{\mathrm{s}}"],
            },
            "unicode": {u.degree: "°′″", u.hourangle: "ʰᵐˢ"},
        }
        # Default separators are as for generic.
        separators[None] = separators["generic"]
        # For Angle "latex_inline" is the same as "latex"
        if format == "latex_inline":
            format = "latex"

        # Create an iterator so we can format each element of what
        # might be an array.
        if not decimal and (unit == u.degree or unit == u.hourangle):
            # Sexagesimal.
            if sep == "fromunit":
                if format not in separators:
                    raise ValueError(f"Unknown format '{format}'")
                sep = separators[format][unit]
            func = functools.partial(
                formats._decimal_to_sexagesimal_string,
                precision=precision,
                sep=sep,
                pad=pad,
                fields=fields,
            )
        else:
            func = ("{:g}" if precision is None else f"{{0:0.{precision}f}}").format
            # Don't add unit by default for decimal.
            # TODO: could we use Quantity.to_string() here?
            if not (decimal and format is None):
                unit_string = unit.to_string(format=format)
                if format == "latex":
                    # Remove $ and add space in front if unit is not a superscript.
                    if "^" in unit_string:
                        unit_string = unit_string[1:-1]
                    else:
                        unit_string = r"\;" + unit_string[1:-1]
                elif len(unit_string) > 1:
                    # Length one for angular units can only happen for
                    # superscript degree, arcmin, arcsec, hour, minute, second,
                    # and those should not get an extra space.
                    unit_string = " " + unit_string

                format_func = func
                func = lambda x: format_func(x) + unit_string

        def do_format(val):
            # Check if value is not nan to avoid ValueErrors when turning it into
            # a hexagesimal string.
            if np.isnan(val):
                return "nan"
            s = func(float(val))
            if alwayssign and not s.startswith("-"):
                s = "+" + s
            return f"${s}$" if format == "latex" else s

        format_ufunc = np.vectorize(do_format, otypes=["U"])
        result = format_ufunc(self.to_value(unit))
        return result if result.ndim else result[()]

    def _wrap_at(self, wrap_angle):
        """
        Implementation that assumes ``angle`` is already validated
        and that wrapping is inplace.
        """
        # Convert the wrap angle and 360 degrees to the native unit of
        # this Angle, then do all the math on raw Numpy arrays rather
        # than Quantity objects for speed.
        a360 = u.degree.to(self.unit, 360.0)
        wrap_angle = wrap_angle.to_value(self.unit)
        self_angle = self.view(np.ndarray)
        if NUMPY_LT_2_0:
            # Ensure ndim>=1 so that comparison is done using the angle dtype.
            self_angle = self_angle[np.newaxis]
        else:
            # Use explicit float to ensure casting to self_angle.dtype (NEP 50).
            wrap_angle = float(wrap_angle)
        wrap_angle_floor = wrap_angle - a360
        # Do the wrapping, but only if any angles need to be wrapped
        #
        # Catch any invalid warnings from the floor division.

        # See if any wrapping is necessary and return early otherwise.
        # It is useful to avoid this since the array may be read-only
        # (e.g. due to broadcasting).
        # Note that since comparisons with NaN always return False,
        # this also ensures that no adjustments are made for a
        # read-only array with some NaN but otherwise OK elements.
        out_of_range = (self_angle < wrap_angle_floor) | (self_angle >= wrap_angle)
        if not out_of_range.any():
            return

        wraps = (self_angle - wrap_angle_floor) // a360

        self_angle -= wraps * a360
        # Rounding errors can cause problems.
        self_angle[self_angle >= wrap_angle] -= a360
        self_angle[self_angle < wrap_angle_floor] += a360

    def wrap_at(self, wrap_angle, inplace=False):
        """
        Wrap the `~astropy.coordinates.Angle` object at the given ``wrap_angle``.

        This method forces all the angle values to be within a contiguous
        360 degree range so that ``wrap_angle - 360d <= angle <
        wrap_angle``. By default a new Angle object is returned, but if the
        ``inplace`` argument is `True` then the `~astropy.coordinates.Angle`
        object is wrapped in place and nothing is returned.

        For instance::

          >>> from astropy.coordinates import Angle
          >>> import astropy.units as u
          >>> a = Angle([-20.0, 150.0, 350.0] * u.deg)

          >>> a.wrap_at(360 * u.deg).degree  # Wrap into range 0 to 360 degrees  # doctest: +FLOAT_CMP
          array([340., 150., 350.])

          >>> a.wrap_at('180d', inplace=True)  # Wrap into range -180 to 180 degrees  # doctest: +FLOAT_CMP
          >>> a.degree  # doctest: +FLOAT_CMP
          array([-20., 150., -10.])

        Parameters
        ----------
        wrap_angle : angle-like
            Specifies a single value for the wrap angle.  This can be any
            object that can initialize an `~astropy.coordinates.Angle` object,
            e.g. ``'180d'``, ``180 * u.deg``, or ``Angle(180, unit=u.deg)``.

        inplace : bool
            If `True` then wrap the object in place instead of returning
            a new `~astropy.coordinates.Angle`

        Returns
        -------
        out : Angle or None
            If ``inplace is False`` (default), return new
            `~astropy.coordinates.Angle` object with angles wrapped accordingly.
            Otherwise wrap in place and return `None`.
        """
        wrap_angle = Angle(wrap_angle, copy=False)  # Convert to an Angle
        if not inplace:
            self = self.copy()
        self._wrap_at(wrap_angle)
        return None if inplace else self

    def is_within_bounds(self, lower=None, upper=None):
        """
        Check if all angle(s) satisfy ``lower <= angle < upper``.

        If ``lower`` is not specified (or `None`) then no lower bounds check is
        performed.  Likewise ``upper`` can be left unspecified.  For example::

          >>> from astropy.coordinates import Angle
          >>> import astropy.units as u
          >>> a = Angle([-20, 150, 350] * u.deg)
          >>> a.is_within_bounds('0d', '360d')
          False
          >>> a.is_within_bounds(None, '360d')
          True
          >>> a.is_within_bounds(-30 * u.deg, None)
          True

        Parameters
        ----------
        lower : angle-like or None
            Specifies lower bound for checking.  This can be any object
            that can initialize an `~astropy.coordinates.Angle` object, e.g. ``'180d'``,
            ``180 * u.deg``, or ``Angle(180, unit=u.deg)``.
        upper : angle-like or None
            Specifies upper bound for checking.  This can be any object
            that can initialize an `~astropy.coordinates.Angle` object, e.g. ``'180d'``,
            ``180 * u.deg``, or ``Angle(180, unit=u.deg)``.

        Returns
        -------
        is_within_bounds : bool
            `True` if all angles satisfy ``lower <= angle < upper``
        """
        return bool(
            (lower is None or (Angle(lower) <= self).all())
            and (upper is None or (self < Angle(upper)).all())
        )

    def _str_helper(self, format=None):
        if self.isscalar:
            return self.to_string(format=format)
        return np.array2string(
            self, formatter={"all": lambda x: x.to_string(format=format)}
        )

    def __str__(self):
        return self._str_helper()

    def _repr_latex_(self):
        return self._str_helper(format="latex")


def _no_angle_subclass(obj):
    """Return any Angle subclass objects as an Angle objects.

    This is used to ensure that Latitude and Longitude change to Angle
    objects when they are used in calculations (such as lon/2.)
    """
    if isinstance(obj, tuple):
        return tuple(_no_angle_subclass(_obj) for _obj in obj)

    return obj.view(Angle) if isinstance(obj, (Latitude, Longitude)) else obj


class Latitude(Angle):
    """
    Latitude-like angle(s) which must be in the range -90 to +90 deg.

    A Latitude object is distinguished from a pure
    :class:`~astropy.coordinates.Angle` by virtue of being constrained
    so that::

      -90.0 * u.deg <= angle(s) <= +90.0 * u.deg

    Any attempt to set a value outside that range will result in a
    `ValueError`.

    The input angle(s) can be specified either as an array, list,
    scalar, tuple (see below), string,
    :class:`~astropy.units.Quantity` or another
    :class:`~astropy.coordinates.Angle`.

    The input parser is flexible and supports all of the input formats
    supported by :class:`~astropy.coordinates.Angle`.

    Parameters
    ----------
    angle : array, list, scalar, `~astropy.units.Quantity`, `~astropy.coordinates.Angle`
        The angle value(s). If a tuple, will be interpreted as ``(h, m, s)``
        or ``(d, m, s)`` depending on ``unit``. If a string, it will be
        interpreted following the rules described for
        :class:`~astropy.coordinates.Angle`.

        If ``angle`` is a sequence or array of strings, the resulting
        values will be in the given ``unit``, or if `None` is provided,
        the unit will be taken from the first given value.

    unit : unit-like, optional
        The unit of the value specified for the angle.  This may be
        any string that `~astropy.units.Unit` understands, but it is
        better to give an actual unit object.  Must be an angular
        unit.

    Raises
    ------
    `~astropy.units.UnitsError`
        If a unit is not provided or it is not an angular unit.
    `TypeError`
        If the angle parameter is an instance of :class:`~astropy.coordinates.Longitude`.
    """

    def __new__(cls, angle, unit=None, **kwargs):
        # Forbid creating a Lat from a Long.
        if isinstance(angle, Longitude) or (
            isinstance(angle, str) and angle.endswith(("E", "W"))
        ):
            raise TypeError("A Latitude angle cannot be created from a Longitude angle")
        self = super().__new__(cls, angle, unit=unit, **kwargs)
        self._validate_angles()
        return self

    def _validate_angles(self, angles=None):
        """Check that angles are between -90 and 90 degrees.
        If not given, the check is done on the object itself.
        """
        # Convert the lower and upper bounds to the "native" unit of
        # this angle.  This limits multiplication to two values,
        # rather than the N values in `self.value`.  Also, the
        # comparison is performed on raw arrays, rather than Quantity
        # objects, for speed.
        if angles is None:
            angles = self

        # For speed, compare using "is", which is not strictly guaranteed to hold,
        # but if it doesn't we'll just convert correctly in the 'else' clause.
        if angles.unit is u.deg:
            limit = 90
        elif angles.unit is u.rad:
            limit = 0.5 * np.pi
        else:
            limit = u.degree.to(angles.unit, 90.0)

        angles_view = angles.view(np.ndarray)
        if NUMPY_LT_2_0:
            # Ensure ndim>=1 so that comparison is done using the angle dtype.
            # Otherwise, e.g., np.array(np.pi/2, 'f4') > np.pi/2 will yield True.
            angles_view = angles_view[np.newaxis]

        if np.any(np.abs(angles_view) > limit):
            if np.size(angles) < 5:
                raise ValueError(
                    "Latitude angle(s) must be within -90 deg <= angle "
                    f"<= 90 deg, got {angles.to(u.degree)}"
                )
            else:
                raise ValueError(
                    "Latitude angle(s) must be within -90 deg <= angle "
                    f"<= 90 deg, got {angles.min().to(u.degree)} <= "
                    f"angle <= {angles.max().to(u.degree)}"
                )

    def __setitem__(self, item, value):
        # Forbid assigning a Long to a Lat.
        if isinstance(value, Longitude):
            raise TypeError("A Longitude angle cannot be assigned to a Latitude angle")
        # first check bounds
        if value is not np.ma.masked:
            self._validate_angles(value)
        super().__setitem__(item, value)

    # Any calculation should drop to Angle
    def __array_ufunc__(self, *args, **kwargs):
        results = super().__array_ufunc__(*args, **kwargs)
        return _no_angle_subclass(results)


class LongitudeInfo(u.QuantityInfo):
    _represent_as_dict_attrs = u.QuantityInfo._represent_as_dict_attrs + ("wrap_angle",)


class Longitude(Angle):
    """
    Longitude-like angle(s) which are wrapped within a contiguous 360 degree range.

    A ``Longitude`` object is distinguished from a pure
    :class:`~astropy.coordinates.Angle` by virtue of a ``wrap_angle``
    property.  The ``wrap_angle`` specifies that all angle values
    represented by the object will be in the range::

      wrap_angle - 360 * u.deg <= angle(s) < wrap_angle

    The default ``wrap_angle`` is 360 deg.  Setting ``wrap_angle=180 *
    u.deg`` would instead result in values between -180 and +180 deg.
    Setting the ``wrap_angle`` attribute of an existing ``Longitude``
    object will result in re-wrapping the angle values in-place.

    The input angle(s) can be specified either as an array, list,
    scalar, tuple, string, :class:`~astropy.units.Quantity`
    or another :class:`~astropy.coordinates.Angle`.

    The input parser is flexible and supports all of the input formats
    supported by :class:`~astropy.coordinates.Angle`.

    Parameters
    ----------
    angle : tuple or angle-like
        The angle value(s). If a tuple, will be interpreted as ``(h, m s)`` or
        ``(d, m, s)`` depending on ``unit``. If a string, it will be interpreted
        following the rules described for :class:`~astropy.coordinates.Angle`.

        If ``angle`` is a sequence or array of strings, the resulting
        values will be in the given ``unit``, or if `None` is provided,
        the unit will be taken from the first given value.

    unit : unit-like ['angle'], optional
        The unit of the value specified for the angle.  This may be
        any string that `~astropy.units.Unit` understands, but it is
        better to give an actual unit object.  Must be an angular
        unit.

    wrap_angle : angle-like or None, optional
        Angle at which to wrap back to ``wrap_angle - 360 deg``.
        If ``None`` (default), it will be taken to be 360 deg unless ``angle``
        has a ``wrap_angle`` attribute already (i.e., is a ``Longitude``),
        in which case it will be taken from there.

    Raises
    ------
    `~astropy.units.UnitsError`
        If a unit is not provided or it is not an angular unit.
    `TypeError`
        If the angle parameter is an instance of :class:`~astropy.coordinates.Latitude`.
    """

    _wrap_angle = None
    _default_wrap_angle = Angle(360 * u.deg)
    info = LongitudeInfo()

    def __new__(cls, angle, unit=None, wrap_angle=None, **kwargs):
        # Forbid creating a Long from a Lat.
        if isinstance(angle, Latitude) or (
            isinstance(angle, str) and angle.endswith(("N", "S"))
        ):
            raise TypeError(
                "A Longitude angle cannot be created from a Latitude angle."
            )
        self = super().__new__(cls, angle, unit=unit, **kwargs)
        if wrap_angle is None:
            wrap_angle = getattr(angle, "wrap_angle", self._default_wrap_angle)
        self.wrap_angle = wrap_angle  # angle-like b/c property setter
        return self

    def __setitem__(self, item, value):
        # Forbid assigning a Lat to a Long.
        if isinstance(value, Latitude):
            raise TypeError("A Latitude angle cannot be assigned to a Longitude angle")
        super().__setitem__(item, value)
        self._wrap_at(self.wrap_angle)

    @property
    def wrap_angle(self):
        return self._wrap_angle

    @wrap_angle.setter
    def wrap_angle(self, value):
        self._wrap_angle = Angle(value, copy=False)
        self._wrap_at(self.wrap_angle)

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self._wrap_angle = getattr(obj, "_wrap_angle", self._default_wrap_angle)

    # Any calculation should drop to Angle
    def __array_ufunc__(self, *args, **kwargs):
        results = super().__array_ufunc__(*args, **kwargs)
        return _no_angle_subclass(results)
