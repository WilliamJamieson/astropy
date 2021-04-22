# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Tests for spline models and fitters"""
# pylint: disable=invalid-name
from astropy.utils.exceptions import AstropyUserWarning
import warnings

import pytest
import unittest.mock as mk
import numpy as np

from numpy.testing import assert_allclose

from astropy.utils.compat.optional_deps import HAS_SCIPY  # noqa

from astropy.utils.exceptions import (AstropyUserWarning,)
from astropy.modeling.core import (ModelDefinitionError,)
from astropy.modeling.spline import (_Spline, Spline1D, Spline2D, SplineFitter, SplineLSQFitter)


class TestSpline:
    def setup_class(self):
        self.spline = mk.MagicMock()

        self.bounding_box = mk.MagicMock()
        self.bbox = mk.MagicMock()
        self.bounding_box.bbox = self.bbox

        self.num_opt = 3
        self.optional_inputs = {f'test{i}': mk.MagicMock() for i in range(self.num_opt)}
        self.extra_kwargs = {f'new{i}': mk.MagicMock() for i in range(self.num_opt)}

    def test_spline(self):
        class Spline(_Spline):
            _spline = None

        spl = Spline()
        assert spl._spline is None
        assert spl.spline is None

        spl.spline = self.spline
        assert spl._spline == self.spline
        assert spl.spline == self.spline

        assert spl.spline is not None
        with pytest.warns(AstropyUserWarning,
                          match=r'Spline already defined for this model.*'):
            spl.spline = mk.MagicMock()

    def test_reset_spline(self):
        class Spline(_Spline):
            _spline = None

        spl = Spline()
        spl.spline = self.spline
        assert spl.spline is not None

        spl.reset_spline()
        assert spl.spline is None
        spl.spline = self.spline
        assert spl.spline == self.spline

    def test_bbox(self):
        class Spline(_Spline):
            n_inputs = 1
            _spline = None
            _bounding_box = None

            @property
            def bounding_box(self):
                if self._bounding_box is None:
                    raise NotImplementedError
                else:
                    return self._bounding_box

        spl = Spline()
        assert spl.bbox == [None, None]
        spl.n_inputs = 2
        assert spl.bbox == [None, None, None, None]

        spl._bounding_box = self.bounding_box
        assert spl.bbox == self.bbox

    def test__optional_arg(self):
        class Spline(_Spline):
            pass

        spl = Spline()
        assert spl._optional_arg('test') == '_test'

    def test__create_optional_inputs(self):
        class Spline(_Spline):
            optional_inputs = self.optional_inputs

            def __init__(self):
                self._create_optional_inputs()

        spl = Spline()
        for arg in self.optional_inputs:
            attribute = spl._optional_arg(arg)
            assert hasattr(spl, attribute)
            assert getattr(spl, attribute) is None

        with pytest.raises(ValueError,
                           match=r"Optional argument .* already exists in this class!"):
            spl._create_optional_inputs()

    def test__intercept_optional_inputs(self):
        class Spline(_Spline):
            optional_inputs = self.optional_inputs

            def __init__(self):
                self._create_optional_inputs()

        spl = Spline()
        new_kwargs = spl._intercept_optional_inputs(**self.extra_kwargs)
        for arg, value in self.optional_inputs.items():
            attribute = spl._optional_arg(arg)
            assert getattr(spl, attribute) is None
        assert new_kwargs == self.extra_kwargs

        kwargs = self.extra_kwargs.copy()
        for arg in self.optional_inputs:
            kwargs[arg] = mk.MagicMock()
        new_kwargs = spl._intercept_optional_inputs(**kwargs)
        for arg, value in self.optional_inputs.items():
            attribute = spl._optional_arg(arg)
            assert getattr(spl, attribute) is not None
            assert getattr(spl, attribute) == kwargs[arg]
            assert getattr(spl, attribute) != value
            assert arg not in new_kwargs
        assert new_kwargs == self.extra_kwargs
        assert kwargs != self.extra_kwargs

        with pytest.raises(RuntimeError,
                           match=r".* has already been set, something has gone wrong!"):
            spl._intercept_optional_inputs(**kwargs)

    def test__get_optional_inputs(self):
        class Spline(_Spline):
            optional_inputs = self.optional_inputs

            def __init__(self):
                self._create_optional_inputs()

        spl = Spline()

        # No options passed in and No options set
        new_kwargs = spl._get_optional_inputs(**self.extra_kwargs)
        for arg, value in self.optional_inputs.items():
            assert new_kwargs[arg] == value
        for arg, value in self.extra_kwargs.items():
            assert new_kwargs[arg] == value
        assert len(new_kwargs) == (len(self.optional_inputs) + len(self.extra_kwargs))

        # No options passed in and Options set
        kwargs = self.extra_kwargs.copy()
        for arg in self.optional_inputs:
            kwargs[arg] = mk.MagicMock()
        spl._intercept_optional_inputs(**kwargs)
        new_kwargs = spl._get_optional_inputs(**self.extra_kwargs)
        assert new_kwargs == kwargs
        for arg in self.optional_inputs:
            attribute = spl._optional_arg(arg)
            assert getattr(spl, attribute) is None

        # Options passed in
        set_kwargs = self.extra_kwargs.copy()
        for arg in self.optional_inputs:
            kwargs[arg] = mk.MagicMock()
        spl._intercept_optional_inputs(**set_kwargs)
        kwargs = self.extra_kwargs.copy()
        for arg in self.optional_inputs:
            kwargs[arg] = mk.MagicMock()
        assert set_kwargs != kwargs
        new_kwargs = spl._get_optional_inputs(**kwargs)
        assert new_kwargs == kwargs


@pytest.mark.skipif('not HAS_SCIPY')
class TestSpline1D:
    """Test Spline 1D"""

    def setup_class(self):
        np.random.seed(42)
        self.npts = 50

        self.x = np.linspace(-3, 3, self.npts)
        self.y = np.exp(-self.x**2) + 0.1 * np.random.randn(self.npts)
        self.w = np.arange(self.npts)
        self.t = [-1, 0, 1]

        self.npts_out = 1000
        self.xs = np.linspace(-3, 3, self.npts_out)

    def generate_spline(self, w=None, bbox=[None]*2, k=None, s=None,
                        ext=None, check_finte=None):
        if k is None:
            k = 3
        if ext is None:
            ext = 0
        if check_finte is None:
            check_finte = False

        from scipy.interpolate import UnivariateSpline

        return UnivariateSpline(self.x, self.y, w=w, bbox=bbox, k=k, s=s,
                                ext=ext, check_finite=check_finte)

    def generate_LSQ_spline(self, w=None, bbox=[None]*2, k=None, ext=None,
                            check_finte=None):
        if k is None:
            k = 3
        if ext is None:
            ext = 0
        if check_finte is None:
            check_finte = False

        from scipy.interpolate import LSQUnivariateSpline

        return LSQUnivariateSpline(self.x, self.y, self.t, w=w, bbox=bbox,
                                   k=k, ext=ext, check_finite=check_finte)

    def test___init__(self):
        # check  defaults
        spl = Spline1D()
        assert spl._k == 3
        assert spl._ext == 0
        assert not spl._check_finite
        assert spl._nu is None

        # check non-defaults
        spl = Spline1D(1, 2, True)
        assert spl._k == 1
        assert spl._ext == 2
        assert spl._check_finite
        assert spl._nu is None

    def test_spline(self):
        spl = Spline1D()
        assert spl.spline is None

        spl.spline = self.generate_spline()
        truth = self.generate_spline()
        assert spl.spline == spl._spline
        assert (spl.spline(self.xs) == truth(self.xs)).all()

        assert spl.spline is not None
        with pytest.warns(AstropyUserWarning,
                          match=r'Spline already defined for this model.*'):
            spl.spline = self.generate_spline()

    def test_reset_spline(self):
        spl = Spline1D()
        spl._spline = self.generate_spline()

        spl.reset_spline()
        assert spl._spline is None

    def test_evaluate(self):
        spl = Spline1D()
        spl._spline = self.generate_spline()
        truth = self.generate_spline()

        assert (spl.evaluate(self.xs) == truth(self.xs)).all()

        # direct derivative set
        assert (spl.evaluate(self.xs, nu=0) == truth(self.xs, nu=0)).all()
        assert (spl.evaluate(self.xs, nu=1) == truth(self.xs, nu=1)).all()
        assert (spl.evaluate(self.xs, nu=2) == truth(self.xs, nu=2)).all()
        assert (spl.evaluate(self.xs, nu=3) == truth(self.xs, nu=3)).all()

        # direct derivative call overrides internal
        spl._nu = 4
        assert (spl.evaluate(self.xs, nu=0) == truth(self.xs, nu=0)).all()
        assert (spl.evaluate(self.xs, nu=1) == truth(self.xs, nu=1)).all()
        assert (spl.evaluate(self.xs, nu=2) == truth(self.xs, nu=2)).all()
        assert (spl.evaluate(self.xs, nu=3) == truth(self.xs, nu=3)).all()

        # internal sets derivative and then gets reset
        spl._nu = 0
        assert (spl.evaluate(self.xs) == truth(self.xs, nu=0)).all()
        assert spl._nu is None
        spl._nu = 1
        assert (spl.evaluate(self.xs) == truth(self.xs, nu=1)).all()
        assert spl._nu is None
        spl._nu = 2
        assert (spl.evaluate(self.xs) == truth(self.xs, nu=2)).all()
        assert spl._nu is None
        spl._nu = 3
        assert (spl.evaluate(self.xs) == truth(self.xs, nu=3)).all()
        assert spl._nu is None

    def test___call__(self):
        spl = Spline1D()
        truth = np.random.rand(self.npts_out)

        with mk.patch.object(Spline1D, 'evaluate', autospec=True,
                             return_value=truth) as mkEval:
            value = spl(self.xs)
            assert (value == truth).all()
            assert mkEval.call_args_list == [mk.call(spl, self.xs)]
            assert spl._nu is None

            mkEval.reset_mock()
            value = spl(self.xs, nu=1)
            assert (value == truth).all()
            assert mkEval.call_args_list == [mk.call(spl, self.xs)]
            assert spl._nu == 1

            mkEval.reset_mock()
            with pytest.raises(RuntimeError,
                               match=r"nu has already been set.*"):
                spl(self.xs, nu=2)
            assert mkEval.call_args_list == []
            assert spl._nu == 1

    def test_bbox(self):
        spl = Spline1D()
        assert spl.bbox == [None, None]

        spl.bounding_box = (1, 2)
        assert spl.bbox == [1, 2]

    def test_fit_spline(self):
        spl = Spline1D()
        truth = self.generate_spline()
        spl.fit_spline(self.x, self.y)
        assert spl._spline is not None

        assert (spl(self.xs) == truth(self.xs)).all()
        assert (spl(self.xs, nu=1) == truth(self.xs, nu=1)).all()
        assert (spl(self.xs, nu=2) == truth(self.xs, nu=2)).all()
        assert (spl(self.xs, nu=3) == truth(self.xs, nu=3)).all()

        # Test warning
        spl = Spline1D()
        spl._spline = self.generate_spline()
        assert spl._spline is not None
        with pytest.warns(AstropyUserWarning,
                          match=r'Spline already defined for this model.*'):
            spl.fit_spline(self.x, self.y)

        spl = Spline1D(k=1)
        truth = self.generate_spline(k=1)
        spl.fit_spline(self.x, self.y)
        assert spl._spline is not None

        assert (spl(self.xs) == truth(self.xs)).all()
        assert (spl(self.xs, nu=1) == truth(self.xs, nu=1)).all()

        spl = Spline1D()
        spl.bounding_box = (-4, 4)
        truth = self.generate_spline(bbox=(-4, 4))
        spl.fit_spline(self.x, self.y,)
        assert spl._spline is not None

        assert (spl(self.xs) == truth(self.xs)).all()
        assert (spl(self.xs, nu=1) == truth(self.xs, nu=1)).all()
        assert (spl(self.xs, nu=2) == truth(self.xs, nu=2)).all()
        assert (spl(self.xs, nu=3) == truth(self.xs, nu=3)).all()

        spl = Spline1D()
        truth = self.generate_spline(self.w)
        spl.fit_spline(self.x, self.y, w=self.w)
        assert spl._spline is not None

        assert (spl(self.xs) == truth(self.xs)).all()
        assert (spl(self.xs, nu=1) == truth(self.xs, nu=1)).all()
        assert (spl(self.xs, nu=2) == truth(self.xs, nu=2)).all()
        assert (spl(self.xs, nu=3) == truth(self.xs, nu=3)).all()

    def test_SplineFitter(self):
        fitter = SplineFitter()
        model = Spline1D()
        truth = self.generate_spline()

        fit = fitter(model, self.x, self.y)
        assert id(fit) != id(model)
        assert model._spline is None
        assert fit._spline is not None
        assert (fit(self.xs) == truth(self.xs)).all()

        with pytest.raises(ValueError,
                           match=r"1D model can only have 2 data points."):
            fitter(model, self.x, self.y, self.w)

        with pytest.raises(ModelDefinitionError,
                           match=r"Only spline models are compatible with this fitter"):
            fitter(mk.MagicMock(), self.x, self.y)

    def test_fit_LSQ_spline(self):
        spl = Spline1D()
        truth = self.generate_LSQ_spline()
        spl.fit_LSQ_spline(self.x, self.y, self.t)
        assert spl._spline is not None

        assert (spl(self.xs) == truth(self.xs)).all()
        assert (spl(self.xs, nu=1) == truth(self.xs, nu=1)).all()
        assert (spl(self.xs, nu=2) == truth(self.xs, nu=2)).all()
        assert (spl(self.xs, nu=3) == truth(self.xs, nu=3)).all()

        # Test warning
        spl._spline = self.generate_spline()
        assert spl._spline is not None
        with pytest.warns(AstropyUserWarning,
                          match=r'Spline already defined for this model.*'):
            spl.fit_LSQ_spline(self.x, self.y, self.t)

        spl = Spline1D(k=1)
        truth = self.generate_LSQ_spline(k=1)
        spl.fit_LSQ_spline(self.x, self.y, self.t)
        assert spl._spline is not None

        assert (spl(self.xs) == truth(self.xs)).all()
        assert (spl(self.xs, nu=1) == truth(self.xs, nu=1)).all()

        spl = Spline1D()
        spl.bounding_box = (-4, 4)
        truth = self.generate_LSQ_spline(bbox=(-4, 4))
        spl.fit_LSQ_spline(self.x, self.y, self.t)
        assert spl._spline is not None

        assert (spl(self.xs) == truth(self.xs)).all()
        assert (spl(self.xs, nu=1) == truth(self.xs, nu=1)).all()
        assert (spl(self.xs, nu=2) == truth(self.xs, nu=2)).all()
        assert (spl(self.xs, nu=3) == truth(self.xs, nu=3)).all()

        spl = Spline1D()
        truth = self.generate_LSQ_spline(self.w)
        spl.fit_LSQ_spline(self.x, self.y, self.t, w=self.w)
        assert spl._spline is not None

        assert (spl(self.xs) == truth(self.xs)).all()
        assert (spl(self.xs, nu=1) == truth(self.xs, nu=1)).all()
        assert (spl(self.xs, nu=2) == truth(self.xs, nu=2)).all()
        assert (spl(self.xs, nu=3) == truth(self.xs, nu=3)).all()

    def test_SplineLSQFitter(self):
        fitter = SplineLSQFitter()
        model = Spline1D()
        truth = self.generate_LSQ_spline()

        fit = fitter(model, self.t, self.x, self.y)
        assert id(fit) != id(model)
        assert model._spline is None
        assert fit._spline is not None
        assert (fit(self.xs) == truth(self.xs)).all()

        with pytest.raises(ValueError,
                           match=r"1D model can only have 2 data points."):
            fitter(model, self.t, self.x, self.y, self.w)

        with pytest.raises(ModelDefinitionError,
                           match=r"Only spline models are compatible with this fitter"):
            fitter(mk.MagicMock(), self.t, self.x, self.y)


@pytest.mark.skipif('not HAS_SCIPY')
class TestSpline2D:
    """Test Spline 2D"""

    def setup_class(self):
        np.random.seed(42)
        self.npts = 50

        self.x = np.linspace(-3, 3, self.npts)
        self.y = np.linspace(-3, 3, self.npts)
        self.z = np.exp(-self.x**2 - self.y**2) + 0.1 * np.random.randn(self.npts)
        self.w = np.random.rand(self.npts)
        self.tx = [-1, 0, 1]
        self.ty = [-1, 0, 1]

        self.npts_out = 1000
        self.xs = np.linspace(-3, 3, self.npts_out)
        self.ys = np.linspace(-3, 3, self.npts_out)

    def generate_spline(self, w=None, bbox=[None]*4, kx=None, ky=None,
                        s=None, eps=None):
        if kx is None:
            kx = 3
        if ky is None:
            ky = 3
        if eps is None:
            eps = 1e-16

        from scipy.interpolate import SmoothBivariateSpline

        return SmoothBivariateSpline(self.x, self.y, self.z, w=w, bbox=bbox,
                                     kx=kx, ky=ky, s=s, eps=eps)

    def generate_LSQ_spline(self, w=None, bbox=[None]*4, kx=None, ky=None,
                            eps=None):
        if kx is None:
            kx = 3
        if ky is None:
            ky = 3
        if eps is None:
            eps = 1e-16

        from scipy.interpolate import LSQBivariateSpline

        return LSQBivariateSpline(self.x, self.y, self.z, self.tx, self.ty,
                                  w=w, bbox=bbox, kx=kx, ky=ky, eps=eps)

    def test___init__(self):
        # check  defaults
        spl = Spline2D()
        assert spl._kx == 3
        assert spl._ky == 3
        assert spl._eps == 1e-16
        assert spl._dx is None
        assert spl._dy is None

        # check non-defaults
        spl = Spline2D(1, 2, 3)
        assert spl._kx == 1
        assert spl._ky == 2
        assert spl._eps == 3
        assert spl._dx is None
        assert spl._dy is None

    def test_spline(self):
        spl = Spline2D()
        assert spl.spline is None

        spl.spline = self.generate_spline()
        truth = self.generate_spline()
        assert spl.spline == spl._spline
        assert (spl.spline(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        assert spl.spline is not None
        with pytest.warns(AstropyUserWarning,
                          match=r'Spline already defined for this model.*'):
            spl.spline = self.generate_spline()

    def test_reset_spline(self):
        spl = Spline2D()
        spl._spline = self.generate_spline()

        spl.reset_spline()
        assert spl._spline is None

    def test_evaluate(self):
        spl = Spline2D()
        spl._spline = self.generate_spline()
        truth = self.generate_spline()

        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        # direct derivative set
        assert (spl.evaluate(self.xs, self.ys, dx=0) == truth(self.xs, self.ys, dx=0)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=1) == truth(self.xs, self.ys, dx=1)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=2) == truth(self.xs, self.ys, dx=2)).all()

        assert (spl.evaluate(self.xs, self.ys, dy=0) == truth(self.xs, self.ys, dy=0)).all()
        assert (spl.evaluate(self.xs, self.ys, dy=1) == truth(self.xs, self.ys, dy=1)).all()
        assert (spl.evaluate(self.xs, self.ys, dy=2) == truth(self.xs, self.ys, dy=2)).all()

        assert (spl.evaluate(self.xs, self.ys, dx=1, dy=1) ==
                truth(self.xs, self.ys, dx=1, dy=1)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=2, dy=1) ==
                truth(self.xs, self.ys, dx=2, dy=1)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=1, dy=2) ==
                truth(self.xs, self.ys, dx=1, dy=2)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=2, dy=2) ==
                truth(self.xs, self.ys, dx=2, dy=2)).all()

        # direct derivative call overrides internal
        spl._dx = 3
        assert (spl.evaluate(self.xs, self.ys, dx=0) == truth(self.xs, self.ys, dx=0)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=1) == truth(self.xs, self.ys, dx=1)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=2) == truth(self.xs, self.ys, dx=2)).all()
        spl._dx = None
        spl._dy = 3
        assert (spl.evaluate(self.xs, self.ys, dy=0) == truth(self.xs, self.ys, dy=0)).all()
        assert (spl.evaluate(self.xs, self.ys, dy=1) == truth(self.xs, self.ys, dy=1)).all()
        assert (spl.evaluate(self.xs, self.ys, dy=2) == truth(self.xs, self.ys, dy=2)).all()
        spl._dx = 3
        spl._dy = 3
        assert (spl.evaluate(self.xs, self.ys, dx=1, dy=1) ==
                truth(self.xs, self.ys, dx=1, dy=1)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=2, dy=1) ==
                truth(self.xs, self.ys, dx=2, dy=1)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=1, dy=2) ==
                truth(self.xs, self.ys, dx=1, dy=2)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=2, dy=2) ==
                truth(self.xs, self.ys, dx=2, dy=2)).all()
        spl._dx = None
        spl._dy = None

        # internal sets derivative and then gets reset
        spl._dx = 0
        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys, dx=0)).all()
        assert spl._dx is None
        spl._dx = 1
        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys, dx=1)).all()
        assert spl._dx is None
        spl._dx = 2
        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys, dx=2)).all()
        assert spl._dx is None
        spl._dy = 0
        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys, dy=0)).all()
        assert spl._dy is None
        spl._dy = 1
        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys, dy=1)).all()
        assert spl._dy is None
        spl._dy = 2
        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys, dy=2)).all()
        assert spl._dy is None
        spl._dx = 1
        spl._dy = 1
        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys, dx=1, dy=1)).all()
        assert spl._dx is None
        assert spl._dy is None
        spl._dx = 2
        spl._dy = 1
        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys, dx=2, dy=1)).all()
        assert spl._dx is None
        assert spl._dy is None
        spl._dx = 1
        spl._dy = 2
        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys, dx=1, dy=2)).all()
        assert spl._dx is None
        assert spl._dy is None
        spl._dx = 2
        spl._dy = 2
        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys, dx=2, dy=2)).all()
        assert spl._dx is None
        assert spl._dy is None

    def test___call__(self):
        spl = Spline2D()
        truth = np.random.rand(self.npts_out)

        with mk.patch.object(Spline2D, 'evaluate', autospec=True,
                             return_value=truth) as mkEval:
            value = spl(self.xs, self.ys)
            assert (value == truth).all()
            assert mkEval.call_args_list == [mk.call(spl, self.xs, self.ys)]
            assert spl._dx is None
            assert spl._dy is None

            mkEval.reset_mock()
            value = spl(self.xs, self.ys, dx=1)
            assert (value == truth).all()
            assert mkEval.call_args_list == [mk.call(spl, self.xs, self.ys)]
            assert spl._dx == 1
            assert spl._dy is None

            mkEval.reset_mock()
            with pytest.raises(RuntimeError,
                               match=r"dx has already been set.*"):
                spl(self.xs, self.ys, dx=2)
            assert mkEval.call_args_list == []
            assert spl._dx == 1
            assert spl._dy is None

            spl._dx = None
            mkEval.reset_mock()
            value = spl(self.xs, self.ys, dy=1)
            assert (value == truth).all()
            assert mkEval.call_args_list == [mk.call(spl, self.xs, self.ys)]
            assert spl._dx is None
            assert spl._dy == 1

            mkEval.reset_mock()
            with pytest.raises(RuntimeError,
                               match=r"dy has already been set.*"):
                spl(self.xs, self.ys, dy=2)
            assert mkEval.call_args_list == []
            assert spl._dx is None
            assert spl._dy == 1

    def test_bbox(self):
        spl = Spline2D()
        assert spl.bbox == [None, None, None, None]

        spl.bounding_box = ((1, 2), (3, 4))
        assert spl.bbox == [1, 2, 3, 4]

    def test_fit_spline(self):
        spl = Spline2D()
        truth = self.generate_spline()
        spl.fit_spline(self.x, self.y, self.z)
        assert spl._spline is not None

        assert (spl(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        assert (spl(self.xs, self.ys, dx=1) == truth(self.xs, self.ys, dx=1)).all()
        assert (spl(self.xs, self.ys, dx=2) == truth(self.xs, self.ys, dx=2)).all()

        assert (spl(self.xs, self.ys, dy=1) == truth(self.xs, self.ys, dy=1)).all()
        assert (spl(self.xs, self.ys, dy=2) == truth(self.xs, self.ys, dy=2)).all()

        assert (spl(self.xs, self.ys, dx=1, dy=1) ==
                truth(self.xs, self.ys, dx=1, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=1) ==
                truth(self.xs, self.ys, dx=2, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=1, dy=2) ==
                truth(self.xs, self.ys, dx=1, dy=2)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=2) ==
                truth(self.xs, self.ys, dx=2, dy=2)).all()

        # Test warning
        spl = Spline2D()
        spl._spline = self.generate_spline()
        assert spl._spline is not None
        with pytest.warns(AstropyUserWarning,
                          match=r'Spline already defined for this model.*'):
            spl.fit_spline(self.x, self.y, self.z)

        spl = Spline2D(kx=1, ky=1)
        truth = self.generate_spline(kx=1, ky=1)
        spl.fit_spline(self.x, self.y, self.z)
        assert spl._spline is not None

        assert (spl(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        spl = Spline2D()
        spl.bounding_box = ((-4, 4), (-4, 4))
        truth = self.generate_spline(bbox=(-4, 4, -4, 4))
        spl.fit_spline(self.x, self.y, self.z)
        assert spl._spline is not None

        assert (spl(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        assert (spl(self.xs, self.ys, dx=1) == truth(self.xs, self.ys, dx=1)).all()
        assert (spl(self.xs, self.ys, dx=2) == truth(self.xs, self.ys, dx=2)).all()

        assert (spl(self.xs, self.ys, dy=1) == truth(self.xs, self.ys, dy=1)).all()
        assert (spl(self.xs, self.ys, dy=2) == truth(self.xs, self.ys, dy=2)).all()

        assert (spl(self.xs, self.ys, dx=1, dy=1) ==
                truth(self.xs, self.ys, dx=1, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=1) ==
                truth(self.xs, self.ys, dx=2, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=1, dy=2) ==
                truth(self.xs, self.ys, dx=1, dy=2)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=2) ==
                truth(self.xs, self.ys, dx=2, dy=2)).all()

        spl = Spline2D()
        truth = self.generate_spline(self.w)
        spl.fit_spline(self.x, self.y, self.z, w=self.w)
        assert spl._spline is not None

        assert (spl(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        assert (spl(self.xs, self.ys, dx=1) == truth(self.xs, self.ys, dx=1)).all()
        assert (spl(self.xs, self.ys, dx=2) == truth(self.xs, self.ys, dx=2)).all()

        assert (spl(self.xs, self.ys, dy=1) == truth(self.xs, self.ys, dy=1)).all()
        assert (spl(self.xs, self.ys, dy=2) == truth(self.xs, self.ys, dy=2)).all()

        assert (spl(self.xs, self.ys, dx=1, dy=1) ==
                truth(self.xs, self.ys, dx=1, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=1) ==
                truth(self.xs, self.ys, dx=2, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=1, dy=2) ==
                truth(self.xs, self.ys, dx=1, dy=2)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=2) ==
                truth(self.xs, self.ys, dx=2, dy=2)).all()

    def test_SplineFitter(self):
        fitter = SplineFitter()
        model = Spline2D()
        truth = self.generate_spline()

        fit = fitter(model, self.x, self.y, self.z)
        assert id(fit) != id(model)
        assert model._spline is None
        assert fit._spline is not None
        assert (fit(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        with pytest.raises(ValueError,
                           match=r"2D model must have 3 data points."):
            fitter(model, self.x, self.y)

        with pytest.raises(ModelDefinitionError,
                           match=r"Only spline models are compatible with this fitter"):
            fitter(mk.MagicMock(), self.x, self.y)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_fit_LSQ_spline(self):
        spl = Spline2D()
        truth = self.generate_LSQ_spline()
        spl.fit_LSQ_spline(self.x, self.y, self.z, self.tx, self.ty)
        assert spl._spline is not None

        assert (spl(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        assert (spl(self.xs, self.ys, dx=1) == truth(self.xs, self.ys, dx=1)).all()
        assert (spl(self.xs, self.ys, dx=2) == truth(self.xs, self.ys, dx=2)).all()

        assert (spl(self.xs, self.ys, dy=1) == truth(self.xs, self.ys, dy=1)).all()
        assert (spl(self.xs, self.ys, dy=2) == truth(self.xs, self.ys, dy=2)).all()

        assert (spl(self.xs, self.ys, dx=1, dy=1) ==
                truth(self.xs, self.ys, dx=1, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=1) ==
                truth(self.xs, self.ys, dx=2, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=1, dy=2) ==
                truth(self.xs, self.ys, dx=1, dy=2)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=2) ==
                truth(self.xs, self.ys, dx=2, dy=2)).all()

        # Test warning
        spl = Spline2D()
        spl._spline = self.generate_LSQ_spline()
        assert spl._spline is not None
        with pytest.warns(AstropyUserWarning,
                          match=r'Spline already defined for this model.*'):
            spl.fit_LSQ_spline(self.x, self.y, self.z, self.tx, self.ty)

        spl = Spline2D(kx=1, ky=1)
        truth = self.generate_LSQ_spline(kx=1, ky=1)
        spl.fit_LSQ_spline(self.x, self.y, self.z, self.tx, self.ty)
        assert spl._spline is not None

        assert (spl(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        spl = Spline2D()
        spl.bounding_box = ((-4, 4), (-4, 4))
        truth = self.generate_LSQ_spline(bbox=(-4, 4, -4, 4))
        spl.fit_LSQ_spline(self.x, self.y, self.z, self.tx, self.ty)
        assert spl._spline is not None

        assert (spl(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        assert (spl(self.xs, self.ys, dx=1) == truth(self.xs, self.ys, dx=1)).all()
        assert (spl(self.xs, self.ys, dx=2) == truth(self.xs, self.ys, dx=2)).all()

        assert (spl(self.xs, self.ys, dy=1) == truth(self.xs, self.ys, dy=1)).all()
        assert (spl(self.xs, self.ys, dy=2) == truth(self.xs, self.ys, dy=2)).all()

        assert (spl(self.xs, self.ys, dx=1, dy=1) ==
                truth(self.xs, self.ys, dx=1, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=1) ==
                truth(self.xs, self.ys, dx=2, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=1, dy=2) ==
                truth(self.xs, self.ys, dx=1, dy=2)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=2) ==
                truth(self.xs, self.ys, dx=2, dy=2)).all()

        spl = Spline2D()
        truth = self.generate_LSQ_spline(self.w)
        spl.fit_LSQ_spline(self.x, self.y, self.z, self.tx, self.ty, w=self.w)
        assert spl._spline is not None

        assert (spl(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        assert (spl(self.xs, self.ys, dx=1) == truth(self.xs, self.ys, dx=1)).all()
        assert (spl(self.xs, self.ys, dx=2) == truth(self.xs, self.ys, dx=2)).all()

        assert (spl(self.xs, self.ys, dy=1) == truth(self.xs, self.ys, dy=1)).all()
        assert (spl(self.xs, self.ys, dy=2) == truth(self.xs, self.ys, dy=2)).all()

        assert (spl(self.xs, self.ys, dx=1, dy=1) ==
                truth(self.xs, self.ys, dx=1, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=1) ==
                truth(self.xs, self.ys, dx=2, dy=1)).all()
        assert (spl(self.xs, self.ys, dx=1, dy=2) ==
                truth(self.xs, self.ys, dx=1, dy=2)).all()
        assert (spl(self.xs, self.ys, dx=2, dy=2) ==
                truth(self.xs, self.ys, dx=2, dy=2)).all()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_SplineLSQFitter(self):
        fitter = SplineLSQFitter()
        model = Spline2D()
        truth = self.generate_LSQ_spline()

        fit = fitter(model, (self.tx, self.ty), self.x, self.y, self.z)
        assert id(fit) != id(model)
        assert model._spline is None
        assert fit._spline is not None
        assert (fit(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        with pytest.raises(ValueError,
                           match=r"2D model must have 3 data points."):
            fitter(model, (self.tx, self.ty), self.x, self.y)

        with pytest.raises(ValueError,
                           match=r"Must have both x and y knots defined"):
            fitter(model, self.tx, self.x, self.y, self.z)

        with pytest.raises(ModelDefinitionError,
                           match=r"Only spline models are compatible with this fitter"):
            fitter(mk.MagicMock(), self.tx, self.x, self.y)
