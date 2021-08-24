# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Tests for spline models and fitters"""
# pylint: disable=invalid-name
from astropy.utils.exceptions import AstropyUserWarning
import warnings

import pytest
import unittest.mock as mk
from numpy.testing import assert_allclose

import numpy as np

from numpy.testing import assert_allclose

from astropy.utils.compat.optional_deps import HAS_SCIPY  # noqa

from astropy.utils.exceptions import (AstropyUserWarning,)
from astropy.modeling.core import (ModelDefinitionError,)
from astropy.modeling.spline import (_Spline, Spline1D, Spline2D, NewSpline1D)
from astropy.modeling.fitting import SplineFitter
from astropy.modeling.parameters import Parameter

npts = 50
nknots = 10
np.random.seed(42)
test_w = np.random.rand(npts)
test_t = [-1, 0, 1]
noise = np.random.randn(npts)


class TestSpline:
    def setup_class(self):
        self.bounding_box = mk.MagicMock()
        self.bbox = mk.MagicMock()
        self.bounding_box.bbox = self.bbox

        self.num_opt = 3
        self.optional_inputs = {f'test{i}': mk.MagicMock() for i in range(self.num_opt)}
        self.extra_kwargs = {f'new{i}': mk.MagicMock() for i in range(self.num_opt)}

    def test__init_spline(self):
        class Spline(_Spline):
            optional_inputs = {'test': 'test'}

            def __init__(self, dim=None):
                self.spline_dimension = dim

        spl = Spline()
        spl._init_spline()
        assert spl._t == None
        assert spl._c == None
        assert spl._k == None
        assert spl._test == None

        # check non-defaults
        spl = Spline(1)
        spl._init_spline(1, 2, 3)
        assert spl._t == 1
        assert spl._c == 2
        assert spl._k == 3
        assert spl._test == None

        spl = Spline()
        # check that dimensions are checked
        with pytest.raises(ValueError,
                           match=r"The dimensions for knots and degree do not agree!"):
            spl._init_spline((1, 2), 3, 4)

    def test__check_dimension(self):
        class Spline(_Spline):
            pass

        spl = Spline()
        t_effect = [0,   1, 1, 2, 3]
        k_effect = [any, 0, 1, 2, 1]
        effects = [val for pair in zip(t_effect, k_effect) for val in pair]
        call_arg = [mk.call(spl, '_t'), mk.call(spl, '_k')]
        with mk.patch.object(_Spline, '_get_dimension', autospec=True,
                             side_effect=effects) as mkGet:
            # t is None
            spl._check_dimension()
            mkGet.call_args_list == call_arg

            mkGet.reset_mock()
            # t is not None, but k is None
            spl._check_dimension()
            mkGet.call_args_list == call_arg

            mkGet.reset_mock()
            # t is 1D and k is 1D
            spl._check_dimension()
            mkGet.call_args_list == call_arg

            mkGet.reset_mock()
            # t is 2D and k is 2D
            spl._check_dimension()
            mkGet.call_args_list == call_arg

            mkGet.reset_mock()
            # t is 3D and k is 1D
            with pytest.raises(ValueError,
                               match=r"The dimensions for knots and degree do not agree!"):
                spl._check_dimension()
            mkGet.call_args_list == call_arg

    def test__get_dimension(self):
        class Spline(_Spline):
            def __init__(self, test=None, dim=None):
                self.test = test
                self.spline_dimension = dim

        # 0-D data
        spl = Spline()
        assert spl.test == None
        assert spl._get_dimension('test') == 0

        # 1-D data
        spl = Spline(1)
        assert spl.test == 1
        assert spl._get_dimension('test') == 1
        spl = Spline(np.arange(npts))
        assert (spl.test == np.arange(npts)).all()
        assert spl._get_dimension('test') == 1

        # 1-D data, with warning
        spl = Spline((1,))
        assert spl.test == (1,)
        with pytest.warns(AstropyUserWarning):
            assert spl._get_dimension('test') == 1
        assert spl.test == 1
        spl = Spline((np.arange(npts),))
        assert len(spl.test) == 1
        assert (spl.test[0] == np.arange(npts)).all()
        with pytest.warns(AstropyUserWarning):
            assert spl._get_dimension('test') == 1
        assert (spl.test == np.arange(npts)).all()

        # 1-D data, with error
        spl = Spline(1, 2)
        assert spl.test == 1
        with pytest.raises(RuntimeError):
            spl._get_dimension('test')
        spl = Spline(np.arange(npts), 2)
        assert (spl.test == np.arange(npts)).all()
        with pytest.raises(RuntimeError):
            spl._get_dimension('test')

        # 2-D data
        spl = Spline((1, 2))
        assert spl.test == (1, 2)
        assert spl._get_dimension('test') == 2
        spl = Spline((np.arange(npts), np.arange(npts)))
        assert len(spl.test) == 2
        assert (spl.test[0] == np.arange(npts)).all()
        assert (spl.test[1] == np.arange(npts)).all()
        assert spl._get_dimension('test') == 2

        # 3-D data
        spl = Spline((1, 2, 3))
        assert spl.test == (1, 2, 3)
        assert spl._get_dimension('test') == 3
        spl = Spline((np.arange(npts), np.arange(npts), np.arange(npts)))
        assert len(spl.test) == 3
        assert (spl.test[0] == np.arange(npts)).all()
        assert (spl.test[1] == np.arange(npts)).all()
        assert (spl.test[2] == np.arange(npts)).all()
        assert spl._get_dimension('test') == 3

    def test_reset(self):
        class Spline(_Spline):
            _t = 1
            _c = 2
            _k = 3

        spl = Spline()
        assert spl._t == 1
        assert spl._c == 2
        assert spl._k == 3

        spl.reset()
        assert spl._t is None
        assert spl._c is None
        assert spl._k is None

    def test__has_tck(self):
        class Spline(_Spline):
            def __init__(self):
                self._init_spline()

        spl = Spline()
        assert not spl._has_tck

        spl._t = mk.MagicMock()
        assert not spl._has_tck

        spl._c = mk.MagicMock()
        assert not spl._has_tck

        spl._k = mk.MagicMock()
        assert spl._has_tck

    def test_knots(self):
        class Spline(_Spline):
            def __init__(self):
                self._init_spline()

        spl = Spline()
        with pytest.warns(AstropyUserWarning):
            assert spl.knots == spl._t == None

        with mk.patch.object(_Spline, '_check_dimension', autospec=True) as mkCheck:
            spl.knots = np.arange(npts)
            assert mkCheck.call_args_list == [mk.call(spl)]
            assert (spl.knots == spl._t).all()
            assert (spl.knots == np.arange(npts)).all()

            mkCheck.reset_mock()
            with mk.patch.object(_Spline, '_has_tck', new_callable=mk.PropertyMock,
                                 side_effect=[True, False]) as mkHas:
                with mk.patch.object(_Spline, 'reset', autospec=True) as mkReset:
                    knots = np.random.rand(npts)
                    assert (spl.knots != knots).all()
                    with pytest.warns(AstropyUserWarning):
                        spl.knots = knots
                    assert mkHas.call_args_list == [mk.call()]
                    assert mkReset.call_args_list == [mk.call(spl)]
                    assert mkCheck.call_args_list == [mk.call(spl)]
                    assert (spl.knots == spl._t).all()
                    assert (spl.knots == knots).all()

                    mkHas.reset_mock()
                    mkReset.reset_mock()
                    mkCheck.reset_mock()
                    knots = np.random.rand(npts)
                    assert (spl.knots != knots).all()
                    spl.knots = knots
                    assert mkHas.call_args_list == [mk.call()]
                    assert mkReset.call_args_list == []
                    assert mkCheck.call_args_list == [mk.call(spl)]
                    assert (spl.knots == spl._t).all()
                    assert (spl.knots == knots).all()

    def test_coeffs(self):
        class Spline(_Spline):
            def __init__(self):
                self._init_spline()

        spl = Spline()
        with pytest.warns(AstropyUserWarning):
            assert spl.coeffs == spl._c == None

        spl.coeffs = np.arange(npts)
        assert (spl.coeffs == spl._c).all()
        assert (spl.coeffs == np.arange(npts)).all()

        with mk.patch.object(_Spline, '_has_tck', new_callable=mk.PropertyMock,
                             side_effect=[True, False]) as mkHas:
            with mk.patch.object(_Spline, 'reset', autospec=True) as mkReset:
                coeffs = np.random.rand(npts)
                assert (spl.coeffs != coeffs).all()
                with pytest.warns(AstropyUserWarning):
                    spl.coeffs = coeffs
                assert mkHas.call_args_list == [mk.call()]
                assert mkReset.call_args_list == [mk.call(spl)]
                assert (spl.coeffs == spl._c).all()
                assert (spl.coeffs == coeffs).all()

                mkHas.reset_mock()
                mkReset.reset_mock()
                coeffs = np.random.rand(npts)
                assert (spl.coeffs != coeffs).all()
                spl.coeffs = coeffs
                assert mkHas.call_args_list == [mk.call()]
                assert mkReset.call_args_list == []
                assert (spl.coeffs == spl._c).all()
                assert (spl.coeffs == coeffs).all()

    def test_degree(self):
        class Spline(_Spline):
            def __init__(self):
                self._init_spline()

        spl = Spline()
        with pytest.warns(AstropyUserWarning):
            assert spl.degree == spl._k == None

        with mk.patch.object(_Spline, '_check_dimension', autospec=True) as mkCheck:
            spl.degree = 1
            assert mkCheck.call_args_list == [mk.call(spl)]
            assert spl.degree == spl._k
            assert spl.degree == 1

            mkCheck.reset_mock()
            with mk.patch.object(_Spline, '_has_tck', new_callable=mk.PropertyMock,
                                 side_effect=[True, False]) as mkHas:
                with mk.patch.object(_Spline, 'reset', autospec=True) as mkReset:
                    with pytest.warns(AstropyUserWarning):
                        spl.degree = 2
                    assert mkHas.call_args_list == [mk.call()]
                    assert mkReset.call_args_list == [mk.call(spl)]
                    assert mkCheck.call_args_list == [mk.call(spl)]
                    assert spl.degree == spl._k
                    assert spl.degree == 2

                    mkHas.reset_mock()
                    mkReset.reset_mock()
                    mkCheck.reset_mock()
                    spl.degree = 3
                    assert mkHas.call_args_list == [mk.call()]
                    assert mkReset.call_args_list == []
                    assert mkCheck.call_args_list == [mk.call(spl)]
                    assert spl.degree == spl._k
                    assert spl.degree == 3

    def test_tck(self):
        class Spline(_Spline):
            pass

        spl = Spline()

        # test get value
        with mk.patch.object(_Spline, '_has_tck', new_callable=mk.PropertyMock,
                             side_effect=[True, False]) as mkHas:
            with pytest.raises(NotImplementedError):
                spl.tck
            assert mkHas.call_args_list == [mk.call()]

            mkHas.reset_mock()
            with pytest.raises(RuntimeError):
                spl.tck
            assert mkHas.call_args_list == [mk.call()]

        # test set value
        with pytest.raises(NotImplementedError):
            spl.tck = mk.MagicMock()

    def test_spline(self):
        class Spline(_Spline):
            pass

        spl = Spline()

        # test get value
        with pytest.raises(NotImplementedError):
            spl.spline

        with mk.patch.object(_Spline, '_get_spline', autospec=True) as mkGet:
            assert spl.spline == mkGet.return_value
            assert mkGet.call_args_list == [mk.call(spl)]

        # test set value
        spline = mk.MagicMock()
        with mk.patch.object(_Spline, 'tck', new_callable=mk.PropertyMock) as mkTck:
            spl.spline = spline
            assert mkTck.call_args_list == [mk.call(spline)]

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


fitting_variables_1D = ('w', 'k', 's', 't')
fitting_tests_1D = [
    (None,   1, None, None),
    (None,   2, None, None),
    (None,   3, None, None),
    (None,   4, None, None),
    (None,   5, None, None),
    (test_w, 3, None, None),
    (test_w, 1, None, None),
    (None,   3, npts, None),
    (None,   1, npts, None),
    (None,   3, 3,    None),
    (None,   1, 3,    None),
    (None,   3, None, test_t),
    (None,   1, None, test_t),
    (None,   1, npts, test_t),
]

lsq_variables_1D = ('w', 'k')
lsq_tests_1D = [
    (None,   1),
    (None,   2),
    (None,   3),
    (None,   4),
    (None,   5),
    (test_w, 3),
    (test_w, 1),
]


@pytest.mark.skipif('not HAS_SCIPY')
class TestSpline1D:
    """Test Spline 1D"""

    def setup_class(self):
        def func(x, noise):
            return np.exp(-x**2) + 0.1*noise

        self.x = np.linspace(-3, 3, npts)
        self.y = func(self.x, noise)

        arg_sort = np.argsort(self.x)
        np.random.shuffle(arg_sort)

        self.x_s = self.x[arg_sort]
        self.y_s = func(self.x_s, noise[arg_sort])

        self.npts_out = 1000
        self.xs = np.linspace(-3, 3, self.npts_out)

        self.t = np.linspace(-3, 3, nknots)[1:-1]

    def generate_spline(self, w=None, bbox=[None]*2, k=None, s=None, t=None):
        if k is None:
            k = 3

        from scipy.interpolate import splrep, BSpline

        tck, fp, ier, msg = splrep(self.x, self.y, w=w, xb=bbox[0], xe=bbox[1],
                                   k=k, s=s, t=t, full_output=1)

        return BSpline(*tck), fp, ier, msg

    def generate_lsq_spline(self, w=None, k=None):
        if k is None:
            k = 3

        from scipy.interpolate import splrep, BSpline

        tck, fp, ier, msg = splrep(self.x, self.y, k=k, t=self.t, w=w, full_output=1)

        return BSpline(*tck), fp, ier, msg

    def check_fit_spline(self, spl, x, y, fp, ier, msg, w=None, k=3, s=None, t=None):
        if (s is not None) and (t is not None):
            with pytest.warns(AstropyUserWarning):
                test_fp, test_ier, test_msg = \
                    spl.fit_spline(x, y, w=w, k=k, s=s, t=t)
        else:
            test_fp, test_ier, test_msg = \
                spl.fit_spline(x, y, w=w, k=k, s=s, t=t)

        assert fp == test_fp
        assert ier == test_ier
        assert msg == test_msg

    def check_fitter(self, fitter, spl, fp, ier, msg, w=None, k=3, s=None, t=None):
        assert fitter.fit_info['fp'] is None
        assert fitter.fit_info['ier'] is None
        assert fitter.fit_info['msg'] is None

        if (s is not None) and (t is not None):
            with pytest.warns(AstropyUserWarning):
                fit = fitter(spl, self.x, self.y, w=w, k=k, s=s, t=t)
        else:
            fit = fitter(spl, self.x, self.y, w=w, k=k, s=s, t=t)

        assert fitter.fit_info['fp'] == fp
        assert fitter.fit_info['ier'] == ier
        assert fitter.fit_info['msg'] == msg

        return fit

    def check_spline(self, spl, truth, nu, k=3):
        if nu > k + 1:
            with pytest.raises(RuntimeError):
                spl.evaluate(self.xs, nu=nu)
        else:
            assert (spl(self.xs, nu=nu) == truth(self.xs, nu=nu)).all()

    def check_fit(self, spl, truth, k=3):
        assert (truth.t == spl.knots).all()
        assert (truth.c == spl.coeffs).all()
        assert truth.k == spl.degree

        assert (spl(self.xs) == truth(self.xs)).all()

        for nu in range(1, 7):
            self.check_spline(spl, truth, nu, k)

        # Test warning
        with pytest.warns(AstropyUserWarning):
            spl.fit_spline(self.x, self.y)

    def run_fit_check(self, spl, w=None, k=3, s=None, t=None, bbox=[None]*2):
        truth, fp, ier, msg = self.generate_spline(w=w, k=k, s=s, t=t, bbox=bbox)

        spl.reset()
        self.check_fit_spline(spl, self.x, self.y, fp, ier, msg,
                              w=w, k=k, s=s, t=t)
        self.check_fit(spl, truth, k=k)

        spl.reset()
        self.check_fit_spline(spl, self.x.tolist(), self.y.tolist(),
                              fp, ier, msg, w=w, k=k, s=s, t=t)
        self.check_fit(spl, truth, k=k)

        spl.reset()
        self.check_fit_spline(spl, self.x_s, self.y_s, fp, ier, msg,
                              w=w, k=k, s=s, t=t)
        self.check_fit(spl, truth, k=k)

        spl.reset()
        self.check_fit_spline(spl, self.x_s.tolist(), self.y_s.tolist(),
                              fp, ier, msg, w=w, k=k, s=s, t=t)
        self.check_fit(spl, truth, k=k)

    @pytest.mark.parametrize(fitting_variables_1D, fitting_tests_1D)
    def test_fit_spline(self, w, k, s, t):
        spl = Spline1D()

        # Normal
        self.run_fit_check(spl, w=w, k=k, s=s, t=t)

        spl.reset()
        bbox = (-4, 4)
        spl.bounding_box = bbox
        self.run_fit_check(spl, w=w, k=k, s=s, t=t, bbox=bbox)

    @pytest.mark.parametrize(lsq_variables_1D, lsq_tests_1D)
    def test_fit_spline_lsq(self, w, k):
        spl = Spline1D()
        truth, fp, ier, msg = self.generate_lsq_spline(w=w, k=k)

        spl.fit_spline(self.x, self.y, t=self.t, w=w, k=k)
        self.check_fit(spl, truth, k=k)

        spl.reset()
        spl.fit_spline(self.x.tolist(), self.y.tolist(), t=self.t, w=w, k=k)
        self.check_fit(spl, truth, k=k)

        spl.reset()
        self.check_fit_spline(spl, self.x, self.y, fp, ier, msg,
                              w=w, k=k, t=self.t)
        self.check_fit(spl, truth, k=k)

    @pytest.mark.parametrize(fitting_variables_1D, fitting_tests_1D)
    def test_SplineFitter(self, w, k, s, t):
        fitter = SplineFitter()
        spl = Spline1D()

        # Main check
        truth, fp, ier, msg = self.generate_spline(w=w, k=k, s=s, t=t)
        fit = self.check_fitter(fitter, spl, fp, ier, msg,
                                w=w, k=k, s=s, t=t)
        assert id(fit) != id(spl)
        self.check_fit(fit, truth, k=k)

        # Check defaults
        truth, fp, ier, msg = self.generate_spline()
        fit = fitter(spl, self.x, self.y)
        assert fitter.fit_info['fp'] == fp
        assert fitter.fit_info['ier'] == ier
        assert fitter.fit_info['msg'] == msg
        assert id(fit) != id(spl)
        self.check_fit(fit, truth)

        # Test bad input
        with pytest.raises(ValueError):
            fitter(spl, self.x, self.y, mk.MagicMock(), w=w, k=k, s=s, t=t)

        # Test bad model
        with pytest.raises(ModelDefinitionError,
                           match=r"Only spline models are compatible with this fitter"):
            fitter(mk.MagicMock(), self.x, self.y, w=w, k=k, s=s, t=t)

    @pytest.mark.parametrize(lsq_variables_1D, lsq_tests_1D)
    def test_SplineFitter_lsq(self, w, k):
        fitter = SplineFitter()
        spl = Spline1D()
        truth, fp, ier, msg = self.generate_lsq_spline(w=w, k=k)

        fit = fitter(spl, self.x, self.y, t=self.t, k=k, w=w)
        assert id(fit) != id(spl)
        self.check_fit(fit, truth, k=k)

        fit = fitter(spl, self.x.tolist(), self.y.tolist(), t=self.t, k=k, w=w)
        assert id(fit) != id(spl)
        self.check_fit(fit, truth, k=k)

        fitter = SplineFitter()
        fit = self.check_fitter(fitter, spl, fp, ier, msg,
                                w=w, k=k, t=self.t)
        assert id(fit) != id(spl)
        self.check_fit(fit, truth, k=k)

    def test___init__(self):
        # check  defaults
        spl = Spline1D()
        assert spl._t == None
        assert spl._c == None
        assert spl._k == None
        assert spl._nu == None

        # check non-defaults
        spl = Spline1D(1, 2, 3)
        assert spl._t == 1
        assert spl._c == 2
        assert spl._k == 3
        assert spl._nu == None

        # check that dimensions are checked
        with pytest.raises(RuntimeError):
            Spline1D((1, 2), 3, 4)

    def test_tck(self):
        spl = Spline1D()

        # No tck defined
        with pytest.raises(RuntimeError):
            spl.tck

        # Basic set
        spl.tck = (1, 2, 3)
        assert spl.tck == (1, 2, 3)
        assert spl.knots == spl._t == 1
        assert spl.coeffs == spl._c == 2
        assert spl.degree == spl._k == 3

        spl.reset()
        # Realistic set
        bspline = self.generate_spline()[0]
        spl.tck = bspline
        assert spl.tck == bspline.tck
        assert (spl.knots == spl._t).all()
        assert (spl.coeffs == spl._c).all()
        assert spl.degree == spl._k
        assert (spl.knots == bspline.tck[0]).all()
        assert (spl.coeffs == bspline.tck[1]).all()
        assert spl.degree == bspline.tck[2]

        spl.reset()
        # Tuple of incorrect length
        with pytest.raises(NotImplementedError):
            spl.tck = (1, 2, 3, 4)

        spl.reset()
        # Arbitrary input error
        with pytest.raises(NotImplementedError):
            spl.tck = mk.MagicMock()

    def test_spline(self):
        spl = Spline1D()
        bspline = self.generate_spline()[0]
        spl.spline = bspline

        assert spl.spline.tck == bspline.tck

    def test_evaluate(self):
        spl = Spline1D()
        truth = self.generate_spline()[0]
        spl.spline = truth

        assert (spl.evaluate(self.xs) == truth(self.xs)).all()

        # direct derivative set
        assert (spl.evaluate(self.xs, nu=0) == truth(self.xs, nu=0)).all()
        assert (spl.evaluate(self.xs, nu=1) == truth(self.xs, nu=1)).all()
        assert (spl.evaluate(self.xs, nu=2) == truth(self.xs, nu=2)).all()
        assert (spl.evaluate(self.xs, nu=3) == truth(self.xs, nu=3)).all()
        assert (spl.evaluate(self.xs, nu=4) == truth(self.xs, nu=4)).all()

        with pytest.raises(RuntimeError):
            spl.evaluate(self.xs, nu=5)

        # direct derivative call overrides internal
        spl._nu = 5
        assert (spl.evaluate(self.xs, nu=0) == truth(self.xs, nu=0)).all()
        assert (spl.evaluate(self.xs, nu=1) == truth(self.xs, nu=1)).all()
        assert (spl.evaluate(self.xs, nu=2) == truth(self.xs, nu=2)).all()
        assert (spl.evaluate(self.xs, nu=3) == truth(self.xs, nu=3)).all()
        assert (spl.evaluate(self.xs, nu=4) == truth(self.xs, nu=4)).all()

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
        spl._nu = 4
        assert (spl.evaluate(self.xs) == truth(self.xs, nu=4)).all()
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

    def test_unfit_spline(self):
        spl = Spline1D()
        assert (0 == spl(self.xs)).all()

    def test_derivative(self):
        spl = Spline1D()
        spl.spline = self.generate_spline()[0]

        der = spl.derivative()
        assert der.degree == 2
        assert_allclose(der.evaluate(self.xs),       spl.evaluate(self.xs, nu=1))
        assert_allclose(der.evaluate(self.xs, nu=1), spl.evaluate(self.xs, nu=2))
        assert_allclose(der.evaluate(self.xs, nu=2), spl.evaluate(self.xs, nu=3))
        assert_allclose(der.evaluate(self.xs, nu=3), spl.evaluate(self.xs, nu=4))

        der = spl.derivative(nu=2)
        assert der.degree == 1
        assert_allclose(der.evaluate(self.xs),       spl.evaluate(self.xs, nu=2))
        assert_allclose(der.evaluate(self.xs, nu=1), spl.evaluate(self.xs, nu=3))
        assert_allclose(der.evaluate(self.xs, nu=2), spl.evaluate(self.xs, nu=4))

        der = spl.derivative(nu=3)
        assert der.degree == 0
        assert_allclose(der.evaluate(self.xs),       spl.evaluate(self.xs, nu=3))
        assert_allclose(der.evaluate(self.xs, nu=1), spl.evaluate(self.xs, nu=4))

        with pytest.raises(ValueError):
            spl.derivative(nu=4)

    def test_antiderivative(self):
        spl = Spline1D()
        spl.spline = self.generate_spline()[0]

        anti = spl.antiderivative()
        assert anti.degree == 4
        assert_allclose(spl.evaluate(self.xs),       anti.evaluate(self.xs, nu=1))
        assert_allclose(spl.evaluate(self.xs, nu=1), anti.evaluate(self.xs, nu=2))
        assert_allclose(spl.evaluate(self.xs, nu=2), anti.evaluate(self.xs, nu=3))
        assert_allclose(spl.evaluate(self.xs, nu=3), anti.evaluate(self.xs, nu=4))
        assert_allclose(spl.evaluate(self.xs, nu=4), anti.evaluate(self.xs, nu=5))

        anti = spl.antiderivative(nu=2)
        assert anti.degree == 5
        assert_allclose(spl.evaluate(self.xs),       anti.evaluate(self.xs, nu=2))
        assert_allclose(spl.evaluate(self.xs, nu=1), anti.evaluate(self.xs, nu=3))
        assert_allclose(spl.evaluate(self.xs, nu=2), anti.evaluate(self.xs, nu=4))
        assert_allclose(spl.evaluate(self.xs, nu=3), anti.evaluate(self.xs, nu=5))
        assert_allclose(spl.evaluate(self.xs, nu=4), anti.evaluate(self.xs, nu=6))

        with pytest.raises(ValueError):
            spl.antiderivative(nu=3)

    def test_bbox(self):
        spl = Spline1D()
        assert spl.bbox == [None, None]

        spl.bounding_box = (1, 2)
        assert spl.bbox == [1, 2]

    def test__sort_xy(self):
        spline = Spline1D()

        assert not (self.x == self.x_s).all()
        assert not (self.y == self.y_s).all()

        # No sort
        x_n, y_n = spline._sort_xy(self.x_s, self.y_s, False)
        assert (x_n == self.x_s).all()
        assert (y_n == self.y_s).all()

        # Sort
        x_n, y_n = spline._sort_xy(self.x_s, self.y_s)
        assert (x_n == self.x).all()
        assert (y_n == self.y).all()

        x_n, y_n = spline._sort_xy(self.x_s.tolist(), self.y_s.tolist())
        assert (x_n == self.x).all()
        assert (y_n == self.y).all()


fitting_variables_2D = ('w', 'kx', 'ky', 's', 'tx', 'ty')
fitting_tests_2D = [
    (None,   1, 3, None, None,   None),
    (None,   2, 3, None, None,   None),
    (None,   4, 3, None, None,   None),
    (None,   5, 3, None, None,   None),
    (None,   3, 1, None, None,   None),
    (None,   3, 2, None, None,   None),
    (None,   3, 4, None, None,   None),
    (None,   3, 5, None, None,   None),
    (None,   1, 1, None, None,   None),
    (None,   2, 2, None, None,   None),
    (None,   3, 3, None, None,   None),
    (None,   4, 4, None, None,   None),
    (None,   5, 5, None, None,   None),
    (test_w, 3, 3, None, None,   None),
    (test_w, 1, 1, None, None,   None),
    (None,   3, 3, npts, None,   None),
    (None,   1, 1, npts, None,   None),
    (None,   3, 3, 3,    None,   None),
    (None,   1, 1, 3,    None,   None),
    (None,   3, 3, None, test_t, test_t),
    (None,   1, 1, None, test_t, test_t),
    (None,   3, 3, npts, test_t, test_t),
    (None,   1, 1, npts, test_t, test_t),
    (None,   3, 3, None, test_t, None),
    (None,   1, 1, None, None,   test_t),
]

lsq_variables_2D = ('w', 'kx', 'ky')
lsq_tests_2D = [
    (None,   1, 3),
    (None,   2, 3),
    (None,   4, 3),
    (None,   5, 3),
    (None,   3, 1),
    (None,   3, 2),
    (None,   3, 4),
    (None,   3, 5),
    (None,   1, 1),
    (None,   2, 2),
    (None,   3, 3),
    (None,   4, 4),
    (None,   5, 5),
    (test_w, 3, 3),
    (test_w, 1, 1),
]


@pytest.mark.skipif('not HAS_SCIPY')
class TestSpline2D:
    """Test Spline 2D"""

    def setup_class(self):
        np.random.seed(42)

        self.x = np.linspace(-3, 3, npts)
        self.y = np.linspace(-3, 3, npts)
        self.z = np.exp(-self.x**2 - self.y**2) + 0.1 * np.random.randn(npts)

        self.npts_out = 1000
        self.xs = np.linspace(-3, 3, self.npts_out)
        self.ys = np.linspace(-3, 3, self.npts_out)

        self.tx = np.linspace(-3, 3, nknots)[1:-1]
        self.ty = np.linspace(-3, 3, nknots)[1:-1]

    def generate_spline(self, w=None, bbox=[None]*4, kx=None, ky=None,
                        s=None, tx=None, ty=None):
        if kx is None:
            kx = 3
        if ky is None:
            ky = 3

        from scipy.interpolate import bisplrep, BivariateSpline

        tck, fp, ier, msg = bisplrep(self.x, self.y, self.z, w=w,
                                     xb=bbox[0], xe=bbox[1], yb=bbox[2], ye=bbox[3],
                                     kx=kx, ky=ky, s=s, tx=tx, ty=ty,
                                     full_output=1)

        return BivariateSpline._from_tck(tck), fp, ier, msg

    def generate_lsq_spline(self, w=None, bbox=[None]*4, kx=None, ky=None):
        if kx is None:
            kx = 3
        if ky is None:
            ky = 3

        from scipy.interpolate import bisplrep, BivariateSpline

        tck, fp, ier, msg = bisplrep(self.x, self.y, self.z, w=w,
                                     xb=bbox[0], xe=bbox[1], yb=bbox[2], ye=bbox[3],
                                     kx=kx, ky=ky, tx=self.tx, ty=self.ty,
                                     full_output=1)

        return BivariateSpline._from_tck(tck), fp, ier, msg

    def check_fit_spline(self, spl, fp, ier, msg, w=None, kx=3, ky=3,
                         s=None, tx=None, ty=None):
        if ((tx is None) and (ty is not None)) or ((tx is not None) and (ty is None)):
            with pytest.raises(ValueError):
                spl.fit_spline(self.x, self.y, self.z, w=w, kx=kx, ky=ky,
                               s=s, tx=tx, ty=ty)
            return False
        elif (s is not None) and (tx is not None):
            with pytest.warns(AstropyUserWarning):
                test_fp, test_ier, test_msg = \
                    spl.fit_spline(self.x, self.y, self.z, w=w,
                                   kx=kx, ky=ky, s=s, tx=tx, ty=ty)
        else:
            test_fp, test_ier, test_msg = \
                spl.fit_spline(self.x, self.y, self.z, w=w,
                               kx=kx, ky=ky, s=s, tx=tx, ty=ty)

        assert fp == test_fp
        assert ier == test_ier
        assert msg == test_msg

        return True

    def check_fitter(self, fitter, spl, fp, ier, msg, w=None, kx=3, ky=3,
                     s=None, tx=None, ty=None):

        assert fitter.fit_info['fp'] is None
        assert fitter.fit_info['ier'] is None
        assert fitter.fit_info['msg'] is None

        if ((tx is None) and (ty is not None)) or ((tx is not None) and (ty is None)):
            with pytest.raises(ValueError):
                fitter(spl, self.x, self.y, self.z,
                       w=w, k=(kx, ky), s=s, t=(tx, ty))
            return False, None
        elif (s is not None) and (tx is not None):
            with pytest.warns(AstropyUserWarning):
                fit = fitter(spl, self.x, self.y, self.z,
                             w=w, k=(kx, ky), s=s, t=(tx, ty))
        else:
            fit = fitter(spl, self.x, self.y, self.z,
                         w=w, k=(kx, ky), s=s, t=(tx, ty))

        assert fitter.fit_info['fp'] == fp
        assert fitter.fit_info['ier'] == ier
        assert fitter.fit_info['msg'] == msg

        return True, fit

    def check_spline(self, spl, truth, dx, dy, kx=3, ky=3):
        if dx > kx - 1:
            with pytest.raises(RuntimeError):
                spl.evaluate(self.xs, self.ys, dx=dx, dy=dy)
        elif dy > ky - 1:
            with pytest.raises(RuntimeError):
                spl.evaluate(self.xs, self.ys, dx=dx, dy=dy)
        else:
            assert (spl(self.xs, self.ys, dx=dx, dy=dy) ==
                    truth(self.xs, self.ys, dx=dx, dy=dy)).all()

    def check_fit(self, spl, truth, kx=3, ky=3):
        assert (spl.knots[0] == truth.get_knots()[0]).all()
        assert (spl.coeffs == truth.get_coeffs()).all()
        assert spl.degree == tuple(truth.degrees)

        assert (spl(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        for dx in range(1, 7):
            for dy in range(1, 7):
                self.check_spline(spl, truth, dx, dy, kx, ky)

        # Test warning
        with pytest.warns(AstropyUserWarning):
            spl.fit_spline(self.x, self.y, self.z)

    @pytest.mark.parametrize(fitting_variables_2D, fitting_tests_2D)
    def test_fit_spline(self, w, kx, ky, s, tx, ty):
        spl = Spline2D()

        truth, fp, ier, msg = self.generate_spline()
        check = self.check_fit_spline(spl, fp, ier, msg)
        if check:
            self.check_fit(spl, truth)

        spl.reset()
        truth, fp, ier, msg = self.generate_spline(w=w, kx=kx, ky=ky,
                                                   s=s, tx=tx, ty=ty)
        check = self.check_fit_spline(spl, fp, ier, msg, w=w, kx=kx, ky=ky,
                                      s=s, tx=tx, ty=ty)
        if check:
            self.check_fit(spl, truth, kx=kx, ky=ky)

        spl.reset()
        spl.bounding_box = ((-4, 4), (-4, 4))
        truth, fp, ier, msg = self.generate_spline(w=w, kx=kx, ky=ky,
                                                   s=s, tx=tx, ty=ty,
                                                   bbox=spl.bbox)
        check = self.check_fit_spline(spl, fp, ier, msg, w=w, kx=kx, ky=ky,
                                      s=s, tx=tx, ty=ty)
        if check:
            self.check_fit(spl, truth, kx=kx, ky=ky)

    @pytest.mark.parametrize(lsq_variables_2D, lsq_tests_2D)
    def test_fit_spline_lsq(self, w, kx, ky):
        spl = Spline2D()
        truth, fp, ier, msg = self.generate_lsq_spline(w=w, kx=kx, ky=ky)

        spl.fit_spline(self.x, self.y, self.z, tx=self.tx, ty=self.ty,
                       w=w, kx=kx, ky=ky)
        self.check_fit(spl, truth, kx=kx, ky=ky)

        spl.reset()
        self.check_fit_spline(spl, fp, ier, msg, w=w, kx=kx, ky=ky)
        self.check_fit(spl, truth, kx=kx, ky=ky)

    @pytest.mark.parametrize(fitting_variables_2D, fitting_tests_2D)
    def test_SplineFitter(self, w, kx, ky, s, tx, ty):
        fitter = SplineFitter()
        spl = Spline2D()

        # Main check
        truth, fp, ier, msg = self.generate_spline(w=w, kx=kx, ky=ky,
                                                   s=s, tx=tx, ty=ty)
        check, fit = self.check_fitter(fitter, spl, fp, ier, msg,
                                       w=w, kx=kx, ky=ky,
                                       s=s, tx=tx, ty=ty)
        if check:
            assert id(fit) != id(spl)
            self.check_fit(fit, truth, kx=kx, ky=ky)

        # Check defaults
        truth, fp, ier, msg = self.generate_spline()
        fit = fitter(spl, self.x, self.y, self.z)
        assert fitter.fit_info['fp'] == fp
        assert fitter.fit_info['ier'] == ier
        assert fitter.fit_info['msg'] == msg
        assert id(fit) != id(spl)
        self.check_fit(fit, truth)

        # No z data
        with pytest.raises(ValueError):
            fitter(spl, self.x, self.y,
                   w=w, k=(kx, ky), s=s, t=(tx, ty))

        # Single k
        with pytest.raises(ValueError):
            fitter(spl, self.x, self.y, self.z,
                   w=w, k=kx, s=s, t=(tx, ty))

        # Single t
        if tx is not None:
            with pytest.raises(ValueError):
                fitter(spl, self.x, self.y, self.z,
                       w=w, k=(kx, ky), s=s, t=tx)

        # Bad model input
        with pytest.raises(ModelDefinitionError,
                           match=r"Only spline models are compatible with this fitter"):
            fitter(mk.MagicMock(), self.x, self.y, self.z,
                   w=w, k=(kx, ky), s=s, t=(tx, ty))

    @pytest.mark.parametrize(lsq_variables_2D, lsq_tests_2D)
    def test_SplineFitter_lsq(self, w, kx, ky):
        fitter = SplineFitter()
        spl = Spline2D()
        truth, fp, ier, msg = self.generate_lsq_spline(w=w, kx=kx, ky=ky)

        fit = fitter(spl, self.x, self.y, self.z, t=(self.tx, self.ty), w=w, k=(kx, ky))
        assert id(fit) != id(spl)
        self.check_fit(fit, truth, kx=kx, ky=ky)

        fit = fitter(spl, self.x.tolist(), self.y.tolist(), self.z.tolist(),
                     t=(self.tx, self.ty), w=w, k=(kx, ky))
        assert id(fit) != id(spl)
        self.check_fit(fit, truth, kx=kx, ky=ky)

        fitter = SplineFitter()
        _, fit = self.check_fitter(fitter, spl, fp, ier, msg, w=w, kx=kx, ky=ky,
                                   tx=self.tx, ty=self.ty)
        assert id(fit) != id(spl)
        self.check_fit(fit, truth, kx=kx, ky=ky)

    def test___init__(self):
        # check  defaults
        spl = Spline2D()
        assert spl._t == None
        assert spl._c == None
        assert spl._k == None
        assert spl._dx is None
        assert spl._dy is None

        # check non-defaults
        spl = Spline2D((1, 2), 3, (4, 5))
        assert spl._t == (1, 2)
        assert spl._c == 3
        assert spl._k == (4, 5)
        assert spl._dx is None
        assert spl._dy is None

        # check that dimensions are checked
        with pytest.raises(RuntimeError):
            Spline2D((1, 2), 3, 4)

    def test_tck(self):
        spl = Spline2D()

        # No tck
        with pytest.raises(RuntimeError):
            spl.tck

        # Basic set
        spl.tck = (1, 2, 3, 4, 5)
        assert spl.tck == (1, 2, 3, 4, 5)
        assert spl.knots == spl._t == (1, 2)
        assert spl.coeffs == spl._c == 3
        assert spl.degree == spl._k == (4, 5)
        spl.reset()
        spl.tck = ((1, 2), 3, (4, 5))
        assert spl.tck == (1, 2, 3, 4, 5)
        assert spl.knots == spl._t == (1, 2)
        assert spl.coeffs == spl._c == 3
        assert spl.degree == spl._k == (4, 5)

        spl.reset()
        # Realistic set
        bspline = self.generate_spline()[0]
        spl.tck = bspline
        assert len(spl.tck) == 5
        assert spl.tck[:2] == tuple(bspline.get_knots())
        assert (spl.tck[2] == bspline.get_coeffs()).all()
        assert spl.tck[3:] == tuple(bspline.degrees)
        assert spl.knots == spl._t
        assert (spl.coeffs == spl._c).all()
        assert spl.degree == spl._k
        assert spl.knots == tuple(bspline.get_knots())
        assert (spl.coeffs == bspline.get_coeffs()).all()
        assert spl.degree == tuple(bspline.degrees)

        # Bad tck tuple lengths
        spl.reset()
        for idx in range(1, 11):
            if idx in [3, 5]:
                continue
            print(idx)
            with pytest.raises(ValueError) as err:
                spl.tck = tuple(range(idx))
            assert str(err.value) == \
                'tck must be of length 3 or 5'
            with pytest.raises(ValueError) as err:
                spl.tck = list(range(idx))
            assert str(err.value) == \
                'tck must be of length 3 or 5'

        spl.reset()
        with pytest.raises(NotImplementedError):
            spl.tck = mk.MagicMock()

    def test_spline(self):
        spl = Spline2D()
        bspline = self.generate_spline()[0]
        spl.spline = bspline

        assert spl.spline.get_knots() == tuple(bspline.get_knots())
        assert (spl.spline.get_coeffs() == bspline.get_coeffs()).all()
        assert spl.spline.degrees == tuple(bspline.degrees)

    def test_evaluate(self):
        spl = Spline2D()
        truth = self.generate_spline()[0]
        spl.spline = truth

        assert (spl.evaluate(self.xs, self.ys) == truth(self.xs, self.ys)).all()

        # direct derivative set
        assert (spl.evaluate(self.xs, self.ys, dx=0) == truth(self.xs, self.ys, dx=0)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=1) == truth(self.xs, self.ys, dx=1)).all()
        assert (spl.evaluate(self.xs, self.ys, dx=2) == truth(self.xs, self.ys, dx=2)).all()

        with pytest.raises(RuntimeError):
            spl.evaluate(self.xs, self.ys, dx=3)

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
        with pytest.raises(RuntimeError):
            spl.evaluate(self.xs, self.ys, dy=3)

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

    def test_unfit_spline(self):
        spl = Spline2D()
        assert (0 == spl(self.xs, self.ys)).all()

    def test_bbox(self):
        spl = Spline2D()
        assert spl.bbox == [None, None, None, None]

        spl.bounding_box = ((1, 2), (3, 4))
        assert spl.bbox == [1, 2, 3, 4]


@pytest.mark.skipif('not HAS_SCIPY')
class TestNewSpline1D:
    def setup_class(self):
        def func(x, noise):
            return np.exp(-x**2) + 0.1*noise

        self.x = np.linspace(-3, 3, npts)
        self.y = func(self.x, noise)

        arg_sort = np.argsort(self.x)
        np.random.shuffle(arg_sort)

        self.x_s = self.x[arg_sort]
        self.y_s = func(self.x_s, noise[arg_sort])

        self.npts_out = 1000
        self.xs = np.linspace(-3, 3, self.npts_out)

        self.t = np.linspace(-3, 3, nknots)[1:-1]

    def check_parameter(self, spl, name, value, new_value=None):
        assert hasattr(spl, name)
        param = getattr(spl, name)
        assert isinstance(param, Parameter)
        assert param.name == name
        assert param.value == value
        assert param.model == spl
        if new_value is not None:
            param.value = new_value

    def check_knots(self, spl, knots, base_name, value):
        for idx, name in enumerate(knots):
            assert base_name in name
            assert idx == int(name.split(base_name)[-1])
            knot_name = f"{base_name}{idx}"
            assert knot_name == name
            self.check_parameter(spl, knot_name, value)

    def check_coeffs(self, spl, value, shift=5):
        assert len(spl._coeff_names) == len(spl._knot_names)
        for idx, name in enumerate(spl._knot_names):
            assert name in spl._coeff_names[idx]
            assert "_coeff" == spl._coeff_names[idx].split(name)[-1]
            coeff_name = f"{name}_coeff"
            assert coeff_name == spl._coeff_names[idx]
            new_value = idx + shift
            self.check_parameter(spl, coeff_name, value, new_value)

    def update_knots(self, spl, knots, value):
        for name in knots:
            param = getattr(spl, name)
            param.value = value
            assert param.value == value

    def test__init__with_no_knot_information(self):
        spl = NewSpline1D()
        assert spl._degree == 3
        assert spl._nknots is None
        assert spl._user_knots is False
        assert spl._t is None
        assert spl._c is None

        # Check all knot names created
        assert len(spl._lower_knot_names) == 0
        assert len(spl._upper_knot_names) == 0
        assert len(spl._interior_knot_names) == 0

    def test___init__with_number_of_knots(self):
        spl = NewSpline1D(10)

        # Check baseline data
        assert spl._degree == 3
        assert spl._nknots == 10
        assert spl._user_knots is False

        # Check vector data
        assert len(spl._t) == 18
        t = np.zeros(18)
        t[-4:] = 1
        assert (spl._t == t).all()
        assert len(spl._c) == 18
        assert (spl._c == np.zeros(18)).all()

        # Check all parameter names created:
        assert len(spl._lower_knot_names) == 4
        assert len(spl._upper_knot_names) == 4
        assert len(spl._interior_knot_names) == 10
        assert len(spl._coeff_names) == 18

        # Check knot values:
        self.check_knots(spl, spl._lower_knot_names, "knot_lower", 0)
        self.check_knots(spl, spl._upper_knot_names, "knot_upper", 1)
        self.check_knots(spl, spl._interior_knot_names, "knot", 0)

        # Check coeff values:
        assert len(spl._coeff_names) == 18
        self.check_coeffs(spl, 0)

    def test___init__with_full_custom_knots(self):
        t = np.arange(20)
        spl = NewSpline1D(knots=t)

        # Check baseline data
        assert spl._degree == 3
        assert spl._nknots == 12
        assert spl._user_knots is True

        # Check vector data
        assert (spl._t == t).all()
        assert len(spl._c) == 20
        assert (spl._c == np.zeros(20)).all()

        # Check all parameter names created
        assert len(spl._lower_knot_names) == 4
        assert len(spl._upper_knot_names) == 4
        assert len(spl._interior_knot_names) == 12
        assert len(spl._coeff_names) == 20

        # Check knot values:
        idx = 0
        for name in spl._lower_knot_names:
            self.check_parameter(spl, name, t[idx])
            idx += 1
        assert idx == 4
        for name in spl._interior_knot_names:
            self.check_parameter(spl, name, t[idx])
            idx += 1
        assert idx == 16
        for name in spl._upper_knot_names:
            self.check_parameter(spl, name, t[idx])
            idx += 1
        assert idx == 20
        for name in spl._coeff_names:
            self.check_parameter(spl, name, 0)

    def test___init__with_interior_custom_knots(self):
        t = np.arange(1, 20)
        spl = NewSpline1D(knots=t, bounds=[0, 20])
        # Check baseline data
        assert spl._degree == 3
        assert spl._nknots == 19
        assert spl._user_knots is True

        # Check vector data
        assert len(spl._t) == 27
        assert (spl._t[4:-4] == t).all()
        assert (spl._t[:4] == 0).all()
        assert (spl._t[-4:] == 20).all()

        assert len(spl._c) == 27
        assert (spl._c == np.zeros(27)).all()

    def test___init__errors(self):
        # Bad knot type
        knots = 3.5
        with pytest.raises(ValueError) as err:
            NewSpline1D(knots=knots)
        assert str(err.value) ==\
            f"Knots: {knots} must be iterable or value"

        # Not enough knots
        for idx in range(8):
            with pytest.raises(ValueError) as err:
                NewSpline1D(knots=np.arange(idx))
            assert str(err.value) ==\
                "Must have at least 8 knots."

        # Bad scipy spline
        t = np.arange(20)[::-1]
        with pytest.raises(ValueError):
            NewSpline1D(knots=t)

    def test_parameter_array_link(self):
        spl = NewSpline1D(10)
        # Check knot vector -> knot parameter link
        self.check_knots(spl, spl._lower_knot_names, "knot_lower", 0)
        self.check_knots(spl, spl._upper_knot_names, "knot_upper", 1)
        self.check_knots(spl, spl._interior_knot_names, "knot", 0)
        spl._t[0:4] = 2
        self.check_knots(spl, spl._lower_knot_names, "knot_lower", 2)
        self.check_knots(spl, spl._upper_knot_names, "knot_upper", 1)
        self.check_knots(spl, spl._interior_knot_names, "knot", 0)
        spl._t[-4:] = 3
        self.check_knots(spl, spl._lower_knot_names, "knot_lower", 2)
        self.check_knots(spl, spl._upper_knot_names, "knot_upper", 3)
        self.check_knots(spl, spl._interior_knot_names, "knot", 0)
        spl._t[4:-4] = 4
        self.check_knots(spl, spl._lower_knot_names, "knot_lower", 2)
        self.check_knots(spl, spl._upper_knot_names, "knot_upper", 3)
        self.check_knots(spl, spl._interior_knot_names, "knot", 4)

        # Check knot parameter -> knot vector link
        self.update_knots(spl, spl._lower_knot_names, 5)
        assert (spl._t[0:4] == 5).all()
        assert (spl._t[-4:] == 3).all()
        assert (spl._t[4:-4] == 4).all()
        self.update_knots(spl, spl._upper_knot_names, 6)
        assert (spl._t[0:4] == 5).all()
        assert (spl._t[-4:] == 6).all()
        assert (spl._t[4:-4] == 4).all()
        self.update_knots(spl, spl._interior_knot_names, 7)
        assert (spl._t[0:4] == 5).all()
        assert (spl._t[-4:] == 6).all()
        assert (spl._t[4:-4] == 7).all()

        assert len(spl._coeff_names) == 18
        self.check_coeffs(spl, 0)
        # Check coeff param -> coeff vector set
        assert (spl._c == (np.arange(18) + 5)).all()
        spl._c = (np.zeros(18) + 37)
        self.check_coeffs(spl, 37)
        assert (spl._c == (np.arange(18) + 5)).all()

    def test_two_splines(self):
        spl0 = NewSpline1D(10)
        spl1 = NewSpline1D(15, 2)

        assert spl0._degree == 3
        assert spl0._nknots == 10
        assert len(spl0._t) == 18
        t = np.zeros(18)
        t[-4:] = 1
        assert (spl0._t == t).all()
        assert len(spl0._c) == 18
        assert (spl0._c == np.zeros(18)).all()
        assert spl1._degree == 2
        assert spl1._nknots == 15
        assert len(spl1._t) == 21
        t = np.zeros(21)
        t[-3:] = 1
        assert (spl1._t == t).all()
        assert len(spl1._c) == 21
        assert (spl1._c == np.zeros(21)).all()

        # Check all knot names created
        assert len(spl0._lower_knot_names) == 4
        assert len(spl0._upper_knot_names) == 4
        assert len(spl0._interior_knot_names) == 10
        assert len(spl1._lower_knot_names) == 3
        assert len(spl1._upper_knot_names) == 3
        assert len(spl1._interior_knot_names) == 15

        # Check knot vector -> knot parameter link
        self.check_knots(spl0, spl0._lower_knot_names, "knot_lower", 0)
        self.check_knots(spl0, spl0._upper_knot_names, "knot_upper", 1)
        self.check_knots(spl0, spl0._interior_knot_names, "knot", 0)
        self.check_knots(spl1, spl1._lower_knot_names, "knot_lower", 0)
        self.check_knots(spl1, spl1._upper_knot_names, "knot_upper", 1)
        self.check_knots(spl1, spl1._interior_knot_names, "knot", 0)
        spl0._t[0:4] = 1
        spl1._t[0:3] = 2
        self.check_knots(spl0, spl0._lower_knot_names, "knot_lower", 1)
        self.check_knots(spl0, spl0._upper_knot_names, "knot_upper", 1)
        self.check_knots(spl0, spl0._interior_knot_names, "knot", 0)
        self.check_knots(spl1, spl1._lower_knot_names, "knot_lower", 2)
        self.check_knots(spl1, spl1._upper_knot_names, "knot_upper", 1)
        self.check_knots(spl1, spl1._interior_knot_names, "knot", 0)
        spl0._t[-4:] = 3
        spl1._t[-3:] = 4
        self.check_knots(spl0, spl0._lower_knot_names, "knot_lower", 1)
        self.check_knots(spl0, spl0._upper_knot_names, "knot_upper", 3)
        self.check_knots(spl0, spl0._interior_knot_names, "knot", 0)
        self.check_knots(spl1, spl1._lower_knot_names, "knot_lower", 2)
        self.check_knots(spl1, spl1._upper_knot_names, "knot_upper", 4)
        self.check_knots(spl1, spl1._interior_knot_names, "knot", 0)
        spl0._t[4:-4] = 5
        spl1._t[3:-3] = 6
        self.check_knots(spl0, spl0._lower_knot_names, "knot_lower", 1)
        self.check_knots(spl0, spl0._upper_knot_names, "knot_upper", 3)
        self.check_knots(spl0, spl0._interior_knot_names, "knot", 5)
        self.check_knots(spl1, spl1._lower_knot_names, "knot_lower", 2)
        self.check_knots(spl1, spl1._upper_knot_names, "knot_upper", 4)
        self.check_knots(spl1, spl1._interior_knot_names, "knot", 6)

        # Check knot parameter -> knot vector link
        self.update_knots(spl0, spl0._lower_knot_names, 7)
        self.update_knots(spl1, spl1._lower_knot_names, 8)
        assert (spl0._t[0:4] == 7).all()
        assert (spl0._t[-4:] == 3).all()
        assert (spl0._t[4:-4] == 5).all()
        assert (spl1._t[0:3] == 8).all()
        assert (spl1._t[-3:] == 4).all()
        assert (spl1._t[3:-3] == 6).all()
        self.update_knots(spl0, spl0._upper_knot_names, 9)
        self.update_knots(spl1, spl1._upper_knot_names, 10)
        assert (spl0._t[0:4] == 7).all()
        assert (spl0._t[-4:] == 9).all()
        assert (spl0._t[4:-4] == 5).all()
        assert (spl1._t[0:3] == 8).all()
        assert (spl1._t[-3:] == 10).all()
        assert (spl1._t[3:-3] == 6).all()
        self.update_knots(spl0, spl0._interior_knot_names, 11)
        self.update_knots(spl1, spl1._interior_knot_names, 12)
        assert (spl0._t[0:4] == 7).all()
        assert (spl0._t[-4:] == 9).all()
        assert (spl0._t[4:-4] == 11).all()
        assert (spl1._t[0:3] == 8).all()
        assert (spl1._t[-3:] == 10).all()
        assert (spl1._t[3:-3] == 12).all()

        assert len(spl0._coeff_names) == 18
        assert len(spl1._coeff_names) == 21
        self.check_coeffs(spl0, 0, 11)
        self.check_coeffs(spl1, 0, 17)
        # Check coeff param -> coeff vector set
        assert (spl0._c == (np.arange(18) + 11)).all()
        assert (spl1._c == (np.arange(21) + 17)).all()
        spl0._c = (np.zeros(18) + 37)
        spl1._c = (np.zeros(21) + 48)
        self.check_coeffs(spl0, 37, 81)
        self.check_coeffs(spl1, 48, 32)
        assert (spl0._c == (np.arange(18) + 81)).all()
        assert (spl1._c == (np.arange(21) + 32)).all()
