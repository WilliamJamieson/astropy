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
from astropy.modeling.core import (ModelDefinitionError, FittableModel)
from astropy.modeling.spline import (_Spline, Spline1D, Spline2D,
                                     _NewSpline, NewSpline1D)
from astropy.modeling.fitting import (SplineFitter, NewInterpolatingSplineFitter,
                                      NewSmoothingSplineFitter, NewLSQSplineFitter)
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


new_variables_1D = ('w', 'k')
new_tests_1D = [
    (None,   1),
    (None,   2),
    (None,   3),
    (None,   4),
    (None,   5),
    (test_w, 1),
    (test_w, 2),
    (test_w, 3),
    (test_w, 4),
    (test_w, 5),
]


class TestNewSpline:
    def setup_class(self):
        self.num_opt = 3
        self.optional_inputs = {f'test{i}': mk.MagicMock() for i in range(self.num_opt)}
        self.extra_kwargs = {f'new{i}': mk.MagicMock() for i in range(self.num_opt)}

    def test___init__(self):
        class Spline(_NewSpline):
            optional_inputs = {'test': 'test'}

        # empty spline
        spl = Spline()
        assert spl._knot_names == []
        assert spl._coeff_names == []
        assert spl._t is None
        assert spl._c is None
        assert spl._user_knots is False
        assert spl._degree is None
        assert spl._test == None

        # Call _init_spline
        with mk.patch.object(_NewSpline, '_init_spline',
                             autospec=True) as mkInit:
            # No call (knots=None)
            spl = Spline()
            assert mkInit.call_args_list == []

            knots = mk.MagicMock()
            coeffs = mk.MagicMock()
            bounds = mk.MagicMock()
            spl = Spline(knots=knots, coeffs=coeffs, bounds=bounds)
            assert mkInit.call_args_list == \
                [mk.call(spl, knots, coeffs, bounds)]

            assert spl._t is None
            assert spl._c is None
            assert spl._user_knots is False
            assert spl._degree is None
            assert spl._test == None

        # Coeffs but no knots
        with pytest.raises(ValueError) as err:
            Spline(coeffs=mk.MagicMock())
        assert str(err.value) == \
            "If one passes a coeffs vector one needs to also pass a knots vector!"

    def test_param_names(self):
        # no parameters
        spl = _NewSpline()
        assert spl.param_names == ()

        knot_names = tuple([mk.MagicMock() for _ in range(3)])
        spl._knot_names = list(knot_names)
        assert spl.param_names == knot_names

        coeff_names = tuple([mk.MagicMock() for _ in range(3)])
        spl._coeff_names = list(coeff_names)
        assert spl.param_names == knot_names + coeff_names

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

    def test_evaluate(self):
        class Spline(_NewSpline):
            optional_inputs = self.optional_inputs

        spl = Spline()

        # No options passed in and No options set
        new_kwargs = spl.evaluate(**self.extra_kwargs)
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
        new_kwargs = spl.evaluate(**self.extra_kwargs)
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
        new_kwargs = spl.evaluate(**kwargs)
        assert new_kwargs == kwargs

    def test___call__(self):
        spl = _NewSpline()

        args = tuple([mk.MagicMock() for _ in range(3)])
        kwargs = {f"test{idx}": mk.MagicMock() for idx in range(3)}
        new_kwargs = {f"new_test{idx}": mk.MagicMock() for idx in range(3)}
        with mk.patch.object(_NewSpline, "_intercept_optional_inputs",
                             autospec=True, return_value=new_kwargs) as mkIntercept:
            with mk.patch.object(FittableModel, "__call__",
                                 autospec=True) as mkCall:
                assert mkCall.return_value == spl(*args, **kwargs)
                assert mkCall.call_args_list == \
                    [mk.call(spl, *args, **new_kwargs)]
                assert mkIntercept.call_args_list == \
                    [mk.call(spl, **kwargs)]

    def test__create_parameter(self):
        print('')
        np.random.seed(37)
        base_vec = np.random.random(20)
        test = base_vec.copy()
        fixed_test = base_vec.copy()

        class Spline(_NewSpline):
            @property
            def test(self):
                return test

            @property
            def fixed_test(self):
                return fixed_test

        spl = Spline()
        assert (spl.test == test).all()
        assert (spl.fixed_test == fixed_test).all()

        for index in range(20):
            name = f"test_name{index}"
            spl._create_parameter(name, index, 'test')
            spl._coeff_names.append(name)
            assert hasattr(spl, name)
            param = getattr(spl, name)
            assert isinstance(param, Parameter)
            assert param.model == spl
            assert param.fixed is False
            assert param.value == test[index] == spl.test[index] == base_vec[index]
            new_set = np.random.random()
            param.value = new_set
            assert spl.test[index] == new_set
            assert spl.test[index] != base_vec[index]
            new_get = np.random.random()
            spl.test[index] = new_get
            assert param.value == new_get
            assert param.value != new_set

        # test the parameters initialize correctly
        spl._initialize_parameters((), {})
        spl._initialize_slices()
        assert (spl.parameters == spl.test).all()

        for index in range(20):
            name = f"fixed_test_name{index}"
            spl._create_parameter(name, index, 'fixed_test', True)
            spl._knot_names.append(name)
            assert hasattr(spl, name)
            param = getattr(spl, name)
            assert isinstance(param, Parameter)
            assert param.model == spl
            assert param.fixed is True
            assert param.value == fixed_test[index] == spl.fixed_test[index] == base_vec[index]
            new_set = np.random.random()
            param.value = new_set
            assert spl.fixed_test[index] == new_set
            assert spl.fixed_test[index] != base_vec[index]
            new_get = np.random.random()
            spl.fixed_test[index] = new_get
            assert param.value == new_get
            assert param.value != new_set

        # test the parameters initialize correctly
        spl._initialize_parameters((), {})
        spl._initialize_slices()
        assert (spl.parameters == np.concatenate((spl.fixed_test, spl.test))).all()

    def test__create_parameters(self):
        np.random.seed(37)
        test = np.random.random(20)

        class Spline(_NewSpline):
            @property
            def test(self):
                return test

        spl = Spline()

        fixed = mk.MagicMock()
        with mk.patch.object(_NewSpline, '_create_parameter',
                             autospec=True) as mkCreate:
            params = spl._create_parameters("test_param", "test", fixed)
            assert params == [f"test_param{idx}" for idx in range(20)]
            assert mkCreate.call_args_list == \
                [mk.call(spl, f"test_param{idx}", idx, 'test', fixed) for idx in range(20)]

    def test__init_parameters(self):
        spl = _NewSpline()

        with pytest.raises(NotImplementedError) as err:
            spl._init_parameters()
        assert str(err.value) == \
            "This needs to be implemented"

    def test__init_data(self):
        spl = _NewSpline()

        with pytest.raises(NotImplementedError) as err:
            spl._init_data(mk.MagicMock(), mk.MagicMock(), mk.MagicMock())
        assert str(err.value) == \
            "This needs to be implemented"
        with pytest.raises(NotImplementedError) as err:
            spl._init_data(mk.MagicMock(), mk.MagicMock())
        assert str(err.value) == \
            "This needs to be implemented"

    def test__init_spline(self):
        spl = _NewSpline()

        knots = mk.MagicMock()
        coeffs = mk.MagicMock()
        bounds = mk.MagicMock()
        with mk.patch.object(_NewSpline, "_init_parameters",
                             autospec=True) as mkInitParam:
            with mk.patch.object(_NewSpline, "_init_data",
                                 autospec=True) as mkInitData:
                with mk.patch.object(_NewSpline, "_initialize_parameters",
                                     autospec=True) as mkParameters:
                    with mk.patch.object(_NewSpline, "_initialize_slices",
                                         autospec=True) as mkSlices:
                        main = mk.MagicMock()
                        main.attach_mock(mkInitParam, 'init_param')
                        main.attach_mock(mkInitData, 'init_data')
                        main.attach_mock(mkParameters, 'parameters')
                        main.attach_mock(mkSlices, 'slices')

                        spl._init_spline(knots, coeffs, bounds)
                        assert main.mock_calls == [
                            mk.call.init_data(spl, knots, coeffs, bounds),
                            mk.call.init_param(spl),
                            mk.call.parameters(spl, (), {}),
                            mk.call.slices(spl),
                        ]


@pytest.mark.skipif('not HAS_SCIPY')
class TestNewSpline1D:
    def setup_class(self):
        def func(x, noise=0):
            return np.exp(-x**2) + 0.1*noise

        self.x = np.linspace(-3, 3, npts)
        self.y = func(self.x, noise)
        self.truth = func(self.x)

        arg_sort = np.argsort(self.x)
        np.random.shuffle(arg_sort)

        self.x_s = self.x[arg_sort]
        self.y_s = func(self.x_s, noise[arg_sort])

        self.npts_out = 1000
        self.xs = np.linspace(-3, 3, self.npts_out)

        self.t = np.linspace(-3, 3, nknots)[1:-1]

    def check_parameter(self, spl, base_name, name, index, value, fixed):
        assert base_name in name
        assert index == int(name.split(base_name)[-1])
        knot_name = f"{base_name}{index}"
        assert knot_name == name
        assert hasattr(spl, name)
        param = getattr(spl, name)
        assert isinstance(param, Parameter)
        assert param.name == name
        assert param.value == value(index)
        assert param.model == spl
        assert param.fixed is fixed

    def check_parameters(self, spl, params, base_name, value, fixed):
        for idx, name in enumerate(params):
            self.check_parameter(spl, base_name, name, idx, value, fixed)

    def update_parameters(self, spl, knots, value):
        for name in knots:
            param = getattr(spl, name)
            param.value = value
            assert param.value == value

    def test__init__with_no_knot_information(self):
        spl = NewSpline1D()
        assert spl._degree == 3
        assert spl._user_knots is False
        assert spl._t is None
        assert spl._c is None

        # Check no parameters created
        assert len(spl._knot_names) == 0
        assert len(spl._coeff_names) == 0

    def test___init__with_number_of_knots(self):
        spl = NewSpline1D(knots=10)

        # Check baseline data
        assert spl._degree == 3
        assert spl._user_knots is False
        assert spl._nu == None

        # Check vector data
        assert len(spl._t) == 18
        t = np.zeros(18)
        t[-4:] = 1
        assert (spl._t == t).all()
        assert len(spl._c) == 18
        assert (spl._c == np.zeros(18)).all()

        # Check all parameter names created:
        assert len(spl._knot_names) == 18
        assert len(spl._coeff_names) == 18

        # Check knot values:
        def value0(idx):
            if idx < 18 - 4:
                return 0
            else:
                return 1
        self.check_parameters(spl, spl._knot_names, "knot", value0, True)

        # Check coeff values:
        def value1(idx):
            return 0
        self.check_parameters(spl, spl._coeff_names, "coeff", value1, False)

        # Check parameters
        knots = np.array([value0(idx) for idx in range(18)])
        coeffs = np.array([value1(idx) for idx in range(18)])
        parameters = np.concatenate((knots, coeffs))
        assert (spl.parameters == parameters).all()

    def test___init__with_full_custom_knots(self):
        t = 17*np.arange(20) - 32
        spl = NewSpline1D(knots=t)

        # Check baseline data
        assert spl._degree == 3
        assert spl._user_knots is True
        assert spl._nu == None

        # Check vector data
        assert (spl._t == t).all()
        assert len(spl._c) == 20
        assert (spl._c == np.zeros(20)).all()

        # Check all parameter names created
        assert len(spl._knot_names) == 20
        assert len(spl._coeff_names) == 20

        # Check knot values:
        def value0(idx):
            return t[idx]
        self.check_parameters(spl, spl._knot_names, "knot", value0, True)

        # Check coeff values
        def value1(idx):
            return 0
        self.check_parameters(spl, spl._coeff_names, "coeff", value1, False)

        # Check parameters
        knots = np.array([value0(idx) for idx in range(20)])
        coeffs = np.array([value1(idx) for idx in range(20)])
        parameters = np.concatenate((knots, coeffs))
        assert (spl.parameters == parameters).all()

    def test___init__with_interior_custom_knots(self):
        t = np.arange(1, 20)
        spl = NewSpline1D(knots=t, bounds=[0, 20])
        # Check baseline data
        assert spl._degree == 3
        assert spl._user_knots is True
        assert spl._nu == None

        # Check vector data
        assert len(spl._t) == 27
        assert (spl._t[4:-4] == t).all()
        assert (spl._t[:4] == 0).all()
        assert (spl._t[-4:] == 20).all()

        assert len(spl._c) == 27
        assert (spl._c == np.zeros(27)).all()

        # Check knot values:
        def value0(idx):
            if idx < 4:
                return 0
            elif idx >= 19 + 4:
                return 20
            else:
                return t[idx-4]
        self.check_parameters(spl, spl._knot_names, "knot", value0, True)

        # Check coeff values
        def value1(idx):
            return 0
        self.check_parameters(spl, spl._coeff_names, "coeff", value1, False)

        # Check parameters
        knots = np.array([value0(idx) for idx in range(27)])
        coeffs = np.array([value1(idx) for idx in range(27)])
        parameters = np.concatenate((knots, coeffs))
        assert (spl.parameters == parameters).all()

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

        # Check knot base values
        def value0(idx):
            if idx < 18 - 4:
                return 0
            else:
                return 1
        self.check_parameters(spl, spl._knot_names, "knot", value0, True)

        # Check knot vector -> knot parameter link
        t = np.arange(18)
        spl._t = t.copy()

        def value1(idx):
            return t[idx]
        self.check_parameters(spl, spl._knot_names, "knot", value1, True)

        # Check knot parameter -> knot vector link
        self.update_parameters(spl, spl._knot_names, 3)
        assert (spl._t[:] == 3).all()

        # Check coeff base values
        def value2(idx):
            return 0
        self.check_parameters(spl, spl._coeff_names, "coeff", value2, False)

        # Check coeff vector -> coeff parameter link
        c = 5 * np.arange(18) + 18
        spl._c = c.copy()

        def value3(idx):
            return c[idx]
        self.check_parameters(spl, spl._coeff_names, "coeff", value3, False)

        # Check coeff parameter -> coeff vector link
        self.update_parameters(spl, spl._coeff_names, 4)
        assert (spl._c[:] == 4).all()

    def test_two_splines(self):
        spl0 = NewSpline1D(knots=10)
        spl1 = NewSpline1D(knots=15, degree=2)

        assert spl0._degree == 3
        assert len(spl0._t) == 18
        t = np.zeros(18)
        t[-4:] = 1
        assert (spl0._t == t).all()
        assert len(spl0._c) == 18
        assert (spl0._c == np.zeros(18)).all()
        assert spl1._degree == 2
        assert len(spl1._t) == 21
        t = np.zeros(21)
        t[-3:] = 1
        assert (spl1._t == t).all()
        assert len(spl1._c) == 21
        assert (spl1._c == np.zeros(21)).all()

        # Check all knot names created
        assert len(spl0._knot_names) == 18
        assert len(spl1._knot_names) == 21

        # Check knot base values
        def value0(idx):
            if idx < 18 - 4:
                return 0
            else:
                return 1
        self.check_parameters(spl0, spl0._knot_names, "knot", value0, True)

        def value1(idx):
            if idx < 21 - 3:
                return 0
            else:
                return 1
        self.check_parameters(spl1, spl1._knot_names, "knot", value1, True)

        # Check knot vector -> knot parameter link
        t0 = 7 * np.arange(18) + 27
        t1 = 11 * np.arange(21) + 19
        spl0._t[:] = t0.copy()
        spl1._t[:] = t1.copy()

        def value2(idx):
            return t0[idx]
        self.check_parameters(spl0, spl0._knot_names, "knot", value2, True)

        def value3(idx):
            return t1[idx]
        self.check_parameters(spl1, spl1._knot_names, "knot", value3, True)

        # Check knot parameter -> knot vector link
        self.update_parameters(spl0, spl0._knot_names, 3)
        self.update_parameters(spl1, spl1._knot_names, 4)
        assert (spl0._t[:] == 3).all()
        assert (spl1._t[:] == 4).all()

        # Check all coeff names created
        assert len(spl0._coeff_names) == 18
        assert len(spl1._coeff_names) == 21

        # Check coeff base values
        def value4(idx):
            return 0
        self.check_parameters(spl0, spl0._coeff_names, "coeff", value4, False)
        self.check_parameters(spl1, spl1._coeff_names, "coeff", value4, False)

        # Check coeff vector -> coeff parameter link
        c0 = 17 * np.arange(18) + 14
        c1 = 37 * np.arange(21) + 47
        spl0._c[:] = c0.copy()
        spl1._c[:] = c1.copy()

        def value5(idx):
            return c0[idx]
        self.check_parameters(spl0, spl0._coeff_names, "coeff", value5, False)

        def value6(idx):
            return c1[idx]
        self.check_parameters(spl1, spl1._coeff_names, "coeff", value6, False)

        # Check coeff parameter -> coeff vector link
        self.update_parameters(spl0, spl0._coeff_names, 5)
        self.update_parameters(spl1, spl1._coeff_names, 6)
        assert (spl0._t[:] == 3).all()
        assert (spl1._t[:] == 4).all()
        assert (spl0._c[:] == 5).all()
        assert (spl1._c[:] == 6).all()

    def test__knot_names(self):
        # no parameters
        spl = NewSpline1D()
        assert spl._knot_names == []

        # some parameters
        knot_names = [f"knot{idx}" for idx in range(18)]

        spl = NewSpline1D(10)
        assert spl._knot_names == knot_names

    def test__coeff_names(self):
        # no parameters
        spl = NewSpline1D()
        assert spl._coeff_names == []

        # some parameters
        coeff_names = [f"coeff{idx}" for idx in range(18)]

        spl = NewSpline1D(10)
        assert spl._coeff_names == coeff_names

    def test_param_names(self):
        # no parameters
        spl = NewSpline1D()
        assert spl.param_names == ()

        # some parameters
        knot_names = [f"knot{idx}" for idx in range(18)]
        coeff_names = [f"coeff{idx}" for idx in range(18)]
        param_names = knot_names + coeff_names

        spl = NewSpline1D(10)
        assert spl.param_names == tuple(param_names)

    def test_t(self):
        # no parameters
        spl = NewSpline1D()
        # test get
        assert spl._t is None
        assert (spl.t == [0, 0, 0, 0, 1, 1, 1, 1]).all()
        # test set
        with pytest.raises(ValueError) as err:
            spl.t = mk.MagicMock()
        assert str(err.value) ==\
            "The model parameters must be initialized before setting knots."

        # with parameters
        spl = NewSpline1D(10)
        # test get
        t = np.zeros(18)
        t[-4:] = 1
        assert (spl._t == t).all()
        assert (spl.t == t).all()
        # test set
        spl.t = (np.arange(18) + 15)
        assert (spl._t == (np.arange(18) + 15)).all()
        assert (spl.t == (np.arange(18) + 15)).all()
        assert (spl.t != t).all()
        # set error
        for idx in range(30):
            if idx == 18:
                continue
            with pytest.raises(ValueError) as err:
                spl.t = np.arange(idx)
            assert str(err.value) == \
                "There must be exactly as many knots as previously defined."

    def test_c(self):
        # no parameters
        spl = NewSpline1D()
        # test get
        assert spl._c is None
        assert (spl.c == [0, 0, 0, 0, 0, 0, 0, 0]).all()
        # test set
        with pytest.raises(ValueError) as err:
            spl.c = mk.MagicMock()
        assert str(err.value) ==\
            "The model parameters must be initialized before setting coeffs."

        # with parameters
        spl = NewSpline1D(10)
        # test get
        assert (spl._c == np.zeros(18)).all()
        assert (spl.c == np.zeros(18)).all()
        # test set
        spl.c = (np.arange(18) + 15)
        assert (spl._c == (np.arange(18) + 15)).all()
        assert (spl.c == (np.arange(18) + 15)).all()
        assert (spl.c != np.zeros(18)).all()
        # set error
        for idx in range(30):
            if idx == 18:
                continue
            with pytest.raises(ValueError) as err:
                spl.c = np.arange(idx)
            assert str(err.value) == \
                "There must be exactly as many coeffs as previously defined."

    def test_degree(self):
        # default degree
        spl = NewSpline1D()
        # test get
        assert spl._degree == 3
        assert spl.degree == 3
        # test set
        spl.degree = 3
        assert spl._degree == 3
        assert spl.degree == 3
        # test set error
        with pytest.raises(ValueError) as err:
            spl.degree = 19
        assert str(err.value) ==\
            "The value of degree cannot be changed!"

        # non-default degree
        spl = NewSpline1D(degree=2)
        # test get
        assert spl._degree == 2
        assert spl.degree == 2
        # test set
        spl.degree = 2
        assert spl._degree == 2
        assert spl.degree == 2
        # test set error
        with pytest.raises(ValueError) as err:
            spl.degree = 19
        assert str(err.value) ==\
            "The value of degree cannot be changed!"

    def test__initialized(self):
        # no parameters
        spl = NewSpline1D()
        assert spl._initialized is False

        # with parameters
        spl = NewSpline1D(knots=10, degree=2)
        assert spl._initialized is True

    def test_tck(self):
        # no parameters
        spl = NewSpline1D()
        # test get
        assert (spl.t == [0, 0, 0, 0, 1, 1, 1, 1]).all()
        assert (spl.c == [0, 0, 0, 0, 0, 0, 0, 0]).all()
        assert spl.degree == 3
        tck = spl.tck
        assert (tck[0] == spl.t).all()
        assert (tck[1] == spl.c).all()
        assert tck[2] == spl.degree
        # test set
        assert spl._t is None
        assert spl._c is None
        assert spl._knot_names == []
        assert spl._coeff_names == []
        t = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5])
        np.random.seed(619)
        c = np.random.random(12)
        k = 3
        spl.tck = (t, c, k)
        assert (spl._t == t).all()
        assert (spl._c == c).all()
        assert spl.degree == k

        def value0(idx):
            return t[idx]
        self.check_parameters(spl, spl._knot_names, "knot", value0, True)

        def value1(idx):
            return c[idx]
        self.check_parameters(spl, spl._coeff_names, "coeff", value1, False)

        # with parameters
        spl = NewSpline1D(knots=10, degree=2)
        # test get
        t = np.zeros(16)
        t[-3:] = 1
        assert (spl.t == t).all()
        assert (spl.c == np.zeros(16)).all()
        assert spl.degree == 2
        tck = spl.tck
        assert (tck[0] == spl.t).all()
        assert (tck[1] == spl.c).all()
        assert tck[2] == spl.degree
        # test set
        t = 5*np.arange(16) + 11
        c = 7*np.arange(16) + 13
        k = 2
        spl.tck = (t, c, k)
        assert (spl.t == t).all()
        assert (spl.c == c).all()
        assert spl.degree == k
        tck = spl.tck
        assert (tck[0] == spl.t).all()
        assert (tck[1] == spl.c).all()
        assert tck[2] == spl.degree

    def test_bspline(self):
        from scipy.interpolate import BSpline
        # no parameters
        spl = NewSpline1D()
        bspline = spl.bspline

        assert isinstance(bspline, BSpline)
        assert (bspline.tck[0] == spl.tck[0]).all()
        assert (bspline.tck[1] == spl.tck[1]).all()
        assert bspline.tck[2] == spl.tck[2]

        t = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5, 5])
        np.random.seed(619)
        c = np.random.random(12)
        k = 3

        def value0(idx):
            return t[idx]

        def value1(idx):
            return c[idx]

        # set (bspline)
        spl = NewSpline1D()
        assert spl._t is None
        assert spl._c is None
        assert spl._knot_names == []
        assert spl._coeff_names == []
        bspline = BSpline(t, c, k)
        spl.bspline = bspline
        assert (spl._t == t).all()
        assert (spl._c == c).all()
        assert spl.degree == k
        self.check_parameters(spl, spl._knot_names, "knot", value0, True)
        self.check_parameters(spl, spl._coeff_names, "coeff", value1, False)

        # set (tuple spline)
        spl = NewSpline1D()
        assert spl._t is None
        assert spl._c is None
        assert spl._knot_names == []
        assert spl._coeff_names == []
        spl.bspline = (t, c, k)
        assert (spl._t == t).all()
        assert (spl._c == c).all()
        assert spl.degree == k
        self.check_parameters(spl, spl._knot_names, "knot", value0, True)
        self.check_parameters(spl, spl._coeff_names, "coeff", value1, False)

        # with parameters
        spl = NewSpline1D(knots=10, degree=2)
        bspline = spl.bspline

        assert isinstance(bspline, BSpline)
        assert (bspline.tck[0] == spl.tck[0]).all()
        assert (bspline.tck[1] == spl.tck[1]).all()
        assert bspline.tck[2] == spl.tck[2]

    def test_knots(self):
        # no parameters
        spl = NewSpline1D()
        assert spl.knots == []

        # with parameters
        spl = NewSpline1D(10)
        knots = spl.knots
        assert len(knots) == 18

        for knot in knots:
            assert isinstance(knot, Parameter)
            assert hasattr(spl, knot.name)
            assert getattr(spl, knot.name) == knot

    def test_coeffs(self):
        # no parameters
        spl = NewSpline1D()
        assert spl.coeffs == []

        # with parameters
        spl = NewSpline1D(10)
        coeffs = spl.coeffs
        assert len(coeffs) == 18

        for coeff in coeffs:
            assert isinstance(coeff, Parameter)
            assert hasattr(spl, coeff.name)
            assert getattr(spl, coeff.name) == coeff

    def test__init_parameters(self):
        spl = NewSpline1D()

        with mk.patch.object(NewSpline1D, '_create_parameters',
                             autospec=True) as mkCreate:
            spl._init_parameters()
            assert mkCreate.call_args_list == [
                mk.call(spl, "knot", "t", fixed=True),
                mk.call(spl, "coeff", "c")
            ]

    def test__init_bounds(self):
        spl = NewSpline1D()

        has_bounds, lower, upper = spl._init_bounds()
        assert has_bounds is False
        assert (lower == [0, 0, 0, 0]).all()
        assert (upper == [1, 1, 1, 1]).all()
        assert spl._user_bounding_box is None

        has_bounds, lower, upper = spl._init_bounds((-5, 5))
        assert has_bounds is True
        assert (lower == [-5, -5, -5, -5]).all()
        assert (upper == [5, 5, 5, 5]).all()
        assert spl._user_bounding_box == (-5, 5)

    def test__init_knots(self):
        np.random.seed(19)
        lower = np.random.random(4)
        upper = np.random.random(4)

        # Integer
        with mk.patch.object(NewSpline1D, "bspline",
                             new_callable=mk.PropertyMock) as mkBspline:
            spl = NewSpline1D()
            assert spl._t is None
            spl._init_knots(10, mk.MagicMock(), lower, upper)
            t = np.concatenate((lower, np.zeros(10), upper))
            assert (spl._t == t).all()
            assert mkBspline.call_args_list == [mk.call()]

        # vector with bounds
        with mk.patch.object(NewSpline1D, "bspline",
                             new_callable=mk.PropertyMock) as mkBspline:
            knots = np.random.random(10)
            spl = NewSpline1D()
            assert spl._t is None
            spl._init_knots(knots, True, lower, upper)
            t = np.concatenate((lower, knots, upper))
            assert (spl._t == t).all()
            assert mkBspline.call_args_list == [mk.call()]

        # vector with no bounds
        with mk.patch.object(NewSpline1D, "bspline",
                             new_callable=mk.PropertyMock) as mkBspline:
            knots = np.random.random(10)
            spl = NewSpline1D()
            assert spl._t is None
            spl._init_knots(knots, False, lower, upper)
            assert (spl._t == knots).all()
            assert mkBspline.call_args_list == [mk.call()]

            # error
            for num in range(8):
                knots = np.random.random(num)
                spl = NewSpline1D()
                assert spl._t is None
                with pytest.raises(ValueError) as err:
                    spl._init_knots(knots, False, lower, upper)
                assert str(err.value) == \
                    "Must have at least 8 knots."

        # Error
        spl = NewSpline1D()
        assert spl._t is None
        with pytest.raises(ValueError) as err:
            spl._init_knots(0.5, False, lower, upper)
        assert str(err.value) ==\
            "Knots: 0.5 must be iterable or value"

    def test__init_coeffs(self):
        np.random.seed(492)
        # No coeffs
        with mk.patch.object(NewSpline1D, "bspline",
                             new_callable=mk.PropertyMock) as mkBspline:
            spl = NewSpline1D()
            assert spl._c is None
            spl._t = [1, 2, 3, 4]
            spl._init_coeffs()
            assert (spl._c == [0, 0, 0, 0]).all()
            assert mkBspline.call_args_list == [mk.call()]

        # Some coeffs
        with mk.patch.object(NewSpline1D, "bspline",
                             new_callable=mk.PropertyMock) as mkBspline:
            coeffs = np.random.random(10)
            spl = NewSpline1D()
            assert spl._c is None
            spl._init_coeffs(coeffs)
            assert (spl._c == coeffs).all()
            assert mkBspline.call_args_list == [mk.call()]

    def test__init_data(self):
        spl = NewSpline1D()

        knots = mk.MagicMock()
        coeffs = mk.MagicMock()
        bounds = mk.MagicMock()
        has_bounds = mk.MagicMock()
        lower = mk.MagicMock()
        upper = mk.MagicMock()
        with mk.patch.object(NewSpline1D, '_init_bounds', autospec=True,
                             return_value=(has_bounds, lower, upper)) as mkBounds:
            with mk.patch.object(NewSpline1D, '_init_knots',
                                 autospec=True) as mkKnots:
                with mk.patch.object(NewSpline1D, '_init_coeffs',
                                     autospec=True) as mkCoeffs:
                    main = mk.MagicMock()
                    main.attach_mock(mkBounds, 'bounds')
                    main.attach_mock(mkKnots, 'knots')
                    main.attach_mock(mkCoeffs, 'coeffs')

                    spl._init_data(knots, coeffs, bounds)
                    assert main.mock_calls == [
                        mk.call.bounds(spl, bounds),
                        mk.call.knots(spl, knots, has_bounds, lower, upper),
                        mk.call.coeffs(spl, coeffs)
                    ]

    def test_evaluate(self):
        spl = NewSpline1D()

        args = tuple([mk.MagicMock() for _ in range(3)])
        kwargs = {f"test{idx}": mk.MagicMock() for idx in range(3)}
        new_kwargs = {f"new_test{idx}": mk.MagicMock() for idx in range(3)}

        with mk.patch.object(_NewSpline, 'evaluate', autospec=True,
                             return_value=new_kwargs) as mkEval:
            with mk.patch.object(NewSpline1D, "bspline",
                                 new_callable=mk.PropertyMock) as mkBspline:
                assert mkBspline.return_value.return_value == spl.evaluate(*args, **kwargs)
                assert mkBspline.return_value.call_args_list == \
                    [mk.call(args[0], **new_kwargs)]
                assert mkBspline.call_args_list == [mk.call()]
                assert mkEval.call_args_list == \
                    [mk.call(spl, *args, **kwargs)]

        # Error
        for idx in range(5, 8):
            with mk.patch.object(_NewSpline, 'evaluate', autospec=True,
                                 return_value={'nu': idx}):
                with pytest.raises(RuntimeError) as err:
                    spl.evaluate(*args, **kwargs)
                assert str(err.value) == \
                    "Cannot evaluate a derivative of order higher than 4"

    def check_knots_created(self, spl, k):
        def value0(idx):
            return self.x[0]

        def value1(idx):
            return self.x[-1]

        for idx in range(k + 1):
            name = f"knot{idx}"
            self.check_parameter(spl, "knot", name, idx, value0, True)

            index = len(spl.t) - (k + 1) + idx
            name = f"knot{index}"
            self.check_parameter(spl, "knot", name, index, value1, True)

        def value3(idx):
            return spl.t[idx]

        assert len(spl._knot_names) == len(spl.t)
        for idx, name in enumerate(spl._knot_names):
            assert name == f"knot{idx}"
            self.check_parameter(spl, "knot", name, idx, value3, True)

    def check_coeffs_created(self, spl):
        def value(idx):
            return spl.c[idx]

        assert len(spl._coeff_names) == len(spl.c)
        for idx, name in enumerate(spl._coeff_names):
            assert name == f"coeff{idx}"
            self.check_parameter(spl, "coeff", name, idx, value, False)

    @pytest.mark.parametrize(new_variables_1D, new_tests_1D)
    def test_interpolate_data(self, w, k):
        spl = NewSpline1D(degree=k)
        assert spl._t is None
        assert spl._c is None
        assert spl.degree == k

        fit_spline = spl.interpolate_data(self.x, self.y, w=w)
        assert len(spl.t) == (len(self.x) + k + 1) == len(spl._knot_names)
        self.check_knots_created(spl, k)
        self.check_coeffs_created(spl)

        from scipy.interpolate import InterpolatedUnivariateSpline
        assert isinstance(fit_spline, InterpolatedUnivariateSpline)
        spline = InterpolatedUnivariateSpline(self.x, self.y, w=w, k=k)

        assert (spl.t == spline._eval_args[0]).all()
        assert (spl.c == spline._eval_args[1]).all()

        assert (spl.t == fit_spline._eval_args[0]).all()
        assert (spl.c == spline._eval_args[1]).all()

        assert spline.get_residual() == 0
        assert spline.get_residual() == fit_spline.get_residual()
        assert (spl(self.x) == spline(self.x)).all()
        assert (spl(self.x) == fit_spline(self.x)).all()

        assert_allclose(spl(self.x), self.y)
        assert_allclose(spl(self.x), self.truth, atol=1)

        # test warning
        knots = np.linspace(self.x[0], self.x[-1], len(self.x) + k + 1)
        spl = NewSpline1D(knots=knots, degree=k)
        with pytest.warns(AstropyUserWarning):
            spl.interpolate_data(self.x, self.y, w=w)

    def test_interpolate(self):
        x = mk.MagicMock()
        y = mk.MagicMock()

        # Defaults
        with mk.patch.object(NewSpline1D, 'interpolate_data',
                             autospec=True) as mkInterpolate:
            spl = NewSpline1D.interpolate(x, y)
            assert isinstance(spl, NewSpline1D)
            assert spl._t is None
            assert spl._c is None
            assert spl.degree == 3
            assert mkInterpolate.call_args_list == \
                [mk.call(spl, x, y, w=None, bbox=[None, None])]

        # No Defaults
        k = mk.MagicMock()
        w = mk.MagicMock()
        bbox = mk.MagicMock()
        with mk.patch.object(NewSpline1D, 'interpolate_data',
                             autospec=True) as mkInterpolate:
            spl = NewSpline1D.interpolate(x, y, k=k, w=w, bbox=bbox)
            assert isinstance(spl, NewSpline1D)
            assert spl._t is None
            assert spl._c is None
            assert spl.degree == k
            assert mkInterpolate.call_args_list == \
                [mk.call(spl, x, y, w=w, bbox=bbox)]

    @pytest.mark.parametrize(new_variables_1D, new_tests_1D)
    def test_NewInterpolatingSplineFitter(self, w, k):
        fitter = NewInterpolatingSplineFitter()

        spl = NewSpline1D(degree=k)
        assert spl._t is None
        assert spl._c is None
        assert spl.degree == k

        fit_spl = fitter(spl, self.x, self.y, weights=w)
        assert isinstance(fit_spl, NewSpline1D)
        assert spl._t is None
        assert fit_spl._t is not None
        assert spl._c is None
        assert fit_spl._c is not None
        assert spl.degree == k == fit_spl.degree

        from scipy.interpolate import InterpolatedUnivariateSpline
        fit_spline = fitter.fit_info['fit_spline']
        assert isinstance(fit_spline, InterpolatedUnivariateSpline)
        assert fit_spline.get_residual() == fitter.fit_info['resid']

        assert (fit_spl.t == fit_spline._eval_args[0]).all()
        assert (fit_spl.c == fit_spline._eval_args[1]).all()

        assert (fit_spl(self.x) == fit_spline(self.x)).all()

        assert_allclose(fit_spl(self.x), self.y, atol=1)
        assert_allclose(fit_spl(self.x), self.truth, atol=1)

    @pytest.mark.parametrize(new_variables_1D, new_tests_1D)
    @pytest.mark.parametrize('s', [None, 0.01])
    def test_smoothing_data(self, w, k, s):
        spl = NewSpline1D(degree=k)
        assert spl._t is None
        assert spl._c is None
        assert spl.degree == k

        fit_spline = spl.smoothing_fit(self.x, self.y, s=s, w=w)
        self.check_knots_created(spl, k)
        self.check_coeffs_created(spl)

        from scipy.interpolate import UnivariateSpline
        assert isinstance(fit_spline, UnivariateSpline)
        spline = UnivariateSpline(self.x, self.y, w=w, k=k, s=s)

        assert (spl.t == spline._eval_args[0]).all()
        assert (spl.c == spline._eval_args[1]).all()

        assert (spl.t == fit_spline._eval_args[0]).all()
        assert (spl.c == fit_spline._eval_args[1]).all()

        assert (spl(self.x) == spline(self.x)).all()
        assert (spl(self.x) == fit_spline(self.x)).all()

        assert_allclose(spl(self.x), self.y, atol=1)
        assert_allclose(spl(self.x), self.truth, atol=1)

    def test_smoothing(self):
        x = mk.MagicMock()
        y = mk.MagicMock()

        # Defaults
        with mk.patch.object(NewSpline1D, 'smoothing_fit',
                             autospec=True) as mkSmoothing:
            spl = NewSpline1D.smoothing(x, y)
            assert isinstance(spl, NewSpline1D)
            assert spl._t is None
            assert spl._c is None
            assert spl.degree == 3
            assert mkSmoothing.call_args_list == \
                [mk.call(spl, x, y, s=None, w=None, bbox=[None, None])]

        # No Defaults
        k = mk.MagicMock()
        s = mk.MagicMock()
        w = mk.MagicMock()
        bbox = mk.MagicMock()
        with mk.patch.object(NewSpline1D, 'smoothing_fit',
                             autospec=True) as mkSmoothing:
            spl = NewSpline1D.smoothing(x, y, k=k, s=s, w=w, bbox=bbox)
            assert isinstance(spl, NewSpline1D)
            assert spl._t is None
            assert spl._c is None
            assert spl.degree == k
            assert mkSmoothing.call_args_list == \
                [mk.call(spl, x, y, s=s, w=w, bbox=bbox)]

    @pytest.mark.parametrize(new_variables_1D, new_tests_1D)
    @pytest.mark.parametrize('s', [None, 0.01])
    def test_NewSmoothingSplineFitter(self, w, k, s):
        fitter = NewSmoothingSplineFitter()

        spl = NewSpline1D(degree=k)
        assert spl._t is None
        assert spl._c is None
        assert spl.degree == k

        fit_spl = fitter(spl, self.x, self.y, s=s, weights=w)
        assert isinstance(fit_spl, NewSpline1D)
        assert spl._t is None
        assert fit_spl._t is not None
        assert spl._c is None
        assert fit_spl._c is not None
        assert spl.degree == k == fit_spl.degree

        from scipy.interpolate import UnivariateSpline
        fit_spline = fitter.fit_info['fit_spline']
        assert isinstance(fit_spline, UnivariateSpline)
        assert fit_spline.get_residual() == fitter.fit_info['resid']

        assert (fit_spl.t == fit_spline._eval_args[0]).all()
        assert (fit_spl.c == fit_spline._eval_args[1]).all()

        assert (fit_spl(self.x) == fit_spline(self.x)).all()

        assert_allclose(fit_spl(self.x), self.y, atol=1)
        assert_allclose(fit_spl(self.x), self.truth, atol=1)

    @pytest.mark.parametrize(new_variables_1D, new_tests_1D)
    def test_lsq_fit(self, w, k):
        knots = [-1, 0, 1]
        t = np.concatenate(([self.x[0]]*(k + 1), knots, [self.x[-1]]*(k + 1)))
        c = np.zeros(len(t))

        # With knots preset
        spl = NewSpline1D(knots=knots, degree=k, bounds=[self.x[0], self.x[-1]])
        assert (spl.t == t).all()
        assert (spl.c == c).all()
        assert (spl.t_interior == knots).all()
        assert spl.degree == k

        fit_spline = spl.lsq_fit(self.x, self.y, w=w)
        assert len(spl.t) == len(t) == len(spl._knot_names)
        self.check_knots_created(spl, k)
        self.check_coeffs_created(spl)

        from scipy.interpolate import LSQUnivariateSpline
        assert isinstance(fit_spline, LSQUnivariateSpline)
        spline = LSQUnivariateSpline(self.x, self.y, knots, w=w, k=k)

        assert (spl.t == spline._eval_args[0]).all()
        assert (spl.c == spline._eval_args[1]).all()

        assert (spl.t == fit_spline._eval_args[0]).all()
        assert (spl.c == fit_spline._eval_args[1]).all()

        assert_allclose(spline.get_residual(), 0.1, atol=1)
        assert spline.get_residual() == fit_spline.get_residual()
        assert (spl(self.x) == spline(self.x)).all()
        assert (spl(self.x) == fit_spline(self.x)).all()

        assert_allclose(spl(self.x), self.y, atol=1)
        assert_allclose(spl(self.x), self.truth, atol=1)

        # Pass knots via fitter function
        with pytest.warns(AstropyUserWarning):
            spl.lsq_fit(self.x, self.y, knots, w=w)

        # Pass no knots
        spl = NewSpline1D(degree=k)
        with pytest.raises(RuntimeError) as err:
            spl.lsq_fit(self.x, self.y, w=w)
        assert str(err.value) ==\
            "No knots have been provided"

    def test_lsq(self):
        x = mk.MagicMock()
        y = mk.MagicMock()
        t = mk.MagicMock()

        # Defaults
        with mk.patch.object(NewSpline1D, 'lsq_fit',
                             autospec=True) as mkLSQ:
            spl = NewSpline1D.lsq(x, y, t)
            assert isinstance(spl, NewSpline1D)
            assert spl._t is None
            assert spl._c is None
            assert spl.degree == 3
            assert mkLSQ.call_args_list == \
                [mk.call(spl, x, y, t, w=None, bbox=[None, None])]

        # No Defaults
        k = mk.MagicMock()
        w = mk.MagicMock()
        bbox = mk.MagicMock()
        with mk.patch.object(NewSpline1D, 'lsq_fit',
                             autospec=True) as mkLSQ:
            spl = NewSpline1D.lsq(x, y, t, k=k, w=w, bbox=bbox)
            assert isinstance(spl, NewSpline1D)
            assert spl._t is None
            assert spl._c is None
            assert spl.degree == k
            assert mkLSQ.call_args_list == \
                [mk.call(spl, x, y, t, w=w, bbox=bbox)]

    @pytest.mark.parametrize(new_variables_1D, new_tests_1D)
    def test_NewLSQSplineFitter(self, w, k):
        knots = [-1, 0, 1]
        t = np.concatenate(([self.x[0]]*(k + 1), knots, [self.x[-1]]*(k + 1)))
        c = np.zeros(len(t))

        fitter = NewLSQSplineFitter()

        # With knots preset
        spl = NewSpline1D(knots=knots, degree=k, bounds=[self.x[0], self.x[-1]])
        assert (spl.t == t).all()
        assert (spl.c == c).all()
        assert spl.degree == k

        fit_spl = fitter(spl, self.x, self.y, weights=w)
        assert isinstance(fit_spl, NewSpline1D)
        assert (spl.t == t).all()
        assert (fit_spl.t == t).all()
        assert (spl.c == c).all()
        assert (spl.c != fit_spl.c).any()
        assert spl.degree == k == fit_spl.degree

        from scipy.interpolate import LSQUnivariateSpline
        fit_spline = fitter.fit_info['fit_spline']
        assert isinstance(fit_spline, LSQUnivariateSpline)
        assert fit_spline.get_residual() == fitter.fit_info['resid']

        assert (fit_spl.t == fit_spline._eval_args[0]).all()
        assert (fit_spl.c == fit_spline._eval_args[1]).all()

        assert (fit_spl(self.x) == fit_spline(self.x)).all()

        assert_allclose(fit_spl(self.x), self.y, atol=1)
        assert_allclose(fit_spl(self.x), self.truth, atol=1)

    @pytest.mark.parametrize(new_variables_1D, new_tests_1D)
    @pytest.mark.parametrize('s', [None, 0.01])
    def test_splrep_data_no_knots(self, w, k, s):
        spl = NewSpline1D(degree=k)
        assert spl._t is None
        assert spl._c is None
        assert spl.degree == k

        spl.splrep_data(self.x, self.y, w=w, s=s)
        self.check_knots_created(spl, k)
        self.check_coeffs_created(spl)

        from scipy.interpolate import splrep, BSpline
        tck = splrep(self.x, self.y, w=w, k=k, s=s)
        assert (spl.t == tck[0]).all()
        assert (spl.c == tck[1]).all()
        spline = BSpline(*tck)

        assert (spl(self.x) == spline(self.x)).all()

        assert_allclose(spl(self.x), self.y, atol=1)
        assert_allclose(spl(self.x), self.truth, atol=1)

    @pytest.mark.parametrize(new_variables_1D, new_tests_1D)
    def test_splrep_data_with_knots(self, w, k):
        knots = [-1, 0, 1]
        t = np.concatenate(([self.x[0]]*(k + 1), knots, [self.x[-1]]*(k + 1)))
        c = np.zeros(len(t))

        # With knots preset
        spl = NewSpline1D(knots=knots, degree=k, bounds=[self.x[0], self.x[-1]])
        assert (spl.t == t).all()
        assert (spl.c == c).all()
        assert (spl.t_interior == knots).all()
        assert spl.degree == k

        spl.splrep_data(self.x, self.y, w=w)
        self.check_knots_created(spl, k)
        self.check_coeffs_created(spl)

        from scipy.interpolate import splrep, BSpline
        tck = splrep(self.x, self.y, w=w, k=k, t=knots)
        assert (spl.t == tck[0]).all()
        assert (spl.c == tck[1]).all()
        spline = BSpline(*tck)

        assert (spl(self.x) == spline(self.x)).all()

        assert_allclose(spl(self.x), self.y, atol=1)
        assert_allclose(spl(self.x), self.truth, atol=1)

        # With no knots present
        spl = NewSpline1D(degree=k)
        assert spl._t is None
        assert spl._c is None
        assert spl.degree == k

        spl.splrep_data(self.x, self.y, w=w, t=knots)
        self.check_knots_created(spl, k)
        self.check_coeffs_created(spl)

        from scipy.interpolate import splrep, BSpline
        tck = splrep(self.x, self.y, w=w, k=k, t=knots)
        assert (spl.t == tck[0]).all()
        assert (spl.c == tck[1]).all()
        spline = BSpline(*tck)

        assert (spl(self.x) == spline(self.x)).all()

        assert_allclose(spl(self.x), self.y, atol=1)
        assert_allclose(spl(self.x), self.truth, atol=1)

        # test warning
        with pytest.warns(AstropyUserWarning):
            spl.splrep_data(self.x, self.y, w=w, t=knots)

    def test_splrep(self):
        x = mk.MagicMock()
        y = mk.MagicMock()

        # Defaults
        with mk.patch.object(NewSpline1D, 'splrep_data',
                             autospec=True) as mkSplrep:
            spl = NewSpline1D.splrep(x, y)
            assert isinstance(spl, NewSpline1D)
            assert spl._t is None
            assert spl._c is None
            assert spl.degree == 3
            assert mkSplrep.call_args_list == \
                [mk.call(spl, x, y, w=None, s=None, task=0, t=None, bbox=[None, None])]

        # No Defaults
        k = mk.MagicMock()
        w = mk.MagicMock()
        s = mk.MagicMock()
        task = mk.MagicMock()
        t = mk.MagicMock()
        bbox = mk.MagicMock()
        with mk.patch.object(NewSpline1D, 'splrep_data',
                             autospec=True) as mkSplrep:
            spl = NewSpline1D.splrep(x, y, k=k, w=w, s=s, task=task, t=t, bbox=bbox)
            assert isinstance(spl, NewSpline1D)
            assert spl._t is None
            assert spl._c is None
            assert spl.degree == k
            assert mkSplrep.call_args_list == \
                [mk.call(spl, x, y, w=w, s=s, task=task, t=t, bbox=bbox)]

    def generate_spline(self, w=None, bbox=[None]*2, k=None, s=None, t=None):
        if k is None:
            k = 3

        from scipy.interpolate import splrep, BSpline

        tck = splrep(self.x, self.y, w=w, xb=bbox[0], xe=bbox[1],
                     k=k, s=s, t=t)

        return BSpline(*tck)

    def test_derivative(self):
        bspline = self.generate_spline()

        spl = NewSpline1D()
        spl.bspline = bspline
        assert (spl.t == bspline.t).all()
        assert (spl.c == bspline.c).all()
        assert spl.degree == bspline.k

        # 1st derivative
        d_bspline = bspline.derivative(nu=1)
        assert_allclose(d_bspline(self.xs),       bspline(self.xs, nu=1))
        assert_allclose(d_bspline(self.xs, nu=1), bspline(self.xs, nu=2))
        assert_allclose(d_bspline(self.xs, nu=2), bspline(self.xs, nu=3))
        assert_allclose(d_bspline(self.xs, nu=3), bspline(self.xs, nu=4))

        der = spl.derivative()
        assert (der.t == d_bspline.t).all()
        assert (der.c == d_bspline.c).all()
        assert der.degree == d_bspline.k == 2
        assert_allclose(der.evaluate(self.xs),       spl.evaluate(self.xs, nu=1))
        assert_allclose(der.evaluate(self.xs, nu=1), spl.evaluate(self.xs, nu=2))
        assert_allclose(der.evaluate(self.xs, nu=2), spl.evaluate(self.xs, nu=3))
        assert_allclose(der.evaluate(self.xs, nu=3), spl.evaluate(self.xs, nu=4))

        # 2nd derivative
        d_bspline = bspline.derivative(nu=2)
        assert_allclose(d_bspline(self.xs),       bspline(self.xs, nu=2))
        assert_allclose(d_bspline(self.xs, nu=1), bspline(self.xs, nu=3))
        assert_allclose(d_bspline(self.xs, nu=2), bspline(self.xs, nu=4))

        der = spl.derivative(nu=2)
        assert (der.t == d_bspline.t).all()
        assert (der.c == d_bspline.c).all()
        assert der.degree == d_bspline.k == 1
        assert_allclose(der.evaluate(self.xs),       spl.evaluate(self.xs, nu=2))
        assert_allclose(der.evaluate(self.xs, nu=1), spl.evaluate(self.xs, nu=3))
        assert_allclose(der.evaluate(self.xs, nu=2), spl.evaluate(self.xs, nu=4))

        # 3rd derivative
        d_bspline = bspline.derivative(nu=3)
        assert_allclose(d_bspline(self.xs),       bspline(self.xs, nu=3))
        assert_allclose(d_bspline(self.xs, nu=1), bspline(self.xs, nu=4))

        der = spl.derivative(nu=3)
        assert (der.t == d_bspline.t).all()
        assert (der.c == d_bspline.c).all()
        assert der.degree == d_bspline.k == 0
        assert_allclose(der.evaluate(self.xs),       spl.evaluate(self.xs, nu=3))
        assert_allclose(der.evaluate(self.xs, nu=1), spl.evaluate(self.xs, nu=4))

        # Too many derivatives
        for nu in range(4, 9):
            with pytest.raises(ValueError) as err:
                spl.derivative(nu=nu)
            assert str(err.value) == \
                "Must have nu <= 3"

    def test_antiderivative(self):
        bspline = self.generate_spline()

        spl = NewSpline1D()
        spl.bspline = bspline

        # 1st antiderivative
        a_bspline = bspline.antiderivative(nu=1)
        assert_allclose(bspline(self.xs),       a_bspline(self.xs, nu=1))
        assert_allclose(bspline(self.xs, nu=1), a_bspline(self.xs, nu=2))
        assert_allclose(bspline(self.xs, nu=2), a_bspline(self.xs, nu=3))
        assert_allclose(bspline(self.xs, nu=3), a_bspline(self.xs, nu=4))
        assert_allclose(bspline(self.xs, nu=4), a_bspline(self.xs, nu=5))

        anti = spl.antiderivative()
        assert (anti.t == a_bspline.t).all()
        assert (anti.c == a_bspline.c).all()
        assert anti.degree == a_bspline.k == 4
        assert_allclose(spl.evaluate(self.xs),       anti.evaluate(self.xs, nu=1))
        assert_allclose(spl.evaluate(self.xs, nu=1), anti.evaluate(self.xs, nu=2))
        assert_allclose(spl.evaluate(self.xs, nu=2), anti.evaluate(self.xs, nu=3))
        assert_allclose(spl.evaluate(self.xs, nu=3), anti.evaluate(self.xs, nu=4))
        assert_allclose(spl.evaluate(self.xs, nu=4), anti.evaluate(self.xs, nu=5))

        # 2nd antiderivative
        a_bspline = bspline.antiderivative(nu=2)
        assert_allclose(bspline(self.xs),       a_bspline(self.xs, nu=2))
        assert_allclose(bspline(self.xs, nu=1), a_bspline(self.xs, nu=3))
        assert_allclose(bspline(self.xs, nu=2), a_bspline(self.xs, nu=4))
        assert_allclose(bspline(self.xs, nu=3), a_bspline(self.xs, nu=5))
        assert_allclose(bspline(self.xs, nu=4), a_bspline(self.xs, nu=6))

        anti = spl.antiderivative(nu=2)
        assert (anti.t == a_bspline.t).all()
        assert (anti.c == a_bspline.c).all()
        assert anti.degree == a_bspline.k == 5
        assert_allclose(spl.evaluate(self.xs),       anti.evaluate(self.xs, nu=2))
        assert_allclose(spl.evaluate(self.xs, nu=1), anti.evaluate(self.xs, nu=3))
        assert_allclose(spl.evaluate(self.xs, nu=2), anti.evaluate(self.xs, nu=4))
        assert_allclose(spl.evaluate(self.xs, nu=3), anti.evaluate(self.xs, nu=5))
        assert_allclose(spl.evaluate(self.xs, nu=4), anti.evaluate(self.xs, nu=6))

        # Too many anti derivatives
        for nu in range(3, 9):
            with pytest.raises(ValueError) as err:
                spl.antiderivative(nu=nu)
            assert str(err.value) == \
                f"Supported splines can have max degree 5, antiderivative degree will be {nu + 3}"
