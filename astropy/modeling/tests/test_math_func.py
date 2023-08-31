# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
import pytest
from numpy.testing import assert_allclose

from astropy.modeling.models import math

x = np.linspace(-20, 360, 100)


@pytest.mark.filterwarnings(r"ignore:.*:RuntimeWarning")
@pytest.mark.parametrize("name", math.__all__)
def test_math(name):
    model_class = getattr(math, name)
    assert model_class.__module__ == "astropy.modeling.models._math_functions"
    model = model_class()
    func = getattr(np, model.func.__name__)
    if model.n_inputs == 1:
        assert_allclose(model(x), func(x))
    elif model.n_inputs == 2:
        assert_allclose(model(x, x), func(x, x))

    assert math.ModUfunc is math.RemainderUfunc
    assert math.DivideUfunc is math.True_divideUfunc
