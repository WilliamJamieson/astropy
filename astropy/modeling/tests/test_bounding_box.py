# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
import unittest.mock as mk

from astropy.modeling.bounding_box import (_BaseInterval, Interval, _ignored_interval,
                                           BoundingDomain, BoundingBox,
                                           _BaseSliceArgument, SliceArgument, SliceArguments,
                                           CompoundBoundingBox)
from astropy.modeling.models import Gaussian1D, Gaussian2D
from astropy.modeling.core import Model
import astropy.units as u


class TestInterval:
    def test_create(self):
        lower = mk.MagicMock()
        upper = mk.MagicMock()
        interval = Interval(lower, upper)
        assert isinstance(interval, _BaseInterval)
        assert interval.lower == lower
        assert interval.upper == upper
        assert interval == (lower, upper)

        assert interval.__repr__() == \
            f"Interval(lower={lower}, upper={upper})"

    def test__validate_shape(self):
        message = "An interval must be some sort of sequence of length 2"
        lower = mk.MagicMock()
        upper = mk.MagicMock()
        interval = Interval(lower, upper)

        # Passes (2,)
        interval._validate_shape((1, 2))
        interval._validate_shape([1, 2])
        interval._validate_shape((1*u.m, 2*u.m))
        interval._validate_shape([1*u.m, 2*u.m])

        # Passes (1, 2)
        interval._validate_shape(((1, 2),))
        interval._validate_shape(([1, 2],))
        interval._validate_shape([(1, 2)])
        interval._validate_shape([[1, 2]])
        interval._validate_shape(((1*u.m, 2*u.m),))
        interval._validate_shape(([1*u.m, 2*u.m],))
        interval._validate_shape([(1*u.m, 2*u.m)])
        interval._validate_shape([[1*u.m, 2*u.m]])

        # Passes (2, 0)
        interval._validate_shape((mk.MagicMock(), mk.MagicMock()))
        interval._validate_shape([mk.MagicMock(), mk.MagicMock()])

        # Passes with array inputs:
        interval._validate_shape((np.array([-2.5, -3.5]), np.array([2.5, 3.5])))
        interval._validate_shape((np.array([-2.5, -3.5, -4.5]),
                                  np.array([2.5, 3.5, 4.5])))

        # Fails shape (no units)
        with pytest.raises(ValueError) as err:
            interval._validate_shape((1, 2, 3))
        assert str(err.value) == message
        with pytest.raises(ValueError) as err:
            interval._validate_shape([1, 2, 3])
        assert str(err.value) == message
        with pytest.raises(ValueError) as err:
            interval._validate_shape([[1, 2, 3], [4, 5, 6]])
        assert str(err.value) == message
        with pytest.raises(ValueError) as err:
            interval._validate_shape(1)
        assert str(err.value) == message

        # Fails shape (units)
        message = "An interval must be some sort of sequence of length 2"
        with pytest.raises(ValueError) as err:
            interval._validate_shape((1*u.m, 2*u.m, 3*u.m))
        assert str(err.value) == message
        with pytest.raises(ValueError) as err:
            interval._validate_shape([1*u.m, 2*u.m, 3*u.m])
        assert str(err.value) == message
        with pytest.raises(ValueError) as err:
            interval._validate_shape([[1*u.m, 2*u.m, 3*u.m], [4*u.m, 5*u.m, 6*u.m]])
        assert str(err.value) == message
        with pytest.raises(ValueError) as err:
            interval._validate_shape(1*u.m)
        assert str(err.value) == message

        # Fails shape (arrays):
        with pytest.raises(ValueError) as err:
            interval._validate_shape((np.array([-2.5, -3.5]),
                                      np.array([2.5, 3.5]),
                                      np.array([3, 4])))
        assert str(err.value) == message
        with pytest.raises(ValueError) as err:
            interval._validate_shape((np.array([-2.5, -3.5]), [2.5, 3.5]))
        assert str(err.value) == message

    def test__validate_bounds(self):
        # Passes
        assert Interval._validate_bounds(1, 2) == (1, 2)
        assert Interval._validate_bounds(1*u.m, 2*u.m) == (1*u.m, 2*u.m)

        interval = Interval._validate_bounds(np.array([-2.5, -3.5]), np.array([2.5, 3.5]))
        assert (interval.lower == np.array([-2.5, -3.5])).all()
        assert (interval.upper == np.array([2.5, 3.5])).all()

        # Fails
        with pytest.warns(RuntimeWarning,
                          match="Invalid interval: upper bound 1 is strictly "
                          r"less than lower bound 2\."):
            Interval._validate_bounds(2, 1)
        with pytest.warns(RuntimeWarning,
                          match=r"Invalid interval: upper bound 1\.0 m is strictly "
                          r"less than lower bound 2\.0 m\."):
            Interval._validate_bounds(2*u.m, 1*u.m)

    def test_validate(self):
        # Passes
        assert Interval.validate((1, 2)) == (1, 2)
        assert Interval.validate([1, 2]) == (1, 2)
        assert Interval.validate((1*u.m, 2*u.m)) == (1*u.m, 2*u.m)
        assert Interval.validate([1*u.m, 2*u.m]) == (1*u.m, 2*u.m)

        assert Interval.validate(((1, 2),)) == (1, 2)
        assert Interval.validate(([1, 2],)) == (1, 2)
        assert Interval.validate([(1, 2)]) == (1, 2)
        assert Interval.validate([[1, 2]]) == (1, 2)
        assert Interval.validate(((1*u.m, 2*u.m),)) == (1*u.m, 2*u.m)
        assert Interval.validate(([1*u.m, 2*u.m],)) == (1*u.m, 2*u.m)
        assert Interval.validate([(1*u.m, 2*u.m)]) == (1*u.m, 2*u.m)
        assert Interval.validate([[1*u.m, 2*u.m]]) == (1*u.m, 2*u.m)

        interval = Interval.validate((np.array([-2.5, -3.5]),
                                      np.array([2.5, 3.5])))
        assert (interval.lower == np.array([-2.5, -3.5])).all()
        assert (interval.upper == np.array([2.5, 3.5])).all()
        interval = Interval.validate((np.array([-2.5, -3.5, -4.5]),
                                     np.array([2.5, 3.5, 4.5])))
        assert (interval.lower == np.array([-2.5, -3.5, -4.5])).all()
        assert (interval.upper == np.array([2.5, 3.5, 4.5])).all()

        # Fail shape
        with pytest.raises(ValueError):
            Interval.validate((1, 2, 3))

        # Fail bounds
        with pytest.warns(RuntimeWarning):
            Interval.validate((2, 1))

    def test_outside(self):
        interval = Interval.validate((0, 1))

        assert (interval.outside(np.linspace(-1, 2, 13)) ==
                [True, True, True, True,
                 False, False, False, False, False,
                 True, True, True, True]).all()

    def test_domain(self):
        interval = Interval.validate((0, 1))
        assert (interval.domain(0.25) == np.linspace(0, 1, 5)).all()

    def test__ignored_interval(self):
        assert _ignored_interval.lower == -np.inf
        assert _ignored_interval.upper == np.inf

        for num in [0, -1, -100, 3.14, 10**100, -10**100]:
            assert not num < _ignored_interval[0]
            assert num > _ignored_interval[0]

            assert not num > _ignored_interval[1]
            assert num < _ignored_interval[1]

            assert not (_ignored_interval.outside(np.array([num]))).all()


class TestBoundingDomain:
    def test_create(self):
        model = mk.MagicMock()
        bounding_box = BoundingDomain(model)

        assert bounding_box._model == model

    def test__prepare_inputs(self):
        bounding_box = BoundingDomain(mk.MagicMock())

        with pytest.raises(NotImplementedError) as err:
            bounding_box.prepare_inputs(mk.MagicMock(), mk.MagicMock())
        assert str(err.value) == \
            "This has not been implemented for BoundingDomain."

    def test__base_ouput(self):
        bounding_box = BoundingDomain(mk.MagicMock())

        # Simple shape
        input_shape = (13,)
        output = bounding_box._base_output(input_shape, 0)
        assert (output == 0).all()
        assert output.shape == input_shape
        output = bounding_box._base_output(input_shape, np.nan)
        assert (np.isnan(output)).all()
        assert output.shape == input_shape
        output = bounding_box._base_output(input_shape, 14)
        assert (output == 14).all()
        assert output.shape == input_shape

        # Complex shape
        input_shape = (13, 7)
        output = bounding_box._base_output(input_shape, 0)
        assert (output == 0).all()
        assert output.shape == input_shape
        output = bounding_box._base_output(input_shape, np.nan)
        assert (np.isnan(output)).all()
        assert output.shape == input_shape
        output = bounding_box._base_output(input_shape, 14)
        assert (output == 14).all()
        assert output.shape == input_shape

    def test__all_out_output(self):
        model = mk.MagicMock()
        bounding_box = BoundingDomain(model)

        # Simple shape
        model.n_outputs = 1
        input_shape = (13,)
        output, output_unit = bounding_box._all_out_output(input_shape, 0)
        assert (np.array(output) == 0).all()
        assert np.array(output).shape == (1, 13)
        assert output_unit is None

        # Complex shape
        model.n_outputs = 6
        input_shape = (13, 7)
        output, output_unit = bounding_box._all_out_output(input_shape, 0)
        assert (np.array(output) == 0).all()
        assert np.array(output).shape == (6, 13, 7)
        assert output_unit is None

    def test__modify_output(self):
        bounding_box = BoundingDomain(mk.MagicMock())
        valid_index = mk.MagicMock()
        input_shape = mk.MagicMock()
        fill_value = mk.MagicMock()

        # Simple shape
        with mk.patch.object(BoundingDomain, '_base_output', autospec=True,
                             return_value=np.asanyarray(0)) as mkBase:
            assert (np.array([1, 2, 3]) ==
                    bounding_box._modify_output([1, 2, 3], valid_index, input_shape, fill_value)).all()
            assert mkBase.call_args_list == [mk.call(input_shape, fill_value)]

        # Replacement
        with mk.patch.object(BoundingDomain, '_base_output', autospec=True,
                             return_value=np.array([1, 2, 3, 4, 5, 6])) as mkBase:
            assert (np.array([7, 2, 8, 4, 9, 6]) ==
                    bounding_box._modify_output([7, 8, 9], np.array([[0, 2, 4]]), input_shape, fill_value)).all()
            assert mkBase.call_args_list == [mk.call(input_shape, fill_value)]

    def test__prepare_outputs(self):
        bounding_box = BoundingDomain(mk.MagicMock())
        valid_index = mk.MagicMock()
        input_shape = mk.MagicMock()
        fill_value = mk.MagicMock()

        valid_outputs = [mk.MagicMock() for _ in range(3)]
        effects = [mk.MagicMock() for _ in range(3)]
        with mk.patch.object(BoundingDomain, '_modify_output', autospec=True,
                             side_effect=effects) as mkModify:
            assert effects == bounding_box._prepare_outputs(valid_outputs, valid_index,
                                                            input_shape, fill_value)
            assert mkModify.call_args_list == \
                [mk.call(bounding_box, valid_outputs[idx], valid_index, input_shape, fill_value)
                 for idx in range(3)]

    def test_prepare_outputs(self):
        model = mk.MagicMock()
        bounding_box = BoundingDomain(model)

        valid_outputs = mk.MagicMock()
        valid_index = mk.MagicMock()
        input_shape = mk.MagicMock()
        fill_value = mk.MagicMock()

        with mk.patch.object(BoundingDomain, '_prepare_outputs', autospec=True) as mkPrepare:
            # Reshape valid_outputs
            model.n_outputs = 1
            assert mkPrepare.return_value == \
                bounding_box.prepare_outputs(valid_outputs, valid_index, input_shape, fill_value)
            assert mkPrepare.call_args_list == \
                [mk.call(bounding_box, [valid_outputs], valid_index, input_shape, fill_value)]
            mkPrepare.reset_mock()

            # No reshape valid_outputs
            model.n_outputs = 2
            assert mkPrepare.return_value == \
                bounding_box.prepare_outputs(valid_outputs, valid_index, input_shape, fill_value)
            assert mkPrepare.call_args_list == \
                [mk.call(bounding_box, valid_outputs, valid_index, input_shape, fill_value)]

    def test__get_valid_outputs_unit(self):
        bounding_box = BoundingDomain(mk.MagicMock())

        # Don't get unit
        assert bounding_box._get_valid_outputs_unit(mk.MagicMock(), False) is None

        # Get unit from unitless
        assert bounding_box._get_valid_outputs_unit(7, True) is None

        # Get unit
        assert bounding_box._get_valid_outputs_unit(25 * u.m, True) == u.m

    def test__evaluate_model(self):
        bounding_box = BoundingDomain(mk.MagicMock())

        evaluate = mk.MagicMock()
        valid_inputs = mk.MagicMock()
        input_shape = mk.MagicMock()
        valid_index = mk.MagicMock()
        fill_value = mk.MagicMock()
        with_units = mk.MagicMock()

        with mk.patch.object(BoundingDomain, '_get_valid_outputs_unit',
                             autospec=True) as mkGet:
            with mk.patch.object(BoundingDomain, 'prepare_outputs',
                                 autospec=True) as mkPrepare:
                assert bounding_box._evaluate_model(evaluate, valid_inputs,
                                                    valid_index, input_shape,
                                                    fill_value, with_units) == \
                    (mkPrepare.return_value, mkGet.return_value)
                assert mkPrepare.call_args_list == \
                    [mk.call(bounding_box, evaluate.return_value, valid_index,
                             input_shape, fill_value)]
                assert mkGet.call_args_list == \
                    [mk.call(evaluate.return_value, with_units)]
                assert evaluate.call_args_list == \
                    [mk.call(valid_inputs)]

    def test__evaluate(self):
        bounding_box = BoundingDomain(mk.MagicMock())

        evaluate = mk.MagicMock()
        inputs = mk.MagicMock()
        input_shape = mk.MagicMock()
        fill_value = mk.MagicMock()
        with_units = mk.MagicMock()

        valid_inputs = mk.MagicMock()
        valid_index = mk.MagicMock()

        effects = [(valid_inputs, valid_index, True), (valid_inputs, valid_index, False)]
        with mk.patch.object(BoundingDomain, 'prepare_inputs', autospec=True,
                             side_effect=effects) as mkPrepare:
            with mk.patch.object(BoundingDomain, '_all_out_output',
                                 autospec=True) as mkAll:
                with mk.patch.object(BoundingDomain, '_evaluate_model',
                                     autospec=True) as mkEvaluate:
                    # all_out
                    assert bounding_box._evaluate(evaluate, inputs, input_shape,
                                                  fill_value, with_units) == \
                        mkAll.return_value
                    assert mkAll.call_args_list == \
                        [mk.call(bounding_box, input_shape, fill_value)]
                    assert mkEvaluate.call_args_list == []
                    assert mkPrepare.call_args_list == \
                        [mk.call(bounding_box, input_shape, inputs)]

                    mkAll.reset_mock()
                    mkPrepare.reset_mock()

                    # not all_out
                    assert bounding_box._evaluate(evaluate, inputs, input_shape,
                                                  fill_value, with_units) == \
                        mkEvaluate.return_value
                    assert mkAll.call_args_list == []
                    assert mkEvaluate.call_args_list == \
                        [mk.call(bounding_box, evaluate, valid_inputs, valid_index,
                                 input_shape, fill_value, with_units)]
                    assert mkPrepare.call_args_list == \
                        [mk.call(bounding_box, input_shape, inputs)]

    def test__set_outputs_unit(self):
        bounding_box = BoundingDomain(mk.MagicMock())

        # set no unit
        assert 27 == bounding_box._set_outputs_unit(27, None)

        # set unit
        assert 27 * u.m == bounding_box._set_outputs_unit(27, u.m)

    def test_evaluate(self):
        bounding_box = BoundingDomain(Gaussian2D())

        evaluate = mk.MagicMock()
        inputs = mk.MagicMock()
        fill_value = mk.MagicMock()

        outputs = mk.MagicMock()
        valid_outputs_unit = mk.MagicMock()
        value = (outputs, valid_outputs_unit)
        with mk.patch.object(BoundingDomain, '_evaluate',
                             autospec=True, return_value=value) as mkEvaluate:
            with mk.patch.object(BoundingDomain, '_set_outputs_unit',
                                 autospec=True) as mkSet:
                with mk.patch.object(Model, 'input_shape', autospec=True) as mkShape:
                    with mk.patch.object(Model, 'bbox_with_units',
                                         new_callable=mk.PropertyMock) as mkUnits:
                        assert mkSet.return_value == \
                            bounding_box.evaluate(evaluate, inputs, fill_value)
                        assert mkSet.call_args_list == \
                            [mk.call(outputs, valid_outputs_unit)]
                        assert mkEvaluate.call_args_list == \
                            [mk.call(bounding_box, evaluate, inputs, mkShape.return_value,
                                     fill_value, mkUnits.return_value)]
                        assert mkShape.call_args_list == \
                            [mk.call(bounding_box._model, inputs)]
                        assert mkUnits.call_args_list == [mk.call()]


class TestBoundingBox:
    def test_create(self):
        intervals = ()
        model = mk.MagicMock()
        bounding_box = BoundingBox(intervals, model)

        assert isinstance(bounding_box, BoundingDomain)
        assert bounding_box._intervals == {}
        assert bounding_box._model == model
        assert bounding_box._ignored == []
        assert bounding_box._order == 'C'

        # Set optional
        intervals = {}
        model = mk.MagicMock()
        bounding_box = BoundingBox(intervals, model, order='test')

        assert isinstance(bounding_box, BoundingDomain)
        assert bounding_box._intervals == {}
        assert bounding_box._model == model
        assert bounding_box._ignored == []
        assert bounding_box._order == 'test'

        # Set interval
        intervals = (1, 2)
        model = mk.MagicMock()
        model.n_inputs = 1
        model.inputs = ['x']
        bounding_box = BoundingBox(intervals, model)

        assert isinstance(bounding_box, BoundingDomain)
        assert bounding_box._intervals == {0: (1, 2)}
        assert bounding_box._model == model

        # Set ignored
        intervals = (1, 2)
        model = mk.MagicMock()
        model.n_inputs = 2
        model.inputs = ['x', 'y']
        bounding_box = BoundingBox(intervals, model, ignored=[1])

        assert isinstance(bounding_box, BoundingDomain)
        assert bounding_box._intervals == {0: (1, 2)}
        assert bounding_box._model == model
        assert bounding_box._ignored == [1]

        intervals = ((1, 2), (3, 4))
        model = mk.MagicMock()
        model.n_inputs = 3
        model.inputs = ['x', 'y', 'z']
        bounding_box = BoundingBox(intervals, model, ignored=[2], order='F')

        assert isinstance(bounding_box, BoundingDomain)
        assert bounding_box._intervals == {0: (1, 2), 1: (3, 4)}
        assert bounding_box._model == model
        assert bounding_box._ignored == [2]
        assert bounding_box._order == 'F'

    def test_intervals(self):
        intervals = {0: Interval(1, 2)}
        model = mk.MagicMock()
        model.n_inputs = 1
        model.inputs = ['x']
        bounding_box = BoundingBox(intervals, model)

        assert bounding_box._intervals == intervals
        assert bounding_box.intervals == intervals

    def test_order(self):
        intervals = {}
        model = mk.MagicMock()
        bounding_box = BoundingBox(intervals, model, order='test')

        assert bounding_box._order == 'test'
        assert bounding_box.order == 'test'

    def test_ignored(self):
        ignored = [0]
        model = mk.MagicMock()
        model.n_inputs = 1
        model.inputs = ['x']
        bounding_box = BoundingBox({}, model, ignored=ignored)

        assert bounding_box._ignored == ignored
        assert bounding_box.ignored == ignored

    def test__get_name(self):
        intervals = {0: Interval(1, 2)}
        model = mk.MagicMock()
        model.n_inputs = 1
        model.inputs = ['x']
        bounding_box = BoundingBox(intervals, model)

        index = mk.MagicMock()
        name = mk.MagicMock()
        model.inputs = mk.MagicMock()
        model.inputs.__getitem__.return_value = name
        assert bounding_box._get_name(index) == name
        assert model.inputs.__getitem__.call_args_list == [mk.call(index)]

    def test_named_intervals(self):
        intervals = {idx: Interval(idx, idx + 1) for idx in range(4)}
        model = mk.MagicMock()
        model.n_inputs = 4
        model.inputs = [mk.MagicMock() for _ in range(4)]
        bounding_box = BoundingBox(intervals, model)

        named = bounding_box.named_intervals
        assert isinstance(named, dict)
        for name, interval in named.items():
            assert name in model.inputs
            assert intervals[model.inputs.index(name)] == interval
        for index, name in enumerate(model.inputs):
            assert index in intervals
            assert name in named
            assert intervals[index] == named[name]

    def test_ignored_inputs(self):
        intervals = {idx: Interval(idx, idx + 1) for idx in range(4)}
        model = mk.MagicMock()
        ignored = list(range(4, 8))
        model.n_inputs = 8
        model.inputs = [mk.MagicMock() for _ in range(8)]
        bounding_box = BoundingBox(intervals, model, ignored=ignored)

        inputs = bounding_box.ignored_inputs
        assert isinstance(inputs, list)
        for index, _input in enumerate(inputs):
            assert _input in model.inputs
            assert model.inputs[index + 4] == _input
        for index, _input in enumerate(model.inputs):
            if _input in inputs:
                assert inputs[index - 4] == _input
            else:
                assert index < 4

    def test___repr__(self):
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)

        assert bounding_box.__repr__() ==\
            "BoundingBox(\n" +\
            "    intervals={\n" +\
            "        x: Interval(lower=-1, upper=1)\n" +\
            "        y: Interval(lower=-4, upper=4)\n" +\
            "    }\n" +\
            "    model=Gaussian2D(inputs=('x', 'y'))\n" +\
            "    order='C'\n" +\
            ")"

        intervals = {0: Interval(-1, 1)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals, ignored=['y'])

        assert bounding_box.__repr__() ==\
            "BoundingBox(\n" +\
            "    intervals={\n" +\
            "        x: Interval(lower=-1, upper=1)\n" +\
            "    }\n" +\
            "    ignored=['y']\n" +\
            "    model=Gaussian2D(inputs=('x', 'y'))\n" +\
            "    order='C'\n" +\
            ")"

    def test___call__(self):
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)

        args = tuple([mk.MagicMock() for _ in range(3)])
        kwargs = {f"test{idx}": mk.MagicMock() for idx in range(3)}

        with pytest.raises(RuntimeError) as err:
            bounding_box(*args, **kwargs)
        assert str(err.value) ==\
            "This bounding box is fixed by the model and does not have " +\
            "adjustable parameters."

    def test__get_index(self):
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)

        # Pass input name
        assert bounding_box._get_index('x') == 0
        assert bounding_box._get_index('y') == 1

        # Pass invalid input name
        with pytest.raises(ValueError) as err:
            bounding_box._get_index('z')
        assert str(err.value) ==\
            "'z' is not one of the inputs: ('x', 'y')."

        # Pass valid index
        assert bounding_box._get_index(0) == 0
        assert bounding_box._get_index(1) == 1
        assert bounding_box._get_index(np.int32(0)) == 0
        assert bounding_box._get_index(np.int32(1)) == 1
        assert bounding_box._get_index(np.int64(0)) == 0
        assert bounding_box._get_index(np.int64(1)) == 1

        # Pass invalid index
        with pytest.raises(IndexError) as err:
            bounding_box._get_index(2)
        assert str(err.value) ==\
            "Integer key: 2 must be < 2."
        with pytest.raises(IndexError) as err:
            bounding_box._get_index(np.int32(2))
        assert str(err.value) ==\
            "Integer key: 2 must be < 2."
        with pytest.raises(IndexError) as err:
            bounding_box._get_index(np.int64(2))
        assert str(err.value) ==\
            "Integer key: 2 must be < 2."

        # Pass invalid key
        value = mk.MagicMock()
        with pytest.raises(ValueError) as err:
            bounding_box._get_index(value)
        assert str(err.value) ==\
            f"Key value: {value} must be string or integer."

    def test__validate_ignored(self):
        model = Gaussian2D()
        bounding_box = BoundingBox({}, model)

        # Pass
        assert bounding_box._validate_ignored(None) == []
        assert bounding_box._validate_ignored(['x', 'y']) == [0, 1]
        assert bounding_box._validate_ignored([0, 1]) == [0, 1]
        assert bounding_box._validate_ignored([np.int32(0), np.int64(1)]) == [0, 1]

        # Fail
        with pytest.raises(ValueError):
            bounding_box._validate_ignored([mk.MagicMock()])
        with pytest.raises(ValueError):
            bounding_box._validate_ignored(['z'])
        with pytest.raises(IndexError):
            bounding_box._validate_ignored([3])
        with pytest.raises(IndexError):
            bounding_box._validate_ignored([np.int32(3)])
        with pytest.raises(IndexError):
            bounding_box._validate_ignored([np.int64(3)])

    def test___len__(self):
        intervals = {0: Interval(-1, 1)}
        model = Gaussian1D()
        bounding_box = BoundingBox.validate(model, intervals)
        assert len(bounding_box) == 1 == len(bounding_box._intervals)

        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)
        assert len(bounding_box) == 2 == len(bounding_box._intervals)

        bounding_box._intervals = {}
        assert len(bounding_box) == 0 == len(bounding_box._intervals)

    def test___contains__(self):
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)

        # Contains with keys
        assert 'x' in bounding_box
        assert 'y' in bounding_box
        assert 'z' not in bounding_box

        # Contains with index
        assert 0 in bounding_box
        assert 1 in bounding_box
        assert 2 not in bounding_box

        # General not in
        assert mk.MagicMock() not in bounding_box

        # Contains with ignored
        del bounding_box['y']

        # Contains with keys
        assert 'x' in bounding_box
        assert 'y' in bounding_box
        assert 'z' not in bounding_box

        # Contains with index
        assert 0 in bounding_box
        assert 1 in bounding_box
        assert 2 not in bounding_box

    def test___getitem__(self):
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)

        # Get using input key
        assert bounding_box['x'] == (-1, 1)
        assert bounding_box['y'] == (-4, 4)

        # Fail with input key
        with pytest.raises(ValueError):
            bounding_box['z']

        # Get using index
        assert bounding_box[0] == (-1, 1)
        assert bounding_box[1] == (-4, 4)
        assert bounding_box[np.int32(0)] == (-1, 1)
        assert bounding_box[np.int32(1)] == (-4, 4)
        assert bounding_box[np.int64(0)] == (-1, 1)
        assert bounding_box[np.int64(1)] == (-4, 4)

        # Fail with index
        with pytest.raises(IndexError):
            bounding_box[2]
        with pytest.raises(IndexError):
            bounding_box[np.int32(2)]
        with pytest.raises(IndexError):
            bounding_box[np.int64(2)]

        # get ignored interval
        del bounding_box[0]
        assert bounding_box[0] == _ignored_interval
        assert bounding_box[1] == (-4, 4)

        del bounding_box[1]
        assert bounding_box[0] == _ignored_interval
        assert bounding_box[1] == _ignored_interval

    def test__get_order(self):
        intervals = {0: Interval(-1, 1)}
        model = Gaussian1D()
        bounding_box = BoundingBox.validate(model, intervals)

        # Success (default 'C')
        assert bounding_box._order == 'C'
        assert bounding_box._get_order() == 'C'
        assert bounding_box._get_order('C') == 'C'
        assert bounding_box._get_order('F') == 'F'

        # Success (default 'F')
        bounding_box._order = 'F'
        assert bounding_box._order == 'F'
        assert bounding_box._get_order() == 'F'
        assert bounding_box._get_order('C') == 'C'
        assert bounding_box._get_order('F') == 'F'

        # Error
        order = mk.MagicMock()
        with pytest.raises(ValueError) as err:
            bounding_box._get_order(order)
        assert str(err.value) ==\
            "order must be either 'C' (C/python order) or " +\
            f"'F' (Fortran/mathematical order), got: {order}."

    def test_bounding_box(self):
        # 1D
        intervals = {0: Interval(-1, 1)}
        model = Gaussian1D()
        bounding_box = BoundingBox.validate(model, intervals)
        assert bounding_box.bounding_box() == (-1, 1)
        assert bounding_box.bounding_box(mk.MagicMock()) == (-1, 1)

        # > 1D
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)
        assert bounding_box.bounding_box() == ((-4, 4), (-1, 1))
        assert bounding_box.bounding_box('C') == ((-4, 4), (-1, 1))
        assert bounding_box.bounding_box('F') == ((-1, 1), (-4, 4))

    def test___eq__(self):
        intervals = {0: Interval(-1, 1)}
        model = Gaussian1D()
        bounding_box = BoundingBox.validate(model.copy(), intervals.copy())

        assert bounding_box == bounding_box
        assert bounding_box == BoundingBox.validate(model.copy(), intervals.copy())
        assert bounding_box == (-1, 1)

        assert not (bounding_box == mk.MagicMock())
        assert not (bounding_box == (-2, 2))
        assert not (bounding_box == BoundingBox.validate(model, {0: Interval(-2, 2)}))

        # Respect ordering
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        model = Gaussian2D()
        bounding_box_1 = BoundingBox.validate(model, intervals)
        bounding_box_2 = BoundingBox.validate(model, intervals, order='F')
        assert bounding_box_1._order == 'C'
        assert bounding_box_1 == ((-4, 4), (-1, 1))
        assert not (bounding_box_1 == ((-1, 1), (-4, 4)))

        assert bounding_box_2._order == 'F'
        assert not (bounding_box_2 == ((-4, 4), (-1, 1)))
        assert bounding_box_2 == ((-1, 1), (-4, 4))

        assert bounding_box_1 == bounding_box_2

        # Respect ignored
        model = Gaussian2D()
        bounding_box_1._ignored = [mk.MagicMock()]
        bounding_box_2._ignored = [mk.MagicMock()]
        assert bounding_box_1._ignored != bounding_box_2._ignored
        assert not (bounding_box_1 == bounding_box_2)

    def test__setitem__(self):
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, {}, ignored=[0, 1])
        assert bounding_box._ignored == [0, 1]

        # USING Intervals directly
        # Set interval using key
        assert 0 not in bounding_box.intervals
        assert 0 in bounding_box.ignored
        bounding_box['x'] = Interval(-1, 1)
        assert 0 in bounding_box.intervals
        assert 0 not in bounding_box.ignored
        assert isinstance(bounding_box['x'], Interval)
        assert bounding_box['x'] == (-1, 1)

        assert 1 not in bounding_box.intervals
        assert 1 in bounding_box.ignored
        bounding_box['y'] = Interval(-4, 4)
        assert 1 in bounding_box.intervals
        assert 1 not in bounding_box.ignored
        assert isinstance(bounding_box['y'], Interval)
        assert bounding_box['y'] == (-4, 4)

        del bounding_box['x']
        del bounding_box['y']

        # Set interval using index
        assert 0 not in bounding_box.intervals
        assert 0 in bounding_box.ignored
        bounding_box[0] = Interval(-1, 1)
        assert 0 in bounding_box.intervals
        assert 0 not in bounding_box.ignored
        assert isinstance(bounding_box[0], Interval)
        assert bounding_box[0] == (-1, 1)

        assert 1 not in bounding_box.intervals
        assert 1 in bounding_box.ignored
        bounding_box[1] = Interval(-4, 4)
        assert 1 in bounding_box.intervals
        assert 1 not in bounding_box.ignored
        assert isinstance(bounding_box[1], Interval)
        assert bounding_box[1] == (-4, 4)

        del bounding_box[0]
        del bounding_box[1]

        # USING tuples
        # Set interval using key
        assert 0 not in bounding_box.intervals
        assert 0 in bounding_box.ignored
        bounding_box['x'] = (-1, 1)
        assert 0 in bounding_box.intervals
        assert 0 not in bounding_box.ignored
        assert isinstance(bounding_box['x'], Interval)
        assert bounding_box['x'] == (-1, 1)

        assert 1 not in bounding_box.intervals
        assert 1 in bounding_box.ignored
        bounding_box['y'] = (-4, 4)
        assert 1 in bounding_box.intervals
        assert 1 not in bounding_box.ignored
        assert isinstance(bounding_box['y'], Interval)
        assert bounding_box['y'] == (-4, 4)

        del bounding_box['x']
        del bounding_box['y']

        # Set interval using index
        assert 0 not in bounding_box.intervals
        assert 0 in bounding_box.ignored
        bounding_box[0] = (-1, 1)
        assert 0 in bounding_box.intervals
        assert 0 not in bounding_box.ignored
        assert isinstance(bounding_box[0], Interval)
        assert bounding_box[0] == (-1, 1)

        assert 1 not in bounding_box.intervals
        assert 1 in bounding_box.ignored
        bounding_box[1] = (-4, 4)
        assert 1 in bounding_box.intervals
        assert 1 not in bounding_box.ignored
        assert isinstance(bounding_box[1], Interval)
        assert bounding_box[1] == (-4, 4)

        # Model set support
        model = Gaussian1D([0.1, 0.2], [0, 0], [5, 7], n_models=2)
        bounding_box = BoundingBox({}, model)
        # USING Intervals directly
        # Set interval using key
        assert 'x' not in bounding_box
        bounding_box['x'] = Interval(np.array([-1, -2]), np.array([1, 2]))
        assert 'x' in bounding_box
        assert isinstance(bounding_box['x'], Interval)
        assert (bounding_box['x'].lower == np.array([-1, -2])).all()
        assert (bounding_box['x'].upper == np.array([1, 2])).all()
        # Set interval using index
        bounding_box._intervals = {}
        assert 0 not in bounding_box
        bounding_box[0] = Interval(np.array([-1, -2]), np.array([1, 2]))
        assert 0 in bounding_box
        assert isinstance(bounding_box[0], Interval)
        assert (bounding_box[0].lower == np.array([-1, -2])).all()
        assert (bounding_box[0].upper == np.array([1, 2])).all()
        # USING tuples
        # Set interval using key
        bounding_box._intervals = {}
        assert 'x' not in bounding_box
        bounding_box['x'] = (np.array([-1, -2]), np.array([1, 2]))
        assert 'x' in bounding_box
        assert isinstance(bounding_box['x'], Interval)
        assert (bounding_box['x'].lower == np.array([-1, -2])).all()
        assert (bounding_box['x'].upper == np.array([1, 2])).all()
        # Set interval using index
        bounding_box._intervals = {}
        assert 0 not in bounding_box
        bounding_box[0] = (np.array([-1, -2]), np.array([1, 2]))
        assert 0 in bounding_box
        assert isinstance(bounding_box[0], Interval)
        assert (bounding_box[0].lower == np.array([-1, -2])).all()
        assert (bounding_box[0].upper == np.array([1, 2])).all()

    def test___delitem__(self):
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)

        # Using index
        assert 0 in bounding_box.intervals
        assert 0 not in bounding_box.ignored
        assert 0 in bounding_box
        assert 'x' in bounding_box
        del bounding_box[0]
        assert 0 not in bounding_box.intervals
        assert 0 in bounding_box.ignored
        assert 0 in bounding_box
        assert 'x' in bounding_box

        # Using key
        assert 1 in bounding_box.intervals
        assert 1 not in bounding_box.ignored
        assert 0 in bounding_box
        assert 'y' in bounding_box
        del bounding_box['y']
        assert 1 not in bounding_box.intervals
        assert 1 in bounding_box.ignored
        assert 0 in bounding_box
        assert 'y' in bounding_box

    def test__validate_dict(self):
        model = Gaussian2D()
        bounding_box = BoundingBox({}, model)

        # Input name keys
        intervals = {'x': Interval(-1, 1), 'y': Interval(-4, 4)}
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        bounding_box._validate_dict(intervals)
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-1, 1)
        assert 'y' in bounding_box
        assert bounding_box['y'] == (-4, 4)
        assert len(bounding_box.intervals) == 2

        # Input index
        bounding_box._intervals = {}
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        assert 0 not in bounding_box
        assert 1 not in bounding_box
        bounding_box._validate_dict(intervals)
        assert 0 in bounding_box
        assert bounding_box[0] == (-1, 1)
        assert 1 in bounding_box
        assert bounding_box[1] == (-4, 4)
        assert len(bounding_box.intervals) == 2

        # Model set support
        model = Gaussian1D([0.1, 0.2], [0, 0], [5, 7], n_models=2)
        bounding_box = BoundingBox({}, model)
        # name keys
        intervals = {'x': Interval(np.array([-1, -2]), np.array([1, 2]))}
        assert 'x' not in bounding_box
        bounding_box._validate_dict(intervals)
        assert 'x' in bounding_box
        assert (bounding_box['x'].lower == np.array([-1, -2])).all()
        assert (bounding_box['x'].upper == np.array([1, 2])).all()
        # input index
        bounding_box._intervals = {}
        intervals = {0: Interval(np.array([-1, -2]), np.array([1, 2]))}
        assert 0 not in bounding_box
        bounding_box._validate_dict(intervals)
        assert 0 in bounding_box
        assert (bounding_box[0].lower == np.array([-1, -2])).all()
        assert (bounding_box[0].upper == np.array([1, 2])).all()

    def test__validate_sequence(self):
        model = Gaussian2D()
        bounding_box = BoundingBox({}, model)

        # Default order
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        bounding_box._validate_sequence(((-4, 4), (-1, 1)))
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-1, 1)
        assert 'y' in bounding_box
        assert bounding_box['y'] == (-4, 4)
        assert len(bounding_box.intervals) == 2

        # C order
        bounding_box._intervals = {}
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        bounding_box._validate_sequence(((-4, 4), (-1, 1)), order='C')
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-1, 1)
        assert 'y' in bounding_box
        assert bounding_box['y'] == (-4, 4)
        assert len(bounding_box.intervals) == 2

        # Fortran order
        bounding_box._intervals = {}
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        bounding_box._validate_sequence(((-4, 4), (-1, 1)), order='F')
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-4, 4)
        assert 'y' in bounding_box
        assert bounding_box['y'] == (-1, 1)
        assert len(bounding_box.intervals) == 2

        # Invalid order
        bounding_box._intervals = {}
        order = mk.MagicMock()
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        with pytest.raises(ValueError):
            bounding_box._validate_sequence(((-4, 4), (-1, 1)), order=order)
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        assert len(bounding_box.intervals) == 0

    def test__n_inputs(self):
        model = Gaussian2D()

        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        bounding_box = BoundingBox.validate(model, intervals)
        assert bounding_box._n_inputs == 2

        intervals = {0: Interval(-1, 1)}
        bounding_box = BoundingBox.validate(model, intervals, ignored=['y'])
        assert bounding_box._n_inputs == 1

    def test__validate_iterable(self):
        model = Gaussian2D()
        bounding_box = BoundingBox({}, model)

        # Pass sequence Default order
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        bounding_box._validate_iterable(((-4, 4), (-1, 1)))
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-1, 1)
        assert 'y' in bounding_box
        assert bounding_box['y'] == (-4, 4)
        assert len(bounding_box.intervals) == 2

        # Pass sequence
        bounding_box._intervals = {}
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        bounding_box._validate_iterable(((-4, 4), (-1, 1)), order='F')
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-4, 4)
        assert 'y' in bounding_box
        assert bounding_box['y'] == (-1, 1)
        assert len(bounding_box.intervals) == 2

        # Pass Dict
        bounding_box._intervals = {}
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        assert 0 not in bounding_box
        assert 1 not in bounding_box
        bounding_box._validate_iterable(intervals)
        assert 0 in bounding_box
        assert bounding_box[0] == (-1, 1)
        assert 1 in bounding_box
        assert bounding_box[1] == (-4, 4)
        assert len(bounding_box.intervals) == 2

        # Pass with ignored
        bounding_box._intervals = {}
        bounding_box._ignored = [1]
        intervals = {0: Interval(-1, 1)}
        assert 0 not in bounding_box.intervals
        bounding_box._validate_iterable(intervals)
        assert 0 in bounding_box.intervals
        assert bounding_box[0] == (-1, 1)

        # Invalid iterable
        bounding_box._intervals = {}
        bounding_box._ignored = []
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        with pytest.raises(ValueError) as err:
            bounding_box._validate_iterable(((-4, 4), (-1, 1), (-3, 3)))
        assert str(err.value) ==\
            "Found 3 intervals, but must have exactly 2."
        assert len(bounding_box.intervals) == 0
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        bounding_box._ignored = [1]
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        with pytest.raises(ValueError) as err:
            bounding_box._validate_iterable(intervals)
        assert str(err.value) ==\
            "Found 2 intervals, but must have exactly 1."
        assert len(bounding_box.intervals) == 0
        bounding_box._ignored = []
        intervals = {0: Interval(-1, 1)}
        with pytest.raises(ValueError) as err:
            bounding_box._validate_iterable(intervals)
        assert str(err.value) ==\
            "Found 1 intervals, but must have exactly 2."
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        assert len(bounding_box.intervals) == 0

    def test__validate(self):
        model = Gaussian2D()
        bounding_box = BoundingBox({}, model)

        # Pass sequence Default order
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        bounding_box._validate(((-4, 4), (-1, 1)))
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-1, 1)
        assert 'y' in bounding_box
        assert bounding_box['y'] == (-4, 4)
        assert len(bounding_box.intervals) == 2

        # Pass sequence
        bounding_box._intervals = {}
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        bounding_box._validate(((-4, 4), (-1, 1)), order='F')
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-4, 4)
        assert 'y' in bounding_box
        assert bounding_box['y'] == (-1, 1)
        assert len(bounding_box.intervals) == 2

        # Pass Dict
        bounding_box._intervals = {}
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        assert 'x' not in bounding_box
        assert 'y' not in bounding_box
        bounding_box._validate(intervals)
        assert 0 in bounding_box
        assert bounding_box[0] == (-1, 1)
        assert 1 in bounding_box
        assert bounding_box[1] == (-4, 4)
        assert len(bounding_box.intervals) == 2

        # Pass single with ignored
        intervals = {0: Interval(-1, 1)}
        bounding_box = BoundingBox({}, model, ignored=[1])

        assert 0 not in bounding_box.intervals
        assert 1 not in bounding_box.intervals
        bounding_box._validate(intervals)
        assert 0 in bounding_box.intervals
        assert bounding_box[0] == (-1, 1)
        assert 1 not in bounding_box.intervals
        assert len(bounding_box.intervals) == 1

        # Pass single
        model = Gaussian1D()
        bounding_box = BoundingBox({}, model)

        assert 'x' not in bounding_box
        bounding_box._validate((-1, 1))
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-1, 1)
        assert len(bounding_box.intervals) == 1

        # Model set support
        model = Gaussian1D([0.1, 0.2], [0, 0], [5, 7], n_models=2)
        bounding_box = BoundingBox({}, model)
        sequence = (np.array([-1, -2]), np.array([1, 2]))
        assert 'x' not in bounding_box
        bounding_box._validate(sequence)
        assert 'x' in bounding_box
        assert (bounding_box['x'].lower == np.array([-1, -2])).all()
        assert (bounding_box['x'].upper == np.array([1, 2])).all()

    def test_validate(self):
        model = Gaussian2D()

        # Pass sequence Default order
        bounding_box = BoundingBox.validate(model, ((-4, 4), (-1, 1)))
        assert (bounding_box._model.parameters == model.parameters).all()
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-1, 1)
        assert 'y' in bounding_box
        assert bounding_box['y'] == (-4, 4)
        assert len(bounding_box.intervals) == 2

        # Pass sequence
        bounding_box = BoundingBox.validate(model, ((-4, 4), (-1, 1)), order='F')
        assert (bounding_box._model.parameters == model.parameters).all()
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-4, 4)
        assert 'y' in bounding_box
        assert bounding_box['y'] == (-1, 1)
        assert len(bounding_box.intervals) == 2

        # Pass Dict
        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        bounding_box = BoundingBox.validate(model, intervals, order='F')
        assert (bounding_box._model.parameters == model.parameters).all()
        assert 0 in bounding_box
        assert bounding_box[0] == (-1, 1)
        assert 1 in bounding_box
        assert bounding_box[1] == (-4, 4)
        assert len(bounding_box.intervals) == 2
        assert bounding_box.order == 'F'

        # Pass BoundingBox
        bbox = bounding_box
        bounding_box = BoundingBox.validate(model, bbox)
        assert (bounding_box._model.parameters == model.parameters).all()
        assert 0 in bounding_box
        assert bounding_box[0] == (-1, 1)
        assert 1 in bounding_box
        assert bounding_box[1] == (-4, 4)
        assert len(bounding_box.intervals) == 2
        assert bounding_box.order == 'F'

        # Pass single ignored
        intervals = {0: Interval(-1, 1)}
        bounding_box = BoundingBox.validate(model, intervals, ignored=['y'])
        assert (bounding_box._model.parameters == model.parameters).all()
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-1, 1)
        assert 'y' in bounding_box
        assert bounding_box['y'] == _ignored_interval
        assert len(bounding_box.intervals) == 1

        # Pass single
        bounding_box = BoundingBox.validate(Gaussian1D(), (-1, 1))
        assert (bounding_box._model.parameters == Gaussian1D().parameters).all()
        assert 'x' in bounding_box
        assert bounding_box['x'] == (-1, 1)
        assert len(bounding_box.intervals) == 1

        # Model set support
        model = Gaussian1D([0.1, 0.2], [0, 0], [5, 7], n_models=2)
        sequence = (np.array([-1, -2]), np.array([1, 2]))
        bounding_box = BoundingBox.validate(model, sequence)
        assert 'x' in bounding_box
        assert (bounding_box['x'].lower == np.array([-1, -2])).all()
        assert (bounding_box['x'].upper == np.array([1, 2])).all()

    def test_copy(self):
        bounding_box = BoundingBox.validate(Gaussian2D(), ((-4, 4), (-1, 1)))

        new_bounding_box = bounding_box.copy()
        assert bounding_box == new_bounding_box
        assert id(bounding_box) != id(new_bounding_box)

        assert bounding_box.intervals == new_bounding_box.intervals
        assert id(bounding_box.intervals) != id(new_bounding_box.intervals)

        assert bounding_box.ignored == new_bounding_box.ignored
        assert id(bounding_box.ignored) != id(new_bounding_box.ignored)

        assert bounding_box._model == new_bounding_box._model
        assert id(bounding_box._model) == id(new_bounding_box._model)

        assert bounding_box._order == new_bounding_box._order
        assert id(bounding_box._order) == id(new_bounding_box._order)

    def test_fix_inputs(self):
        bounding_box = BoundingBox.validate(Gaussian2D(), ((-4, 4), (-1, 1)))

        new_bounding_box = bounding_box.fix_inputs(Gaussian1D(), [1])
        assert not (bounding_box == new_bounding_box)

        assert (new_bounding_box._model.parameters == Gaussian1D().parameters).all()
        assert 'x' in new_bounding_box
        assert new_bounding_box['x'] == (-1, 1)
        assert 'y' not in new_bounding_box
        assert len(new_bounding_box.intervals) == 1

    def test_dimension(self):
        intervals = {0: Interval(-1, 1)}
        model = Gaussian1D()
        bounding_box = BoundingBox.validate(model, intervals)
        assert bounding_box.dimension == 1 == len(bounding_box._intervals)

        intervals = {0: Interval(-1, 1), 1: Interval(-4, 4)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)
        assert bounding_box.dimension == 2 == len(bounding_box._intervals)

        bounding_box._intervals = {}
        assert bounding_box.dimension == 0 == len(bounding_box._intervals)

    def test_domain(self):
        intervals = {0: Interval(-1, 1), 1: Interval(0, 2)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)

        # test defaults
        assert (np.array(bounding_box.domain(0.25)) ==
                np.array([np.linspace(0, 2, 9), np.linspace(-1, 1, 9)])).all()

        # test C order
        assert (np.array(bounding_box.domain(0.25, 'C')) ==
                np.array([np.linspace(0, 2, 9), np.linspace(-1, 1, 9)])).all()

        # test Fortran order
        assert (np.array(bounding_box.domain(0.25, 'F')) ==
                np.array([np.linspace(-1, 1, 9), np.linspace(0, 2, 9)])).all()

        # test error order
        order = mk.MagicMock()
        with pytest.raises(ValueError):
            bounding_box.domain(0.25, order)

    def test__outside(self):
        intervals = {0: Interval(-1, 1), 1: Interval(0, 2)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)

        # Normal array input, all inside
        x = np.linspace(-1, 1, 13)
        y = np.linspace(0, 2, 13)
        input_shape = x.shape
        inputs = (x, y)
        outside_index, all_out = bounding_box._outside(input_shape, inputs)
        assert (outside_index == [False for _ in range(13)]).all()
        assert not all_out and isinstance(all_out, bool)

        # Normal array input, some inside and some outside
        x = np.linspace(-2, 1, 13)
        y = np.linspace(0, 3, 13)
        input_shape = x.shape
        inputs = (x, y)
        outside_index, all_out = bounding_box._outside(input_shape, inputs)
        assert (outside_index ==
                [True, True, True, True,
                 False, False, False, False, False,
                 True, True, True, True]).all()
        assert not all_out and isinstance(all_out, bool)

        # Normal array input, all outside
        x = np.linspace(2, 3, 13)
        y = np.linspace(-2, -1, 13)
        input_shape = x.shape
        inputs = (x, y)
        outside_index, all_out = bounding_box._outside(input_shape, inputs)
        assert (outside_index == [True for _ in range(13)]).all()
        assert not all_out and isinstance(all_out, bool)

        # Scalar input inside bounding_box
        inputs = (0.5, 0.5)
        input_shape = (1,)
        outside_index, all_out = bounding_box._outside(input_shape, inputs)
        assert (outside_index == [False]).all()
        assert not all_out and isinstance(all_out, bool)

        # Scalar input outside bounding_box
        inputs = (2, -1)
        input_shape = (1,)
        outside_index, all_out = bounding_box._outside(input_shape, inputs)
        assert (outside_index == [True]).all()
        assert all_out and isinstance(all_out, bool)

    def test__valid_index(self):
        intervals = {0: Interval(-1, 1), 1: Interval(0, 2)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)

        # Normal array input, all inside
        x = np.linspace(-1, 1, 13)
        y = np.linspace(0, 2, 13)
        input_shape = x.shape
        inputs = (x, y)
        valid_index, all_out = bounding_box._valid_index(input_shape, inputs)
        assert len(valid_index) == 1
        assert (valid_index[0] == [idx for idx in range(13)]).all()
        assert not all_out and isinstance(all_out, bool)

        # Normal array input, some inside and some outside
        x = np.linspace(-2, 1, 13)
        y = np.linspace(0, 3, 13)
        input_shape = x.shape
        inputs = (x, y)
        valid_index, all_out = bounding_box._valid_index(input_shape, inputs)
        assert len(valid_index) == 1
        assert (valid_index[0] == [4, 5, 6, 7, 8]).all()
        assert not all_out and isinstance(all_out, bool)

        # Normal array input, all outside
        x = np.linspace(2, 3, 13)
        y = np.linspace(-2, -1, 13)
        input_shape = x.shape
        inputs = (x, y)
        valid_index, all_out = bounding_box._valid_index(input_shape, inputs)
        assert len(valid_index) == 1
        assert (valid_index[0] == []).all()
        assert all_out and isinstance(all_out, bool)

        # Scalar input inside bounding_box
        inputs = (0.5, 0.5)
        input_shape = (1,)
        valid_index, all_out = bounding_box._valid_index(input_shape, inputs)
        assert len(valid_index) == 1
        assert (valid_index[0] == [0]).all()
        assert not all_out and isinstance(all_out, bool)

        # Scalar input outside bounding_box
        inputs = (2, -1)
        input_shape = (1,)
        valid_index, all_out = bounding_box._valid_index(input_shape, inputs)
        assert len(valid_index) == 1
        assert (valid_index[0] == []).all()
        assert all_out and isinstance(all_out, bool)

    def test_prepare_inputs(self):
        intervals = {0: Interval(-1, 1), 1: Interval(0, 2)}
        model = Gaussian2D()
        bounding_box = BoundingBox.validate(model, intervals)

        # Normal array input, all inside
        x = np.linspace(-1, 1, 13)
        y = np.linspace(0, 2, 13)
        input_shape = x.shape
        inputs = (x, y)
        new_inputs, valid_index, all_out = bounding_box.prepare_inputs(input_shape, inputs)
        assert (np.array(new_inputs) == np.array(inputs)).all()
        assert len(valid_index) == 1
        assert (valid_index[0] == [idx for idx in range(13)]).all()
        assert not all_out and isinstance(all_out, bool)

        # Normal array input, some inside and some outside
        x = np.linspace(-2, 1, 13)
        y = np.linspace(0, 3, 13)
        input_shape = x.shape
        inputs = (x, y)
        new_inputs, valid_index, all_out = bounding_box.prepare_inputs(input_shape, inputs)
        assert (np.array(new_inputs) ==
                np.array(
                    [
                        [x[4], x[5], x[6], x[7], x[8]],
                        [y[4], y[5], y[6], y[7], y[8]],
                    ]
                )).all()
        assert len(valid_index) == 1
        assert (valid_index[0] == [4, 5, 6, 7, 8]).all()
        assert not all_out and isinstance(all_out, bool)

        # Normal array input, all outside
        x = np.linspace(2, 3, 13)
        y = np.linspace(-2, -1, 13)
        input_shape = x.shape
        inputs = (x, y)
        new_inputs, valid_index, all_out = bounding_box.prepare_inputs(input_shape, inputs)
        assert new_inputs == []
        assert len(valid_index) == 1
        assert (valid_index[0] == []).all()
        assert all_out and isinstance(all_out, bool)

        # Scalar input inside bounding_box
        inputs = (0.5, 0.5)
        input_shape = (1,)
        new_inputs, valid_index, all_out = bounding_box.prepare_inputs(input_shape, inputs)
        assert (np.array(new_inputs) == np.array([[0.5], [0.5]])).all()
        assert len(valid_index) == 1
        assert (valid_index[0] == [0]).all()
        assert not all_out and isinstance(all_out, bool)

        # Scalar input outside bounding_box
        inputs = (2, -1)
        input_shape = (1,)
        new_inputs, valid_index, all_out = bounding_box.prepare_inputs(input_shape, inputs)
        assert new_inputs == []
        assert len(valid_index) == 1
        assert (valid_index[0] == []).all()
        assert all_out and isinstance(all_out, bool)


class TestSliceArgument:
    def test_create(self):
        index = mk.MagicMock()
        ignore = mk.MagicMock()
        model = Gaussian2D()
        argument = SliceArgument(index, ignore, model)

        assert isinstance(argument, _BaseSliceArgument)
        assert argument.index == index
        assert argument.ignore == ignore
        assert argument == (index, ignore)
        assert (argument._model.parameters == Gaussian2D().parameters).all()

    def test_validate(self):
        model = Gaussian2D()

        # default integer
        assert SliceArgument.validate(model, 0) == (0, False)
        assert SliceArgument.validate(model, 1) == (1, False)

        # default string
        assert SliceArgument.validate(model, 'x') == (0, False)
        assert SliceArgument.validate(model, 'y') == (1, False)

        ignore = mk.MagicMock()
        # non-default integer
        assert SliceArgument.validate(model, 0, ignore) == (0, ignore)
        assert SliceArgument.validate(model, 1, ignore) == (1, ignore)

        # non-default string
        assert SliceArgument.validate(model, 'x', ignore) == (0, ignore)
        assert SliceArgument.validate(model, 'y', ignore) == (1, ignore)

        # Fail
        with pytest.raises(ValueError):
            SliceArgument.validate(model, 'z')
        with pytest.raises(ValueError):
            SliceArgument.validate(model, mk.MagicMock())
        with pytest.raises(IndexError):
            SliceArgument.validate(model, 2)

    def test_get_slice(self):
        # single inputs
        inputs = [idx + 17 for idx in range(3)]
        for index in range(3):
            assert SliceArgument(index, mk.MagicMock(), Gaussian2D()).get_slice(*inputs) == inputs[index]

        # numpy array of single inputs
        inputs = [np.array([idx + 11]) for idx in range(3)]
        for index in range(3):
            assert SliceArgument(index, mk.MagicMock(), Gaussian2D()).get_slice(*inputs) == inputs[index]
        inputs = [np.asanyarray(idx + 13) for idx in range(3)]
        for index in range(3):
            assert SliceArgument(index, mk.MagicMock(), Gaussian2D()).get_slice(*inputs) == inputs[index]

        # multi entry numpy array
        inputs = [np.array([idx + 27, idx - 31]) for idx in range(3)]
        for index in range(3):
            assert SliceArgument(index, mk.MagicMock(), Gaussian2D()).get_slice(*inputs) == tuple(inputs[index])

    def test___repr__(self):
        model = Gaussian2D()

        assert SliceArgument(0, False, Gaussian2D()).__repr__() ==\
            "Argument(name='x', ignore=False)"
        assert SliceArgument(0, True, Gaussian2D()).__repr__() ==\
            "Argument(name='x', ignore=True)"

        assert SliceArgument(1, False, Gaussian2D()).__repr__() ==\
            "Argument(name='y', ignore=False)"
        assert SliceArgument(1, True, Gaussian2D()).__repr__() ==\
            "Argument(name='y', ignore=True)"

    def test_get_fixed_value(self):
        model = Gaussian2D()
        values = {0: 5, 'y': 7}

        # Get index value
        assert SliceArgument(0, mk.MagicMock(), model).get_fixed_value(values) == 5

        # Get name value
        assert SliceArgument(1, mk.MagicMock(), model).get_fixed_value(values) == 7

        # Fail
        values = {0: 5}
        with pytest.raises(RuntimeError) as err:
            SliceArgument(1, True, model).get_fixed_value(values)
        assert str(err.value) == \
            "Argument(name='y', ignore=True) was not found in {0: 5}"


class TestSliceArguments:
    def test_create(self):
        model = Gaussian2D()
        arguments = SliceArguments((SliceArgument(0, True, model), SliceArgument(1, False, model)), model)

        assert isinstance(arguments, SliceArguments)
        assert (arguments._model.parameters == Gaussian2D().parameters).all()
        assert arguments == ((0, True), (1, False))

    def test___repr__(self):
        model = Gaussian2D()
        arguments = SliceArguments((SliceArgument(0, True, model), SliceArgument(1, False, model)), model)

        assert arguments.__repr__() ==\
            "SliceArguments(\n" +\
            "    Argument(name='x', ignore=True)\n" +\
            "    Argument(name='y', ignore=False)\n" +\
            ")"

    def test_ignore(self):
        model = Gaussian2D()
        assert SliceArguments((SliceArgument(0, True, model),
                               SliceArgument(1, True, model)), model).ignore == [0, 1]
        assert SliceArguments((SliceArgument(0, True, model),
                               SliceArgument(1, False, model)), model).ignore == [0]
        assert SliceArguments((SliceArgument(0, False, model),
                               SliceArgument(1, True, model)), model).ignore == [1]
        assert SliceArguments((SliceArgument(0, False, model),
                               SliceArgument(1, False, model)), model).ignore == []

    def test_validate(self):
        # Valid
        arguments = SliceArguments.validate(Gaussian2D(), ((0, True), (1, False)))
        assert isinstance(arguments, SliceArguments)
        assert (arguments._model.parameters == Gaussian2D().parameters).all()
        assert arguments == ((0, True), (1, False))

        arguments = SliceArguments.validate(Gaussian2D(), ((0,), (1,)))
        assert isinstance(arguments, SliceArguments)
        assert (arguments._model.parameters == Gaussian2D().parameters).all()
        assert arguments == ((0, False), (1, False))

        arguments = SliceArguments.validate(Gaussian2D(), (('x', True), ('y', False)))
        assert isinstance(arguments, SliceArguments)
        assert (arguments._model.parameters == Gaussian2D().parameters).all()
        assert arguments == ((0, True), (1, False))

        # Invalid, bad argument
        with pytest.raises(ValueError):
            SliceArguments.validate(Gaussian2D(), ((0, True), ('z', False)))
        with pytest.raises(ValueError):
            SliceArguments.validate(Gaussian2D(), ((mk.MagicMock(), True), (1, False)))
        with pytest.raises(IndexError):
            SliceArguments.validate(Gaussian2D(), ((0, True), (2, False)))

        # Invalid, repeated argument
        with pytest.raises(ValueError) as err:
            SliceArguments.validate(Gaussian2D(), ((0, True), (0, False)))
        assert str(err.value) == \
            "Input: 'x' has been repeated."

        # Invalid, no arguments
        with pytest.raises(ValueError) as err:
            SliceArguments.validate(Gaussian2D(), ())
        assert str(err.value) == \
            "There must be at least one slice argument."

    def test_get_slice(self):
        inputs = [idx + 19 for idx in range(4)]

        assert SliceArguments.validate(Gaussian2D(),
                                       ((0, True), (1, False))).get_slice(*inputs) ==\
            tuple(inputs[:2])
        assert SliceArguments.validate(Gaussian2D(),
                                       ((1, True), (0, False))).get_slice(*inputs) ==\
            tuple(inputs[:2][::-1])
        assert SliceArguments.validate(Gaussian2D(),
                                       ((1, False),)).get_slice(*inputs) ==\
            (inputs[1],)
        assert SliceArguments.validate(Gaussian2D(),
                                       ((0, True),)).get_slice(*inputs) ==\
            (inputs[0],)

    def test_is_slice(self):
        # Is Slice
        assert SliceArguments.validate(Gaussian2D(),
                                       ((0, True), (1, False))).is_slice((0.5, 2.5))
        assert SliceArguments.validate(Gaussian2D(),
                                       ((0, True),)).is_slice((0.5,))

        # Is not slice
        assert not SliceArguments.validate(Gaussian2D(),
                                           ((0, True), (1, False))).is_slice((0.5, 2.5, 3.5))
        assert not SliceArguments.validate(Gaussian2D(),
                                           ((0, True), (1, False))).is_slice((0.5,))
        assert not SliceArguments.validate(Gaussian2D(),
                                           ((0, True), (1, False))).is_slice(0.5)
        assert not SliceArguments.validate(Gaussian2D(),
                                           ((0, True),)).is_slice((0.5, 2.5))
        assert not SliceArguments.validate(Gaussian2D(),
                                           ((0, True),)).is_slice(2.5)

    def test_get_fixed_values(self):
        assert SliceArguments.validate(Gaussian2D(),
                                       ((0, True), (1, False))).get_fixed_values({0: 11, 1: 7}) \
            == (11, 7)
        assert SliceArguments.validate(Gaussian2D(),
                                       ((0, True), (1, False))).get_fixed_values({0: 5, 'y': 47}) \
            == (5, 47)
        assert SliceArguments.validate(Gaussian2D(),
                                       ((0, True), (1, False))).get_fixed_values({'x': 2, 'y': 9}) \
            == (2, 9)
        assert SliceArguments.validate(Gaussian2D(),
                                       ((0, True), (1, False))).get_fixed_values({'x': 12, 1: 19}) \
            == (12, 19)


class TestCompoundBoundingBox:
    def test_create(self):
        model = Gaussian2D()
        slice_args = ((0, True),)
        bounding_boxes = {(1,): (-1, 1), (2,): (-2, 2)}
        create_slice = mk.MagicMock()

        bounding_box = CompoundBoundingBox(bounding_boxes, model, slice_args, create_slice, 'test')
        assert (bounding_box._model.parameters == model.parameters).all()
        assert bounding_box._slice_args == slice_args
        for _slice, bbox in bounding_boxes.items():
            assert _slice in bounding_box._bounding_boxes
            assert bounding_box._bounding_boxes[_slice] == bbox
        for _slice, bbox in bounding_box._bounding_boxes.items():
            assert _slice in bounding_boxes
            assert bounding_boxes[_slice] == bbox
            assert isinstance(bbox, BoundingBox)
        assert bounding_box._bounding_boxes == bounding_boxes
        assert bounding_box._create_slice == create_slice
        assert bounding_box._order == 'test'

    def test___repr__(self):
        model = Gaussian2D()
        slice_args = ((0, True),)
        bounding_boxes = {(1,): (-1, 1), (2,): (-2, 2)}
        bounding_box = CompoundBoundingBox(bounding_boxes, model, slice_args)

        assert bounding_box.__repr__() ==\
            "CompoundBoundingBox(\n" + \
            "    bounding_boxes={\n" + \
            "        (1,) = BoundingBox(\n" + \
            "                intervals={\n" + \
            "                    x: Interval(lower=-1, upper=1)\n" + \
            "                }\n" + \
            "                model=Gaussian2D(inputs=('x', 'y'))\n" + \
            "                order='C'\n" + \
            "            )\n" + \
            "        (2,) = BoundingBox(\n" + \
            "                intervals={\n" + \
            "                    x: Interval(lower=-2, upper=2)\n" + \
            "                }\n" + \
            "                model=Gaussian2D(inputs=('x', 'y'))\n" + \
            "                order='C'\n" + \
            "            )\n" + \
            "    }\n" + \
            "    slice_args = SliceArguments(\n" + \
            "            Argument(name='x', ignore=True)\n" + \
            "        )\n" + \
            ")"

    def test_bounding_boxes(self):
        model = Gaussian2D()
        slice_args = ((0, True),)
        bounding_boxes = {(1,): (-1, 1), (2,): (-2, 2)}
        bounding_box = CompoundBoundingBox(bounding_boxes, model, slice_args)

        assert bounding_box._bounding_boxes == bounding_boxes
        assert bounding_box.bounding_boxes == bounding_boxes

    def test_slice_args(self):
        model = Gaussian2D()
        slice_args = ((0, True),)
        bounding_box = CompoundBoundingBox({}, model, slice_args)

        assert bounding_box._slice_args == slice_args
        assert bounding_box.slice_args == slice_args

    def test_create_slice(self):
        model = Gaussian2D()
        create_slice = mk.MagicMock()
        bounding_box = CompoundBoundingBox({}, model, ((1,),), create_slice)

        assert bounding_box._create_slice == create_slice
        assert bounding_box.create_slice == create_slice

    def test_order(self):
        model = Gaussian2D()
        create_slice = mk.MagicMock()
        bounding_box = CompoundBoundingBox({}, model, ((1,),), create_slice, 'test')

        assert bounding_box._order == 'test'
        assert bounding_box.order == 'test'

    def test__get_slice_key(self):
        bounding_box = CompoundBoundingBox({}, Gaussian2D(), ((1, True),))
        assert len(bounding_box.bounding_boxes) == 0

        # Singlar
        assert bounding_box._get_slice_key(5) == (5,)
        assert bounding_box._get_slice_key((5,)) == (5,)
        assert bounding_box._get_slice_key([5]) == (5,)
        assert bounding_box._get_slice_key(np.asanyarray(5)) == (5,)
        assert bounding_box._get_slice_key(np.array([5])) == (5,)

        # multiple
        assert bounding_box._get_slice_key((5, 19)) == (5, 19)
        assert bounding_box._get_slice_key([5, 19]) == (5, 19)
        assert bounding_box._get_slice_key(np.array([5, 19])) == (5, 19)

    def test___setitem__(self):
        model = Gaussian2D()

        # Ignored argument
        bounding_box = CompoundBoundingBox({}, model, ((1, True),), order='F')
        assert len(bounding_box.bounding_boxes) == 0
        # Valid
        bounding_box[(15, )] = (-15, 15)
        assert len(bounding_box.bounding_boxes) == 1
        assert (15,) in bounding_box._bounding_boxes
        assert isinstance(bounding_box._bounding_boxes[(15,)], BoundingBox)
        assert bounding_box._bounding_boxes[(15,)] == (-15, 15)
        assert bounding_box._bounding_boxes[(15,)].order == 'F'
        # Invalid key
        assert (7, 13) not in bounding_box._bounding_boxes
        with pytest.raises(ValueError) as err:
            bounding_box[(7, 13)] = (-7, 7)
        assert str(err.value) == \
            "(7, 13) is not a slice!"
        assert (7, 13) not in bounding_box._bounding_boxes
        assert len(bounding_box.bounding_boxes) == 1
        # Invalid bounding box
        assert 13 not in bounding_box._bounding_boxes
        with pytest.raises(ValueError):
            bounding_box[(13,)] = ((-13, 13), (-3, 3))
        assert 13 not in bounding_box._bounding_boxes
        assert len(bounding_box.bounding_boxes) == 1

        # No ignored argument
        bounding_box = CompoundBoundingBox({}, model, ((1, False),), order='F')
        assert len(bounding_box.bounding_boxes) == 0
        # Valid
        bounding_box[(15, )] = ((-15, 15), (-6, 6))
        assert len(bounding_box.bounding_boxes) == 1
        assert (15,) in bounding_box._bounding_boxes
        assert isinstance(bounding_box._bounding_boxes[(15,)], BoundingBox)
        assert bounding_box._bounding_boxes[(15,)] == ((-15, 15), (-6, 6))
        assert bounding_box._bounding_boxes[(15,)].order == 'F'
        # Invalid key
        assert (14, 11) not in bounding_box._bounding_boxes
        with pytest.raises(ValueError) as err:
            bounding_box[(14, 11)] = ((-7, 7), (-12, 12))
        assert str(err.value) == \
            "(14, 11) is not a slice!"
        assert (14, 11) not in bounding_box._bounding_boxes
        assert len(bounding_box.bounding_boxes) == 1
        # Invalid bounding box
        assert 13 not in bounding_box._bounding_boxes
        with pytest.raises(ValueError):
            bounding_box[(13,)] = (-13, 13)
        assert 13 not in bounding_box._bounding_boxes
        assert len(bounding_box.bounding_boxes) == 1

    def test__validate(self):
        model = Gaussian2D()
        slice_args = ((0, True),)

        # Tuple slice_args
        bounding_boxes = {(1,): (-1, 1), (2,): (-2, 2)}
        bounding_box = CompoundBoundingBox({}, model, slice_args)
        bounding_box._validate(bounding_boxes)
        for _slice, bbox in bounding_boxes.items():
            assert _slice in bounding_box._bounding_boxes
            assert bounding_box._bounding_boxes[_slice] == bbox
        for _slice, bbox in bounding_box._bounding_boxes.items():
            assert _slice in bounding_boxes
            assert bounding_boxes[_slice] == bbox
            assert isinstance(bbox, BoundingBox)
        assert bounding_box._bounding_boxes == bounding_boxes

    def test___eq__(self):
        bounding_box_1 = CompoundBoundingBox({(1,): (-1, 1), (2,): (-2, 2)}, Gaussian2D(), ((0, True),))
        bounding_box_2 = CompoundBoundingBox({(1,): (-1, 1), (2,): (-2, 2)}, Gaussian2D(), ((0, True),))

        # Equal
        assert bounding_box_1 == bounding_box_2

        # Not equal bounding_boxes
        bounding_box_2[(15,)] = (-15, 15)
        assert not bounding_box_1 == bounding_box_2
        del bounding_box_2._bounding_boxes[(15,)]
        assert bounding_box_1 == bounding_box_2

        # Not equal slice_args
        bounding_box_2._slice_args = SliceArguments.validate(Gaussian2D(), ((0, False),))
        assert not bounding_box_1 == bounding_box_2
        bounding_box_2._slice_args = SliceArguments.validate(Gaussian2D(), ((0, True),))
        assert bounding_box_1 == bounding_box_2

        # Note equal create_slice
        bounding_box_2._create_slice = mk.MagicMock()
        assert not bounding_box_1 == bounding_box_2

    def test_validate(self):
        model = Gaussian2D()
        slice_args = ((0, True),)
        bounding_boxes = {(1,): (-1, 1), (2,): (-2, 2)}
        create_slice = mk.MagicMock()

        # Fail slice_args
        with pytest.raises(RuntimeWarning) as err:
            CompoundBoundingBox.validate(model, bounding_boxes)
        assert str(err.value) == \
            "Slice arguments must be provided prior to model evaluation."

        # Normal validate
        bounding_box = CompoundBoundingBox.validate(model, bounding_boxes, slice_args,
                                                    create_slice, order='F')
        assert (bounding_box._model.parameters == model.parameters).all()
        assert bounding_box._slice_args == slice_args
        assert bounding_box._bounding_boxes == bounding_boxes
        assert bounding_box._create_slice == create_slice
        assert bounding_box._order == 'F'

        # Re-validate
        new_bounding_box = CompoundBoundingBox.validate(model, bounding_box)
        assert bounding_box == new_bounding_box
        assert new_bounding_box._order == 'F'

        # Default order
        bounding_box = CompoundBoundingBox.validate(model, bounding_boxes, slice_args,
                                                    create_slice)
        assert (bounding_box._model.parameters == model.parameters).all()
        assert bounding_box._slice_args == slice_args
        assert bounding_box._bounding_boxes == bounding_boxes
        assert bounding_box._create_slice == create_slice
        assert bounding_box._order == 'C'

    def test___contains__(self):
        model = Gaussian2D()
        slice_args = ((0, True),)
        bounding_boxes = {(1,): (-1, 1), (2,): (-2, 2)}
        bounding_box = CompoundBoundingBox(bounding_boxes, model, slice_args)

        assert (1,) in bounding_box
        assert (2,) in bounding_box

        assert (3,) not in bounding_box
        assert 1 not in bounding_box
        assert 2 not in bounding_box

    def test__create_bounding_box(self):
        model = Gaussian2D()
        create_slice = mk.MagicMock()
        bounding_box = CompoundBoundingBox({}, model, ((1,),), create_slice)

        # Create is successful
        create_slice.return_value = ((-15, 15), (-23, 23))
        assert len(bounding_box._bounding_boxes) == 0
        bbox = bounding_box._create_bounding_box((7,))
        assert isinstance(bbox, BoundingBox)
        assert bbox == ((-15, 15), (-23, 23))
        assert len(bounding_box._bounding_boxes) == 1
        assert (7,) in bounding_box
        assert isinstance(bounding_box[(7,)], BoundingBox)
        assert bounding_box[(7,)] == bbox

        # Create is unsuccessful
        create_slice.return_value = (-42, 42)
        with pytest.raises(ValueError):
            bounding_box._create_bounding_box((27,))

    def test___getitem__(self):
        model = Gaussian2D()
        slice_args = ((0, True),)
        bounding_boxes = {(1,): (-1, 1), (2,): (-2, 2)}
        bounding_box = CompoundBoundingBox(bounding_boxes, model, slice_args)

        # already exists
        assert isinstance(bounding_box[1], BoundingBox)
        assert bounding_box[1] == (-1, 1)
        assert isinstance(bounding_box[(2,)], BoundingBox)
        assert bounding_box[2] == (-2, 2)
        assert isinstance(bounding_box[(1,)], BoundingBox)
        assert bounding_box[(1,)] == (-1, 1)
        assert isinstance(bounding_box[(2,)], BoundingBox)
        assert bounding_box[(2,)] == (-2, 2)

        # no slice
        with pytest.raises(RuntimeError) as err:
            bounding_box[(3,)]
        assert str(err.value) == \
            "No bounding box is defined for slice: (3,)."

        # Create a slice
        bounding_box._create_slice = mk.MagicMock()
        with mk.patch.object(CompoundBoundingBox, '_create_bounding_box',
                             autospec=True) as mkCreate:
            assert bounding_box[(3,)] == mkCreate.return_value
            assert mkCreate.call_args_list == \
                [mk.call(bounding_box, (3,))]

    def test__select_bounding_box(self):
        model = Gaussian2D()
        slice_args = ((0, True),)
        bounding_boxes = {(1,): (-1, 1), (2,): (-2, 2)}
        bounding_box = CompoundBoundingBox(bounding_boxes, model, slice_args)

        inputs = [mk.MagicMock() for _ in range(3)]
        with mk.patch.object(SliceArguments, 'get_slice',
                             autospec=True) as mkSlice:
            with mk.patch.object(CompoundBoundingBox, '__getitem__',
                                 autospec=True) as mkGet:
                assert bounding_box._select_bounding_box(inputs) == mkGet.return_value
                assert mkGet.call_args_list == \
                    [mk.call(bounding_box, mkSlice.return_value)]
                assert mkSlice.call_args_list == \
                    [mk.call(bounding_box.slice_args, *inputs)]

    def test_prepare_inputs(self):
        model = Gaussian2D()
        slice_args = ((0, True),)
        bounding_boxes = {(1,): (-1, 1), (2,): (-2, 2)}
        bounding_box = CompoundBoundingBox(bounding_boxes, model, slice_args)

        input_shape = mk.MagicMock()
        with mk.patch.object(BoundingBox, 'prepare_inputs',
                             autospec=True) as mkPrepare:
            assert bounding_box.prepare_inputs(input_shape, [1, 2, 3]) == mkPrepare.return_value
            assert mkPrepare.call_args_list == \
                [mk.call(bounding_box[(1,)], input_shape, [1, 2, 3])]
            mkPrepare.reset_mock()
            assert bounding_box.prepare_inputs(input_shape, [2, 2, 3]) == mkPrepare.return_value
            assert mkPrepare.call_args_list == \
                [mk.call(bounding_box[(2,)], input_shape, [2, 2, 3])]
            mkPrepare.reset_mock()
