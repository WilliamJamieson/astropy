# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Tests for evaluation IO for Models"""

import pytest
import numpy as np
import unittest.mock as mk

from astropy.modeling import evaluation_io
from astropy.utils import shapes as utils_shapes
from astropy.modeling.utils import _BoundingBox


class TestIoEntry:
    def test___init__(self):
        entry = evaluation_io.IoEntry('name', 1)
        assert entry._name == 'name'
        assert entry._value == np.asanyarray(1)

        entry = evaluation_io.IoEntry('name', 1)
        assert entry._name == 'name'
        assert entry._value == np.asanyarray(1)

    def test___eq__(self):
        assert evaluation_io.IoEntry('name', 1) == evaluation_io.IoEntry('name', 1)
        assert evaluation_io.IoEntry('name', [1, 2]) == evaluation_io.IoEntry('name', [1, 2])
        assert evaluation_io.IoEntry('name', [1, 2]) == evaluation_io.IoEntry('name', [1, 2])

        entry = evaluation_io.IoEntry('name', [1, 2])
        fake_entry = mk.MagicMock()
        fake_entry.name = entry.name
        fake_entry.value = entry.value
        assert not (entry == fake_entry)

    def test_name(self):
        entry = evaluation_io.IoEntry('name', 1)
        assert entry.name == 'name'
        assert entry._name == 'name'

    def test_value(self):
        with mk.patch.object(np, 'asanyarray', autospec=True) as mkNp:
            entry = evaluation_io.IoEntry('name', 1)
            assert mkNp.call_args_list == [mk.call(1, dtype=float)]
            mkNp.reset_mock()

            # Test get
            assert entry.value == mkNp.return_value
            assert entry._value == mkNp.return_value

            # Test set
            entry.value = 2
            assert mkNp.call_args_list == [mk.call(2, dtype=float)]
            assert entry.value == mkNp.return_value
            assert entry._value == mkNp.return_value

    def test_shape(self):
        entry = evaluation_io.IoEntry('name', 1)
        assert entry.shape == tuple()

        entry = evaluation_io.IoEntry('name', [1, 2])
        assert entry.shape == (2,)

        entry = evaluation_io.IoEntry('name', np.ones((2, 2)))
        assert entry.shape == (2, 2)

        entry = evaluation_io.IoEntry('name', np.ones((2, 2, 2)))
        assert entry.shape == (2, 2, 2)


class TestInputEntry:
    def test_input_array(self):
        # Cast scalar to array
        entry = evaluation_io.InputEntry('name', 1)
        assert entry.input_array.shape == np.array([1], dtype=float).shape
        assert entry.input_array.shape != entry.value.shape
        assert entry.input_array == entry.value

        # Leave array alone
        entry = evaluation_io.InputEntry('name', [1, 2])
        assert entry.input_array.shape == np.array([1, 2], dtype=float).shape
        assert entry.input_array.shape == entry.value.shape
        assert (entry.input_array == entry.value).all()

    def test__array_shape(self):
        entry = evaluation_io.InputEntry('name', 1)
        assert entry._array_shape(True) == (1,)
        assert entry._array_shape(False) == tuple()

        entry = evaluation_io.InputEntry('name', [1, 2])
        assert entry._array_shape(True) == (2,)
        assert entry._array_shape(False) == (2,)

    def test_check_input_shape(self):
        entry = evaluation_io.InputEntry('name', 1)
        array_shape = mk.MagicMock()

        # Passes
        shapes = [tuple(), (1,), (4,), (2,)]
        with mk.patch.object(evaluation_io.InputEntry, '_array_shape', autospec=True,
                             side_effect=shapes) as mkShape:
            # Shape is false
            assert entry.check_input_shape(2, 0, array_shape) == shapes[0]
            assert mkShape.call_args_list == [mk.call(entry, array_shape)]
            mkShape.reset_mock()

            # Non-empty correct shape, n_models <= 1, model_set_axis not False
            assert entry.check_input_shape(1, 0, array_shape) == shapes[1]
            assert mkShape.call_args_list == [mk.call(entry, array_shape)]
            mkShape.reset_mock()

            # Non-empty correct shape, n_models > 1, model_set_axis is False
            assert entry.check_input_shape(5, False, array_shape) == shapes[2]
            assert mkShape.call_args_list == [mk.call(entry, array_shape)]
            mkShape.reset_mock()

            # Non-empty correct shape, n_models > 1, model_set_axis not False
            assert entry.check_input_shape(2, 0, array_shape) == shapes[3]
            assert mkShape.call_args_list == [mk.call(entry, array_shape)]

        # Fails
        shapes = [(2,), (1,)]
        with mk.patch.object(evaluation_io.InputEntry, '_array_shape', autospec=True,
                             side_effect=shapes) as mkShape:
            # len(shape) < model_set_axis + 1
            with pytest.raises(ValueError, match=r"For model_set_axis=.*"):
                entry.check_input_shape(2, 1, array_shape)
            assert mkShape.call_args_list == [mk.call(entry, array_shape)]
            mkShape.reset_mock()

            # shape[model_set_axis] != n_models
            with pytest.raises(ValueError, match=r"Input argument .*"):
                entry.check_input_shape(2, 0, array_shape)
            assert mkShape.call_args_list == [mk.call(entry, array_shape)]

    def test__check_broadcast(self):
        entry = evaluation_io.InputEntry('name', 1)
        input_shape = mk.MagicMock()
        param = mk.MagicMock()
        param_shape = mk.MagicMock()

        effects = [mk.MagicMock(), evaluation_io.IncompatibleShapeError(1, 2, 3, 4)]
        with mk.patch.object(evaluation_io, 'check_broadcast', autospec=True,
                             side_effect=effects) as mkCheck:
            assert entry._check_broadcast(input_shape, param, param_shape) == effects[0]
            assert mkCheck.call_args_list == [mk.call(input_shape, param_shape)]
            mkCheck.reset_mock()

            with pytest.raises(ValueError):
                entry._check_broadcast(input_shape, param, param_shape)

    def test__get_param_broadcast(self):
        entry = evaluation_io.InputEntry('name', 1)
        param = mk.MagicMock()
        with mk.patch.object(evaluation_io.InputEntry, '_check_broadcast',
                             autospec=True) as mkCheck:
            with mk.patch.object(evaluation_io.InputEntry, 'shape',
                                 new_callable=mk.PropertyMock) as mkShape:
                # Standard broadcast
                assert entry._get_param_broadcast(param, True) == mkCheck.return_value
                assert mkCheck.call_args_list == [mk.call(entry, mkShape.return_value, param, param.shape)]
                assert mkShape.call_args_list == [mk.call()]
                mkCheck.reset_mock()
                mkShape.reset_mock()

                # No standard broadcast
                assert entry._get_param_broadcast(param, False) == mkShape.return_value
                assert mkCheck.call_args_list == []
                assert mkShape.call_args_list == [mk.call()]

    def test__update_param_broadcast(self):
        entry = evaluation_io.InputEntry('name', 1)
        broadcast = (2,)
        param = mk.MagicMock()
        standard_broadcasting = mk.MagicMock()

        effects = [tuple(), (1,), (3,), (3, 4)]
        with mk.patch.object(evaluation_io.InputEntry, '_get_param_broadcast',
                             autospec=True, side_effect=effects) as mkGet:
            # broadcast longer than new
            assert entry._update_param_broadcast(broadcast, param, standard_broadcasting) == (2,)
            assert mkGet.call_args_list == [mk.call(entry, param, standard_broadcasting)]
            mkGet.reset_mock()

            # broadcast has same length as new, but larger
            assert entry._update_param_broadcast(broadcast, param, standard_broadcasting) == (2,)
            assert mkGet.call_args_list == [mk.call(entry, param, standard_broadcasting)]
            mkGet.reset_mock()

            # broadcast has same length as new, but smaller
            assert entry._update_param_broadcast(broadcast, param, standard_broadcasting) == (3,)
            assert mkGet.call_args_list == [mk.call(entry, param, standard_broadcasting)]
            mkGet.reset_mock()

            # broadcast shorter than new
            assert entry._update_param_broadcast(broadcast, param, standard_broadcasting) == (3, 4)
            assert mkGet.call_args_list == [mk.call(entry, param, standard_broadcasting)]

    def test_broadcast(self):
        entry = evaluation_io.InputEntry('name', 1)
        params = [mk.MagicMock() for _ in range(3)]
        standard_broadcasting = mk.MagicMock()

        effects = [mk.MagicMock() for _ in range(3)]
        with mk.patch.object(evaluation_io.InputEntry, '_update_param_broadcast',
                             autospec=True, side_effect=effects) as mkUpdate:
            with mk.patch.object(evaluation_io.InputEntry, 'shape',
                                 new_callable=mk.PropertyMock) as mkShape:
                # Has params
                assert entry.broadcast(params, standard_broadcasting) == effects[2]
                assert mkUpdate.call_args_list == \
                    [mk.call(entry, (),         params[0], standard_broadcasting),
                     mk.call(entry, effects[0], params[1], standard_broadcasting),
                     mk.call(entry, effects[1], params[2], standard_broadcasting)]
                assert mkShape.call_args_list == []
                mkUpdate.reset_mock()

                # Has no params
                assert entry.broadcast([], standard_broadcasting) == mkShape.return_value
                assert mkUpdate.call_args_list == []
                assert mkShape.call_args_list == [mk.call()]

    def test__remove_axes_from_shape(self):
        entry = evaluation_io.InputEntry('name', 1)

        # len(shape) == 0
        assert entry._remove_axes_from_shape((), mk.MagicMock()) == ()

        shape = tuple(range(5))
        # axis < 0
        assert entry._remove_axes_from_shape(shape, -1) == (0, 1, 2, 3)
        assert entry._remove_axes_from_shape(shape, -2) == (0, 1, 2, 4)
        assert entry._remove_axes_from_shape(shape, -3) == (0, 1, 3, 4)
        assert entry._remove_axes_from_shape(shape, -4) == (0, 2, 3, 4)
        assert entry._remove_axes_from_shape(shape, -5) == (1, 2, 3, 4)
        # axis >= len(shape)
        assert entry._remove_axes_from_shape(shape, 5) == ()
        assert entry._remove_axes_from_shape(shape, 6) == ()
        assert entry._remove_axes_from_shape(shape, 7) == ()
        # 0 <= axis < len(shape)
        assert entry._remove_axes_from_shape(shape, 0) == (1, 2, 3, 4)
        assert entry._remove_axes_from_shape(shape, 1) == (2, 3, 4)
        assert entry._remove_axes_from_shape(shape, 2) == (3, 4)
        assert entry._remove_axes_from_shape(shape, 3) == (4,)

    def test__get_param_shape(self):
        entry = evaluation_io.InputEntry('name', 1)
        input_shape = mk.MagicMock()
        param = mk.MagicMock()
        model_set_axis = mk.MagicMock()

        with mk.patch.object(evaluation_io.InputEntry, '_remove_axes_from_shape',
                             autospec=True) as mkRemove:
            with mk.patch.object(evaluation_io.InputEntry, '_check_broadcast',
                                 autospec=True) as mkCheck:
                assert entry._get_param_shape(input_shape, param, model_set_axis) \
                    == mkRemove.return_value
                assert mkRemove.call_args_list == \
                    [mk.call(param.shape, model_set_axis)]
                assert mkCheck.call_args_list == \
                    [mk.call(entry, input_shape, param, mkRemove.return_value)]

    def test__update_max_param_shape(self):
        entry = evaluation_io.InputEntry('name', 1)
        max_param_shape = (mk.MagicMock(), mk.MagicMock())
        input_shape = mk.MagicMock()
        param = mk.MagicMock()
        model_set_axis = mk.MagicMock()

        with mk.patch.object(evaluation_io.InputEntry, '_get_param_shape',
                             autospec=True) as mkGet:
            # no update
            param.shape = (mk.MagicMock(), mk.MagicMock(), mk.MagicMock())
            assert entry._update_max_param_shape(max_param_shape, input_shape, param, model_set_axis) ==\
                max_param_shape
            assert mkGet.call_args_list == [mk.call(entry, input_shape, param, model_set_axis)]
            mkGet.reset_mock()

            # update
            param.shape = (mk.MagicMock(), mk.MagicMock(), mk.MagicMock(), mk.MagicMock())
            assert entry._update_max_param_shape(max_param_shape, input_shape, param, model_set_axis) ==\
                mkGet.return_value
            assert mkGet.call_args_list == [mk.call(entry, input_shape, param, model_set_axis)]
            mkGet.reset_mock()

    def test__max_param_shape(self):
        entry = evaluation_io.InputEntry('name', 1)
        input_shape = mk.MagicMock()
        params = [mk.MagicMock() for _ in range(3)]
        model_set_axis = mk.MagicMock()

        effects = [mk.MagicMock() for _ in range(3)]
        with mk.patch.object(evaluation_io.InputEntry, '_update_max_param_shape',
                             autospec=True, side_effect=effects) as mkUpdate:
            assert entry._max_param_shape(input_shape, params, model_set_axis) == effects[2]
            assert mkUpdate.call_args_list == \
                [mk.call(entry, (),         input_shape, params[0], model_set_axis),
                 mk.call(entry, effects[0], input_shape, params[1], model_set_axis),
                 mk.call(entry, effects[1], input_shape, params[2], model_set_axis)]

    def test__new_input_value(self):
        entry = evaluation_io.InputEntry('name', 1)
        input_ndim = mk.MagicMock()
        max_param_shape = mk.MagicMock()
        model_set_axis = mk.MagicMock()

        with mk.patch.object(evaluation_io.InputEntry, '_new_input_no_axis',
                             autospec=True) as mkNoAxis:
            with mk.patch.object(evaluation_io.InputEntry, '_new_input_axis',
                                 autospec=True) as mkAxis:
                # Axis
                assert entry._new_input_value(input_ndim, max_param_shape,
                                              model_set_axis, True) == \
                    mkAxis.return_value
                assert mkNoAxis.call_args_list == []
                assert mkAxis.call_args_list == \
                    [mk.call(entry, input_ndim, max_param_shape, model_set_axis, True)]
                mkAxis.reset_mock()

                # No axis
                assert entry._new_input_value(input_ndim, max_param_shape,
                                              model_set_axis, False) == \
                    mkNoAxis.return_value
                assert mkNoAxis.call_args_list == \
                    [mk.call(entry, input_ndim, max_param_shape, model_set_axis)]
                assert mkAxis.call_args_list == []

    def test__get_input_shape(self):
        entry = evaluation_io.InputEntry('name', 1)

        value = tuple(range(5))
        with mk.patch.object(evaluation_io.InputEntry, 'shape',
                             new_callable=mk.PropertyMock, return_value=value) as mkShape:
            # No adjustment
            assert entry._get_input_shape(1, mk.MagicMock()) == value
            assert mkShape.call_args_list == [mk.call()]
            mkShape.reset_mock()
            assert entry._get_input_shape(2, False) == value
            assert mkShape.call_args_list == [mk.call()]
            mkShape.reset_mock()

            # Adjustment
            assert entry._get_input_shape(2, 0) == (1, 2, 3, 4)
            assert mkShape.call_args_list == [mk.call(), mk.call()]
            mkShape.reset_mock()
            assert entry._get_input_shape(2, 1) == (0, 2, 3, 4)
            assert mkShape.call_args_list == [mk.call(), mk.call()]
            mkShape.reset_mock()
            assert entry._get_input_shape(2, 2) == (0, 1, 3, 4)
            assert mkShape.call_args_list == [mk.call(), mk.call()]
            mkShape.reset_mock()
            assert entry._get_input_shape(2, 3) == (0, 1, 2, 4)
            assert mkShape.call_args_list == [mk.call(), mk.call()]
            mkShape.reset_mock()
            assert entry._get_input_shape(2, 4) == (0, 1, 2, 3)
            assert mkShape.call_args_list == [mk.call(), mk.call()]
            mkShape.reset_mock()
            assert entry._get_input_shape(2, 5) == (0, 1, 2, 3, 4)
            assert mkShape.call_args_list == [mk.call(), mk.call()]
            mkShape.reset_mock()
            assert entry._get_input_shape(2, 6) == (0, 1, 2, 3, 4)
            assert mkShape.call_args_list == [mk.call(), mk.call()]

    def test_new_input(self):
        entry = evaluation_io.InputEntry('name', 1)
        params = mk.MagicMock()
        model_set_axis = mk.MagicMock()
        n_models = mk.MagicMock()
        model_set_axis_input = mk.MagicMock()

        value = (mk.MagicMock(), mk.MagicMock())
        with mk.patch.object(evaluation_io.InputEntry, '_get_input_shape',
                             autospec=True) as mkGet:
            with mk.patch.object(evaluation_io.InputEntry, '_max_param_shape',
                                 autospec=True) as mkMax:
                with mk.patch.object(evaluation_io.InputEntry, '_new_input_value',
                                     autospec=True, return_value=value) as mkNew:
                    new_input, pivot = entry.new_input(params, model_set_axis,
                                                       n_models, model_set_axis_input)
                    assert pivot == value[1]
                    assert isinstance(new_input, evaluation_io.InputEntry)
                    assert new_input.name == 'name'
                    assert (new_input.value == value[0]).all()

                    assert mkNew.call_args_list == \
                        [mk.call(entry, mkGet.return_value.__len__.return_value,
                                 mkMax.return_value, model_set_axis, model_set_axis_input)]
                    assert mkGet.return_value.__len__.call_args_list == [mk.call()]
                    assert mkGet.call_args_list == \
                        [mk.call(entry, n_models, model_set_axis_input)]
                    assert mkMax.call_args_list == \
                        [mk.call(entry, mkGet.return_value, params, model_set_axis)]

    def test_reduce_to_bounding_box(self):
        entry = evaluation_io.InputEntry('name', np.arange(0, 5))
        valid_index = [1, 3, 4]
        assert entry.reduce_to_bounding_box(valid_index) == \
            evaluation_io.InputEntry('name', [1, 3, 4])

        entry = evaluation_io.InputEntry('name', 1)
        valid_index = [0]
        assert entry.reduce_to_bounding_box(valid_index) == \
            evaluation_io.InputEntry('name', 1)


class TestInputs:
    def test___init__(self):
        inputs = evaluation_io.Inputs({'test': evaluation_io.InputEntry('test', 1)})
        assert inputs._inputs == {'test': evaluation_io.InputEntry('test', 1)}

    def test___eq__(self):
        inputs1 = evaluation_io.Inputs({'test': evaluation_io.InputEntry('test', 1)})
        inputs2 = evaluation_io.Inputs({'test': evaluation_io.InputEntry('test', 1)})

        # Equal
        assert inputs1 == inputs2

        # Not matching inputs
        inputs2._inputs = mk.MagicMock()
        assert not (inputs1 == inputs2)

        # Not matching type
        assert not (inputs1 == mk.MagicMock())

    def test_inputs(self):
        inputs = evaluation_io.Inputs({'test': evaluation_io.InputEntry('test', 1)})
        assert inputs._inputs == {'test': evaluation_io.InputEntry('test', 1)}
        assert inputs.inputs == {'test': evaluation_io.InputEntry('test', 1)}

    def test_n_inputs(self):
        inputs = evaluation_io.Inputs({'test': evaluation_io.InputEntry('test', 1)})
        assert inputs.n_inputs == 1

        inputs = evaluation_io.Inputs({'test': evaluation_io.InputEntry('test', 1),
                                       'next': evaluation_io.InputEntry('next', 2)})
        assert inputs.n_inputs == 2

    def test_copy(self):
        inputs = evaluation_io.Inputs({'test': evaluation_io.InputEntry('test', 1)})
        with mk.patch.object(evaluation_io, 'deepcopy', autospec=True) as mkDeepcopy:
            assert inputs.copy() == mkDeepcopy.return_value
            assert mkDeepcopy.call_args_list == [mk.call(inputs)]

    def test__check_input_shape(self):
        entries = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        inputs = evaluation_io.Inputs(entries)

        check_args = [entry.check_input_shape.return_value for entry in entries.values()]

        n_models = mk.MagicMock()
        model_set_axis = mk.MagicMock()
        array_shape = mk.MagicMock()

        effects = [mk.MagicMock(), None]
        with mk.patch.object(evaluation_io, 'check_broadcast', autospec=True,
                             side_effect=effects) as mkCheck:
            # Success
            assert inputs.check_input_shape(n_models, model_set_axis,
                                            array_shape) == effects[0]
            for index, entry in enumerate(entries.values()):
                assert entry.check_input_shape.call_args_list == \
                    [mk.call(n_models, model_set_axis, array_shape)]
                entry.check_input_shape.reset_mock()
            assert mkCheck.call_args_list == [mk.call(*check_args)]
            mkCheck.reset_mock()

            # Fail
            with pytest.raises(ValueError):
                inputs.check_input_shape(n_models, model_set_axis, array_shape)
            for index, entry in enumerate(entries.values()):
                assert entry.check_input_shape.call_args_list == \
                    [mk.call(n_models, model_set_axis, array_shape)]
                entry.check_input_shape.reset_mock()
            assert mkCheck.call_args_list == [mk.call(*check_args)]

    def test__get_broadcasts(self):
        entries = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        inputs = evaluation_io.Inputs(entries)

        params = mk.MagicMock()
        standard_broadcasting = mk.MagicMock()

        broadcasts = inputs._get_broadcasts(params, standard_broadcasting)
        assert broadcasts == [entry.broadcast.return_value for entry in entries.values()]
        for entry in entries.values():
            assert entry.broadcast.call_args_list == [mk.call(params, standard_broadcasting)]

    def test__extend_broadcasts(self):
        # No inputs
        inputs = evaluation_io.Inputs({})

        broadcasts = []
        inputs._extend_broadcasts(3, broadcasts)
        assert broadcasts == [(), (), (), ()]

        broadcasts = [(1,)]
        inputs._extend_broadcasts(3, broadcasts)
        assert broadcasts == [(1,), (1,), (1,), (1,)]

        broadcasts = []
        inputs._extend_broadcasts(0, broadcasts)
        assert broadcasts == []

        broadcasts = [(1,)]
        inputs._extend_broadcasts(0, broadcasts)
        assert broadcasts == [(1,)]

        # Some inputs
        entries = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        inputs = evaluation_io.Inputs(entries)

        broadcasts = [(1,), (2,), (3,)]
        inputs._extend_broadcasts(4, broadcasts)
        assert broadcasts == [(1,), (2,), (3,), (1,)]

        broadcasts = [(1,), (2,), (3,)]
        inputs._extend_broadcasts(3, broadcasts)
        assert broadcasts == [(1,), (2,), (3,)]

        broadcasts = []
        inputs._extend_broadcasts(3, broadcasts)
        assert broadcasts == []

        broadcasts = []
        inputs._extend_broadcasts(4, broadcasts)
        assert broadcasts == [(), ()]

    def test_broadcast(self):
        inputs = evaluation_io.Inputs({})

        params = mk.MagicMock()
        standard_broadcasting = mk.MagicMock()
        n_outputs = mk.MagicMock()

        with mk.patch.object(evaluation_io.Inputs, '_get_broadcasts',
                             autospec=True) as mkGet:
            with mk.patch.object(evaluation_io.Inputs, '_extend_broadcasts',
                                 autospec=True) as mkExtend:
                assert inputs.broadcast(params, standard_broadcasting, n_outputs) ==\
                    mkGet.return_value
                assert mkGet.call_args_list == \
                    [mk.call(inputs, params, standard_broadcasting)]
                assert mkExtend.call_args_list == \
                    [mk.call(inputs, n_outputs, mkGet.return_value)]

    def test__new_inputs(self):
        entries = {f'x{idx}': evaluation_io.InputEntry(f'x{idx}', mk.MagicMock())
                   for idx in range(3)}
        inputs = evaluation_io.Inputs(entries)
        params = mk.MagicMock()
        model_set_axis = mk.MagicMock()
        n_models = mk.MagicMock()
        model_set_axis_input = mk.MagicMock()

        old_entries = entries.copy()
        new_inputs = [(mk.MagicMock(), mk.MagicMock()) for _ in range(3)]
        with mk.patch.object(evaluation_io.InputEntry, 'new_input',
                             autospec=True, side_effect=new_inputs) as mkNew:
            pivots = inputs._new_inputs(params, model_set_axis, n_models, model_set_axis_input)
            assert pivots == [_input[1] for _input in new_inputs]
            assert inputs._inputs == {f'x{idx}': new_inputs[idx][0]
                                      for idx in range(3)}
            assert mkNew.call_args_list == \
                [mk.call(old_entries[f'x{idx}'], params, model_set_axis,
                         n_models, model_set_axis_input) for idx in range(3)]

    def test_pivots(self):
        entries = {f'x{idx}': evaluation_io.InputEntry(f'x{idx}', mk.MagicMock())
                   for idx in range(3)}
        inputs = evaluation_io.Inputs(entries)
        params = mk.MagicMock()
        model_set_axis = mk.MagicMock()
        n_models = mk.MagicMock()
        model_set_axis_input = mk.MagicMock()

        pivots = [mk.MagicMock() for _ in range(3)]
        unmodified_pivots = pivots.copy()
        with mk.patch.object(evaluation_io.Inputs, '_new_inputs',
                             autospec=True, return_value=pivots) as mkNew:
            with mk.patch.object(evaluation_io.Inputs, 'n_inputs',
                                 new_callable=mk.PropertyMock,
                                 return_value=3) as mkInputs:
                # No extension (n_inputs >= n_outputs)
                assert inputs.pivots(params, model_set_axis, n_models,
                                     1, model_set_axis_input) == pivots
                assert mkNew.call_args_list == \
                    [mk.call(inputs, params, model_set_axis, n_models, model_set_axis_input)]
                assert mkInputs.call_args_list == [mk.call()]
                mkNew.reset_mock()
                mkInputs.reset_mock()

                # Extension (n_inputs < n_outputs)
                assert inputs.pivots(params, model_set_axis, n_models,
                                     6, model_set_axis_input) == \
                    unmodified_pivots + [model_set_axis_input for _ in range(3)]
                assert mkNew.call_args_list == \
                    [mk.call(inputs, params, model_set_axis, n_models, model_set_axis_input)]
                assert mkInputs.call_args_list == [mk.call(), mk.call()]

    def test__reduce_to_bounding_box(self):
        entries = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        inputs = evaluation_io.Inputs(entries)

        valid_index = mk.MagicMock()
        all_out = mk.MagicMock()
        reduced = inputs._reduce_to_bounding_box(valid_index)
        assert isinstance(reduced, evaluation_io.Inputs)
        assert len(reduced.inputs) == len(entries) == 3
        for name, entry in entries.items():
            assert name in reduced.inputs
            assert reduced.inputs[name] == entry.reduce_to_bounding_box.return_value
            assert entry.reduce_to_bounding_box.call_args_list == \
                [mk.call(valid_index)]
        for name, entry in reduced.inputs.items():
            assert name in entries
            assert entries[name].reduce_to_bounding_box.return_value == entry
            assert entries[name].reduce_to_bounding_box.call_args_list == \
                [mk.call(valid_index)]

    def test_reduce_to_bounding_box(self):
        entries = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        inputs = evaluation_io.Inputs(entries)

        valid_index = mk.MagicMock()

        with mk.patch.object(evaluation_io.Inputs, '_reduce_to_bounding_box',
                             autospec=True) as mkReduce:
            with mk.patch.object(evaluation_io.Inputs, 'copy',
                                 autospec=True) as mkCopy:
                # Test all out
                assert inputs.reduce_to_bounding_box(valid_index, True) == \
                    mkCopy.return_value
                assert mkCopy.call_args_list == [mk.call(inputs)]
                assert mkReduce.call_args_list == []
                mkCopy.reset_mock()

                # Test actually reduce
                assert inputs.reduce_to_bounding_box(valid_index, False) == \
                    mkReduce.return_value
                assert mkReduce.call_args_list == \
                    [mk.call(inputs, valid_index)]
                assert mkCopy.call_args_list == []


class TestOptional:
    def test___init__(self):
        # Defaults
        inputs = evaluation_io.Optional({'option': 2})
        assert inputs._optional == {'option': 2}
        assert inputs._model_options == evaluation_io.modeling_options
        assert inputs._pass_through == {}
        assert inputs._model_set_axis == 0

        # Optionals
        modeling_options = {name: mk.MagicMock() for name in evaluation_io.modeling_options}
        inputs = evaluation_io.Optional({'option': 2}, modeling_options, {'pass': 1}, 2)
        assert inputs._optional == {'option': 2}
        assert inputs._model_options == modeling_options
        assert inputs._pass_through == {'pass': 1}
        assert inputs._model_set_axis == 2

        # Error
        with pytest.raises(ValueError):
            evaluation_io.Optional(mk.MagicMock(), mk.MagicMock(),
                                   mk.MagicMock(), mk.MagicMock())

    def test___eq__(self):
        inputs1 = evaluation_io.Optional({'option': 2}, evaluation_io.modeling_options,
                                         {'pass': 1}, 2)
        inputs2 = evaluation_io.Optional({'option': 2}, evaluation_io.modeling_options,
                                         {'pass': 1}, 2)

        # Equal
        assert inputs1 == inputs2

        # Not matching model_set_axis
        inputs2._model_set_axis = 4
        assert not (inputs1 == inputs2)

        # Not matching pass_through
        inputs2._model_set_axis = inputs1._model_set_axis
        inputs2._pass_through = mk.MagicMock()
        assert not (inputs1 == inputs2)

        # Not matching model_options
        inputs2._pass_through = inputs1._pass_through
        inputs2._model_options = mk.MagicMock()
        assert not (inputs1 == inputs2)

        # Not matching optional
        inputs2._model_options = inputs1._model_options
        inputs2._optional = mk.MagicMock()
        assert not (inputs1 == inputs2)

        # Not matching type
        assert not (inputs1 == mk.MagicMock())

    def test_optional(self):
        inputs = evaluation_io.Optional({'option': 2})
        assert inputs._optional == {'option': 2}
        assert inputs.optional == {'option': 2}

    def test_model_options(self):
        inputs = evaluation_io.Optional({'option': 2})

        # Test get
        assert inputs._model_options == evaluation_io.modeling_options
        assert inputs.model_options == evaluation_io.modeling_options

        # Test set and get successfully
        modeling_options = {name: mk.MagicMock() for name in evaluation_io.modeling_options}
        inputs.model_options = modeling_options
        assert inputs._model_options == modeling_options
        assert inputs.model_options == modeling_options

        # Test set fail
        for name in evaluation_io.modeling_options:
            new_model_options = modeling_options.copy()
            del new_model_options[name]
            with pytest.raises(ValueError, match=f"Modeling option {name} must be set!"):
                inputs.model_options = new_model_options

    def test_default_model_set_axis(self):
        model_set_axis = mk.MagicMock()
        inputs = evaluation_io.Optional({'option': 2}, model_set_axis=model_set_axis)

        assert inputs._model_set_axis == model_set_axis
        assert inputs.default_model_set_axis == model_set_axis

    def test__get_model_option(self):
        inputs = evaluation_io.Optional({'option': 2})
        model_options = mk.MagicMock()
        model_options.__contains__.side_effect = [True, False]
        inputs._model_options = model_options

        # Test success
        assert inputs._get_model_option('test') == model_options.__getitem__.return_value
        assert model_options.__getitem__.call_args_list == [mk.call('test')]
        assert model_options.__contains__.call_args_list == [mk.call('test')]
        model_options.reset_mock()

        with pytest.raises(RuntimeError, match=r"Option .*"):
            inputs._get_model_option('test')
        assert model_options.__getitem__.call_args_list == []
        assert model_options.__contains__.call_args_list == [mk.call('test')]

    def test_model_set_axis(self):
        model_set_axis = mk.MagicMock()
        inputs = evaluation_io.Optional({'option': 2}, model_set_axis=model_set_axis)

        effects = [mk.MagicMock(), None]
        with mk.patch.object(evaluation_io.Optional, '_get_model_option',
                             autospec=True, side_effect=effects) as mkGet:
            assert inputs.model_set_axis == effects[0]
            assert mkGet.call_args_list == [mk.call(inputs, 'model_set_axis')]

            mkGet.reset_mock()
            assert inputs.model_set_axis == model_set_axis
            assert mkGet.call_args_list == [mk.call(inputs, 'model_set_axis')]

    def test_with_bounding_box(self):
        inputs = evaluation_io.Optional({'option': 2})
        with mk.patch.object(evaluation_io.Optional, '_get_model_option',
                             autospec=True) as mkGet:
            assert inputs.with_bounding_box == mkGet.return_value
            assert mkGet.call_args_list == [mk.call(inputs, 'with_bounding_box')]

    def test_fill_value(self):
        inputs = evaluation_io.Optional({'option': 2})
        with mk.patch.object(evaluation_io.Optional, '_get_model_option',
                             autospec=True) as mkGet:
            assert inputs.fill_value == mkGet.return_value
            assert mkGet.call_args_list == [mk.call(inputs, 'fill_value')]

    def test_equivalencies(self):
        inputs = evaluation_io.Optional({'option': 2})
        with mk.patch.object(evaluation_io.Optional, '_get_model_option',
                             autospec=True) as mkGet:
            assert inputs.equivalencies == mkGet.return_value
            assert mkGet.call_args_list == [mk.call(inputs, 'equivalencies')]

    def test_inputs_map(self):
        inputs = evaluation_io.Optional({'option': 2})
        with mk.patch.object(evaluation_io.Optional, '_get_model_option',
                             autospec=True) as mkGet:
            assert inputs.inputs_map == mkGet.return_value
            assert mkGet.call_args_list == [mk.call(inputs, 'inputs_map')]

    def test_pass_through(self):
        inputs = evaluation_io.Optional({'option': 2})
        assert inputs._pass_through == {}
        assert inputs.pass_through == {}

        inputs = evaluation_io.Optional({'option': 2}, pass_through={'pass': 7})
        assert inputs._pass_through == {'pass': 7}
        assert inputs.pass_through == {'pass': 7}

    def test_validate(self):
        # Test pass_through is empty
        inputs = evaluation_io.Optional({'option': 2})
        assert len(inputs.pass_through) == 0
        inputs.validate(True)
        inputs.validate(False)

        # Test pass_through is non-empty
        inputs = evaluation_io.Optional({'option': 2},
                                        pass_through={'test': mk.MagicMock()})
        assert len(inputs.pass_through) > 0
        inputs.validate(True)
        with pytest.raises(RuntimeError):
            inputs.validate(False)


class TestInputData:
    def test___init__(self):
        # Defaults
        inputs = evaluation_io.InputData()
        assert inputs._format_info is None
        assert inputs._valid_index is None
        assert inputs._all_out is None

        # Optionals
        inputs = evaluation_io.InputData([3], np.array([7]), True)
        assert inputs._format_info == [3]
        assert inputs._valid_index == np.array([7])
        assert inputs._all_out

    def test___eq__(self):
        inputs1 = evaluation_io.InputData([3], np.array([7]), True)
        inputs2 = evaluation_io.InputData([3], np.array([7]), True)

        # Equal
        assert inputs1 == inputs2

        # Not matching all_out
        inputs2._all_out = mk.MagicMock()
        assert not (inputs1 == inputs2)

        # Not matching valid_index
        inputs2._all_out = inputs1._all_out
        inputs2._valid_index = np.ndarray([4])
        assert not (inputs1 == inputs2)

        # Not matching format_info
        inputs2._valid_index = inputs1._valid_index
        inputs2._format_info = mk.MagicMock()
        assert not (inputs1 == inputs2)

        # Not matching type
        assert not (inputs1 == mk.MagicMock())

    def test_format_info(self):
        inputs = evaluation_io.InputData()
        # None value
        assert inputs._format_info is None
        assert inputs.format_info == []

        # Not None
        inputs.format_info = [3, 4]
        assert inputs._format_info == [3, 4]
        assert inputs.format_info == [3, 4]

    def test_valid_index(self):
        inputs = evaluation_io.InputData()

        # Test get if None
        assert inputs._valid_index is None
        assert (inputs.valid_index == np.array([])).all()

        # Test set and get
        inputs.valid_index = np.array([1, 2, 3])
        assert (inputs._valid_index == np.array([1, 2, 3])).all()
        assert (inputs.valid_index == np.array([1, 2, 3])).all()

    def test_all_out(self):
        inputs = evaluation_io.InputData()
        # None value
        assert inputs._all_out is None
        assert inputs.all_out == False

        # Test set and get
        inputs.all_out = True
        assert inputs.all_out == True
        inputs.all_out = False
        assert inputs.all_out == False

    def test_copy(self):
        inputs = evaluation_io.InputData()
        with mk.patch.object(evaluation_io, 'deepcopy', autospec=True) as mkDeepcopy:
            assert inputs.copy() == mkDeepcopy.return_value
            assert mkDeepcopy.call_args_list == [mk.call(inputs)]

    def test_reduce_to_bounding_box(self):
        inputs = evaluation_io.InputData()
        valid_index = mk.MagicMock()
        all_out = mk.MagicMock()

        new_inputs = evaluation_io.InputData([3], np.array([7]), True)
        assert not (inputs == new_inputs)
        with mk.patch.object(evaluation_io.InputData, 'copy',
                             autospec=True, return_value=new_inputs) as mkCopy:
            with mk.patch.object(evaluation_io.InputData, 'valid_index',
                                 new_callable=mk.PropertyMock) as mkValid:
                with mk.patch.object(evaluation_io.InputData, 'all_out',
                                     new_callable=mk.PropertyMock) as mkAll:
                    reduced = inputs.reduce_to_bounding_box(valid_index, all_out)
                    assert mkCopy.call_args_list == [mk.call(inputs)]
                    assert mkValid.call_args_list == [mk.call(valid_index)]
                    assert mkAll.call_args_list == [mk.call(all_out)]
        assert reduced == new_inputs


class TestEvaluationInputs:
    def test___init__(self):
        inputs = mk.MagicMock()
        optional = mk.MagicMock()
        data = mk.MagicMock()
        evaluation = evaluation_io.EvaluationInputs(inputs, optional, data)
        assert evaluation._inputs == inputs
        assert evaluation._optional == optional
        assert evaluation._data == data

    def test_inputs(self):
        inputs = mk.MagicMock()
        optional = mk.MagicMock()
        data = mk.MagicMock()
        evaluation = evaluation_io.EvaluationInputs(inputs, optional, data)

        # Test Get
        assert evaluation._inputs == inputs
        assert evaluation.inputs == inputs

        # Test set
        new_inputs = mk.MagicMock()
        evaluation.inputs = new_inputs
        assert evaluation._inputs == new_inputs
        assert evaluation.inputs == new_inputs

    def test_optional(self):
        inputs = mk.MagicMock()
        optional = mk.MagicMock()
        data = mk.MagicMock()
        evaluation = evaluation_io.EvaluationInputs(inputs, optional, data)

        # Test Get
        assert evaluation._optional == optional
        assert evaluation.optional == optional

        # Test set
        new_optional = mk.MagicMock()
        evaluation.optional = new_optional
        assert evaluation._optional == new_optional
        assert evaluation.optional == new_optional

    def test_data(self):
        inputs = mk.MagicMock()
        optional = mk.MagicMock()
        data = mk.MagicMock()
        evaluation = evaluation_io.EvaluationInputs(inputs, optional, data)

        # Test Get
        assert evaluation._data == data
        assert evaluation.data == data

        # Test set
        new_data = mk.MagicMock()
        evaluation.data = new_data
        assert evaluation._data == new_data
        assert evaluation.data == new_data

    def test_format_info(self):
        inputs = mk.MagicMock()
        optional = mk.MagicMock()
        data = mk.MagicMock()
        evaluation = evaluation_io.EvaluationInputs(inputs, optional, data)

        # Test Get
        assert evaluation._data.format_info == data.format_info
        assert evaluation.format_info == data.format_info

        # Test set
        new_format_info = mk.MagicMock()
        evaluation.format_info = new_format_info
        assert evaluation._data.format_info == new_format_info
        assert evaluation.format_info == new_format_info

    def test_check_input_shape(self):
        inputs = mk.MagicMock()
        optional = mk.MagicMock()
        data = mk.MagicMock()
        evaluation = evaluation_io.EvaluationInputs(inputs, optional, data)
        n_models = mk.MagicMock()

        evaluation.check_input_shape(n_models)
        assert inputs.check_input_shape.call_args_list == \
            [mk.call(n_models, optional.model_set_axis, False)]

    def test_evaluation_inputs(self):
        inputs = mk.MagicMock()
        optional = mk.MagicMock()
        evaluation = evaluation_io.EvaluationInputs.evaluation_inputs(inputs, optional)
        assert evaluation._inputs == inputs
        assert evaluation._optional == optional
        assert evaluation._data == evaluation_io.InputData()

    def test_set_format_info(self):
        inputs = evaluation_io.Inputs({})
        optional = mk.MagicMock()
        evaluation = evaluation_io.EvaluationInputs.evaluation_inputs(inputs, optional)

        params = mk.MagicMock()
        standard_broadcasting = mk.MagicMock()
        model_set_axis = mk.MagicMock()
        n_outputs = mk.MagicMock()

        with mk.patch.object(evaluation_io.Inputs, 'broadcast',
                             autospec=True) as mkBroadcast:
            with mk.patch.object(evaluation_io.Inputs, 'pivots',
                                 autospec=True) as mkPivots:
                assert evaluation.format_info == []
                evaluation.set_format_info(params, standard_broadcasting, 1,
                                           model_set_axis, n_outputs)
                assert evaluation.format_info == mkBroadcast.return_value
                assert mkBroadcast.call_args_list == \
                    [mk.call(inputs, params, standard_broadcasting, n_outputs)]
                assert mkPivots.call_args_list == []
                mkBroadcast.reset_mock()

                evaluation.set_format_info(params, standard_broadcasting, 2,
                                           model_set_axis, n_outputs)
                assert evaluation.format_info == mkPivots.return_value
                assert mkBroadcast.call_args_list == []
                assert mkPivots.call_args_list == \
                    [mk.call(inputs, params, model_set_axis, 2, n_outputs,
                             optional.model_set_axis)]

    def test_reduce_to_bounding_box(self):
        inputs = evaluation_io.Inputs({})
        optional = mk.MagicMock()
        data = evaluation_io.InputData()
        evaluation = evaluation_io.EvaluationInputs(inputs, optional, data)

        valid_index = mk.MagicMock()
        all_out = mk.MagicMock()

        with mk.patch.object(evaluation_io.Inputs, 'reduce_to_bounding_box',
                             autospec=True) as mkInputs:
            with mk.patch.object(evaluation_io.InputData, 'reduce_to_bounding_box',
                                 autospec=True) as mkData:
                evaluation.reduce_to_bounding_box(valid_index, all_out)
                assert evaluation.inputs == mkInputs.return_value
                assert evaluation.data == mkData.return_value
                assert mkInputs.call_args_list == \
                    [mk.call(inputs, valid_index, all_out)]
                assert mkData.call_args_list == \
                    [mk.call(data, valid_index, all_out)]


class TestOutputEntry:
    def test___init__(self):
        entry = evaluation_io.OutputEntry('name', 1, 2)
        assert entry._name == 'name'
        assert entry._value == np.asanyarray(1)
        assert entry._index == 2

        entry = evaluation_io.OutputEntry('name', 1, 2)
        assert entry._name == 'name'
        assert entry._value == np.asanyarray(1)
        assert entry._index == 2

    def test_index(self):
        entry = evaluation_io.OutputEntry('name', 1, 2)

        # Test get
        assert entry._index == 2
        assert entry.index == 2

        # Test set
        entry.index = 5
        assert entry._index == 5
        assert entry.index == 5

    def test_scalar(self):
        entry = evaluation_io.OutputEntry('name', 1, 2)

        # Test get already scalar
        assert entry.scalar == 1

        # Test turn to scalar
        entry.value = [2, 3, 4]
        assert (entry.scalar == np.array([2, 3, 4])).all()

    def test__new_output(self):
        entry = evaluation_io.OutputEntry('name', 1, 2)
        broadcast_shape = mk.MagicMock()

        value = mk.MagicMock()
        reshape = mk.MagicMock()
        value.reshape.side_effect = [reshape, ValueError('Test')]
        with mk.patch.object(evaluation_io.OutputEntry, 'scalar',
                             new_callable=mk.PropertyMock) as mkScalar:
            with mk.patch.object(evaluation_io.OutputEntry, 'value',
                                 new_callable=mk.PropertyMock) as mkValue:
                mkValue.return_value = value

                # No broadcast_info
                assert entry._new_output(()) == mkScalar.return_value
                assert mkScalar.call_args_list == [mk.call()]
                assert mkValue.call_args_list == []
                mkScalar.reset_mock()

                # Broadcast info and reshape
                assert entry._new_output(broadcast_shape) == reshape
                assert mkScalar.call_args_list == []
                assert mkValue.call_args_list == [mk.call()]
                assert value.reshape.call_args_list == [mk.call(broadcast_shape)]
                mkValue.reset_mock()
                value.reset_mock()

                # Broadcast info and no reshape
                assert entry._new_output(broadcast_shape) == mkScalar.return_value
                assert mkScalar.call_args_list == [mk.call()]
                assert mkValue.call_args_list == [mk.call()]
                assert value.reshape.call_args_list == [mk.call(broadcast_shape)]

    def test__check_broadcast(self):
        entry = evaluation_io.OutputEntry('name', 1, 0)
        format_info = (mk.MagicMock(), mk.MagicMock())

        effects = [mk.MagicMock(), IndexError('test'), TypeError('test')]
        with mk.patch.object(evaluation_io, 'check_broadcast', autospec=True,
                             side_effect=effects) as mkCheck:
            # No issue
            assert entry._check_broadcast(format_info) == effects[0]
            assert mkCheck.call_args_list == [mk.call(*format_info)]
            mkCheck.reset_mock()

            # IndexError
            assert entry._check_broadcast(format_info) == format_info[0]
            assert mkCheck.call_args_list == [mk.call(*format_info)]
            mkCheck.reset_mock()

            # ValueError
            assert entry._check_broadcast(format_info) == format_info[0]
            assert mkCheck.call_args_list == [mk.call(*format_info)]
            mkCheck.reset_mock()

    def test_prepare_output_single_model(self):
        entry = evaluation_io.OutputEntry('name', 1, 0)
        format_info = (mk.MagicMock(), mk.MagicMock())

        broadcast_shapes = [None, mk.MagicMock()]
        with mk.patch.object(evaluation_io.OutputEntry, '_check_broadcast',
                             autospec=True, side_effect=broadcast_shapes) as mkCheck:
            with mk.patch.object(evaluation_io.OutputEntry, '_new_output',
                                 autospec=True) as mkNew:
                # No broadcast_shape
                assert entry.prepare_output_single_model(format_info) == entry
                assert mkCheck.call_args_list == [mk.call(entry, format_info)]
                assert mkNew.call_args_list == []
                mkCheck.reset_mock()

                # Has broadcast_shape
                assert entry.prepare_output_single_model(format_info) == \
                    evaluation_io.OutputEntry('name', mkNew.return_value, 0)
                assert mkCheck.call_args_list == [mk.call(entry, format_info)]
                assert mkNew.call_args_list == [mk.call(entry, broadcast_shapes[1])]

    def test_prepare_output_model_set(self):
        entry = evaluation_io.OutputEntry('name', mk.MagicMock(), 0)
        pivots = [mk.MagicMock(), mk.MagicMock()]
        model_set_axis = mk.MagicMock()

        lt_effects = [True, True, False]
        ne_effects = [True, False]
        pivots[0].__lt__.side_effect = lt_effects
        pivots[0].__ne__.side_effect = ne_effects
        with mk.patch.object(np, 'rollaxis', autospec=True) as mkRoll:
            with mk.patch.object(evaluation_io.OutputEntry, 'value',
                                 new_callable=mk.PropertyMock) as mkValue:
                # Make change (pivot < ndim and pivot != model_set_axis)
                new_entry = entry.prepare_output_model_set(pivots, model_set_axis)
                assert isinstance(new_entry, evaluation_io.OutputEntry)
                assert new_entry._name == "name"
                assert new_entry._index == 0
                assert mkValue.call_args_list == \
                    [mk.call(), mk.call(), mk.call(mkRoll.return_value)]
                assert mkRoll.call_args_list == \
                    [mk.call(mkValue.return_value, pivots[0], model_set_axis)]
                assert pivots[0].__lt__.call_args_list == [mk.call(mkValue.return_value.ndim)]
                assert pivots[0].__ne__.call_args_list == [mk.call(model_set_axis)]
                mkRoll.reset_mock()
                mkValue.reset_mock()
                pivots[0].reset_mock()

                # No change (pivot < ndim and pivot == model_set_axis)
                new_entry = entry.prepare_output_model_set(pivots, model_set_axis)
                assert id(new_entry) == id(entry)
                assert mkValue.call_args_list == [mk.call()]
                assert mkRoll.call_args_list == []
                assert pivots[0].__lt__.call_args_list == [mk.call(mkValue.return_value.ndim)]
                assert pivots[0].__ne__.call_args_list == [mk.call(model_set_axis)]
                mkValue.reset_mock()
                pivots[0].reset_mock()

                # No change (pivot >= ndim)
                new_entry = entry.prepare_output_model_set(pivots, model_set_axis)
                assert id(new_entry) == id(entry)
                assert mkValue.call_args_list == [mk.call()]
                assert mkRoll.call_args_list == []
                assert pivots[0].__lt__.call_args_list == [mk.call(mkValue.return_value.ndim)]
                assert pivots[0].__ne__.call_args_list == []

    def test_prepare_input(self):
        entry = evaluation_io.OutputEntry('name', 1, 0)
        assert entry.prepare_input() == evaluation_io.InputEntry('name', 1)


class TestOutputs:
    def test___init__(self):
        entries = {f"z{idx}": evaluation_io.OutputEntry(f"z{idx}", mk.MagicMock(), idx)
                   for idx in range(3)}
        outputs = evaluation_io.Outputs(entries)
        assert outputs._outputs == entries

    def test_n_outputs(self):
        for index in range(5):
            entries = {f"z{idx}": evaluation_io.OutputEntry(f"z{idx}", mk.MagicMock(), idx)
                       for idx in range(index)}
            outputs = evaluation_io.Outputs(entries)
            assert outputs.n_outputs == index

    def test_ouputs(self):
        entries = {f"z{idx}": evaluation_io.OutputEntry(f"z{idx}", mk.MagicMock(), idx)
                   for idx in range(3)}
        outputs = evaluation_io.Outputs(entries.copy())

        # Test get
        assert outputs.outputs == entries
        assert outputs._outputs == entries

        new_entries = {f"zz{idx}": evaluation_io.OutputEntry(f"zz{idx}", mk.MagicMock(), idx)
                       for idx in range(3)}
        outputs.outputs = new_entries
        assert new_entries != entries
        assert outputs.outputs == new_entries
        assert outputs._outputs == new_entries

    def test_prepare_outputs_single_model(self):
        entries = {f"z{idx}": evaluation_io.OutputEntry(f"z{idx}", mk.MagicMock(), idx)
                   for idx in range(3)}
        outputs = evaluation_io.Outputs(entries)
        format_info = mk.MagicMock()

        new_output = [mk.MagicMock() for _ in range(3)]
        with mk.patch.object(evaluation_io.OutputEntry, 'prepare_output_single_model',
                             autospec=True, side_effect=new_output) as mkPrepare:
            new_outputs = outputs.prepare_outputs_single_model(format_info)
            assert isinstance(new_outputs, evaluation_io.Outputs)
            for index, (name, entry) in enumerate(entries.items()):
                assert name in new_outputs.outputs
                assert new_outputs.outputs[name] == new_output[index]
                assert mkPrepare.call_args_list[index] == mk.call(entry, format_info)
            assert len(mkPrepare.call_args_list) == len(entries)

    def test_prepare_outputs_model_set(self):
        entries = {f"z{idx}": evaluation_io.OutputEntry(f"z{idx}", mk.MagicMock(), idx)
                   for idx in range(3)}
        outputs = evaluation_io.Outputs(entries)
        pivots = mk.MagicMock()
        model_set_axis = mk.MagicMock()

        new_output = [mk.MagicMock() for _ in range(3)]
        with mk.patch.object(evaluation_io.OutputEntry, 'prepare_output_model_set',
                             autospec=True, side_effect=new_output) as mkPrepare:
            new_outputs = outputs.prepare_outputs_model_set(pivots, model_set_axis)
            assert isinstance(new_outputs, evaluation_io.Outputs)
            for index, (name, entry) in enumerate(entries.items()):
                assert name in new_outputs.outputs
                assert new_outputs.outputs[name] == new_output[index]
                assert mkPrepare.call_args_list[index] == mk.call(entry, pivots, model_set_axis)
            assert len(mkPrepare.call_args_list) == len(entries)


class TestIoMetaDataEntry:
    def test___init__(self):
        entry = evaluation_io.IoMetaDataEntry()
        assert entry._data_entry is None
        assert entry._name is None

        entry = evaluation_io.IoMetaDataEntry('test')
        assert entry._data_entry is None
        assert entry._name == 'test'

    def test_name(self):
        # test get error
        entry = evaluation_io.IoMetaDataEntry()
        with pytest.raises(RuntimeError):
            entry.name

        # test set and get without error
        entry.name = 'test'
        assert entry.name == 'test'
        assert entry._name == 'test'
        entry = evaluation_io.IoMetaDataEntry('test')
        assert entry.name == 'test'
        assert entry._name == 'test'

        # test set with error
        with pytest.raises(ValueError):
            entry.name = 'new_test'
        assert entry.name == 'test'
        assert entry._name == 'test'

    def test_create_entry(self):
        with pytest.raises(NotImplementedError):
            evaluation_io.IoMetaDataEntry.create_entry(mk.MagicMock(), test=mk.MagicMock())

    def test__create_io_entry(self):
        entry = evaluation_io.IoMetaDataEntry('test')
        value = mk.MagicMock()

        # No data_entry specified
        assert entry._create_io_entry(value) == value

        # data_entry specified, not a data_entry type
        data_entry = evaluation_io.InputEntry
        entry._data_entry = data_entry
        assert entry._create_io_entry(value) == evaluation_io.InputEntry('test', value)

        # data_entry specified, data_entry type
        value = evaluation_io.InputEntry(mk.MagicMock(), mk.MagicMock())
        assert entry._create_io_entry(value) == value

    def test_get_from_kwargs(self):
        entry = evaluation_io.IoMetaDataEntry('test')

        with mk.patch.object(evaluation_io.IoMetaDataEntry, '_create_io_entry',
                             autospec=True) as mkCreate:
            # Has entry
            data_kwargs = {}
            kwargs = {'test': mk.MagicMock(), 'other': mk.MagicMock()}
            new_kwargs = entry.get_from_kwargs(data_kwargs, **kwargs)
            assert data_kwargs == {'test': mkCreate.return_value}
            assert mkCreate.call_args_list == [mk.call(entry, kwargs['test'])]
            assert new_kwargs == {'other': kwargs['other']}
            mkCreate.reset_mock()

            # No entry
            data_kwargs = {}
            kwargs = {'other': mk.MagicMock()}
            new_kwargs = entry.get_from_kwargs(data_kwargs, **kwargs)
            assert data_kwargs == {}
            assert mkCreate.call_args_list == []
            assert new_kwargs == kwargs


class TestIoMetaData:
    def test___init__(self):
        meta_data = evaluation_io.IoMetaData()
        assert meta_data._data == {}
        assert meta_data._data_entry is None

        data = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        meta_data = evaluation_io.IoMetaData(data)
        assert meta_data._data == data
        assert meta_data._data_entry is None

        with mk.patch.object(evaluation_io.IoMetaData, 'data',
                             new_callable=mk.PropertyMock) as mkData:
            data = mk.MagicMock()
            meta_data = evaluation_io.IoMetaData(data)
            assert meta_data._data == {}
            assert mkData.call_args_list == [mk.call(data)]
            assert meta_data._data_entry is None

    def test__fill_defaults(self):
        meta_data = evaluation_io.IoMetaData()

        with pytest.raises(NotImplementedError):
            meta_data._fill_defaults(mk.MagicMock())

    def test_create_defaults(self):
        n_data = mk.MagicMock()
        with mk.patch.object(evaluation_io.IoMetaData, '_fill_defaults',
                             autospec=True) as mkFill:
            meta_data = evaluation_io.IoMetaData.create_defaults(n_data)
            assert isinstance(meta_data, evaluation_io.IoMetaData)
            assert mkFill.call_args_list == [mk.call(meta_data, n_data)]

    def test_validate(self):
        meta_data = evaluation_io.IoMetaData()

        with pytest.raises(NotImplementedError):
            meta_data.validate(mk.MagicMock(), mk.MagicMock(),
                               test1=mk.MagicMock(), test2=mk.MagicMock())

    def test__process_meta_data(self):
        meta_data = evaluation_io.IoMetaData()

        # No data entry
        value = mk.MagicMock()
        assert meta_data._process_meta_data(value) == value

        # Create a data entry
        data_entry = mk.MagicMock()
        meta_data._data_entry = data_entry

        # None input
        assert meta_data._process_meta_data(None) == {}

        # List input
        value = [mk.MagicMock(), mk.MagicMock()]
        entries = [mk.MagicMock(), mk.MagicMock()]
        data_entry.create_entry.side_effect = entries

        data = meta_data._process_meta_data(value)
        assert isinstance(data, dict)
        assert len(data) == 2
        assert len(data_entry.create_entry.call_args_list) == 2
        for index, (name, entry) in enumerate(data.items()):
            assert name == entries[index].name
            assert entry == entries[index]
            assert data_entry.create_entry.call_args_list[index] == \
                mk.call(value[index], pos=index)
        data_entry.reset_mock()

        # Dict input
        value = {'entry0': mk.MagicMock(), 'entry1': mk.MagicMock()}
        entries = [mk.MagicMock(), mk.MagicMock()]
        data_entry.create_entry.side_effect = entries

        data = meta_data._process_meta_data(value)
        assert isinstance(data, dict)
        assert len(data) == 2
        assert len(data_entry.create_entry.call_args_list) == 2
        for index, (name, entry) in enumerate(data.items()):
            assert name == entries[index].name
            assert entry == entries[index]
        for index, (name, entry) in enumerate(value.items()):
            assert data_entry.create_entry.call_args_list[index] == \
                mk.call(entry, name=name)

        # Other input
        with pytest.raises(ValueError):
            meta_data._process_meta_data(mk.MagicMock())

    def test_reset_data(self):
        data = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        meta_data = evaluation_io.IoMetaData(data)
        assert meta_data._data == data != {}
        meta_data.reset_data()
        assert meta_data._data == {}

    def test_data(self):
        data = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        meta_data = evaluation_io.IoMetaData(data)

        # Test Get
        assert meta_data.data == data
        assert meta_data._data == data

        # Test Set and Get
        new_data = mk.MagicMock()
        with mk.patch.object(evaluation_io.IoMetaData, '_process_meta_data',
                             autospec=True) as mkProcess:
            meta_data.data = new_data
        assert meta_data.data == mkProcess.return_value
        assert meta_data._data == mkProcess.return_value
        assert mkProcess.call_args_list == [mk.call(meta_data, new_data)]

    def test_get_inputs_from_kwargs(self):
        data = {f'x{idx}': evaluation_io.IoMetaDataEntry(f'x{idx}') for idx in range(3)}
        meta_data = evaluation_io.IoMetaData(data)
        kwargs = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        kwargs['other'] = mk.MagicMock()

        data_kwargs, new_kwargs = meta_data.get_from_kwargs(**kwargs)
        assert data_kwargs == {f'x{idx}': kwargs[f'x{idx}'] for idx in range(3)}
        assert new_kwargs == {'other': kwargs['other']}
        assert len(kwargs) == 4


class TestInputMetaDataEntry:
    def test___init__(self):
        entry = evaluation_io.InputMetaDataEntry()
        assert entry._name is None
        assert entry._pos is None
        assert entry._bounding_box is None

        entry = evaluation_io.InputMetaDataEntry('test', 1, _BoundingBox((-1, 1)))
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box == _BoundingBox((-1, 1))

    def test_pos(self):
        # test get error
        entry = evaluation_io.InputMetaDataEntry()
        with pytest.raises(RuntimeError):
            entry.pos

        # test set and get without error
        entry.pos = 1
        assert entry.pos == 1
        assert entry._pos == 1
        entry = evaluation_io.InputMetaDataEntry('test', 1)
        assert entry.pos == 1
        assert entry._pos == 1

        # test set with error
        with pytest.raises(ValueError):
            entry.pos = 2
        assert entry.pos == 1
        assert entry._pos == 1

    def test_bounding_box(self):
        # test get error
        entry = evaluation_io.InputMetaDataEntry()
        with pytest.raises(NotImplementedError):
            entry.bounding_box

        # test set and get without error
        value = mk.MagicMock()
        bbox = mk.MagicMock()
        with mk.patch.object(evaluation_io, '_BoundingBox', autospec=True,
                             return_value=bbox) as mkBbox:
            entry.bounding_box = value
            assert entry.bounding_box == bbox
            assert entry._bounding_box == bbox
            assert mkBbox.call_args_list == [mk.call(value)]

        # test set as None
        entry.bounding_box = None
        assert entry._bounding_box is None
        with pytest.raises(NotImplementedError):
            entry.bounding_box

    def test_create_entry(self):
        # test pass input metadata entry in
        input_value = evaluation_io.InputMetaDataEntry()
        entry = evaluation_io.InputMetaDataEntry.create_entry(input_value)
        assert entry == input_value
        entry = evaluation_io.InputMetaDataEntry.create_entry(input_value, name='test')
        assert entry == input_value
        entry = evaluation_io.InputMetaDataEntry.create_entry(input_value, pos=1)
        assert entry == input_value
        entry = evaluation_io.InputMetaDataEntry.create_entry(input_value, name='test', pos=1)
        assert entry == input_value

        # test pass tuple in
        entry = evaluation_io.InputMetaDataEntry.create_entry((1, (-1, 1)))
        assert entry._name is None
        assert entry._pos == 1
        assert entry._bounding_box == (-1, 1)
        entry = evaluation_io.InputMetaDataEntry.create_entry((1, (-1, 1)), name='test')
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box == (-1, 1)
        entry = evaluation_io.InputMetaDataEntry.create_entry((1, (-1, 1)), name='test', pos=2)
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box == (-1, 1)
        entry = evaluation_io.InputMetaDataEntry.create_entry((1,))
        assert entry._name is None
        assert entry._pos == 1
        assert entry._bounding_box is None
        entry = evaluation_io.InputMetaDataEntry.create_entry((1,), name='test')
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box is None
        entry = evaluation_io.InputMetaDataEntry.create_entry((1,), name='test', pos=2)
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box is None

        # test pass str in
        entry = evaluation_io.InputMetaDataEntry.create_entry('test')
        assert entry._name == 'test'
        assert entry._pos is None
        assert entry._bounding_box is None
        entry = evaluation_io.InputMetaDataEntry.create_entry('test', name=mk.MagicMock())
        assert entry._name == 'test'
        assert entry._pos is None
        assert entry._bounding_box is None
        entry = evaluation_io.InputMetaDataEntry.create_entry('test', name=mk.MagicMock(), pos=1)
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box is None

        # test pass bad input
        with pytest.raises(ValueError):
            evaluation_io.InputMetaDataEntry.create_entry(mk.MagicMock())

    def test_create_input(self):
        entry = evaluation_io.InputMetaDataEntry('test', 1)

        base_args = [mk.MagicMock(), mk.MagicMock()]
        base_kwargs = {'thing': mk.MagicMock()}
        with mk.patch.object(evaluation_io.InputMetaDataEntry, '_create_io_entry',
                             autospec=True) as mkCreate:
            # Is a kwarg
            args = tuple(base_args)
            kwargs = base_kwargs.copy()
            kwargs['test'] = mk.MagicMock()
            assert entry.create_input(*args, **kwargs) == \
                (mkCreate.return_value, args)
            assert mkCreate.call_args_list == \
                [mk.call(entry, kwargs['test'])]
            mkCreate.reset_mock()

            # Is not kwarg
            args = tuple(base_args)
            kwargs = base_kwargs.copy()
            assert entry.create_input(*args, **kwargs) == \
                (mkCreate.return_value, (base_args[1],))
            assert mkCreate.call_args_list == \
                [mk.call(entry, base_args[0])]

    def test__outside(self):
        entries = {f'x{idx}': evaluation_io.InputEntry(f'x{idx}', mk.MagicMock()) for idx in range(3)}
        inputs = evaluation_io.Inputs(entries)
        evaluation = evaluation_io.EvaluationInputs(inputs, mk.MagicMock(), mk.MagicMock())

        with mk.patch.object(evaluation_io.InputMetaDataEntry, 'bounding_box',
                             new_callable=mk.PropertyMock) as mkBbox:
            # Entry is not in inputs
            entry = evaluation_io.InputMetaDataEntry(mk.MagicMock())
            with pytest.raises(RuntimeError):
                entry._outside(evaluation)
            assert mkBbox.call_args_list == [mk.call(None)]
            mkBbox.reset_mock()

            for idx in range(3):
                print(idx)
                entry = evaluation_io.InputMetaDataEntry(f'x{idx}')
                outside, shape = entry._outside(evaluation)
                assert outside == mkBbox.return_value.outside.return_value
                assert shape == entries[f'x{idx}'].input_array.shape
                assert mkBbox.return_value.outside.call_args_list == \
                    [mk.call(entries[f'x{idx}'].input_array)]
                assert mkBbox.call_args_list == [mk.call(None), mk.call()]
                mkBbox.reset_mock()

    def test_update_outside(self):
        entry = evaluation_io.InputMetaDataEntry()
        entries = {f'x{idx}': evaluation_io.InputEntry(f'x{idx}', mk.MagicMock()) for idx in range(3)}
        inputs = evaluation_io.Inputs(entries)
        evaluation = evaluation_io.EvaluationInputs(inputs, mk.MagicMock(), mk.MagicMock())
        outside = np.zeros((5,), dtype=bool)

        # Array Inputs
        return_data = [np.array([True, False, False, True, True]), (1,)]
        with mk.patch.object(evaluation_io.InputMetaDataEntry, '_outside',
                             autospec=True, return_value=return_data) as mkOutside:
            values, all_out = entry.update_outside(outside, False, evaluation)
            assert (values == return_data[0]).all()
            assert not all_out
            assert mkOutside.call_args_list == [mk.call(entry, evaluation)]
            mkOutside.reset_mock()
            assert not outside.any()

            values, all_out = entry.update_outside(values, False, evaluation)
            assert (values == return_data[0]).all()
            assert not all_out
            assert mkOutside.call_args_list == [mk.call(entry, evaluation)]

        # Scalar input
        return_data = [np.array([True, False, False, True, True]), ()]
        with mk.patch.object(evaluation_io.InputMetaDataEntry, '_outside',
                             autospec=True, return_value=return_data) as mkOutside:
            values, all_out = entry.update_outside(outside, False, evaluation)
            assert (values == return_data[0]).all()
            assert not all_out
            assert mkOutside.call_args_list == [mk.call(entry, evaluation)]
            mkOutside.reset_mock()
            assert not outside.any()

            values, all_out = entry.update_outside(values, False, evaluation)
            assert (values == return_data[0]).all()
            assert not all_out
            assert mkOutside.call_args_list == [mk.call(entry, evaluation)]

        # All out
        return_data = [np.asanyarray(True), ()]
        with mk.patch.object(evaluation_io.InputMetaDataEntry, '_outside',
                             autospec=True, return_value=return_data) as mkOutside:
            values, all_out = entry.update_outside(outside, False, evaluation)
            assert values.all()
            assert all_out
            assert mkOutside.call_args_list == [mk.call(entry, evaluation)]
            mkOutside.reset_mock()
            assert not outside.any()

            values, all_out = entry.update_outside(outside, False, evaluation)
            assert values.all()
            assert all_out
            assert mkOutside.call_args_list == [mk.call(entry, evaluation)]


class TestInputMetaData:
    def test___init__(self):
        meta_data = evaluation_io.InputMetaData(3)
        assert meta_data._n_inputs == 3
        assert meta_data._data == {}
        assert meta_data._bounding_box is None

        inputs = mk.MagicMock()
        bbox = mk.MagicMock()
        with mk.patch.object(evaluation_io.IoMetaData, '__init__', autospec=True) as mkInit:
            with mk.patch.object(evaluation_io.InputMetaData, 'bounding_box',
                                 new_callable=mk.PropertyMock) as mkBbox:
                meta_data = evaluation_io.InputMetaData(3, inputs, bbox)
                meta_data._n_inputs = 3
                meta_data._data = mkInit.return_value
                mkInit.call_args_list == [mk.call(meta_data, inputs)]
                mkBbox.call_args_list == [mk.call(bbox)]

    def test__fill_defaults(self):
        # n_inputs = 0
        meta_data = evaluation_io.InputMetaData(0)
        meta_data._fill_defaults(0)
        assert meta_data._n_inputs == 0 == len(meta_data._data)

        # n_inputs = 1
        meta_data = evaluation_io.InputMetaData(1)
        meta_data._fill_defaults(1)
        assert meta_data._n_inputs == 1 == len(meta_data._data)
        assert 'x' in meta_data._data
        _input = meta_data._data['x']
        assert isinstance(_input, evaluation_io.InputMetaDataEntry)
        assert _input._name == 'x'
        assert _input._pos == 0
        assert _input._bounding_box is None

        # n_inputs = 2
        meta_data = evaluation_io.InputMetaData(2)
        meta_data._fill_defaults(2)
        assert meta_data._n_inputs == 2 == len(meta_data._data)
        assert 'x' in meta_data._data
        _input = meta_data._data['x']
        assert isinstance(_input, evaluation_io.InputMetaDataEntry)
        assert _input._name == 'x'
        assert _input._pos == 0
        assert _input._bounding_box is None
        assert 'y' in meta_data._data
        _input = meta_data._data['y']
        assert isinstance(_input, evaluation_io.InputMetaDataEntry)
        assert _input._name == 'y'
        assert _input._pos == 1
        assert _input._bounding_box is None

        # n_inputs >= 3
        for idx in range(3, 5):
            meta_data = evaluation_io.InputMetaData(idx)
            meta_data._fill_defaults(idx)
            assert meta_data._n_inputs == idx == len(meta_data._data)
            for index, (name, _input) in enumerate(meta_data._data.items()):
                assert name == f'x{index}'
                assert isinstance(_input, evaluation_io.InputMetaDataEntry)
                assert _input._name == name
                assert _input._pos == index
                assert _input._bounding_box is None

    def test_n_inputs(self):
        n_inputs = mk.MagicMock()
        meta_data = evaluation_io.InputMetaData(n_inputs)
        assert meta_data._data == {}

        # Test get
        assert meta_data._n_inputs == n_inputs
        assert meta_data.n_inputs == n_inputs

        value = mk.MagicMock()
        with mk.patch.object(evaluation_io.InputMetaData, 'reset_data',
                             autospec=True) as mkReset:
            with mk.patch.object(evaluation_io.InputMetaData, '_fill_defaults',
                                 autospec=True) as mkFill:
                main = mk.MagicMock()
                main.attach_mock(mkReset, 'reset')
                main.attach_mock(mkFill, 'fill')

                meta_data.n_inputs = value
                assert meta_data._n_inputs == value
                assert meta_data.n_inputs == value
                assert main.mock_calls == \
                    [mk.call.reset(meta_data),
                     mk.call.fill(meta_data, value)]

        # Test Set 3
        meta_data.n_inputs = 3
        assert meta_data._n_inputs == 3
        assert meta_data.n_inputs == 3
        assert len(meta_data._data) == 3
        for index, (name, _input) in enumerate(meta_data._data.items()):
            assert name == f'x{index}'
            assert isinstance(_input, evaluation_io.InputMetaDataEntry)
            assert _input._name == name
            assert _input._pos == index
            assert _input._bounding_box is None

        # Test Set 4
        meta_data.n_inputs = 4
        assert meta_data._n_inputs == 4
        assert meta_data.n_inputs == 4
        assert len(meta_data._data) == 4
        for index, (name, _input) in enumerate(meta_data._data.items()):
            assert name == f'x{index}'
            assert isinstance(_input, evaluation_io.InputMetaDataEntry)
            assert _input._name == name
            assert _input._pos == index
            assert _input._bounding_box is None

    def test_validate(self):
        meta_data = evaluation_io.InputMetaData(1)

        # Fail inputs size
        meta_data._n_inputs = 2
        meta_data._data = {'x': mk.MagicMock()}
        with pytest.raises(ValueError, match=r"n_inputs .*"):
            meta_data.validate()
        meta_data._n_inputs = 1

        # Fail input name
        meta_data._data = {'x': evaluation_io.InputMetaDataEntry('y')}
        with pytest.raises(ValueError, match=r"Input: .* key .*"):
            meta_data.validate()

        # Fail input position
        meta_data._data = {'x': evaluation_io.InputMetaDataEntry('x', pos=1)}
        with pytest.raises(ValueError, match=r"Input: .* position .*"):
            meta_data.validate()

        # Fix fails
        meta_data._data = {'x': evaluation_io.InputMetaDataEntry('x', pos=0)}
        meta_data.validate()

    def test_inputs(self):
        meta_data = evaluation_io.InputMetaData.create_defaults(3)

        # Test get
        assert meta_data.data == meta_data.inputs
        for index, (name, _input) in enumerate(meta_data._data.items()):
            assert name == f'x{index}'
            assert isinstance(_input, evaluation_io.InputMetaDataEntry)
            assert _input._name == name
            assert _input._pos == index
            assert _input._bounding_box is None

        # Test set
        inputs = {f"new_x{idx}": evaluation_io.InputMetaDataEntry(f"new_x{idx}", idx) for idx in range(3)}
        meta_data.inputs = inputs
        assert meta_data.data == inputs
        assert meta_data.inputs == inputs

    def test_bounding_box_get(self):
        inputs = evaluation_io.InputMetaData(1)

        # Test fail to get
        assert inputs._bounding_box is None
        with pytest.raises(NotImplementedError):
            inputs.bounding_box

        # Test get
        bbox = mk.MagicMock()
        inputs._bounding_box = bbox
        assert inputs.bounding_box == bbox

    def test__reverse_bounding_box(self):
        inputs = evaluation_io.InputMetaData(1)

        # Test when n_inputs == 1
        assert inputs.n_inputs == 1
        bbox = mk.MagicMock()
        inputs._bounding_box = bbox
        assert inputs._reverse_bounding_box() == [bbox]

        # Test when n_inputs > 1
        for index in range(2, 4):
            inputs.reset_data()
            inputs.n_inputs = index
            assert inputs.n_inputs == index > 1
            bbox = [mk.MagicMock() for _ in range(index)]
            inputs._bounding_box = bbox
            assert inputs._reverse_bounding_box() == bbox[::-1]

    def test__distribute_bounding_box(self):
        # n_inputs == 1
        input_data = evaluation_io.InputMetaData.create_defaults(1)
        bbox = (mk.MagicMock(), mk.MagicMock())
        input_data._bounding_box = bbox
        input_data._distribute_bounding_box()
        assert input_data._data['x'].bounding_box == bbox

        # n_inputs > 1
        for index in range(2, 4):
            input_data = evaluation_io.InputMetaData.create_defaults(index)
            bbox = [(mk.MagicMock(), mk.MagicMock()) for _ in range(index)]
            input_data._bounding_box = bbox
            input_data._distribute_bounding_box()
            for _input in input_data._data.values():
                assert _input.bounding_box == bbox[index - 1 - _input.pos]

    def test_bounding_box_set(self):
        input_data = evaluation_io.InputMetaData.create_defaults(1)
        bbox = (mk.MagicMock(), mk.MagicMock())

        # Set an actual box 1D
        input_data.bounding_box = bbox
        assert input_data._data['x'].bounding_box == bbox
        assert input_data.bounding_box == bbox

        # try to set a box
        input_data.bounding_box = None
        assert input_data._bounding_box is None

        input_data.bounding_box = bbox
        input_data.bounding_box = NotImplemented
        assert input_data._bounding_box is NotImplemented

        # Set an actual box > 1D
        for index in range(2, 4):
            input_data = evaluation_io.InputMetaData.create_defaults(index)
            bbox = [(mk.MagicMock(), mk.MagicMock()) for _ in range(index)]
            input_data.bounding_box = bbox
            for _input in input_data._data.values():
                assert _input.bounding_box == bbox[index - 1 - _input.pos]

    def test__check_inputs(self):
        meta_data = evaluation_io.InputMetaData.create_defaults(3)
        assert meta_data._n_inputs == 3

        # Too many args
        with pytest.raises(ValueError, match=r"Too many .*"):
            meta_data._check_inputs(1, 2, 3, 4)
        with pytest.raises(ValueError, match=r"Too many .*"):
            meta_data._check_inputs(1, 2, 3, a=4)
        with pytest.raises(ValueError, match=r"Too many .*"):
            meta_data._check_inputs(1, 2, b=3, a=4)
        with pytest.raises(ValueError, match=r"Too many .*"):
            meta_data._check_inputs(1, c=2, b=3, a=4)
        with pytest.raises(ValueError, match=r"Too many .*"):
            meta_data._check_inputs(d=1, c=2, b=3, a=4)

        # Too few args
        with pytest.raises(ValueError, match=r"Too few .*"):
            meta_data._check_inputs()
        with pytest.raises(ValueError, match=r"Too few .*"):
            meta_data._check_inputs(1)
        with pytest.raises(ValueError, match=r"Too few .*"):
            meta_data._check_inputs(1, 2)
        with pytest.raises(ValueError, match=r"Too few .*"):
            meta_data._check_inputs(1, a=2)
        with pytest.raises(ValueError, match=r"Too few .*"):
            meta_data._check_inputs(b=1, a=2)

    def test__create_inputs(self):
        meta_data = evaluation_io.InputMetaData.create_defaults(3)
        true_inputs = \
            evaluation_io.Inputs({f'x{idx}': evaluation_io.InputEntry(f'x{idx}', idx)
                                  for idx in range(3)})

        assert true_inputs == meta_data._create_inputs(0, 1, 2)
        assert true_inputs == meta_data._create_inputs(0, 1, x2=2)
        assert true_inputs == meta_data._create_inputs(0, 2, x1=1)
        assert true_inputs == meta_data._create_inputs(1, 2, x0=0)
        assert true_inputs == meta_data._create_inputs(0, x1=1, x2=2)
        assert true_inputs == meta_data._create_inputs(1, x0=0, x2=2)
        assert true_inputs == meta_data._create_inputs(2, x0=0, x1=1)
        assert true_inputs == meta_data._create_inputs(x0=0, x1=1, x2=2)

    def test_get_inputs(self):
        meta_data = evaluation_io.InputMetaData.create_defaults(3)
        args = (mk.MagicMock(), mk.MagicMock())
        kwargs = {f'test{idx}': mk.MagicMock() for idx in range(3)}

        input_kwargs = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        new_kwargs = {f'new_test{idx}': mk.MagicMock() for idx in range(3)}
        get_return = (input_kwargs, new_kwargs)
        create_return = mk.MagicMock()
        with mk.patch.object(evaluation_io.InputMetaData, 'get_from_kwargs',
                             autospec=True, return_value=get_return) as mkGet:
            with mk.patch.object(evaluation_io.InputMetaData, '_check_inputs',
                                 autospec=True) as mkCheck:
                with mk.patch.object(evaluation_io.InputMetaData, '_create_inputs',
                                     autospec=True, return_value=create_return) as mkCreate:
                    main = mk.MagicMock()
                    main.attach_mock(mkGet, 'get')
                    main.attach_mock(mkCheck, 'check')
                    main.attach_mock(mkCreate, 'create')

                    model_inputs, model_kwargs = meta_data.get_inputs(*args, **kwargs)
                    assert model_inputs == create_return
                    assert model_kwargs == new_kwargs
                    assert main.mock_calls == [mk.call.get(meta_data, **kwargs),
                                               mk.call.check(meta_data, *args, **input_kwargs),
                                               mk.call.create(meta_data, *args, **input_kwargs)]

    def test__outside(self):
        meta_data = evaluation_io.InputMetaData.create_defaults(3)
        n_models = mk.MagicMock()
        model_set_axis = mk.MagicMock()
        inputs = mk.MagicMock()

        update_effects = [(mk.MagicMock(), mk.MagicMock()) for _ in range(3)]
        with mk.patch.object(evaluation_io.InputMetaDataEntry, 'update_outside',
                             autospec=True, side_effect=update_effects) as mkUpdate:
            with mk.patch.object(np, 'zeros', autospec=True) as mkZeros:
                outside_inputs, all_out = meta_data._outside(n_models, model_set_axis, inputs)
                assert outside_inputs == update_effects[2][0]
                assert all_out == update_effects[2][1]

                assert mkUpdate.call_args_list == \
                    [
                        mk.call(meta_data._data['x0'],
                                mkZeros.return_value, False, inputs),
                        mk.call(meta_data._data['x1'],
                                update_effects[0][0], update_effects[0][1], inputs),
                        mk.call(meta_data._data['x2'],
                                update_effects[1][0], update_effects[1][1], inputs),
                    ]
                assert mkZeros.call_args_list == \
                    [mk.call(inputs.inputs.check_input_shape.return_value, dtype=bool)]
                assert inputs.inputs.check_input_shape.call_args_list == \
                    [mk.call(n_models, model_set_axis, True)]

    def test__get_valid_index(self):
        meta_data = evaluation_io.InputMetaData.create_defaults(3)
        n_models = mk.MagicMock()
        model_set_axis = mk.MagicMock()
        inputs = mk.MagicMock()

        valid_index_options = [
            [[mk.MagicMock(), mk.MagicMock()], mk.MagicMock()],
            [[], mk.MagicMock()]
        ]

        outside_return = (mk.MagicMock(), mk.MagicMock())
        atleast_return = mk.MagicMock()
        atleast_return.nonzero.side_effect = valid_index_options
        with mk.patch.object(evaluation_io.InputMetaData, '_outside',
                             autospec=True, return_value=outside_return) as mkOutside:
            with mk.patch.object(np, 'logical_not', autospec=True) as mkNot:
                with mk.patch.object(np, 'atleast_1d', autospec=True,
                                     return_value=atleast_return) as mkAtleast:
                    # Do not update all_out
                    valid_index, all_out = \
                        meta_data._get_valid_index(n_models, model_set_axis, inputs)
                    assert valid_index == valid_index_options[0]
                    assert all_out == outside_return[1]
                    assert atleast_return.nonzero.call_args_list == [mk.call()]
                    assert mkAtleast.call_args_list == [mk.call(mkNot.return_value)]
                    assert mkNot.call_args_list == [mk.call(outside_return[0])]
                    assert mkOutside.call_args_list == [mk.call(meta_data, n_models,
                                                                model_set_axis, inputs)]
                    atleast_return.reset_mock()
                    mkAtleast.reset_mock()
                    mkNot.reset_mock()
                    mkOutside.reset_mock()

                    # Do update all_out
                    valid_index, all_out = \
                        meta_data._get_valid_index(n_models, model_set_axis, inputs)
                    assert valid_index == valid_index_options[1]
                    assert all_out == True
                    assert atleast_return.nonzero.call_args_list == [mk.call()]
                    assert mkAtleast.call_args_list == [mk.call(mkNot.return_value)]
                    assert mkNot.call_args_list == [mk.call(outside_return[0])]
                    assert mkOutside.call_args_list == [mk.call(meta_data, n_models,
                                                                model_set_axis, inputs)]

    def test_enforce_bounding_box(self):
        meta_data = evaluation_io.InputMetaData.create_defaults(3)
        n_models = mk.MagicMock()
        model_set_axis = mk.MagicMock()
        optional = evaluation_io.Optional({})
        inputs = evaluation_io.EvaluationInputs(mk.MagicMock(), optional, mk.MagicMock())

        get_return = (mk.MagicMock(), mk.MagicMock())
        effects = [True, False]
        with mk.patch.object(evaluation_io.InputMetaData, '_get_valid_index',
                             autospec=True, return_value=get_return) as mkGet:
            with mk.patch.object(evaluation_io.EvaluationInputs, 'reduce_to_bounding_box',
                                 autospec=True) as mkReduce:
                with mk.patch.object(evaluation_io.Optional, 'with_bounding_box',
                                     new_callable=mk.PropertyMock,
                                     side_effect=effects) as mkWith:
                    # Enforce the bounding box
                    meta_data.enforce_bounding_box(n_models, model_set_axis, inputs)
                    assert mkReduce.call_args_list == \
                        [mk.call(inputs, get_return[0], get_return[1])]
                    assert mkGet.call_args_list == [mk.call(meta_data, n_models,
                                                            model_set_axis, inputs)]
                    assert mkWith.call_args_list == [mk.call()]
                    mkReduce.reset_mock()
                    mkGet.reset_mock()
                    mkWith.reset_mock()

                    # Don't Enforce the bounding box
                    meta_data.enforce_bounding_box(n_models, model_set_axis, inputs)
                    assert mkReduce.call_args_list == []
                    assert mkGet.call_args_list == []
                    assert mkWith.call_args_list == [mk.call()]


class TestOptionalMetaDataEntry:
    def test___init__(self):
        entry = evaluation_io.OptionalMetaDataEntry()
        assert entry._name is None
        assert entry._default is None

        entry = evaluation_io.OptionalMetaDataEntry('test', 1)
        assert entry._name == 'test'
        assert entry._default == 1

    def test_default(self):
        entry = evaluation_io.OptionalMetaDataEntry()
        assert entry.default is None
        assert entry._default is None

        entry = evaluation_io.OptionalMetaDataEntry(default=1)
        assert entry.default == 1
        assert entry._default == 1

    def test_create_entry(self):
        # test pass input metadata entry in
        input_value = evaluation_io.OptionalMetaDataEntry()
        entry = evaluation_io.OptionalMetaDataEntry.create_entry(input_value)
        assert entry == input_value
        entry = evaluation_io.OptionalMetaDataEntry.create_entry(input_value, name='test')
        assert entry == input_value
        entry = evaluation_io.OptionalMetaDataEntry.create_entry(input_value, pos=1)
        assert entry == input_value
        entry = evaluation_io.OptionalMetaDataEntry.create_entry(input_value, name='test', pos=1)
        assert entry == input_value

        # test pass tuple in
        entry = evaluation_io.OptionalMetaDataEntry.create_entry((1,))
        assert entry._name is None
        assert entry._default == 1
        entry = evaluation_io.OptionalMetaDataEntry.create_entry((1,), name='test')
        assert entry._name == 'test'
        assert entry._default == 1
        entry = evaluation_io.OptionalMetaDataEntry.create_entry((1,), name='test', pos=2)
        assert entry._name == 'test'
        assert entry._default == 1

        # test pass str in
        entry = evaluation_io.OptionalMetaDataEntry.create_entry('test')
        assert entry._name == 'test'
        assert entry._default is None
        entry = evaluation_io.OptionalMetaDataEntry.create_entry('test', name=mk.MagicMock())
        assert entry._name == 'test'
        assert entry._default is None
        entry = evaluation_io.OptionalMetaDataEntry.create_entry('test', name=mk.MagicMock(), pos=1)
        assert entry._name == 'test'
        assert entry._default is None

        # test pass bad input
        with pytest.raises(ValueError):
            evaluation_io.OptionalMetaDataEntry.create_entry(mk.MagicMock())


class TestOptionalMetaData:
    def test___init__(self):
        # Default
        meta_data = evaluation_io.OptionalMetaData()
        assert meta_data._data == {}
        assert meta_data._pass_optional == False
        assert meta_data._model_set_axis == 0

        meta_data = evaluation_io.OptionalMetaData({'option': evaluation_io.OptionalMetaDataEntry('option', 1)}, True, 3)
        assert len(meta_data._data) == 1
        assert 'option' in meta_data._data
        assert isinstance(meta_data._data['option'], evaluation_io.OptionalMetaDataEntry)
        assert meta_data._data['option'].name == 'option'
        assert meta_data._data['option'].default == 1
        assert meta_data._pass_optional == True
        assert meta_data._model_set_axis == 3

    def test_pass_optional(self):
        meta_data = evaluation_io.OptionalMetaData()

        # Test Get
        assert meta_data._pass_optional == False
        assert meta_data.pass_optional == False

        # Test Set
        meta_data.pass_optional = True
        assert meta_data._pass_optional == True
        assert meta_data.pass_optional == True

    def test_model_set_axis(self):
        meta_data = evaluation_io.OptionalMetaData()

        # Test Get
        assert meta_data._model_set_axis == 0
        assert meta_data.model_set_axis == 0

        # Test Set
        meta_data.model_set_axis = 3
        assert meta_data._model_set_axis == 3
        assert meta_data.model_set_axis == 3

    def test__fill_defaults(self):
        meta_data = evaluation_io.OptionalMetaData()
        meta_data._fill_defaults(3)
        assert len(meta_data._data) == 3
        for index, (name, optional) in enumerate(meta_data._data.items()):
            assert name == f"optional_{index}"
            assert isinstance(optional, evaluation_io.OptionalMetaDataEntry)
            assert optional.name == name
            assert optional.default is None

    def test__get__model_options(self):
        meta_data = evaluation_io.OptionalMetaData.create_defaults(1)
        model_set_axis = mk.MagicMock()
        meta_data.model_set_axis = model_set_axis

        # No optionals and no kwargs
        input_kwargs = {}
        optional = {}

        true_optional = evaluation_io.Optional({}, model_set_axis=model_set_axis)
        assert meta_data._get_model_options(optional) == true_optional
        assert meta_data._get_model_options(optional, **input_kwargs) == \
            true_optional

        # Optional with no model options and no kwargs
        for index in range(3):
            key = f'z{index}'
            optional[key] = mk.MagicMock()
            true_optional = evaluation_io.Optional(optional, model_set_axis=model_set_axis)

            assert meta_data._get_model_options(optional) == true_optional
            assert meta_data._get_model_options(optional, **input_kwargs) == \
                true_optional

        true_options = optional.copy()
        true_model_options = evaluation_io.modeling_options.copy()
        # Optional with model options and no kwargs
        for index, key in enumerate(evaluation_io.modeling_options):
            optional[key] = mk.MagicMock()
            true_model_options[key] = optional[key]
            true_optional = evaluation_io.Optional(true_options,
                                                   true_model_options,
                                                   model_set_axis=model_set_axis)

            assert meta_data._get_model_options(optional) == true_optional
            assert len(optional) == index + 4
            assert meta_data._get_model_options(optional, **input_kwargs) == \
                true_optional
            assert len(optional) == index + 4

        # Optional with model options and disjoint kwargs
        for index in range(3):
            key = f'a{index}'
            input_kwargs[key] = mk.MagicMock
            true_optional = evaluation_io.Optional(true_options,
                                                   true_model_options,
                                                   input_kwargs,
                                                   model_set_axis=model_set_axis)

            assert meta_data._get_model_options(optional, **input_kwargs) == \
                true_optional
            assert len(optional) == 3 + len(evaluation_io.modeling_options)

        optional = true_options.copy()
        true_kwargs = input_kwargs.copy()
        true_model_options = evaluation_io.modeling_options.copy()
        # Optional and model options in kwargs
        for index, key in enumerate(evaluation_io.modeling_options):
            input_kwargs[key] = mk.MagicMock()
            true_model_options[key] = input_kwargs[key]
            true_optional = evaluation_io.Optional(true_options,
                                                   true_model_options,
                                                   true_kwargs,
                                                   model_set_axis=model_set_axis)

            assert meta_data._get_model_options(optional, **input_kwargs) == \
                true_optional
            assert len(optional) == 3

    def test_get_optional(self):
        meta_data = evaluation_io.OptionalMetaData.create_defaults(1)
        pass_optional = mk.MagicMock()
        meta_data.pass_optional = pass_optional
        kwargs = {'test': mk.MagicMock()}

        get_return = (mk.MagicMock(), {'other': mk.MagicMock()})
        options_return = evaluation_io.Optional(mk.MagicMock())
        with mk.patch.object(evaluation_io.OptionalMetaData, 'get_from_kwargs',
                             autospec=True, return_value=get_return) as mkGet:
            with mk.patch.object(evaluation_io.OptionalMetaData, '_get_model_options',
                                 autospec=True, return_value=options_return) as mkOptions:
                with mk.patch.object(evaluation_io.Optional, 'validate',
                                     autospec=True) as mkValidate:
                    assert meta_data.get_optional(**kwargs) == options_return
                    assert mkOptions.call_args_list == \
                        [mk.call(meta_data, get_return[0], **get_return[1])]
                    assert mkGet.call_args_list == \
                        [mk.call(meta_data, **kwargs)]
                    assert mkValidate.call_args_list == \
                        [mk.call(options_return, pass_optional)]


class TestOutputMetaDataEntry:
    def test___init__(self):
        entry = evaluation_io.OutputMetaDataEntry()
        assert entry._name is None
        assert entry._pos is None

        entry = evaluation_io.OutputMetaDataEntry('test', 1)
        assert entry._name == 'test'
        assert entry._pos == 1

    def test_pos(self):
        # test get error
        entry = evaluation_io.OutputMetaDataEntry()
        with pytest.raises(RuntimeError):
            entry.pos

        # test set and get without error
        entry.pos = 1
        assert entry.pos == 1
        assert entry._pos == 1
        entry = evaluation_io.OutputMetaDataEntry('test', 1)
        assert entry.pos == 1
        assert entry._pos == 1

        # test set with error
        with pytest.raises(ValueError):
            entry.pos = 2
        assert entry.pos == 1
        assert entry._pos == 1

    def test_create_entry(self):
        # test pass input metadata entry in
        input_value = evaluation_io.OutputMetaDataEntry()
        entry = evaluation_io.OutputMetaDataEntry.create_entry(input_value)
        assert entry == input_value
        entry = evaluation_io.OutputMetaDataEntry.create_entry(input_value, name='test')
        assert entry == input_value
        entry = evaluation_io.OutputMetaDataEntry.create_entry(input_value, pos=1)
        assert entry == input_value
        entry = evaluation_io.OutputMetaDataEntry.create_entry(input_value, name='test', pos=1)
        assert entry == input_value

        # test pass tuple in
        entry = evaluation_io.OutputMetaDataEntry.create_entry((1,))
        assert entry._name is None
        assert entry._pos == 1
        entry = evaluation_io.OutputMetaDataEntry.create_entry((1,), name='test')
        assert entry._name == 'test'
        assert entry._pos == 1
        entry = evaluation_io.OutputMetaDataEntry.create_entry((1,), name='test', pos=2)
        assert entry._name == 'test'
        assert entry._pos == 1

        # test pass str in
        entry = evaluation_io.OutputMetaDataEntry.create_entry('test')
        assert entry._name == 'test'
        assert entry._pos is None
        entry = evaluation_io.OutputMetaDataEntry.create_entry('test', name=mk.MagicMock())
        assert entry._name == 'test'
        assert entry._pos is None
        entry = evaluation_io.OutputMetaDataEntry.create_entry('test', name=mk.MagicMock(), pos=1)
        assert entry._name == 'test'
        assert entry._pos == 1

        # test pass bad input
        with pytest.raises(ValueError):
            evaluation_io.OutputMetaDataEntry.create_entry(mk.MagicMock())


class TestOutputMetaData:
    def test___init__(self):
        meta_data = evaluation_io.OutputMetaData(3)
        assert meta_data._n_outputs == 3
        assert meta_data._data == {}

        outputs = mk.MagicMock()
        with mk.patch.object(evaluation_io.IoMetaData, '__init__', autospec=True) as mkInit:
            meta_data = evaluation_io.OutputMetaData(3, outputs)
            meta_data._n_outputs = 3
            meta_data._data = mkInit.return_value
            mkInit.call_args_list == [mk.call(meta_data, outputs)]

    def test_n_outputs(self):
        n_outputs = mk.MagicMock()
        meta_data = evaluation_io.OutputMetaData(n_outputs)
        assert meta_data._data == {}

        # Test get
        assert meta_data._n_outputs == n_outputs
        assert meta_data.n_outputs == n_outputs

        value = mk.MagicMock()
        with mk.patch.object(evaluation_io.OutputMetaData, 'reset_data',
                             autospec=True) as mkReset:
            with mk.patch.object(evaluation_io.OutputMetaData, '_fill_defaults',
                                 autospec=True) as mkFill:
                main = mk.MagicMock()
                main.attach_mock(mkReset, 'reset')
                main.attach_mock(mkFill, 'fill')

                meta_data.n_outputs = value
                assert meta_data._n_outputs == value
                assert meta_data.n_outputs == value
                assert main.mock_calls == \
                    [mk.call.reset(meta_data),
                     mk.call.fill(meta_data, value)]

        # Test Set 3
        meta_data.n_outputs = 3
        assert meta_data._n_outputs == 3
        assert meta_data.n_outputs == 3
        assert len(meta_data._data) == 3
        for index, (name, _output) in enumerate(meta_data._data.items()):
            assert name == f'y{index}'
            assert isinstance(_output, evaluation_io.OutputMetaDataEntry)
            assert _output._name == name
            assert _output._pos == index

        # Test Set 4
        meta_data.n_outputs = 4
        assert meta_data._n_outputs == 4
        assert meta_data.n_outputs == 4
        assert len(meta_data._data) == 4
        for index, (name, _output) in enumerate(meta_data._data.items()):
            assert name == f'y{index}'
            assert isinstance(_output, evaluation_io.OutputMetaDataEntry)
            assert _output._name == name
            assert _output._pos == index

    def test_validate(self):
        meta_data = evaluation_io.OutputMetaData(1)

        # Fail outputs size
        meta_data._n_outputs = 2
        meta_data._data = {'x': mk.MagicMock()}
        with pytest.raises(ValueError, match=r"n_outputs .*"):
            meta_data.validate()
        meta_data._n_outputs = 1

        # Fail output name
        meta_data._data = {'x': evaluation_io.OutputMetaDataEntry('y')}
        with pytest.raises(ValueError, match=r"Output: .* key .*"):
            meta_data.validate()

        # Fail output position
        meta_data._data = {'x': evaluation_io.OutputMetaDataEntry('x', pos=1)}
        with pytest.raises(ValueError, match=r"Output: .* position .*"):
            meta_data.validate()

        # Fix fails
        meta_data._data = {'x': evaluation_io.OutputMetaDataEntry('x', pos=0)}
        meta_data.validate()

    def test_outputs(self):
        meta_data = evaluation_io.OutputMetaData.create_defaults(3)

        # Test get
        assert meta_data.data == meta_data.outputs
        for index, (name, _output) in enumerate(meta_data._data.items()):
            assert name == f'y{index}'
            assert isinstance(_output, evaluation_io.OutputMetaDataEntry)
            assert _output._name == name
            assert _output._pos == index

        # Test set
        outputs = {f"new_y{idx}": evaluation_io.OutputMetaDataEntry(f"new_y{idx}", idx) for idx in range(3)}
        meta_data.outputs = outputs
        assert meta_data.data == outputs
        assert meta_data.outputs == outputs


class TestMetaData:
    def test___init__(self):
        inputs = mk.MagicMock()
        optional = mk.MagicMock()
        outputs = mk.MagicMock()

        meta_data = evaluation_io.MetaData(inputs, optional, outputs)
        assert meta_data._inputs == inputs
        assert meta_data._optional == optional
        assert meta_data._outputs == outputs
        assert meta_data._n_models == 1
        assert meta_data._standard_broadcasting == True

        n_models = mk.MagicMock()
        standard_broadcasting = mk.MagicMock()

        meta_data = evaluation_io.MetaData(inputs, optional, outputs,
                                              n_models, standard_broadcasting)
        assert meta_data._inputs == inputs
        assert meta_data._optional == optional
        assert meta_data._n_models == n_models
        assert meta_data._standard_broadcasting == standard_broadcasting

    def test_create_defaults(self):
        n_inputs = mk.MagicMock()
        n_outputs = mk.MagicMock()
        n_models = mk.MagicMock()
        standard_broadcasting = mk.MagicMock()
        pass_optional = mk.MagicMock()
        model_set_axis = mk.MagicMock()
        optional = evaluation_io.OptionalMetaData.create_defaults(0)

        with mk.patch.object(evaluation_io.InputMetaData, 'create_defaults',
                             autospec=True) as mkInputs:
            with mk.patch.object(evaluation_io.OptionalMetaData, 'create_defaults',
                                 autospec=True) as mkOptional:
                with mk.patch.object(evaluation_io.OutputMetaData, 'create_defaults',
                                     autospec=True) as mkOutputs:
                    # Defaults
                    meta_data = evaluation_io.MetaData.create_defaults(n_inputs)
                    assert meta_data._inputs == mkInputs.return_value
                    assert meta_data._optional == mkOptional.return_value
                    assert meta_data._outputs == mkOutputs.return_value
                    assert meta_data._n_models == 1
                    assert meta_data._standard_broadcasting == True
                    assert mkInputs.call_args_list == [mk.call(n_inputs)]
                    assert mkOptional.call_args_list == \
                        [mk.call(0, pass_optional=False, model_set_axis=0)]
                    assert mkOutputs.call_args_list == [mk.call(1)]
                    mkInputs.reset_mock()
                    mkOptional.reset_mock()
                    mkOutputs.reset_mock()

                    # No Defaults
                    meta_data = evaluation_io.MetaData.create_defaults(n_inputs,
                                                                          n_outputs,
                                                                          n_models,
                                                                          standard_broadcasting,
                                                                          pass_optional,
                                                                          model_set_axis,
                                                                          optional)
                    assert meta_data._inputs == mkInputs.return_value
                    assert meta_data._optional == optional
                    assert optional.pass_optional == pass_optional
                    assert optional.model_set_axis == model_set_axis
                    assert meta_data._outputs == mkOutputs.return_value
                    assert meta_data._n_models == n_models
                    assert meta_data._standard_broadcasting == standard_broadcasting
                    assert mkInputs.call_args_list == [mk.call(n_inputs)]
                    assert mkOptional.call_args_list == []
                    assert mkOutputs.call_args_list == [mk.call(n_outputs)]

    def test_n_inputs(self):
        inputs = evaluation_io.InputMetaData.create_defaults(3)
        meta_data = evaluation_io.MetaData(inputs, mk.MagicMock(), mk.MagicMock())

        with mk.patch.object(evaluation_io.InputMetaData, 'n_inputs',
                             new_callable=mk.PropertyMock) as mkInputs:
            # Test Get
            assert meta_data.n_inputs == mkInputs.return_value
            assert mkInputs.call_args_list == [mk.call()]
            mkInputs.reset_mock()

            # Test Set
            n_inputs = mk.MagicMock()
            meta_data.n_inputs = n_inputs
            assert mkInputs.call_args_list == [mk.call(n_inputs)]

    def test_n_outputs(self):
        outputs = evaluation_io.OutputMetaData.create_defaults(3)
        meta_data = evaluation_io.MetaData(mk.MagicMock(), mk.MagicMock(),
                                              outputs)

        with mk.patch.object(evaluation_io.OutputMetaData, 'n_outputs',
                             new_callable=mk.PropertyMock) as mkOutputs:
            # Test Get
            assert meta_data.n_outputs == mkOutputs.return_value
            assert mkOutputs.call_args_list == [mk.call()]
            mkOutputs.reset_mock()

            # Test Set
            n_outputs = mk.MagicMock()
            meta_data.n_outputs = n_outputs
            assert mkOutputs.call_args_list == [mk.call(n_outputs)]

    def test_n_models(self):
        n_models = mk.MagicMock()
        meta_data = evaluation_io.MetaData(mk.MagicMock(), mk.MagicMock(),
                                              mk.MagicMock(), n_models=n_models)
        # Test get
        assert meta_data._n_models == n_models
        assert meta_data.n_models == n_models

        # Test set
        new_n_models = mk.MagicMock()
        meta_data.n_models = new_n_models
        assert meta_data._n_models == new_n_models
        assert meta_data.n_models == new_n_models

    def test_model_set_axis(self):
        optional = evaluation_io.OptionalMetaData.create_defaults(3)
        meta_data = evaluation_io.MetaData(mk.MagicMock(), optional,
                                              mk.MagicMock())

        with mk.patch.object(evaluation_io.OptionalMetaData, 'model_set_axis',
                             new_callable=mk.PropertyMock) as mkModel:
            # Test Get
            assert meta_data.model_set_axis == mkModel.return_value
            assert mkModel.call_args_list == [mk.call()]
            mkModel.reset_mock()

            # Test Set
            model_set_axis = mk.MagicMock()
            meta_data.model_set_axis = model_set_axis
            assert mkModel.call_args_list == [mk.call(model_set_axis)]

    def test_standard_broadcasting(self):
        standard_broadcasting = mk.MagicMock()
        meta_data = evaluation_io.MetaData(mk.MagicMock(), mk.MagicMock(), mk.MagicMock(),
                                              standard_broadcasting=standard_broadcasting)
        # Test get
        assert meta_data._standard_broadcasting == standard_broadcasting
        assert meta_data.standard_broadcasting == standard_broadcasting

        # Test set
        new_standard_broadcasting = mk.MagicMock()
        meta_data.standard_broadcasting = new_standard_broadcasting
        assert meta_data._standard_broadcasting == new_standard_broadcasting
        assert meta_data.standard_broadcasting == new_standard_broadcasting

    def test_inputs(self):
        inputs = evaluation_io.InputMetaData.create_defaults(3)
        optional = evaluation_io.OptionalMetaData.create_defaults(3)
        meta_data = evaluation_io.MetaData(inputs, optional, mk.MagicMock())

        # Test get
        assert meta_data.inputs == ('x0', 'x1', 'x2')

        # Test set
        value = mk.MagicMock()
        with mk.patch.object(evaluation_io.InputMetaData, 'inputs',
                             new_callable=mk.PropertyMock) as mkInputs:
            with mk.patch.object(evaluation_io.OptionalMetaData, 'validate',
                                 autospec=True) as mkValidate:
                meta_data.inputs = value
                assert mkInputs.call_args_list == [mk.call(value), mk.call()]
                assert mkValidate.call_args_list == \
                    [mk.call(optional, mkInputs.return_value)]

    def test_optional(self):
        inputs = evaluation_io.InputMetaData.create_defaults(3)
        optional = evaluation_io.OptionalMetaData.create_defaults(3)
        meta_data = evaluation_io.MetaData(inputs, optional, mk.MagicMock())

        # Test get
        assert meta_data.optional == ('optional_0', 'optional_1', 'optional_2')

        # Test set
        value = mk.MagicMock()
        with mk.patch.object(evaluation_io.OptionalMetaData, 'optional',
                             new_callable=mk.PropertyMock) as mkOptional:
            with mk.patch.object(evaluation_io.InputMetaData, 'inputs',
                                 new_callable=mk.PropertyMock) as mkInputs:
                with mk.patch.object(evaluation_io.OptionalMetaData, 'validate',
                                     autospec=True) as mkValidate:
                    meta_data.optional = value
                    assert mkOptional.call_args_list == [mk.call(value)]
                    assert mkInputs.call_args_list == [mk.call()]
                    assert mkValidate.call_args_list == \
                        [mk.call(optional, mkInputs.return_value)]

    def test_outputs(self):
        outputs = evaluation_io.OutputMetaData.create_defaults(3)
        meta_data = evaluation_io.MetaData(mk.MagicMock(), mk.MagicMock(),
                                              outputs)

        # Test get
        assert meta_data.outputs == ('y0', 'y1', 'y2')

        # Test set
        value = mk.MagicMock()
        with mk.patch.object(evaluation_io.OutputMetaData, 'outputs',
                             new_callable=mk.PropertyMock) as mkOutputs:
                meta_data.outputs = value
                assert mkOutputs.call_args_list == [mk.call(value)]

    def test_evaluation_inputs(self):
        inputs = evaluation_io.InputMetaData.create_defaults(3)
        optional = evaluation_io.OptionalMetaData.create_defaults(3)
        meta_data = evaluation_io.MetaData(inputs, optional, mk.MagicMock())

        args = (mk.MagicMock(), mk.MagicMock())
        kwargs = {'test': mk.MagicMock()}

        inputs_return = (mk.MagicMock(), {'inputs_test': mk.MagicMock()})
        with mk.patch.object(evaluation_io.InputMetaData, 'get_inputs',
                             autospec=True, return_value=inputs_return) as mkInputs:
            with mk.patch.object(evaluation_io.OptionalMetaData, 'get_optional',
                                 autospec=True) as mkOptional:
                with mk.patch.object(evaluation_io.EvaluationInputs, 'evaluation_inputs',
                                     autospec=True) as mkEvaluation:
                    assert meta_data.evaluation_inputs(*args, **kwargs) == \
                        mkEvaluation.return_value
                    assert mkEvaluation.call_args_list == \
                        [mk.call(inputs_return[0], mkOptional.return_value)]
                    assert mkOptional.call_args_list == \
                        [mk.call(optional, **inputs_return[1])]
                    assert mkInputs.call_args_list == \
                        [mk.call(inputs, *args, **kwargs)]

    def test_set_format_info(self):
        meta_data = evaluation_io.MetaData(mk.MagicMock(), mk.MagicMock(),
                                              mk.MagicMock())
        params = mk.MagicMock()
        inputs = evaluation_io.EvaluationInputs(mk.MagicMock(), mk.MagicMock(),
                                                mk.MagicMock())

        with mk.patch.object(evaluation_io.MetaData, 'standard_broadcasting',
                             new_callable=mk.PropertyMock) as mkStd:
            with mk.patch.object(evaluation_io.MetaData, 'n_models',
                                 new_callable=mk.PropertyMock) as mkModels:
                with mk.patch.object(evaluation_io.MetaData, 'model_set_axis',
                                     new_callable=mk.PropertyMock) as mkAxis:
                    with mk.patch.object(evaluation_io.MetaData, 'n_outputs',
                                         new_callable=mk.PropertyMock) as mkOutputs:
                        with mk.patch.object(evaluation_io.EvaluationInputs, 'set_format_info',
                                             autospec=True) as mkSet:
                            meta_data.set_format_info(params, inputs)
                            assert mkSet.call_args_list == \
                                [mk.call(inputs, params, mkStd.return_value,
                                         mkModels.return_value, mkAxis.return_value,
                                         mkOutputs.return_value)]
                            assert mkStd.call_args_list == [mk.call()]
                            assert mkModels.call_args_list == [mk.call()]
                            assert mkAxis.call_args_list == [mk.call()]
                            assert mkOutputs.call_args_list == [mk.call()]

    def test_enforce_bounding_box(self):
        inputs = evaluation_io.InputMetaData.create_defaults(3)
        meta_data = evaluation_io.MetaData(inputs, mk.MagicMock(), mk.MagicMock())
        evaluation = mk.MagicMock()

        with mk.patch.object(evaluation_io.MetaData, 'n_models',
                             new_callable=mk.PropertyMock) as mkModels:
            with mk.patch.object(evaluation_io.MetaData, 'model_set_axis',
                                 new_callable=mk.PropertyMock) as mkAxis:
                with mk.patch.object(evaluation_io.InputMetaData, 'enforce_bounding_box',
                                     autospec=True) as mkEnforce:
                    meta_data.enforce_bounding_box(evaluation)
                    assert mkEnforce.call_args_list == \
                        [mk.call(inputs, mkModels.return_value, mkAxis.return_value, evaluation)]
                    assert mkModels.call_args_list == [mk.call()]
                    assert mkAxis.call_args_list == [mk.call()]

    def test_process_inputs(self):
        meta_data = evaluation_io.MetaData(mk.MagicMock(), mk.MagicMock(),
                                              mk.MagicMock())
        params = mk.MagicMock()
        inputs = evaluation_io.EvaluationInputs(mk.MagicMock(), mk.MagicMock(),
                                                mk.MagicMock())

        with mk.patch.object(evaluation_io.EvaluationInputs, 'check_input_shape',
                             autospec=True) as mkCheck:
            with mk.patch.object(evaluation_io.MetaData, 'n_models',
                                 new_callable=mk.PropertyMock) as mkModels:
                with mk.patch.object(evaluation_io.MetaData, 'set_format_info',
                                     autospec=True) as mkSet:
                    with mk.patch.object(evaluation_io.MetaData, 'enforce_bounding_box',
                                         autospec=True) as mkEnforce:
                        main = mk.MagicMock()
                        main.attach_mock(mkCheck, 'check')
                        main.attach_mock(mkSet, 'set')
                        main.attach_mock(mkEnforce, 'enforce')

                        meta_data.process_inputs(params, inputs)
                        assert main.mock_calls == \
                            [
                                mk.call.check(inputs, mkModels.return_value),
                                mk.call.set(meta_data, params, inputs),
                                mk.call.enforce(meta_data, inputs)
                            ]
                        assert mkModels.call_args_list == [mk.call()]

    def test_prepare_inputs(self):
        meta_data = evaluation_io.MetaData(mk.MagicMock(), mk.MagicMock(),
                                              mk.MagicMock())
        params = mk.MagicMock()
        args = (mk.MagicMock(), mk.MagicMock())
        kwargs = {'test': mk.MagicMock()}

        with mk.patch.object(evaluation_io.MetaData, 'evaluation_inputs',
                             autospec=True) as mkInputs:
            with mk.patch.object(evaluation_io.MetaData, 'process_inputs',
                                 autospec=True) as mkProcess:
                assert meta_data.prepare_inputs(params, *args, **kwargs) == \
                    mkInputs.return_value
                assert mkProcess.call_args_list == \
                    [mk.call(meta_data, params, mkInputs.return_value)]
                assert mkInputs.call_args_list == \
                    [mk.call(meta_data, *args, **kwargs)]
