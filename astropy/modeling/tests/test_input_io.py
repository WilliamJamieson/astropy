# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Tests for evaluation IO for Models"""

import pytest
import numpy as np
import unittest.mock as mk

from astropy.modeling import input_io
from astropy.utils import shapes as utils_shapes
from astropy.modeling.utils import _BoundingBox


class TestInputEntry:
    def test___init__(self):
        entry = input_io.InputEntry('name', 1)
        assert entry._name == 'name'
        assert entry._input == np.asanyarray(1)

        entry = input_io.InputEntry('name', 1)
        assert entry._name == 'name'
        assert entry._input == np.asanyarray(1)

    def test___eq__(self):
        assert input_io.InputEntry('name', 1) == input_io.InputEntry('name', 1)
        assert input_io.InputEntry('name', [1, 2]) == input_io.InputEntry('name', [1, 2])
        assert input_io.InputEntry('name', [1, 2]) == input_io.InputEntry('name', [1, 2])

        entry = input_io.InputEntry('name', [1, 2])
        fake_entry = mk.MagicMock()
        fake_entry.name = entry.name
        fake_entry.input = entry.input
        assert not (entry == fake_entry)

    def test_name(self):
        entry = input_io.InputEntry('name', 1)
        assert entry.name == 'name'
        assert entry._name == 'name'

    def test_input(self):
        with mk.patch.object(np, 'asanyarray', autospec=True) as mkNp:
            entry = input_io.InputEntry('name', 1)
            assert mkNp.call_args_list == [mk.call(1, dtype=float)]
            mkNp.reset_mock()

            # Test get
            assert entry.input == mkNp.return_value
            assert entry._input == mkNp.return_value

            # Test set
            entry.input = 2
            assert mkNp.call_args_list == [mk.call(2, dtype=float)]
            assert entry.input == mkNp.return_value
            assert entry._input == mkNp.return_value

    def test_shape(self):
        entry = input_io.InputEntry('name', 1)
        assert entry.shape == tuple()

        entry = input_io.InputEntry('name', [1, 2])
        assert entry.shape == (2,)

        entry = input_io.InputEntry('name', np.ones((2, 2)))
        assert entry.shape == (2, 2)

        entry = input_io.InputEntry('name', np.ones((2, 2, 2)))
        assert entry.shape == (2, 2, 2)

    def test_input_array(self):
        # Cast scalar to array
        entry = input_io.InputEntry('name', 1)
        assert entry.input_array.shape == np.array([1], dtype=float).shape
        assert entry.input_array.shape != entry.input.shape
        assert entry.input_array == entry.input

        # Leave array alone
        entry = input_io.InputEntry('name', [1, 2])
        assert entry.input_array.shape == np.array([1, 2], dtype=float).shape
        assert entry.input_array.shape == entry.input.shape
        assert (entry.input_array == entry.input).all()

    def test__array_shape(self):
        entry = input_io.InputEntry('name', 1)
        assert entry._array_shape(True) == (1,)
        assert entry._array_shape(False) == tuple()

        entry = input_io.InputEntry('name', [1, 2])
        assert entry._array_shape(True) == (2,)
        assert entry._array_shape(False) == (2,)

    def test_check_input_shape(self):
        entry = input_io.InputEntry('name', 1)
        array_shape = mk.MagicMock()

        # Passes
        shapes = [tuple(), (1,), (4,), (2,)]
        with mk.patch.object(input_io.InputEntry, '_array_shape', autospec=True,
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
        with mk.patch.object(input_io.InputEntry, '_array_shape', autospec=True,
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

    def test__get_param_broadcast(self):
        entry = input_io.InputEntry('name', 1)
        param = mk.MagicMock()

        effects = [mk.MagicMock(), input_io.IncompatibleShapeError(1, 2, 3, 4)]
        with mk.patch.object(input_io, 'check_broadcast', autospec=True,
                             side_effect=effects) as mkCheck:
            with mk.patch.object(input_io.InputEntry, 'shape',
                                 new_callable=mk.PropertyMock) as mkShape:
                # Standard broadcast success
                assert entry._get_param_broadcast(param, True) == effects[0]
                assert mkCheck.call_args_list == [mk.call(mkShape.return_value, param.shape)]
                assert mkShape.call_args_list == [mk.call()]
                mkCheck.reset_mock()
                mkShape.reset_mock()

                # Standard broadcast fail
                with pytest.raises(ValueError):
                    entry._get_param_broadcast(param, True)
                assert mkCheck.call_args_list == [mk.call(mkShape.return_value, param.shape)]
                assert mkShape.call_args_list == [mk.call(), mk.call()]
                mkCheck.reset_mock()
                mkShape.reset_mock()

                # No standard broadcast
                assert entry._get_param_broadcast(param, False) == mkShape.return_value
                assert mkCheck.call_args_list == []
                assert mkShape.call_args_list == [mk.call()]

    def test__update_param_broadcast(self):
        entry = input_io.InputEntry('name', 1)
        broadcast = (2,)
        param = mk.MagicMock()
        standard_broadcasting = mk.MagicMock()

        effects = [tuple(), (1,), (3,), (3, 4)]
        with mk.patch.object(input_io.InputEntry, '_get_param_broadcast',
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
        entry = input_io.InputEntry('name', 1)
        params = [mk.MagicMock() for _ in range(3)]
        standard_broadcasting = mk.MagicMock()

        effects = [mk.MagicMock() for _ in range(3)]
        with mk.patch.object(input_io.InputEntry, '_update_param_broadcast',
                             autospec=True, side_effect=effects) as mkUpdate:
            with mk.patch.object(input_io.InputEntry, 'shape',
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

    def test_reduce_to_bounding_box(self):
        entry = input_io.InputEntry('name', np.arange(0, 5))
        valid_index = [1, 3, 4]
        assert entry.reduce_to_bounding_box(valid_index, True) == \
            input_io.InputEntry('name', [1, 3, 4])
        assert entry.reduce_to_bounding_box(valid_index, False) == \
            input_io.InputEntry('name', [1, 3, 4])

        entry = input_io.InputEntry('name', 1)
        valid_index = [0]
        assert entry.reduce_to_bounding_box(valid_index, True) == \
            input_io.InputEntry('name', 1)
        with pytest.raises(IndexError):
            entry.reduce_to_bounding_box(valid_index, False)


class TestInputs:
    def test___init__(self):
        inputs = input_io.Inputs({'test': input_io.InputEntry('test', 1)}, {'option': 2}, [3])
        assert inputs._inputs == {'test': input_io.InputEntry('test', 1)}
        assert inputs._optional == {'option': 2}
        assert inputs._format_info == [3]

    def test_n_inputs(self):
        inputs = input_io.Inputs({'test': input_io.InputEntry('test', 1)}, {'option': 2}, [3])
        assert inputs.n_inputs == 1

        inputs = input_io.Inputs({'test': input_io.InputEntry('test', 1),
                                  'next': input_io.InputEntry('next', 2)}, {'option': 2}, [3])
        assert inputs.n_inputs == 2

    def test__check_input_shape(self):
        inputs = input_io.Inputs({}, {}, [])
        entries = {f"x{idx}": mk.MagicMock() for idx in range(3)}
        inputs._inputs = entries

        check_args = [entry.check_input_shape.return_value for entry in entries.values()]

        n_models = mk.MagicMock()
        model_set_axis = mk.MagicMock()
        array_shape = mk.MagicMock()

        effects = [mk.MagicMock(), None]
        with mk.patch.object(input_io, 'check_broadcast', autospec=True,
                             side_effect=effects) as mkCheck:
            # Success
            assert inputs.check_input_shape(n_models, model_set_axis, array_shape) == effects[0]
            for entry in entries.values():
                assert entry.check_input_shape.call_args_list == \
                    [mk.call(n_models, model_set_axis, array_shape)]
                entry.check_input_shape.reset_mock()
            assert mkCheck.call_args_list == [mk.call(*check_args)]
            mkCheck.reset_mock()

            # Fail
            with pytest.raises(ValueError):
                inputs.check_input_shape(n_models, model_set_axis, array_shape)
            for entry in entries.values():
                assert entry.check_input_shape.call_args_list == \
                    [mk.call(n_models, model_set_axis, array_shape)]
                entry.check_input_shape.reset_mock()
            assert mkCheck.call_args_list == [mk.call(*check_args)]

    def test_reduce_to_bounding_box(self):
        entries = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        inputs = input_io.Inputs(entries, mk.MagicMock(), mk.MagicMock())

        valid_index = mk.MagicMock()
        array_shape = mk.MagicMock()
        reduced = inputs.reduce_to_bounding_box(valid_index, array_shape)
        assert isinstance(reduced, input_io.Inputs)
        assert reduced.optional == inputs.optional
        assert reduced.format_info == inputs.format_info
        assert len(reduced.inputs) == len(entries) == 3
        for name, entry in entries.items():
            assert name in reduced.inputs
            assert reduced.inputs[name] == entry.reduce_to_bounding_box.return_value
            assert entry.reduce_to_bounding_box.call_args_list == \
                [mk.call(valid_index, array_shape)]
        for name, entry in reduced.inputs.items():
            assert name in entries
            assert entries[name].reduce_to_bounding_box.return_value == entry
            assert entries[name].reduce_to_bounding_box.call_args_list == \
                [mk.call(valid_index, array_shape)]


class TestIoMetaDataEntry:
    def test___init__(self):
        entry = input_io.IoMetaDataEntry()
        assert entry._name is None

        entry = input_io.IoMetaDataEntry('test')
        assert entry._name == 'test'

    def test_name(self):
        # test get error
        entry = input_io.IoMetaDataEntry()
        with pytest.raises(RuntimeError):
            entry.name

        # test set and get without error
        entry.name = 'test'
        assert entry.name == 'test'
        assert entry._name == 'test'
        entry = input_io.IoMetaDataEntry('test')
        assert entry.name == 'test'
        assert entry._name == 'test'

        # test set with error
        with pytest.raises(ValueError):
            entry.name = 'new_test'
        assert entry.name == 'test'
        assert entry._name == 'test'

    def test_create_entry(self):
        with pytest.raises(NotImplementedError):
            input_io.IoMetaDataEntry.create_entry(mk.MagicMock(), test=mk.MagicMock())


class TestInputMetaDataEntry:
    def test___init__(self):
        entry = input_io.InputMetaDataEntry()
        assert entry._name is None
        assert entry._pos is None
        assert entry._bounding_box is None

        entry = input_io.InputMetaDataEntry('test', 1, _BoundingBox((-1, 1)))
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box == _BoundingBox((-1, 1))

    def test_pos(self):
        # test get error
        entry = input_io.InputMetaDataEntry()
        with pytest.raises(RuntimeError):
            entry.pos

        # test set and get without error
        entry.pos = 1
        assert entry.pos == 1
        assert entry._pos == 1
        entry = input_io.InputMetaDataEntry('test', 1)
        assert entry.pos == 1
        assert entry._pos == 1

        # test set with error
        with pytest.raises(ValueError):
            entry.pos = 2
        assert entry.pos == 1
        assert entry._pos == 1

    def test_bounding_box(self):
        # test get error
        entry = input_io.InputMetaDataEntry()
        with pytest.raises(NotImplementedError):
            entry.bounding_box

        # test set and get without error
        value = mk.MagicMock()
        bbox = mk.MagicMock()
        with mk.patch.object(input_io, '_BoundingBox', autospec=True,
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
        input_value = input_io.InputMetaDataEntry()
        entry = input_io.InputMetaDataEntry.create_entry(input_value)
        assert entry == input_value
        entry = input_io.InputMetaDataEntry.create_entry(input_value, name='test')
        assert entry == input_value
        entry = input_io.InputMetaDataEntry.create_entry(input_value, pos=1)
        assert entry == input_value
        entry = input_io.InputMetaDataEntry.create_entry(input_value, name='test', pos=1)
        assert entry == input_value

        # test pass tuple in
        entry = input_io.InputMetaDataEntry.create_entry((1, (-1, 1)))
        assert entry._name is None
        assert entry._pos == 1
        assert entry._bounding_box == (-1, 1)
        entry = input_io.InputMetaDataEntry.create_entry((1, (-1, 1)), name='test')
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box == (-1, 1)
        entry = input_io.InputMetaDataEntry.create_entry((1, (-1, 1)), name='test', pos=2)
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box == (-1, 1)
        entry = input_io.InputMetaDataEntry.create_entry((1,))
        assert entry._name is None
        assert entry._pos == 1
        assert entry._bounding_box is None
        entry = input_io.InputMetaDataEntry.create_entry((1,), name='test')
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box is None
        entry = input_io.InputMetaDataEntry.create_entry((1,), name='test', pos=2)
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box is None

        # test pass str in
        entry = input_io.InputMetaDataEntry.create_entry('test')
        assert entry._name == 'test'
        assert entry._pos is None
        assert entry._bounding_box is None
        entry = input_io.InputMetaDataEntry.create_entry('test', name=mk.MagicMock())
        assert entry._name == 'test'
        assert entry._pos is None
        assert entry._bounding_box is None
        entry = input_io.InputMetaDataEntry.create_entry('test', name=mk.MagicMock(), pos=1)
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._bounding_box is None

        # test pass bad input
        with pytest.raises(ValueError):
            input_io.InputMetaDataEntry.create_entry(mk.MagicMock())

    def test_outside(self):
        entry = input_io.InputMetaDataEntry()
        value = mk.MagicMock()
        with mk.patch.object(input_io.InputMetaDataEntry, 'bounding_box',
                             new_callable=mk.PropertyMock) as mkBbox:
            assert entry.outside(value) == mkBbox.return_value.outside.return_value
            assert mkBbox.call_args_list == [mk.call()]
            assert mkBbox.return_value.outside.call_args_list == [mk.call(value)]


class TestOptionalMetaDataEntry:
    def test___init__(self):
        entry = input_io.OptionalMetaDataEntry()
        assert entry._name is None
        assert entry._default is None

        entry = input_io.OptionalMetaDataEntry('test', 1)
        assert entry._name == 'test'
        assert entry._default == 1

    def test_default(self):
        entry = input_io.OptionalMetaDataEntry()
        assert entry.default is None
        assert entry._default is None

        entry = input_io.OptionalMetaDataEntry(default=1)
        assert entry.default == 1
        assert entry._default == 1

    def test_create_entry(self):
        # test pass input metadata entry in
        input_value = input_io.OptionalMetaDataEntry()
        entry = input_io.OptionalMetaDataEntry.create_entry(input_value)
        assert entry == input_value
        entry = input_io.OptionalMetaDataEntry.create_entry(input_value, name='test')
        assert entry == input_value
        entry = input_io.OptionalMetaDataEntry.create_entry(input_value, pos=1)
        assert entry == input_value
        entry = input_io.OptionalMetaDataEntry.create_entry(input_value, name='test', pos=1)
        assert entry == input_value

        # test pass tuple in
        entry = input_io.OptionalMetaDataEntry.create_entry((1,))
        assert entry._name is None
        assert entry._default == 1
        entry = input_io.OptionalMetaDataEntry.create_entry((1,), name='test')
        assert entry._name == 'test'
        assert entry._default == 1
        entry = input_io.OptionalMetaDataEntry.create_entry((1,), name='test', pos=2)
        assert entry._name == 'test'
        assert entry._default == 1

        # test pass str in
        entry = input_io.OptionalMetaDataEntry.create_entry('test')
        assert entry._name == 'test'
        assert entry._default is None
        entry = input_io.OptionalMetaDataEntry.create_entry('test', name=mk.MagicMock())
        assert entry._name == 'test'
        assert entry._default is None
        entry = input_io.OptionalMetaDataEntry.create_entry('test', name=mk.MagicMock(), pos=1)
        assert entry._name == 'test'
        assert entry._default is None

        # test pass bad input
        with pytest.raises(ValueError):
            input_io.OptionalMetaDataEntry.create_entry(mk.MagicMock())


class TestInputMetaData:
    def test___init__(self):
        inputs = input_io.InputMetaData(1)
        assert inputs._n_inputs == 1
        assert inputs._inputs == {}
        assert inputs._optional == {}
        assert not inputs._pass_optional

        with mk.patch.object(input_io.InputMetaData, 'inputs',
                             new_callable=mk.PropertyMock) as mkInputs:
            with mk.patch.object(input_io.InputMetaData, 'optional',
                                 new_callable=mk.PropertyMock) as mkOptional:
                mk_inputs = mk.MagicMock()
                mk_optional = mk.MagicMock()
                inputs = input_io.InputMetaData(1, mk_inputs, mk_optional, True)

                assert inputs._n_inputs == 1
                assert inputs._pass_optional

                assert mkInputs.call_args_list == [mk.call(mk_inputs)]
                assert mkOptional.call_args_list == [mk.call(mk_optional)]

    def test__fill_defaults(self):
        # 1D
        inputs = input_io.InputMetaData(1)
        inputs._fill_defaults()
        assert inputs._n_inputs == 1
        assert not inputs._pass_optional
        assert inputs._optional == {}
        assert len(inputs._inputs) == 1
        assert 'x' in inputs._inputs
        assert inputs._inputs['x'].name == 'x'
        assert inputs._inputs['x'].pos == 0
        assert inputs._inputs['x']._bounding_box is None

        # 2D
        inputs = input_io.InputMetaData(2)
        inputs._fill_defaults()
        assert inputs._n_inputs == 2
        assert not inputs._pass_optional
        assert inputs._optional == {}
        assert len(inputs._inputs) == 2
        assert 'x' in inputs._inputs
        assert inputs._inputs['x'].name == 'x'
        assert inputs._inputs['x'].pos == 0
        assert inputs._inputs['x']._bounding_box is None
        assert 'y' in inputs._inputs
        assert inputs._inputs['y'].name == 'y'
        assert inputs._inputs['y'].pos == 1
        assert inputs._inputs['y']._bounding_box is None

        # 3D
        inputs = input_io.InputMetaData(3)
        inputs._fill_defaults()
        assert inputs._n_inputs == 3
        assert not inputs._pass_optional
        assert inputs._optional == {}
        assert len(inputs._inputs) == 3
        assert 'x0' in inputs._inputs
        assert inputs._inputs['x0'].name == 'x0'
        assert inputs._inputs['x0'].pos == 0
        assert inputs._inputs['x0']._bounding_box is None
        assert 'x1' in inputs._inputs
        assert inputs._inputs['x1'].name == 'x1'
        assert inputs._inputs['x1'].pos == 1
        assert inputs._inputs['x1']._bounding_box is None
        assert 'x2' in inputs._inputs
        assert inputs._inputs['x2'].name == 'x2'
        assert inputs._inputs['x2'].pos == 2
        assert inputs._inputs['x2']._bounding_box is None

    def test_create_defaults(self):
        with mk.patch.object(input_io.InputMetaData, '_fill_defaults',
                             autospec=True) as mkFill:
            inputs = input_io.InputMetaData.create_defaults(1)
            assert inputs._n_inputs == 1
            assert not inputs._pass_optional
            assert inputs._inputs == {}
            assert inputs._optional == {}
            assert mkFill.call_args_list == [mk.call(inputs)]

            mkFill.reset_mock()
            optional = {'z': input_io.OptionalMetaDataEntry('z')}
            inputs = input_io.InputMetaData.create_defaults(1, optional, True)
            assert inputs._n_inputs == 1
            assert inputs._pass_optional
            assert inputs._inputs == {}
            assert inputs._optional == optional
            assert mkFill.call_args_list == [mk.call(inputs)]

    def test_n_inputs(self):
        inputs = input_io.InputMetaData(1)

        # Test get
        assert inputs.n_inputs == 1
        assert inputs._n_inputs == 1

        # Test set
        assert inputs._inputs == {}
        with mk.patch.object(input_io.InputMetaData, '_fill_defaults',
                             autospec=True) as mkFill:
            inputs.n_inputs = 2
            assert inputs.n_inputs == 2
            assert inputs._n_inputs == 2
            assert mkFill.call_args_list == [mk.call(inputs)]
            assert inputs._inputs == {}

    def test_pass_optional(self):
        inputs = input_io.InputMetaData(1)

        # Test get
        assert not inputs.pass_optional
        assert not inputs._pass_optional

        # Test set
        inputs.pass_optional = True
        assert inputs.pass_optional
        assert inputs._pass_optional

    def test_inputs_get(self):
        inputs = input_io.InputMetaData.create_defaults(1)
        assert inputs.inputs == ['x']
        inputs = input_io.InputMetaData.create_defaults(2)
        assert inputs.inputs == ['x', 'y']
        inputs = input_io.InputMetaData.create_defaults(3)
        assert inputs.inputs == ['x0', 'x1', 'x2']

    def test_optional_get(self):
        optional = {'z': input_io.OptionalMetaDataEntry('z')}
        inputs = input_io.InputMetaData.create_defaults(1, optional=optional)
        assert inputs.optional == ['z']
        optional = {'z0': input_io.OptionalMetaDataEntry('z0'),
                    'z1': input_io.OptionalMetaDataEntry('z1')}
        inputs = input_io.InputMetaData.create_defaults(1, optional=optional)
        assert inputs.optional == ['z0', 'z1']
        optional = {'z0': input_io.OptionalMetaDataEntry('z0'),
                    'z1': input_io.OptionalMetaDataEntry('z1'),
                    'z2': input_io.OptionalMetaDataEntry('z2')}
        inputs = input_io.InputMetaData.create_defaults(1, optional=optional)
        assert inputs.optional == ['z0', 'z1', 'z2']

    def test_reset_inputs(self):
        inputs = input_io.InputMetaData.create_defaults(3)
        assert inputs.inputs == ['x0', 'x1', 'x2']

        inputs.reset_inputs()
        assert inputs.inputs == []

    def test_reset_optional(self):
        optional = {'z0': input_io.OptionalMetaDataEntry('z0'),
                    'z1': input_io.OptionalMetaDataEntry('z1'),
                    'z2': input_io.OptionalMetaDataEntry('z2')}
        inputs = input_io.InputMetaData.create_defaults(1, optional=optional)
        assert inputs.optional == ['z0', 'z1', 'z2']

        inputs.reset_optional()
        assert inputs.optional == []

    def test_validate(self):
        inputs = input_io.InputMetaData(1)
        inputs = input_io.InputMetaData(1, {}, {})

        # Fail inputs size
        inputs._n_inputs = 2
        inputs._inputs = {'x': mk.MagicMock()}
        with pytest.raises(ValueError, match=r"n_inputs .*"):
            inputs.validate()
        inputs._n_inputs = 1

        # Fail input name
        inputs._inputs = {'x': input_io.InputMetaDataEntry('y')}
        with pytest.raises(ValueError, match=r"Input: .* key .*"):
            inputs.validate()

        # Fail input position
        inputs._inputs = {'x': input_io.InputMetaDataEntry('x', pos=1)}
        with pytest.raises(ValueError, match=r"Input: .* position .*"):
            inputs.validate()

        # Fix fails
        inputs._inputs = {'x': input_io.InputMetaDataEntry('x', pos=0)}
        inputs.validate()

        # Fail optional name
        inputs._optional = {'x': input_io.OptionalMetaDataEntry('y')}
        with pytest.raises(ValueError, match=r"Optional: .* key .*"):
            inputs.validate()

        # Fail optional is normal input
        inputs._optional = {'x': input_io.OptionalMetaDataEntry('x')}
        with pytest.raises(ValueError, match=r"Optional: .* is both .*"):
            inputs.validate()

        # Fix fails
        inputs._optional = {'z': input_io.OptionalMetaDataEntry('z')}

    def test__process_inputs_atr(self):
        inputs = input_io.InputMetaData(1)

        # None input
        assert inputs._process_inputs_atr(None, input_io.IoMetaDataEntry) == {}

        # List input
        value = [mk.MagicMock(), mk.MagicMock()]
        entries = [mk.MagicMock(), mk.MagicMock()]
        with mk.patch.object(input_io.IoMetaDataEntry, 'create_entry',
                             autospec=True, side_effect=entries) as mkCreate:
            process_inputs = inputs._process_inputs_atr(value, input_io.IoMetaDataEntry)
            assert isinstance(process_inputs, dict)
            assert len(process_inputs) == 2
            assert len(mkCreate.call_args_list) == 2
            for index, (name, entry) in enumerate(process_inputs.items()):
                assert name == entries[index].name
                assert entry == entries[index]
                assert mkCreate.call_args_list[index] == mk.call(value[index], pos=index)

        # Dict input
        value = {'entry0': mk.MagicMock(), 'entry1': mk.MagicMock()}
        entries = [mk.MagicMock(), mk.MagicMock()]
        with mk.patch.object(input_io.IoMetaDataEntry, 'create_entry',
                             autospec=True, side_effect=entries) as mkCreate:
            process_inputs = inputs._process_inputs_atr(value, input_io.IoMetaDataEntry)
            assert isinstance(process_inputs, dict)
            assert len(process_inputs) == 2
            assert len(mkCreate.call_args_list) == 2
            for index, (name, entry) in enumerate(process_inputs.items()):
                assert name == entries[index].name
                assert entry == entries[index]
            for index, (name, entry) in enumerate(value.items()):
                assert mkCreate.call_args_list[index] == mk.call(entry, name=name)

        # Other input
        with pytest.raises(ValueError):
            inputs._process_inputs_atr(mk.MagicMock(), input_io.IoMetaDataEntry)

    def test__set_inputs_atr(self):
        value = mk.MagicMock()
        data_entry = mk.MagicMock()
        inputs = input_io.InputMetaData(1)

        # Test non-existent
        assert not hasattr(inputs, 'test')
        with pytest.raises(AttributeError):
            inputs._set_inputs_atr('test', value, data_entry)

        # Test fail for none empty
        test_value = {1: mk.MagicMock()}
        setattr(inputs, 'test', test_value)
        assert inputs.test == test_value
        assert len(inputs.test) != 0
        with pytest.raises(RuntimeError):
            inputs._set_inputs_atr('test', value, data_entry)
        assert inputs.test == test_value

        # test success
        setattr(inputs, 'test', {})
        assert inputs.test == {}
        assert len(inputs.test) == 0
        with mk.patch.object(input_io.InputMetaData, '_process_inputs_atr',
                             autospec=True) as mkProcess:
            inputs._set_inputs_atr('test', value, data_entry)
            assert inputs.test == mkProcess.return_value
            assert mkProcess.call_args_list == [mk.call(value, data_entry)]

    def test_inputs_setter(self):
        value = mk.MagicMock()
        inputs = input_io.InputMetaData(1)
        assert inputs.inputs == []

        with mk.patch.object(input_io.InputMetaData, '_set_inputs_atr',
                             autospec=True) as mkSet:
            with mk.patch.object(input_io.InputMetaData, 'validate',
                                 autospec=True) as mkValidate:
                main = mk.MagicMock()
                main.attach_mock(mkSet, 'set')
                main.attach_mock(mkValidate, 'validate')

                inputs.inputs = value
                assert main.mock_calls == [mk.call.set(inputs, '_inputs', value, input_io.InputMetaDataEntry),
                                           mk.call.validate(inputs)]
                assert inputs.inputs == []

    def test_optional_setter(self):
        value = mk.MagicMock()
        inputs = input_io.InputMetaData(1)
        assert inputs.inputs == []

        with mk.patch.object(input_io.InputMetaData, '_set_inputs_atr',
                             autospec=True) as mkSet:
            with mk.patch.object(input_io.InputMetaData, 'validate',
                                 autospec=True) as mkValidate:
                main = mk.MagicMock()
                main.attach_mock(mkSet, 'set')
                main.attach_mock(mkValidate, 'validate')

                inputs.optional = value
                assert main.mock_calls == [mk.call.set(inputs, '_optional', value, input_io.OptionalMetaDataEntry),
                                           mk.call.validate(inputs)]
                assert inputs.inputs == []

    def test_get_input_data(self):
        inputs = input_io.InputMetaData.create_defaults(3)

        data = inputs.get_input_data('x0')
        assert isinstance(data, input_io.InputMetaDataEntry)
        assert data.name == 'x0'
        assert data.pos == 0
        assert data._bounding_box is None

        data = inputs.get_input_data('x1')
        assert isinstance(data, input_io.InputMetaDataEntry)
        assert data.name == 'x1'
        assert data.pos == 1
        assert data._bounding_box is None

        data = inputs.get_input_data('x2')
        assert isinstance(data, input_io.InputMetaDataEntry)
        assert data.name == 'x2'
        assert data.pos == 2
        assert data._bounding_box is None

    def test_get_optional_data(self):
        optional = {'z0': input_io.OptionalMetaDataEntry('z0'),
                    'z1': input_io.OptionalMetaDataEntry('z1'),
                    'z2': input_io.OptionalMetaDataEntry('z2')}
        inputs = input_io.InputMetaData.create_defaults(1, optional=optional)
        assert inputs.get_optional_data('z0') == optional['z0']
        assert inputs.get_optional_data('z1') == optional['z1']
        assert inputs.get_optional_data('z2') == optional['z2']

    def test__fill_optional(self):
        optional = {'z0': input_io.OptionalMetaDataEntry('z0', mk.MagicMock()),
                    'z1': input_io.OptionalMetaDataEntry('z1', mk.MagicMock()),
                    'z2': input_io.OptionalMetaDataEntry('z2', mk.MagicMock())}
        inputs = input_io.InputMetaData.create_defaults(1, optional=optional)

        # No user passed, global and model optional disjoint
        kwargs = inputs._fill_optional()
        for name, value in kwargs.items():
            if name in optional:
                assert value == optional[name].default
            elif name in input_io.modeling_options:
                if isinstance(value, float) and np.isnan(value):
                    assert name == 'fill_value'
                else:
                    assert value == input_io.modeling_options[name]
            else:
                assert False
        for name, data in optional.items():
            assert name in kwargs
            assert kwargs[name] == data.default
        for name, value in input_io.modeling_options.items():
            assert name in kwargs
            if isinstance(value, float) and np.isnan(value):
                assert name == 'fill_value'
            else:
                assert kwargs[name] == value
        assert len(kwargs) == len(optional) + len(input_io.modeling_options)

        # User passed, global and model optional disjoint
        input_kwargs = {'z0': mk.MagicMock(), 'fill_value': 0}
        kwargs = inputs._fill_optional(**input_kwargs)
        assert kwargs != input_kwargs
        for name, value in kwargs.items():
            if name in input_kwargs:
                assert value == input_kwargs[name]
            elif name in optional:
                assert value == optional[name].default
            elif name in input_io.modeling_options:
                if isinstance(value, float) and np.isnan(value):
                    assert name == 'fill_value'
                else:
                    assert value == input_io.modeling_options[name]
            else:
                assert False
        for name, data in optional.items():
            assert name in kwargs
            if name in input_kwargs:
                assert kwargs[name] == input_kwargs[name]
            else:
                assert kwargs[name] == data.default
        for name, value in input_io.modeling_options.items():
            assert name in kwargs
            if name in input_kwargs:
                assert kwargs[name] == input_kwargs[name]
            else:
                assert kwargs[name] == value
        assert len(kwargs) == len(optional) + len(input_io.modeling_options)

        optional = {'z0': input_io.OptionalMetaDataEntry('z0', mk.MagicMock()),
                    'z1': input_io.OptionalMetaDataEntry('z1', mk.MagicMock()),
                    'z2': input_io.OptionalMetaDataEntry('z2', mk.MagicMock()),
                    'fill_value': input_io.OptionalMetaDataEntry('fill_value', mk.MagicMock()),
                    'with_bounding_box': input_io.OptionalMetaDataEntry('with_bounding_box', mk.MagicMock())}
        inputs = input_io.InputMetaData.create_defaults(1, optional=optional)

        # No user passed, global and model optional not disjoint
        kwargs = inputs._fill_optional()
        for name, value in kwargs.items():
            if name in optional:
                assert value == optional[name].default
            elif name in input_io.modeling_options:
                assert value == input_io.modeling_options[name]
            else:
                assert False
        for name, data in optional.items():
            assert name in kwargs
            assert kwargs[name] == data.default
        for name, value in input_io.modeling_options.items():
            assert name in kwargs
            if name in optional:
                assert kwargs[name] == optional[name].default
            else:
                assert kwargs[name] == value
        assert len(kwargs) == len(optional) + len(input_io.modeling_options) - 2

        # User passed, global and model optional not disjoint
        input_kwargs = {'z0': mk.MagicMock(), 'fill_value': 0, 'model_set_axis': 2}
        kwargs = inputs._fill_optional(**input_kwargs)
        assert kwargs != input_kwargs
        for name, value in kwargs.items():
            if name in input_kwargs:
                assert value == input_kwargs[name]
            elif name in optional:
                assert value == optional[name].default
            elif name in input_io.modeling_options:
                assert value == input_io.modeling_options[name]
            else:
                assert False
        for name, data in optional.items():
            assert name in kwargs
            if name in input_kwargs:
                assert kwargs[name] == input_kwargs[name]
            else:
                assert kwargs[name] == data.default
        for name, value in input_io.modeling_options.items():
            assert name in kwargs
            if name in input_kwargs:
                assert kwargs[name] == input_kwargs[name]
            elif name in optional:
                assert kwargs[name] == optional[name].default
            else:
                assert kwargs[name] == value
        assert len(kwargs) == len(optional) + len(input_io.modeling_options) - 2

    def test__input_kwargs(self):
        inputs = input_io.InputMetaData.create_defaults(3)

        with mk.patch.object(input_io.InputMetaData, '_fill_optional',
                             autospec=True) as mkFill:
            kwargs = {'x0': 1, 'x2': 2, 'test': 3}
            input_kwargs, optional = inputs._input_kwargs(**kwargs)

            assert optional != kwargs
            assert input_kwargs == {'x0': 1, 'x2': 2}
            assert optional == mkFill.return_value

            assert mkFill.call_args_list == [mk.call(inputs, test=3)]

    def test__get_inputs(self):
        inputs = input_io.InputMetaData.create_defaults(3)
        true_inputs = {f'x{idx}': input_io.InputEntry(f'x{idx}', idx)
                       for idx in range(3)}

        assert true_inputs == inputs._get_inputs(0, 1, 2)
        assert true_inputs == inputs._get_inputs(0, 1, x2=2)
        assert true_inputs == inputs._get_inputs(0, 2, x1=1)
        assert true_inputs == inputs._get_inputs(1, 2, x0=0)
        assert true_inputs == inputs._get_inputs(0, x1=1, x2=2)
        assert true_inputs == inputs._get_inputs(1, x0=0, x2=2)
        assert true_inputs == inputs._get_inputs(2, x0=0, x1=1)
        assert true_inputs == inputs._get_inputs(x0=0, x1=1, x2=2)

    def test__check_inputs(self):
        inputs = input_io.InputMetaData.create_defaults(3)
        assert inputs._n_inputs == 3

        # Too many args
        with pytest.raises(RuntimeError, match=r"Too many .*"):
            inputs._check_inputs(1, 2, 3, 4)
        with pytest.raises(RuntimeError, match=r"Too many .*"):
            inputs._check_inputs(1, 2, 3, a=4)
        with pytest.raises(RuntimeError, match=r"Too many .*"):
            inputs._check_inputs(1, 2, b=3, a=4)
        with pytest.raises(RuntimeError, match=r"Too many .*"):
            inputs._check_inputs(1, c=2, b=3, a=4)
        with pytest.raises(RuntimeError, match=r"Too many .*"):
            inputs._check_inputs(d=1, c=2, b=3, a=4)

        # Too few args
        with pytest.raises(RuntimeError, match=r"Too few .*"):
            inputs._check_inputs()
        with pytest.raises(RuntimeError, match=r"Too few .*"):
            inputs._check_inputs(1)
        with pytest.raises(RuntimeError, match=r"Too few .*"):
            inputs._check_inputs(1, 2)
        with pytest.raises(RuntimeError, match=r"Too few .*"):
            inputs._check_inputs(1, a=2)
        with pytest.raises(RuntimeError, match=r"Too few .*"):
            inputs._check_inputs(b=1, a=2)

    def test__check_optional(self):
        optional = {'z0': input_io.OptionalMetaDataEntry('z0'),
                    'z1': input_io.OptionalMetaDataEntry('z1'),
                    'z2': input_io.OptionalMetaDataEntry('z2')}
        inputs = input_io.InputMetaData.create_defaults(1, optional=optional)
        assert not inputs._pass_optional

        # Passes when not pass_through
        inputs._check_optional()
        kwargs = input_io.modeling_options.copy()
        inputs._check_optional(**kwargs)
        for name in optional:
            kwargs[name] = mk.MagicMock()
            inputs._check_optional(**kwargs)

        # Fail when not pass_through
        kwargs['test'] = mk.MagicMock()
        with pytest.raises(RuntimeError):
            inputs._check_optional(**kwargs)

        # No fail when pass optional enabled
        inputs._pass_optional = True
        inputs._check_optional(**kwargs)

    def test_evaluation_inputs(self):
        optional = {'z0': input_io.OptionalMetaDataEntry('z0', 'z0'),
                    'z1': input_io.OptionalMetaDataEntry('z1', 'z0'),
                    'z2': input_io.OptionalMetaDataEntry('z2', 'z0')}
        inputs = input_io.InputMetaData.create_defaults(3, optional=optional)
        true_inputs = {f'x{idx}': input_io.InputEntry(f'x{idx}', idx)
                       for idx in range(3)}

        # No optional
        true_optional = input_io.modeling_options.copy()
        true_eval = input_io.Inputs(true_inputs, true_optional, [])
        for name, value in optional.items():
            true_optional[name] = value.default
        assert inputs.evaluation_inputs(0, 1, 2)          == true_eval
        assert inputs.evaluation_inputs(0, 1, x2=2)       == true_eval
        assert inputs.evaluation_inputs(0, 2, x1=1)       == true_eval
        assert inputs.evaluation_inputs(1, 2, x0=0)       == true_eval
        assert inputs.evaluation_inputs(0, x1=1, x2=2)    == true_eval
        assert inputs.evaluation_inputs(1, x0=0, x2=2)    == true_eval
        assert inputs.evaluation_inputs(2, x0=0, x1=1)    == true_eval
        assert inputs.evaluation_inputs(x0=0, x1=1, x2=2) == true_eval

        # Optional
        true_optional['z0'] = 0
        true_eval = input_io.Inputs(true_inputs, true_optional, [])
        assert inputs.evaluation_inputs(0, 1, 2, z0=0) == true_eval

    def test__get_outside(self):
        entries = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        inputs = input_io.Inputs(entries, mk.MagicMock(), mk.MagicMock())
        input_data = input_io.InputMetaData(3)
        inputs_data = {f'x{idx}': mk.MagicMock() for idx in range(3)}
        input_data._inputs = inputs_data

        # Test get outside inputs
        for name in entries:
            assert input_data._get_outside(inputs, name, True) ==\
                (inputs_data[name].outside.return_value, entries[name].input_array.shape)
            assert inputs_data[name].outside.call_args_list == \
                [mk.call(entries[name].input_array)]

            assert input_data._get_outside(inputs, name, False) ==\
                (inputs_data[name].outside.return_value, entries[name].input.shape)
            assert inputs_data[name].outside.call_args_list == \
                [mk.call(entries[name].input_array), mk.call(entries[name].input)]

        # Test Error check
        with pytest.raises(RuntimeError):
            input_data._get_outside(inputs, mk.MagicMock(), mk.MagicMock())

    def test__update_outside_inputs(self):
        input_data = input_io.InputMetaData(3)
        inputs = mk.MagicMock()
        name = mk.MagicMock()
        array_shape = mk.MagicMock()

        # Test array input
        outside_inputs = np.zeros((5,), dtype=bool)
        return_data = [np.array([True, False, False, True, True]), (1,)]
        with mk.patch.object(input_io.InputMetaData, '_get_outside', autospec=True,
                             return_value=return_data) as mkGet:
            values, all_out = input_data._update_outside_inputs(outside_inputs,
                                                                False, inputs,
                                                                name, array_shape)
            assert (values == return_data[0]).all()
            assert not all_out
            assert mkGet.call_args_list == \
                [mk.call(input_data, inputs, name, array_shape)]
            mkGet.reset_mock()

            values, all_out = input_data._update_outside_inputs(values,
                                                                False, inputs,
                                                                name, array_shape)
            assert (values == return_data[0]).all()
            assert not all_out
            assert mkGet.call_args_list == \
                [mk.call(input_data, inputs, name, array_shape)]

        # Test scalar input
        outside_inputs = np.zeros((5,), dtype=bool)
        return_data = [np.array([True, False, False, True, True]), ()]
        with mk.patch.object(input_io.InputMetaData, '_get_outside', autospec=True,
                             return_value=return_data) as mkGet:
            values, all_out = input_data._update_outside_inputs(outside_inputs,
                                                                False, inputs,
                                                                name, array_shape)
            assert (values == return_data[0]).all()
            assert not all_out
            assert mkGet.call_args_list == \
                [mk.call(input_data, inputs, name, array_shape)]

        outside_inputs = np.zeros((5,), dtype=bool)
        return_data = [np.asanyarray(True), ()]
        with mk.patch.object(input_io.InputMetaData, '_get_outside', autospec=True,
                             return_value=return_data) as mkGet:
            values, all_out = input_data._update_outside_inputs(outside_inputs,
                                                                False, inputs,
                                                                name, array_shape)
            assert values.all()
            assert all_out
            assert mkGet.call_args_list == \
                [mk.call(input_data, inputs, name, array_shape)]

    def test__outside_inputs(self):
        input_data = input_io.InputMetaData.create_defaults(3)
