# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Tests for evaluation IO for Models"""

import pytest
import numpy as np
import unittest.mock as mk

from astropy.modeling import input_io


class TestIoMetaDataEntry(object):
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
        assert entry._ordinality == 'continuous'

        entry = input_io.InputMetaDataEntry('test', 1, 'discrete')
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._ordinality == 'discrete'

        with pytest.raises(ValueError):
            input_io.InputMetaDataEntry(ordinality='test')

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

    def test_ordinality(self):
        # test get
        entry = input_io.InputMetaDataEntry()
        assert entry.ordinality == 'continuous'
        assert entry._ordinality == 'continuous'
        entry.ordinality = 'discrete'
        assert entry.ordinality == 'discrete'
        assert entry._ordinality == 'discrete'

        entry = input_io.InputMetaDataEntry(ordinality='discrete')
        assert entry.ordinality == 'discrete'
        assert entry._ordinality == 'discrete'
        entry.ordinality = 'continuous'
        assert entry.ordinality == 'continuous'
        assert entry._ordinality == 'continuous'

        # test set with error
        with pytest.raises(ValueError):
            entry.ordinality = 'test'
        entry.ordinality = 'continuous'
        entry._ordinality = 'continuous'

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
        entry = input_io.InputMetaDataEntry.create_entry((1, 'discrete'))
        assert entry._name is None
        assert entry._pos == 1
        assert entry._ordinality == 'discrete'
        entry = input_io.InputMetaDataEntry.create_entry((1, 'discrete'), name='test')
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._ordinality == 'discrete'
        entry = input_io.InputMetaDataEntry.create_entry((1, 'discrete'), name='test', pos=2)
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._ordinality == 'discrete'
        entry = input_io.InputMetaDataEntry.create_entry((1,))
        assert entry._name is None
        assert entry._pos == 1
        assert entry._ordinality == 'continuous'
        entry = input_io.InputMetaDataEntry.create_entry((1,), name='test')
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._ordinality == 'continuous'
        entry = input_io.InputMetaDataEntry.create_entry((1,), name='test', pos=2)
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._ordinality == 'continuous'

        # test pass str in
        entry = input_io.InputMetaDataEntry.create_entry('test')
        assert entry._name == 'test'
        assert entry._pos is None
        assert entry._ordinality == 'continuous'
        entry = input_io.InputMetaDataEntry.create_entry('test', name=mk.MagicMock())
        assert entry._name == 'test'
        assert entry._pos is None
        assert entry._ordinality == 'continuous'
        entry = input_io.InputMetaDataEntry.create_entry('test', name=mk.MagicMock(), pos=1)
        assert entry._name == 'test'
        assert entry._pos == 1
        assert entry._ordinality == 'continuous'

        # test pass bad input
        with pytest.raises(ValueError):
            input_io.InputMetaDataEntry.create_entry(mk.MagicMock())


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
        assert inputs._inputs['x'].ordinality == 'continuous'

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
        assert inputs._inputs['x'].ordinality == 'continuous'
        assert 'y' in inputs._inputs
        assert inputs._inputs['y'].name == 'y'
        assert inputs._inputs['y'].pos == 1
        assert inputs._inputs['y'].ordinality == 'continuous'

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
        assert inputs._inputs['x0'].ordinality == 'continuous'
        assert 'x1' in inputs._inputs
        assert inputs._inputs['x1'].name == 'x1'
        assert inputs._inputs['x1'].pos == 1
        assert inputs._inputs['x1'].ordinality == 'continuous'
        assert 'x2' in inputs._inputs
        assert inputs._inputs['x2'].name == 'x2'
        assert inputs._inputs['x2'].pos == 2
        assert inputs._inputs['x2'].ordinality == 'continuous'

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
        # Test get
        inputs = input_io.InputMetaData(1)
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
        assert data.ordinality == 'continuous'

        data = inputs.get_input_data('x1')
        assert isinstance(data, input_io.InputMetaDataEntry)
        assert data.name == 'x1'
        assert data.pos == 1
        assert data.ordinality == 'continuous'

        data = inputs.get_input_data('x2')
        assert isinstance(data, input_io.InputMetaDataEntry)
        assert data.name == 'x2'
        assert data.pos == 2
        assert data.ordinality == 'continuous'

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
