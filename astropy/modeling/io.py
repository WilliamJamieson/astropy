from astropy.units import equivalencies
import numpy as np
from typing import Dict, List, Tuple
from collections import UserDict

ordinalities = ['continuous', 'discrete']
modeling_options = {
    'model_set_axis': None,
    'with_bounding_box': False,
    'fill_value': np.nan,
    'equivalencies': None,
    'inputs_map': None
}


class IoMetaDataEntry(object):
    def __init__(self, name: str=None):
        self._name = name

    @property
    def name(self) -> str:
        if self._name is None:
            raise RuntimeError('Data entry has no name specified')
        else:
            return self._name

    @name.setter
    def name(self, value):
        if self._name is None:
            self._name = value
        else:
            raise ValueError(f'Data entry has name already specified')

    @classmethod
    def create_entry(cls, input_value, **kwargs):
        raise NotImplementedError


class InputMetaDataEntry(IoMetaDataEntry):
    def __init__(self, name: str=None, pos: int=None):
        super().__init__(name)

        self._pos = pos

    @property
    def pos(self) -> int:
        if self._pos is None:
            raise RuntimeError(f'Input {self.name} has no position specified')
        else:
            return self._pos

    @pos.setter
    def pos(self, value: int):
        if self._pos is None:
            self._pos = value
        else:
            raise ValueError(f'Input {self.name} has position specified already')

    @classmethod
    def create_entry(cls, input_value, name=None, pos=None):
        if isinstance(input_value, InputMetaDataEntry):
            return input_value
        elif isinstance(input_value, tuple):
            return cls(name, *input_value)
        elif isinstance(input_value, str):
            return cls(input_value, pos)
        else:
            raise ValueError(f'{input_value} is not a valid way to set an input')


class OptionalMetaDataEntry(IoMetaDataEntry):
    def __init__(self, name: str=None, default=None, ordinality: str='continuous'):
        super().__init__(name)
        self._default = default
        self.ordinality = ordinality

    @property
    def ordinality(self) -> str:
        return self._ordinality

    @ordinality.setter
    def ordinality(self, value: str):
        if value in ordinalities:
            self._ordinality = value
        else:
            raise ValueError(f'{value} is not one of {ordinalities}')

    @property
    def default(self):
        return self._default

    @classmethod
    def create_entry(cls, input_value, name=None, pos=None):
        if isinstance(input_value, InputMetaDataEntry):
            return input_value
        elif isinstance(input_value, tuple):
            return cls(name, *input_value)
        elif isinstance(input_value, str):
            return cls(input_value)
        else:
            raise ValueError(f'{input_value} is not a valid way to set an input')


class InputMetaData(object):
    def __init__(self, n_inputs: int,
                 inputs: Dict[str, InputMetaDataEntry]=None,
                 optional: Dict[str, OptionalMetaDataEntry]=None):
        self._n_inputs = n_inputs

        self._inputs = None
        self._optional = None

        self.inputs = inputs
        self.optional = optional

    @classmethod
    def create_defaults(cls, n_inputs: int, optional=None):
        new = cls(n_inputs, optional=optional)

        if n_inputs == 1:
            new.inputs = ['x']
        elif n_inputs == 2:
            new.inputs = ['x', 'y']
        else:
            new.inputs = [f'x{idx}' for idx in range(n_inputs)]

        return new

    def validate(self):
        if self._inputs is not None:
            if self._n_inputs != len(self._inputs):
                raise ValueError('n_inputs must match the number of entries in inputs.')

            for pos, (input_name, input_data) in enumerate(self._inputs.items()):
                if pos != input_data.pos:
                    raise ValueError(f"Input: {input_data.name}'s position is information is incorrect.'")
                if input_name != input_data.name:
                    raise ValueError(f"Input: {input_data.name}'s key information is incorrect'")

        if self._optional is not None:
            for input_name, input_data in self._optional.items():
                if input_name != input_data.name:
                    raise ValueError(f"Input: {input_data.name}'s key information is incorrect'")
                if (self._inputs is not None) and (input_name in self._inputs):
                    raise ValueError(f"Input: {input_name} is both optional and non-optional")

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @n_inputs.setter
    def n_inputs(self, value):
        self._n_inputs = value

    @property
    def inputs(self) -> List[str]:
        return list(self._inputs.keys())

    @property
    def optional(self) -> List[str]:
        return list(self._optional.keys())

    @staticmethod
    def _process_inputs(value, data_entry):
        inputs = {}
        if isinstance(value, list):
            for index, input_value in enumerate(value):
                entry = data_entry.create_entry(input_value, pos=index)
                inputs[entry.name] = entry
        elif isinstance(value, dict):
            for name, input_value in value.items():
                entry = data_entry.create_entry(input_value, name=name)
                inputs[entry.name] = entry
        else:
            raise ValueError(f'{value} is not a valid way to set inputs')

        return inputs

    @inputs.setter
    def inputs(self, value):
        if self._inputs is None:
            if value is not None:
                self._inputs = self._process_inputs(value, InputMetaDataEntry)
                self.validate()
        else:
            raise RuntimeError('Attempting to override currently set inputs')

    @optional.setter
    def optional(self, value):
        if self._optional is None:
            if value is not None:
                self._optional = self._process_inputs(value, OptionalMetaDataEntry)
                self.validate()
        else:
            raise RuntimeError('Attempting to override currently set optional inputs')

    def get_input_data(self, name: str) -> InputMetaDataEntry:
        return self._inputs[name]

    def get_optional_data(self, name: str) -> OptionalMetaDataEntry:
        return self._optional[name]

    def _input_kwargs(self, **kwargs) -> Tuple[dict, dict]:
        input_kwargs = {}

        for name in self._inputs:
            if name in kwargs:
                input_kwargs[name] = kwargs[name]
                del kwargs[name]

        optional = kwargs
        for name, optional_input in self._optional.items():
            if name not in optional:
                optional[name] = optional_input.default
        for name, value in modeling_options.items():
            if name not in optional:
                optional[name] = value

        return input_kwargs, optional

    def evaluation_inputs(self, *args, **kwargs) -> Tuple[dict, dict]:
        args = list(args)
        input_kwargs, optional = self._input_kwargs(**kwargs)

        n_args = len(args) + len(input_kwargs)
        if self._n_inputs < n_args:
            raise RuntimeError(f'Too many input arguments - expected {self._n_inputs}, got {n_args}')
        elif self._n_inputs > n_args:
            raise RuntimeError(f'Too few input arguments - expected {self._n_inputs}, got {n_args}')

        inputs = {}
        for name in self._inputs:
            if name in input_kwargs:
                inputs[name] = InputEntry(input_kwargs[name])
            else:
                inputs[name] = InputEntry(args.pop(0))

        return inputs, optional


class InputEntry(object):
    def __init__(self, input_value, format_info=None):
        self.input = input_value
        self._format_info = format_info

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = np.asanyarray(value, dtype=float)

    @property
    def format_info(self):
        return self._format_info

    @format_info.setter
    def format_info(self, value):
        self._format_info = value


class Inputs(object):
    def __init__(self, inputs: Dict[str, InputEntry], kwargs: Dict[str, InputEntry]):
        self._inputs = inputs
        self._kwargs = kwargs
