# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Evaluation IO for Models"""

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
        raise NotImplementedError('IoMetaDataEntry does not implement this!')


class InputMetaDataEntry(IoMetaDataEntry):
    def __init__(self, name: str=None, pos: int=None, ordinality: str='continuous'):
        super().__init__(name)

        self._pos = pos
        self.ordinality = ordinality

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

    @property
    def ordinality(self) -> str:
        return self._ordinality

    @ordinality.setter
    def ordinality(self, value: str):
        if value in ordinalities:
            self._ordinality = value
        else:
            raise ValueError(f'{value} is not one of {ordinalities}')

    @classmethod
    def create_entry(cls, input_value, *, name=None, pos=None):
        if isinstance(input_value, InputMetaDataEntry):
            return input_value
        elif isinstance(input_value, tuple):
            return cls(name, *input_value)
        elif isinstance(input_value, str):
            return cls(input_value, pos)
        else:
            raise ValueError(f'{input_value} is not a valid way to set an input')


class OptionalMetaDataEntry(IoMetaDataEntry):
    def __init__(self, name: str=None, default=None):
        super().__init__(name)

        self._default = default

    @property
    def default(self):
        return self._default

    @classmethod
    def create_entry(cls, input_value, *, name=None, pos=None):
        if isinstance(input_value, OptionalMetaDataEntry):
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
                 optional: Dict[str, OptionalMetaDataEntry]=None,
                 pass_optional: bool=False):
        self._n_inputs = n_inputs
        self._pass_optional = pass_optional

        self._inputs: Dict[str, InputMetaDataEntry] = {}
        self._optional: Dict[str, OptionalMetaDataEntry] = {}

        if inputs is not None:
            self.inputs = inputs
        if optional is not None:
            self.optional = optional

    def _fill_defaults(self):
        if self._n_inputs == 1:
            self.inputs = ['x']
        elif self.n_inputs == 2:
            self.inputs = ['x', 'y']
        else:
            self.inputs = [f'x{idx}' for idx in range(self.n_inputs)]

    @classmethod
    def create_defaults(cls, n_inputs: int, optional=None, pass_optional=False):
        new = cls(n_inputs, optional=optional, pass_optional=pass_optional)
        new._fill_defaults()

        return new

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @n_inputs.setter
    def n_inputs(self, value):
        self._n_inputs = value
        self._fill_defaults()

    @property
    def pass_optional(self) -> bool:
        return self._pass_optional

    @pass_optional.setter
    def pass_optional(self, value):
        self._pass_optional = value

    @property
    def inputs(self) -> List[str]:
        return list(self._inputs.keys())

    @property
    def optional(self) -> List[str]:
        return list(self._optional.keys())

    def reset_inputs(self):
        self._inputs = {}

    def reset_optional(self):
        self._optional = {}

    def validate(self):
        if (len(self._inputs) > 0) and (self._n_inputs != len(self._inputs)):
            raise ValueError('n_inputs must match the number of entries in inputs.')

        for pos, (input_name, input_data) in enumerate(self._inputs.items()):
            if input_name != input_data.name:
                raise ValueError(f"Input: {input_data.name}'s key information is incorrect'")
            if pos != input_data.pos:
                raise ValueError(f"Input: {input_data.name}'s position is information is incorrect.'")

        for input_name, input_data in self._optional.items():
            if input_name != input_data.name:
                raise ValueError(f"Optional: {input_data.name}'s key information is incorrect'")
            if (self._inputs is not None) and (input_name in self._inputs):
                raise ValueError(f"Optional: {input_name} is both optional and non-optional")

    @staticmethod
    def _process_inputs_atr(value, data_entry) -> dict:
        inputs = {}
        if value is not None:
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

    def _set_inputs_atr(self, atr: str, value, data_entry):
        if hasattr(self, atr):
            input_atr = getattr(self, atr)
            if len(input_atr) != 0:
                raise RuntimeError(f'Attempting to override currently set {atr[1:]}')
        else:
            raise AttributeError('Trying to set non-existent attribute')

        setattr(self, atr, self._process_inputs_atr(value, data_entry))

    @inputs.setter
    def inputs(self, value):
        self._set_inputs_atr('_inputs', value, InputMetaDataEntry)
        self.validate()

    @optional.setter
    def optional(self, value):
        self._set_inputs_atr('_optional', value, OptionalMetaDataEntry)
        self.validate()

    def get_input_data(self, name: str) -> InputMetaDataEntry:
        return self._inputs[name]

    def get_optional_data(self, name: str) -> OptionalMetaDataEntry:
        return self._optional[name]

    def _fill_optional(self, **kwargs) -> dict:
        for name, optional_input in self._optional.items():
            if name not in kwargs:
                kwargs[name] = optional_input.default
        for name, value in modeling_options.items():
            if name not in kwargs:
                kwargs[name] = value

        return kwargs

    def _input_kwargs(self, **kwargs) -> Tuple[dict, dict]:
        input_kwargs = {}

        for name in self._inputs:
            if name in kwargs:
                input_kwargs[name] = kwargs[name]
                del kwargs[name]

        optional = self._fill_optional(**kwargs)

        return input_kwargs, optional

    def _get_inputs(self, *args, **kwargs) -> dict:
        args = list(args)
        inputs = {}
        for name in self._inputs:
            if name in kwargs:
                inputs[name] = InputEntry(kwargs[name])
            else:
                inputs[name] = InputEntry(args.pop(0))

        return inputs

    def _check_inputs(self, *args, **kwargs):
        n_args = len(args) + len(kwargs)
        if self._n_inputs < n_args:
            raise RuntimeError(f'Too many input arguments - expected {self._n_inputs}, got {n_args}')
        elif self._n_inputs > n_args:
            raise RuntimeError(f'Too few input arguments - expected {self._n_inputs}, got {n_args}')

    def _check_optional(self, **kwargs):
        if not self._pass_optional:
            for name in kwargs:
                if not ((name in self._optional) or (name in modeling_options)):
                    raise RuntimeError(f'Keyword: {name} has been passed, no undocumented arguments can be passed through!')

    def evaluation_inputs(self, *args, **kwargs) -> Tuple[dict, dict]:
        input_kwargs, optional = self._input_kwargs(**kwargs)
        self._check_optional(**optional)

        self._check_inputs(*args, **input_kwargs)
        inputs = self._get_inputs(*args, **input_kwargs)

        return inputs, optional


class InputEntry(object):
    def __init__(self, input_value, format_info=None):
        self.input = input_value
        self._format_info = format_info

    def __eq__(self, this):
        if isinstance(this, InputEntry):
            return (self.input == this.input).all() and (self.format_info == this.format_info)
        else:
            return False

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
