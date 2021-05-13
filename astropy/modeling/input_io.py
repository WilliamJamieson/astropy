# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Evaluation IO for Models"""

import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy

from astropy.utils.shapes import check_broadcast, IncompatibleShapeError
from astropy.modeling.utils import _BoundingBox

modeling_options = {
    'model_set_axis': None,
    'with_bounding_box': False,
    'fill_value': np.nan,
    'equivalencies': None,
    'inputs_map': None
}


class InputEntry(object):
    """
    Class to contain a single required input to an `~astropy.modeling.Model`

    Parameters
    ----------
    name : str
        The model's name for the evaluation input
    input_value : np.ndarray
        Contains the value the user passed to model for evaluation

    Methods
    -------
    check_input_shape: Returns error checked shape for this input.
    broadcast: Returns the broadcast shape of this input.
    reduce_to_bounding_box: Update this input to just consider the array
        entries computed to be inside the bounding box.

    Notes
    -----
    - self.input can be used to directly set and access the input. Note
      that when using this for setting the value,
        `np.asanyarray(*, dtype=float)`
      is called to convert the value into a float np.ndarray (possibly
      a scalar array). This is consistent with the requirement that all
      required inputs to `~astropy.modeling.Model` based models use
      numpy float arrays as inputs.
    - self.input_array is a way to access self.input (without modifying
      it) so that the result is always a vector. That is it returns the
      input if the value is non-scalar and converts scalar values to `(1,)`
      shaped arrays. This is so that numpy input broadcasting can be
      supported between scalar and array inputs.
    - self.shape returns the stored shape of the input.
    """
    def __init__(self, name: str, input_value):
        self._name = name
        self.input = input_value

    def __eq__(self, this):
        if isinstance(this, InputEntry):
            return (self.name == this.name) and (self.input == this.input).all()
        else:
            return False

    @property
    def name(self) -> str:
        return self._name

    @property
    def input(self) -> np.ndarray:
        return self._input

    @input.setter
    def input(self, value):
        self._input = np.asanyarray(value, dtype=float)

    @property
    def shape(self) -> tuple:
        return self._input.shape

    @property
    def input_array(self) -> np.ndarray:
        # NOTE: this method is for replacing _prepare_inputs_single_model

        # Ensure that array scalars are always upgrade to 1-D arrays for the
        # sake of consistency with how parameters work.  They will be cast back
        # to scalars at the end
        if not self.shape:
            return self._input.reshape((1,))
        else:
            return self._input

    def _array_shape(self, array_shape: bool) -> tuple:
        """
        This is a helper method for self.check_input_shape.
            It replicates the differences between shape checks (using
            _validate_input_shapes) performed during current model
            evaluation calls. First call assumes scalars are scalars,
            while the second call assumes scalars have been reshaped
            into `(1,)` shaped arrays.

        Parameters
        ----------
        array_shape : bool
            Determines which mode to use.
                True for converted array
                False for scalars remaining un-reshaped

        Returns
        -------
            The required shape tuple
        """
        if array_shape:
            # For call in generic_call
            return self.input_array.shape
        else:
            # Call in Model.prepare_inputs
            return self.shape

    def check_input_shape(self, n_models: int, model_set_axis: int, array_shape: bool) -> tuple:
        """
        This is the single input level computation to check the shape of an
        input.

        Parameters
        ----------
        n_models : int
            Number of models (for model set)
        model_set_axis : int
            Model option to set which dimension of input array is used
            when evaluating model set.
        array_shape : bool
            Determines which mode to use.
                True for converted array
                False for scalars remaining un-reshaped

        Returns
        -------
            An error checked input shape.

        Notes
        -----
        See Inputs.check_input_shape for cross checking of all model
        evaluation inputs against one another
        """
        # NOTE: this method is for replacing _validate_input_shapes

        # NOTE: this is currently in place to exactly replicate the two
        #       calls to _validate_input_shapes

        shape = self._array_shape(array_shape)

        # Ensure that the input's model_set_axis matches the model's
        # n_models
        if shape and (n_models > 1) and (model_set_axis is not False):
            # Note: Scalar inputs *only* get a pass on this
            if len(shape) < model_set_axis + 1:
                raise ValueError(f"For model_set_axis={model_set_axis}, " +
                                 f"all inputs must be at least {model_set_axis + 1}-dimensional.")
            if shape[model_set_axis] != n_models:
                raise ValueError(f"Input argument {self._name} does not have the " +
                                 f"correct dimensions in model_set_axis={model_set_axis} "+
                                 f"for a model set with n_models={n_models}.")

        return shape

    def _get_param_broadcast(self, param, standard_broadcasting: bool) -> tuple:
        """
        Helper method for self._update_param_broadcast.
            This method wraps `~astropy.utils.shapes.check_broadcast` with
            a better error message. It also allows for skipping this check
            if the model is not using standard_broadcasting.

        Parameters
        ----------
        param :
            A model parameter to get the broadcast shape of this evaluation
            input against.
        standard_broadcasting : bool
            Whether or not standard_broadcasting is used by the model.

        Returns
        -------
            broadcast shape of this evaluation input relative to the
            model parameter.
        """
        try:
            if standard_broadcasting:
                return check_broadcast(self.shape, param.shape)
            else:
                return self.shape
        except IncompatibleShapeError:
            raise ValueError(f"Model input argument {self.name} of shape {self.shape} cannot be " +
                             f"broadcast with parameter {param.name} of shape {param.shape}.")

    def _update_param_broadcast(self, broadcast: tuple, param, standard_broadcasting: bool) -> tuple:
        """
        Helper method for self.broadcast.
            Updates this evaluation input's broadcast shape based on the
            passed in model parameter.

        Parameters
        ----------
        broadcast : tuple
            Currently computed broadcast shape
        param :
            A model parameter to get the broadcast shape of this evaluation
            input against.
        standard_broadcasting : bool
            Whether or not standard_broadcasting is used by the model.

        Returns
        -------
            An updated broadcast shape based on the current parameter
        """
        new_broadcast = self._get_param_broadcast(param, standard_broadcasting)

        if  len(new_broadcast) > len(broadcast):
            return new_broadcast
        elif len(new_broadcast) == len(broadcast):
            return max(broadcast, new_broadcast)
        else:
            return broadcast

    def broadcast(self, params: list, standard_broadcasting: bool) -> tuple:
        """
        Determines the broadcast shape of this evaluation input relative
        to all of the model's parameters.

        Parameters
        ----------
        params : list
            A list of all the model's parameter to get this evaluation
            input's broadcast shape in relation to
        standard_broadcasting : bool
            Whether or not standard_broadcasting is used by the model.

        Returns
        -------
            The broadcast shape of this input for the model.

        Notes
        -----
        This is used to generate the format_info for the inputs, which
        will be used to generate the post-evaluation output shapes.
        """
        # NOTE: this method is for replacing _prepare_inputs_single_model

        if not params:
            broadcast = self.shape
        else:
            broadcast = ()

        for param in params:
            broadcast = self._update_param_broadcast(broadcast, param, standard_broadcasting)

        return broadcast

    def reduce_to_bounding_box(self, valid_index, array_shape: bool) -> 'InputEntry':
        """
        Reduces this evaluation input to just the indices computed to
        be inside the model's bounding box.

        Parameters
        ----------
        valid_index : Tuple[np.ndarray, ...]
            The indices of this input found to be corrisponding to points
            within the bounding box of the model
        array_shape : bool
            Determines which mode to use.
                True for converted array
                False for scalars remaining un-reshaped

        Returns
        -------
        A new input entry which has been reduced to just the valid_index
        locations

        Notes
        -----
        See Inputs.reduce_to_bounding_box for collective input computation
        of bounding box reduction.
        """
        # Note, pretty sure array_shape is always true (need to check this)
        if array_shape:
            input_value = self.input_array
        else:
            input_value = self.input

        return InputEntry(self._name, np.array(input_value)[valid_index])


class Inputs(object):
    def __init__(self, inputs: Dict[str, InputEntry], optional: dict,
                 model_options: dict=modeling_options, pass_through: dict={},
                 format_info: list=None, valid_index: np.ndarray=None, all_out: bool=None):
        self._inputs = inputs
        self._optional = optional

        self.model_options = model_options
        self._pass_through = pass_through

        self._format_info = format_info
        self._valid_index = valid_index
        self._all_out = all_out

    def __eq__(self, this):
        if isinstance(this, Inputs):
            return (self.inputs == this.inputs) and \
                (self.optional == this.optional) and \
                (self.model_options == this.model_options) and \
                (self.format_info == this.format_info) and \
                (self.pass_through == this.pass_through) and \
                (self.valid_index == this.valid_index).all() and \
                (self.all_out == this.all_out)
        else:
            return False

    @property
    def n_inputs(self) -> int:
        return len(self._inputs)

    @property
    def inputs(self) -> Dict[str, InputEntry]:
        return self._inputs

    @property
    def optional(self) -> dict:
        return self._optional

    @property
    def model_options(self) -> dict:
        return self._model_options

    @model_options.setter
    def model_options(self, value: dict):
        for name in modeling_options:
            if name not in value:
                raise ValueError(f"Modeling option {name} must be set!")
        else:
            self._model_options = value

    def _get_model_option(self, name: str):
        if name in self._model_options:
            return self._model_options[name]
        else:
            raise RuntimeError(f'Option "{name}" must be set!')

    @property
    def model_set_axis(self):
        return self._get_model_option('model_set_axis')

    @property
    def with_bounding_box(self):
        return self._get_model_option('with_bounding_box')

    @property
    def fill_value(self):
        return self._get_model_option('fill_value')

    @property
    def equivalencies(self):
        return self._get_model_option('equivalencies')

    @property
    def inputs_map(self):
        return self._get_model_option('inputs_map')

    @property
    def pass_through(self) -> dict:
        return self._pass_through

    @property
    def format_info(self) -> list:
        if self._format_info is None:
            return []
        else:
            return self._format_info

    @property
    def valid_index(self) -> np.ndarray:
        if self._valid_index is None:
            return np.empty(0)
        else:
            return self._valid_index

    @valid_index.setter
    def valid_index(self, value):
        self._valid_index = value

    @property
    def all_out(self) -> bool:
        if self._all_out is None:
            return False
        else:
            return self._all_out

    @all_out.setter
    def all_out(self, value):
        self._all_out = value

    def copy(self) -> 'Inputs':
        return deepcopy(self)

    def check_input_shape(self, n_models: int, array_shape: bool):
        # NOTE: this method is for replacing _validate_input_shapes
        input_shape = check_broadcast(*[_input.check_input_shape(n_models, self.model_set_axis, array_shape)
                                        for _input in self._inputs.values()])
        if input_shape is None:
            raise ValueError("All inputs must have identical shapes or must be scalars.")

        return input_shape

    def _broadcast(self, params: list, standard_broadcasting: bool, n_outputs: int) -> list:
        # NOTE: this method is for replacing _prepare_inputs_single_model

        # TODO: could this be a dictionary?
        broadcasts = [_input.broadcast(params, standard_broadcasting) for _input in self._inputs.values()]

        if n_outputs > self.n_inputs:
            extra_outputs = n_outputs - self.n_inputs
            if not broadcasts:
                # If there were no inputs then the broadcasts list is empty
                # just add a None since there is no broadcasting of outputs and
                # inputs necessary (see _prepare_outputs_single_model)

                # TODO: check for broadcast None checks
                broadcasts.append(tuple())
            # TODO: check why its always the first one
            broadcasts.extend([broadcasts[0]] * extra_outputs)

        return broadcasts

    def get_format_info(self, n_models: int, params: list,
                        standard_broadcasting: bool, n_outputs: int):
        # Note: this method is for replacing ~Model.prepare_inputs

        self.check_input_shape(n_models, False)
        # TODO: add units handling

        self._format_info = self._broadcast(params, standard_broadcasting, n_outputs)

    def _reduce_to_bounding_box(self, valid_index, all_out: bool, array_shape: bool) -> 'Inputs':
        inputs = {}
        for name, _input in self._inputs.items():
            inputs[name] = _input.reduce_to_bounding_box(valid_index, array_shape)

        return Inputs(inputs, self.optional, self.model_options, self.pass_through,
                      self.format_info, valid_index, all_out)

    def reduce_to_bounding_box(self, valid_index, all_out: bool, array_shape: bool) -> 'Inputs':
        # if not all inputs are not in bounding box, adjust them
        if all_out:
            new = self.copy()
            new.valid_index = valid_index
            new.all_out = all_out
            return new
        else:
            return self._reduce_to_bounding_box(valid_index, all_out, array_shape)

    def prepare_inputs(self, n_models: int, array_shape: bool):
        # equivalent of _validate_input_shapes
        self.check_input_shape(n_models, array_shape)

        # TODO: do unit checking?


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
    def __init__(self, name: str=None, pos: int=None, bounding_box: _BoundingBox=None):
        super().__init__(name)

        self._pos = pos
        self.bounding_box = bounding_box

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
    def bounding_box(self) -> _BoundingBox:
        if self._bounding_box is None:
            raise NotImplementedError(f'No bounding_box has been assigned to input {self._name}')
        else:
            return self._bounding_box

    @bounding_box.setter
    def bounding_box(self, value):
        if value is None:
            self._bounding_box = None
        else:
            self._bounding_box = _BoundingBox(value)

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

    def outside(self, value):
        return self.bounding_box.outside(value)


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
                 n_models: int=1,
                 n_outputs: int=1,
                 standard_broadcasting: bool=True,
                 bounding_box: _BoundingBox=None,
                 pass_optional: bool=False):
        self._n_inputs = n_inputs
        self._n_models = n_models
        self._n_outputs = n_outputs
        self._standard_broadcasting = standard_broadcasting
        self._pass_optional = pass_optional

        self._inputs: Dict[str, InputMetaDataEntry] = {}
        self._optional: Dict[str, OptionalMetaDataEntry] = {}

        if inputs is not None:
            self.inputs = inputs
        if optional is not None:
            self.optional = optional

        self.bounding_box = bounding_box

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
    def n_models(self) -> int:
        return self._n_models

    @n_models.setter
    def n_models(self, value):
        self._n_models = value

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @n_outputs.setter
    def n_outputs(self, value):
        self._n_outputs = value

    @property
    def standard_broadcasting(self) -> bool:
        return self._standard_broadcasting

    @standard_broadcasting.setter
    def standard_broadcasting(self, value):
        self._standard_broadcasting = value

    @property
    def pass_optional(self) -> bool:
        return self._pass_optional

    @pass_optional.setter
    def pass_optional(self, value):
        self._pass_optional = value

    @property
    def bounding_box(self) -> _BoundingBox:
        if self._bounding_box is None:
            raise NotImplementedError('No bounding_box has been assigned')
        else:
            return self._bounding_box

    def _reverse_bounding_box(self):
        if self.n_inputs > 1:
            return self.bounding_box[::-1]
        else:
            return [self.bounding_box]

    def _distribute_bounding_box(self):
        bbox = self._reverse_bounding_box()

        for _input in self._inputs.values():
            _input.bounding_box = bbox[_input.pos]

    @bounding_box.setter
    def bounding_box(self, value):
        if value is None:
            self._bounding_box = value
        else:
            if (self.n_inputs == 1 and len(value) == 2) or \
                    (self.n_inputs > 1 and len(value) == self.n_inputs):
                self._bounding_box = _BoundingBox(value)
                self._distribute_bounding_box()
            else:
                raise ValueError('Invalid bounding box passed')

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

    def _get_inputs_from_kwargs(self, **kwargs) -> Tuple[dict, dict]:
        input_kwargs = {}

        for name in self._inputs:
            if name in kwargs:
                input_kwargs[name] = kwargs[name]
                del kwargs[name]

        return input_kwargs, kwargs

    def _check_inputs(self, *args, **kwargs):
        n_args = len(args) + len(kwargs)
        if self._n_inputs < n_args:
            raise RuntimeError(f'Too many input arguments - expected {self._n_inputs}, got {n_args}')
        elif self._n_inputs > n_args:
            raise RuntimeError(f'Too few input arguments - expected {self._n_inputs}, got {n_args}')

    def _create_inputs(self, *args, **kwargs) -> Dict[str, InputEntry]:
        args = list(args)
        inputs = {}
        for name in self._inputs:
            if name in kwargs:
                inputs[name] = InputEntry(name, kwargs[name])
            else:
                inputs[name] = InputEntry(name, args.pop(0))

        return inputs

    def _get_inputs(self, *args, **kwargs) -> Tuple[Dict[str, InputEntry], dict]:
        input_kwargs, kwargs = self._get_inputs_from_kwargs(**kwargs)
        self._check_inputs(*args, **input_kwargs)

        return self._create_inputs(*args, **input_kwargs), kwargs

    def _fill_optional(self, **kwargs) -> Tuple[dict, dict]:
        optional = {}
        for name, optional_input in self._optional.items():
            if name in kwargs:
                value = kwargs[name]
                del kwargs[name]
            else:
                value = optional_input.default

            optional[name] = value

        return optional, kwargs

    def _fill_model_options(self, optional: dict, **kwargs) -> Tuple[dict, dict, dict]:
        model_options = {}
        options = optional.copy()
        for name, default in modeling_options.items():
            if name in optional:
                value = optional[name]
                del options[name]
            elif name in kwargs:
                value = kwargs[name]
                del kwargs[name]
            else:
                value = default

            model_options[name] = value

        return options, model_options, kwargs

    def _get_options(self, **kwargs) -> Tuple[dict, dict, dict]:
        optional, kwargs = self._fill_optional(**kwargs)
        optional, modeling_options, kwargs = self._fill_model_options(optional, **kwargs)

        if (not self._pass_optional) and (len(kwargs) > 0):
            raise RuntimeError(f'Unknown optional arguments: {kwargs.keys()} have been passed, argument pass through is off.')

        return optional, modeling_options, kwargs

    def evaluation_inputs(self, *args, **kwargs) -> Inputs:
        inputs, kwargs = self._get_inputs(*args, **kwargs)
        optional, modeling_options, kwargs = self._get_options(**kwargs)

        return Inputs(inputs, optional, modeling_options, kwargs)

    def _get_outside(self, inputs: Inputs, name: str, array_shape: bool) \
            -> Tuple[np.ndarray, tuple]:
        if name in inputs.inputs:
            value = inputs.inputs[name]
            if array_shape:
                value = value.input_array
            else:
                value = value.input

            return self._inputs[name].outside(value), value.shape
        else:
            raise RuntimeError(f'Input: {name} not present in inputs')

    def _update_outside_inputs(self, outside_inputs: np.ndarray, all_out: bool,
                               inputs: Inputs, name: str, array_shape: bool) \
            -> Tuple[np.ndarray, bool]:
        outside, shape = self._get_outside(inputs, name, array_shape)

        outside_inputs |= outside

        if not shape and outside_inputs.all():
            all_out = True

        return outside_inputs, all_out

    def _outside_inputs(self, inputs: Inputs, array_shape: bool) -> Tuple[np.ndarray, bool]:
        input_shape = inputs.check_input_shape(self._n_models, array_shape)

        outside_inputs = np.zeros(input_shape, dtype=bool)
        all_out = False
        for name in self._inputs:
            outside_inputs, all_out = self._update_outside_inputs(outside_inputs, all_out,
                                                                  inputs, name, array_shape)

        return outside_inputs, all_out

    def _get_valid_index(self, inputs: Inputs, array_shape: bool) \
            -> Tuple[Tuple[np.ndarray, ...], bool]:
        outside_inputs, all_out = self._outside_inputs(inputs, array_shape)

        # get an array with indices of valid inputs
        valid_index = np.atleast_1d(np.logical_not(outside_inputs)).nonzero()
        if len(valid_index[0]) == 0:
            all_out = True

        return valid_index, all_out

    def bounding_box_inputs(self, inputs: Inputs, array_shape: bool):
        # NOTE: this is to replace prepare_bounding_box_inputs
        valid_index, all_out = self._get_valid_index(inputs, array_shape)

        return inputs.reduce_to_bounding_box(valid_index, all_out, array_shape)

    def prepare_inputs(self, params: list, *args, **kwargs) -> Inputs:
        # Process inputs into wrapper
        inputs = self.evaluation_inputs(*args, **kwargs)

        # equivalent of _validate_input_shapes
        inputs.check_input_shape(self._n_models, False)

        # TODO: equivalent of self._validate_input_shapes

        # equivalent of _prepare_inputs_single_model
        inputs.get_format_info(self._n_models, params,
                               self._standard_broadcasting, self._n_outputs)

        # enforce bounding_box
        if inputs.with_bounding_box:
            inputs = self.bounding_box(inputs, True)

        return inputs
