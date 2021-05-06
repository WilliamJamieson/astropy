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


class IoEntry(object):
    def __init__(self, name: str, value):
        self._name = name
        self.value = value

    def __eq__(self, this):
        if isinstance(this, IoEntry):
            return (self.name == this.name) and (self.value == this.value).all()
        else:
            return False

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, value):
        self._value = np.asanyarray(value, dtype=float)

    @property
    def shape(self) -> tuple:
        return self._value.shape


class InputEntry(IoEntry):
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
    check_input_shape:
        Returns error checked shape for this input.
    broadcast:
        Returns the broadcast shape of this input.
    reduce_to_bounding_box:
        Update this input to just consider the array entries computed
        to be inside the bounding box.

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
    @property
    def input_array(self) -> np.ndarray:
        # NOTE: this method is for replacing _prepare_inputs_single_model

        # Ensure that array scalars are always upgrade to 1-D arrays for the
        # sake of consistency with how parameters work.  They will be cast back
        # to scalars at the end
        if not self.shape:
            return self._value.reshape((1,))
        else:
            return self._value

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
        See OldInputs.check_input_shape for cross checking of all model
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

    def _check_broadcast(self, input_shape: tuple, param, param_shape: tuple) -> tuple:
        try:
            return check_broadcast(input_shape, param_shape)
        except IncompatibleShapeError:
            raise ValueError(f"Model input argument {self.name} of shape {input_shape} cannot be " +
                             f"broadcast with parameter {param.name} of shape {param_shape}.")

    def _get_param_broadcast(self, param, standard_broadcasting: bool) -> tuple:
        if standard_broadcasting:
            return self._check_broadcast(self.shape, param, param.shape)
        else:
            return self.shape

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

        if len(new_broadcast) > len(broadcast):
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
            input's broadcast shape in relation to.
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

    @staticmethod
    def _remove_axes_from_shape(shape: tuple, axis: int) -> tuple:
        """
        Given a shape tuple as the first input, construct a new one by  removing
        that particular axis from the shape and all preceding axes. Negative axis
        numbers are permitted, where the axis is relative to the last axis.
        """
        if len(shape) == 0:
            return shape

        if axis < 0:
            axis = len(shape) + axis
            return shape[:axis] + shape[axis+1:]

        if axis >= len(shape):
            axis = len(shape)-1

        return shape[axis+1:]

    def _get_param_shape(self, input_shape: tuple, param, model_set_axis: int) -> tuple:
        param_shape = self._remove_axes_from_shape(param.shape, model_set_axis)
        self._check_broadcast(input_shape, param, param_shape)

        return param_shape

    def _update_max_param_shape(self, max_param_shape: tuple, input_shape: tuple, param, model_set_axis: int) -> tuple:
        param_shape = self._get_param_shape(input_shape, param, model_set_axis)
        if len(param.shape) - 1 > len(max_param_shape):
            max_param_shape = param_shape

        return max_param_shape

    def _max_param_shape(self, input_shape: tuple, params: list, model_set_axis: int) -> tuple:
        max_param_shape = ()
        for param in params:
            max_param_shape = self._update_max_param_shape(max_param_shape, input_shape, param, model_set_axis)

        return max_param_shape

    def _new_input_no_axis(self, input_ndim: int, max_param_shape: tuple, model_set_axis: int) -> Tuple[np.ndarray, int]:
        if len(max_param_shape) > input_ndim:
            n_new_axes = 1 + len(max_param_shape) - input_ndim
            new_axes = (1,) * n_new_axes
            new_shape = new_axes + self.shape
            pivot = model_set_axis
        else:
            pivot = input_ndim - len(max_param_shape)
            new_shape = (self.shape[:pivot] + (1,) +
                         self.shape[pivot:])

        return self._value.reshape(new_shape), pivot

    def _new_input_axis(self, input_ndim: int, max_param_shape: tuple,
                        model_set_axis: int, model_set_axis_input: int) -> Tuple[np.ndarray, int]:
        if len(max_param_shape) >= input_ndim:
            n_new_axes = len(max_param_shape) - input_ndim
            pivot = model_set_axis
            new_axes = (1,) * n_new_axes
            new_shape = (self.shape[:pivot + 1] + new_axes +
                         self.shape[pivot + 1:])
            new_input = self._value.reshape(new_shape)
        else:
            pivot = self._value.ndim - len(max_param_shape) - 1
            new_input = np.rollaxis(self._value, model_set_axis_input,
                                    pivot + 1)

        return new_input, pivot

    def _new_input(self, input_ndim: int, max_param_shape: tuple,
                   model_set_axis: int, model_set_axis_input: int) -> Tuple[np.ndarray, int]:

        if model_set_axis_input is False:
            return self._new_input_no_axis(input_ndim, max_param_shape,
                                           model_set_axis)
        else:
            return self._new_input_axis(input_ndim, max_param_shape,
                                        model_set_axis, model_set_axis_input)

    def _get_input_shape(self, n_models: int, model_set_axis_input: int) -> tuple:
        if n_models > 1 and model_set_axis_input is not False:
            return (self.shape[:model_set_axis_input] +
                    self.shape[model_set_axis_input + 1:])
        else:
            return self.shape

    def new_input(self, params: list, model_set_axis: int,
                  n_models: int, model_set_axis_input: int) -> Tuple['InputEntry', int]:
        input_shape = self._get_input_shape(n_models, model_set_axis_input)
        max_param_shape = self._max_param_shape(input_shape, params, model_set_axis)

        new_input, pivot = self._new_input(len(input_shape), max_param_shape,
                                           model_set_axis, model_set_axis_input)

        return InputEntry(self.name, new_input), pivot

    def reduce_to_bounding_box(self, valid_index) -> 'InputEntry':
        """
        Reduces this evaluation input to just the indices computed to
        be inside the model's bounding box.

        Parameters
        ----------
        valid_index : Tuple[np.ndarray, ...]
            The indices of this input found to be corrisponding to points
            within the bounding box of the model

        Returns
        -------
        A new input entry which has been reduced to just the valid_index
        locations

        Notes
        -----
        See OldInputs.reduce_to_bounding_box for collective input computation
        of bounding box reduction.
        """

        # Always requires scalars to be arrays to work, so using input_array version
        return InputEntry(self._name, np.array(self.input_array)[valid_index])


class Inputs(object):
    def __init__(self, inputs: Dict[str, InputEntry]):
        self._inputs = inputs

    def __eq__(self, this):
        if isinstance(this, Inputs):
            return (self.inputs == this.inputs)
        else:
            return False

    @property
    def inputs(self) -> Dict[str, InputEntry]:
        return self._inputs

    @property
    def n_inputs(self) -> int:
        return len(self._inputs)

    def copy(self) -> 'Inputs':
        return deepcopy(self)

    def check_input_shape(self, n_models: int, model_set_axis: int, array_shape: bool):
        """
        Checks all the evaluation input shapes, then finds the broadcast
        shape for all the inputs.
            This method replaces _validate_input_shapes

        Parameters
        ----------
        n_models : int
            Number of models (for model set)
        array_shape : bool
            Determines which mode to use.
                True for converted array
                False for scalars remaining un-reshaped

        Returns
        -------
            An error checked and broadcasted shape for all the evaluation
            inputs.
        """
        # NOTE: this method is for replacing _validate_input_shapes
        input_shape = check_broadcast(*[_input.check_input_shape(n_models, model_set_axis, array_shape)
                                        for _input in self._inputs.values()])
        if input_shape is None:
            raise ValueError("All inputs must have identical shapes or must be scalars.")

        return input_shape

    def _get_broadcasts(self, params: list, standard_broadcasting: bool) -> list:
        """
        Helper function for self.broadcast.
            Calls and records the result of broadcast on each evaluation input

        Parameters
        ----------
        params : list
            A list of all the model's parameter to get this evaluation
            input's broadcast shape in relation to.
        standard_broadcasting : bool
            Whether or not standard_broadcasting is used by the model.

        Returns
        -------
            list of input broadcasts in input order
        """

        # TODO: could this be a dictionary?
        return [_input.broadcast(params, standard_broadcasting)
                for _input in self._inputs.values()]

    def _extend_broadcasts(self, n_outputs: int, broadcasts: list):
        """
        Helper function for self.broadcast.
            Extends the broadcast results when there are more outputs
            than inputs.

        Parameters
        ----------
        n_outputs : int
            The number of outputs for the model being evaluated.
        broadcasts : list
            The broadcast shapes of the inputs

        Notes
        -----
        This simply modifies broadcasts in place if necessary.
        """
        # Note that this is copied from _prepare_inputs_single_model. It needs improvement.
        if n_outputs > self.n_inputs:
            extra_outputs = n_outputs - self.n_inputs
            if not broadcasts:
                # If there were no inputs then the broadcasts list is empty
                # just add a None since there is no broadcasting of outputs and
                # inputs necessary (see _prepare_outputs_single_model)

                # TODO: check for broadcast None checks
                broadcasts.append(())
            # TODO: check why its always the first one
            broadcasts.extend([broadcasts[0]] * extra_outputs)

    def broadcast(self, params: list, standard_broadcasting: bool, n_outputs: int) -> list:
        """
        Creates all the broadcast information and stores it in self._broadcast_info
            This method is for replacing _prepare_inputs_single_model

        Parameters
        ----------
        params : list
            A list of all the model's parameter to get this evaluation
            input's broadcast shape in relation to.
        standard_broadcasting : bool
            Whether or not standard_broadcasting is used by the model.
        n_outputs : int
            Number of outputs of the model being evaluated.

        Notes
        -----
        This is the only way to set the broadcast_info for inputs.
        """
        # NOTE: this method is for replacing _prepare_inputs_single_model

        broadcasts = self._get_broadcasts(params, standard_broadcasting)
        self._extend_broadcasts(n_outputs, broadcasts)

        return broadcasts

    def _new_inputs(self, params: list, model_set_axis: int,
                    n_models: int, model_set_axis_input: int) -> list:
        pivots = []
        for name, _input in self._inputs.items():
            new_input, pivot = _input.new_input(params, model_set_axis,
                                                n_models, model_set_axis_input)
            self._inputs[name] = new_input
            pivots.append(pivot)

        return pivots

    def new_inputs(self, params: list, model_set_axis: int, n_models: int,
                   n_outputs: int, model_set_axis_input: int) -> list:
        pivots = self._new_inputs(params, model_set_axis, n_models, model_set_axis_input)
        if self.n_inputs < n_outputs:
            pivots.extend([model_set_axis_input] * (n_outputs - self.n_inputs))

        return pivots

    def _reduce_to_bounding_box(self, valid_index) -> 'Inputs':
        """
        Helper function for self.reduce_to_bounding_box
            Performs the bounding_box reduction.

        Parameters
        ----------
        valid_index : Tuple[np.ndarray, ...]
            The indices of this input found to be corrisponding to points
            within the bounding box of the model
        all_out : bool
            If all of the evaulation inputs are in bounding_box

        Returns
        -------
        A new set of OldInputs which have been adjusted
        """
        inputs = {name: _input.reduce_to_bounding_box(valid_index)
                  for name, _input in self._inputs.items()}

        return Inputs(inputs)

    def reduce_to_bounding_box(self, valid_index, all_out: bool) -> 'Inputs':
        """
        Reduces a set of inputs to be inside the bounding box of a model

        Parameters
        ----------
        valid_index : Tuple[np.ndarray, ...]
            The indices of this input found to be corrisponding to points
            within the bounding box of the model
        all_out : bool
            If all of the evaulation inputs are in bounding_box.
                True returns copy of inputs with valid_index and all_out set.
                False returns a reduced version

        Returns
        -------
        A new set of OldInputs which have been adjusted
        """
        # if not all inputs are not in bounding box, adjust them
        if all_out:
            return self.copy()
        else:
            return self._reduce_to_bounding_box(valid_index)


class Optional(object):
    def __init__(self, optional: dict,
                 model_options: dict=modeling_options,
                 pass_through: dict={},
                 model_set_axis: int=0):
        self._optional = optional
        self.model_options = model_options
        self._pass_through = pass_through
        self._model_set_axis = model_set_axis

    def __eq__(self, this):
        if isinstance(this, Optional):
            return (self.optional == this.optional) and \
                (self.model_options == this.model_options) and \
                (self.pass_through == this.pass_through) and \
                (self.model_set_axis == this.model_set_axis)
        else:
            return False

    @property
    def optional(self) -> dict:
        return self._optional

    @property
    def model_options(self) -> dict:
        return self._model_options

    @model_options.setter
    def model_options(self, value: dict):
        """
        Includes checking to make sure all options are set
            (should happen by default)
        """
        for name in modeling_options:
            if name not in value:
                raise ValueError(f"Modeling option {name} must be set!")
        else:
            self._model_options = value

    def _get_model_option(self, name: str):
        """
        Get a modeling_option by name.
        """
        if name in self._model_options:
            return self._model_options[name]
        else:
            raise RuntimeError(f'Option "{name}" must be set!')

    @property
    def model_set_axis(self):
        value = self._get_model_option('model_set_axis')
        if value is None:
            return self._model_set_axis
        else:
            return value

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


class InputData(object):
    def __init__(self, format_info: list=None,
                 valid_index: np.ndarray=None,
                 all_out: bool=None):
        self._format_info = format_info
        self._valid_index = valid_index
        self._all_out = all_out

    def __eq__(self, this):
        if isinstance(this, InputData):
            return (self.format_info == this.format_info) and \
                (self.valid_index == this.valid_index).all() and \
                (self.all_out == this.all_out)
        else:
            return False

    @property
    def format_info(self) -> list:
        if self._format_info is None:
            return []
        else:
            return self._format_info

    @format_info.setter
    def format_info(self, value):
        self._format_info = value

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

    def copy(self) -> 'InputData':
        return deepcopy(self)

    def reduce_to_bounding_box(self, valid_index, all_out: bool) -> 'InputData':
        new = self.copy()
        new.valid_index = valid_index
        new.all_out = all_out

        return new


class EvaluationInputs(object):
    def __init__(self, inputs: Inputs, optional: Optional, data: InputData):
        self._inputs = inputs
        self._optional = optional
        self._data = data

    @property
    def inputs(self) -> Inputs:
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        self._inputs = value

    @property
    def optional(self) -> Optional:
        return self._optional

    @optional.setter
    def optional(self, value):
        self._optional = value

    @property
    def data(self) -> InputData:
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def format_info(self) -> list:
        return self._data.format_info

    @format_info.setter
    def format_info(self, value):
        self._data.format_info = value

    def check_input_shape(self, n_models: int):
        self._inputs.check_input_shape(n_models, self._optional.model_set_axis, False)

    @classmethod
    def evaluation_inputs(cls, inputs: Inputs, optional: Optional) -> 'EvaluationInputs':
        return cls(inputs, optional, InputData())

    def set_format_info(self, params: list, standard_broadcasting: bool,
                        n_models: int, model_set_axis: int, n_outputs: int):
        if n_models == 1:
            format_info = self._inputs.broadcast(params, standard_broadcasting, n_outputs)
        else:
            format_info = self._inputs.new_inputs(params, model_set_axis, n_models, n_outputs,
                                                  self._optional.model_set_axis)
        self.format_info = format_info

    def reduce_to_bounding_box(self, valid_index, all_out: bool):
        self.inputs = self._inputs.reduce_to_bounding_box(valid_index, all_out)
        self.data = self._data.reduce_to_bounding_box(valid_index, all_out)


class OutputEntry(IoEntry):
    def __init__(self, name: str, value, index: int):
        super().__init__(name, value)
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def scalar(self):
        """
        Return the entry as a scalar if possible
        """
        try:
            return self._value.item()
        except ValueError:
            return self._value

    def _new_output(self, broadcast_shape: tuple):
        if not broadcast_shape:
            return self.scalar
        else:
            try:
                return self.value.reshape(broadcast_shape)
            except ValueError:
                return self.scalar

    def _check_broadcast(self, format_info):
        try:
            return check_broadcast(*format_info)
        except (IndexError, TypeError):
            return format_info[self._index]

    def prepare_output_single_model(self, format_info) -> "OutputEntry":
        broadcast_shape = self._check_broadcast(format_info)

        if broadcast_shape is None:
            return self
        else:
            return OutputEntry(self.name, self._new_output(broadcast_shape), self._index)

    def prepare_output_model_set(self, pivots: list, model_set_axis: int) -> "OutputEntry":
        pivot = pivots[self._index]
        if pivot < self.value.ndim and pivot != model_set_axis:
            return OutputEntry(self.name,
                               np.rollaxis(self.value, pivot, model_set_axis),
                               self._index)
        else:
            return self

    def prepare_input(self) -> InputEntry:
        return InputEntry(self.name, self.value)


class Outputs(object):
    def __init__(self, outputs: Dict[str, OutputEntry]):
        self._outputs = outputs

    @property
    def n_outputs(self) -> int:
        return len(self._outputs)

    @property
    def outputs(self) -> Dict[str, OutputEntry]:
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value

    def prepare_outputs_single_model(self, format_info) -> "Outputs":
        outputs = {}
        for name, output in self._outputs.items():
            outputs[name] = output.prepare_output_single_model(format_info)

        return Outputs(outputs)

    def prepare_outputs_model_set(self, pivots: list, model_set_axis: int) -> "Outputs":
        outputs = {}
        for name, output in self._outputs.items():
            outputs[name] = output.prepare_output_model_set(pivots, model_set_axis)

        return Outputs(outputs)


class IoMetaDataEntry(object):
    """
    Base class of meta data entries

    Parameters
    ----------
    name : str
        Name of the piece of data
    """
    _data_entry = None

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

    def _create_io_entry(self, value):
        if self._data_entry is None or isinstance(value, self._data_entry):
            return value
        else:
            return self._data_entry(self.name, value)

    def get_from_kwargs(self, data_kwargs: dict, **kwargs) -> dict:
        if self.name in kwargs:
            data_kwargs[self.name] = self._create_io_entry(kwargs[self.name])
            del kwargs[self.name]

        return kwargs


class IoMetaData(object):
    _data_entry = None
    def __init__(self, data: dict = None, **kwargs):
        self._data = {}

        if data is not None:
            self.data = data

    def _fill_defaults(self, n_data: int):
        """
        Helper function for create_defaults
            Generates the required number of inputs under the standard
            names.

        Parameters
        ----------
        n_data : int
            The number of data entries
        """

        raise NotImplementedError('This has not been implemented')

    @classmethod
    def create_defaults(cls, n_data: int, **kwargs):
        """
        Create InputMetaData object with default values.

        Parameters
        ----------
        n_data : int
            The number of required evaluation inputs for the model.
        """
        new = cls(n_data=n_data, **kwargs)
        new._fill_defaults(n_data)

        return new

    def validate(self, *args, **kwargs):
        """
        Validates that the ALL the metadata is self consistent.
        """

        raise NotImplementedError('This has not been implemented')

    def _process_meta_data(self, value) -> dict:
        """
        Helper method for self._set_inputs_atr
            Creates the correct input entry object

        Parameters
        ----------
        value : any
            The data to be stored
        """
        if self._data_entry is None:
            return value
        else:
            data = {}
            if value is not None:
                if isinstance(value, list) or isinstance(value, tuple):
                    for index, input_value in enumerate(value):
                        entry = self._data_entry.create_entry(input_value, pos=index)
                        data[entry.name] = entry
                elif isinstance(value, dict):
                    for name, input_value in value.items():
                        entry = self._data_entry.create_entry(input_value, name=name)
                        data[entry.name] = entry
                else:
                    raise ValueError(f'{value} is not a valid way to set inputs')

            return data

    def reset_data(self):
        self._data = {}

    @property
    def data(self) -> dict:
        return self._data

    @data.setter
    def data(self, value):
        self._data = self._process_meta_data(value)

    def get_from_kwargs(self, **kwargs) -> Tuple[dict, dict]:
        """
        Extracts the corresponding inputs which have been passed as kwargs

        Parameters
        ----------
        kwargs :
            The kwargs passed to the model for evaluation

        Returns
        tuple(
            Dictionary of the required inputs passed as kwargs,
            kwargs with the required inputs removed
        )
        """
        data_kwargs = {}

        for entry in self._data.values():
            kwargs = entry.get_from_kwargs(data_kwargs, **kwargs)

        return data_kwargs, kwargs


class InputMetaDataEntry(IoMetaDataEntry):
    """
    Contains meta data on a single required input for a model.

    Parameters
    ----------
    name : str
        Name of the input
    pos : int
        Position of the input in ordered set of inputs
    bounding_box : _BoundingBox (1-D)
        Bounding box for the single input in question

    Constructors
    ------------
    create_entry:
        Creates the entry from standardized inputs. This is for backwards
        compatibility with previous model evaluation IO system.

    Methods
    -------
    outside:
        Determines what parts of an array are outside the bounding box
    """
    _data_entry = InputEntry

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
        if value is None or value is NotImplemented:
            self._bounding_box = value
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

    def create_input(self, inputs: Dict[str, InputEntry], *args, **kwargs) -> tuple:
        args = list(args)
        if self.name in kwargs:
            value = self._create_io_entry(kwargs[self.name])
        else:
            value = self._create_io_entry(args.pop(0))
        inputs[self.name] = value

        return tuple(args)

    def _outside(self, inputs: EvaluationInputs) -> Tuple[np.ndarray, tuple]:
        if self.name in inputs.inputs.inputs:
            value = inputs.inputs.inputs[self.name].input_array

            return self.bounding_box.outside(value), value.shape
        else:
            raise RuntimeError(f'Input: {self.name} not present in inputs')

    def update_outside(self, outside: np.ndarray, all_out: bool,
                       inputs: EvaluationInputs) -> Tuple[np.ndarray, bool]:
        update = outside.copy()

        current, shape = self._outside(inputs)
        update |= current

        if not shape and update.all():
            all_out = True

        return update, all_out


class InputMetaData(IoMetaData):
    _data_entry = InputMetaDataEntry

    def __init__(self, n_data: int,
                 inputs: Dict[str, InputMetaDataEntry] = None,
                 bounding_box: _BoundingBox=None):
        super().__init__(inputs)
        self._n_inputs = n_data
        self.bounding_box = bounding_box

    def _fill_defaults(self, n_data: int):
        """
        Helper function for create_defaults
            Generates the required number of inputs under the standard
            names.
        """
        if n_data == 1:
            self.inputs = ['x']
        elif n_data == 2:
            self.inputs = ['x', 'y']
        else:
            self.inputs = [f'x{idx}' for idx in range(n_data)]

    @property
    def n_inputs(self) -> int:
        return self._n_inputs

    @n_inputs.setter
    def n_inputs(self, value):
        self.reset_data()
        self._n_inputs = value
        self._fill_defaults(value)

    def validate(self):
        """
        Validates that the ALL the input metadata is self consistent.
        """
        if (len(self._data) > 0) and (self._n_inputs != len(self._data)):
            raise ValueError('n_inputs must match the number of entries in inputs.')

        for pos, (name, _input) in enumerate(self._data.items()):
            if name != _input.name:
                raise ValueError(f"Input: {_input.name}'s key information is incorrect'")
            if pos != _input.pos:
                raise ValueError(f"Input: {_input.name}'s position is " +
                                 "information is incorrect.'")

    @property
    def inputs(self) -> Dict[str, InputMetaDataEntry]:
        return self.data

    @inputs.setter
    def inputs(self, value):
        self.data = value
        if self._n_inputs == 0:
            self._n_inputs = len(self._data)
        self.validate()

    @property
    def bounding_box(self) -> _BoundingBox:
        if self._bounding_box is None:
            raise NotImplementedError('No bounding_box has been assigned')
        else:
            return self._bounding_box

    def _reverse_bounding_box(self):
        """
        Helper function for self._distribute_bounding_box.
            This reverses the order of the bounding_box entries so that
            input meta data can get the correct one.

        Notes
        -----
        Other design decisions for `~astropy.Modeling` have made it so
        that multidimensional bounding boxes store the individual dimension
        boxes in reverse order to the normal order of passing a coordinate.
        This method reverses this order to work around this design choice.
        """

        if self.n_inputs > 1:
            return self.bounding_box[::-1]
        else:
            return [self.bounding_box]

    def _distribute_bounding_box(self):
        """
        Helper function for bounding_box setter.
            This function passes the correct parts of the bounding box
            to the correct required inputs.
        """
        bbox = self._reverse_bounding_box()

        for _input in self._data.values():
            _input.bounding_box = bbox[_input.pos]

    @bounding_box.setter
    def bounding_box(self, value):
        if value is None or value is NotImplemented:
            self._bounding_box = value
        else:
            self._bounding_box = _BoundingBox(value)
            self._distribute_bounding_box()

    def _check_inputs(self, *args, **kwargs):
        """
        Helper function for self._get_inputs
            Performs basic consistency check on model evaluation arguments

        Parameters
        ----------
        args :
            The positional arguments passed to model for evaluation
        kwargs :
            The required arguments passed to the model as kwargs for
            evaluation
        """
        n_args = len(args) + len(kwargs)
        if self._n_inputs < n_args:
            raise ValueError(f'Too many input arguments - expected {self._n_inputs}, got {n_args}')
        elif self._n_inputs > n_args:
            raise ValueError(f'Too few input arguments - expected {self._n_inputs}, got {n_args}')

    def _create_inputs(self, *args, **kwargs) -> Inputs:
        """
        Helper function for self._get_inputs
            Turns user evaluation inputs into InputEntry objects

        Parameters
        ----------
        args :
            The positional arguments passed to model for evaluation
        kwargs :
            The required arguments passed to the model as kwargs for
            evaluation

        Returns
        -------
        Dictionary of input arguments
        """
        inputs = {}
        for _input in self._data.values():
            args = _input.create_input(inputs, *args, **kwargs)

        return Inputs(inputs)

    def get_inputs(self, *args, **kwargs) -> Tuple[Inputs, dict]:
        """
        Helper function for self.evaluation_inputs
            Performs all the steps necessary to check and create all the InputEntry
            objects

        Parameters
        ----------
        args :
            The positional arguments passed to model for evaluation
        kwargs :
            The kwargs passed to the model for evaluation

        Returns
        -------
        tuple(
            Dictionary of input arguments,
            kwargs with no required inputs included
        )
        """
        input_kwargs, kwargs = self.get_from_kwargs(**kwargs)
        self._check_inputs(*args, **input_kwargs)

        return self._create_inputs(*args, **input_kwargs), kwargs

    def _outside(self, n_models: int, model_set_axis: int, inputs: EvaluationInputs) -> Tuple[np.ndarray, bool]:
        """
        Helper function for self._get_valid_index
            Deterimines indices of input arrays that are in/out of bounding
            box.

        Parameters
        ----------
        n_models: int
            The number of models
        inputs : OldInputs
            The processed evaluation inputs object

        Returns
        -------
        tuple(
            boolean array specifying if indices are inside/outside bounding box,
            if all of inputs are inside/outside bounding_box.
        )
        """
        # NOTE: array_shape will always be True for bounding box
        input_shape = inputs.inputs.check_input_shape(n_models, model_set_axis, True)
        outside = np.zeros(input_shape, dtype=bool)

        all_out = False
        for _input in self._data.values():
            outside, all_out = _input.update_outside(outside, all_out, inputs)

        return outside, all_out

    def _get_valid_index(self, n_models: int, model_set_axis: int, inputs: EvaluationInputs) \
            -> Tuple[Tuple[np.ndarray, ...], bool]:
        """
        Helper function for self.bounding_box_inputs
            Generates the list of valid indices for the input arrays

        Parameters
        ----------
        n_models: int
            The number of models
        inputs : OldInputs
            The processed evaluation inputs object

        Returns
        -------
        tuple(
            Array specifying which indices are inside bounding box
            if all of inputs are inside/outside bounding_box.
        )
        """
        outside_inputs, all_out = self._outside(n_models, model_set_axis, inputs)

        # get an array with indices of valid inputs
        valid_index = np.atleast_1d(np.logical_not(outside_inputs)).nonzero()
        if len(valid_index[0]) == 0:
            all_out = True

        return valid_index, all_out

    def enforce_bounding_box(self, n_models: int, model_set_axis: int, inputs: EvaluationInputs):
        """
        Creates an OldInputs object whose values are all inside the bounding box

        Parameters
        ----------
        n_models: int
            The number of models
        inputs : OldInputs
            The processed evaluation inputs object

        Returns
        -------
        new inputs object with only inputs inside bounding box
        """
        # NOTE: this is to replace prepare_bounding_box_inputs
        if inputs.optional.with_bounding_box:
            valid_index, all_out = self._get_valid_index(n_models, model_set_axis, inputs)
            inputs.reduce_to_bounding_box(valid_index, all_out)


class OptionalMetaDataEntry(IoMetaDataEntry):
    """
    Contains meta data on a single optional input for a model.

    Parameters
    ----------
    name : str
        Name of data
    default : any
        default value for the input

    Constructors
    ------------
    create_entry:
        Creates the entry from standardized inputs. This is for backwards
        compatibility with previous model evaluation IO system.
    """
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

    def get_from_kwargs(self, optional: dict, **kwargs) -> dict:
        if self.name in kwargs:
            value = kwargs[self.name]
            del kwargs[self.name]
        else:
            value = self.default
        optional[self.name] = value

        return kwargs


class OptionalMetaData(IoMetaData):
    _data_entry = OptionalMetaDataEntry

    def __init__(self, optional: Dict[str, OptionalMetaDataEntry]=None,
                 pass_optional: bool=False,
                 model_set_axis: int=0,
                 **kwargs):
        super().__init__(optional)
        self._pass_optional = pass_optional
        self._model_set_axis = model_set_axis

    @property
    def pass_optional(self) -> bool:
        return self._pass_optional

    @pass_optional.setter
    def pass_optional(self, value):
        self._pass_optional = value

    @property
    def model_set_axis(self) -> int:
        return self._model_set_axis

    @model_set_axis.setter
    def model_set_axis(self, value):
        self._model_set_axis = value

    def _fill_defaults(self, n_data: int):
        """
        Helper function for create_defaults
            Generates the required number of inputs under the standard
            names.

        Parameters
        ----------
        n_data : int
            The number of data entries
        """
        self.optional = [f'optional_{idx}' for idx in range(n_data)]

    def validate(self, inputs: dict):
        """
        Validates that the ALL the input metadata is self consistent.
        """
        for input_name, input_data in self._data.items():
            if input_name != input_data.name:
                raise ValueError(f"Optional: {input_data.name}'s key information is incorrect'")
            if (inputs is not None) and (input_name in inputs):
                raise ValueError(f"Optional: {input_name} is both optional and non-optional")

    @property
    def optional(self) -> Dict[str, OptionalMetaDataEntry]:
        return self.data

    @optional.setter
    def optional(self, value):
        self.data = value

    def _get_model_options(self, optional: dict, **kwargs) -> Tuple[dict, dict, dict]:
        """
        Helper function for self._get_options
            Fills in all the optional inputs

        Parameters
        ----------
        optional : dict
            The optional inputs
        kwargs :
            User's kwargs with the required inputs and optional removed

        Returns
        -------
        tuple(
            Dictionary of optional inputs (no modeling options),
            modeling options,
            kwargs with required, optional inputs, and modeling options removed.
        )

        Notes
        -----
        The metadata object allows one to override the default modeling_options
        with model specific defaults. These options are transfered here from
        optional input storage to model_options.
        """
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

    def get_optional(self, **kwargs) -> Optional:
        """
        Helper function for self.evaluation_inputs
            Read all the kwargs

        Parameters
        ----------
        kwargs :
            User's kwargs with the required inputsremoved

        Returns
        -------
        tuple(
            Dictionary of optional inputs (no modeling options),
            modeling options,
            all remaining kwargs
        )

        Notes
        -----
        If pass_through is disabled, an error will be raised if there
        are any unaccounted for kwargs.
        """
        optional, kwargs = self.get_from_kwargs(**kwargs)
        optional, model_options, pass_through = self._get_model_options(optional, **kwargs)

        if (not self.pass_optional) and (len(pass_through) > 0):
            raise RuntimeError(f'Unknown optional arguments: {kwargs.keys()} ' +
                               'have been passed, argument pass through is off.')

        return Optional(optional, model_options, pass_through, self._model_set_axis)


class OutputMetaDataEntry(IoMetaDataEntry):
    _data_entry = OutputEntry

    def __init__(self, name: str=None, pos: int=None):
        super().__init__(name)
        self._pos = pos

    @property
    def pos(self) -> int:
        if self._pos is None:
            raise RuntimeError(f'Output {self.name} has no position specified')
        else:
            return self._pos

    @pos.setter
    def pos(self, value: int):
        if self._pos is None:
            self._pos = value
        else:
            raise ValueError(f'Output {self.name} has position specified already')

    @classmethod
    def create_entry(cls, input_value, *, name=None, pos=None):
        if isinstance(input_value, OutputMetaDataEntry):
            return input_value
        elif isinstance(input_value, tuple):
            return cls(name, *input_value)
        elif isinstance(input_value, str):
            return cls(input_value, pos)
        else:
            raise ValueError(f'{input_value} is not a valid way to set an output')


class OutputMetaData(IoMetaData):
    _data_entry = OutputMetaDataEntry

    def __init__(self, n_data: int,
                 outputs: Dict[str, OutputMetaDataEntry] = None):
        super().__init__(outputs)
        self._n_outputs = n_data

    def _fill_defaults(self, n_data: int):
        """
        Helper function for create_defaults
            Generates the required number of outputs under the standard
            names.
        """

        self.outputs = [f'y{idx}' for idx in range(n_data)]

    @property
    def n_outputs(self) -> int:
        return self._n_outputs

    @n_outputs.setter
    def n_outputs(self, value):
        self.reset_data()
        self._n_outputs = value
        self._fill_defaults(value)

    def validate(self):
        """
        Validates that the ALL the input metadata is self consistent.
        """
        if (len(self._data) > 0) and (self._n_outputs != len(self._data)):
            raise ValueError('n_outputs must match the number of entries in outputs.')

        for pos, (name, output) in enumerate(self._data.items()):
            if name != output.name:
                raise ValueError(f"Output: {output.name}'s key information is incorrect'")
            if pos != output.pos:
                raise ValueError(f"Output: {output.name}'s position is " +
                                 "information is incorrect.'")

    @property
    def outputs(self) -> Dict[str, OutputMetaDataEntry]:
        return self.data

    @outputs.setter
    def outputs(self, value):
        self.data = value
        self.validate()


class MetaData(object):
    def __init__(self, inputs: InputMetaData,
                 optional: OptionalMetaData,
                 outputs: OutputMetaData,
                 n_models: int=1,
                 standard_broadcasting: bool=True):
        self._inputs = inputs
        self._optional = optional
        self._outputs = outputs
        self._n_models = n_models
        self._standard_broadcasting = standard_broadcasting

    @classmethod
    def create_defaults(cls, n_inputs: int,
                        n_outputs: int=1,
                        n_models: int=1,
                        standard_broadcasting: bool=True,
                        pass_optional: bool=False,
                        model_set_axis: int=0,
                        optional: OptionalMetaData=None):
        inputs = InputMetaData.create_defaults(n_inputs)

        if optional is None:
            optional = OptionalMetaData.create_defaults(0, pass_optional=pass_optional,
                                                        model_set_axis=model_set_axis)
        else:
            optional.pass_optional = pass_optional
            optional.model_set_axis = model_set_axis

        outputs = OutputMetaData.create_defaults(n_outputs)

        return cls(inputs, optional, outputs, n_models, standard_broadcasting)

    @property
    def n_inputs(self) -> int:
        return self._inputs.n_inputs

    @n_inputs.setter
    def n_inputs(self, value):
        self._inputs.n_inputs = value

    @property
    def n_outputs(self) -> int:
        return self._outputs.n_outputs

    @n_outputs.setter
    def n_outputs(self, value):
        self._outputs.n_outputs = value

    @property
    def n_models(self) -> int:
        return self._n_models

    @n_models.setter
    def n_models(self, value):
        self._n_models = value

    @property
    def model_set_axis(self) -> int:
        return self._optional.model_set_axis

    @model_set_axis.setter
    def model_set_axis(self, value):
        self._optional.model_set_axis = value

    @property
    def standard_broadcasting(self) -> bool:
        return self._standard_broadcasting

    @standard_broadcasting.setter
    def standard_broadcasting(self, value):
        self._standard_broadcasting = value

    @property
    def bounding_box(self) -> _BoundingBox:
        return self._inputs.bounding_box

    @bounding_box.setter
    def bounding_box(self, value):
        self._inputs.bounding_box = value

    @property
    def inputs(self) -> Tuple[str]:
        return tuple(self._inputs.inputs.keys())

    @inputs.setter
    def inputs(self, value):
        self._inputs.inputs = value
        self._optional.validate(self._inputs.inputs)

    @property
    def optional(self) -> Tuple[str]:
        return tuple(self._optional.optional.keys())

    @optional.setter
    def optional(self, value):
        self._optional.optional = value
        self._optional.validate(self._inputs.inputs)

    @property
    def outputs(self) -> Tuple[str]:
        return tuple(self._outputs.outputs.keys())

    @outputs.setter
    def outputs(self, value):
        self._outputs.outputs = value

    def evaluation_inputs(self, *args, **kwargs) -> EvaluationInputs:
        """
        Turn the user's evaluation inputs to a model into an OldInputs object

        Parameters
        ----------
        args :
            The positional arguments passed to model for evaluation
        kwargs :
            The kwargs passed to the model for evaluation

        Returns
        -------
        Validated OldInputs object
        """
        inputs, kwargs = self._inputs.get_inputs(*args, **kwargs)
        optional = self._optional.get_optional(**kwargs)

        return EvaluationInputs.evaluation_inputs(inputs, optional)

    def set_format_info(self, params: list, inputs: EvaluationInputs):
        inputs.set_format_info(params, self.standard_broadcasting,
                               self.n_models, self.model_set_axis, self.n_outputs)

    def enforce_bounding_box(self, inputs: EvaluationInputs):
        self._inputs.enforce_bounding_box(self.n_models, self.model_set_axis, inputs)

    def process_inputs(self, params: list, inputs: EvaluationInputs):
        inputs.check_input_shape(self.n_models)
        # TODO: enforce units.

        # Generate the format info for the inputs
        self.set_format_info(params, inputs)

        # Enforce the bounding box
        self.enforce_bounding_box(inputs)

    def prepare_inputs(self, params: list, *args, **kwargs) -> EvaluationInputs:
        # Process user inputs into the wrapper
        inputs = self.evaluation_inputs(*args, **kwargs)

        # Process the input wrapper
        self.process_inputs(params, inputs)

        return inputs
