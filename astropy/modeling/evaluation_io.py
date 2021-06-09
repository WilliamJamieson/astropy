# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Evaluation IO for Models"""

import numpy as np
from typing import Dict, List, Tuple, Union, TYPE_CHECKING
from copy import deepcopy
from itertools import chain

from astropy.utils import isiterable
from astropy.utils.shapes import check_broadcast, IncompatibleShapeError
from astropy.modeling.utils import _BoundingBox, _combine_equivalency_dict
from astropy.units import UnitBase, dimensionless_unscaled, Quantity, UnitsError

if TYPE_CHECKING:
    from astropy.modeling.core import Model

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

    @property
    def is_quantity(self) -> bool:
        return isinstance(self._value, Quantity)

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
        Helper function for self._get_param_shape
            Given a shape tuple as the first input, construct a new one by  removing
            that particular axis from the shape and all preceding axes. Negative axis
            numbers are permitted, where the axis is relative to the last axis.

        Parameters
        ----------
        shape : tuple
            The shape of the parameter
        axis : int
            The axis to reshape parameter around
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
        """
        Helper function for self._update_max_param_shape
            Gets the shape of the input relative to param

        Parameters
        ----------
        input_shape : int
            The shape of the input within the model set
        param :
            A model parameter
        model_set_axis : int
            The default model_set_axis

        Returns
        -------
        The shape of the input relative to param
        """
        param_shape = self._remove_axes_from_shape(param.shape, model_set_axis)
        self._check_broadcast(input_shape, param, param_shape)

        return param_shape

    def _update_max_param_shape(self, max_param_shape: tuple, input_shape: tuple, param, model_set_axis: int) -> tuple:
        """
        Helper function for self._max_param_shape
            Update's the max_param_shape with respect to the current parameter

        Parameters
        ----------
        max_param_shape : tuple
            The current max_param_shape
        input_shape : int
            The shape of the input within the model set
        param :
            A model parameter
        model_set_axis : int
            The default model_set_axis

        Returns
        -------
        The maximum parameter shape for this input relative to param
        """
        param_shape = self._get_param_shape(input_shape, param, model_set_axis)
        if len(param.shape) - 1 > len(max_param_shape):
            max_param_shape = param_shape

        return max_param_shape

    def _max_param_shape(self, input_shape: tuple, params: list, model_set_axis: int) -> tuple:
        """
        Helper function for self.new_input
            Finds the maximum shape of this input vs the parameters

        Parameters
        ----------
        input_shape : int
            The shape of the input within the model set
        params : list
            A list of all the model's parameter to get this evaluation
            input's broadcast shape in relation to.
        model_set_axis : int
            The default model_set_axis

        Returns
        -------
        The maximum parameter shape for this input.
        """
        max_param_shape = ()
        for param in params:
            max_param_shape = self._update_max_param_shape(max_param_shape, input_shape, param, model_set_axis)

        return max_param_shape

    def _new_input_no_axis(self, input_ndim: int, max_param_shape: tuple, model_set_axis: int) -> Tuple[np.ndarray, int]:
        """
        Helper function for self._new_input_value
            Adjust the input's internal array to be compatible with model set,
            when user no user supplied model_set_axis.

        Parameters
        ----------
        input_ndim : int
            Number of input dimensions
        max_param_shape : tuple
            The compatible broadcast shape of the model set
        model_set_axis : int
            The default model_set_axis

        Returns
        -------
        tuple(
            new input array value,
            pivot value
        )
        """
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
        """
        Helper function for self._new_input_value
            Adjust the input's internal array to be compatible with model set,
            when user supplied model_set_axis is given

        Parameters
        ----------
        input_ndim : int
            Number of input dimensions
        max_param_shape : tuple
            The compatible broadcast shape of the model set
        model_set_axis : int
            The default model_set_axis
        model_set_axis_input : int
            The model_set_axis passed in at evaluation

        Returns
        -------
        tuple(
            new input array value,
            pivot value
        )
        """
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

    def _new_input_value(self, input_ndim: int, max_param_shape: tuple,
                   model_set_axis: int, model_set_axis_input: int) -> Tuple[np.ndarray, int]:
        """
        Helper function for self.new_input
            Adjust the input's internal array to be compatible with model set

        Parameters
        ----------
        input_ndim : int
            Number of input dimensions
        max_param_shape : tuple
            The compatible broadcast shape of the model set
        model_set_axis : int
            The default model_set_axis
        model_set_axis_input : int
            The model_set_axis passed in at evaluation

        Returns
        -------
        tuple(
            new input array value,
            pivot value
        )
        """
        if model_set_axis_input is False:
            return self._new_input_no_axis(input_ndim, max_param_shape,
                                           model_set_axis)
        else:
            return self._new_input_axis(input_ndim, max_param_shape,
                                        model_set_axis, model_set_axis_input)

    def _get_input_shape(self, n_models: int, model_set_axis_input: int) -> tuple:
        """
        Helper function for self.new_input
            Gets the shape of this input relative to the model set

        Parameters
        ----------
        n_models : int
            The number of models in the model set
        model_set_axis_input : int
            The model_set_axis passed in at evaluation

        Returns
        -------
        The input's shape in the model set.
        """
        if n_models > 1 and model_set_axis_input is not False:
            return (self.shape[:model_set_axis_input] +
                    self.shape[model_set_axis_input + 1:])
        else:
            return self.shape

    def new_input(self, params: list, model_set_axis: int,
                  n_models: int, model_set_axis_input: int) -> Tuple['InputEntry', int]:
        """
        Creates a new version of this input and a pivot when evaluating a model set.

        Parameters
        ----------
        params : list
            A list of all the model's parameter to get this evaluation
            input's broadcast shape in relation to.
        model_set_axis : int
            The default model_set_axis
        n_models : int
            The number of models in the model set
        model_set_axis_input : int
            The model_set_axis passed in at evaluation

        Returns
        -------
        tuple(
            New InputEntry adjusted to the model set,
            The pivot of that entry.
        )
        """
        input_shape = self._get_input_shape(n_models, model_set_axis_input)
        max_param_shape = self._max_param_shape(input_shape, params, model_set_axis)

        input_value, pivot = self._new_input_value(len(input_shape), max_param_shape,
                                                   model_set_axis, model_set_axis_input)

        return InputEntry(self.name, input_value), pivot

    def _reduce_value_to_bounding_box(self, valid_index):
        unit = getattr(self._value, 'unit', None)
        value = np.array(self.input_array)[valid_index]
        if unit is not None:
            value = Quantity(value, unit, copy=False)

        return value

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
        return InputEntry(self._name, self._reduce_value_to_bounding_box(valid_index))

    def _convert_unit_value(self, value: Quantity, unit: UnitBase,
                            equivalencies: dict, strict: bool):
        # If equivalencies have been specified, we need to
        # convert the input to the input units - this is
        # because some equivalencies are non-linear, and
        # we need to be sure that we evaluate the model in
        # its own frame of reference. If input_units_strict
        # is set, we also need to convert to the input units.
        if len(equivalencies) > 0 or strict:
            self._value = value.to(unit, equivalencies=equivalencies[self._name])

    def _not_equivalent_error(self, name: str, value: Quantity, unit: UnitBase):
        # We consider the following two cases separately so as
        # to be able to raise more appropriate/nicer exceptions
        if unit is dimensionless_unscaled:
            raise UnitsError(f"{name}: Units of input '{self._name}', {value.unit} "
                             f"({value.unit.physical_type}), "
                             "could not be converted to required dimensionless input")
        else:
            raise UnitsError(f"{name}: Units of input '{self._name}', {value.unit} "
                             f"({value.unit.physical_type}), could not be converted "
                             f"to required input units of {unit} ({unit.physical_type})")

    def _convert_unit(self, name: str, value: Quantity, unit: UnitBase,
                      model: 'Model', equivalencies: dict):
        if value.unit.is_equivalent(unit, equivalencies=equivalencies[self._name]):
            self._convert_unit_value(value, unit, equivalencies,
                                     model.input_units_strict[self._name])
        else:
            self._not_equivalent_error(name, value, unit)

    def _validate_dimensionless(self, name: str, unit: UnitBase, model):
        # If we allow dimensionless input, we add the units to the
        # input values without conversion, otherwise we raise an
        # exception.
        if (not model.input_units_allow_dimensionless[self._name]) and \
                (unit is not dimensionless_unscaled) and (unit is not None):
            if np.any(self.value != 0):
                raise UnitsError(f"{name}: Units of input '{self._name}', (dimensionless), "
                                 "could not be converted to required input units of "
                                 f"{unit} ({unit.physical_type})")

    def _get_unit(self, model: 'Model') -> Union[UnitBase, None]:
        return model.input_units.get(self._name, None)

    def convert_unit(self, model: 'Model', equivalencies: dict):
        unit = self._get_unit(model)

        if (unit is not None):
            name = model.name or model.__class__.__name__
            if isinstance(self._value, Quantity):
                self._convert_unit(name, self._value, unit, model, equivalencies)
            else:
                self._validate_dimensionless(name, unit, model)


class Inputs(object):
    def __init__(self, inputs: Dict[str, InputEntry]):
        self._inputs = inputs

    def __eq__(self, this):
        if isinstance(this, Inputs):
            return (self.inputs == this.inputs)
        else:
            return False

    @property
    def are_quantity(self) -> bool:
        return any([entry.is_quantity for entry in self._inputs.values()])

    @property
    def names(self) -> Tuple[str]:
        return tuple(self._inputs.keys())

    @property
    def inputs(self) -> Dict[str, InputEntry]:
        return self._inputs

    @property
    def values(self) -> List[np.ndarray]:
        return [entry.input_array for entry in self._inputs.values()]

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
        Creates all the broadcast information.
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
        """
        # NOTE: this method is for replacing _prepare_inputs_single_model

        broadcasts = self._get_broadcasts(params, standard_broadcasting)
        self._extend_broadcasts(n_outputs, broadcasts)

        return broadcasts

    def _new_inputs(self, params: list, model_set_axis: int,
                    n_models: int, model_set_axis_input: int) -> list:
        """
        Helper function for self.pivots.
            Creates the pivot information for just the inputs, while
            adjusting the inputs for the model set.

        Parameters
        ----------
        params : list
            A list of all the model's parameter to get this evaluation
            input's broadcast shape in relation to.
        model_set_axis : int
            The default model_set_axis
        n_models : int
            The number of models in the model set
        model_set_axis_input : int
            The model_set_axis passed in at evaluation

        Returns
        -------
        The pivot information from just the inputs
        """
        pivots = []
        for name, _input in self._inputs.items():
            self._inputs[name], pivot = _input.new_input(params, model_set_axis,
                                                n_models, model_set_axis_input)
            pivots.append(pivot)

        return pivots

    def pivots(self, params: list, model_set_axis: int, n_models: int,
                   n_outputs: int, model_set_axis_input: int) -> list:
        """
        Creates all the pivot information.
            This method is for replacing _prepare_inputs_model_set

        Parameters
        ----------
        params : list
            A list of all the model's parameter to get this evaluation
            input's broadcast shape in relation to.
        model_set_axis : int
            The default model_set_axis
        n_models : int
            The number of models in the model set
        n_outputs : int
            Number of outputs of the model being evaluated.
        model_set_axis_input : int
            The model_set_axis passed in at evaluation

        Returns
        -------
        The pivot information
        """
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
        A new set of Inputs which have been adjusted
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
        A new set of Inputs which have been adjusted
        """
        # if not all inputs are not in bounding box, adjust them
        if all_out:
            return self.copy()
        else:
            return self._reduce_to_bounding_box(valid_index)

    def convert_units(self, model: 'Model', equivalencies: dict):

        # We now iterate over the different inputs and make sure that their
        # units are consistent with those specified in input_units.
        for entry in self._inputs.values():
            entry.convert_unit(model, equivalencies)


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
                (self.default_model_set_axis == this.default_model_set_axis)
        else:
            return False

    @property
    def optional(self) -> dict:
        return self._optional

    @property
    def values(self) -> list:
        return [value for value in self._optional.values()]

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

    @property
    def default_model_set_axis(self) -> int:
        return self._model_set_axis

    def _get_model_option(self, name: str):
        """
        Get a modeling_option by name.
        """
        if name in self._model_options:
            return self._model_options[name]
        else:
            raise RuntimeError(f'Option "{name}" must be set!')

    @property
    def model_set_axis(self) -> int:
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

    def validate(self, pass_optional: bool):
        """
        Validates if the Optional arguments fit with passing optional
        arguments or not.

        Parameters
        ----------
        pass_optional : bool
            Whether or not, undefined optional arguments will be passed.
        """
        if (not pass_optional) and (len(self.pass_through) > 0):
            raise RuntimeError(f'Unknown optional arguments: {self.pass_through.keys()} ' +
                               'have been passed, argument pass through is off.')


class InputData(object):
    def __init__(self, shape: tuple=None,
                 format_info: list=None,
                 valid_index: np.ndarray=None,
                 all_out: bool=None,
                 with_bounding_box: bool=False):
        self._shape = shape
        self._format_info = format_info
        self._valid_index = valid_index
        self._all_out = all_out
        self._with_bounding_box = with_bounding_box

    def __eq__(self, this):
        if isinstance(this, InputData):
            return (self.shape == this.shape) and \
                (self.format_info == this.format_info) and \
                (self.valid_index == this.valid_index).all() and \
                (self.all_out == this.all_out) and \
                (self.with_bounding_box == this.with_bounding_box)
        else:
            return False

    @property
    def shape(self) -> tuple:
        if self._shape is None:
            return ()
        else:
            return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

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

    @property
    def with_bounding_box(self) -> bool:
        return self._with_bounding_box

    @with_bounding_box.setter
    def with_bounding_box(self, value):
        self._with_bounding_box = value

    def copy(self) -> 'InputData':
        return deepcopy(self)

    def reduce_to_bounding_box(self, valid_index, all_out: bool, input_shape: tuple) -> 'InputData':
        new = self.copy()
        new.valid_index = valid_index
        new.all_out = all_out
        new.shape = input_shape
        new.with_bounding_box = True

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
    def model_set_axis(self) -> int:
        return self._optional.model_set_axis

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

    @property
    def values(self) -> list:
        return list(chain(self._inputs.values, self._optional.values))

    def check_input_shape(self, n_models: int):
        """
        Validates the inputs all have compatible shapes

        Parameters
        ----------
        n_models : int
            The number of models in the model set.
        """
        self._inputs.check_input_shape(n_models, self._optional.model_set_axis, False)

    @classmethod
    def evaluation_inputs(cls, inputs: Inputs, optional: Optional) -> 'EvaluationInputs':
        """
        Construct an EvaluationInputs object, where the input data is empty.

        Parameters
        ----------
        inputs : Inputs
            The wrapped user required inputs
        optional: Optional
            The wrapped user optional inputs

        Returns
        -------
        An EvaluationInputs inputs object with empty input data.
        """
        return cls(inputs, optional, InputData())

    def set_format_info(self, params: list, standard_broadcasting: bool,
                        n_models: int, model_set_axis: int, n_outputs: int):
        """
        Creates and sets the format_info for the EvaluationInputs

        Parameters
        ----------
        params : list
            A list of all the model's parameter to get this evaluation
            input's broadcast shape in relation to.
        standard_broadcasting : bool
            If model uses standard numpy broadcasting
        n_models : int
            The number of models in the model set.
        model_set_axis : int
            The default model_set_axis for the model
        n_outputs : int
            The number of expected outputs
        """
        if n_models == 1:
            format_info = self._inputs.broadcast(params, standard_broadcasting, n_outputs)
        else:
            format_info = self._inputs.pivots(params, model_set_axis, n_models, n_outputs,
                                              self._optional.model_set_axis)
        self.format_info = format_info

    def reduce_to_bounding_box(self, valid_index, all_out: bool, input_shape: tuple):
        """
        Reduce the inputs in this object down to just those in the bounding_box,
        and record where valid indices are located.

        Parameters
        ----------
        valid_index : Tuple[np.ndarray, ...]
            The indices of this input found to be corrisponding to points
            within the bounding box of the model
        all_out : bool
            If all the indices will be used
        """
        self.inputs = self._inputs.reduce_to_bounding_box(valid_index, all_out)
        self.data = self._data.reduce_to_bounding_box(valid_index, all_out, input_shape)

    def _equivalencies(self, model) -> dict:
        # If a leaflist is provided that means this is in the context of
        # a compound model and it is necessary to create the appropriate
        # alias for the input coordinate name for the equivalencies dict
        inputs_map = self._optional.inputs_map
        equivalencies = self._optional.equivalencies

        if inputs_map:
            edict = {}
            for mod, mapping in inputs_map:
                if model is mod:
                    edict[mapping[0]] = equivalencies[mapping[1]]
        else:
            edict = equivalencies

        return edict

    def _combine_equivalency_dict(self, model) -> dict:
        return _combine_equivalency_dict(self._inputs.names,
                                         self._equivalencies(model),
                                         model.input_units_equivalencies)

    def enforce_units(self, model):
        if model.input_units is not None:
            input_equivalencies = self._combine_equivalency_dict(model)
            self._inputs.convert_units(model, input_equivalencies)


class OutputEntry(IoEntry):
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

    def _check_broadcast(self, format_info, index: int):
        try:
            return check_broadcast(*format_info)
        except (IndexError, TypeError):
            return format_info[index]

    def prepare_output_single_model(self, format_info, index: int) -> "OutputEntry":
        broadcast_shape = self._check_broadcast(format_info, index)

        if broadcast_shape is None:
            return self
        else:
            return OutputEntry(self.name, self._new_output(broadcast_shape))

    def prepare_output_model_set(self, pivots: list, model_set_axis: int, index: int) -> "OutputEntry":
        pivot = pivots[index]
        if pivot < self.value.ndim and pivot != model_set_axis:
            return OutputEntry(self.name,
                               np.rollaxis(self.value, pivot, model_set_axis))
        else:
            return self

    def prepare_input(self) -> InputEntry:
        return InputEntry(self.name, self.value)

    def process_units(self, return_units: dict) -> "OutputEntry":
        unit = return_units.get(self._name, None)
        value = Quantity(self._value, unit, subok=True)

        return OutputEntry(self._name, value)


class Outputs(object):
    def __init__(self, outputs: Dict[str, OutputEntry]):
        self._outputs = outputs

    def __eq__(self, this):
        if isinstance(this, Outputs):
            return (self.outputs == this.outputs)
        else:
            return False

    @property
    def names(self) -> List[str]:
        return list(self._outputs.keys())

    @property
    def n_outputs(self) -> int:
        return len(self._outputs)

    @property
    def outputs(self) -> Dict[str, OutputEntry]:
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        self._outputs = value

    @property
    def scalars(self):
        scalars = tuple([entry.scalar for entry in self._outputs.values()])
        if self.n_outputs == 1:
            return scalars[0]
        else:
            return scalars

    def prepare_outputs_single_model(self, format_info: list) -> "Outputs":
        outputs = {}
        for index, (name, output) in enumerate(self._outputs.items()):
            outputs[name] = output.prepare_output_single_model(format_info, index)

        return Outputs(outputs)

    def prepare_outputs_model_set(self, pivots: list, model_set_axis: int) -> "Outputs":
        outputs = {}
        for index, (name, output) in enumerate(self._outputs.items()):
            outputs[name] = output.prepare_output_model_set(pivots, model_set_axis, index)

        return Outputs(outputs)

    def _prepare_outputs(self, n_models: int, inputs: EvaluationInputs) -> "Outputs":
        if n_models == 1:
            return self.prepare_outputs_single_model(inputs.format_info)
        else:
            return self.prepare_outputs_model_set(inputs.format_info, inputs.model_set_axis)

    def _return_units(self, return_units) -> dict:
        if self.n_outputs == 1 and not isiterable(return_units):
            return {self.names[0]: return_units}
        else:
            return return_units

    def _process_units(self, return_units: dict) -> "Outputs":
        outputs = {}
        for name, entry in self._outputs.items():
            outputs[name] = entry.process_units(return_units)

        return Outputs(outputs)

    def process_units(self, inputs: EvaluationInputs, return_units) -> "Outputs":
        if return_units and inputs.inputs.are_quantity:
            return self._process_units(self._return_units(return_units))
        else:
            return self

    def prepare_outputs(self, n_models: int, inputs: EvaluationInputs, return_units) -> "Outputs":
        outputs = self._prepare_outputs(n_models, inputs)

        return outputs.process_units(inputs, return_units)


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
        """
        Helper function for self.get_from_kwargs and InputMetaData.create_input
            Attempts to Wrap the input value in the right evaluation entry wrapper.

        Parameters
        ----------
        value :
            The users input value to wrap

        Returns
        -------
            The input value correctly wrapped.
        """
        if self._data_entry is None or isinstance(value, self._data_entry):
            return value
        else:
            return self._data_entry(self.name, value)

    def get_from_kwargs(self, data_kwargs: dict, **kwargs) -> dict:
        """
        If the evaluation input described is present in the kwargs,
        it is extracted and wrapped from the kwargs and added to a dictionary.

        Parameters
        ----------
        data_kwargs : dict
            The dictionary to add the input to if it is present.
        **kwargs :
            The remaining set of kwargs passed as evaluation inputs by
            the user

        Returns
        -------
        The kwargs without the entry described by this object.
        """
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

    def _fill_defaults(self, n_data: int, **kwargs):
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
        new._fill_defaults(n_data, **kwargs)

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

    def __init__(self, name: str=None, pos: int=None, unit: UnitBase=None,
                 bounding_box: _BoundingBox=None):
        super().__init__(name)

        self._pos = pos
        self._unit = unit
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
    def unit(self) -> Union[UnitBase, None]:
        return self._unit

    @unit.setter
    def unit(self, value):
        if self._unit is not None and self._unit != dimensionless_unscaled:
            raise ValueError(f"Cannot override existing units for input {self.name}")
        else:
            self._unit = value

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

    def create_input(self, *args, **kwargs) -> Tuple[InputEntry, tuple]:
        """
        Creates an InputEntry object for the input described by this set of
        meta data and then adds that entry to a dictionary under the name of
        the input.

        Parameters
        ----------
        *args :
            The positional arguments passed to model for evaluation
        **kwargs :
            The required arguments passed to the model as kwargs for
            evaluation

        Returns
        -------
        tuple(
            The users inputs wrapped in an Inputs object,
            Tuple of remaining positional arguments.
        )
        """
        args = list(args)
        if self.name in kwargs:
            value = self._create_io_entry(kwargs[self.name])
        else:
            value = self._create_io_entry(args.pop(0))

        return value, tuple(args)

    def _outside(self, inputs: EvaluationInputs) -> Tuple[np.ndarray, tuple]:
        """
        Helper function for self.update_outside.
            Get the boolean array of where the input for this entry is
            outside its bounding_box

        Parameters
        ----------
        inputs : EvaluationInputs
            The processed evaluation inputs

        Returns
        -------
        tuple(
            The array booleans indicating positions outside bounding_box,
            shape of input
        )
        """
        if self.name in inputs.inputs.inputs:
            value = inputs.inputs.inputs[self.name].input_array

            return self.bounding_box.outside(value), value.shape
        else:
            raise RuntimeError(f'Input: {self.name} not present in inputs')

    def update_outside(self, outside: np.ndarray, all_out: bool,
                       inputs: EvaluationInputs) -> Tuple[np.ndarray, bool]:
        """
        Updates array of outside positions based on the data from this entry

        Parameters
        ----------
        outside : bool np.ndarray
            Array of booleans indicating which positions are outside the
            bounding_box
        all_out : bool
            If all the input positions will be used
        inputs : EvaluationInputs
            The processed evaluation inputs

        Returns
        -------
        tuple(
            Updated outside array,
            If all the input positions are inside the box
        )
        """
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
                 bounding_box: _BoundingBox=None, **kwargs):
        super().__init__(inputs)
        self._n_inputs = n_data
        self.bounding_box = bounding_box

    def _fill_defaults(self, n_data: int, **kwargs):
        """
        Helper function for create_defaults
            Generates the required number of inputs under the standard
            names.
        """

        if 'n_outputs' in kwargs:
            n_outputs = kwargs['n_outputs']
        else:
            n_outputs = 1

        if n_data == 1 and n_outputs == 1:
            self.inputs = ['x']
        elif n_data == 2 and n_outputs == 1:
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

    def set_unit(self, name: str, unit: UnitBase):
        if name in self._data:
            self._data[name].unit = unit
        else:
            raise RuntimeError(f'No input {name} found')

    def get_unit(self, name: str) -> UnitBase:
        if name in self._data:
            return self._data[name].unit
        else:
            raise RuntimeError(f'No input {name} found')

    @property
    def units(self) -> Union[Dict[str, UnitBase], None]:
        units = {name: entry.unit for name, entry in self._data.items() if entry.unit is not None}
        if units == {}:
            return None
        else:
            return units

    @units.setter
    def units(self, value: Dict[str, UnitBase]):
        for name, unit in value.items():
            self.set_unit(name, unit)

    def _check_inputs(self, *args, **kwargs):
        """
        Helper function for self.get_inputs
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
        Helper function for self.get_inputs
            Turns user evaluation inputs into Input object. This is accomplished
            by extracting the InputEntry object for each required input. These
            objects are then turned into a complete Inputs object.

        Parameters
        ----------
        args :
            The positional arguments passed to model for evaluation
        kwargs :
            The required arguments passed to the model as kwargs for
            evaluation

        Returns
        -------
        The users inputs wrapped in an Inputs object
        """
        inputs = {}
        for name, _input in self._data.items():
            inputs[name], args = _input.create_input(*args, **kwargs)

        return Inputs(inputs)

    def get_inputs(self, *args, **kwargs) -> Tuple[Inputs, dict]:
        """
        Performs extracts the Inputs object from the user supplied
        evaluation arguments.

        Parameters
        ----------
        args :
            The positional arguments passed to model for evaluation
        kwargs :
            The kwargs passed to the model for evaluation

        Returns
        -------
        tuple(
            Inputs object
            kwargs with no required inputs included
        )

        Notes
        -----
        Performs a basic consistency check of the user supplied evaluation
        inputs while generation of the Inputs Object
        """
        input_kwargs, kwargs = self.get_from_kwargs(**kwargs)
        self._check_inputs(*args, **input_kwargs)

        return self._create_inputs(*args, **input_kwargs), kwargs

    def _outside(self, n_models: int, model_set_axis: int, inputs: EvaluationInputs) -> Tuple[np.ndarray, bool, tuple]:
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

        return outside, all_out, input_shape

    def _get_valid_index(self, n_models: int, model_set_axis: int, inputs: EvaluationInputs) \
            -> Tuple[Tuple[np.ndarray, ...], bool, tuple]:
        """
        Helper function for self.enforce_bounding_box
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
        outside_inputs, all_out, input_shape = self._outside(n_models, model_set_axis, inputs)

        # get an array with indices of valid inputs
        valid_index = np.atleast_1d(np.logical_not(outside_inputs)).nonzero()
        if len(valid_index[0]) == 0:
            all_out = True

        return valid_index, all_out, input_shape

    def enforce_bounding_box(self, n_models: int, model_set_axis: int, inputs: EvaluationInputs):
        """
        Updates the EvaluationInputs object so that its values are all inside the bounding box

        Parameters
        ----------
        n_models: int
            The number of models
        inputs : EvaluationInputs
            The processed evaluation inputs object
        """
        # NOTE: this is to replace prepare_bounding_box_inputs
        if inputs.optional.with_bounding_box:
            valid_index, all_out, input_shape = self._get_valid_index(n_models, model_set_axis, inputs)
            inputs.reduce_to_bounding_box(valid_index, all_out, input_shape)


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

    def _fill_defaults(self, n_data: int, **kwargs):
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

    def _get_model_options(self, optional: dict, **kwargs) -> Optional:
        """
        Helper function for self.get_options
            Fills in all the optional inputs

        Parameters
        ----------
        optional : dict
            The optional inputs described by this object
        kwargs :
            User's kwargs with the required inputs and optional removed

        Returns
        -------
        The Optional inputs object

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

        return Optional(options, model_options, kwargs, self._model_set_axis)

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
        options = self._get_model_options(optional, **kwargs)
        options.validate(self.pass_optional)

        return options


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

    @staticmethod
    def _create_base_result(input_data: InputData, fill_value) -> np.ndarray:
        return np.zeros(input_data.shape) + fill_value

    @staticmethod
    def _modify_unit(result: np.ndarray, value: np.ndarray) -> np.ndarray:
        unit = getattr(value, 'unit', None)
        if unit is not None:
            result = Quantity(result, unit, copy=False)

        return result

    def _modifly_result(self, result: np.ndarray, value: np.ndarray, input_data: InputData, fill_value) -> np.ndarray:
        if result.shape:
            result[input_data.valid_index] = value
        else:
            result = np.array(value)

        return self._modify_unit(result, value)

    def _create_bounding_box_output(self, value: np.ndarray, input_data: InputData, fill_value) -> OutputEntry:
        result = self._create_base_result(input_data, fill_value)

        if not input_data.all_out:
            result = self._modifly_result(result, value, input_data, fill_value)

        return OutputEntry(self.name, result)

    def _create_output(self, value: np.ndarray, input_data: InputData, fill_value) -> OutputEntry:
        if input_data.with_bounding_box:
            return self._create_bounding_box_output(value, input_data, fill_value)
        else:
            return OutputEntry(self.name, value)

    def create_output(self, input_data: InputData, fill_value, *args) -> Tuple[OutputEntry, tuple]:
        args = list(args)
        value = self._create_output(args.pop(0), input_data, fill_value)

        return value, tuple(args)


class OutputMetaData(IoMetaData):
    _data_entry = OutputMetaDataEntry

    def __init__(self, n_data: int,
                 outputs: Dict[str, OutputMetaDataEntry] = None, **kwargs):
        super().__init__(outputs)
        self._n_outputs = n_data

    def _fill_defaults(self, n_data: int, **kwargs):
        """
        Helper function for create_defaults
            Generates the required number of outputs under the standard
            names.
        """

        if 'n_inputs' in kwargs:
            n_inputs = kwargs['n_inputs']
        else:
            n_inputs = 0

        if n_inputs == 1 and n_data == 1:
            self.outputs = ['y']
        elif n_inputs == 2 and n_data == 1:
            self.outputs = ['z']
        else:
            self.outputs = [f'x{idx}' for idx in range(n_data)]

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

    def _create_outputs(self, input_data: InputData, fill_value, *args) -> Outputs:
        outputs = {}
        for name, _output in self._data.items():
            outputs[name], args = _output.create_output(input_data, fill_value, *args)

        if len(args) > 0:
            raise RuntimeError('Too many outputs have been generated by model')

        return Outputs(outputs)

    def get_outputs(self, inputs: EvaluationInputs, results) -> Outputs:
        input_data = inputs.data
        fill_value = inputs.optional.fill_value

        if self._n_outputs == 1:
            results = [results,]
        return self._create_outputs(input_data, fill_value, *results)


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
        inputs = InputMetaData.create_defaults(n_inputs, n_outputs=n_outputs)

        if optional is None:
            optional = OptionalMetaData.create_defaults(0, pass_optional=pass_optional,
                                                        model_set_axis=model_set_axis)
        else:
            optional.pass_optional = pass_optional
            optional.model_set_axis = model_set_axis

        outputs = OutputMetaData.create_defaults(n_outputs, n_inputs=n_inputs)

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

    def set_input_unit(self, name: str, unit: UnitBase):
        self._inputs.set_unit(name, unit)

    def get_input_unit(self, name: str) -> UnitBase:
        return self._inputs.get_unit(name)

    @property
    def input_units(self) -> Union[Dict[str, UnitBase], None]:
        return self._inputs.units

    @input_units.setter
    def input_units(self, value: Dict[str, UnitBase]):
        self._inputs.units = value

    def evaluation_inputs(self, *args, **kwargs) -> EvaluationInputs:
        """
        Turn the user's evaluation inputs to a model into an EvaluationInputs object

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
        """
        Sets the format_info for the EvaluationInputs object.

        Parameters
        ----------
        params : list
            A list of all the model's parameter to get this evaluation
            input's broadcast shape in relation to.
        inputs : EvaluationInputs
            The wrapped user inputs.
        """
        inputs.set_format_info(params, self.standard_broadcasting,
                               self.n_models, self.model_set_axis, self.n_outputs)

    def enforce_bounding_box(self, inputs: EvaluationInputs):
        """
        Enforces the bounding_box on EvaluationInputs

        Parameters
        ----------
        inputs : EvaluationInputs
            The wrapped user inputs.
        """
        self._inputs.enforce_bounding_box(self.n_models, self.model_set_axis, inputs)

    def process_inputs(self, params: list,
                       model: 'Model',
                       inputs: EvaluationInputs):
        """
        Process the EvaluationInputs object in light of the meta_data:
            Checks input shapes
            Generates the internal format data for inputs
            Enforces units (if needed)
            Enforces bounding_box (if needed)

        Parameters
        ----------
        params : list
            A list of all the model's parameter to get this evaluation
            input's broadcast shape in relation to.
        inputs : EvaluationInputs
            The wrapped user inputs.
        """
        inputs.check_input_shape(self.n_models)
        inputs.enforce_units(model)

        # Generate the format info for the inputs
        self.set_format_info(params, inputs)

        # Enforce the bounding box
        self.enforce_bounding_box(inputs)

    def prepare_inputs(self, params: list, model: 'Model',
                       *args, **kwargs) -> EvaluationInputs:
        # Process user inputs into the wrapper
        inputs = self.evaluation_inputs(*args, **kwargs)

        # Process the input wrapper
        self.process_inputs(params, model, inputs)

        return inputs

    def prepare_outputs(self, inputs: EvaluationInputs, return_units, results) -> Outputs:
        outputs = self._outputs.get_outputs(inputs, results)

        return outputs.prepare_outputs(self._n_models, inputs, return_units)
