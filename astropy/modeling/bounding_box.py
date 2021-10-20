# Licensed under a 3-clause BSD style license - see LICENSE.rst


"""
This module is to contain an improved bounding box
"""

from collections import namedtuple
from typing import TYPE_CHECKING, Dict, List, Tuple, Callable, Any

from astropy.utils import isiterable
from astropy.units import Quantity

import warnings
import numpy as np


__all__ = ['Interval', 'BoundingDomain', 'BoundingBox', 'SliceArgument',
           'SliceArguments', 'CompoundBoundingBox']


_BaseInterval = namedtuple('_BaseInterval', "lower upper")


class Interval(_BaseInterval):
    """
    A single input's bounding box interval.

    Parameters
    ----------
    lower : float
        The lower bound of the interval

    upper : float
        The upper bound of the interval

    Methods
    -------
    validate :
        Contructs a valid interval

    outside :
        Determine which parts of an input array are outside the interval.

    domain :
        Contructs a discretization of the points inside the interval.
    """

    @staticmethod
    def _validate_shape(interval):
        """Validate the shape of an interval representation"""
        MESSAGE = """An interval must be some sort of sequence of length 2"""

        try:
            shape = np.shape(interval)
        except TypeError:
            try:
                # np.shape does not work with lists of Quantities
                if len(interval) == 1:
                    interval = interval[0]
                shape = np.shape([b.to_value() for b in interval])
            except (ValueError, TypeError, AttributeError):
                raise ValueError(MESSAGE)

        valid_shape = shape in ((2,), (1, 2), (2, 0))
        if not valid_shape:
            valid_shape = (len(shape) > 0) and (shape[0] == 2) and \
                all(isinstance(b, np.ndarray) for b in interval)

        if not isiterable(interval) or not valid_shape:
            raise ValueError(MESSAGE)

    @classmethod
    def _validate_bounds(cls, lower, upper):
        """Validate the bounds are reasonable and construct an interval from them."""
        if (np.asanyarray(lower) > np.asanyarray(upper)).all():
            warnings.warn(f"Invalid interval: upper bound {upper} "
                          f"is strictly less than lower bound {lower}.", RuntimeWarning)

        return cls(lower, upper)

    @classmethod
    def validate(cls, interval):
        """
        Construct and validate an interval

        Parameters
        ----------
        interval : iterable
            A representation of the interval.

        Returns
        -------
        A validated interval.
        """
        cls._validate_shape(interval)

        if len(interval) == 1:
            interval = tuple(interval[0])
        else:
            interval = tuple(interval)

        return cls._validate_bounds(interval[0], interval[1])

    def outside(self, _input: np.ndarray):
        """
        Parameters
        ----------
        _input : np.ndarray
            The evaluation input in the form of an array.

        Returns
        -------
        Boolean array indicating which parts of _input are outside the interval:
            True  -> position outside interval
            False -> position inside  interval
        """
        return np.logical_or(_input < self.lower, _input > self.upper)

    def domain(self, resolution):
        return np.arange(self.lower, self.upper + resolution, resolution)


# The interval where all ignored inputs can be found.
_ignored_interval = Interval.validate((-np.inf, np.inf))


class BoundingDomain(object):
    """
    Base class for BoundingBox and CompoundBoundingBox.
        This is where all the `~astropy.modeling.core.Model` evaluation
        code for evaluating with a bounding box is because it is common
        to both types of bounding box.

    Parameters
    ----------
    model : `~astropy.modeling.Model`
        The Model this bounding domain is for.

    prepare_inputs :
        Generates the necessary input information so that model can
        be evaluated only for input points entirely inside bounding_box.
        This needs to be implemented by a subclass. Note that most of
        the implementation is in BoundingBox.

    prepare_outputs :
        Fills the output values in for any input points outside the
        bounding_box.

    evaluate :
        Performs a complete model evaluation while enforcing the bounds
        on the inputs and returns a complete output.
    """

    def __init__(self, model):
        self._model = model

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "This bounding box is fixed by the model and does not have "
            "adjustable parameters.")

    def prepare_inputs(self, input_shape, inputs) -> Tuple[Any, Any, Any]:
        """
        Get prepare the inputs with respect to the bounding box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        all_out: bool
            if all of the inputs are outside the bounding_box
        """
        raise NotImplementedError("This has not been implemented for BoundingDomain.")

    @staticmethod
    def _base_output(input_shape, fill_value):
        """
        Create a baseline output, assuming that the entire input is outside
        the bounding box

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        An array of the correct shape containing all fill_value
        """
        return np.zeros(input_shape) + fill_value

    def _all_out_output(self, input_shape, fill_value):
        """
        Create output if all inputs are outside the domain

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        A full set of outputs for case that all inputs are outside domain.
        """

        return [self._base_output(input_shape, fill_value)
                for _ in range(self._model.n_outputs)], None

    def _modify_output(self, valid_output, valid_index, input_shape, fill_value):
        """
        For a single output fill in all the parts corresponding to inputs
        outside the bounding box.

        Parameters
        ----------
        valid_output : numpy array
            The output from the model corresponding to inputs inside the
            bounding box
        valid_index : numpy array
            array of all indices of inputs inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        An output array with all the indices corresponding to inputs
        outside the bounding box filled in by fill_value
        """
        output = self._base_output(input_shape, fill_value)
        if not output.shape:
            output = np.array(valid_output)
        else:
            output[valid_index] = valid_output

        return output

    def _prepare_outputs(self, valid_outputs, valid_index, input_shape, fill_value):
        """
        Fill in all the outputs of the model corresponding to inputs
        outside the bounding_box.

        Parameters
        ----------
        valid_outputs : list of numpy array
            The list of outputs from the model corresponding to inputs
            inside the bounding box
        valid_index : numpy array
            array of all indices of inputs inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box

        Returns
        -------
        List of filled in output arrays.
        """
        outputs = []
        for valid_output in valid_outputs:
            outputs.append(self._modify_output(valid_output, valid_index, input_shape, fill_value))

        return outputs

    def prepare_outputs(self, valid_outputs, valid_index, input_shape, fill_value):
        """
        Fill in all the outputs of the model corresponding to inputs
        outside the bounding_box, adjusting any single output model so that
        its output becomes a list of containing that output.

        Parameters
        ----------
        valid_outputs : list
            The list of outputs from the model corresponding to inputs
            inside the bounding box
        valid_index : array_like
            array of all indices of inputs inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        """
        if self._model.n_outputs == 1:
            valid_outputs = [valid_outputs]

        return self._prepare_outputs(valid_outputs, valid_index, input_shape, fill_value)

    @staticmethod
    def _get_valid_outputs_unit(valid_outputs, with_units: bool):
        """
        Get the unit for outputs if one is required.

        Parameters
        ----------
        valid_outputs : list of numpy array
            The list of outputs from the model corresponding to inputs
            inside the bounding box
        with_units : bool
            whether or not a unit is required
        """

        if with_units:
            return getattr(valid_outputs, 'unit', None)

    def _evaluate_model(self, evaluate: Callable, valid_inputs, valid_index,
                        input_shape, fill_value, with_units: bool):
        """
        Evaluate the model using the given evaluate routine

        Parameters
        ----------
        evaluate : Callable
            callable which takes in the valid inputs to evaluate model
        valid_inputs : list of numpy arrays
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : numpy array
            array of all indices inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        with_units : bool
            whether or not a unit is required

        Returns
        -------
        outputs :
            list containing filled in output values
        valid_outputs_unit :
            the unit that will be attached to the outputs
        """

        valid_outputs = evaluate(valid_inputs)
        valid_outputs_unit = self._get_valid_outputs_unit(valid_outputs, with_units)

        return self.prepare_outputs(valid_outputs, valid_index,
                                    input_shape, fill_value), valid_outputs_unit

    def _evaluate(self, evaluate: Callable, inputs, input_shape,
                  fill_value, with_units: bool):
        """
        Perform model evaluation steps:
            prepare_inputs -> evaluate -> prepare_outputs

        Parameters
        ----------
        evaluate : Callable
            callable which takes in the valid inputs to evaluate model
        valid_inputs : list of numpy arrays
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : numpy array
            array of all indices inside the bounding box
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        with_units : bool
            whether or not a unit is required

        Returns
        -------
        outputs :
            list containing filled in output values
        valid_outputs_unit :
            the unit that will be attached to the outputs
        """

        valid_inputs, valid_index, all_out = self.prepare_inputs(input_shape, inputs)

        if all_out:
            return self._all_out_output(input_shape, fill_value)
        else:
            return self._evaluate_model(evaluate, valid_inputs, valid_index,
                                        input_shape, fill_value, with_units)

    @staticmethod
    def _set_outputs_unit(outputs, valid_outputs_unit):
        """
        Perform full model evaluation steps:
            prepare_inputs -> evaluate -> prepare_outputs -> set output units

        Parameters
        ----------
        outputs :
            list containing filled in output values
        valid_outputs_unit :
            the unit that will be attached to the outputs

        Returns
        -------
        List containing filled in output values and units
        """

        if valid_outputs_unit is not None:
            return Quantity(outputs, valid_outputs_unit, copy=False)

        return outputs

    def evaluate(self, evaluate: Callable, inputs, fill_value):
        """
        Perform full model evaluation steps:
            prepare_inputs -> evaluate -> prepare_outputs -> set output units

        Parameters
        ----------
        evaluate : callable
            callable which takes in the valid inputs to evaluate model
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        fill_value : float
            The value which will be assigned to inputs which are outside
            the bounding box
        """
        input_shape = self._model.input_shape(inputs)
        # NOTE: CompoundModel does not currently support units during
        #   evaluation for bounding_box so this feature is turned off
        #   for CompoundModel(s).
        outputs, valid_outputs_unit = self._evaluate(evaluate, inputs, input_shape,
                                                     fill_value, self._model.bbox_with_units)
        return self._set_outputs_unit(outputs, valid_outputs_unit)


def get_name(model, index: int):
    """Get the input name corresponding to the input index"""
    return model.inputs[index]


def get_index(model, key) -> int:
    """
    Get the input index corresponding to the given key.
        Can pass in either:
            the string name of the input or
            the input index itself.
    """
    if isinstance(key, str):
        if key in model.inputs:
            index = model.inputs.index(key)
        else:
            raise ValueError(f"'{key}' is not one of the inputs: {model.inputs}.")
    elif np.issubdtype(type(key), np.integer):
        if key < len(model.inputs):
            index = key
        else:
            raise IndexError(f"Integer key: {key} must be < {len(model.inputs)}.")
    else:
        raise ValueError(f"Key value: {key} must be string or integer.")

    return index


class BoundingBox(BoundingDomain):
    """
    A model's bounding box

    Parameters
    ----------
    intervals : dict
        A dictionary containing all the intervals for each model input
            keys   -> input index
            values -> interval for that index

    model : `~astropy.modeling.Model`
        The Model this bounding_box is for.

    ignored : list
        A list containing all the inputs (index) which will not be
        checked for whether or not their elements are in/out of an interval.

    order : optional, str
        The ordering that is assumed for the tuple representation of this
        bounding_box. Options: 'C': C/Python order, e.g. z, y, x.
        (default), 'F': Fortran/mathematical notation order, e.g. x, y, z.

    Methods
    -------
    validate :
        Constructs a valid bounding_box from any of the allowed
        respresentations of a bounding_box.

    bounding_box :
        Contructs a tuple respresentation

    domain :
        Contructs a discretization of the points inside the bounding_box
    """

    def __init__(self, intervals: Dict[int, Interval], model,
                 ignored: List[int] = None, order: str = 'C'):
        super().__init__(model)
        self._order = order

        self._ignored = self._validate_ignored(ignored)

        self._intervals = {}
        if intervals != () and intervals != {}:
            self._validate(intervals, order=order)

    @property
    def intervals(self) -> Dict[int, Interval]:
        """Return bounding_box labeled using input positions"""
        return self._intervals

    @property
    def order(self) -> str:
        return self._order

    @property
    def ignored(self) -> List[int]:
        return self._ignored

    def _get_name(self, index: int):
        """Get the input name corresponding to the input index"""
        return get_name(self._model, index)

    @property
    def named_intervals(self) -> Dict[str, Interval]:
        """Return bounding_box labeled using input names"""
        return {self._get_name(index): bbox for index, bbox in self._intervals.items()}

    @property
    def ignored_inputs(self) -> List[str]:
        return [self._get_name(index) for index in self._ignored]

    def __repr__(self):
        parts = [
            'BoundingBox(',
            '    intervals={'
        ]

        for name, interval in self.named_intervals.items():
            parts.append(f"        {name}: {interval}")

        parts.append('    }')
        if len(self._ignored) > 0:
            parts.append(f"    ignored={self.ignored_inputs}")

        parts.append(f'    model={self._model.__class__.__name__}(inputs={self._model.inputs})')
        parts.append(f"    order='{self._order}'")
        parts.append(')')

        return '\n'.join(parts)

    def _get_index(self, key) -> int:
        """
        Get the input index corresponding to the given key.
            Can pass in either:
                the string name of the input or
                the input index itself.
        """

        return get_index(self._model, key)

    def _validate_ignored(self, ignored: list) -> List[int]:
        if ignored is None:
            return []
        else:
            return [self._get_index(key) for key in ignored]

    def __len__(self):
        return len(self._intervals)

    def __contains__(self, key):
        try:
            return self._get_index(key) in self._intervals or self._ignored
        except (IndexError, ValueError):
            return False

    def __getitem__(self, key):
        """Get bounding_box entries by either input name or input index"""
        index = self._get_index(key)
        if index in self._ignored:
            return _ignored_interval
        else:
            return self._intervals[self._get_index(key)]

    def _get_order(self, order: str = None) -> str:
        """
        Get if bounding_box is C/python ordered or Fortran/mathematically
        ordered
        """
        if order is None:
            order = self._order

        if order not in ('C', 'F'):
            raise ValueError("order must be either 'C' (C/python order) or "
                             f"'F' (Fortran/mathematical order), got: {order}.")

        return order

    def bounding_box(self, order: str = None):
        """
        Return the old tuple of tuples representation of the bounding_box
            order='C' corresponds to the old bounding_box ordering
            order='F' corresponds to the gwcs bounding_box ordering.
        """
        if len(self._intervals) == 1:
            return self._intervals[0]
        else:
            order = self._get_order(order)
            inputs = self._model.inputs
            if order == 'C':
                inputs = inputs[::-1]

            return tuple([self[input_name] for input_name in inputs])

    def __eq__(self, value):
        """Note equality can be either with old representation or new one."""
        if isinstance(value, tuple):
            return self.bounding_box() == value
        elif isinstance(value, BoundingBox):
            return (self.intervals == value.intervals) and (self.ignored == value.ignored)
        else:
            return False

    def __setitem__(self, key, value):
        """Validate and store interval under key (input index or input name)."""
        index = self._get_index(key)
        if index in self._ignored:
            self._ignored.remove(index)

        self._intervals[index] = Interval.validate(value)

    def __delitem__(self, key):
        """Delete stored interval"""
        index = self._get_index(key)
        del self._intervals[index]
        self._ignored.append(index)

    def _validate_dict(self, bounding_box: dict):
        """Validate passing dictionary of intervals and setting them."""
        for key, value in bounding_box.items():
            self[key] = value

    def _validate_sequence(self, bounding_box, order: str = None):
        """Validate passing tuple of tuples representation (or related) and setting them."""
        order = self._get_order(order)
        if order == 'C':
            # If bounding_box is C/python ordered, it needs to be reversed
            # to be in Fortran/mathematical/input order.
            bounding_box = bounding_box[::-1]

        for index, value in enumerate(bounding_box):
            self[index] = value

    @property
    def _n_inputs(self) -> int:
        return self._model.n_inputs - len(self._ignored)

    def _validate_iterable(self, bounding_box, order: str = None):
        """Validate and set any iterable representation"""
        if len(bounding_box) != self._n_inputs:
            raise ValueError(f"Found {len(bounding_box)} intervals, "
                             f"but must have exactly {self._n_inputs}.")

        if isinstance(bounding_box, dict):
            self._validate_dict(bounding_box)
        else:
            self._validate_sequence(bounding_box, order)

    def _validate(self, bounding_box, order: str = None):
        """Validate and set any representation"""
        if self._n_inputs == 1 and not isinstance(bounding_box, dict):
            self[0] = bounding_box
        else:
            self._validate_iterable(bounding_box, order)

    @classmethod
    def validate(cls, model, bounding_box,
                 ignored: list = None, order: str = 'C'):
        """
        Construct a valid bounding box for a model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model for which this will be a bounding_box
        bounding_box : dict, tuple
            A possible representation of the bounding box
        order : optional, str
            The order that a tuple representation will be assumed to be
                Default: 'C'
        """
        if isinstance(bounding_box, BoundingBox):
            order = bounding_box.order
            bounding_box = bounding_box.intervals

        new = cls({}, model, ignored=ignored, order=order)
        new._validate(bounding_box)

        return new

    def copy(self):
        return BoundingBox(self._intervals.copy(), self._model,
                           ignored=self._ignored.copy(), order=self._order)

    def fix_inputs(self, model, fixed_inputs: list):
        """
        Fix the bounding_box for a `fix_inputs` compound model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The new model for which this will be a bounding_box
        fixed_inputs : list
            List if inputs that have been fixed in this bounding_box
        """

        new = self.copy()

        for _input in fixed_inputs:
            del new[_input]

        return BoundingBox.validate(model, new.named_intervals,
                                    order=new._order)

    @property
    def dimension(self):
        return len(self)

    def domain(self, resolution, order: str = None):
        inputs = self._model.inputs
        order = self._get_order(order)
        if order == 'C':
            inputs = inputs[::-1]

        return [self[input_name].domain(resolution) for input_name in inputs]

    def _outside(self,  input_shape, inputs):
        """
        Get all the input positions which are outside the bounding_box,
        so that the corresponding outputs can be filled with the fill
        value (default NaN).

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        outside_index : bool-numpy array
            True  -> position outside bounding_box
            False -> position inside  bounding_box
        all_out : bool
            if all of the inputs are outside the bounding_box
        """
        all_out = False

        outside_index = np.zeros(input_shape, dtype=bool)
        for index, _input in enumerate(inputs):
            _input = np.asanyarray(_input)
            outside = self[index].outside(_input)

            if _input.shape:
                outside_index[outside] = True
            else:
                outside_index |= outside
                if outside_index:
                    all_out = True

        return outside_index, all_out

    def _valid_index(self, input_shape, inputs):
        """
        Get the indices of all the inputs inside the bounding_box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_index : numpy array
            array of all indices inside the bounding box
        all_out : bool
            if all of the inputs are outside the bounding_box
        """
        outside_index, all_out = self._outside(input_shape, inputs)

        valid_index = np.atleast_1d(np.logical_not(outside_index)).nonzero()
        if len(valid_index[0]) == 0:
            all_out = True

        return valid_index, all_out

    def prepare_inputs(self, input_shape, inputs) -> Tuple[Any, Any, Any]:
        """
        Get prepare the inputs with respect to the bounding box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        all_out: bool
            if all of the inputs are outside the bounding_box
        """

        valid_index, all_out = self._valid_index(input_shape, inputs)

        valid_inputs = []
        if not all_out:
            for _input in inputs:
                if input_shape:
                    valid_inputs.append(np.atleast_1d(_input)[valid_index])
                else:
                    valid_inputs.append(_input)

        return valid_inputs, valid_index, all_out


_BaseSliceArgument = namedtuple('_BaseSliceArgument', "index ignore")


class SliceArgument(_BaseSliceArgument):
    """
    Contains a single CompoundBoundingBox slicing input.

    Parameters
    ----------
    index : int
        The index of the input in the input list

    ignore : bool
        Whether or not this input will be ignored by the bounding box.

    model:  `~astropy.modeling.Model`
        The Model this argument is for.

    Methods
    -------
    validate :
        Returns a valid SliceArgument for a given model.

    get_slice :
        Returns the value of the input for use in finding the correct
        bounding_box.

    get_fixed_value :
        Gets the slicing value from a fix_inputs set of values.
    """

    _model = None

    def __new__(cls, index, ignore, model):
        self = super().__new__(cls, index, ignore)
        self._model = model

        return self

    @classmethod
    def validate(cls, model, argument, ignored: bool = False):
        """
        Construct a valid slice argument for a CompoundBoundingBox.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model for which this will be an argument for.
        argument : int or str
            A representation of which evaluation input to use
        ignored : optional, bool
            Whether or not to ignore this argument in the BoundingBox.

        Returns
        -------
        Validated slice_argument
        """
        return cls(get_index(model, argument), ignored, model)

    def get_slice(self, *inputs):
        """
        Get the slice value corresponding to this argument

        Parameters
        ----------
        *inputs :
            All the processed model evaluation inputs.
        """
        _slice = inputs[self.index]
        if isiterable(_slice):
            if len(_slice) == 1:
                return _slice[0]
            else:
                return tuple(_slice)
        return _slice

    def __repr__(self):
        return f"Argument(name='{get_name(self._model, self.index)}', ignore={self.ignore})"

    def get_fixed_value(self, values: dict):
        """
        Gets the value fixed input corresponding to this argument

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model this is an argument for

        values : dict
            Dictionary of fixed inputs.
        """
        if self.index in values:
            return values[self.index]
        else:
            name = get_name(self._model, self.index)
            if name in values:
                return values[name]
            else:
                raise RuntimeError(f"{self} was not found in {values}")


class SliceArguments(tuple):
    """
    Contains the CompoundBoundingBox slicing description

    Parameters
    ----------
    input_ :
        The SliceArgument values

    model : `~astropy.modeling.Model`
        The Model this is the SliceArguments for.

    Methods
    -------
    validate :
        Returns a valid SliceArguments for its model.

    get_slice :
        Returns the slice a set of inputs corresponds to.

    is_slice :
        Determines if a slice is correctly formatted for this CompoundBoundingBox.

    get_fixed_value :
        Gets the slice from a fix_inputs set of values.
    """

    _model = None

    def __new__(cls, input_: Tuple[SliceArgument], model):
        self = super().__new__(cls, input_)
        self._model = model

        return self

    def __repr__(self):
        parts = ['SliceArguments(']
        for argument in self:
            parts.append(
                f"    {argument}"
            )
        parts.append(')')

        return '\n'.join(parts)

    @property
    def ignore(self):
        """Get the list of ignored inputs"""
        return [argument.index for argument in self if argument.ignore]

    @classmethod
    def validate(cls, model, arguments):
        """
        Construct a valid Slice description for a CompoundBoundingBox.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model for which this will be an argument for.
        arguments :
            The individual argument informations
        """
        inputs = []
        for argument in arguments:
            _input = SliceArgument.validate(model, *argument)
            if _input.index in [this.index for this in inputs]:
                raise ValueError(f"Input: '{get_name(model, _input.index)}' has been repeated.")
            inputs.append(_input)

        if len(inputs) == 0:
            raise ValueError("There must be at least one slice argument.")

        return cls(tuple(inputs), model)

    def get_slice(self, *inputs):
        """
        Get the slice corresponding to these inputs

        Parameters
        ----------
        *inputs :
            All the processed model evaluation inputs.
        """
        return tuple([argument.get_slice(*inputs) for argument in self])

    def is_slice(self, _slice):
        """
        Determine if this is a reasonable slice

        Parameters
        ----------
        _slice : tuple
            The slice to check
        """
        return isinstance(_slice, tuple) and len(_slice) == len(self)

    def get_fixed_values(self, values: dict):
        """
        Gets the value fixed input corresponding to this argument

        Parameters
        ----------
        values : dict
            Dictionary of fixed inputs.
        """
        return tuple([argument.get_fixed_value(values) for argument in self])


class CompoundBoundingBox(BoundingDomain):
    """
    A model's compound bounding box

    Parameters
    ----------
    bounding_boxes : dict
        A dictionary containing all the BoundingBoxes that are possible
            keys   -> _slice (extracted from model inputs)
            values -> BoundingBox

    model : `~astropy.modeling.Model`
        The Model this bounding_box is for.

    slice_args : SliceArguments
        A description of how to extract the slices from model inputs.

    create_slice : optional
        A method which takes in the slice and the model to return a
        valid bounding corresponding to that slice. This can be used
        to construct new bounding_boxes for previously undefined slices.
        These new boxes are then stored for future lookups.

    order : optional, str
        The ordering that is assumed for the tuple representation of the
        bounding_boxes.

    Methods
    -------
    validate :
        Contructs a valid complex bounding_box
    """
    def __init__(self, bounding_boxes: Dict[Any, BoundingBox], model,
                 slice_args: SliceArguments, create_slice: Callable = None, order: str = 'C'):
        super().__init__(model)
        self._order = order

        self._create_slice = create_slice
        self._slice_args = SliceArguments.validate(model, slice_args)

        self._bounding_boxes = {}
        self._validate(bounding_boxes)

    def __repr__(self):
        parts = ['CompoundBoundingBox(',
                 '    bounding_boxes={']
        # bounding_boxes
        for _slice, bbox in self._bounding_boxes.items():
            bbox_repr = bbox.__repr__().split('\n')
            parts.append(f"        {_slice} = {bbox_repr.pop(0)}")
            for part in bbox_repr:
                parts.append(f"            {part}")
        parts.append('    }')

        # slice_args
        slice_args_repr = self._slice_args.__repr__().split('\n')
        parts.append(f"    slice_args = {slice_args_repr.pop(0)}")
        for part in slice_args_repr:
            parts.append(f"        {part}")
        parts.append(')')

        return '\n'.join(parts)

    @property
    def bounding_boxes(self) -> Dict[Any, BoundingBox]:
        return self._bounding_boxes

    @property
    def slice_args(self) -> SliceArguments:
        return self._slice_args

    @slice_args.setter
    def slice_args(self, value):
        self._slice_args = SliceArguments.validate(self._model, value)

    @property
    def create_slice(self):
        return self._create_slice

    @property
    def order(self) -> str:
        return self._order

    @staticmethod
    def _get_slice_key(key):
        if isiterable(key):
            return tuple(key)
        else:
            return (key,)

    def __setitem__(self, key, value):
        _slice = self._get_slice_key(key)
        if not self._slice_args.is_slice(_slice):
            raise ValueError(f"{_slice} is not a slice!")

        self._bounding_boxes[_slice] = BoundingBox.validate(self._model, value,
                                                            self._slice_args.ignore,
                                                            order=self._order)

    def _validate(self, bounding_boxes: dict):
        for _slice, bounding_box in bounding_boxes.items():
            self[_slice] = bounding_box

    def __eq__(self, value):
        if isinstance(value, CompoundBoundingBox):
            return (self.bounding_boxes == value.bounding_boxes) and \
                (self.slice_args == value.slice_args) and \
                (self.create_slice == value.create_slice)
        else:
            return False

    @classmethod
    def validate(cls, model, bounding_box: dict, slice_args=None, create_slice=None,
                 order: str = 'C'):
        """
        Construct a valid compound bounding box for a model.

        Parameters
        ----------
        model : `~astropy.modeling.Model`
            The model for which this will be a bounding_box
        bounding_box : dict
            Dictionary of possible bounding_box respresentations
        slice_args : optional
            Description of the slice arguments
        create_slice : optional, callable
            Method for generating new slices
        order : optional, str
            The order that a tuple representation will be assumed to be
                Default: 'C'
        """
        if isinstance(bounding_box, CompoundBoundingBox):
            if slice_args is None:
                slice_args = bounding_box.slice_args
            if create_slice is None:
                create_slice = bounding_box.create_slice
            order = bounding_box.order
            bounding_box = bounding_box.bounding_boxes

        if slice_args is None:
            warnings.warn("Slice arguments must be provided prior to "
                          "model evaluation.", RuntimeWarning)

        return cls(bounding_box, model, slice_args, create_slice, order)

    def __contains__(self, key):
        return key in self._bounding_boxes

    def _create_bounding_box(self, _slice):
        self[_slice] = self._create_slice(_slice, model=self._model)

        return self[_slice]

    def __getitem__(self, key):
        _slice = self._get_slice_key(key)
        if _slice in self:
            return self._bounding_boxes[_slice]
        elif self._create_slice is not None:
            return self._create_bounding_box(_slice)
        else:
            raise RuntimeError(f"No bounding box is defined for slice: {_slice}.")

    def _select_bounding_box(self, inputs) -> BoundingBox:
        _slice = self._slice_args.get_slice(*inputs)

        return self[_slice]

    def prepare_inputs(self, input_shape, inputs) -> Tuple[Any, Any, Any]:
        """
        Get prepare the inputs with respect to the bounding box.

        Parameters
        ----------
        input_shape : tuple
            The shape that all inputs have be reshaped/broadcasted into
        inputs : list
            List of all the model inputs

        Returns
        -------
        valid_inputs : list
            The inputs reduced to just those inputs which are all inside
            their respective bounding box intervals
        valid_index : array_like
            array of all indices inside the bounding box
        all_out: bool
            if all of the inputs are outside the bounding_box
        """

        bounding_box = self._select_bounding_box(inputs)
        return bounding_box.prepare_inputs(input_shape, inputs)
