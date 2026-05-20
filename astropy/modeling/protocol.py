from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Parameter(Protocol):
    """
    Protocol describing the public API for an astropy model parameter.

    This protocol defines the interface for parameter objects used in models.
    Parameters represent model coefficients, support units, constraints, and
    can be used in arithmetic expressions.

    Examples
    --------
    >>> p: Parameter
    >>> p.value = 3.0
    >>> p.unit
    'm'
    >>> p.fixed = True
    >>> p.bounds = (0, 10)
    >>> p2 = p.copy(default=5.0)

    Entry Importance
    ----------------
    Core:
    `name`, `value`, `fixed`, `tied`, `bounds`, `min`, `max`, `copy()`, `validate()`

    Common:
    `description`, `default`, `unit`, `quantity`, `shape`, `size`, `model`

    Optional/Specialized:
    `std`, `prior`, `posterior`, arithmetic/comparison dunder methods
    """

    name: str
    description: str
    default: float | np.ndarray | None
    value: float | np.ndarray
    unit: Any | None
    quantity: Any | None
    fixed: bool
    tied: Callable | bool
    bounds: tuple[float | None, float | None]
    min: float | None
    max: float | None
    std: float | np.ndarray | None
    prior: Any
    posterior: Any
    shape: tuple[int, ...]
    size: int
    model: Any

    def copy(
        self,
        *,
        name: str = ...,
        description: str = ...,
        default: Any = ...,
        unit: Any = ...,
        getter: Any = ...,
        setter: Any = ...,
        fixed: bool = ...,
        tied: Any = ...,
        min: Any = ...,
        max: Any = ...,
        bounds: Any = ...,
        prior: Any = ...,
        posterior: Any = ...,
    ) -> Parameter: ...

    def validate(self, value: Any) -> None: ...

    # Arithmetic and comparison dunder methods
    def __add__(self, other: Any) -> Any: ...
    def __radd__(self, other: Any) -> Any: ...
    def __sub__(self, other: Any) -> Any: ...
    def __rsub__(self, other: Any) -> Any: ...
    def __mul__(self, other: Any) -> Any: ...
    def __rmul__(self, other: Any) -> Any: ...
    def __truediv__(self, other: Any) -> Any: ...
    def __rtruediv__(self, other: Any) -> Any: ...
    def __pow__(self, other: Any) -> Any: ...
    def __rpow__(self, other: Any) -> Any: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __lt__(self, other: Any) -> bool: ...
    def __le__(self, other: Any) -> bool: ...
    def __gt__(self, other: Any) -> bool: ...
    def __ge__(self, other: Any) -> bool: ...


@runtime_checkable
class Fitter(Protocol):
    """
    Protocol describing the public API for astropy model fitters.

    Fitters implement algorithms to optimize model parameters to fit data.
    They support constraints, provide fit information, and are callable.

    Examples
    --------
    >>> fitter: Fitter
    >>> result = fitter(model, x, y)
    >>> fitter.fit_info
    {'residuals': ..., 'rank': ...}

    Entry Importance
    ----------------
    Core:
    `supported_constraints`, `__call__()`

    Common:
    `fit_info`, `objective_function()`

    Optional/Specialized:
    `_add_fitting_uncertainties()`, `supports_masked_input`
    """

    supported_constraints: list[str]
    fit_info: dict[str, Any]

    def __call__(
        self,
        model: Any,
        x: Any,
        y: Any,
        z: Any = None,
        weights: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...
    def objective_function(self, fps: Any, *args: Any, **kwargs: Any) -> Any: ...
    def _add_fitting_uncertainties(self, *args: Any, **kwargs: Any) -> Any: ...

    # Optional: for fitters supporting masked input
    supports_masked_input: bool


@runtime_checkable
class FittingWithOutlierRemoval(Protocol):
    """
    Protocol for iterative outlier-removal wrappers around fitters.

    This interface captures the public API of objects that alternate fitting
    and masking of outliers, then return both the fitted model and final mask.

    Entry Importance
    ----------------
    Core:
    `fitter`, `outlier_func`, `niter`, `__call__()`

    Common:
    `outlier_kwargs`, `fit_info`

    Optional/Specialized:
    `__str__()`, `__repr__()`
    """

    fitter: Fitter
    outlier_func: Callable
    niter: int
    outlier_kwargs: dict[str, Any]
    fit_info: dict[str, Any]

    def __call__(
        self,
        model: Any,
        x: Any,
        y: Any,
        z: Any = None,
        weights: Any = None,
        *,
        inplace: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, np.ndarray]: ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...


@runtime_checkable
class JointFitter(Protocol):
    """
    Protocol for fitters that jointly fit multiple models with shared parameters.

    Joint fitters maintain collections of models and shared-parameter metadata,
    and fit all models simultaneously.

    Entry Importance
    ----------------
    Core:
    `models`, `jointparams`, `initvals`, `model_to_fit_params()`,
    `objective_function()`, `__call__()`

    Common:
    `fitparams`, `modeldims`, `ndim`
    """

    models: list[Any]
    initvals: list[Any]
    jointparams: dict[Any, list[str]]
    fitparams: list[Any]
    modeldims: list[int]
    ndim: int

    def model_to_fit_params(self) -> list[Any]: ...
    def objective_function(self, fps: Any, *args: Any) -> Any: ...
    def __call__(self, *args: Any) -> None: ...


@runtime_checkable
class Optimization(Protocol):
    """
    Protocol describing the public API for optimization backends used by fitters.

    Optimizers wrap a numerical optimization routine and expose shared controls
    such as iteration limits and convergence tolerances.

    Entry Importance
    ----------------
    Core:
    `supported_constraints`, `__call__()`

    Common:
    `maxiter`, `eps`, `acc`, `opt_method`, `fit_info`
    """

    supported_constraints: list[str]
    fit_info: dict[str, Any]
    maxiter: int
    eps: float
    acc: float
    opt_method: Callable

    def __call__(
        self,
        objfunc: Callable,
        initval: Any,
        fargs: tuple[Any, ...],
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]: ...


@runtime_checkable
class Statistic(Protocol):
    """
    Protocol for statistic functions used during fitting.

    A statistic function receives measured values, the current model, optional
    weights, and one or more independent-variable arrays.

    Entry Importance
    ----------------
    Core:
    `__call__()`
    """

    def __call__(
        self,
        measured_vals: Any,
        updated_model: Any,
        weights: Any,
        *coords: Any,
    ) -> float: ...


@runtime_checkable
class Covariance(Protocol):
    """
    Protocol for covariance results attached to fitted models.

    Entry Importance
    ----------------
    Core:
    `cov_matrix`, `param_names`, `__getitem__()`

    Common:
    `pprint()`, `__repr__()`
    """

    cov_matrix: Any
    param_names: list[str] | tuple[str, ...]

    def pprint(self, max_lines: int, round_val: int) -> str: ...
    def __getitem__(self, params: tuple[str, str] | tuple[int, int]) -> Any: ...
    def __repr__(self) -> str: ...


@runtime_checkable
class StandardDeviations(Protocol):
    """
    Protocol for per-parameter fitting uncertainties.

    Entry Importance
    ----------------
    Core:
    `param_names`, `stds`, `__getitem__()`

    Common:
    `pprint()`, `__repr__()`
    """

    param_names: list[str] | tuple[str, ...]
    stds: list[float | None]

    def pprint(self, max_lines: int, round_val: int) -> str: ...
    def __getitem__(self, param: str | int) -> float | None: ...
    def __repr__(self) -> str: ...


@runtime_checkable
class BoundingDomain(Protocol):
    """
    Protocol for bounding-domain objects used to clip model evaluation.

    This covers shared behavior implemented by model bounding-box variants.

    Entry Importance
    ----------------
    Core:
    `model`, `fix_inputs()`, `prepare_inputs()`, `prepare_outputs()`, `evaluate()`

    Common:
    `order`, `ignored`, `ignored_inputs`
    """

    model: Any
    order: str
    ignored: list[int]
    ignored_inputs: list[str]

    def fix_inputs(self, model: Any, fixed_inputs: dict[Any, Any]) -> Any: ...
    def prepare_inputs(self, input_shape: Any, inputs: Any) -> tuple[Any, Any, Any]: ...
    def prepare_outputs(
        self,
        valid_outputs: Any,
        valid_index: Any,
        input_shape: Any,
        fill_value: Any,
    ) -> Any: ...
    def evaluate(
        self, evaluate: Callable, inputs: Any, fill_value: Any
    ) -> tuple[Any, ...]: ...


@runtime_checkable
class ModelBoundingBox(BoundingDomain, Protocol):
    """
    Protocol for per-model bounding box containers.

    Entry Importance
    ----------------
    Core:
    `intervals`, `bounding_box()`, `validate()`

    Common:
    `named_intervals`, `dimension`, `copy()`, `has_interval()`, `domain()`

    Inherited Core from `BoundingDomain`:
    `fix_inputs()`, `prepare_inputs()`, `prepare_outputs()`, `evaluate()`
    """

    intervals: dict[int, Any]
    named_intervals: dict[str, Any]
    dimension: int

    def copy(self, ignored: list[int] | None = None) -> ModelBoundingBox: ...
    def has_interval(self, key: Any) -> bool: ...
    def bounding_box(
        self, order: str | None = None
    ) -> tuple[float, float] | tuple[tuple[float, float], ...]: ...
    def domain(self, resolution: Any, order: str | None = None) -> list[np.ndarray]: ...

    @classmethod
    def validate(
        cls,
        model: Any,
        bounding_box: Any,
        ignored: list | None = None,
        order: str = "C",
        **kwargs: Any,
    ) -> ModelBoundingBox: ...


@runtime_checkable
class Model(Protocol):
    """
    Protocol describing the public API for an astropy model.

    This protocol defines the interface that all astropy modeling classes
    should implement. Models can represent either single models or "model sets"
    (multiple copies of the same model type with different parameter values).

    Models can be evaluated on input data, combined with other models using
    operators, and fitted to data using the fitting subpackage.

    Notes
    -----
    **Input/Output Attributes**: The `inputs` and `outputs` properties can be
    reassigned after model creation by directly assigning to these attributes,
    which updates the model's I/O names.

    **Top-level Utility Functions**: While not part of this protocol, the
    following top-level functions work with models:

    - `astropy.modeling.bind_bounding_box()` - Attach a bounding box to a model
    - `astropy.modeling.bind_compound_bounding_box()` - Attach compound bounding box
    - `astropy.modeling.fix_inputs()` - Fix model input values to create a reduced model
    - `astropy.modeling.custom_model()` - Decorator to create custom model classes
    - `astropy.modeling.compose_models_with_units()` - Compose models handling units

    **Common Patterns in Subclasses**:

    - Subclasses typically define `n_inputs` and `n_outputs` as class attributes
    - The `evaluate()` method signature varies per model (inputs are model-specific)
    - Custom models often define `bounding_box()` as a method to compute dynamic bounds
    - Parameter constraints (fixed, bounds, tied) are commonly set during initialization

    Examples
    --------
    Basic model usage and composition::

        from astropy.modeling.models import Gaussian1D, Polynomial1D
        from astropy.modeling import fix_inputs, bind_bounding_box

        # Create and evaluate a model
        g = Gaussian1D(amplitude=1, mean=0, stddev=1)
        y = g(x)

        # Combine models with operators
        p = Polynomial1D(degree=2)
        compound = g | p  # Composition: p(g(x))

        # Fix an input to reduce dimensionality
        g_fixed = fix_inputs(g, {0: 2.5})  # Fix x=2.5

        # Attach a bounding box
        g.bounding_box = (-5, 5)
        # or use bind_bounding_box
        from astropy.modeling import bind_bounding_box
        bind_bounding_box(g, (-5, 5))

    **Real-World Usage**:

    This protocol is implemented by custom models in:

    - **GWCS** — 16 custom models for coordinate transforms (CartesianToSpherical,
      SphericalToCartesian, WavelengthFromGratingEquation, etc.) and
      advanced selector logic (RegionsSelector)
    - **JWST** — WCS transforms and spectroscopy models
    - **stdatamodels** — 24+ coordinate and spectroscopy transform models
    - **Roman Space Telescope** — Calibration pipeline models
    - **Photutils** — Photometry analysis models
    - **DKIST** — Solar telescope data processing models

    All successfully implement this protocol and demonstrate that it covers
    diverse use cases from simple coordinate transforms to complex selector
    logic with conditional behavior.

    Entry Importance
    ----------------
    Core:
    `n_inputs`, `n_outputs`, `param_names`, `parameters`, `inputs`, `outputs`,
    `__call__()`, `evaluate()`, `copy()`, `__len__()`

    Common:
    `fittable`, `linear`, `standard_broadcasting`, `parameter_constraints`,
    `model_constraints`, `name`, `meta`, `param_sets`, `model_set_axis`,
    `fixed`, `bounds`, `tied`, `has_fixed`, `has_bounds`, `has_tied`,
    `eqcons`, `ineqcons`, `inverse`, `bounding_box`, `render()`,
    `prepare_inputs()`, `prepare_outputs()`, `deepcopy()`, `__repr__()`,
    `__str__()`, compound operators (`__add__`, `__sub__`, `__mul__`,
    `__truediv__`, `__pow__`, `__or__`, `__and__`)

    Optional/Specialized:
    `has_inverse`, `has_user_inverse`, `has_user_bounding_box`, `cov_matrix`,
    `stds`, `separable`, `sync_constraints`, `input_units_strict`,
    `input_units_allow_dimensionless`, `uses_quantity`, `input_units`,
    `output_units` (attribute), `fit_deriv`, `without_units_for_data()`,
    `with_units_from_data()`, `output_units(...)` (method), `coerce_units()`
    """

    # *** Class Attributes ***
    fittable: bool
    """
    Boolean flag indicating whether this model can be fitted to data.
    Fittable models inherit from FittableModel.
    """

    linear: bool
    """
    Boolean flag indicating whether the model is linear in its parameters.
    Used to determine which fitting algorithms can be applied.
    """

    standard_broadcasting: bool
    """
    Boolean flag indicating whether standard numpy broadcasting rules apply
    to this model's inputs/outputs.
    """

    param_names: tuple[str, ...]
    """
    Tuple of names of the model's parameters, in the order they should be
    passed to the model's constructor.
    """

    parameter_constraints: dict[str, Any]
    """
    Dictionary describing the types of constraints that can be set on
    model parameters (e.g., 'fixed', 'bounds', 'tied').
    """

    model_constraints: tuple[str, ...]
    """
    Tuple of constraint types that apply to the model as a whole
    (e.g., 'eqcons' for equality constraints, 'ineqcons' for inequality).
    """

    n_inputs: int
    """Number of input dimensions to this model."""

    n_outputs: int
    """Number of output dimensions from this model."""

    # *** Instance Attributes and Properties ***

    name: str | None
    """
    Optional user-provided name for this model instance.
    Useful for identifying individual components of compound models.
    """

    meta: dict[str, Any]
    """
    Optional dictionary of user-defined metadata attached to this model.
    How this is used and interpreted is up to the user or use case.
    """

    inputs: tuple[str, ...]
    """
    Tuple of input names for this model.
    By default: ('x',) for 1D, ('x', 'y') for 2D, etc.
    This property is writable; assigning a new tuple updates the model's input names.
    """

    outputs: tuple[str, ...]
    """
    Tuple of output names for this model.
    By default: ('y',) for 1D, ('z',) for 2D, etc.
    This property is writable; assigning a new tuple updates the model's output names.
    """

    parameters: np.ndarray
    """
    Flattened array of all parameter values in all parameter sets.
    """

    param_sets: list
    """
    List of parameter sets, one item per model in a model set.
    Each item is an array of that parameter's values.
    """

    model_set_axis: int | None
    """
    The axis index for model sets (models with multiple parameter sets).
    Indicates which axis of parameter arrays corresponds to different models.
    """

    fixed: dict[str, bool]
    """
    Dictionary mapping parameter names to their fixed constraint status.
    True means the parameter is held fixed during fitting.
    """

    bounds: dict[str, tuple[float | None, float | None]]
    """
    Dictionary mapping parameter names to their bounds as (min, max) tuples.
    None values indicate no bound in that direction.
    """

    tied: dict[str, Callable | bool]
    """
    Dictionary mapping parameter names to their tied constraint.
    Values are callables that define the linking relationship, or False.
    """

    has_fixed: bool
    """Boolean indicating if the model has any fixed constraints."""

    has_bounds: bool
    """Boolean indicating if the model has any bounds constraints."""

    has_tied: bool
    """Boolean indicating if the model has any tied constraints."""

    eqcons: list
    """List of model-level equality constraints."""

    ineqcons: list
    """List of model-level inequality constraints."""

    inverse: Model
    """
    Returns a new Model instance representing the inverse transform.
    Raises NotImplementedError if no inverse is defined.
    Can be manually assigned a custom inverse model.
    """

    has_inverse: bool
    """Boolean indicating if the model has an analytic or user-defined inverse."""

    has_user_inverse: bool
    """
    Boolean indicating if a custom inverse has been assigned by the user
    (as opposed to a built-in inverse).
    """

    bounding_box: Any  # ModelBoundingBox or tuple
    """
    Bounding box tuple defining the valid region for model evaluation.
    Format depends on n_inputs (1D: (x_low, x_high), 2D: ((y_low, y_high), (x_low, x_high)), etc.)
    """

    has_user_bounding_box: bool
    """Boolean indicating if a custom bounding box has been assigned by the user."""

    cov_matrix: Any | None
    """
    Covariance matrix of parameters, set by fitter if available.
    """

    stds: list | None
    """
    Standard deviation of parameters, derived from covariance matrix if available.
    """

    separable: bool
    """
    Boolean flag indicating whether the model is separable (can be decomposed into
    independent components for each input dimension).
    Raises NotImplementedError if not defined for the model.
    """

    sync_constraints: bool
    """
    Boolean indicating whether accessing constraints automatically checks
    constituent models' current values. Defaults to True, but should be False
    during fitting for performance.
    """

    input_units_strict: dict[str, bool]
    """
    Dictionary mapping input names to whether strict unit checking is enforced.
    If True, input values must have exactly the specified units.
    """

    input_units_allow_dimensionless: dict[str, bool]
    """
    Dictionary mapping input names to whether dimensionless inputs are allowed.
    When True, dimensionless inputs gain the specified input_units.
    """

    uses_quantity: bool
    """
    Boolean indicating whether this model uses astropy Quantity objects.
    True if created with Quantity parameters or has no parameters.
    """

    input_units: dict[str, Any] | None
    """
    Dictionary mapping input names to their expected units.
    Can be set as a class attribute by model subclasses.
    """

    output_units: dict[str, Any] | None
    """
    Dictionary mapping output names to their expected units.
    Typically computed from input units and model parameters.
    """

    fit_deriv: Callable | None
    """
    For fittable models: a function to compute the Jacobian matrix
    (derivatives with respect to parameters) for use by fitting algorithms.
    Similar interface to the model's evaluate method.
    """

    # *** Methods ***

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Evaluate the model on supplied inputs.

        Parameters
        ----------
        *args : tuple
            Input data. Can be positional arguments corresponding to inputs,
            or passed as keyword arguments using input names.
        model_set_axis : int, optional
            Axis indicating model sets (for multi-model evaluation).
        with_bounding_box : bool, optional
            If True, evaluate only within bounding box bounds (default False).
        fill_value : float, optional
            Value to use outside bounding box (default np.nan).
        equivalencies : dict, optional
            Unit equivalencies to apply to inputs.
        inputs_map : dict, optional
            Mapping of input names to data.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        output : ndarray or Quantity
            Model output evaluated at the input points.
        """
        ...

    def evaluate(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Evaluate the model (abstract method to be overridden in subclasses).

        Subclasses must implement this method to define the actual model evaluation.

        Parameters
        ----------
        *args : tuple
            Input values for each input dimension.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        output : ndarray
            Model output.
        """
        ...

    def copy(self) -> Model:
        """
        Return a copy of this model.

        Uses a deep copy so that all model attributes, including parameter values,
        are copied as well.

        Returns
        -------
        model_copy : Model
            A deep copy of this model instance.
        """
        ...

    def deepcopy(self) -> Model:
        """
        Return a deep copy of this model.

        Returns
        -------
        model_copy : Model
            A deep copy of this model instance.
        """
        ...

    def render(
        self, out: np.ndarray | None = None, coords: list | None = None
    ) -> np.ndarray:
        """
        Render the model on a grid.

        Parameters
        ----------
        out : ndarray, optional
            If provided, render into this array.
        coords : list of ndarray, optional
            Coordinate arrays for each dimension.

        Returns
        -------
        out : ndarray
            Array containing the rendered model.
        """
        ...

    def prepare_inputs(
        self,
        *inputs: Any,
        model_set_axis: int | None = None,
        equivalencies: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        """
        Prepare and validate input data for model evaluation.

        Handles unit conversion, shape validation, and other preprocessing.

        Parameters
        ----------
        *inputs : tuple
            Input values to prepare.
        model_set_axis : int, optional
            Axis for model sets.
        equivalencies : dict, optional
            Unit equivalencies to apply.
        **kwargs : dict
            Additional options.

        Returns
        -------
        inputs : tuple
            Prepared input values.
        """
        ...

    def prepare_outputs(
        self, broadcasted_shapes: Any, *outputs: Any, **kwargs: Any
    ) -> tuple[Any, ...]:
        """
        Prepare output data after model evaluation.

        Handles reshaping and unit conversion of outputs.

        Parameters
        ----------
        broadcasted_shapes : any
            Information about output shapes from broadcasting.
        *outputs : tuple
            Output values from evaluate.
        **kwargs : dict
            Additional options.

        Returns
        -------
        outputs : tuple
            Prepared output values.
        """
        ...

    def without_units_for_data(self, **kwargs: Any) -> Model:
        """
        Return a model instance with units stripped from parameters.

        This is needed for fitting with unitless data while maintaining
        unit information on parameters.

        Parameters
        ----------
        **kwargs : dict
            Input and output data as keyword arguments (e.g., x=..., y=...).

        Returns
        -------
        model : Model
            Copy of model with units stripped from parameters.
        """
        ...

    def with_units_from_data(self, **kwargs: Any) -> Model:
        """
        Return a model with units attached based on input data units.

        Parameters
        ----------
        **kwargs : dict
            Input and output data as keyword arguments (e.g., x=..., y=...).

        Returns
        -------
        model : Model
            Model with units applied to parameters based on data.
        """
        ...

    def output_units(self, **kwargs: Any) -> dict[str, Any]:
        """
        Compute expected output units based on input data units.

        Parameters
        ----------
        **kwargs : dict
            Input and output data units.

        Returns
        -------
        output_units : dict
            Mapping of output names to their units.
        """
        ...

    def coerce_units(
        self,
        input_units: dict[str, Any] | None = None,
        return_units: dict[str, Any] | None = None,
        input_units_equivalencies: dict[str, Any] | None = None,
        input_units_allow_dimensionless: bool = False,
    ) -> Model:
        """
        Coerce model to a specific unit system.

        Parameters
        ----------
        input_units : dict, optional
            Target units for inputs.
        return_units : dict, optional
            Target units for outputs.
        input_units_equivalencies : dict, optional
            Unit equivalencies for inputs.
        input_units_allow_dimensionless : bool, optional
            Allow dimensionless inputs (default False).

        Returns
        -------
        model : Model
            Coerced model instance.
        """
        ...

    def __len__(self) -> int:
        """
        Return the number of models in a model set.

        For single models, returns 1. For model sets, returns the number of
        parameter sets.

        Returns
        -------
        n_models : int
            Number of models.
        """
        ...

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the model.

        Returns
        -------
        repr : str
            Detailed representation.
        """
        ...

    def __str__(self) -> str:
        """
        Return a string representation of the model.

        Returns
        -------
        str : str
            String representation.
        """
        ...

    # *** Operators for creating compound models ***

    def __add__(self, other: Model) -> Model:
        """Create a compound model by addition."""
        ...

    def __sub__(self, other: Model) -> Model:
        """Create a compound model by subtraction."""
        ...

    def __mul__(self, other: Model) -> Model:
        """Create a compound model by multiplication."""
        ...

    def __truediv__(self, other: Model) -> Model:
        """Create a compound model by division."""
        ...

    def __pow__(self, other: Model) -> Model:
        """Create a compound model by exponentiation."""
        ...

    def __or__(self, other: Model) -> Model:
        """Create a compound model by composition (left | right means right(left(x)))."""
        ...

    def __and__(self, other: Model) -> Model:
        """Create a compound model by joining inputs."""
        ...


# Additional Notes on Parameter Definition (for custom model subclasses)
# =====================================================================
# Custom Model subclasses typically define parameters as class attributes using
# the `Parameter` class from `astropy.modeling.parameters`. Key patterns:
#
# - `Parameter(default=value, bounds=(min, max))` - Set parameter bounds
# - `Parameter(default=value, fixed=True)` - Fixed parameters
# - `Parameter(default=value, tied=tie_function)` - Tied parameters
# - `Parameter(getter=func, setter=func)` - Custom value transformation
# - `Parameter(unit=u.deg)` - Associate units with parameter
#
# For example, in a custom model:
#
#    class MyModel(Fittable1DModel):
#        amplitude = Parameter(default=1, bounds=(0, None))
#        center = Parameter(default=0, unit=u.deg)
#        width = Parameter(default=1, bounds=(0.001, None))
#
#        def evaluate(self, x, amplitude, center, width):
#            return amplitude * exp(-(x - center)**2 / width**2)
#
# Advanced Patterns Used in Real Models
# =====================================
# 1. **Model composition in evaluate()**: Complex transforms combining multiple
#    sub-operations or sub-models via & and | operators
#
# 2. **Conditional evaluation**: Models may check `with_bounding_box` parameter
#    in __call__ to conditionally apply bounds checking
#
# 3. **Unit handling**: Custom `_parameter_units_for_data_units()` method for
#    inferring parameter units from data units
#
# 4. **Separability**: Setting `_separable` property to enable decomposition
#    of multi-dimensional models into independent components
#
# 5. **Compound bounding boxes**: Advanced models use `CompoundBoundingBox` for
#    condition-dependent bounding box behavior
#
# See astropy.modeling documentation for detailed Parameter and custom model
# definition examples.

# Additional Notes on Advanced Model Patterns
# ============================================
# Beyond the core protocol, some model subclasses define custom properties
# that provide additional configuration or metadata:
#
# **Custom @property methods** (seen in GWCS and similar packages):
# - `wrap_lon_at` — Custom property for coordinate wrapping control
# - `input_units` — Property that returns expected input units
# - `output_units` — Property computed from inputs and parameters
# - `atol` — Tolerance parameter (e.g., for label mapping in selectors)
#
# These custom properties are NOT part of the Model protocol but represent
# valid extensions used for model-specific configuration. Subclasses are
# free to define such properties as needed.
#
# **Complex Model Patterns**:
# - **Selector Models** (e.g., RegionsSelector): Use internal state to map
#   inputs to specific transforms (region-based conditional evaluation)
# - **Inverse Transforms**: Custom models often implement both forward and
#   inverse transforms via the `inverse` property
# - **Parameter Arrays**: Spectroscopy models may use array parameters for
#   storing coefficient tables or calibration data
