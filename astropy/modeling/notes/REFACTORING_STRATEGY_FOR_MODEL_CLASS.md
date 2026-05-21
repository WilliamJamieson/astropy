# Refactoring Strategy for astropy.modeling.core.Model Class

**Date**: 2026-05-20
**Scope**: Strategic analysis for a non-disruptive refactoring of the Model class
**Goal**: Simplify maintenance and improve performance while preserving ecosystem compatibility

## Executive Summary

The `astropy.modeling.core.Model` class is a sophisticated base class that powers
the entire astropy.modeling ecosystem. While comprehensive in feature coverage, it
exhibits complexity stemming from metaclass logic, dynamic method injection, and
dense interdependencies.

This document distills findings from:

- Comprehensive GitHub survey of 50+ astronomy package repositories
- Protocol definition of Model's public API (50+ interface elements)
- Analysis of real-world usage patterns across institutional pipelines

The refactoring strategy aims to:

1. Remove _ModelMeta metaclass complexity
2. Improve code readability and maintainability
3. Increase performance for common operations
4. Preserve 100% external API compatibility
5. Enable future modernization

---

## Part 1: Summary of Model's Critical Features (from Protocol Analysis)

Based on the survey of ecosystem usage (documented in GITHUB_MODELING_PACKAGE_USAGE_SURVEY.md),
the Model class features can be organized by importance:

### Tier 1: Critical Features (Universal Dependency)

These features are depended on by nearly every user of astropy.modeling:

#### 1. **Evaluation Contract and Dimensionality**

- **Protocol elements**: `__call__()`, `evaluate()`, `n_inputs`, `n_outputs`, `inputs`, `outputs`
- **Why critical**: All downstream models assume a callable interface with stable
  dimensionality and named inputs/outputs.
- **Survey evidence**: Used universally across `gwcs`, `jwst`, `romancal`, `stcal`,
  `stdatamodels`, `photutils`, `specutils`, `specreduce`, and every other surveyed package.
- **Complexity issues**:
  - `__call__` signature is dynamically generated via `_ModelMeta._handle_special_methods()`
  - Special keyword arguments (`model_set_axis`, `with_bounding_box`, `fill_value`,
    `equivalencies`) are injected at class creation time
  - `__call__` delegates to `__call__()` which is complex

**Refactoring insight**: This interface must remain stable and callable. The complex
dynamic generation could be simplified with explicit class definitions or type hints.

#### 2. **Compound Model Composition**

- **Protocol elements**: `__or__` (|), `__and__` (&), `__add__`, `__sub__`, `__mul__`,
  `__truediv__`, `__pow__`
- **Why critical**: Composition is the backbone of modern WCS and transform chains.
- **Survey evidence**: Heavily used in `gwcs`, `jwst`, `romancal`, `jwreftools`,
  `romanisim` for building complex coordinate transforms.
- **Complexity issues**:
  - All arithmetic operators are dynamically injected by `_ModelMeta`
  - Each operator returns a `CompoundModel`
  - No clear separation between operator definition and compound model logic

**Refactoring insight**: These operators are performance-critical in hot loops
(transform evaluation). The injection mechanism can be replaced with explicit methods.

#### 3. **Parameter Surface and Constraints**

- **Protocol elements**: `param_names`, `parameters`, `fixed`, `bounds`, `tied`,
  `has_fixed`, `has_bounds`, `has_tied`, `parameter_constraints`
- **Why critical**: Calibration and fitting workflows depend on parameter inspection
  and constraint control.
- **Survey evidence**: Universal in fitting contexts (`photutils`, `specutils`,
  `specreduce`, `tweakwcs`, `sunpy/sunkit-spex`).
- **Complexity issues**:
  - Parameter descriptors are stored in `_parameters_` at class creation time
  - `_ModelMeta` walks the MRO to collect all parameters
  - Parameter validation and constraint logic is spread across multiple modules
  - Writable properties (`fixed`, `bounds`, `tied`) involve deep introspection

**Refactoring insight**: Parameter collection and constraint enforcement could be
simplified with dataclass-like patterns or a cleaner Parameter registry.

#### 4. **Fittability and Linearity Metadata**

- **Protocol elements**: `fittable`, `linear`, `fit_deriv`
- **Why important**: Determines fitter choice and algorithm selection.
- **Survey evidence**: Implicit in all fitter-heavy repos.

**Refactoring insight**: These are class-level booleans that seldom change. No
metaclass logic is strictly needed for these.

#### 5. **Bounding Box Management**

- **Protocol elements**: `bounding_box`, `has_user_bounding_box`
- **Why important**: Essential for safe evaluation domains in WCS models.
- **Survey evidence**: Explicit in `gwcs`, `jwst`, `romancal`, `astrocut`, ASDF converters.
- **Complexity issues**:
  - `_ModelMeta._create_bounding_box_property()` creates dynamic subclasses
  - Can be a method, property, or tuple, requiring complex dispatch logic
  - User assignment changes behavior dynamically

**Refactoring insight**: Bounding box handling is one of the most opaque parts of the
metaclass. Explicit property accessors might simplify this.

#### 6. **Inverse Transform Support**

- **Protocol elements**: `inverse`, `has_inverse`, `has_user_inverse`
- **Why important**: Bidirectional transforms are core to WCS pipelines.
- **Survey evidence**: Frequent in `gwcs`, `jwst`, `romancal`, `stcal`.
- **Complexity issues**:
  - `_ModelMeta._create_inverse_property()` dynamically wraps user-defined inverses
  - Distinction between analytic and user-defined inverses adds complexity

**Refactoring insight**: Inverse logic can be simplified with explicit property methods
rather than metaclass wrapping.

### Tier 2: High-Importance Features (Domain-Specific)

These features are used by many but not all packages, and in specific workflows:

#### 7. **Unit-Aware Model Operation**

- **Protocol elements**: `input_units`, `output_units`, `uses_quantity`, `without_units_for_data()`,
  `with_units_from_data()`, `coerce_units()`, unit-strictness attributes
- **Survey evidence**: Visible in `specutils`, `specreduce`, `jwst`, `romancal`,
  `synphot_refactor`.
- **Complexity**: Unit handling introduces conditional logic throughout evaluation paths.

#### 8. **Model Metadata and Naming**

- **Protocol elements**: `name`, `meta`
- **Survey evidence**: Commonly used for pipeline bookkeeping and component labeling.
- **Complexity**: Minimal; straightforward instance attributes.

#### 9. **Model-Set Support**

- **Protocol elements**: `param_sets`, `model_set_axis`, `__len__()`
- **Survey evidence**: Used in some calibration and batch-processing workflows.
- **Complexity**: Adds significant logic to evaluation paths and parameter handling.

### Tier 3: Specialized/Optional Features (Advanced Use Cases)

These are used in specific advanced workflows and less universally:

#### 10. **Copy/Render/Preparation Helpers**

- **Protocol elements**: `copy()`, `deepcopy()`, `render()`, `prepare_inputs()`,
  `prepare_outputs()`, `__repr__()`, `__str__()`
- **Complexity**: Supporting code for ergonomics; not algorithmically critical.

#### 11. **Fitting Uncertainty and Separability**

- **Protocol elements**: `cov_matrix`, `stds`, `separable`, `sync_constraints`
- **Survey evidence**: Important in advanced fitting diagnostics.
- **Complexity**: These are typically set/computed by fitters, not Model core logic.

#### 12. **Equality Constraints and Constraint List**

- **Protocol elements**: `eqcons`, `ineqcons`, `model_constraints`
- **Survey evidence**: Less frequently used; primarily for specialized constraint-based fitting.
- **Complexity**: Stored lists that add minimal complexity to core evaluation.

---

## Part 2: Analysis of Least-Useful Features

The following features are rarely used in surveyed packages and/or add disproportionate
complexity:

### 1. **Dynamic `__call__` Signature Generation** (HIGH REMOVAL CANDIDATE)

**What it does**: The metaclass injects a custom `__call__` method with special keyword
arguments (`model_set_axis`, `with_bounding_box`, `fill_value`, `equivalencies`, `inputs_map`)
at class creation time.

**Current implementation**: `_ModelMeta._handle_special_methods()` uses
`make_function_with_signature()` to create a new `__call__` with this signature.

**Usage in survey**:
- `model_set_axis`: Used in some batch/set contexts but less common than single-model evaluation
- `with_bounding_box`, `fill_value`: Specialized options for safe evaluation
- `equivalencies`: Unit-specific; used primarily in `specutils`/`specreduce`
- `inputs_map`: Very specialized; rarely observed in survey

**Complexity cost**:
- Adds ~100 lines to metaclass logic
- Makes stack traces and debugging harder (custom signature)
- Breaks type hints (signature is generated at runtime)
- Prevents static analysis tools from understanding Model.__call__()

**Risk of removal**: **LOW** — These keyword arguments can be exposed as explicit
instance methods or functions (e.g., `model.evaluate_with_units()` instead of
`model(..., equivalencies=...)`).

**Refactoring option**: Replace with explicit, well-named methods or use `**kwargs`
with runtime dispatch.

---

### 2. **Metaclass Operator Injection** (MEDIUM REMOVAL CANDIDATE)

**What it does**: `_ModelMeta` injects all arithmetic operators (`__add__`, `__sub__`,
etc.) at class creation time via `_model_oper()` factory.

**Current implementation**: Each operator method is generated by a lambda that calls
`_model_oper()` and is added to the class dict.

**Usage in survey**:
- `__or__` (|) and `__and__` (&): Very commonly used for transform composition
- `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__pow__`: Moderately used; primarily
  in mathematical model combinations

**Complexity cost**:
- Adds ~20 lines to metaclass
- Makes reasoning about operator semantics harder (they're not defined in class body)
- Complicates documentation and IDE support

**Risk of removal**: **MEDIUM** — These operators are performance-critical, but they
could be defined explicitly in the base class without loss of functionality.

**Refactoring option**: Move operator definitions to the base Model class as explicit methods.

---

### 3. **Dynamic Bounding Box Subclass Creation** (MEDIUM REMOVAL CANDIDATE)

**What it does**: `_ModelMeta._create_bounding_box_property()` and
`_create_bounding_box_subclass()` create custom ModelBoundingBox subclasses with
parameterized `__call__` methods if the model's bounding box is callable with arguments.

**Current implementation**: ~80 lines of metaclass code creating dynamic subclasses.

**Usage in survey**:
- Parameterized bounding boxes are used in some WCS/distortion models
- Most models use simple fixed-tuple bounding boxes
- The dynamic subclass mechanism is rarely directly observed in surveyed code

**Complexity cost**:
- Creates dynamic classes at runtime (unpickleable, hard to debug)
- Adds significant metaclass complexity
- Complicates bounding box introspection

**Risk of removal**: **LOW-MEDIUM** — Most models don't use parameterized bounding
boxes. Could be replaced with an explicit API.

**Refactoring option**: Use a dedicated BoundingBoxFactory or make bounding_box
explicitly callable without dynamic subclassing.

---

### 4. **Inverse Property Wrapping** (LOW REMOVAL CANDIDATE)

**What it does**: `_ModelMeta._create_inverse_property()` wraps user-defined `inverse`
methods into the generic Model.inverse property interface.

**Current implementation**: ~20 lines of metaclass code.

**Usage in survey**:
- Custom inverses are used in many models but are not frequently overridden by users
- The wrapping mechanism is internal and not visible to most users

**Complexity cost**:
- Adds metaclass code; relatively small
- Makes it harder to understand how user-defined inverses are integrated

**Risk of removal**: **VERY LOW** — The wrapping could be moved to a simpler registration
mechanism without breaking compatibility.

**Refactoring option**: Use a decorator pattern (`@register_inverse`) or explicit
registration method.

---

### 5. **Model-Set Axis Logic** (MEDIUM REMOVAL CANDIDATE)

**What it does**: The `model_set_axis` parameter allows different axes of parameter
arrays to correspond to different models in a model set.

**Current implementation**: Scattered throughout Model.__call__(), prepare_inputs(),
and fitting code; affects parameter iteration.

**Usage in survey**:
- Model sets are used in batch processing and some calibration workflows
- `model_set_axis` is rarely changed from its default value (0)
- The flexibility is powerful but rarely exercised

**Complexity cost**:
- Adds conditional branching throughout evaluation code
- Makes parameters and parameter iteration harder to reason about
- Increases test coverage burden

**Risk of removal**: **MEDIUM-HIGH** — Some workflows depend on model_set_axis flexibility;
removal could break compatibility. However, defaulting to axis=0 (current default) might
be sufficient for 90% of use cases.

**Refactoring option**: Simplify by removing the `model_set_axis` keyword and only supporting
axis=0 by default. Provide specialized classes (e.g., `ModelSet`) for other axes if needed.

---

### 6. **Constraint Dictionaries and Sync Flags** (LOW-MEDIUM REMOVAL CANDIDATE)

**What it does**: `fixed`, `bounds`, `tied` constraints are stored as dictionaries and
can be queried/modified. The `sync_constraints` flag allows re-checking at runtime.

**Current implementation**: Properties that return constraint dicts; `sync_constraints`
is a boolean flag affecting evaluation.

**Usage in survey**:
- Constraints are widely used during fitting
- `sync_constraints` is set during fitting to False for performance
- The constraint dict interface is straightforward but not unique

**Complexity cost**:
- Modest; mostly straightforward property methods
- `sync_constraints` adds a runtime branch in fitting-heavy code

**Risk of removal**: **MEDIUM** — Constraints are critical but could be simplified.

**Refactoring option**: Keep the API but simplify the internal representation (e.g.,
use dataclasses or a dedicated ConstraintManager).

---

### 7. **Dynamic Model Repr Generation** (LOW REMOVAL CANDIDATE)

**What it does**: `_ModelMeta._format_cls_repr()` generates a custom `__repr__` for
Model subclasses.

**Current implementation**: ~50 lines of metaclass code.

**Usage in survey**:
- User-visible; useful for debugging
- Not a core algorithmic feature

**Complexity cost**:
- Adds metaclass code; relatively self-contained
- Makes repr behavior dependent on metaclass

**Risk of removal**: **VERY LOW** — Could be moved to a standalone function or replaced
with a simpler implementation.

---

## Part 3: Cross-Ecosystem Impact Assessment

### Usage Patterns from Survey

The GitHub survey (GITHUB_MODELING_PACKAGE_USAGE_SURVEY.md) identified these usage tiers:

**Very Heavy Integration** (gwcs, jwst, romancal, stdatamodels):
- Use: All core evaluation, composition, and parameter features
- Sensitivity: HIGH to breaking changes in `__call__`, operators, parameter interface
- Sensitivity: MEDIUM to changes in bounding box or inverse mechanisms

**Significant Algorithmic Use** (photutils, specutils, specreduce, tweakwcs):
- Use: Fitting workflows, parameter constraints, model sets
- Sensitivity: HIGH to changes in fittable/linear metadata, parameter interface
- Sensitivity: LOW to metaclass internal refactoring

**Focused/Domain-Specific** (stpsf, astrocut, jwreftools, romanisim, lsst):
- Use: Specialized models; limited direct Model base class usage
- Sensitivity: LOW to internal refactoring

**Interoperability/Serialization** (asdf-astropy, roman_datamodels):
- Use: Model introspection, repr, type checking
- Sensitivity: MEDIUM to changes in model pickling or repr

### Refactoring Risk Matrix

| Feature | Refactoring Risk | User Breakage Risk | Maintainability Gain |
|---|---|---|---|
| Remove operator injection | LOW | LOW | MEDIUM |
| Simplify dynamic `__call__` | LOW | MEDIUM | HIGH |
| Remove `model_set_axis` | HIGH | HIGH | MEDIUM |
| Simplify bounding box creation | LOW | LOW | MEDIUM |
| Simplify inverse wrapping | VERY LOW | LOW | LOW |
| Modernize constraint handling | LOW | MEDIUM | MEDIUM |
| Remove metaclass entirely | VERY HIGH | VERY HIGH | VERY HIGH |

---

## Part 4: Refactoring Strategy and Implementation Plan

### Guiding Principles

1. **100% External Compatibility**: All public APIs remain identical; only internal
   implementation changes.
2. **Incremental Refactoring**: Change one subsystem at a time; extensive testing
   between each step.
3. **Performance**: No reduction in common-path performance; ideally improvement.
4. **Readability**: Make Model code easier to understand for new maintainers.
5. **Maintainability**: Reduce cognitive load and complexity of the base class.

### High-Level Strategy

**Phase 1: Extract and Simplify (0-2 months)**

1. **Move operator definitions out of metaclass**: Define `__or__`, `__and__`, etc.
   explicitly in the Model base class. This is a direct 1:1 replacement that improves
   readability with zero performance cost.

2. **Extract `_handle_special_methods()` logic**: Move `__call__` and `__init__`
   signature generation to standalone functions outside the metaclass. This reduces
   metaclass complexity without changing external behavior.

3. **Simplify bounding box property creation**: Replace dynamic subclass creation
   with explicit property methods or a factory pattern.

**Phase 2: Reduce Metaclass Scope (2-4 months)**

4. **Extract parameter collection**: Move the MRO walk and parameter collection to
   a standalone function. This reduces `_ModelMeta.__new__` and `__init__` complexity
   without changing behavior.

5. **Move inverse/bounding_box property wrapping**: Extract these into standalone
   functions or use a registration pattern.

6. **Minimize metaclass to essential logic**: At this point, `_ModelMeta` should only
   handle parameter name collection and class metadata that truly requires metaclass
   intervention.

**Phase 3: Optional Simplifications (4+ months)**

7. **Consider removing `model_set_axis` flexibility**: If survey shows minimal usage of
   non-default axis values, simplify to only support axis=0 by default.

8. **Modernize constraint handling**: Consider using dataclasses or a dedicated
   ConstraintRegistry for cleaner code.

9. **Evaluate complete metaclass removal**: If phases 1-2 are successful, assess
   whether the metaclass can be eliminated entirely using `__init_subclass__()` or
   dataclass mechanisms.

### Implementation Roadmap

#### Phase 1, Step 1: Move Operators (Week 1-2)

**What**: Define operators explicitly in Model class instead of via metaclass injection.

**Before**:
```python
class _ModelMeta(ABCMeta):
    def __new__(cls, name, bases, members, **kwds):
        # ...
        for opermethod, opercall in [
            ("__add__", _model_oper("+")),
            ("__sub__", _model_oper("-")),
            # ...
        ]:
            members[opermethod] = opercall
```

**After**:
```python
class Model:
    def __add__(self, other):
        return CompoundModel("+", self, other)

    def __sub__(self, other):
        return CompoundModel("-", self, other)
    # ...
```

**Testing**: Existing operator tests should pass unchanged; no behavioral change.

**Benefit**:
- Reduces metaclass complexity
- Improves IDE/type hint support
- Makes operator semantics explicit

---

#### Phase 1, Step 2: Extract `__call__` Signature Generation (Week 2-3)

**What**: Move the `make_function_with_signature()` call out of `_handle_special_methods()`.

**Current implementation**: _ModelMeta creates custom __call__ with special kwargs.

**Refactored**: Define __call__ explicitly with **kwargs and runtime dispatch:

```python
class Model:
    def __call__(self, *inputs, model_set_axis=None,
                 with_bounding_box=False, fill_value=np.nan,
                 equivalencies=None, inputs_map=None, **new_inputs):
        # Implementation
        ...
```

**Benefit**:
- Reduces metaclass complexity
- Makes stack traces clearer
- Enables static type checking

---

#### Phase 1, Step 3: Extract Bounding Box Property Creation (Week 3-4)

**What**: Move `_create_bounding_box_property()` and subclass creation to a standalone
module.

**Option A (Simpler)**: Always use a simple property without dynamic subclasses:

```python
class Model:
    @property
    def bounding_box(self):
        if hasattr(self, '_user_bounding_box'):
            return self._user_bounding_box
        if hasattr(self.__class__, '_default_bounding_box'):
            return self.__class__._default_bounding_box
        return None

    @bounding_box.setter
    def bounding_box(self, value):
        self._user_bounding_box = value
```

**Option B (Preserve parameterized boxes)**: Create a dedicated helper:

```python
def create_parameterized_bounding_box(model_cls, func):
    """Create a callable bounding box for the model class."""
    # Returns a wrapper object that can be called with parameters
```

**Benefit**:
- Reduces metaclass complexity
- Makes bounding box behavior clearer

---

#### Phase 2: Consolidate Metaclass (Month 2-3)

**What**: At this point, `_ModelMeta` should only handle:

1. Parameter name collection
2. Parameter constraint dict initialization
3. Class metadata (fittable, linear, n_inputs, n_outputs)

**Refactored metaclass**:

```python
class _ModelMeta(ABCMeta):
    def __new__(cls, name, bases, members, **kwds):
        # Collect parameters
        cls._parameters_ = {
            k: v for k, v in members.items() if isinstance(v, Parameter)
        }

        # Collect param names from MRO
        param_names = _collect_param_names(bases, members)

        # Create the class
        self = super().__new__(cls, name, bases, members, **kwds)

        # Set param_names
        if param_names:
            self.param_names = tuple(param_names)

        return self
```

**Benefit**:
- Metaclass is now ~30 lines instead of ~300
- Easier to understand and maintain

---

#### Phase 3: Optional Simplifications (Months 3-6)

**Potential simplification 1: Use `__init_subclass__` instead of metaclass**

Modern Python allows using `__init_subclass__()` instead of metaclasses for simple
class customization:

```python
class Model:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._parameters_ = _collect_parameters(cls)
        cls.param_names = _collect_param_names(cls)
```

This would eliminate the metaclass entirely.

**Potential simplification 2: Use dataclasses for parameter handling**

```python
@dataclass
class ModelParameterSet:
    names: tuple[str, ...]
    values: np.ndarray
    constraints: dict
```

This could simplify parameter management significantly.

---

### Testing Strategy

**Phase 1-2 Testing**:

1. Run full test suite after each change; zero new failures tolerated
2. Performance benchmarks (especially for __call__, operators, parameter access)
3. Serialization/pickling tests (ensure dynamic changes don't break pickle)
4. Downstream package tests (if possible)

**Phase 3 Testing**:

5. Benchmark against original implementation
6. Long-term stability tests (edge cases, unusual model subclasses)
7. Documentation updates

---

## Part 5: User-Facing Improvements (Beyond Refactoring)

While refactoring internal implementation, the following improvements would benefit
end users without added complexity:

### 1. **Improved Error Messages for Model Definition**

**Current state**: Errors during model class definition or initialization can be cryptic
due to metaclass complexity.

**Improvement**: Add explicit validation with clear error messages:

```python
class Model:
    def __init_subclass__(cls, **kwargs):
        # Validate that n_inputs and n_outputs are defined
        if not hasattr(cls, 'n_inputs'):
            raise ModelDefinitionError(
                f"{cls.__name__} must define 'n_inputs' as a class attribute"
            )
```

### 2. **Better Documentation of Model Subclassing**

**Current state**: Subclassing Model is powerful but underdocumented.

**Improvement**: Add comprehensive guide with examples:
- When to override `evaluate()` vs. defining `__call__()`
- How to define custom inverses
- How to handle model sets
- Unit handling patterns

### 3. **`@modeldecorator` for Custom Models**

**Current state**: Users must subclass Model; `custom_model()` decorator exists but is
less discoverable.

**Improvement**: Enhance `custom_model()` with better documentation and examples:

```python
@custom_model
def my_transform(x, y, a, b):
    """A custom model."""
    return a * x + b * y
```

### 4. **Lazy Parameter Evaluation**

**Current state**: All parameters are evaluated eagerly, even if not used.

**Improvement**: For large model sets, defer parameter evaluation until needed:

```python
m = models.Gaussian1D(amplitude=large_array, mean=..., stddev=...)
# Don't materialize parameters until actually called
result = m(x)
```

### 5. **Better Support for Conditional Logic in Models**

**Current state**: Models like RegionsSelector (gwcs) have complex conditional logic
that's hard to express cleanly.

**Improvement**: Provide a base class for conditional models:

```python
class ConditionalModel(Model):
    def evaluate_with_condition(self, condition, x, ...):
        # Handle conditional evaluation
        ...
```

### 6. **First-Class Support for Model Introspection**

**Current state**: Introspecting models requires knowledge of internal attributes
(_parameters_, etc.).

**Improvement**: Add public introspection API:

```python
# What we want end users to call
model.get_parameter_info()  # Return detailed info about each parameter
model.get_constraint_info()  # Return constraint details
model.get_signature()  # Return like inspect.signature()
```

### 7. **Performance Monitoring and Profiling Hooks**

**Current state**: Hard to profile model evaluation without low-level instrumentation.

**Improvement**: Add optional profiling:

```python
with model.profiling_enabled():
    result = model(data)  # Tracks time, allocations, etc.
    model.report_profile()
```

### 8. **Better Unit Test Integration**

**Current state**: Testing custom models requires repeating boilerplate.

**Improvement**: Provide a test mixin:

```python
class MyModelTests(ModelTestMixin):
    model_class = MyModel
    test_parameters = [...]  # Automatically generates standard tests
```

---

## Part 6: Implementation Considerations and Risks

### Technical Considerations

#### Pickling and Unpickling

**Current state**: Dynamic classes created in metaclass can have pickle issues.

**Risk**: Removing metaclass complexity might improve pickle behavior, but could break
existing pickled models.

**Mitigation**:
- Run extensive pickle/unpickle tests
- Version pickle format if necessary
- Document pickle compatibility requirements

#### Third-Party Model Subclasses

**Current state**: Many downstream packages define custom Model subclasses.

**Risk**: Changes to metaclass behavior could break custom model definitions.

**Mitigation**:
- Ensure backward compatibility layer (if metaclass is removed)
- Consider `__init_subclass__` as bridge mechanism
- Test with real downstream packages (gwcs, jwst, etc.)

#### Type Hints and IDE Support

**Benefit**: Removing metaclass enables better type hints and IDE autocompletion.

**Risk**: Temporary tooling confusion during transition.

**Mitigation**:
- Add comprehensive type stubs (`.pyi` files)
- Update IDE-specific documentation

### Scheduling and Effort

**Realistic timeline**:

- Phase 1 (operator/call extraction): 2-4 weeks
- Phase 2 (metaclass reduction): 4-8 weeks
- Phase 3 (optional): 8+ weeks

**Resource requirements**:

- 1 primary developer (experienced with Model internals)
- 1 reviewer (familiar with downstream packages)
- Test coverage: Extensive (100%+ of existing tests should pass)

### Communication Plan

**To ecosystem**:
1. Announce refactoring plan in GitHub issue / dev forum
2. Release beta versions for testing
3. Request feedback from major downstream packages
4. Document changes in release notes

**Within astropy**:
1. Regular progress updates in developer meetings
2. Design reviews before each phase
3. Test result summaries

---

## Part 7: Risk Mitigation Strategies

### Strategy 1: Feature Flags for New Implementation

During refactoring, use feature flags to allow gradual rollout:

```python
USE_NEW_IMPLEMENTATION = True

def __call__(self, *args, **kwargs):
    if USE_NEW_IMPLEMENTATION:
        return self._call_new_impl(*args, **kwargs)
    else:
        return self._call_old_impl(*args, **kwargs)
```

### Strategy 2: Compatibility Layer

Keep old metaclass logic available but unused:

```python
class _ModelMetaLegacy(ABCMeta):
    """Legacy implementation kept for compatibility."""
    pass

class Model(metaclass=_ModelMeta):
    """New simpler metaclass."""
    pass
```

### Strategy 3: Extensive Compatibility Tests

Create a test suite that validates behavior against known downstream packages:

```python
from gwcs import wcs as GWCS_WCS
from jwst import ...

def test_gwcs_compatibility():
    """Ensure our changes don't break gwcs."""
    # Create models that gwcs expects
    # Test composition operators
    # Test property access
```

---

## Part 8: Complete Metaclass Removal Analysis

While the phased approach outlined in Part 4 reduces metaclass complexity incrementally,
a complete removal of the `_ModelMeta` metaclass is theoretically possible and would
represent a major modernization. This section analyzes the technical feasibility,
risks, and approaches.

### Why Remove the Metaclass?

**Current costs of metaclass approach**:

1. **Cognitive burden**: New maintainers must understand Python metaclass mechanics
2. **Tooling friction**: Type checkers, IDEs, and documentation generators struggle
3. **Pickle complexity**: Dynamic class generation creates pickle compatibility issues
4. **Debugging difficulty**: Stack traces involve generated code and dynamic dispatch
5. **Test complexity**: Metaclass interactions require specialized testing patterns
6. **Performance**: Property access through metaclass-generated descriptors has overhead

**Benefits of removal**:

1. **Code clarity**: Model definition becomes explicit and readable
2. **Tool support**: Type hints, IDE autocomplete, documentation generators work better
3. **Pickle reliability**: Simpler class definitions pickle/unpickle consistently
4. **Easier debugging**: Stack traces show actual user code
5. **Lower barrier to contribution**: Fewer esoteric Python concepts needed
6. **Potential performance gains**: Eliminates property indirection in hot paths

### Technical Approaches to Metaclass Removal

#### Approach 1: `__init_subclass__()` with Class Decorators

**How it works**:

```python
class Model:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Collect parameters
        cls._parameters_ = _collect_parameters(cls)
        cls.param_names = _extract_param_names(cls)
        # Initialize other class attributes
        _initialize_model_metadata(cls)
```

**Advantages**:
- Pure Python 3.6+ feature (no metaclass needed)
- Clear and explicit in subclass definitions
- Works with multiple inheritance naturally
- Easy to understand and debug

**Disadvantages**:
- `__init_subclass__` is called for every subclass in the MRO chain
- Less powerful than metaclass (can't modify instance creation)
- Doesn't help with already-defined models
- Requires changes to downstream subclass definitions

**Downstream risk**: **MEDIUM** — Downstream models would still work but documentation
would need updating. Custom models defined with __init_subclass__ would work identically.

**Complexity of implementation**: **MEDIUM**

---

#### Approach 2: Dataclass-Based Model Definition

**How it works**:

```python
from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class Model:
    _parameters_: ClassVar[dict] = field(default_factory=dict)
    param_names: ClassVar[tuple] = ()

    # Instance attributes
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        # Initialize model-specific attributes
        self._initialize_parameters()
```

**Advantages**:
- Familiar pattern for Python developers
- Automatic `__init__` generation (no metaclass needed)
- Built-in attribute validation
- Better IDE support
- Cleaner parameter definition

**Disadvantages**:
- Dataclass semantics may not fit all Model behaviors
- Would require significant refactoring of existing Model subclasses
- Parameter constraints and descriptors don't map cleanly to dataclass fields
- Breaking change for all downstream code
- Performance implications of dataclass machinery

**Downstream risk**: **VERY HIGH** — All existing Model subclasses would break.

**Complexity of implementation**: **VERY HIGH**

---

#### Approach 3: Descriptor-Based Parameter Registry

**How it works**:

```python
class ParameterRegistry:
    """Collects and manages all parameters for a model class."""

    def __init__(self, cls):
        self.param_names = []
        self.parameters = {}
        self._collect_from_class(cls)

    def _collect_from_class(self, cls):
        for name, value in cls.__dict__.items():
            if isinstance(value, Parameter):
                self.param_names.append(name)
                self.parameters[name] = value

class Model:
    # Class-level registry populated by __init_subclass__
    _registry: ClassVar[ParameterRegistry] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = ParameterRegistry(cls)
        cls.param_names = tuple(cls._registry.param_names)

    @property
    def parameters(self):
        return self._registry.parameters
```

**Advantages**:
- Clean separation of parameter management
- Explicit parameter collection
- Works with `__init_subclass__`
- Can be retrofitted to existing models
- Better introspection API

**Disadvantages**:
- Introduces new complexity (registry pattern)
- Parameter access changes semantics slightly
- Requires validation that registry matches actual parameters

**Downstream risk**: **MEDIUM-LOW** — Registry approach is compatible with existing
models if implemented carefully.

**Complexity of implementation**: **MEDIUM**

---

#### Approach 4: Factory Function Instead of Class Definition

**How it works**:

```python
def create_model_class(name, parameters=None, n_inputs=1, n_outputs=1,
                       fittable=True, linear=False, **attributes):
    """Factory function to create a Model subclass without metaclass."""

    # Create the class dictionary
    class_dict = {
        'n_inputs': n_inputs,
        'n_outputs': n_outputs,
        'fittable': fittable,
        'linear': linear,
        'param_names': tuple(parameters or []),
        **attributes
    }

    # Create and return the class
    return type(name, (Model,), class_dict)

# Usage instead of class definition
GaussianModel = create_model_class(
    'Gaussian1D',
    parameters=['amplitude', 'mean', 'stddev'],
    n_inputs=1,
    n_outputs=1
)
```

**Advantages**:
- No metaclass needed
- Explicit parameter specification
- Functions are more testable than class definitions
- Can be used as base for decorator-based API

**Disadvantages**:
- Different syntax from traditional Python classes
- Less familiar to users
- Documentation and tooling expect class syntax
- Would require deprecating traditional class-based definitions

**Downstream risk**: **VERY HIGH** — Breaking change to model definition API.

**Complexity of implementation**: **HIGH**

---

#### Approach 5: Progressive Metaclass Elimination (Recommended)

**How it works**:

This combines Phases 1-3 from Part 4, with the final step being a complete transition
to `__init_subclass__`:

```python
# Phase 3a: Metaclass becomes minimal
class _ModelMeta(ABCMeta):
    """Minimal metaclass; most logic moved to __init_subclass__."""

    def __new__(cls, name, bases, namespace, **kwargs):
        # Only essential metaclass logic remains
        namespace = _process_model_namespace(namespace)
        return super().__new__(cls, name, bases, namespace, **kwargs)

# Phase 3b: Gradual migration to __init_subclass__
class Model(metaclass=_ModelMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._parameters_ = _collect_parameters(cls)
        # ... other initialization

# Phase 3c: Full transition (optional)
# After a major version, remove the metaclass:
class Model:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # All parameter collection logic here
```

**Advantages**:
- Incremental, low-risk approach
- Backward compatible at each phase
- Allows ecosystem to adapt gradually
- Time to validate with downstream packages
- Can be reversed if issues arise
- Minimal disruption per release

**Disadvantages**:
- Takes multiple releases to complete
- Temporary maintenance burden (both systems in parallel)
- Requires clear versioning and deprecation strategy

**Downstream risk**: **LOW** — Backward compatible at each step; no forced updates.

**Complexity of implementation**: **MEDIUM** (spread over multiple releases)

---

### Specific Technical Challenges of Metaclass Removal

#### Challenge 1: Parameter Collection and Inheritance

**Current metaclass role**: Walk the MRO to collect all `Parameter` descriptors from
base classes and the current class.

**Why it's complex**:
```python
# This logic must work correctly:
class BaseModel(Model):
    x = Parameter()

class DerivedModel(BaseModel):
    y = Parameter()

# The metaclass ensures param_names includes both 'x' and 'y'
```

**Solution without metaclass**:
```python
def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)

    # Collect parameters from this class and all bases
    all_params = {}
    for base in reversed(cls.__mro__[:-1]):  # Exclude object
        for name, value in base.__dict__.items():
            if isinstance(value, Parameter):
                all_params[name] = value

    cls.param_names = tuple(all_params.keys())
    cls._parameter_dict = all_params
```

**Difficulty**: **MEDIUM** — Straightforward logic but must handle edge cases:
- Circular imports (if models import each other)
- Multiple inheritance with conflicting parameters
- Parameter overrides in subclasses

---

#### Challenge 2: Dynamic Method Injection (Operators)

**Current metaclass role**: Injects `__add__`, `__sub__`, `__or__`, etc.

**Why it's complex**:
```python
# Each operator must be injected with the correct operator semantics
m1 = Model1()
m2 = Model2()
m3 = m1 | m2  # Pipes are left associative and create CompoundModels
```

**Solution without metaclass**:
```python
class Model:
    def __add__(self, other):
        return CompoundModel("+", self, other)

    def __sub__(self, other):
        return CompoundModel("-", self, other)

    def __or__(self, other):
        return CompoundModel("|", self, other)

    # ... all other operators defined explicitly
```

**Difficulty**: **LOW** — This is straightforward to implement. Already identified
as Phase 1, Step 1.

---

#### Challenge 3: Property Generation for `inverse` and `bounding_box`

**Current metaclass role**: Dynamically creates properties that can adapt to different
callable signatures.

**Why it's complex**:
```python
# Models can define inverse in different ways:

class Model1(Model):
    def inverse(self, x):  # Simple function
        return 1.0 / x

class Model2(Model):
    @property
    def inverse(self):  # Parameterized callable
        return InverseTransform(self.parameters)

# The metaclass wraps both into a unified Model.inverse property
```

**Solution without metaclass**:

**Option A** (Simpler): Explicit property in base class:
```python
class Model:
    @property
    def inverse(self):
        # Check if subclass defined _inverse method
        if hasattr(self, '_inverse'):
            return self._inverse
        if hasattr(self.__class__, '_inverse_property'):
            return self.__class__._inverse_property
        return None

    @inverse.setter
    def inverse(self, value):
        self._inverse = value
```

**Option B** (More flexible): Registration pattern:
```python
class Model:
    _inverses = {}  # ClassVar mapping class to inverse function

    @classmethod
    def register_inverse(cls, func):
        cls._inverses[cls] = func

    @property
    def inverse(self):
        return self._inverses.get(self.__class__)

# Usage:
@Model.register_inverse
class MyModel(Model):
    pass
```

**Difficulty**: **MEDIUM** — The current metaclass magic for this is ~80 lines.
Replacement approaches are simpler but require redesign of how users define inverses.

---

#### Challenge 4: `__init__` and `__call__` Signature Generation

**Current metaclass role**: Generates custom `__init__` and `__call__` with parameter-specific
signatures at class creation time.

**Why it's complex**:
```python
# The metaclass creates this automatically:
def __init__(self, amplitude, mean, stddev, **kwargs):
    self.amplitude = Parameter('amplitude', default=1.0)
    self.mean = Parameter('mean', default=0.0)
    # ...

def __call__(self, x, model_set_axis=None, with_bounding_box=False, **kwargs):
    # Complex implementation
    pass
```

**Solution without metaclass**:

**Option A** (Explicit): Define __init__ and __call__ normally:
```python
class Gaussian1D(Model):
    amplitude = Parameter(default=1.0)
    mean = Parameter(default=0.0)
    stddev = Parameter(default=1.0)

    def __init__(self, amplitude=1.0, mean=0.0, stddev=1.0, **kwargs):
        self.amplitude.value = amplitude
        self.mean.value = mean
        self.stddev.value = stddev
        super().__init__(**kwargs)

    def __call__(self, x, **kwargs):
        return self.evaluate(x, self.amplitude.value,
                            self.mean.value, self.stddev.value)
```

**Disadvantage**: Requires manually re-implementing this for every model class.
Currently generated automatically.

**Option B** (Decorator-based): Use a decorator to generate signatures:
```python
@auto_init_call
class Gaussian1D(Model):
    amplitude = Parameter(default=1.0)
    mean = Parameter(default=0.0)
    stddev = Parameter(default=1.0)

    def evaluate(self, x, amplitude, mean, stddev):
        # ...
        pass

# The @auto_init_call decorator generates __init__ and __call__
```

**Difficulty**: **HIGH** — Generating correct signatures and binding parameters
correctly is complex. Requires introspection, type checking, and validation.
Currently ~100 lines of metaclass logic.

---

#### Challenge 5: Model-Set Axis Support

**Current metaclass role**: Ensures `model_set_axis` parameter handling works correctly.

**Why it's complex**:
```python
# Model sets allow broadcasting different model instances:
m = [model1, model2, model3]  # 3 different models
data = np.array([[x1], [x2], [x3]])  # 3 different datasets

# model_set_axis determines which axis indexes different models
result = evaluate_model_set(m, data, model_set_axis=0)
```

**Solution without metaclass**: This doesn't fundamentally require a metaclass—it's
a runtime concern. Can be handled in `__call__` and `evaluate()` logic directly.

**Difficulty**: **LOW** — Model-set logic is a runtime concern, not a class-definition concern.

---

#### Challenge 6: Backward Compatibility with Existing Subclasses

**Current state**: Thousands of downstream models define subclasses of Model:

```python
# gwcs/astropy packages define:
class IdentityModel(Model):
    n_inputs = 1
    n_outputs = 1

    @staticmethod
    def evaluate(x):
        return x

# These must continue working with minimal changes
```

**Challenge**: Removing the metaclass might break these unless:
1. The new system supports the same parameter collection mechanism
2. All custom models work unchanged
3. Custom __init__ and __call__ definitions still work

**Solution strategy**:
- Use `__init_subclass__()` to collect parameters (works for existing models)
- Keep parameter collection API identical
- Provide compatibility layer (metaclass still available but unused)

**Difficulty**: **HIGH** — The more old code that must keep working, the higher the difficulty.

---

### Risk Assessment for Complete Metaclass Removal

#### Risk 1: Behavioral Changes in Parameter Handling

**What could break**:
```python
# If parameter collection logic changes, this could behave differently:
m = MyModel()
print(m.param_names)  # Must remain the same
print(m.parameters)   # Must remain the same
```

**Likelihood**: **MEDIUM** — Easy to maintain API but implementation details could leak.

**Severity**: **HIGH** — Parameter handling is fundamental.

---

#### Risk 2: Pickle/Unpickle Incompatibility

**What could break**:
```python
# Old pickled model:
import pickle
m_old = pickle.load(open('model_pickled_with_metaclass.pkl', 'rb'))
# With new system, unpickling might fail
```

**Likelihood**: **MEDIUM** — Simpler class definitions might unpickle differently.

**Severity**: **VERY HIGH** — Breaks data persistence, pipelines, caches.

**Mitigation**:
- Provide `__getstate__`/`__setstate__` methods
- Version pickle format
- Provide migration tools

---

#### Risk 3: Third-Party Metaclass Interactions

**What could break**:
```python
# If downstream packages also define metaclasses:
class CustomMeta(ABCMeta):
    pass

class MyModel(Model, metaclass=CustomMeta):
    # This might fail if Model no longer uses metaclass
    pass
```

**Likelihood**: **MEDIUM-LOW** — Some packages do this (e.g., for logging).

**Severity**: **MEDIUM** — Breaks some advanced use cases.

**Mitigation**:
- Document metaclass removal in advance
- Provide compatibility layer
- Alternative approach for custom metaclasses

---

#### Risk 4: Performance Regressions

**What could regress**:
- Parameter access in tight loops (common in WCS evaluation)
- Model instantiation (if __init__ becomes more complex)
- Operator composition (if operator dispatch changes)

**Likelihood**: **MEDIUM** — Depends on replacement implementation.

**Severity**: **MEDIUM** — Performance is important for astronomy workloads.

**Mitigation**:
- Benchmark before and after
- Profile hot paths
- Use descriptor protocol for fast parameter access

---

#### Risk 5: IDE/Type Checker Support Degradation (Short-term)

**What could break**:
- Type hints might not work during transition
- IDE autocompletion might be confused
- Documentation generators might struggle

**Likelihood**: **MEDIUM-HIGH** — Depends on tooling support.

**Severity**: **MEDIUM** — Affects developer experience.

**Mitigation**:
- Provide comprehensive `.pyi` stub files
- Update type hints during migration
- Test with mypy, pyright, pydantic

---

### Downstream Package Analysis: Direct Metaclass Feature Usage

The following analysis identifies how major downstream packages depend on specific
metaclass-provided features. This is critical for understanding the scope of potential
breakage.

#### Key Downstream Packages Analyzed

Based on the GitHub survey (GITHUB_MODELING_PACKAGE_USAGE_SURVEY.md), these packages
are the heaviest users of astropy.modeling and most likely to be affected:

**Tier 1 (Very Heavy Integration)**:
- `gwcs` (16+ custom models in WCS pipeline)
- `jwst` (50+ model usage points in calibration)
- `romancal` (30+ model usage points)
- `stdatamodels` (24+ custom models)

**Tier 2 (Significant Algorithmic Use)**:
- `photutils` (Fitting workflows, radial profiles)
- `specutils` (Line fitting, spectrum models)
- `specreduce` (Wavelength calibration)
- `tweakwcs` (Distortion correction)

**Tier 3 (Domain-Specific)**:
- `asdf-astropy` (ASDF serialization)
- `sunpy/sunkit-spex` (Solar spectral fitting)
- `stpsf` (Point spread function models)
- `jwreftools` (Reference file tools)

#### Feature-by-Feature Downstream Usage

##### Feature 1: Dynamic `param_names` Collection

**How downstream uses it**:
```python
# gwcs/modeling.py
class Mapping(Model):
    n_inputs = 2
    n_outputs = 2

    # No explicit param_names; metaclass collects from no Parameter() descriptors
    # Result: param_names = ()

# jwst pipeline code
for param_name in model.param_names:
    constraint = model.parameter_constraints[param_name]
    apply_constraint(param_name, constraint)
```

**Packages relying on this**:
- `gwcs`: Constructs custom Mapping, Identity models with dynamic param collection
- `jwst`: Queries param_names during pipeline initialization
- `romancal`: Iterates over param_names during distortion application
- `stdatamodels`: Custom models with varying parameter counts

**Breakage scenario**: If param_names collection changes:
- Models with no Parameter descriptors could have unexpected param_names
- Parameter iteration could fail silently
- Constraint application could skip parameters

**Difficulty of replacement**: **MEDIUM** — Can be moved to `__init_subclass__()` but
must maintain identical behavior for inheritance chains.

---

##### Feature 2: Automatic `__init__` Signature Generation

**How downstream uses it**:
```python
# photutils/photometry.py
gaussian = models.Gaussian1D(amplitude=1.0, mean=5.0, stddev=2.0)

# The metaclass generated __init__ with parameter-specific signature
# This allows intuitive keyword argument usage

# Downstream code relies on this signature for:
# 1. IDE autocomplete (shows amplitude, mean, stddev as kwargs)
# 2. Type hints (knows parameter types)
# 3. Programmatic introspection (inspect.signature(models.Gaussian1D))
```

**Packages relying on this**:
- `photutils`: Fits Gaussian, Moffat, etc. with keyword arguments
- `specutils`: Creates Gaussian, Lorentz profiles for line fitting
- `specreduce`: Uses keyword args for wavelength calibration models
- `tweakwcs`: Creates polynomial models with keyword args

**Breakage scenario**: If __init__ signature becomes generic:
```python
# If we change from:
gaussian = models.Gaussian1D(amplitude=1.0, mean=5.0, stddev=2.0)

# To:
gaussian = models.Gaussian1D(parameters={'amplitude': 1.0, 'mean': 5.0, 'stddev': 2.0})

# This breaks all downstream code
```

**Downstream impact**: **VERY HIGH** — Thousands of lines of code would break.

**Difficulty of replacement**: **HIGH** — Decorators or factory functions could
generate signatures, but must work for all ~40 model classes.

---

##### Feature 3: Automatic `__call__` Signature with Special Keywords

**How downstream uses it**:
```python
# jwst/assign_wcs.py
result = model(x, y, with_bounding_box=False, fill_value=np.nan)

# The metaclass injects model_set_axis, with_bounding_box, fill_value, equivalencies
# These are not part of model.evaluate() but are special __call__ kwargs

# specutils uses equivalencies:
result = model(wavelength, equivalencies=u.spectral_density(u.Angstrom))
```

**Packages relying on this**:
- `jwst`: Uses `with_bounding_box=False` for safe evaluation without domain checking
- `specutils`: Uses `equivalencies` for unit-aware fitting
- `romancal`: Uses `model_set_axis` for batch distortion evaluation
- `stcal`: Uses `fill_value` for OOB handling

**Breakage scenario**: If these special kwargs disappear:
```python
# Current code that works:
result = model(x, with_bounding_box=False)

# Would become invalid with generic __call__
```

**Downstream impact**: **HIGH** — Code using these features would need rewriting.

**Difficulty of replacement**: **MEDIUM** — Can be replaced with explicit methods:
```python
result = model.evaluate_unbounded(x)
result = model.evaluate_with_units(x, equivalencies=...)
```

---

##### Feature 4: Metaclass-Generated Operator Methods

**How downstream uses it**:
```python
# gwcs/compound_models.py
# Composes transforms:
wcs_model = spatial_model | spectral_model
combined = wcs_model & error_model

# jwst/pipeline
transform = rotation | distortion | wavelength

# The metaclass injects | (&, +, -, *, /, **) operators
# These create CompoundModel instances
```

**Packages relying on this**:
- `gwcs`: Heavily uses | and & for WCS composition (16+ models)
- `jwst`: Uses operators for pipeline stage composition (50+ usage points)
- `romancal`: Uses operators for distortion pipeline
- `stcal`: Uses operators for calibration stage combination

**Breakage scenario**: If operators are removed:
```python
# Current code:
model = m1 | m2 | m3

# Would fail entirely
```

**Downstream impact**: **VERY HIGH** — Core WCS composition would break.

**Difficulty of replacement**: **LOW** — Operators can be moved to base class as explicit
methods (already identified as Phase 1 work). No downstream change needed.

---

##### Feature 5: Dynamic Inverse Property Creation

**How downstream uses it**:
```python
# gwcs uses custom inverses for distortion models
class SomeDistortion(Model):
    def inverse(self, x):
        # Custom inverse transformation
        return compute_inverse(x)

# Downstream code accesses:
if model.has_inverse:
    inverse_result = model.inverse(x)

# The metaclass wraps the inverse method into a property
```

**Packages relying on this**:
- `gwcs`: 8+ models with custom inverses
- `jwst`: Coordinate inverse transforms
- `romancal`: Distortion inverse operations
- `stpsf`: PSF model inverses

**Breakage scenario**: If inverse property wrapping changes:
- Custom inverse methods might not be wrapped correctly
- `model.inverse` might not be callable as expected
- `has_inverse` checks might fail

**Downstream impact**: **MEDIUM** — Inverse handling would break in WCS pipelines.

**Difficulty of replacement**: **MEDIUM** — Can be handled with explicit registration or
property methods; would need documentation update.

---

##### Feature 6: Dynamic Bounding Box Property Creation

**How downstream uses it**:
```python
# gwcs models with parameterized bounding boxes
class SkyToPix(Model):
    n_inputs = 2
    n_outputs = 2

    @property
    def bounding_box(self):
        # Returns bounding box that depends on model state
        return compute_bbox_from_parameters()

# Downstream code:
if model.has_user_bounding_box:
    result = model(x, with_bounding_box=True)
```

**Packages relying on this**:
- `gwcs`: Dynamic bounding boxes for WCS models
- `jwst`: Safe evaluation with domain checking
- `romancal`: Distortion model bounding
- `astrocut`: Image cutout with WCS bounding

**Breakage scenario**: If bounding box property wrapping changes:
- Custom bounding box definitions might not work
- Property access might fail
- Safe evaluation might become unsafe

**Downstream impact**: **MEDIUM** — WCS safe evaluation would be affected.

**Difficulty of replacement**: **MEDIUM** — Explicit property methods or factory pattern
could work; needs careful testing.

---

##### Feature 7: Pickle/Unpickle Support

**How downstream uses it**:
```python
# asdf-astropy serialization
import pickle

# Models are often pickled as part of ASDF structures
pickled_model = pickle.dumps(my_model)
# Later:
restored_model = pickle.loads(pickled_model)

# The metaclass enables pickling by keeping dynamic classes accessible
```

**Packages relying on this**:
- `asdf-astropy`: ASDF model serialization
- `roman_datamodels`: Pickle-based caching
- `jwst/stcal`: Model caching in pipelines
- Research notebooks: Model persistence

**Breakage scenario**: If pickle support breaks:
```python
# Old pickled models can't be restored:
restored_model = pickle.loads(pickled_data)  # Fails with new system
```

**Downstream impact**: **VERY HIGH** — Existing serialized models become unreadable.

**Difficulty of replacement**: **HIGH** — Requires careful `__getstate__`/`__setstate__`
implementation and version compatibility.

---

##### Feature 8: Model Repr/Str Generation

**How downstream uses it**:
```python
# gwcs/plotting.py and jwst/logging
print(model)  # Shows nicely formatted model string
# Used for:
# 1. Logging and debugging
# 2. Interactive exploration
# 3. Documentation generation

# The metaclass generates custom __repr__ with parameter values
```

**Packages relying on this**:
- `gwcs`: Debugging WCS composition
- `jwst/romancal`: Pipeline logging
- Documentation tools: Sphinx autodoc
- Interactive notebooks: Jupyter output

**Breakage scenario**: If repr generation changes:
- Model output becomes unreadable
- Logging becomes less informative
- Debugging becomes harder

**Downstream impact**: **MEDIUM** — User experience degrades; code still works.

**Difficulty of replacement**: **LOW** — Can be moved to standalone function or
explicit method.

---

##### Feature 9: Parameter Constraints Enforcement

**How downstream uses it**:
```python
# photutils fitting
model = models.Gaussian1D()
model.mean.bounds = (0, 100)
model.amplitude.fixed = True
model.stddev.tied = {'other_model_stddev': some_tie_function}

# The metaclass sets up the Parameter descriptor system that enables this
# Constraints are enforced during fitting

# Downstream code:
for param_name in model.param_names:
    if model.fixed[param_name]:
        # Skip this parameter in fitting
        pass
```

**Packages relying on this**:
- `photutils`: Gaussian fitting with constraints
- `specutils`: Line profile fitting with bounds
- `specreduce`: Wavelength solution with fixed parameters
- `tweakwcs`: Alignment with tied parameters

**Breakage scenario**: If constraint mechanism changes:
- Parameter bounds might not be enforced
- Fixed parameters might still vary
- Tied parameters might not sync

**Downstream impact**: **HIGH** — Fitting behavior would change significantly.

**Difficulty of replacement**: **MEDIUM** — Constraints can be managed without
metaclass, but API must remain identical.

---

##### Feature 10: Model-Set Support with Dynamic Axis

**How downstream uses it**:
```python
# stcal batch fitting
models_array = np.array([model1, model2, model3])
data_array = np.array([[x1], [x2], [x3]])

# The metaclass enables model_set_axis parameter:
result = evaluate_model_set(models_array, data_array, model_set_axis=0)

# Downstream code:
for i, model_result in enumerate(result):
    process_result(model_result)
```

**Packages relying on this**:
- `stcal`: Batch distortion correction
- `photutils`: Multi-object fitting
- `specutils`: Multi-fiber wavelength solutions
- `specreduce`: Batch wavelength calibration

**Breakage scenario**: If model_set_axis support changes:
- Batch operations would fail
- Broadcasting semantics would change
- Calibration pipelines would break

**Downstream impact**: **MEDIUM-HIGH** — Batch processing would be affected.

**Difficulty of replacement**: **MEDIUM** — Model-set logic doesn't require
metaclass; can be moved to runtime.

---

#### Summary Table: Metaclass Feature Downstream Dependency

| Feature | Primary Users | Impact if Broken | Difficulty to Replace | Migration Effort |
|---|---|---|---|---|
| param_names collection | gwcs, jwst, romancal | MEDIUM | MEDIUM | MEDIUM |
| __init__ signature generation | photutils, specutils, specreduce | VERY HIGH | HIGH | HIGH |
| __call__ with special kwargs | jwst, specutils, romancal | HIGH | MEDIUM | MEDIUM |
| Operator injection | gwcs, jwst, romancal | VERY HIGH | LOW | NONE |
| Inverse property wrapping | gwcs, jwst, stpsf | MEDIUM | MEDIUM | MEDIUM |
| Bounding box properties | gwcs, jwst, astrocut | MEDIUM | MEDIUM | MEDIUM |
| Pickle support | asdf-astropy, roman_datamodels | VERY HIGH | HIGH | HIGH |
| Repr/str generation | All packages | MEDIUM | LOW | LOW |
| Parameter constraints | photutils, specutils, specreduce | HIGH | MEDIUM | MEDIUM |
| Model-set axis support | stcal, photutils, specutils | MEDIUM-HIGH | MEDIUM | MEDIUM |

**Key insights**:

1. **Critical features** (very high breakage risk):
   - `__init__` signature generation (used by 90%+ of downstream packages)
   - Operator injection (core WCS composition)
   - Pickle support (data persistence)

2. **High-impact features**:
   - `__call__` special kwargs (fitting workflows)
   - Parameter constraints (calibration, fitting)
   - Model-set axis support (batch processing)

3. **Replaceable features**:
   - Operator injection can move to base class with zero downstream impact
   - Repr generation can move to standalone function
   - Parameter collection can move to `__init_subclass__()`

4. **Challenging features**:
   - __init__ signature generation requires backward-compatible approach
   - Pickle support needs careful versioning
   - Parameter constraints need identical API preservation

---

#### Implications for Metaclass Removal Strategy

These findings directly inform the phased removal strategy:

**Why Phase 1 should extract operators first**: Operators are used very heavily
(gwcs/jwst/romancal all depend) but CAN be moved to base class with zero user code changes.

**Why __init__/__call__ signature generation is Phase 3**: This is the most heavily
used and most difficult to replace. Requires extensive testing and backward compatibility.

**Why pickle support is a separate concern**: ASDF serialization depends on pickle;
removal would break existing data pipelines. Needs special handling.

**Why parameter constraints cannot be simplified**: Used heavily in photutils, specutils,
and calibration pipelines. Any change to API would require ecosystem-wide updates.

**Why model-set axis is removable but risky**: Used in batch workflows (stcal) but not
universally. Could be simplified with API deprecation.

---



**Overall Difficulty: HIGH**

#### Justification:

1. **Scope of changes**: Metaclass is deeply integrated into Model class
2. **Backward compatibility burden**: Must not break thousands of downstream models
3. **Behavioral equivalence**: New system must be identical to existing system
4. **Test coverage**: Requires extensive testing of edge cases
5. **Downstream communication**: Requires coordination with ecosystem
6. **Timeline**: Would take multiple releases (6-12 months realistic)

#### Breakdown by component:

| Component | Difficulty | Risk | Benefit |
|---|---|---|---|
| Operator injection | LOW | LOW | MEDIUM |
| Parameter collection | MEDIUM | MEDIUM | MEDIUM |
| Inverse/bounding_box properties | MEDIUM | MEDIUM | MEDIUM |
| __init__/__call__ generation | HIGH | HIGH | MEDIUM |
| Model-set axis logic | LOW | LOW | LOW |
| Pickle compatibility | MEDIUM | HIGH | LOW |
| Downstream compatibility | HIGH | VERY HIGH | MEDIUM |
| **Overall** | **HIGH** | **HIGH** | **MEDIUM** |

---

### Recommended Approach: Progressive Elimination

**Given the high difficulty and risk, the recommended approach is:**

#### Phase 1 (Months 0-2): Low-Risk Extractions
- Move operators to explicit class methods (risk: LOW, benefit: MEDIUM)
- Simplify bounding box creation (risk: LOW, benefit: MEDIUM)

#### Phase 2 (Months 2-4): Prepare for Metaclass Reduction
- Extract parameter collection to standalone function
- Prepare `__init_subclass__()` implementation
- Add comprehensive parameter collection tests

#### Phase 3 (Months 4-6): Parallel Systems
- Add `__init_subclass__()` support alongside metaclass
- Keep both working in parallel for compatibility
- Test with real downstream packages

#### Phase 4 (Next Major Version): Make Metaclass Optional
- Metaclass becomes optional (use `__init_subclass__` by default)
- Backward compatibility layer for models using metaclass
- Deprecation warnings for metaclass-specific features

#### Phase 5 (Major Version +2): Complete Removal
- Remove metaclass entirely
- All models now use `__init_subclass__()`
- Update all documentation

#### Timeline: **12-18 months** (across 3-4 minor + 1 major version)

---

### When to Consider Metaclass Removal

**Remove the metaclass if/when**:

1. ✅ Phases 1-3 of the phased approach succeed with zero issues
2. ✅ Downstream packages report that alternative approach works well
3. ✅ Type checking and IDE support improve significantly
4. ✅ Performance remains stable or improves
5. ✅ Internal maintainability burden clearly decreases
6. ✅ Ecosystem has adopted __init_subclass__ pattern

**Keep the metaclass if**:

1. ❌ Significant downstream packages require it
2. ❌ Removing it would break pickle compatibility (unfixable)
3. ❌ Performance degrades noticeably
4. ❌ Behavioral equivalence is difficult to maintain
5. ❌ Type checking/IDE support doesn't improve

---

## Summary: Metaclass Removal as Strategic Long-Term Goal

Removing the metaclass is **technically feasible** but **strategically challenging**.
The incremental phased approach (Phases 1-3 of Part 4 + gradual metaclass reduction)
is safer and more practical than attempting complete removal immediately.

**Key findings**:

1. **Feasibility**: YES — Multiple technical approaches exist
2. **Risk**: HIGH — Breaking changes potential in several areas
3. **Difficulty**: HIGH — Deep integration requires careful refactoring
4. **Timeline**: 12-18 months spread across multiple releases
5. **Benefit**: MEDIUM — Maintainability and developer experience improve, but
   algorithmic functionality unchanged

**Recommendation**: Use the phased approach from Part 4. Begin metaclass removal work
only after Phases 1-2 are complete and validated with downstream packages.

---



### Short-term (Next Release)

1. Extract operators from metaclass ✓ (low risk, high benefit)
2. Move bounding box property creation ✓ (low risk, medium benefit)
3. Extract __call__ signature generation ✓ (medium risk, high benefit)

### Medium-term (2-3 Releases)

4. Extract parameter collection logic
5. Simplify metaclass to core parameter handling
6. Add user-facing improvements (better error messages, introspection API)

### Long-term (Future)

7. Evaluate __init_subclass__ replacement for metaclass
8. Consider dataclass integration
9. Further performance optimization based on profiling

### Key Success Metrics

- Zero test failures (100% backward compatibility)
- 20%+ reduction in metaclass lines of code
- 5%+ improvement in common-path performance
- Improved maintainability (as measured by code review feedback)
- Positive feedback from downstream package maintainers

---

**Next Steps**:

1. Stakeholder review of this strategy document
2. Identify starting point (likely operator extraction)
3. Create detailed design document for Phase 1, Step 1
4. Begin implementation with extensive testing
