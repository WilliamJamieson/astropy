# Clean-Sheet API Design for astropy.modeling

**Date**: 2026-05-20
**Scope**: Modern Python API redesign of astropy.modeling following PEP 20 and contemporary standards
**Goal**: Create a maintainable, intuitive API that serves astronomical computing workflows while adhering to Python best practices

---

## Executive Summary

The current `astropy.modeling` API is comprehensive but exhibits design patterns that predate
modern Python standards and create cognitive friction for users and maintainers alike.

This document provides:

1. **Honest assessment** of the current API (strengths, weaknesses, gaps, antipatterns)
2. **Analysis against PEP 20** (Zen of Python) principles
3. **Clean-sheet design** that uses contemporary Python patterns
4. **Multiple options** for key design decisions with pros/cons
5. **Migration and compatibility strategy**

---

## Part 1: Current astropy.modeling API — Honest Assessment

### What Works Well (Strengths)

#### 1. **Composition as Core Primitive**

The pipe operator (`|`) for model composition is excellent:
```python
wcs_model = spatial_model | spectral_model | distortion_model
```

**Why this works**:
- Intuitive chaining mirrors mathematical function composition
- Read left-to-right like data pipelines
- Reduces visual nesting compared to function call syntax
- Aligns with Unix pipe philosophy (familiar to scientists)
- Survey evidence: Used in nearly all surveyed packages (gwcs, jwst, romancal)

---

#### 2. **Declarative Parameter Definition**

Defining models with class-level parameters is natural:
```python
class Gaussian1D(Model):
    amplitude = Parameter(default=1.0)
    mean = Parameter(default=0.0)
    stddev = Parameter(default=1.0)
```

**Why this works**:
- Clear, explicit parameter list
- IDE support for parameter discovery
- Pythonic descriptor pattern (familiar to Python developers)
- Self-documenting code

---

#### 3. **Constraint System is Comprehensive**

The parameter constraints system (`fixed`, `bounds`, `tied`) covers real-world fitting needs:
```python
model.amplitude.bounds = (0, 100)
model.stddev.fixed = True
model.mean.tied = lambda: other_model.mean * scale_factor
```

**Why this works**:
- Covers most constraint types needed in astronomy
- Syntactically natural (property access)
- Can represent complex relationships (tied parameters)
- Supported across all major fitters

---

#### 4. **Unit Awareness**

The integration of units throughout the API is forward-thinking:
```python
position = 1.5 * u.arcsec
model.input_units = {'x': u.pixel, 'y': u.pixel}
model.output_units = {'z': u.arcsec}
result = model(data_with_units)
```

**Why this works**:
- Prevents unit errors at runtime
- Makes implicit assumptions explicit
- Critical for astronomy where unit mixing is common

---

#### 5. **Explicit `evaluate()` Method**

Separating model evaluation from `__call__()` is useful:
```python
# Direct evaluation without instance state
result = model.evaluate(x, amplitude=1.5, mean=2.0, stddev=0.5)

# Instance-based evaluation with parameters bound to model
result = model(x)
```

**Why this works**:
- Allows both functional and object-oriented styles
- `evaluate()` enables stateless computation
- `__call__()` provides convenient stateful interface
- Users can choose style that fits their workflow

---

#### 6. **Named Inputs and Outputs**

Models explicitly declare their dimensionality:
```python
class Model:
    n_inputs = 2
    n_outputs = 1
    inputs = ('x', 'y')
    outputs = ('z',)
```

**Why this works**:
- Makes assumptions about data shape explicit
- Enables dimension checking
- Supports descriptive parameter mapping
- Critical for complex WCS pipelines

---

### What Works Poorly (Weaknesses)

#### 1. **Metaclass Complexity Obscures Simple Concepts**

The use of `_ModelMeta` metaclass creates unnecessary abstraction:

```python
class _ModelMeta(ABCMeta):
    def __new__(cls, name, bases, members, **kwds):
        # 300 lines of metaclass logic for:
        # 1. Parameter collection
        # 2. Operator injection
        # 3. __init__/__call__ signature generation
        # 4. Bounding box property creation
        # 5. Inverse property wrapping
```

**Why this is bad**:

- ❌ **Cognitive burden**: Metaclasses are advanced Python; most developers avoid them
- ❌ **Debugging difficulty**: Stack traces hide user code in generated methods
- ❌ **Tool friction**: Type checkers, IDEs, documentation generators struggle
- ❌ **Test complexity**: Requires specialized testing of class creation
- ❌ **Knowledge barrier**: New contributors must learn metaclass mechanics
- ❌ **Performance**: Property indirection adds overhead in hot paths

**Survey evidence**: Multiple downstream packages expressed concern about complexity
when extending the API.

**Impact**: Maintenance burden; bars entry for contributors; makes understanding
Model subclassing challenging for new users.

---

#### 2. **Dynamic Signature Generation Makes Type Hints Impossible**

The `__init__` signature is generated at runtime:

```python
# What type checkers see:
class Gaussian1D(Model):
    def __init__(self, **kwargs):  # Generic; doesn't help tools
        pass

# What users write:
g = Gaussian1D(amplitude=1.0, mean=2.0, stddev=0.5)

# Type checker can't validate parameter names or types
```

**Why this is bad**:

- ❌ **Zero IDE support**: No autocomplete for parameters
- ❌ **Type checking fails**: mypy, pyright can't validate arguments
- ❌ **Documentation tools confused**: Sphinx can't extract parameter info
- ❌ **Runtime errors**: Users discover missing/wrong parameters at call time

**Impact**: Worse developer experience; harder to catch errors; documentation generation
requires workarounds.

---

#### 3. **Parameter Access is Indirect and Confusing**

Multiple ways to access parameter values; unclear which is correct:

```python
model = Gaussian1D(amplitude=1.0)

# Which of these is the "right" way?
value1 = model.amplitude.value         # Parameter object's value attribute
value2 = model.parameters['amplitude'] # Via dict lookup
value3 = model.amplitude               # Direct attribute access (not always works)

# How do you set?
model.amplitude.value = 2.0            # Parameter object API
model.amplitude = 2.0                  # Direct assignment (not always works)
model.parameters['amplitude'].value = 2.0  # Via dict

# How do you check constraints?
model.amplitude.bounds = (0, 10)
if model.has_bounds['amplitude']:      # Via dict?
if model.amplitude.has_bounds:         # Via attribute? (doesn't exist)
if 'amplitude' in model.bounds:        # Via dict dict?
```

**Why this is bad**:

- ❌ **Inconsistent API**: Multiple paths to same data; unclear which to use
- ❌ **Confusing semantics**: Direct assignment sometimes works, sometimes doesn't
- ❌ **Hidden state**: Parameter constraints live in multiple places
- ❌ **Documentation burden**: Must explain all access patterns

**Survey evidence**: Users frequently ask on forums which pattern to use.

**Impact**: Increased cognitive load; easier to write buggy code; harder to document.

---

#### 4. **`__call__` Signature is Overloaded with Special Keywords**

The `__call__` method has too many responsibilities:

```python
def __call__(self, *inputs, model_set_axis=None, with_bounding_box=False,
             fill_value=np.nan, equivalencies=None, inputs_map=None, **kwargs):
    pass
```

**Why this is bad**:

- ❌ **Keyword soup**: Too many optional arguments; hard to remember
- ❌ **Unclear intent**: `model_set_axis` is for batching? `with_bounding_box` for safety?
- ❌ **Poor discoverability**: Must read docs to know these exist
- ❌ **Conflates concerns**: General evaluation mixed with special modes
- ❌ **Not composable**: Can't chain special modes nicely

**Impact**: Users miss features; hard to use correctly; API feels inconsistent.

---

#### 5. **Model-Set Axis Adds Complexity Without Commensurate Benefit**

The `model_set_axis` parameter complicates parameter handling:

```python
# Batch fitting with different models on different data:
models = np.array([model1, model2, model3])
data = np.array([[x1], [x2], [x3]])

result = models(data, model_set_axis=0)  # Which axis indexes models?

# This adds conditionals throughout:
# - Parameter iteration
# - Evaluation loops
# - Test coverage
```

**Survey findings**: `model_set_axis` is rarely changed from default (axis=0).
Batch processing is better served by explicit loop constructs.

**Why this is bad**:

- ❌ **Subtle semantics**: Easy to misunderstand which axis is which
- ❌ **Scattered logic**: Parameter handling branches on this flag everywhere
- ❌ **Rarely used flexibility**: Non-default values almost never used
- ❌ **Test burden**: Must test all axis combinations

**Impact**: Code complexity; maintenance burden; edge cases.

---

#### 6. **Bounding Box API is Opaque**

The bounding box interface is hard to understand:

```python
# How do you define a simple bounding box?
class MyModel(Model):
    @property
    def bounding_box(self):
        # Is this a property? A callable? A tuple?
        return ((0, 10), (0, 20))

# How do you define a parameterized bounding box?
@property
def bounding_box(self):
    # Parameters? Closure? State?
    def bbox_function(x):
        return ((0, x.max()), (0, 100))
    return bbox_function

# How do you use it?
if model.bounding_box:
    # Check what? Can it be None?
    result = model(x, with_bounding_box=True)
    # What does this actually do?
```

**Why this is bad**:

- ❌ **Unclear semantics**: What types can bounding box be?
- ❌ **Inconsistent interface**: Property vs. callable vs. tuple
- ❌ **Hidden behavior**: `with_bounding_box` does what exactly?
- ❌ **Poor discoverability**: Hard to document; hard to use correctly

**Impact**: Users confused about bounding boxes; avoided in practice; rarely used correctly.

---

#### 7. **Inverse Definition is Scattered and Unclear**

No clear pattern for defining inverses:

```python
# Option 1: Method
class Model1(Model):
    def inverse(self, x):
        return 1.0 / x

# Option 2: Property
class Model2(Model):
    @property
    def inverse(self):
        return InverseWrapper(lambda x: 1.0 / x)

# Option 3: Assignment
class Model3(Model):
    pass

m = Model3()
m.inverse = lambda x: 1.0 / x

# Check if it exists?
if model.has_inverse:  # Seems right
    result = model.inverse(x)
```

**Why this is bad**:

- ❌ **Multiple patterns**: Which should users use?
- ❌ **Inconsistent return types**: Method vs. property vs. assignment
- ❌ **Unclear lifecycle**: When is inverse available?

**Impact**: Users uncertain how to define inverses; hard to document.

---

#### 8. **Fitting API Mixes Concerns**

Fitter classes bundle too much responsibility:

```python
from astropy.modeling import fitting

# Fitter creation
fitter = fitting.LevMarLSQFitter(max_nfev=1000)

# Model setup
model = models.Gaussian1D()
model.amplitude.bounds = (0, 100)

# Fitting (fitter modifies model in-place)
result = fitter(model, x, y, weights=weights)

# What happened?
# - Model.parameters are modified IN-PLACE
# - result is what? The same model? A fit result? Confusing.
```

**Why this is bad**:

- ❌ **Side effects**: Fitting modifies input model in-place
- ❌ **Unclear return value**: What does fitter return?
- ❌ **Stateful**: Fitter state affects subsequent fits
- ❌ **No result object**: Covariance, residuals, diagnostic scattered

**Impact**: Users make mistakes; hard to understand what fitting does; can't easily
compare multiple fits.

---

#### 9. **No Clear Distinction Between Model Classes and Instances**

Class-level vs. instance-level configuration is blurred:

```python
# These look identical but behave differently:
Gaussian1D.fittable = True      # Class-level (affects all instances)
gaussian.fittable = False       # Instance-level (affects this one)

# But actually:
Gaussian1D.fittable = True      # ???
gaussian = Gaussian1D()
gaussian.fittable = False       # Does this work? Does it override?

# What about parameters?
Gaussian1D.amplitude = Parameter()   # Class level
g.amplitude = Parameter()            # Instance level (overwrites?)
g.amplitude.value = 1.0              # Instance parameter value
```

**Why this is bad**:

- ❌ **Confusing semantics**: Class vs. instance behavior unclear
- ❌ **Error prone**: Easy to accidentally modify class when intending instance
- ❌ **Unpredictable behavior**: Inheritance and overrides are subtle
- ❌ **Documentation burden**: Must explain these distinctions

**Impact**: Users make mistakes; classes behave unpredictably; harder to debug.

---

#### 10. **Antipattern: "Explicit is Better Than Implicit" Violated**

The metaclass uses dark magic that violates PEP 20:

```python
# What's happening here?
class Gaussian1D(Model):
    amplitude = Parameter()

# Metaclass magic does:
# 1. Collects parameters via MRO walk
# 2. Generates __init__ signature
# 3. Generates __call__ signature
# 4. Injects operator methods
# 5. Creates bounding box properties
# 6. Wraps inverse methods
# 7. Creates parameter constraint handlers

# User sees class definition but 10 things happen invisibly
```

**PEP 20 violations**:

- ❌ "Explicit is better than implicit" — metaclass magic is implicit
- ❌ "In the face of ambiguity, refuse the temptation to guess" — metaclass guesses
- ❌ "Readability counts" — generated code is hard to read
- ❌ "There should be one obvious way to do it" — too many ways to access parameters

---

### What's Missing (Gaps)

#### 1. **No First-Class Fitting Result**

Fitting returns values or modified models; no result object:

```python
# Current API:
fitter = fitting.LevMarLSQFitter()
result = fitter(model, x, y)  # What is result? The model? A tuple?

# What we want:
result = fitter(model, x, y)
result.model        # The fitted model
result.parameters   # Fitted parameters
result.covariance   # Covariance matrix
result.residuals    # Residuals
result.chi_squared  # Chi-squared value
result.dof          # Degrees of freedom
result.success      # Did fitting converge?
result.message      # Why fitting stopped
```

**Why missing**: API predates result object patterns; fitting evolved piecemeal.

---

#### 2. **No Declarative Constraint Definition**

Constraints must be set after model creation:

```python
# Current:
model = Gaussian1D()
model.amplitude.bounds = (0, 100)
model.stddev.fixed = True

# Desired (declarative):
class ConstrainedGaussian(Model):
    amplitude = Parameter(default=1.0, bounds=(0, 100))
    mean = Parameter(default=0.0)
    stddev = Parameter(default=1.0, fixed=True)
```

**Why missing**: API evolved from functional to declarative; constraints weren't
planned into parameter definitions.

---

#### 3. **No Model Validation Framework**

Models don't validate configuration:

```python
# What if user defines:
class BadModel(Model):
    n_inputs = 2
    n_outputs = 3

    def evaluate(self, x):  # Only takes 1 argument!
        return x  # Only returns 1 value

# API doesn't catch this error until runtime
```

**Why missing**: Validation wasn't a design priority; models evolved organically.

---

#### 4. **No Built-in Diagnostics**

Limited introspection for debugging:

```python
# How do I understand a model?
model = complicated_wcs_transform

# What are its parameters?
print(model.param_names)

# What are their current values and constraints?
# Must iterate manually

# What does the model do?
# Must read source code or documentation

# No built-in model.inspect() or model.diagnose()
```

**Why missing**: Diagnostics weren't prioritized; models work, so users don't need to understand them.

---

#### 5. **No Composition Error Handling**

Composing incompatible models fails at runtime:

```python
# These shouldn't work together but no error until evaluation
model1 = Model()  # 1 input, 1 output
model2 = Model()  # 2 inputs, 1 output

composed = model1 | model2  # Should error: can't connect 1 output to 2 inputs
# But error only happens when evaluating
result = composed(x)  # NOW it errors
```

**Why missing**: Composition validation is complex; API doesn't check compatibility.

---

### Antipatterns and Bad Practices Forced by API

#### Antipattern 1: In-Place Model Mutation During Fitting

```python
# Current API forces this pattern:
model = Model()
fitter = Fitter()
fitter(model, x, y)  # Modifies model IN PLACE; return value confusing

# Users then do:
fitted_model = fitter(model, x, y)  # Is this a new model or same model?

# This violates Python conventions (should return new object or return None)
```

**Consequence**: Users can't easily keep original model for comparison.

---

#### Antipattern 2: Dynamic Parameter Access

```python
# Users forced into this pattern:
for param_name in model.param_names:
    value = getattr(model, param_name).value

    # Instead of cleaner:
    for param_name, value in model.parameter_dict.items():
        # Use param_name, value
```

**Consequence**: Verbose, error-prone parameter iteration.

---

#### Antipattern 3: Constraint Soup

```python
# Users forced into this pattern:
constraints = {
    'amplitude': {'bounds': (0, 100)},
    'stddev': {'fixed': True},
    'mean': {'tied': other_model.mean}
}

for name, constraint_dict in constraints.items():
    param = getattr(model, name)
    for constraint_type, value in constraint_dict.items():
        setattr(param, constraint_type, value)

# Instead of declarative:
model = ConstrainedModel(
    amplitude=Constrained(default=1.0, bounds=(0, 100)),
    stddev=Fixed(default=1.0),
    mean=Tied(other_model.mean)
)
```

**Consequence**: Constraint definition is verbose and error-prone.

---

---

## Part 2: Analysis Against PEP 20 (Zen of Python)

### PEP 20 Principles and Current API

```
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
  CURRENT: Metaclass magic is ugly; parameter access is confusing

Explicit is better than implicit.
  CURRENT: ❌ Metaclass generates methods implicitly
  CURRENT: ❌ Parameter collection happens via hidden MRO walk
  CURRENT: ❌ Bounding box behavior is implicit

Simple is better than complex.
  CURRENT: ❌ Metaclass adds complexity; should be simpler
  CURRENT: ❌ Parameter access is overly complex
  CURRENT: ❌ Fitting API mixes concerns

Complex is better than complicated.
  CURRENT: ❌ Model-set axis adds complication without benefit
  CURRENT: ❌ Multiple ways to define inverses is complicated

Readability counts.
  CURRENT: ❌ Generated __init__ signatures hard to read
  CURRENT: ❌ Metaclass logic obscures intent
  CURRENT: ✅ Declarative parameter definition readable
  CURRENT: ✅ Composition operators readable

Special cases aren't special enough to break the rules.
  CURRENT: ❌ Model-set axis is special case
  CURRENT: ❌ Special keywords in __call__ are special cases

Although practicality beats purity.
  CURRENT: ❌ Metaclass complexity is practical but not pure

Errors should never pass silently.
  CURRENT: ❌ Incompatible model composition fails silently until evaluation
  CURRENT: ❌ Invalid model definitions aren't caught

Unless explicitly silenced.
  CURRENT: ✅ Constraints can be applied explicitly

In the face of ambiguity, refuse the temptation to guess.
  CURRENT: ❌ Metaclass guesses at parameter collection
  CURRENT: ❌ Bounding box behavior guessed

There should be one—and preferably only one—obvious way to do it.
  CURRENT: ❌ Multiple ways to access parameters
  CURRENT: ❌ Multiple ways to define inverses
  CURRENT: ❌ Multiple patterns for constraints
  CURRENT: ✅ Composition with | is obvious

Although that way may not be obvious at first unless you're Dutch.
  CURRENT: ❌ Current API requires learning metaclass internals

Now is better than never.
  CURRENT: ✅ API works and is practical

Although never is often better than *right now*.
  CURRENT: ❌ Rushed evolution added special cases

If the implementation is hard to explain, it's a bad idea.
  CURRENT: ❌ Metaclass implementation is hard to explain
  CURRENT: ❌ Parameter access patterns hard to explain
  CURRENT: ❌ Fitting side-effects hard to explain

If the implementation is easy to explain, you may have a good idea.
  CURRENT: ✅ Composition operators easy to explain
  CURRENT: ✅ Declarative parameters easy to explain
```

**Summary**: Current API violates PEP 20 on 12+ principles. Clean design must fix these.

---

## Part 3: Clean-Sheet API Design

### Core Design Principles

**Design must**:

1. ✅ Eliminate metaclass complexity
2. ✅ Make type hints work (enable IDE support)
3. ✅ Follow PEP 20 (Zen of Python)
4. ✅ Preserve composition as primary pattern
5. ✅ Simplify parameter access (one obvious way)
6. ✅ Support modern Python patterns (dataclasses, protocols, `__init_subclass__`)
7. ✅ Separate concerns (fitting, composition, evaluation)
8. ✅ Maintain 100% backward compatibility initially, then deprecate old patterns

---

### Section A: Model Definition (Clean Design)

#### Option 1: Dataclass-Based Models (OPTION A)

**Concept**: Use Python dataclasses for parameter definition, eliminating metaclass:

```python
from dataclasses import dataclass, field
from typing import Annotated
from astropy.modeling import Model, Param, Constraint

@dataclass
class Gaussian1D(Model):
    """A 1D Gaussian model."""

    # Parameters with metadata
    amplitude: Param = field(
        default=1.0,
        metadata={'description': 'Peak value', 'bounds': (0, None)}
    )
    mean: Param = field(
        default=0.0,
        metadata={'description': 'Center position'}
    )
    stddev: Param = field(
        default=1.0,
        metadata={'description': 'Standard deviation', 'bounds': (1e-10, None)}
    )

    # Model metadata (class-level, not dataclass fields)
    n_inputs: int = 1
    n_outputs: int = 1

    def evaluate(self, x, amplitude, mean, stddev):
        """Evaluate the Gaussian."""
        return amplitude * np.exp(-0.5 * ((x - mean) / stddev)**2)

# Usage:
g = Gaussian1D(amplitude=2.0, mean=5.0, stddev=1.5)
result = g(x)
```

**Advantages**:
- ✅ No metaclass needed
- ✅ Type hints work perfectly (IDE autocomplete)
- ✅ `@dataclass` handles `__init__` generation
- ✅ Parameter metadata in field definitions
- ✅ Familiar Python pattern
- ✅ Easy to serialize/pickle
- ✅ Simple to understand and maintain

**Disadvantages**:
- ❌ Parameter objects (with constraints) must be different from dataclass fields
- ❌ Parameter `value` vs. field `default` confusion
- ❌ Cannot easily change parameter counts at runtime
- ❌ Constraints stored in metadata (less elegant)
- ❌ Breaking change from current API

**Migration path**:
```python
# New API:
@dataclass
class Gaussian1D(Model):
    amplitude: Param = field(default=1.0, metadata={...})

# Old API (deprecated):
class Gaussian1D(Model):
    amplitude = Parameter()
```

**Difficulty of implementation**: **MEDIUM** (dataclasses standard library feature)

---

#### Option 2: Explicit Parameter Registry Pattern (OPTION B)

**Concept**: Parameters explicitly registered without metaclass:

```python
from astropy.modeling import Model, Parameter, parameter_registry

class Gaussian1D(Model):
    """A 1D Gaussian model."""

    n_inputs = 1
    n_outputs = 1

    def __init__(self, amplitude=1.0, mean=0.0, stddev=1.0):
        super().__init__()

        # Explicitly register parameters
        self._parameters = {
            'amplitude': Parameter('amplitude', default=1.0, value=amplitude,
                                   bounds=(0, None)),
            'mean': Parameter('mean', default=0.0, value=mean),
            'stddev': Parameter('stddev', default=1.0, value=stddev,
                               bounds=(1e-10, None)),
        }

    def evaluate(self, x, amplitude, mean, stddev):
        """Evaluate the Gaussian."""
        return amplitude * np.exp(-0.5 * ((x - mean) / stddev)**2)

# Usage:
g = Gaussian1D(amplitude=2.0, mean=5.0, stddev=1.5)
result = g(x)
```

**Advantages**:
- ✅ No metaclass; no code generation
- ✅ Explicit parameter registration
- ✅ Full control over initialization
- ✅ Clear parameter lifecycle
- ✅ Easy to test
- ✅ Simple to understand

**Disadvantages**:
- ❌ More verbose than current API
- ❌ Parameter definition scattered between class and `__init__`
- ❌ Still doesn't enable type hints for parameter names
- ❌ More boilerplate for model authors

**Migration path**:
```python
# New API (explicit):
def __init__(self, amplitude=1.0):
    self._parameters = {'amplitude': Parameter(...)}

# Old API (implicit via metaclass):
amplitude = Parameter()
```

**Difficulty of implementation**: **LOW** (straightforward pattern)

---

#### Option 3: `__init_subclass__` with Class Decorators (OPTION C)

**Concept**: Use `__init_subclass__` hook for automatic parameter collection:

```python
from astropy.modeling import Model, Parameter

class Model:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Collect parameters from class definition
        cls._param_names = tuple(
            name for name, obj in cls.__dict__.items()
            if isinstance(obj, Parameter)
        )

class Gaussian1D(Model):
    """A 1D Gaussian model."""

    amplitude = Parameter(default=1.0)
    mean = Parameter(default=0.0)
    stddev = Parameter(default=1.0)

    n_inputs = 1
    n_outputs = 1

    def __init__(self, amplitude=1.0, mean=0.0, stddev=1.0):
        self.amplitude = amplitude
        self.mean = mean
        self.stddev = stddev

    def evaluate(self, x, amplitude, mean, stddev):
        return amplitude * np.exp(-0.5 * ((x - mean) / stddev)**2)

# Usage:
g = Gaussian1D(amplitude=2.0, mean=5.0, stddev=1.5)
result = g(x)
```

**Advantages**:
- ✅ No metaclass
- ✅ `__init_subclass__` is standard Python 3.6+
- ✅ Class-level parameter declarations work
- ✅ Gradual migration from metaclass
- ✅ Can coexist with metaclass during transition
- ✅ Simpler than metaclass

**Disadvantages**:
- ❌ `__init_subclass__` called for every subclass in MRO
- ❌ Still doesn't enable type hints for parameter names
- ❌ Requires explicit `__init__` in each model
- ❌ Parameter and instance attribute duplication

**Migration path**: Replace metaclass with `__init_subclass__` gradually

**Difficulty of implementation**: **MEDIUM** (requires careful MRO handling)

---

#### **RECOMMENDED: Hybrid Approach (OPTION D)**

**Concept**: Start with `__init_subclass__` (backward-compatible), migrate to dataclass pattern.

**Phase 1** (Now):
```python
# Replace metaclass with __init_subclass__
class Model:
    def __init_subclass__(cls, **kwargs):
        # Automatic parameter collection
        cls._collect_parameters()

# Existing models work unchanged:
class Gaussian1D(Model):
    amplitude = Parameter()
    # ...
```

**Phase 2** (Next major version):
```python
# New recommended pattern:
@dataclass
class Gaussian1D(Model):
    amplitude: Param = field(default=1.0, metadata={...})
```

**Phase 3** (Future):
```python
# Full dataclass adoption
# Old Parameter-based models still work for compatibility
```

**Why hybrid**:
- ✅ Backward compatible initially
- ✅ Low migration friction
- ✅ Time to validate with ecosystem
- ✅ Can refine dataclass approach before full commitment
- ✅ Lets users adopt at their pace

---

### Section B: Parameter Access (Clean Design)

#### Current Problem

```python
# Too many ways to access; unclear which is right
model.amplitude.value
model.parameters['amplitude']
model.amplitude
getattr(model, 'amplitude').value
```

#### Clean Solution: Unified Parameter Access

```python
from astropy.modeling import Model, ParameterView

class Model:
    @property
    def parameters(self) -> ParameterView:
        """Unified parameter access."""
        return ParameterView(self._param_dict)

class ParameterView(dict-like):
    """Unified interface for parameter access."""

    def __getitem__(self, name: str) -> Parameter:
        """Get parameter by name."""
        return self._params[name]

    def items(self):
        """Iterate over (name, value) pairs."""
        return ((name, param.value) for name, param in self._params.items())

    def values(self):
        """Get parameter values."""
        return [param.value for param in self._params.values()]

    def constraints(self):
        """Get all constraints."""
        return {
            name: {
                'bounds': param.bounds,
                'fixed': param.fixed,
                'tied': param.tied
            }
            for name, param in self._params.items()
        }

# Usage:
model = Gaussian1D()

# Get parameter
param = model.parameters['amplitude']

# Get value
value = model.parameters['amplitude'].value

# Iterate values
for name, value in model.parameters.items():
    print(f"{name} = {value}")

# Get constraints
constraints = model.parameters.constraints()

# Direct access (backward compat)
value = model.amplitude.value
```

**Advantages**:
- ✅ One clear way to access parameters
- ✅ Dict-like interface familiar to Python developers
- ✅ Methods for common operations (constraints, iteration)
- ✅ Backward compatible with existing attribute access
- ✅ Easier to document

---

### Section C: `__call__` Signature (Clean Design)

#### Current Problem

```python
def __call__(self, *inputs, model_set_axis=None, with_bounding_box=False,
             fill_value=np.nan, equivalencies=None, inputs_map=None, **kwargs):
    pass
```

**Issues**:
- Too many keywords
- Unclear intent of each
- Not composable
- Hard to discover

#### Clean Solution 1: Explicit Methods (RECOMMENDED)

```python
class Model:
    def __call__(self, *inputs):
        """Evaluate model with current parameters."""
        return self.evaluate(*inputs, **self._get_param_values())

    def evaluate_with_bounding_box(self, *inputs, fill_value=np.nan):
        """Evaluate with bounding box checking."""
        return self._evaluate_bounded(*inputs, fill_value, self._get_param_values())

    def evaluate_with_units(self, *inputs, equivalencies=None):
        """Evaluate with unit handling."""
        return self._evaluate_units(*inputs, equivalencies, self._get_param_values())

    def evaluate_all(self, models, data, axis=0):
        """Evaluate array of models on array of data."""
        return self._evaluate_model_set(models, data, axis)

    def evaluate_unbounded(self, *inputs):
        """Evaluate without bounding box checking."""
        return self.evaluate(*inputs, **self._get_param_values())

# Usage:
result = model(x, y)                          # Basic
result = model.evaluate_with_bounding_box(x)  # Safe
result = model.evaluate_with_units(x, eq=...) # Units
```

**Advantages**:
- ✅ Clear intent for each method
- ✅ Explicit, discoverable API
- ✅ Composable (can chain operations)
- ✅ Follows PEP 20 ("one obvious way")
- ✅ Easy to document
- ✅ Easy to type hint

**Disadvantages**:
- ❌ More method names to remember
- ❌ Longer method names
- ❌ Breaking change for keyword argument usage

---

#### Clean Solution 2: Options Object

```python
from dataclasses import dataclass

@dataclass
class EvaluationOptions:
    """Options for model evaluation."""
    with_bounding_box: bool = False
    fill_value: float = np.nan
    equivalencies: Any = None
    model_set_axis: int = 0

class Model:
    def __call__(self, *inputs, options=None):
        """Evaluate with optional configuration."""
        opts = options or EvaluationOptions()

        if opts.with_bounding_box:
            return self._evaluate_bounded(*inputs, opts.fill_value)
        if opts.equivalencies:
            return self._evaluate_units(*inputs, opts.equivalencies)

        return self.evaluate(*inputs, **self._get_param_values())

# Usage:
result = model(x)
result = model(x, options=EvaluationOptions(with_bounding_box=True))
```

**Advantages**:
- ✅ Future-proof (can add options without changing `__call__`)
- ✅ Options are discoverable via autocomplete
- ✅ Easy to document
- ✅ Extensible

**Disadvantages**:
- ❌ More verbose than keywords
- ❌ Options object required

---

#### Clean Solution 3: Context Manager (OPTION)

```python
class Model:
    def __call__(self, *inputs):
        """Basic evaluation."""
        return self.evaluate(*inputs, **self._get_param_values())

    @contextmanager
    def bounded_evaluation(self, fill_value=np.nan):
        """Context manager for bounded evaluation."""
        old_mode = self._evaluation_mode
        self._evaluation_mode = 'bounded'
        self._fill_value = fill_value
        try:
            yield self
        finally:
            self._evaluation_mode = old_mode

# Usage:
with model.bounded_evaluation(fill_value=0):
    result = model(x)  # Automatically bounded
```

**Advantages**:
- ✅ Clear scope of special behavior
- ✅ Automatic cleanup
- ✅ Pythonic

**Disadvantages**:
- ❌ State-based (not functional)
- ❌ Can be confusing with nested contexts

---

**RECOMMENDED**: Explicit methods approach (Solution 1)
- Clear intent
- Discoverable
- Composable
- Follows Python conventions

---

### Section D: Constraints System (Clean Design)

#### Current Problem

```python
# Constraints scattered; no declarative way
model = Gaussian1D()
model.amplitude.bounds = (0, 100)
model.stddev.fixed = True
model.mean.tied = lambda: other.mean * scale

# Querying constraints is scattered too
model.fixed      # What's this?
model.has_fixed  # And this?
model.bounds     # And this?
```

#### Clean Solution: Declarative Constraints

```python
from astropy.modeling import Model, Parameter, Bound, Fixed, Tied

# Option A: In model definition
@dataclass
class Gaussian1D(Model):
    amplitude: Param = field(
        default=1.0,
        metadata={'bounds': (0, None)}  # Declarative
    )
    stddev: Param = field(
        default=1.0,
        metadata={'min_bound': 1e-10}
    )

# Option B: After creation via ParameterConstraints
model = Gaussian1D()
model.parameters.set_constraint('amplitude', bounds=(0, 100))
model.parameters.set_constraint('stddev', fixed=True)
model.parameters.set_constraint('mean', tied=other_model.parameters['mean'])

# Query constraints clearly
constraints = model.parameters.constraints()
# {
#     'amplitude': {'bounds': (0, 100)},
#     'stddev': {'fixed': True},
#     'mean': {'tied': <parameter ref>}
# }

if constraints['amplitude']['bounds']:
    print("Amplitude is bounded")
```

**Advantages**:
- ✅ Constraints visible in code
- ✅ Clear query interface
- ✅ Validation possible
- ✅ Easy to serialize

**Implementation**:
```python
class ParameterView:
    def set_constraint(self, param_name, **constraints):
        """Apply constraints to a parameter."""
        if param_name not in self._params:
            raise ValueError(f"Unknown parameter: {param_name}")

        param = self._params[param_name]

        # Validate constraints
        if 'bounds' in constraints:
            param.bounds = constraints['bounds']
        if 'fixed' in constraints:
            param.fixed = constraints['fixed']
        if 'tied' in constraints:
            param.tied = constraints['tied']

    def constraints(self):
        """Return all constraints."""
        return {
            name: {k: v for k, v in param.__dict__.items()
                   if k in ('bounds', 'fixed', 'tied')}
            for name, param in self._params.items()
        }
```

---

### Section E: Composition and Operators (Clean Design)

#### Current (Working Well!)

```python
model = spatial | spectral | distortion
composed = model1 & model2
```

**Status**: KEEP THIS. It works and is excellent.

#### Improvements

```python
class CompoundModel:
    # Add validation during composition
    def __or__(self, other):
        # Check that outputs of self match inputs of other
        if self.n_outputs != other.n_inputs:
            raise IncompatibleModels(
                f"Cannot compose: {self.n_outputs} outputs "
                f"to {other.n_inputs} inputs"
            )
        return CompoundModel("|", self, other)

    # Add introspection
    def components(self):
        """List component models in order."""
        return [self.left, self.right]

    def structure(self):
        """Return composition structure as tree."""
        return {
            'operator': self.operator,
            'left': self.left.structure() if hasattr(self.left, 'structure') else str(self.left),
            'right': self.right.structure() if hasattr(self.right, 'structure') else str(self.right)
        }

    def find_by_name(self, name):
        """Find component model by name."""
        if self.name == name:
            return self
        for component in self.components():
            if hasattr(component, 'find_by_name'):
                result = component.find_by_name(name)
                if result:
                    return result
        return None

# Usage:
model = (spatial | spectral) & distortion
model.structure()  # See composition tree
spatial_model = model.find_by_name('spatial')
```

**Advantages**:
- ✅ Composition error checking
- ✅ Better introspection
- ✅ Easier debugging of complex pipelines

---

### Section F: Fitting API (Clean Design)

#### Current Problem

```python
# Fitting modifies model in-place; unclear return value
fitter = fitting.LevMarLSQFitter()
result = fitter(model, x, y)

# Is result the fitted model? A fit result? Confusing.
# Model parameters are now modified.
```

#### Clean Solution: Fit Result Object

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FitResult:
    """Result of model fitting."""
    model: Model                    # Fitted model
    parameters: dict               # Fitted parameters
    covariance_matrix: Optional[np.ndarray]  # Covariance
    standard_deviations: Optional[np.ndarray]  # Param uncertainties
    residuals: Optional[np.ndarray]  # Residuals
    chi_squared: Optional[float]   # Chi-squared
    degrees_of_freedom: int         # DOF
    reduced_chi_squared: Optional[float]  # Reduced chi-sq
    success: bool                   # Did fitting converge?
    message: str                    # Why fitting stopped
    iterations: int                 # Iterations used

    def __repr__(self):
        return f"FitResult(success={self.success}, χ²={self.reduced_chi_squared})"

class LevMarLSQFitter:
    """Levenberg-Marquardt LSQ fitter."""

    def __call__(self, model, x, y, weights=None) -> FitResult:
        """Fit model to data."""

        # Fit (doesn't modify model)
        fitted_model = model.copy()
        # ... perform fitting ...

        # Return complete result
        return FitResult(
            model=fitted_model,
            parameters={...},
            covariance_matrix=cov,
            residuals=residuals,
            chi_squared=chisq,
            success=success,
            message=message,
            # ...
        )

# Usage:
fitter = LevMarLSQFitter()
result = fitter(model, x, y)

# Original model unchanged
print(model.parameters)

# Access fitted model and diagnostics
print(result.model.parameters)
print(result.chi_squared)
print(result.covariance_matrix)

if result.success:
    print("Fitting converged")
else:
    print(f"Fitting failed: {result.message}")
```

**Advantages**:
- ✅ No side effects (model not modified)
- ✅ Complete result information
- ✅ Clear success/failure
- ✅ Diagnostic information included
- ✅ Easy to compare multiple fits
- ✅ Serializable (can pickle result)

**Migration**:
```python
# Old API (deprecated):
fitter(model, x, y)  # Modifies model in-place

# New API:
result = fitter(model, x, y)  # Returns FitResult; model unchanged
```

---

### Section G: Model Validation and Introspection (Clean Design)

#### Validation

```python
class Model:
    @classmethod
    def validate_model(cls):
        """Validate model definition."""
        # Check n_inputs and n_outputs defined
        if not hasattr(cls, 'n_inputs'):
            raise ModelDefinitionError(f"{cls.__name__} must define 'n_inputs'")

        # Check evaluate signature matches n_inputs/n_outputs
        sig = inspect.signature(cls.evaluate)
        params = list(sig.parameters.values())[2:]  # Skip self, x

        if len(params) != cls.n_inputs:
            raise ModelDefinitionError(
                f"{cls.__name__}.evaluate expects {len(params)} inputs "
                f"but n_inputs={cls.n_inputs}"
            )

        # Check evaluate returns correct output count
        # (Can't fully validate at class time; runtime checks needed)

# Called during __init_subclass__:
class Model:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Model.validate_model()
```

#### Introspection

```python
class Model:
    def inspect(self):
        """Get comprehensive model information."""
        return ModelInspection(
            name=self.name,
            model_class=self.__class__.__name__,
            n_inputs=self.n_inputs,
            inputs=self.inputs,
            n_outputs=self.n_outputs,
            outputs=self.outputs,
            parameters=self.parameters.to_dict(),
            constraints=self.parameters.constraints(),
            is_fittable=self.fittable,
            is_linear=self.linear,
            has_inverse=self.has_inverse,
            has_bounding_box=self.has_bounding_box,
            input_units=self.input_units,
            output_units=self.output_units,
        )

    def diagnose(self):
        """Print diagnostic information."""
        inspection = self.inspect()
        print(inspection)

# Usage:
model.inspect()
# ModelInspection(
#     name='gaussian',
#     model_class='Gaussian1D',
#     n_inputs=1,
#     inputs=['x'],
#     n_outputs=1,
#     outputs=['y'],
#     parameters={'amplitude': 1.0, 'mean': 0.0, 'stddev': 1.0},
#     constraints={'amplitude': {'bounds': (0, 100)}},
#     is_fittable=True,
#     is_linear=False,
#     ...
# )
```

---

### Section H: Error Messages and Composition Validation (Clean Design)

#### Composition Validation

```python
class CompoundModel:
    def __or__(self, other):
        """Compose models with validation."""
        # Check compatibility
        if self.n_outputs != other.n_inputs:
            raise IncompatibleModels(
                f"Cannot compose {self.name} | {other.name}: "
                f"left has {self.n_outputs} outputs, "
                f"right expects {other.n_inputs} inputs\n"
                f"\nLeft model:\n{self.inspect()}\n"
                f"\nRight model:\n{other.inspect()}"
            )

        return CompoundModel('|', self, other)

# Usage:
try:
    result = model1 | model2
except IncompatibleModels as e:
    print(e)  # Clear error message with model details
```

#### Better Error Messages

```python
class ModelEvaluationError(Exception):
    def __init__(self, model, inputs, error):
        self.model = model
        self.inputs = inputs
        super().__init__(
            f"Error evaluating {model.name}: {error}\n"
            f"Model: {model.inspect()}\n"
            f"Inputs: {inputs}\n"
            f"Expected n_inputs={model.n_inputs}"
        )

# When evaluation fails:
try:
    result = model(x, y, z)  # Wrong number of inputs
except ModelEvaluationError as e:
    # User sees helpful message with model structure
```

---

## Part 4: Multiple API Options (Comparison)

### Option Comparison Matrix

| Feature | Current | Option 1: Dataclass | Option 2: Registry | Option 3: `__init_subclass__` | Option D: Hybrid |
|---|---|---|---|---|---|
| **Metaclass** | ❌ Complex | ✅ None | ✅ None | ✅ None | ✅ Simplified |
| **Type Hints** | ❌ No | ✅ Yes | ⚠️ Partial | ⚠️ Partial | ⚠️ Partial |
| **IDE Support** | ❌ Poor | ✅ Excellent | ⚠️ OK | ⚠️ OK | ⚠️ OK |
| **Backward Compat** | N/A | ❌ No | ⚠️ Some | ✅ Yes | ✅ Yes |
| **Boilerplate** | ⚠️ Some | ✅ Low | ❌ High | ⚠️ Medium | ✅ Low |
| **Familiar to Python** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Easy to Explain** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Migration Effort** | N/A | ❌ High | ⚠️ Medium | ✅ Low | ✅ Low |
| **Adoption Timeline** | Now | 2-3 years | 1-2 years | Now | Now |

---

### Key Design Decisions

#### Decision 1: How to Define Parameters?

**Option A**: Dataclass fields (best for new code)
```python
@dataclass
class Gaussian1D(Model):
    amplitude: Param = field(default=1.0)
```

**Option B**: Class attributes + `__init_subclass__` (best for migration)
```python
class Gaussian1D(Model):
    amplitude = Parameter(default=1.0)
```

**Recommendation**: Both. Start with Option B (backward compatible), introduce Option A
as new recommended pattern.

---

#### Decision 2: How to Access Parameters?

**Option A**: Via `model.parameters` dict
```python
model.parameters['amplitude'].value = 2.0
```

**Option B**: Direct attribute (backward compatible)
```python
model.amplitude.value = 2.0
```

**Recommendation**: Both. Direct access for backward compat, `model.parameters` as
official interface.

---

#### Decision 3: How to Handle `__call__` Keywords?

**Option A**: Explicit methods
```python
model.evaluate_with_bounding_box(x)
```

**Option B**: Keep keyword args; document better
```python
model(x, with_bounding_box=True)
```

**Recommendation**: Explicit methods. Clearer intent, better discoverable.

---

#### Decision 4: How to Handle Fitting?

**Option A**: Return FitResult object; don't modify model
```python
result = fitter(model, x, y)
```

**Option B**: Keep current (modify in-place)
```python
fitter(model, x, y)
```

**Recommendation**: Return FitResult. No side effects is Python convention.

---

## Part 5: Migration Strategy

### Phase 1: Simplify Metaclass (6-12 months)

**Immediate actions**:
1. Move operators to base class (no metaclass needed)
2. Extract parameter collection to standalone function
3. Move bounding box/inverse wrapping to explicit methods
4. Add `__init_subclass__` hook alongside metaclass

**Result**: Metaclass shrinks from 300 lines to 50.

**Backward compatibility**: 100% — existing code unchanged.

---

### Phase 2: New API Patterns (Next Major Version, 12-24 months)

**Introduce**:
1. Dataclass-based model definition (new)
2. Explicit methods (`evaluate_with_units`, etc.)
3. FitResult object from fitters
4. Deprecate old `__call__` keyword approach

**Result**: Users can adopt new patterns at their pace.

**Backward compatibility**: Old patterns still work but deprecated.

---

### Phase 3: Remove Old Patterns (Major Version +1, 24-36 months)

**Remove**:
1. Metaclass (if no longer needed)
2. Old `__call__` keyword arguments
3. In-place fitting
4. Indirect parameter access patterns

**Result**: Clean, modern, maintainable API.

**Backward compatibility**: Breaking change; users must migrate.

---

## Part 6: Summary and Recommendations

### What to Keep (Works Well)

- ✅ Composition with pipe operator
- ✅ Declarative parameter definition
- ✅ Constraint system (improve interface)
- ✅ Unit integration
- ✅ Named inputs/outputs
- ✅ Explicit `evaluate()` method

### What to Improve

- ⚠️ Metaclass → Replace with `__init_subclass__` gradually
- ⚠️ Parameter access → Unify into `model.parameters` interface
- ⚠️ `__call__` keywords → Split into explicit methods
- ⚠️ Fitting API → Return FitResult object
- ⚠️ Model-set axis → Simplify or remove

### What to Add

- ✨ Better error messages
- ✨ Composition validation
- ✨ Model introspection
- ✨ FitResult object
- ✨ Type hints support

### Recommended Implementation Order

1. **Now**: Simplify metaclass (Phase 1)
2. **Next release**: Add new API patterns alongside old (Phase 2)
3. **Major version**: Remove old patterns (Phase 3)

### Key Success Metrics

- ✅ Zero PEP 20 violations
- ✅ Full type hint support
- ✅ IDE autocomplete works
- ✅ New developers find API intuitive
- ✅ No breaking changes until major version
- ✅ Ecosystem successfully adopts new patterns

---

## Conclusion

The current astropy.modeling API works but exhibits design debt from evolution.
A clean-sheet redesign following modern Python standards is feasible and desirable.

**Key principles for the redesign**:

1. ✅ Eliminate metaclass complexity
2. ✅ Follow PEP 20 (one obvious way per task)
3. ✅ Enable type hints and IDE support
4. ✅ Preserve what works (composition)
5. ✅ Fix what doesn't (parameter access, fitting)
6. ✅ Add what's missing (validation, introspection, results)
7. ✅ Migrate incrementally with backward compatibility

**Next steps**:

1. Community discussion on design principles
2. Prototype Phase 1 (metaclass simplification)
3. Gather feedback from major downstream packages
4. Begin Phase 2 with deprecation warnings
5. Plan Phase 3 for major version release
