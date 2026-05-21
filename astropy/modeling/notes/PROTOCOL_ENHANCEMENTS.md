# Protocol Enhancement Recommendations

## Missing Elements Identified from External Repository Analysis

### 1. Top-Level Functions NOT in Protocol

These functions are frequently used but are not part of the Model protocol:

#### bind_bounding_box()
```python
from astropy.modeling import bind_bounding_box

# Usage in jwst/assign_wcs/fgs.py, miri.py, niriss.py, nirspec.py
bound_model = bind_bounding_box(model, bounding_box_dict)
```

**Status**: Function, not method. Should be documented in modeling.core or a functions module.

#### fix_inputs()
```python
from astropy.modeling import fix_inputs

# Usage in jwst/assign_wcs/nirspec.py
fixed_model = fix_inputs(model, x=5.0, y=10.0)
```

**Status**: Function, not method. Similar to bind_bounding_box.

#### custom_model() decorator
Used in some advanced cases for wrapping functions as models.

---

### 2. Parameter Class Enhancement

Current protocol doesn't document Parameter internals. External packages use:

```python
Parameter(setter=np.deg2rad, getter=np.rad2deg)
```

**Finding**: Parameter class supports `setter` and `getter` keyword arguments for automatic unit/scale conversion.

**Protocol Gap**: Not mentioned in Parameter or anywhere

---

### 3. Missing Class Attributes

Found in external models but not in protocol:

```python
class CustomModel(Model):
    _separable = False  # Private attribute used for optimization
    standard_broadcasting = False  # Documented but not widely known
```

**Pattern**: `_separable` is a private attribute that optimizes compound model handling.

**Protocol Status**: `standard_broadcasting` is documented, but `_separable` is private and undocumented.

---

### 4. Attribute Assignability

Protocol documents these as properties (read-only):
```python
inputs: tuple[str, ...]
outputs: tuple[str, ...]
```

**Finding**: External models directly assign these at initialization:
```python
self.inputs = ("x", "y", "angle")
self.outputs = ("dx", "dy", "wavelength")
```

**Protocol Status**: Should clarify these are assignable.

---

### 5. Custom Attribute Pattern

Models store computation-specific attributes beyond parameters:

```python
class Gwa2Slit(Model):
    def __init__(self, slits, models):
        self._slits = []  # Internal storage
        self.slit_ids = []  # Lookup cache
        self.models = models  # Sub-models list
```

**Protocol Gap**: No guidance on custom attribute storage.

---

### 6. Parameter Setter/Getter Support

Found in external models:

```python
class RefractionIndexFromPrism(Model):
    prism_angle = Parameter(setter=np.deg2rad, getter=np.rad2deg)

    def __init__(self, prism_angle, name=None):
        super().__init__(prism_angle=prism_angle)
        # Value stored in radians internally
        # Accessed as degrees via getter
```

**Use Case**: Automatic unit conversion during parameter access.

---

### 7. Model Inverse Creation Pattern

Protocol shows `inverse` property, but doesn't document creation pattern:

```python
def inverse(self):
    """Create an inverse model."""
    return NewModelType(...)  # Creates NEW instance
```

**Protocol Status**: Documented as property returning a Model, but the pattern of creating new instances should be clearer.

---

### 8. Static evaluate() Methods

Protocol doesn't mention that evaluate can be `@staticmethod`:

```python
class IdealToV2V3(Model):
    @staticmethod
    def evaluate(xidl, yidl, v3idlyangle, v2ref, v3ref, vparity):
        ...
```

**Protocol Gap**: Doesn't mention @staticmethod possibility for evaluate.

---

### 9. Model Composition in evaluate()

Complex models use heavy composition in evaluate():

```python
def evaluate(self, x, y, x0, y0, order):
    dxr = astmath.SubtractUfunc()
    wavelength = dxr | tab | lmodel
    model = mapping | Const1D(x00) & Const1D(y00) & wavelength & Const1D(order)
    return model(x, y, x0, y0, order)
```

**Pattern**: Models constructed and evaluated within evaluate() method.

**Protocol Status**: Model operators documented, but not typical usage in custom models.

---

### 10. CompoundBoundingBox Usage

```python
from astropy.modeling.bounding_box import CompoundBoundingBox

# Used in complex models with multiple input regions
bbox = CompoundBoundingBox(...)
```

**Protocol Gap**: Not mentioned as part of bounding_box API.

---

## Recommendations by Priority

### PRIORITY 1: Critical Gaps (Should document)

1. **Top-level functions**: Add to core or new functions module:
   - `bind_bounding_box(model, bounding_box) -> Model`
   - `fix_inputs(model, **kwargs) -> Model`
   - `custom_model(*args, **kwargs)` (if not already documented)

2. **Parameter setter/getter**: Document in Parameter API
   - `Parameter(..., setter=callable, getter=callable)`
   - Use case: Unit conversion, scale factors, etc.

3. **Attribute assignability**: Clarify in protocol that:
   - `model.inputs` is writable
   - `model.outputs` is writable
   - Custom attributes can be added

### PRIORITY 2: Important Clarifications (Should add notes)

1. **Model.inverse** creation pattern:
   - Document that `.inverse` property returns NEW model instances
   - Not a copy, but new instance of different class

2. **_separable attribute**: Document as private optimization flag
   - Used by compound models for performance
   - Not for user override in normal cases

3. **CompoundBoundingBox**: Document as bounding box type
   - Returned by complex models
   - Combines multiple box regions

4. **Model.name** attribute: Add to protocol
   - Instance attribute for model identification
   - Useful in compound models

### PRIORITY 3: Advanced Documentation (Optional)

1. **Model composition patterns in evaluate()**:
   - Document as advanced use case
   - Show example of internal model construction

2. **Static evaluate() methods**:
   - Document that evaluate can be staticmethod
   - Must handle parameter passing carefully

3. **Custom sub-model storage**:
   - Document pattern of storing sub-models in attributes
   - Important for complex transforms

---

## Current Protocol Gaps Summary

| Element | Status | Impact |
|---------|--------|--------|
| bind_bounding_box() | Missing | High - Common in JWST |
| fix_inputs() | Missing | Medium - Some use |
| Parameter setter/getter | Missing | Medium - Unit conversion |
| Writable inputs/outputs | Unclear | Medium - Custom models |
| _separable attribute | Missing | Low - Optimization only |
| Custom attributes | Undocumented | Low - Implementation detail |
| Model.name | Missing | Low - Nice-to-have |
| CompoundBoundingBox | Missing | Low - Advanced |

---

## Implementation Notes

1. **bind_bounding_box() and fix_inputs()** are already in astropy.modeling
   - Just need to be documented/linked in protocol

2. **Parameter setter/getter** is part of existing Parameter implementation
   - Just needs documentation update

3. **inputs/outputs assignability** needs protocol clarification
   - Currently not clear they're writable

4. Most gaps are **documentation/clarity issues** rather than API problems
