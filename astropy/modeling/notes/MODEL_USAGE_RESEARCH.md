# Custom astropy.modeling.Model Subclasses Research Report

## Search Scope
Research conducted on 7 major repositories that depend on astropy.modeling:

| Repository | Status | Findings |
|-----------|--------|----------|
| photutils/photutils | Searched | No custom Model subclasses found |
| spacetelescope/stdatamodels | ✅ Complete | 24+ custom Model subclasses |
| spacetelescope/jwst | ✅ Complete | Heavy Model usage patterns |
| astropy/romancal | ⚠️ Rate Limit | Partial results before API limit |
| astropy/stcal | ⚠️ Rate Limit | Partial results before API limit |
| sunpy/sunpy | Searched | No results found |
| dkist/dkist | Searched | No results found |

---

## Key Findings: Custom Model Subclasses

### 1. stdatamodels/jwst/transforms/models.py - 24 Custom Subclasses

All inherit directly from `astropy.modeling.core.Model`.

#### Coordinate Transform Models
- **Gwa2Slit** / **Slit2Gwa** - GWA ↔ slit coordinate transforms
- **Slit2Msa** / **Slit2MsaLegacy** - Slit ↔ MSA coordinate mapping
- **Msa2Slit** - Inverse mapping with name parameter
- **IdealToV2V3** / **V2V3ToIdeal** - Telescope coordinate systems
- **Rotation3DToGWA** - 3D rotation to GWA frame

#### Spectral Models
- **AngleFromGratingEquation** - Solves grating equation for refracted angle
- **WavelengthFromGratingEquation** - Solves grating equation for wavelength
- **RefractionIndexFromPrism** - NIRSpec prism refraction index

#### Grism Dispersion Models (6 variants)
- **NIRCAMBackwardGrismDispersion**
- **NIRCAMForwardColumnGrismDispersion**
- **NIRCAMForwardRowGrismDispersion**
- **NIRISSBackwardGrismDispersion**
- **NIRISSForwardRowGrismDispersion**
- **NIRISSForwardColumnGrismDispersion**

#### Instrument-Specific Models
- **MIRIWFSSBackwardDispersion** / **MIRIWFSSForwardDispersion**
- **Snell** - Complex NIRSpec prism with Snell's law
- **NirissSOSSModel** - NIRISS SOSS spectroscopy
- **MIRI_AB2Slice** - Array slice mapping
- **Logical** - Logical operations on parameters
- **Unitless2DirCos** / **DirCos2Unitless** - Vector/cosine transforms

---

## Model Patterns NOT Currently in Protocol

### Pattern 1: Direct Parameter Storage as Attributes

Many custom models store computed parameters as instance attributes:

```python
class Snell(Model):
    def __init__(self, angle, kcoef, lcoef, tcoef, tref, pref, temperature, pressure, name=None):
        self.prism_angle = angle
        self.kcoef = np.array(kcoef, dtype=float)
        self.lcoef = np.array(lcoef, dtype=float)
        self.tcoef = np.array(tcoef, dtype=float)
        self.tref = tref
        self.pref = pref
        self.temp = temperature
        self.pressure = pressure
        super(Snell, self).__init__(...)
```

**Impact**: Custom attributes beyond `parameters` property should be accessible.

### Pattern 2: Dynamic Input/Output Name Assignment

Models set custom input/output names at initialization:

```python
class Gwa2Slit(Model):
    def __init__(self, slits, models):
        self.inputs = ("name", "angle1", "angle2", "angle3")
        self.outputs = ("name", "x_slit", "y_slit", "lam")
        super(Gwa2Slit, self).__init__()
```

**Impact**: `inputs` and `outputs` are writable attributes, not just properties.

### Pattern 3: Parameter with Setter/Getter Functions

Using `Parameter()` with `setter` and `getter` callbacks:

```python
class RefractionIndexFromPrism(Model):
    prism_angle = Parameter(setter=np.deg2rad, getter=np.rad2deg)

    def __init__(self, prism_angle, name=None):
        super().__init__(prism_angle=prism_angle, ...)
```

**Impact**: Parameter class supports `setter` and `getter` arguments for unit conversion.

### Pattern 4: Complex Model Composition with Operators

Heavy use of model composition in `evaluate()` methods:

```python
# From _WFSSForwardGrismDispersion.evaluate()
dxr = astmath.SubtractUfunc()
wavelength = dxr | tab | lmodel
model = mapping | Const1D(x00) & Const1D(y00) & wavelength & Const1D(order)
return model(x, y, x0, y0, order)
```

**Impact**: Models are callable with other models in complex pipelines.

### Pattern 5: Static evaluate() Methods

Some models define static evaluate:

```python
class IdealToV2V3(Model):
    @staticmethod
    def evaluate(xidl, yidl, v3idlyangle, v2ref, v3ref, vparity):
        v3idlyangle = np.deg2rad(v3idlyangle)
        v2 = v2ref + vparity * xidl * np.cos(v3idlyangle) + yidl * np.sin(v3idlyangle)
        v3 = v3ref - vparity * xidl * np.sin(v3idlyangle) + yidl * np.cos(v3idlyangle)
        return v2, v3
```

**Impact**: `evaluate()` can be a `@staticmethod` but parameters are still passed.

### Pattern 6: Custom Model Attributes Beyond Parameters

Models access computation-specific attributes:

```python
class Gwa2Slit(Model):
    def __init__(self, slits, models):
        self._slits = []  # Computation storage
        self.slit_ids = []  # Lookup cache
        self.models = models  # Sub-models list
```

**Impact**: Models can have instance attributes for internal state beyond `Parameter` objects.

### Pattern 7: Model.inverse() Returning New Instances

The `inverse()` method creates new model instances:

```python
class IdealToV2V3(Model):
    def inverse(self):
        return V2V3ToIdeal(self.v3idlyangle, self.v2ref, self.v3ref, self.vparity)
```

**Impact**: `inverse()` returns a new Model instance, not self.

### Pattern 8: Tabular1D with Custom Parameters

Using `Tabular1D` with specific bounding behavior:

```python
# From _WFSSForwardGrismDispersion.evaluate()
tab = Tabular1D(alongdisp[so], t[so], bounds_error=False, fill_value=None)
```

**Impact**: `Tabular1D` accepts `bounds_error` and `fill_value` parameters.

### Pattern 9: bind_bounding_box Usage

In jwst files, models are wrapped with bounding boxes:

```python
from astropy.modeling import bind_bounding_box
# Used in fgs.py, miri.py, niriss.py, nirspec.py
```

**Impact**: Common pattern for applying bounding boxes to compound models.

### Pattern 10: Derived Properties from Parameters

Models compute properties from Parameter values:

```python
class _ForwardGrismDispersionBase(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Properties that depend on parameters set after super().__init__()
        if self.dispaxis == "row":
            self.alongdisp_models = self.xmodels
        elif self.dispaxis == "column":
            self.alongdisp_models = self.ymodels
```

**Impact**: Derived attributes computed in `__init__` depend on parameters.

---

## Methods/APIs Used in External Packages

### Frequently Used Model Methods

1. **Model Composition Operators**
   - `&` (and) - Combine models in parallel
   - `|` (or) - Compose models in series
   - Used extensively in evaluate methods for complex transforms

2. **bind_bounding_box()**
   - Applied to models to set pixel coordinate bounds
   - Used in: fgs.py, miri.py, niriss.py, nirspec.py

3. **fix_inputs()**
   - Used in nirspec.py for parameter fixing

4. **Model Calling Convention**
   - `model(x, y, ...)` - Direct evaluation
   - `model.evaluate(x, y, ...)` - Explicit evaluation

5. **Property Access**
   - `model.inputs` / `model.outputs` - Read and write
   - `model.n_inputs` / `model.n_outputs` - Read
   - `model.parameters` - Parameter tuple
   - `model.inverse` / `model.has_user_inverse` - Inverse checking

### Built-in Models Used

- `Const1D`, `Const2D` - Constant value models
- `Mapping` - Input/output routing
- `Rotation2D`, `Rotation3D` - Rotations
- `Tabular1D`, `Tabular2D` - Lookup tables
- `Polynomial1D`, `Polynomial2D` - Polynomial fits
- `Identity` - Identity transform
- `Scale`, `Shift` - Scale and shift operations

### Utility Modules

- `astropy.modeling.models.math` - as `astmath`
  - `SubtractUfunc()` - Subtraction operation as model
- `astropy.modeling.parameters.Parameter` - With setter/getter
- `astropy.modeling.bounding_box.CompoundBoundingBox` - Complex bounds

---

## Protocol Coverage Assessment

### ✅ Well Covered
- Basic Model API (call, evaluate)
- Parameter access and constraints
- Inverse checking
- Bounding box properties
- Input/output naming
- Model composition operators
- Linear/separable/standard_broadcasting attributes

### ⚠️ Partially Covered / Edge Cases
- **Parameter setter/getter functions** - Protocol shows Parameter but not setter/getter
- **Direct attribute storage** - Protocol shows parameters property but not custom attributes
- **Writable inputs/outputs** - Protocol shows these as properties but they're assignable
- **Static evaluate methods** - Signature may vary with parameters
- **Derived properties** - Computed in __init__, not reflected in protocol

### ❌ Not in Protocol
- **bind_bounding_box()** - Top-level function, not method
- **fix_inputs()** - Top-level function, not method
- **Model._separable** - Private attribute used for optimization
- **Model.standard_broadcasting** - Class attribute for broadcasting behavior
- **Model.name** - Instance name attribute
- **Model.meta** - Metadata dictionary

### Missing Functional Patterns
1. Top-level functions: `bind_bounding_box()`, `fix_inputs()`, `custom_model()`
2. Parameter with setter/getter support
3. Custom attribute storage pattern (beyond parameters)
4. Complex model evaluation pipeline composition

---

## Recommendations for Protocol Enhancement

### Priority 1: High Impact
1. Add top-level function signatures:
   - `bind_bounding_box(model, bounding_box) -> Model`
   - `fix_inputs(model, **kwargs) -> Model`

2. Document Parameter setter/getter support:
   ```python
   Parameter(default=None, setter=None, getter=None)
   ```

3. Add `name` and `meta` attributes to protocol

### Priority 2: Medium Impact
1. Document that `inputs` and `outputs` are assignable (not just readable)
2. Add `_separable` and `standard_broadcasting` class attributes
3. Document static method support for `evaluate()`

### Priority 3: Low Impact (Edge Cases)
1. Custom attribute storage pattern
2. Derived property computation in `__init__`
3. Complex model composition patterns in evaluate

---

## Examples of Complex Patterns Found

### Example 1: Grism Dispersion Model (Complex)
File: stdatamodels/jwst/transforms/models.py

```python
class _WFSSForwardGrismDispersion(_ForwardGrismDispersionBase):
    def evaluate(self, x, y, x0, y0, order):
        # Complex composition with multiple models
        mapping = Mapping((0, 1, 2, 3, 4))
        const_models = Const1D(x00) & Const1D(y00) & wavelength & Const1D(order)
        composed = mapping | const_models
        return composed(x, y, x0, y0, order)
```

### Example 2: Parameter with Unit Conversion
```python
class RefractionIndexFromPrism(Model):
    prism_angle = Parameter(setter=np.deg2rad, getter=np.rad2deg)
```

### Example 3: Inverse Model Creation
```python
class IdealToV2V3(Model):
    def inverse(self):
        return V2V3ToIdeal(...)
```

### Example 4: Custom Attributes Storage
```python
class Gwa2Slit(Model):
    def __init__(self, slits, models):
        self._slits = []  # Private storage
        self.slit_ids = []  # Lookup table
        self.models = models  # Sub-models
```

---

## Conclusion

The protocol captures the core Model API well, but external packages rely on several patterns and functions not currently documented:

1. **Top-level functions** - `bind_bounding_box()`, `fix_inputs()`
2. **Parameter advanced features** - setter/getter functions
3. **Attribute assignment** - models support custom attributes and assignable inputs/outputs
4. **Optimization attributes** - `_separable`, `standard_broadcasting`
5. **Complex composition** - Heavy use of model operators in evaluate methods

Most gaps are **advanced usage patterns** rather than core API problems. The protocol could be strengthened by documenting these patterns and adding support for the top-level functions.
