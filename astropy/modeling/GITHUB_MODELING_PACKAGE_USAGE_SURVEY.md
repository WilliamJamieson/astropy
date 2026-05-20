# GitHub Survey: Package Usage of astropy.modeling

Date: 2026-05-20

## Scope and Method

This is a broad lexical code-search snapshot of GitHub repositories, focused on
Python package repositories in astronomy-related organizations.

Primary search scopes used:

- `astropy`
- `spacetelescope`
- `sunpy`
- `gammapy`
- `lsst`

Primary query style used:

- `"astropy.modeling" language:python`

Important caveats:

- This is not exhaustive across all of GitHub.
- Lexical search can miss indirect usage (e.g., re-exported APIs, aliases).
- Some orgs/repositories had intermittent lexical search failures.

## Summary of Usage Patterns

Across package repos, `astropy.modeling` is used in several recurring ways:

1. Model construction and composition
   - Building transforms and analytic models with `models.*`
   - Composing pipelines with `|` and `&`
2. Fitting and optimization
   - Using fitters such as `TRFLSQFitter`, `LevMarLSQFitter`, `LMLSQFitter`
3. WCS and coordinate transforms
   - Heavy use in `gwcs`/`jwst`/`romancal`-adjacent pipelines
4. Custom model classes
   - Defining `Fittable1DModel`/`Fittable2DModel` subclasses with `Parameter`
5. Tabular / polynomial / projection models
   - `Tabular1D`, `Polynomial1D/2D`, projection and mapping models
6. Serialization and schema interoperability
   - ASDF converters mapping model types to tagged formats

## Package Repository Findings

### astropy organization

- `astropy/photutils`
  - Uses `Model`, `Fittable2DModel`, `Parameter`, common functional models
    (`Gaussian1D/2D`, `Moffat1D`, `Const2D`), and fitters.
  - Typical use: PSF/centroid/profile modeling and fitting workflows.
- `astropy/specutils`
  - Uses polynomial/tabular/mapping models and fitters for continuum and line
    workflows.
  - Typical use: spectral modeling and fitting utilities.
- `astropy/specreduce`
  - Uses `models`, `Model`, `CompoundModel`, fitters, and polynomial models.
  - Typical use: extraction, tracing, wavelength calibration pipelines.
- `astropy/ccdproc`
  - Uses modeling fitters and model objects in reduction utilities.
- `astropy/asdf-astropy`
  - Uses extensive converters for model classes (`CompoundModel`, tabular,
    polynomial, mappings, rotations, bounding boxes, math functions).
  - Typical use: serialization/deserialization of modeling objects.
- `astropy/astroquery`
  - Uses model objects for runtime predictor modeling in utility code.
- `astropy/SPISEA`
  - Uses `astropy.modeling.powerlaws` (e.g., `BrokenPowerLaw1D`).
- `astropy/reproject`
  - Hits were mainly test/data generation usage.

### spacetelescope organization

- `spacetelescope/gwcs`
  - Core dependency on `astropy.modeling.Model`, mappings, projections,
    bounding boxes, and compound-model composition.
  - Typical use: WCS transform graph construction and transform composition.
- `spacetelescope/jwst`
  - Heavy usage in `assign_wcs` and related steps with
    `Identity`, `Mapping`, `Shift`, `Scale`, `RotationSequence3D`, tabular and
    polynomial models.
- `spacetelescope/romancal`
  - Uses mapping/rotation/shift/scale and model-based distortion/resampling
    transforms.
- `spacetelescope/romanisim`
  - Uses `models` for WCS/distortion model construction.
- `spacetelescope/stdatamodels`
  - Uses model classes to represent/reference transform payloads in data models.
- `spacetelescope/stcal`
  - Uses model-typed transform interfaces in alignment/resampling utilities.
- `spacetelescope/roman_datamodels`
  - Uses model objects in datamodel/testing contexts.
- `spacetelescope/stpsf` and `spacetelescope/webbpsf`
  - Use functional models such as `Gaussian1D` in detector/PSF code paths.
- `spacetelescope/tweakwcs`
  - Uses fitters and affine/mapping-related model constructs for WCS correction.
- `spacetelescope/jwreftools`
  - Uses polynomial/mapping/identity and related models to build reference
    transforms.
- `spacetelescope/astrocut`
  - Uses SIP and related model-based distortion/WCS fitting helpers.
- Additional package repos with clear usage:
  - `spacetelescope/jdaviz`
  - `spacetelescope/jwql`
  - `spacetelescope/slitlessutils`
  - `spacetelescope/acstools`
  - `spacetelescope/poppy`
  - `spacetelescope/msaviz`
  - `spacetelescope/pysiaf`
  - `spacetelescope/synphot_refactor`

### sunpy organization

- `sunpy/sunkit-spex`
  - Defines custom `FittableModel`/`Fittable1DModel` components with
    `Parameter`; also uses mappings/tabular and fitting.
- `sunpy/ndcube`
  - Uses `models`, tabular model types, and mapping-related concepts in WCS/
    coordinate helpers.

### gammapy organization

- `gammapy/gammapy`
  - Defines custom `Fittable1DModel` classes and uses standard analytic models
    (`Gaussian1D`, etc.) in astrophysical distributions/catalog workflows.

### lsst organization

- `lsst/atmospec`
  - Uses `models` and `fitting` in extraction logic.
- `lsst/all_sky_phot`
  - Uses projection and affine models in WCS fitting utilities.
- `lsst/rubin_sim`
  - Uses selected model classes (e.g., `Schechter1D`).

## Requested Packages Check (from prior targeted survey)

The following requested package repos were confirmed with explicit source usage
in this survey context:

- `gwcs`, `jwst`, `romancal`, `romanisim`, `stpsf`, `photutils`, `dkist`,
  `stdatamodels`, `roman_datamodels`, `stcal`

Notes:

- `stpipe` did not show strong direct explicit hits in this pass; it is often
  used adjacent to model-using packages rather than as a primary modeling code
  surface.
- Usage in tests/docs/examples was filtered where possible, but some repos mix
  source and examples closely.

## Usage Taxonomy (Package-Level)

- Very heavy/core integration:
  - `gwcs`, `jwst`, `romancal`, `stdatamodels`
- Significant algorithmic use:
  - `photutils`, `specutils`, `specreduce`, `tweakwcs`, `sunpy/sunkit-spex`
- Focused or domain-specific use:
  - `stpsf`, `astrocut`, `jwreftools`, `romanisim`, `lsst/atmospec`
- Interoperability/serialization use:
  - `asdf-astropy`, `roman_datamodels`

## Feature Importance: Model API Usage Mapped to Protocol Elements

Ordered by practical importance in this broad package survey.

### 1. Evaluation contract and dimensionality (critical)

- Protocol elements:
  - `Model.__call__()`
  - `Model.evaluate()`
  - `Model.n_inputs`
  - `Model.n_outputs`
  - `Model.inputs`
  - `Model.outputs`
- Why it is critical:
  - Nearly all downstream usage assumes a callable transform/model with stable
    input/output dimensionality and naming.
- Survey usage:
  - `gwcs`, `jwst`, `romancal`, `stcal`, `stdatamodels` all rely on transform
    callability and strict dimensional expectations in WCS pipelines.
- Example pattern:
  - Declaring transform interfaces as `astropy.modeling.Model` and requiring a
    fixed `(x, y) -> (ra, dec)` style contract.

### 2. Compound model composition (critical)

- Protocol elements:
  - `Model.__or__` (serial composition)
  - `Model.__and__` (parallel composition)
  - `Model.__add__`, `Model.__sub__`, `Model.__mul__`, `Model.__truediv__`,
    `Model.__pow__`
- Why it is critical:
  - Composition is the backbone of modern WCS and calibration transform chains.
- Survey usage:
  - Strongly visible in `gwcs`, `jwst`, `romancal`, `jwreftools`, `romanisim`.
- Example pattern:
  - `Mapping(...) | Polynomial2D(...) | Shift(...)`

### 3. Parameter surface and constraints (high)

- Protocol elements:
  - `Model.param_names`
  - `Model.parameters`
  - `Model.fixed`
  - `Model.bounds`
  - `Model.tied`
  - `Model.has_fixed`, `Model.has_bounds`, `Model.has_tied`
  - `Model.parameter_constraints`
- Why it is high importance:
  - Calibration and fitting workflows depend on parameter inspection and
    constraint control.
- Survey usage:
  - Common in `photutils`, `specutils`, `specreduce`, `tweakwcs`, `sunkit-spex`,
    and instrument calibration repos.
- Example pattern:
  - Construct model -> set constraints -> fit with a selected fitter.

### 4. Fittability and linearity metadata (high)

- Protocol elements:
  - `Model.fittable`
  - `Model.linear`
  - `Model.fit_deriv` (specialized but important where available)
- Why it is high importance:
  - Determines fitter choice and expected numerical behavior.
- Survey usage:
  - Fitter-heavy repos (`photutils`, `specutils`, `specreduce`, `jwql`,
    `acstools`, `slitlessutils`) implicitly depend on this metadata.

### 5. Bounding regions and valid domains (high)

- Protocol elements:
  - `Model.bounding_box`
  - `Model.has_user_bounding_box`
- Why it is high importance:
  - Essential for safe evaluation domains in WCS and distortion models.
- Survey usage:
  - Explicit in `gwcs`, `jwst`, `romancal`, `astrocut`, plus ASDF conversion
    flows handling `ModelBoundingBox`.
- Example pattern:
  - Attach or consume bounding boxes for transform validity windows.

### 6. Inverse transform support (high)

- Protocol elements:
  - `Model.inverse`
  - `Model.has_inverse`
  - `Model.has_user_inverse`
- Why it is high importance:
  - Bidirectional detector/world transforms are a core requirement in pipeline
    WCS systems.
- Survey usage:
  - Frequent in `gwcs`, `jwst`, `romancal`, `stcal` transform interfaces.

### 7. Unit-aware model operation (medium-high)

- Protocol elements:
  - `Model.input_units`
  - `Model.output_units` (attribute)
  - `Model.output_units(...)` (method)
  - `Model.uses_quantity`
  - `Model.without_units_for_data()`
  - `Model.with_units_from_data()`
  - `Model.coerce_units()`
  - `Model.input_units_strict`
  - `Model.input_units_allow_dimensionless`
- Why it matters:
  - Astropy ecosystems are unit-centric; unit-safe modeling is expected in many
    analysis pipelines.
- Survey usage:
  - Seen across spectroscopy/photometry and transform-heavy repositories using
    `astropy.units` together with model evaluation and fitting.

### 8. Model metadata and naming (medium)

- Protocol elements:
  - `Model.name`
  - `Model.meta`
- Why it matters:
  - Useful for pipeline bookkeeping, provenance, and model component labeling.
- Survey usage:
  - Commonly present but less central than transform and fitting interfaces.

### 9. Model-set support (medium)

- Protocol elements:
  - `Model.param_sets`
  - `Model.model_set_axis`
  - `Model.__len__()`
- Why it matters:
  - Enables efficient grouped evaluation/fitting in some workflows.
- Survey usage:
  - Visible in notebook and calibration contexts; less universal than core
    callable/compound behavior.

### 10. Copy/render/prep helper methods (medium-lower)

- Protocol elements:
  - `Model.copy()`
  - `Model.deepcopy()`
  - `Model.render()`
  - `Model.prepare_inputs()`
  - `Model.prepare_outputs()`
  - `Model.__repr__()`, `Model.__str__()`
- Why it matters:
  - Improves ergonomics, diagnostics, and safe manipulation of model instances.
- Survey usage:
  - Common supporting usage rather than primary algorithmic dependency.

### 11. Fitting uncertainty attachments and separability flags (specialized)

- Protocol elements:
  - `Model.cov_matrix`
  - `Model.stds`
  - `Model.separable`
  - `Model.sync_constraints`
- Why it is specialized:
  - Important in advanced fitting diagnostics and decomposition logic, but not
    as universal as call/compose/parameter features.

### Compact Crosswalk Table (Feature -> Protocol -> Repositories)

| Feature | Protocol element(s) in `Model` | Representative package repos |
|---|---|---|
| Evaluation contract and dimensionality | `__call__()`, `evaluate()`, `n_inputs`, `n_outputs`, `inputs`, `outputs` | `gwcs`, `jwst`, `romancal`, `stcal`, `stdatamodels` |
| Compound composition | `__or__`, `__and__`, `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__pow__` | `gwcs`, `jwst`, `romancal`, `jwreftools`, `romanisim` |
| Parameters and constraints | `param_names`, `parameters`, `fixed`, `bounds`, `tied`, `has_fixed`, `has_bounds`, `has_tied`, `parameter_constraints` | `photutils`, `specutils`, `specreduce`, `tweakwcs`, `sunpy/sunkit-spex` |
| Fittability and linearity metadata | `fittable`, `linear`, `fit_deriv` | `photutils`, `specutils`, `specreduce`, `jwql`, `acstools` |
| Bounding domains | `bounding_box`, `has_user_bounding_box` | `gwcs`, `jwst`, `romancal`, `astrocut`, `asdf-astropy` |
| Inverse transforms | `inverse`, `has_inverse`, `has_user_inverse` | `gwcs`, `jwst`, `romancal`, `stcal` |
| Unit-aware operation | `input_units`, `output_units` (attribute/method), `uses_quantity`, `without_units_for_data()`, `with_units_from_data()`, `coerce_units()`, `input_units_strict`, `input_units_allow_dimensionless` | `specutils`, `specreduce`, `jwst`, `romancal`, `synphot_refactor` |
| Metadata and naming | `name`, `meta` | `jwst`, `romancal`, `stdatamodels`, `roman_datamodels` |
| Model-set interfaces | `param_sets`, `model_set_axis`, `__len__()` | `specreduce`, `jdaviz`, `jdat_notebooks` |
| Copy/render/preparation helpers | `copy()`, `deepcopy()`, `render()`, `prepare_inputs()`, `prepare_outputs()`, `__repr__()`, `__str__()` | `photutils`, `specreduce`, `jwreftools` |
| Fit diagnostics/separability | `cov_matrix`, `stds`, `separable`, `sync_constraints` | `photutils`, `specutils`, `tweakwcs` |

## Specific Model Classes by Importance (Broad GitHub Survey)

Ranking basis: number of distinct surveyed package repositories with explicit,
non-test usage.

### Tier 1 (very widely used)

- `Mapping` (7 packages)
- `Identity` (6 packages)
- `Polynomial2D` (6 packages)
- `Shift` (6 packages)

### Tier 2 (widely used)

- `Const1D` (4 packages)
- `Scale` (4 packages)

### Tier 3 (common)

- `AffineTransformation2D` (3 packages)
- `Gaussian1D` (3 packages)

### Tier 4 (recurring, specialized)

- `Pix2Sky_TAN` (2 packages)
- `RotationSequence3D` (2 packages)
- `Tabular1D` (2 packages)

### Tier 5 (package-specific but important in domain workflows)

- `BlackBody`, `Const2D`, `Gaussian2D`, `Linear1D`, `Lorentz1D`, `Moffat1D`,
  `Moffat2D`, `Planar2D`, `Polynomial1D`, `RedshiftScaleFactor`,
  `RotateCelestial2Native`, `Rotation2D`, `SIP`, `Sersic1D`, `Sersic2D`,
  `Sine1D`, `Spline1D`, `Voigt1D` (1 package each)

Interpretation:

- Top tiers are dominated by transform-composition primitives and WCS-related
  models, consistent with the heavy ecosystem demand for coordinate pipelines.
- Mid/lower tiers include domain-specific photometry/spectroscopy profiles and
  calibration-specialized models.

## Representative Snippets (high-level)

- Imports such as:
  - `from astropy.modeling import models, fitting`
  - `from astropy.modeling.models import Identity, Mapping, Shift, Polynomial2D`
  - `from astropy.modeling import Fittable1DModel, Parameter`
  - `from astropy.modeling.bounding_box import ModelBoundingBox`
- Model composition and transform chaining:
  - `model_a | model_b`
  - `model_a & model_b`

## Bottom Line

`astropy.modeling` is broadly used across astronomy package ecosystems as:

- a model-definition framework,
- a fitter interface surface,
- and a transform-composition substrate (especially for WCS pipelines).
