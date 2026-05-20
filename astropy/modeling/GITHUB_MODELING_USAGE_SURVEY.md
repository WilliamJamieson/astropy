# GitHub Survey: astropy.modeling Public API Usage

## Scope

This document summarizes a GitHub survey of downstream usage of the generic public API in `astropy.modeling`.

The survey explicitly focused on generic modeling interfaces and workflows, including:

- `Model`, `CompoundModel`, and fittable model base classes
- `Parameter`
- `fitting` APIs and fitter classes
- `bounding_box` APIs (`ModelBoundingBox`, `CompoundBoundingBox`, bind helpers)
- general utilities and composition (`fix_inputs`, `separable`, mappings/tabular/projections/rotations)

The survey intentionally did **not** focus on enumerating specific built-in model classes (e.g. Gaussian, Polynomial) except where they appeared as evidence of broader API use.

## Requested Repositories

### 1. spacetelescope/gwcs

Status: **Strong active usage**

Observed usage includes:

- core `Model` interface used as transform contracts
- custom model subclasses and custom parameters
- `fitting`, `separable`, `fix_inputs`
- `bounding_box` API (`ModelBoundingBox`, `CompoundBoundingBox`)
- compound model composition throughout WCS pipeline code

Representative paths:

- `gwcs/wcs/_wcs.py`
- `gwcs/wcstools.py`
- `gwcs/geometry.py`
- `gwcs/spectroscopy.py`
- `gwcs/selector.py`

### 2. astropy/photutils

Status: **Strong active usage**

Observed usage includes:

- generic `Model` and `CompoundModel` contracts
- `Fittable2DModel` and `Parameter` in custom modeling code
- fitter usage via `astropy.modeling.fitting` (e.g. TRF/LevMar families)
- model-based PSF/profile APIs expecting generic astropy modeling interfaces

Representative paths:

- `photutils/psf/functional_models.py`
- `photutils/psf/image_models.py`
- `photutils/psf/model_helpers.py`
- `photutils/psf/photometry.py`
- `photutils/datasets/images.py`

### 3. spacetelescope/stdatamodels

Status: **Strong active usage**

Observed usage includes:

- `Model` as transform payload type in datamodels
- transform model implementations based on astropy modeling
- converter integration around modeling objects

Representative paths:

- `src/stdatamodels/jwst/transforms/models.py`
- `src/stdatamodels/jwst/transforms/converters/jwst_models.py`
- `src/stdatamodels/jwst/datamodels/wcs_ref_models.py`

### 4. spacetelescope/stcal

Status: **Moderate active usage**

Observed usage includes:

- model-based transform handling in alignment/resample logic
- typed expectations of generic `Model` transforms (inputs/outputs/inverse semantics)
- use of astropy modeling namespaces in source modules

Representative paths:

- `src/stcal/alignment/util.py`
- `src/stcal/alignment/resample_utils.py`
- `src/stcal/tweakreg/utils.py`

### 5. spacetelescope/jwst

Status: **Very strong active usage**

Observed usage includes:

- heavy use of generic modeling interfaces across WCS/assign/resample/extract code
- `bind_bounding_box`, `bounding_box` APIs, `CompoundModel`
- utility namespaces (mappings, tabular, projections, polynomial)
- fitter usage (`LinearLSQFitter`, others)

Representative paths:

- `jwst/assign_wcs/nirspec.py`
- `jwst/assign_wcs/nircam.py`
- `jwst/assign_wcs/miri.py`
- `jwst/resample/resample.py`
- `jwst/extract_2d/grisms.py`

### 6. spacetelescope/romancal

Status: **Strong active usage**

Observed usage includes:

- model-based WCS construction and transform pipelines
- `bind_bounding_box` usage
- fitter/spline API usage in source-catalog components

Representative paths:

- `romancal/assign_wcs/assign_wcs.py`
- `romancal/resample/_l3_wcs.py`
- `romancal/source_catalog/_utils.py`
- `romancal/skycell/skymap.py`

### 7. sunpy/sunpy

Status: **No direct source-level hits found in this survey run**

Notes:

- direct lexical search in `sunpy/sunpy` did not return source-level imports of `astropy.modeling`
- related SunPy ecosystem projects do actively use astropy modeling (see below)

### 8. DKISTDC/dkist

Status: **Strong active usage**

Observed usage includes:

- `Model`, `CompoundModel`, `Parameter`, and `separable`
- model graph and WCS model composition workflows
- internal model converter/serialization integration

Representative paths:

- `dkist/wcs/models.py`
- `dkist/utils/_model_to_graphviz.py`
- `dkist/io/asdf/converters/models.py`

## Additional Active Projects Found

Beyond the requested repositories, the survey found active usage in:

- `sunpy/ndcube`
- `sunpy/sunkit-spex`
- `spacetelescope/synphot_refactor`
- `spacetelescope/jdaviz`
- `spacetelescope/poppy`
- `spacetelescope/romanisim`
- `spacetelescope/pysiaf`
- `spacetelescope/mirage`
- `spacetelescope/tweakwcs`
- `spacetelescope/drizzlepac`
- `spacetelescope/astrocut`
- `spacetelescope/roman_datamodels`
- `spacetelescope/stistools`
- `spacetelescope/jwql`
- `spacetelescope/spaceKLIP`
- `spacetelescope/jwreftools`

## Summary

The generic public API of `astropy.modeling` is actively used across major downstream astronomy pipelines and tooling ecosystems. The strongest concentration of usage appears in GWCS/JWST/romancal/stdatamodels/dkist, with substantial usage in photutils and moderate but clear usage in stcal.

The only requested repository without direct source-level hits in this run was `sunpy/sunpy`; however, related repositories under the SunPy organization (notably `ndcube` and `sunkit-spex`) do make active use of astropy modeling.

## Protocol Taxonomy Link

The findings in this survey are used to inform the entry-importance taxonomy
(`Core`, `Common`, `Optional/Specialized`) documented in
`astropy/modeling/protocol.py`.

In particular, entries marked as `Core` in protocol docstrings correspond to
API surfaces observed as consistently central across the surveyed downstream
projects, while `Common` and `Optional/Specialized` reflect progressively more
situational usage patterns.

## Method Note

The survey was performed via GitHub code search using lexical queries over target repositories and organizations. A small number of query variants intermittently failed at the API level; where this occurred, findings were corroborated with successful broader queries.
