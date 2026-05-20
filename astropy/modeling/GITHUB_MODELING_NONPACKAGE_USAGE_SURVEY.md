# GitHub Survey: Non-Package / Research-Style Usage of astropy.modeling

Date: 2026-05-20

## Scope and Method

This document summarizes usage in repositories that are primarily notebook,
tutorial, workshop, sandbox, demo, or research/workflow oriented rather than
core library packages.

Primary search method:

- GitHub lexical code search for `"astropy.modeling"`
- Additional notebook-focused passes (`path:notebooks`, `extension:ipynb`)
- Main scopes: `astropy`, `spacetelescope` (plus related hits discovered there)

Caveats:

- Classification as "non-package" is heuristic and based on repository intent
  inferred from names/content patterns.
- Some repos may contain both package code and notebook/tutorial materials.

## Common Non-Package Usage Patterns

In research-style repos, `astropy.modeling` is most often used for:

1. Interactive fitting in notebooks
   - Continuum fitting, line fitting, PSF fitting, polynomial calibration
2. Educational demonstration
   - Introductory model construction, evaluation, and fitter walkthroughs
3. WCS/transformation prototyping
   - Building toy/production-like transform chains with `gwcs` + modeling
4. Synthetic data generation
   - Generating Gaussian/Sersic/PSF-like simulated sources

## Surveyed Non-Package / Research-Style Repositories

### astropy ecosystem

- `astropy/astropy-workshop`
  - Direct notebook-based instruction on modeling concepts, fitters,
    compound models, and model arithmetic.
- `astropy/ccd-reduction-and-photometry-guide`
  - Notebook-driven use of functional models (e.g., `Gaussian2D`, `Const2D`)
    for simulated image construction.
- `astropy/specreduce` notebook sandboxes/docs
  - Research/development notebooks using `models` for extraction/tracing demos.

### spacetelescope ecosystem (notebook/demo-heavy repos)

- `spacetelescope/jdat_notebooks`
  - Extensive use in JWST analysis notebooks for spectral fitting,
    continuum modeling, custom model definitions, and WCS transform handling.
- `spacetelescope/hst_notebooks`
  - Uses `astropy.modeling` for line profiles, polynomial fitting, and
    calibration tutorials.
- `spacetelescope/mast_notebooks`
  - Notebook workflows using `models`/`fitting` for catalog/data analysis.
- `spacetelescope/da5-notebooks`
  - Uses `models`/`fitting` for wavelength registration and PSF workflows.
- `spacetelescope/jdaviz_demo`
  - Demonstrates fitting workflows using `Polynomial1D`, `Spline1D`, etc.
- `spacetelescope/jdat_demo`
  - Demonstrates model-set usage in extraction workflows.
- `spacetelescope/ccsp_supplement`
  - Workflow notebooks using `astropy.modeling.models` in WCS build chains.
- `spacetelescope/astrogrism_sandbox`
  - Sandbox scripts for distortion/grism references using
    `Polynomial2D`, `Mapping`, `Shift`, and `SIP`.

### Additional mixed-use examples

- `spacetelescope/jwst_magic` notebooks/utilities
  - Uses `models` and `fitting` in commissioning-support style notebook code.

## Example Usage Modes Observed

- Functional model imports in notebooks:
  - `Gaussian1D`, `Gaussian2D`, `Moffat*`, `Polynomial1D/2D`, `Spline1D`
- Interactive fitting imports:
  - `from astropy.modeling import fitting`
  - `from astropy.modeling.fitting import LevMarLSQFitter, TRFLSQFitter`
- Compound/transform usage in WCS notebook workflows:
  - `from astropy.modeling import models`
  - model composition for pipeline-like transformations
- Custom model definitions in educational contexts:
  - `Fittable1DModel`, `Parameter`

## Confidence and Limitations

Confidence is high that notebook/research-style repositories in `astropy` and
`spacetelescope` make broad use of `astropy.modeling`.

Limitations:

- Global cross-GitHub discovery is constrained by lexical-search scope.
- Some repositories with sparse indexing may be undercounted.
- The package/non-package boundary can be fuzzy for hybrid repos.

## Bottom Line

Outside library packages, `astropy.modeling` is widely used as the default
interactive modeling and fitting toolkit in astronomy notebook and workflow
repositories, especially for calibration, spectral fitting, PSF/shape modeling,
and WCS transform prototyping.
