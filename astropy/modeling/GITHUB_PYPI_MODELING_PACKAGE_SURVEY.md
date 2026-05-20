# GitHub + PyPI Survey: Packages Similar to astropy.modeling

Date: 2026-05-20

## Scope

This survey lists Python packages that overlap with core `astropy.modeling` use cases:

- defining analytical or parametric models
- fitting model parameters to data
- composing model components into larger workflows

Each entry includes links to both GitHub and PyPI plus a brief description.

## Package Survey

| Package | GitHub | PyPI | Brief description | Notes vs astropy.modeling |
|---|---|---|---|---|
| lmfit | https://github.com/lmfit/lmfit-py | https://pypi.org/project/lmfit/ | High-level nonlinear least-squares fitting toolkit built on SciPy, with rich parameter constraints and fit reports. | Strong alternative for fitting ergonomics and parameter management; less focused on model composition syntax and astronomy-specific transform pipelines. |
| symfit | https://github.com/tBuLi/symfit | https://pypi.org/project/symfit/ | Symbolic fitting framework that uses SymPy expressions to generate constrained numerical fits. | Useful when symbolic model definitions are central; smaller ecosystem and lower activity than major scientific stacks. |
| sherpa | https://github.com/sherpa/sherpa | https://pypi.org/project/sherpa/ | Modeling and fitting package from the Chandra ecosystem with broad optimization/statistical tooling for scientific data analysis. | Strong for astronomy fitting and statistical workflows; heavier framework style than `astropy.modeling` for general-purpose model composition. |
| iminuit | https://github.com/scikit-hep/iminuit | https://pypi.org/project/iminuit/ | Python interface to CERN Minuit2 optimizers, widely used for robust minimization and uncertainty estimation. | Excellent optimizer backend and confidence interval tooling; usually paired with external model definitions rather than providing a full model-class framework. |
| SciPy | https://github.com/scipy/scipy | https://pypi.org/project/scipy/ | Core scientific-computing library with optimization and curve-fitting primitives (`scipy.optimize`). | Foundational fitting backend used by many projects; lower-level API than `astropy.modeling` and no native model-composition abstraction. |
| statsmodels | https://github.com/statsmodels/statsmodels | https://pypi.org/project/statsmodels/ | Statistical modeling library focused on regression, time series, and inference with rich result objects. | Strong for statistical models and diagnostics; less aligned with transform-style model composition used in WCS/instrument pipelines. |
| emcee | https://github.com/dfm/emcee | https://pypi.org/project/emcee/ | Ensemble MCMC sampler for probabilistic parameter estimation. | Complements `astropy.modeling` for Bayesian posterior sampling after model definition; not a direct replacement for deterministic model classes. |
| zfit | https://github.com/zfit/zfit | https://pypi.org/project/zfit/ | TensorFlow-based fitting framework for scalable likelihood fitting (common in HEP). | Useful for large likelihood-driven fits; different ecosystem and workflow assumptions than typical Astropy usage. |

## Quick GitHub Adoption Snapshot

GitHub stars are included as a rough ecosystem signal (not a quality metric):

- SciPy: ~14.7k stars
- statsmodels: ~11.4k stars
- lmfit: ~1.2k stars
- iminuit: ~317 stars
- symfit: ~244 stars
- sherpa: ~164 stars

## Adoption Ranking (Survey Set)

Ranking below is based on GitHub stars for the surveyed projects, as a simple
visibility/adoption proxy:

1. SciPy
2. statsmodels
3. lmfit
4. iminuit
5. symfit
6. sherpa

PyPI download telemetry was partially available during data collection:

- statsmodels (PyPIStats): very high monthly download volume
- sherpa (PyPIStats): modest monthly download volume
- lmfit/symfit/iminuit/scipy: PyPIStats rate-limited or unavailable in this run

## API Usability Comparison (Pros/Cons)

### astropy.modeling (baseline)

Pros:

- Clean model-class abstraction with reusable parameters and constraints.
- Strong model composition semantics for transform pipelines (`|`, `&`).
- Good alignment with astronomy workflows and units-aware use cases.

Cons:

- Less specialized for advanced statistical diagnostics/inference workflows.
- Does not bundle a full Bayesian sampling workflow out of the box.

### lmfit

Pros:

- Very ergonomic parameter API (`Parameters`) with bounds, ties, and reports.
- High productivity for nonlinear least-squares fitting tasks.

Cons:

- Weaker native model-composition pipeline abstractions than astropy.modeling.
- Less astronomy-transform oriented than Astropy/GWCS-style stacks.

### symfit

Pros:

- Symbolic model definitions can improve readability and constraint expression.
- Natural fit when users already build models in SymPy.

Cons:

- Smaller ecosystem/community footprint.
- Symbolic-first workflows can be heavier than purely numerical APIs.

### sherpa

Pros:

- Mature fitting/statistics framework with astronomy heritage.
- Broad optimizer/statistical options in one toolkit.

Cons:

- API style is more framework-heavy than lightweight model classes.
- Less commonly used as a generic composition-first transform layer.

### iminuit

Pros:

- Excellent minimization behavior and uncertainty estimation in practice.
- Highly effective for difficult objective landscapes.

Cons:

- Primarily an optimizer layer, not a full model-definition framework.
- Usually requires pairing with another library for model objects.

### SciPy

Pros:

- Ubiquitous numerical optimization/fitting primitives.
- Stable, widely known APIs with broad ecosystem interoperability.

Cons:

- Lower-level APIs require more boilerplate for model parameter management.
- No native equivalent of astropy.modeling model composition classes.

### statsmodels

Pros:

- Excellent statistical model coverage and diagnostics.
- Rich result objects for inference, tests, and reporting.

Cons:

- Focused on statistical/econometric models rather than transform composition.
- Not a direct substitute for WCS-style model graph construction.

### emcee

Pros:

- Strong Bayesian posterior sampling tooling.
- Easy to layer on top of existing deterministic model definitions.

Cons:

- Not a deterministic fitting/model-definition framework by itself.
- Requires additional setup for likelihood/prior engineering.

### zfit

Pros:

- Likelihood-centric and scalable workflows for large fit problems.
- TensorFlow backend can help with performance and autodiff workflows.

Cons:

- Heavier stack and different mental model than common Astropy workflows.
- Less of a drop-in replacement for `astropy.modeling` in astronomy pipelines.

## What Other Packages Do That astropy.modeling Does Not (or Not as Directly)

- statsmodels:
	- Built-in statistical inference and diagnostics depth (hypothesis tests,
		robust covariance options, econometric/time-series model families) that is
		beyond the core scope of `astropy.modeling`.
- emcee:
	- Native ensemble MCMC Bayesian posterior sampling workflows; Astropy users
		typically integrate external samplers for this.
- iminuit:
	- Minuit2-specific minimization/error-analysis patterns (e.g., MINOS-like
		workflows) not provided as first-class features in `astropy.modeling`.
- symfit:
	- Symbolic-first model specification and symbolic constraint handling as the
		primary API surface.
- sherpa:
	- End-to-end domain-specific fitting/statistics environment with a broad set
		of built-in statistical approaches.
- zfit:
	- TensorFlow-native likelihood modeling with differentiable programming
		patterns and HEP-style fit workflows.

## What astropy.modeling Does Especially Well

- Composable transform/model graphs for astronomy calibration and WCS pipelines.
- Integration style that aligns naturally with Astropy ecosystem components.
- Clear model object semantics that are practical for instrument pipeline code.

## Capability Matrix (Quick View)

Legend:

- Yes: first-class support
- Partial: supported, but usually via external wiring or narrower scope
- No: not a core focus

| Package | Model composition | Parameter constraints | Bayesian inference | Symbolic modeling | Statistical diagnostics depth | WCS/instrument pipeline fit |
|---|---|---|---|---|---|---|
| astropy.modeling | Yes | Yes | Partial | No | Partial | Yes |
| lmfit | Partial | Yes | Partial | No | Partial | Partial |
| symfit | Partial | Yes | Partial | Yes | Partial | No |
| sherpa | Partial | Yes | Partial | No | Yes | Partial |
| iminuit | No | Partial | Partial | No | Partial | Partial |
| SciPy | No | Partial | Partial | No | Partial | Partial |
| statsmodels | Partial | Partial | Partial | No | Yes | No |
| emcee | No | No | Yes | No | Partial | No |
| zfit | Partial | Yes | Partial | No | Partial | No |

Notes on interpretation:

- `astropy.modeling` is strongest when the problem is model/transform composition
	inside astronomy workflows.
- `statsmodels` and `sherpa` provide broader built-in statistical analysis than
	`astropy.modeling`.
- `emcee` and `iminuit` are best viewed as complementary engines for inference
	and optimization rather than drop-in replacements for model class APIs.

## Recommended Stacks by Use Case

### 1. Astronomy calibration and WCS pipelines

Recommended stack:

- `astropy.modeling` + SciPy (+ optional `iminuit`)

Why:

- Keep `astropy.modeling` as the model/transform composition layer.
- Use SciPy fitters/optimizers for broad numerical method coverage.
- Add `iminuit` when objective landscapes are difficult or robust uncertainty
	estimates are needed.

Best fit for:

- Instrument calibration pipelines
- WCS transform graphs
- Production astronomy reduction workflows

### 2. Optimization-heavy deterministic fitting

Recommended stack:

- `lmfit` + `iminuit` (+ SciPy)

Why:

- `lmfit` provides highly ergonomic parameter constraints and reporting.
- `iminuit` improves minimization reliability and error estimation in many
	hard nonlinear problems.
- SciPy remains useful for fallback solvers and specialized routines.

Best fit for:

- Complex nonlinear least-squares models
- Teams prioritizing fitting productivity and fit diagnostics
- Workflows not centered on WCS-style transform composition

### 3. Bayesian and inference-centric modeling

Recommended stack:

- `astropy.modeling` or `lmfit` + `emcee` (+ `statsmodels` when diagnostics are required)

Why:

- Use `astropy.modeling`/`lmfit` to define deterministic model and constraints.
- Use `emcee` to sample posterior distributions and propagate uncertainty.
- Add `statsmodels` where richer statistical tests/diagnostics are part of the
	deliverable.

Best fit for:

- Parameter posterior estimation
- Uncertainty quantification for publication-grade analysis
- Projects combining physical modeling with statistical inference

## Reasonable Additions for astropy.modeling

This section focuses on features that appear missing relative to the surveyed
ecosystem and that could be added incrementally without turning
`astropy.modeling` into a full statistics framework.

### 1. Better uncertainty and interval estimation interfaces

Current gap:

- No first-class, unified API for confidence intervals/profile-likelihood style
	outputs comparable to workflows users get from `iminuit`-style tooling.

Reasonable addition:

- Add an optional uncertainty results object and helper APIs that standardize
	parameter interval reporting across fitters.

Why it is realistic:

- It can be layered on existing fitter outputs without replacing current
	fitting backends.

### 2. First-class posterior sampling integration hooks

Current gap:

- Bayesian workflows are common, but users must manually bridge model
	parameters to external samplers such as `emcee`.

Reasonable addition:

- Provide a lightweight adapter API for converting model + constraints into a
	sampler-ready parameter vector and back.

Why it is realistic:

- This can remain optional and backend-agnostic, avoiding hard dependencies.

### 3. Higher-level parameter ergonomics

Current gap:

- Parameter workflows are powerful but verbose in complex multi-parameter fits
	compared to `lmfit` convenience patterns.

Reasonable addition:

- Add convenience helpers for grouped parameter updates, tied-parameter
	templates, and easier constraint summaries.

Why it is realistic:

- These are additive helper utilities over existing `Parameter` semantics.

### 4. Standardized fit diagnostics summary

Current gap:

- Post-fit diagnostics and reporting are less consolidated than in
	`statsmodels`-style result APIs.

Reasonable addition:

- Add a common fit summary object (residual metrics, fit status, covariance
	availability, warning flags) shared by Astropy fitters.

Why it is realistic:

- Much of this data already exists in fitter outputs and can be normalized.

### 5. Improved symbolic-to-numeric model pathway

Current gap:

- No built-in symbolic model authoring surface similar to `symfit` for users
	who prototype analytically before numerical fitting.

Reasonable addition:

- Add optional utilities to ingest simple SymPy expressions into generated
	Astropy model classes for common scalar/low-dimensional cases.

Why it is realistic:

- This can be scoped narrowly and implemented as an optional bridge utility.

### 6. More explicit plugin interfaces for external optimizers/samplers

Current gap:

- Integrating external engines is possible but not consistently ergonomic.

Reasonable addition:

- Define a small plugin protocol for optimizer/sampler adapters so external
	projects can provide robust integrations without internal patching.

Why it is realistic:

- `astropy.modeling` already has stable model semantics; a plugin contract would
	formalize integration points rather than redesign internals.

## Suggested Priority Order

1. Standardized fit diagnostics summary
2. Better uncertainty and interval estimation interfaces
3. Higher-level parameter ergonomics
4. Explicit optimizer/sampler plugin interfaces
5. Posterior sampling integration hooks
6. Symbolic-to-numeric bridge utilities

Rationale:

- The top items improve daily user workflow with low architectural risk.
- Mid-tier items improve ecosystem interoperability.
- Symbolic integration is valuable but should stay optional and tightly scoped.

## Mini Roadmap Table (Effort/Risk/Impact)

Scale:

- Effort: S (small), M (medium), L (large)
- Risk: Low, Medium, High
- Impact: Low, Medium, High

| Addition | Effort | Risk | Impact | Dependencies | Success criteria |
|---|---|---|---|---|---|
| Standardized fit diagnostics summary | M | Low | High | Common fitter result schema | Users can retrieve a uniform diagnostics object across major fitters. |
| Uncertainty and interval estimation interfaces | M | Medium | High | Diagnostics schema, covariance availability checks | Confidence-interval APIs exist with consistent output format for supported fitters. |
| Higher-level parameter ergonomics | S-M | Low | Medium-High | Existing `Parameter` API | Fewer lines of code for common constraint/tie workflows without behavior regressions. |
| Optimizer/sampler plugin interfaces | M | Medium | High | Stable adapter protocol and tests | External optimizer/sampler adapters integrate without internal patching. |
| Posterior sampling integration hooks | M | Medium | Medium-High | Plugin interfaces, parameter vector adapter | Round-trip model <-> sampler parameter mapping works for representative models. |
| Symbolic-to-numeric bridge utilities | M-L | Medium | Medium | Optional SymPy bridge, codegen validation | Simple symbolic expressions convert into valid Astropy models with tested evaluation parity. |

## Mini Roadmap (Phased)

### Phase 1: Foundation and quick wins

Target additions:

- Standardized fit diagnostics summary
- Higher-level parameter ergonomics

Why first:

- High user value with low-to-moderate implementation risk.
- Creates reusable primitives for later uncertainty and plugin work.

Deliverables:

- Shared fit-result summary object and docs
- Constraint/tie convenience helpers and examples

### Phase 2: Uncertainty standardization

Target additions:

- Uncertainty and interval estimation interfaces

Why second:

- Builds on common diagnostics representation from Phase 1.
- Addresses a major ecosystem gap relative to optimizer-focused tooling.

Deliverables:

- Confidence-interval helper API
- Unified interval output structure across supported fitters

### Phase 3: Interoperability layer

Target additions:

- Optimizer/sampler plugin interfaces
- Posterior sampling integration hooks

Why third:

- Enables stronger ecosystem integration while preserving core Astropy model
	semantics.
- Lets external projects provide maintained adapters.

Deliverables:

- Adapter protocol documentation
- Reference integrations (for example, one optimizer and one sampler path)

### Phase 4: Optional symbolic bridge

Target additions:

- Symbolic-to-numeric bridge utilities

Why fourth:

- Valuable for a subset of users but less central than diagnostics,
	uncertainty, and interoperability work.
- Easier to scope safely after plugin and adapter patterns are established.

Deliverables:

- Narrow, optional SymPy bridge for simple expressions
- Validation tests for generated-model correctness

## Suggested Sequencing Milestones

1. Milestone A: release diagnostics summary + parameter ergonomics helpers
2. Milestone B: release uncertainty/interval API with fitter coverage notes
3. Milestone C: release plugin protocol and one reference external adapter
4. Milestone D: release optional symbolic bridge prototype

## Notes

- This is a practical ecosystem survey, not a strict one-to-one feature equivalence matrix.
- Some entries are direct alternatives (for model fitting APIs), while others are complementary (for optimization, probabilistic inference, or statistics).
- For astronomy pipelines, common real-world stacks combine multiple tools (for example: `astropy.modeling` + `iminuit`/`emcee`, or `astropy.modeling` + SciPy optimizers).
