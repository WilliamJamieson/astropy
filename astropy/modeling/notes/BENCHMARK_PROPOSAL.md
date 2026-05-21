# Benchmark Proposal for `astropy.modeling`

Date: 2026-05-21

## Goal

Define a benchmark suite that captures the key public API usage patterns of
`astropy.modeling` observed across:

- package pipelines (`gwcs`, `jwst`, `romancal`, `stdatamodels`, `photutils`)
- notebook/research workflows (interactive fitting and model prototyping)
- performance hotspots identified in `PERFORMANCE_ANALYSIS.md`

The proposal prioritizes **behavioral representativeness** (real API shapes) and
**regression sensitivity** (benchmarks likely to catch slowdowns early).

## Design Principles

1. Benchmark API workflows, not isolated helper internals.
2. Include both small-array and medium-array regimes.
3. Separate unitless and unit-aware paths.
4. Cover construction cost, evaluation cost, and fitting cost.
5. Track both time and memory where meaningful.
6. Keep each benchmark deterministic and cheap enough for CI.

## Core Usage Coverage Matrix

The suite should explicitly cover these feature groups.

| Feature group | Why it matters | Typical downstream pattern |
|---|---|---|
| Model evaluation contract (`__call__`, `evaluate`) | Universal entry point | Transform and photometry pipelines |
| Compound composition (`|`, `&`, arithmetic) | Backbone of WCS/model graphs | `Mapping | Polynomial2D | Shift` chains |
| Parameter handling and constraints | Critical to fit workflows | `fixed`, `bounds`, `tied` mutation and reads |
| Fitting (linear + nonlinear) | Core science workload | `LinearLSQFitter`, `TRFLSQFitter`, `LevMarLSQFitter` |
| Bounding boxes and masked evaluation | Heavy WCS use | `bind_bounding_box`, `with_bounding_box=True` |
| Inverse transforms | Bidirectional coordinate conversion | detector <-> world |
| Unit-aware operation | Common in spectroscopy/calibration | quantity inputs + equivalencies |
| Utility transforms | High prevalence in surveys | `Mapping`, `Identity`, `Tabular1D`, `Polynomial*` |
| Custom model subclasses | Ecosystem extension point | `Fittable1DModel`/`Model` subclasses |
| Model copy and fit `inplace` behavior | Known perf factor | default copy vs `inplace=True` |

## Proposed Benchmark Groups

## Group A: Evaluation Hot Path

### A1. Scalar and vector unitless evaluation

- Models: `Gaussian1D`, `Polynomial1D`, `Shift`, `Scale`
- Inputs:
  - scalar (`N=1`)
  - small vector (`N=64`)
  - medium vector (`N=4096`)
- Metrics:
  - `time_eval_scalar`, `time_eval_vector_small`, `time_eval_vector_medium`

Purpose: detect per-call overhead inflation in generic `Model.__call__` path.

### A2. 2D grid evaluation

- Models: `Gaussian2D`, `Polynomial2D`
- Shapes: `64x64`, `256x256`
- Metrics: wall time and peak memory

Purpose: detect array coercion/allocation regressions.

### A3. `evaluate()` vs `__call__()`

- Same models/data as A1
- Compare direct `evaluate(...)` and instance `__call__(...)`

Purpose: track overhead introduced by call pipeline wrappers.

## Group B: Compound Model Composition

### B1. Construction cost vs depth

- Build chains of depth 1, 5, 10, 20
- Pattern: `Mapping | Polynomial2D | Shift | Scale ...`
- Metrics: construction time, object size proxy (peak mem)

Purpose: measure model-graph build overhead highlighted in docs.

### B2. Evaluation cost vs depth

- Evaluate compound chains built in B1 on `N=4096` vector and `128x128` grid
- Metrics: wall time scaling by depth

Purpose: detect recursion and composition overhead regressions.

### B3. Parallel composition fanout

- Pattern: `(Identity(1) & Identity(1) & Identity(1)) | Mapping(...)`
- Fanout sizes: 2, 4, 8 branches

Purpose: capture `&` operator and tuple/shape plumbing costs.

## Group C: Parameter and Constraint Operations

### C1. Parameter read/write throughput

- Repeated reads/writes of model parameters
- Include `Parameter` getter/setter conversion case

Purpose: measure descriptor and conversion overhead.

### C2. Constraint mutation and sync

- Set/update `fixed`, `bounds`, `tied` across models with 3, 10, 30 params
- Metrics: mutation time and subsequent first-evaluation time

Purpose: capture constraint bookkeeping costs in realistic fit setup.

### C3. Model copy cost

- `copy()` and `deepcopy()` for simple and deep compound models

Purpose: quantify known fit-path overhead from copying.

## Group D: Fitting Workloads

### D1. Linear fit baseline

- `LinearLSQFitter` with `Linear1D`, `Polynomial1D` (order 3, 7)
- Data sizes: `N=256`, `N=4096`
- Metrics: fit time + residual quality sanity check

Purpose: stable baseline for common fast fits.

### D2. Nonlinear 1D fits

- `TRFLSQFitter` and `LevMarLSQFitter`
- Models: `Gaussian1D`, `Moffat1D`, compound `Gaussian1D + Const1D`
- Cases:
  - unconstrained
  - bounded parameters
  - tied parameter relation

Purpose: cover dominant nonlinear objective/jacobian paths.

### D3. Nonlinear 2D fits

- `TRFLSQFitter` with `Gaussian2D` on `64x64` and `128x128`
- Optional weights variant

Purpose: catch expensive marshalling and Jacobian reshaping regressions.

### D4. `inplace=False` vs `inplace=True`

- Run D1/D2 with both modes
- Metric: time and peak memory delta

Purpose: make copy-related performance impact measurable and guardrail-able.

## Group E: Bounding Box and Domain-Limited Evaluation

### E1. Bounding box evaluation fractions

- Attach box with `bind_bounding_box`
- Evaluate same grid with inside fractions ~100%, ~50%, ~10%
- Compare with and without `with_bounding_box=True`

Purpose: measure mask/allocation behavior under sparse validity.

### E2. Compound bounding box

- Use `CompoundBoundingBox` on a composed transform
- Evaluate on medium grid

Purpose: represent advanced WCS-domain patterns.

## Group F: Units and Equivalencies

### F1. Quantity overhead in evaluation

- Unitless vs Quantity input for same model/data sizes
- Models: `Gaussian1D`, `Polynomial1D`, simple compound chain

Purpose: track unit handling overhead and regressions.

### F2. Equivalencies path

- Evaluate with relevant equivalency set on spectral-like inputs

Purpose: catch overhead shifts in conversion/equivalency handling.

## Group G: Utility and Custom-Model Ecosystem Patterns

### G1. High-frequency utility models

- `Mapping`, `Identity`, `Tabular1D`, `Polynomial2D`, `RotationSequence3D`
- Focus on construction + repeated evaluation

Purpose: directly cover classes most frequently observed in survey notes.

### G2. Custom subclass benchmarks

- Minimal `Fittable1DModel` subclass (instance `evaluate`)
- Minimal `Model` subclass with `@staticmethod evaluate`
- Optional subclass with parameter getter/setter conversion

Purpose: guard extension-point performance for downstream packages.

### G3. Inverse usage

- Benchmark forward-only, inverse-only, and forward+inverse round-trip

Purpose: cover bidirectional transform workflows.

## Benchmark Priority Tiers

## Tier 1 (must land first)

- A1, A2
- B1, B2
- D1, D2, D4
- E1
- F1
- G1

These represent the best balance of ecosystem relevance and regression
sensitivity.

## Tier 2 (next)

- A3, B3
- C1, C2, C3
- D3
- E2
- F2
- G2, G3

These add coverage depth once Tier 1 is stable.

## Suggested Benchmark Module Layout

If implemented with ASV-style layout in-repo:

- `benchmarks/modeling/eval_basic.py` (Group A)
- `benchmarks/modeling/compound.py` (Group B)
- `benchmarks/modeling/parameters.py` (Group C)
- `benchmarks/modeling/fitting.py` (Group D)
- `benchmarks/modeling/bounding_box.py` (Group E)
- `benchmarks/modeling/units.py` (Group F)
- `benchmarks/modeling/utility_custom.py` (Group G)

If Astropy keeps benchmarks external, preserve this grouping as logical modules.

## Measurement Conventions

- Use deterministic RNG seeds.
- Warm-up once during `setup` for JIT-less stability.
- Store synthetic datasets in `setup_cache` where possible.
- Prefer `time_*` and `peakmem_*` metrics.
- Keep each benchmark under ~2 seconds locally for CI usability.

## Acceptance Criteria

A benchmark proposal is considered implemented successfully when:

1. Tier 1 suite exists and runs in CI/asv locally.
2. Each key API feature group in the coverage matrix has at least one benchmark.
3. At least one memory benchmark exists for bounding-box and fitting copy paths.
4. A short benchmark guide documents how to run and interpret regressions.

## Why this covers key API features

This proposal covers all repeatedly observed API surfaces from the notes:

- call/evaluate contract
- model composition operators
- high-usage concrete model classes (`Mapping`, `Identity`, `Polynomial2D`, `Shift`, `Scale`, `Tabular1D`)
- fitting and constraints
- bounding boxes (`bind_bounding_box`, `CompoundBoundingBox`)
- units and equivalencies
- custom model subclass extension points
- inverse transform workflows

Together, these benchmarks should catch regressions in both mainstream
astronomy pipelines and interactive notebook-style usage.
