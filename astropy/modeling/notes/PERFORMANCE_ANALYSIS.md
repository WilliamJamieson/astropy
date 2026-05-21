# astropy.modeling Performance Analysis

## Scope

This document analyzes performance and resource-usage hotspots in
`astropy.modeling`, focusing on model evaluation, fitting, import-time overhead,
and memory behavior.

The analysis is based on direct code inspection of:

- `astropy/modeling/core.py`
- `astropy/modeling/fitting.py`
- `astropy/modeling/parameters.py`
- `astropy/modeling/bounding_box.py`
- `astropy/modeling/_fitting_parallel.py`
- `astropy/modeling/convolution.py`
- `astropy/modeling/__init__.py`
- `astropy/modeling/models/__init__.py`

## Important Note About Runtime Measurements

I attempted to run local microbenchmarks in this worktree, but direct runtime
profiling was blocked by a local extension-build/import issue:
`ModuleNotFoundError: No module named 'astropy.io.votable.fast_converters'`.

As a result, this document is a code-informed performance analysis with
implementation-ready recommendations, rather than a measured benchmark report.

## Executive Summary

Primary likely causes of perceived slowness and high resource usage are:

1. High per-call overhead in the generic model evaluation pipeline
   (`Model.__call__`), even for simple models and "hot loop" usage.
2. Repeated parameter marshalling and array conversions in nonlinear fit
   objective/Jacobian paths.
3. Frequent array allocation/copying in input coercion, bounding box handling,
   weights handling, and constraint synchronization.
4. Expensive default deep-copy behavior in fitters (`inplace=False`), which can
   dominate cost for large or complex models.
5. Heavy import-time loading pattern in modeling namespace modules.
6. Missing focused performance benchmark suite for regression prevention.

## Findings

### 1) Evaluation Path Overhead Is High For Simple Calls

Relevant code:

- `core.py:1076` (`Model.__call__`)
- `core.py:1100` (`_get_renamed_inputs_as_positional`)
- `core.py:930` (`_pre_evaluate`)
- `core.py:2107` (`prepare_inputs` coercion to float arrays)
- `core.py:2125` (`_validate_input_units`)
- `core.py:2217` (`_process_output_units`)

Observed behavior:

- Every model call goes through a broad, feature-rich pipeline:
  kwargs-to-positional remapping, input broadcasting/validation,
  potential unit conversion, bounding-box dispatch, output shaping,
  and output-unit wrapping.
- `prepare_inputs` eagerly does `np.asanyarray(_input, dtype=float)` for every
  input (`core.py:2107`), which can trigger unnecessary copies and dtype
  conversion work.
- The generic path is valuable for correctness/flexibility, but expensive for
  high-frequency numeric use-cases (the exact use-case where users compare
  against custom kernels).

Impact:

- Increased Python overhead per call.
- Extra temporary arrays and allocations.
- Disproportionate cost for small/simple models in tight loops.

### 2) Fitting Performs Repeated Parameter Packing/Unpacking Work

Relevant code:

- `fitting.py:1141` (`_NonLinearLSQFitter.objective_function`)
- `fitting.py:1174` (residual evaluation)
- `fitting.py:2145` (`fitter_to_model_params_array`)
- `fitting.py:2206`/`2241` (`model.parameters = parameters` updates)
- `fitting.py:2244` (`model_to_fit_params`)
- `fitting.py:1204` (`_wrap_deriv`)

Observed behavior:

- Per optimizer iteration, parameters are repeatedly reconstructed from
  constrained/fitted subsets.
- Constraint support (`fixed`, `tied`, bounds) adds significant Python-level
  branching and object/property access.
- Jacobian wrapping performs repeated conversion patterns
  (`np.array(...)`, list comprehensions, `np.ravel(...)`, transpose/moveaxis)
  that are allocation-heavy.

Impact:

- Higher runtime in nonlinear fitting, especially with many parameters and/or
  many iterations.
- Increased memory churn due to temporary arrays.

### 3) Parameter Accessor Design Trades Convenience For Runtime Cost

Relevant code:

- `parameters.py:339` (`Parameter.value` getter)
- `parameters.py:363` (`Parameter.value` setter)
- `parameters.py:360` (`return np.float64(value)`)
- `parameters.py:372`/`374` (array conversion on assignment)
- `core.py:2723` (`_parameters_to_array`)
- `core.py:2738` (`_array_to_parameters`)
- `core.py:2800` (`_param_sets`)

Observed behavior:

- Parameter reads/writes frequently pass through conversion layers
  (including float64 normalization, optional getter/setter wrappers,
  and unit transforms).
- Fitting path relies on repeated synchronization between model parameter
  objects and flattened parameter arrays.

Impact:

- Higher Python overhead and more object/array conversion than a tightly packed
  numeric-core-only implementation.

### 4) Deep Copy Default In Fitters Is Safe But Expensive

Relevant code:

- `core.py:2295`/`2302` (`Model.copy()` -> `copy.deepcopy`)
- `fitting.py:1354` (nonlinear fitters default `inplace=False`)
- `fitting.py:1412` (`_validate_model(..., copy=not inplace)`)

Observed behavior:

- Most fitters default to returning a copy and therefore deep-copy models.
- Deep copies are robust and user-friendly, but costly for large compound
  models, large parameter arrays, and repeated fitting workflows.

Impact:

- Elevated memory usage and startup latency per fit call.

### 5) Bounding Box Enforcement Allocates Large Temporary Arrays

Relevant code:

- `bounding_box.py:315` (`np.zeros(input_shape) + fill_value`)
- `bounding_box.py:871` (`outside_index = np.zeros(input_shape, dtype=bool)`)
- `bounding_box.py:937` (`np.broadcast_to(...)[valid_index]`)

Observed behavior:

- Bounding-box evaluation builds full-size boolean masks and full-size output
  arrays (filled with `fill_value`) before reinserting valid regions.
- This is correct and general, but can be memory-heavy for large grids,
  especially when most points are outside domain.

Impact:

- Peak memory spikes and extra memory bandwidth.

### 6) Compound Model Evaluation Is Recursively Composed

Relevant code:

- `core.py:3460` (`CompoundModel._evaluate`)
- `core.py:3464`/`3466` (recursive left/right calls)

Observed behavior:

- Compound operations are implemented as compositional recursion, which is
  maintainable and extensible.
- Deep/chained compound models increase Python call overhead and intermediate
  tuple/list transformations.

Impact:

- Noticeable overhead for high-depth pipelines relative to fused custom code.

### 7) Parallel Dask Fit Path Can Inflate Memory Depending On Chunking

Relevant code:

- `_fitting_parallel.py:547` (`data.rechunk(...)`)
- `_fitting_parallel.py:334` (broadcasted value expansion in helper access)

Observed behavior:

- Rechunking and broadcast behavior are necessary for generality but can cause
  substantial temporary memory use for large cubes.

Impact:

- Resource pressure and scheduler overhead on large N-D fitting jobs.

### 8) Convolution Model Cache Is Useful But Potentially Large

Relevant code:

- `convolution.py` (`Convolution._get_convolution`, cache behavior)

Observed behavior:

- Caching interpolator state avoids repeated expensive work, but can retain
  large precomputed grids depending on domain/resolution.

Impact:

- Potentially high persistent memory footprint in long sessions.

### 9) Import-Time Cost Is High In Common Modeling Entry Points

Relevant code:

- `modeling/__init__.py:10` (`from . import fitting, models`)
- `modeling/models/__init__.py:9` (`from ...functional_models import *` plus
  multiple star imports)

Observed behavior:

- Importing common modeling namespaces eagerly imports large module sets and
  transitive dependencies.

Impact:

- Slower import/startup and larger baseline memory footprint.

### 10) Benchmark Coverage Gap

I found no dedicated, visible modeling-focused benchmark suite in this
worktree that tracks performance-sensitive paths such as:

- scalar vs vector model evaluation overhead
- unit-aware vs unitless call overhead
- compound model depth scaling
- fitting iteration cost by parameter count/constraint mix
- masked/model-set linear fit paths
- bounding-box memory behavior

Impact:

- Performance regressions are easier to introduce and harder to catch.

## Prioritized Improvement Suggestions

### Priority 0 (Low risk, immediate value)

1. Add an explicit "fast path" in `Model.__call__` for common pure-numeric
   cases:
   - no units
   - no `with_bounding_box`
   - no renamed-input kwargs
   - single model set
   - no output-unit wrapping needed

2. Optimize bounding-box array allocation:
   - replace `np.zeros(shape) + fill` with `np.full(shape, fill)` where
     appropriate.
   - avoid full-size output creation when all points are valid.

3. Make fitter docs/examples more explicit about `inplace=True` for performance
   workflows.

4. Add small internal memoization/precomputation for constraint metadata within
   a single fit invocation (parameter slices, fixed/tied masks, bounds arrays).

### Priority 1 (Moderate refactor, high likely payoff)

1. Reduce per-iteration allocations in nonlinear objective/Jacobian paths:
   - preallocate working buffers for residuals/Jacobians when shapes are fixed
   - replace repeated list/array/ravel conversions with direct ndarray ops
   - minimize transpose/moveaxis churn in `_wrap_deriv`

2. Refactor parameter marshalling:
   - precompute fit-parameter to model-parameter mapping tables once per fit
   - apply in-place updates into contiguous parameter arrays when possible
   - avoid repeated Python-level `getattr` for hot loops

3. Add optimized evaluator for compound models with arithmetic-only chains
   (limited operator subset), reducing recursion and tuple handling.

### Priority 2 (Broader architecture changes)

1. Import-time laziness:
   - move toward lazy exports for heavy model collections and fitting modules.

2. Optional accelerated backend(s):
   - consider opt-in JIT/vectorized backends for selected models or evaluation
     kernels.

3. Memory-aware dask fitting defaults:
   - improve heuristics around chunking and temporary array expansion.

## Detailed Recommendation Breakdown

This section expands the prioritized recommendations into concrete
implementation guidance.

### A) Fast Path For Numeric Model Evaluation

Target code:

- `astropy/modeling/core.py` (`Model.__call__`, `_pre_evaluate`,
  `prepare_inputs`, `_post_evaluate`)

What to implement:

1. Add a guarded fast path in `Model.__call__` that bypasses expensive generic
   handling when all of these are true:
   - positional args only
   - no units on model or inputs
   - no bounding box requested
   - single model (`len(self) == 1`)
   - no output unit conversion required
2. In this path, avoid:
   - renamed-input remapping
   - unit validation/conversion
   - repeated tuple/list output reshaping when unnecessary

Expected benefit:

- Lower per-call overhead for simple repeated evaluations.
- Fewer temporary allocations from conversion/broadcast machinery.

Risk and mitigation:

- Risk: behavior drift for edge cases.
- Mitigation: keep path strictly opt-in by hard guard conditions and add parity
  tests against current behavior for random input shapes.

Validation criteria:

- New benchmark shows improved call throughput for unitless 1D and 2D models.
- No changes in numerical output for existing evaluation tests.

### B) Reduce Bounding Box Allocation Pressure

Target code:

- `astropy/modeling/bounding_box.py` (`_base_output`, `_modify_output`,
  `_outside`, `prepare_inputs`)

What to implement:

1. Replace `np.zeros(input_shape) + fill_value` with `np.full(...)` in
   baseline output construction.
2. Add short-circuit handling to avoid creating full-size output buffers when
   all points are valid.
3. Minimize repeated `broadcast_to(...)[valid_index]` operations by reusing
   computed masks/indices where possible.

Expected benefit:

- Reduced peak memory and fewer large temporary arrays for large grids.

Risk and mitigation:

- Risk: corner-case shape behavior changes for scalar inputs.
- Mitigation: add dedicated tests for scalar, 1D, and multidimensional inputs
  with both all-in and all-out bounding scenarios.

Validation criteria:

- Lower peak memory in bounding-box benchmarks.
- Existing bounding-box correctness tests pass unchanged.

### C) Make In-Place Fitting A First-Class Performance Option

Target code/docs:

- `astropy/modeling/fitting.py` docstrings
- relevant user docs in `docs/modeling/`

What to implement:

1. Add explicit performance note to fitter docstrings highlighting that
   `inplace=True` avoids deep-copy overhead.
2. Update examples to show safe patterns for reusing model instances with
   in-place updates in iterative workflows.

Expected benefit:

- Immediate real-world speed/memory wins without core refactors.

Risk and mitigation:

- Risk: user confusion about mutation semantics.
- Mitigation: include clear warning that model parameters are modified in place.

Validation criteria:

- Documentation examples validated in CI.
- Reduced user confusion in issue reports about fitter allocation cost.

### D) Cache Constraint Metadata Within A Fit Call

Target code:

- `astropy/modeling/fitting.py` (`model_to_fit_params`,
  `fitter_to_model_params_array`, `_NonLinearLSQFitter.__call__`)

What to implement:

1. Build once-per-fit metadata object containing:
   - parameter slices
   - fixed/tied masks
   - bounds arrays
   - fit-parameter index mapping
2. Pass this metadata through objective and Jacobian wrappers to avoid
   recomputing lists and dictionary lookups on every optimizer iteration.

Expected benefit:

- Lower Python overhead in iterative fitting loops.

Risk and mitigation:

- Risk: stale metadata if constraints are mutated mid-fit.
- Mitigation: document constraints as immutable during a fit and assert this in
  debug mode where practical.

Validation criteria:

- Fitter benchmarks improve for medium/high-parameter models.
- Constraint-handling tests remain green.

### E) Cut Allocation Churn In Residual/Jacobian Computation

Target code:

- `astropy/modeling/fitting.py` (`objective_function`, `_wrap_deriv`)

What to implement:

1. Replace repeated `np.array([...])`, `np.ravel(...)`, transpose chains with
   shape-stable ndarray operations.
2. Preallocate and reuse residual/Jacobian work buffers when optimizer API
   allows stable shapes.
3. Keep conversion logic for `col_fit_deriv` but avoid repeated moveaxis and
   list construction.

Expected benefit:

- Faster nonlinear fitting and lower temporary memory pressure.

Risk and mitigation:

- Risk: subtle Jacobian orientation bugs.
- Mitigation: add explicit Jacobian-shape tests for `col_fit_deriv` True/False,
  with and without weights.

Validation criteria:

- Improved Jacobian benchmark timings.
- Numerical parity on fitter regression tests.

### F) Streamline Parameter Marshalling

Target code:

- `astropy/modeling/fitting.py`
- `astropy/modeling/core.py` (parameter array sync points)

What to implement:

1. Consolidate updates into contiguous parameter array operations where
   possible.
2. Avoid repeated attribute lookups for hot loops by prebinding parameter
   metadata.
3. Minimize full `model.parameters = ...` synchronization frequency when
   intermediate object-level updates are not needed.

Expected benefit:

- Reduced Python-object overhead and faster optimizer iteration throughput.

Risk and mitigation:

- Risk: tied-parameter semantics depend on current model state.
- Mitigation: preserve ordering guarantees for tied parameter evaluation and add
  targeted tests.

Validation criteria:

- Parameter-heavy fit benchmarks improve.
- Tied/fixed/bounds behavior remains unchanged.

### G) Optimize Compound Arithmetic Evaluation

Target code:

- `astropy/modeling/core.py` (`CompoundModel._evaluate`)

What to implement:

1. Add optional optimized path for arithmetic-only chains (`+`, `-`, `*`, `/`)
   when no units/bounding/renamed-input handling is needed.
2. Reduce recursion depth overhead by using iterative evaluation plan for
   eligible compound trees.

Expected benefit:

- Better scaling for deep arithmetic compound models.

Risk and mitigation:

- Risk: incompatibility with non-arithmetic operators (`|`, `&`, `fix_inputs`).
- Mitigation: limit optimization strictly to supported operator subset with
  fallback to current generic behavior.

Validation criteria:

- Compound depth benchmark shows improved scaling.
- Full compound-model test suite continues to pass.

### H) Reduce Import-Time Eagerness

Target code:

- `astropy/modeling/__init__.py`
- `astropy/modeling/models/__init__.py`

What to implement:

1. Introduce lazy import/export strategy for heavy modules while preserving
   public API behavior.
2. Avoid eager wildcard imports where possible.

Expected benefit:

- Faster startup and lower baseline memory.

Risk and mitigation:

- Risk: backward-compatibility issues for users relying on import side effects.
- Mitigation: phase rollout with compatibility checks and deprecation notes if
  needed.

Validation criteria:

- Import benchmarks improve for common entry points.
- API import tests continue to pass.

### I) Improve Dask Fitting Memory Heuristics

Target code:

- `astropy/modeling/_fitting_parallel.py`

What to implement:

1. Tune default chunking heuristics to reduce oversized rechunk-induced
   temporaries.
2. Document memory/performance tradeoffs for `preserve_native_chunks`,
   `chunk_n_max`, and scheduler choice.

Expected benefit:

- Better memory behavior in large-cube fitting use cases.

Risk and mitigation:

- Risk: changing defaults may degrade some workloads.
- Mitigation: keep configurable overrides and benchmark representative data
  shapes before changing defaults.

Validation criteria:

- Lower memory peaks in N-D parallel fit benchmarks.
- No regressions in correctness and diagnostics behavior.

## Suggested Benchmark Plan

Add a dedicated `astropy.modeling` benchmark suite (ASV or equivalent) with
tests for:

1. `Model.__call__` throughput:
   - scalar input
   - 1e3 / 1e5 vector inputs
   - kwargs vs positional
   - with/without quantities

2. Compound depth scaling:
   - chained unary/binary compound operations with depth 1, 3, 10, 30

3. Fitter iteration cost:
   - Gaussian1D fit with varying point counts
   - unconstrained vs fixed/tied/bounded fits
   - analytic derivative vs numerical derivative

4. Linear fitter special cases:
   - single model vs model set
   - masked vs unmasked
   - shared vs per-model weights

5. Bounding box memory/time:
   - mostly-inside vs mostly-outside domains
   - large 2D and 3D grids

6. Import-time benchmark:
   - `import astropy.modeling`
   - `from astropy.modeling import models`
   - cold-start memory footprint snapshot

For each benchmark record:

- wall time
- peak allocated memory (if available)
- allocation counts (if available)

## Practical Guidance For Users (Current State)

Until internal optimizations land, users can reduce cost by:

1. Passing plain ndarrays (not `Quantity`) in performance-critical loops when
   unit safety is not required at each call.
2. Reusing model instances and fitters across repeated calls.
3. Using `inplace=True` in fitter calls when safe in user workflows.
4. Avoiding deep compound chains when equivalent fused/custom expressions are
   possible.
5. Being deliberate about bounding-box usage on very large grids.

## Proposed Next Engineering Steps

1. Land benchmark suite first, to define baseline and regression guardrails.
2. Implement `Model.__call__` fast path and measure effect.
3. Optimize nonlinear objective/Jacobian allocation patterns.
4. Address import-time laziness once runtime hotspots are validated.
