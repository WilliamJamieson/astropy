# astropy.modeling Benchmarks

This directory contains a proposed runnable benchmark suite derived from
`astropy/modeling/notes/BENCHMARK_PROPOSAL.md`.

## Files

- `eval_basic.py`: scalar/vector/2D evaluation path benchmarks
- `compound.py`: compound model construction and depth/fanout evaluation
- `fitting.py`: linear and nonlinear fitter workloads (`inplace` included)
- `bounding_box.py`: bounded vs unbounded evaluation behavior
- `units.py`: unitless vs quantity evaluation
- `utility_custom.py`: common utility models and custom subclass evaluation
- `parameters.py`: constraint mutation and post-constraint evaluation
- `inverse_roundtrip.py`: inverse-only and forward+inverse round-trip workloads
- `compound_bbox.py`: selector-based compound bounding-box evaluation
- `jwst_compound.py`: JWST assign_wcs/assign_mtwcs-style large compound models,
  including copy/deepcopy benchmarks
- `common.py`: shared deterministic setup helpers

## Running (ASV)

From repository root (with ASV installed and extension modules built):

```bash
asv run --config asv.ci.conf.json --bench modeling
```

Or for a specific module:

```bash
asv run --config asv.ci.conf.json --bench modeling.eval_basic
```

Quick CI subset (faster runtime, broad coverage):

```bash
asv run --config asv.ci.conf.json --bench "modeling\.(eval_basic|compound|bounding_box|units|utility_custom\.TimeUtilityModels|fitting\.TimeLinearFitting|parameters|inverse_roundtrip|compound_bbox)"
```

Selection presets are defined in:

- `benchmarks/modeling/selections.py`

Full modeling suite:

```bash
asv run --config asv.ci.conf.json --bench modeling
```

## Notes

- Benchmarks are deterministic (fixed RNG seeds).
- Input sizes are tuned for CI-friendliness while preserving scaling sensitivity.
- The local worktree may require building C extensions before direct runtime
  smoke tests succeed.
