"""Selection presets for astropy.modeling ASV benchmark runs.

These regex presets are intended for use with:
    asv run --config asv.ci.conf.json --bench <regex>
"""

# Lightweight subset intended for CI smoke/regression checks.
QUICK_CI_BENCH_REGEX = (
    r"modeling\\.(eval_basic|compound|bounding_box|units|"
    r"utility_custom\\.TimeUtilityModels|fitting\\.TimeLinearFitting|"
    r"parameters|inverse_roundtrip|compound_bbox)"
)

# Full modeling benchmark suite.
FULL_MODELING_BENCH_REGEX = r"modeling"
