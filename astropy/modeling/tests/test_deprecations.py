import importlib
from pathlib import Path

import pytest

from astropy.utils.exceptions import AstropyDeprecationWarning

MODELS_MODULES = [
    file_.name.replace(".py", "")
    for file_ in (Path(__file__).parents[1] / "models").glob("*.py")
    if "__init__" not in file_.name
]


@pytest.mark.parametrize("module", MODELS_MODULES)
def test_test(module):
    new_module = importlib.import_module(f"astropy.modeling.models.{module}")
    with pytest.warns(AstropyDeprecationWarning):
        old_module = importlib.import_module(f"astropy.modeling.{module[1:]}")

    # Check that the API is the same
    for this in new_module.__all__:
        assert this in old_module.__all__
    for this in old_module.__all__:
        assert this in new_module.__all__

    # Check warning for individual imports
    for this in new_module.__all__:
        with pytest.warns(AstropyDeprecationWarning):
            getattr(old_module, this)
