import importlib
import sys
from pathlib import Path

import pytest

from astropy.utils.exceptions import AstropyDeprecationWarning

MODELS_MODULES = [
    file_.name.replace(".py", "")
    for file_ in (Path(__file__).parents[1] / "models").glob("*.py")
    if "__init__" not in file_.name
]


@pytest.mark.parametrize("module", MODELS_MODULES)
def test_api_matches(module):
    """
    Check that we preserve the API of the old modules, but we raise a warning
    in the old modules.
    """
    new_module = importlib.import_module(f"astropy.modeling.models.{module}")

    # If the old module is already imported, delete it
    if (module_name := f"astropy.modeling.{module[1:]}") in sys.modules:
        del sys.modules[module_name]
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


@pytest.mark.parametrize("module", MODELS_MODULES)
def test_old_imports(module):
    """
    Several downstream libraries have the following pattern:

        import astropy.modeling

        foo = astropy.modeling.<old_models_module_name>

    Supporting this with a warning message requires us to do some funky things.

    Note that python only imports modules once unless we delete the module and
    retry the import.
    """

    # If the old module is already imported, delete it
    if (module_name := f"astropy.modeling.{module[1:]}") in sys.modules:
        del sys.modules[module_name]

    # If astropy.modeling is already imported, delete it
    if "astropy.modeling" in sys.modules:
        del sys.modules["astropy.modeling"]

    # Load astropy.modeling
    import astropy.modeling

    with pytest.warns(AstropyDeprecationWarning):
        # attempt to get the old module from astropy.modeling
        getattr(astropy.modeling, module[1:])
