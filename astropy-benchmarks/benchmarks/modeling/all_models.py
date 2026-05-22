"""Benchmarks that evaluate every built-in astropy.modeling model class."""

import numpy as np

from astropy import units as u
from astropy.modeling import (
    functional_models,
    mappings,
    math_functions,
    physical_models,
    polynomial,
    powerlaws,
    projections,
    rotations,
    spline,
    tabular,
)
from astropy.utils.compat.optional_deps import HAS_SCIPY

# Filter out projection base classes and non-instantiable helper objects.
_PROJ_TO_REMOVE = (
    [
        "Projection",
        "Pix2SkyProjection",
        "Sky2PixProjection",
        "Zenithal",
        "Conic",
        "Cylindrical",
        "PseudoCylindrical",
        "PseudoConic",
        "QuadCube",
        "HEALPix",
        "projcodes",
        "Pix2Sky_ZenithalPerspective",
    ]
    + [f"Pix2Sky_{code}" for code in projections.projcodes]
    + [f"Sky2Pix_{code}" for code in projections.projcodes]
)


def _projection_model_names():
    return [name for name in projections.__all__ if name not in _PROJ_TO_REMOVE]


def _iter_model_specs():
    specs = []

    specs.extend(("functional", name) for name in functional_models.__all__)
    specs.extend(
        ("math", name) for name in math_functions.__all__ if name != "ArctanhUfunc"
    )
    specs.extend(("physical", name) for name in physical_models.__all__)
    specs.extend(("powerlaw", name) for name in powerlaws.__all__)
    specs.extend(("projection", name) for name in _projection_model_names())

    specs.extend(
        (
            ("polynomial", "Chebyshev1D"),
            ("polynomial", "Hermite1D"),
            ("polynomial", "Legendre1D"),
            ("polynomial", "Polynomial1D"),
            ("polynomial", "Chebyshev2D"),
            ("polynomial", "Hermite2D"),
            ("polynomial", "Legendre2D"),
            ("polynomial", "Polynomial2D"),
            ("polynomial", "SIP"),
            ("polynomial", "InverseSIP"),
        )
    )

    specs.extend(
        (
            ("mapping", "Mapping"),
            ("mapping", "Identity"),
            ("mapping", "UnitsMapping"),
        )
    )

    specs.extend(
        (
            ("rotation", "RotateCelestial2Native"),
            ("rotation", "RotateNative2Celestial"),
            ("rotation", "EulerAngleRotation"),
            ("rotation", "RotationSequence3D"),
            ("rotation", "SphericalRotationSequence"),
            ("rotation", "Rotation2D"),
        )
    )

    if HAS_SCIPY:
        specs.extend((("tabular", "Tabular1D"), ("tabular", "Tabular2D")))
        specs.append(("spline", "Spline1D"))

    return sorted(specs)


def _instantiate_model(group, name):  # noqa: PLR0911
    if group == "functional":
        return getattr(functional_models, name)()
    if group == "math":
        return getattr(math_functions, name)()
    if group == "physical":
        return getattr(physical_models, name)()
    if group == "powerlaw":
        return getattr(powerlaws, name)()

    if group == "projection":
        cls = getattr(projections, name)
        if name == "AffineTransformation2D":
            return cls(matrix=[[1.0, 0.0], [0.0, 1.0]], translation=[0.0, 0.0])
        return cls()

    if group == "polynomial":
        cls = getattr(polynomial, name)
        if name in {"Chebyshev1D", "Hermite1D", "Legendre1D", "Polynomial1D"}:
            return cls(2)
        if name in {"Chebyshev2D", "Hermite2D", "Legendre2D"}:
            return cls(2, 2)
        if name == "Polynomial2D":
            return cls(2)
        if name == "SIP":
            return cls(crpix=(21.0, 23.0), a_order=2, b_order=3)
        if name == "InverseSIP":
            return cls(ap_order=2, bp_order=3)

    if group == "mapping":
        if name == "Mapping":
            return mappings.Mapping((1, 0))
        if name == "Identity":
            return mappings.Identity(2)
        if name == "UnitsMapping":
            return mappings.UnitsMapping(((u.m, None),))

    if group == "rotation":
        if name == "RotateCelestial2Native":
            return rotations.RotateCelestial2Native(12.0, 23.0, 34.0)
        if name == "RotateNative2Celestial":
            return rotations.RotateNative2Celestial(12.0, 23.0, 34.0)
        if name == "EulerAngleRotation":
            return rotations.EulerAngleRotation(12.0, 23.0, 34.0, "xyz")
        if name == "RotationSequence3D":
            return rotations.RotationSequence3D([12.0, 23.0, 34.0], axes_order="xyz")
        if name == "SphericalRotationSequence":
            return rotations.SphericalRotationSequence([12.0, 23.0, 34.0], "xyz")
        if name == "Rotation2D":
            return rotations.Rotation2D(12.0)

    if group == "tabular":
        if name == "Tabular1D":
            return tabular.Tabular1D(lookup_table=[1.0, 2.0, 3.0, 4.0])
        if name == "Tabular2D":
            return tabular.Tabular2D(lookup_table=[[1.0, 2.0], [3.0, 4.0]])

    if group == "spline" and name == "Spline1D":
        return spline.Spline1D(
            knots=[-5.0, -5.0, -5.0, -5.0, 0.0, 5.0, 5.0, 5.0, 5.0],
            coeffs=[0.0, 1.0, 0.5, -0.5, 0.0],
            degree=3,
        )

    raise RuntimeError(f"No instantiation rule for {group}:{name}")


def _evaluate_model(model):
    x = np.linspace(-0.8, 0.8, 32)
    y = np.linspace(-0.6, 0.6, 32)
    z = np.linspace(-0.4, 0.4, 32)

    if isinstance(model, mappings.UnitsMapping):
        return model(x * u.km)

    n_inputs = model.n_inputs
    if n_inputs == 1:
        return model(x)
    if n_inputs == 2:
        return model(x, y)
    if n_inputs == 3:
        return model(x, y, z)

    values = tuple(np.linspace(-1.0, 1.0, 32) for _ in range(n_inputs))
    return model(*values)


_MODEL_SPECS = _iter_model_specs()
_MODEL_LABELS = [f"{group}:{name}" for group, name in _MODEL_SPECS]


class TimeBuiltinModels:
    """Benchmark one call per built-in model class."""

    params = [_MODEL_LABELS]
    param_names = ["model"]

    def setup(self, model):
        group, name = model.split(":", 1)
        self.instance = _instantiate_model(group, name)

    def time_evaluate(self, model):
        _evaluate_model(self.instance)
