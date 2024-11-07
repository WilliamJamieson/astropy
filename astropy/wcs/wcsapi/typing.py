from typing import TypeAlias, TypeVar

import numpy as np
from typing_extensions import Buffer  # Need 3.12 for this to be in stdlib

from astropy.coordinates import SkyCoord, SpectralCoord
from astropy.time import Time
from astropy.units import Quantity
from astropy.units.typing import Real

Interval: TypeAlias = tuple[Real, Real]

BooleanBuffer: TypeAlias = Buffer | np.typing.NDArray[bool]
ScalarBuffer: TypeAlias = Buffer | np.typing.NDArray[Real]
IndexBuffer: TypeAlias = Buffer | np.typing.NDArray[int]

ScalarArrays: TypeAlias = tuple[Real | ScalarBuffer, ...]
IndexArrays: TypeAlias = tuple[int | IndexBuffer, ...]

OutputCoords: TypeAlias = Real | ScalarBuffer | ScalarArrays
OutputIndex: TypeAlias = int | IndexBuffer | IndexArrays

# Think astropy object, but general protocol is less specific
_HighLevelObject = TypeVar("_HighLevelObject")

WorldAxisComponent: TypeAlias = tuple[str, str | int, str]
WorldAxisClass: TypeAlias = tuple[
    type | str, tuple[int | None, ...], dict[str, _HighLevelObject]
]

# These are the implemented types for the high-level WCS interface
#   but they can be anything from the world_axis_object_classes (hence the type var)
HighLevelObject: TypeAlias = (
    Time | SkyCoord | SpectralCoord | Quantity | _HighLevelObject
)
HighLevelOutput: TypeAlias = HighLevelObject | tuple[HighLevelObject]

ScalarOrNdarray: TypeAlias = Real | np.ndarray
IndexOrNdarray: TypeAlias = int | np.typing.NDArray[np.int_]

InputScalarOrNdarray: TypeAlias = tuple[ScalarOrNdarray, ...]
InputIndexOrNdarray: TypeAlias = tuple[IndexOrNdarray, ...]

OutputScalarOrNdarray: TypeAlias = ScalarOrNdarray | InputScalarOrNdarray
OutputIndexOrNdarray: TypeAlias = IndexOrNdarray | InputIndexOrNdarray

AstropyObject: TypeAlias = Time | SkyCoord | SpectralCoord | Quantity
AstropyInputs: TypeAlias = tuple[AstropyObject]
AstropyOutput: TypeAlias = AstropyObject | AstropyInputs
