import pytest

from astropy.wcs import WCS
from astropy.wcs.wcsapi import fitswcs, high_level_api, low_level_api, protocols


class ConcreteLowLevelWCS(low_level_api.BaseLowLevelWCS):
    """
    Create a concrete class that inherits from the abstract base class
    """

    @property
    def pixel_n_dim(self): ...

    @property
    def world_n_dim(self): ...

    @property
    def world_axis_physical_types(self): ...

    @property
    def world_axis_units(self): ...

    def pixel_to_world_values(self, *pixel_arrays): ...

    def world_to_pixel_values(self, *world_arrays): ...

    @property
    def world_axis_object_components(self): ...

    @property
    def world_axis_object_classes(self): ...


class ConcreteHighLevelWCS(high_level_api.BaseHighLevelWCS):
    @property
    def low_level_wcs(self): ...

    def pixel_to_world(self, *pixel_arrays): ...

    def world_to_pixel(self, *world_objects): ...


@pytest.mark.parametrize(
    "instance",
    [
        ConcreteLowLevelWCS(),
        fitswcs.SlicedLowLevelWCS(WCS(), 1),
        fitswcs.SlicedFITSWCS(WCS(), 1),
        fitswcs.FITSWCSAPIMixin(),
        WCS(),
    ],
)
def test_low_level_wcs(instance):
    """
    Test the low-level wcs objects follow the low level protocol
    """
    # Has to be instance check because of limitations of issubclass for protocols
    assert isinstance(instance, protocols.LowLevelWCS)


@pytest.mark.parametrize(
    "instance",
    [
        ConcreteHighLevelWCS(),
        high_level_api.HighLevelWCSMixin(),
        fitswcs.SlicedFITSWCS(WCS(), 1),
        fitswcs.FITSWCSAPIMixin(),
        WCS(),
    ],
)
def test_high_level_wcs(instance):
    """
    Test the high-level wcs objects follow the high level protocol
    """
    # Has to be instance check because of limitations of issubclass for protocols
    assert isinstance(instance, protocols.HighLevelWCS)
