from astropy.wcs import WCS
from astropy.wcs.wcsapi import BaseLowLevelWCS, protocols


class TestLowLevelWCS:
    def test_base(self):
        """
        Test that the abstract base class for low-level wcs objects f
        """

        class TestLowLevelWCS(BaseLowLevelWCS):
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

        assert isinstance(TestLowLevelWCS(), protocols.LowLevelWCS)

    def test_wcs(self):
        """Test the WCS class is a low-level WCS object"""
        assert isinstance(WCS(), protocols.LowLevelWCS)
