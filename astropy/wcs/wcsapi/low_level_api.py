import abc
import os

import numpy as np

from .protocols import LowLevelWCS, scalar

__all__ = ["BaseLowLevelWCS", "validate_physical_types"]

scalar_or_ndarray = scalar | np.ndarray
index_or_ndarray = int | np.typing.NDArray[np.int_]

input_scalar_or_ndarray = tuple[scalar_or_ndarray, ...]
input_index_or_ndarray = tuple[index_or_ndarray, ...]

output_scalar_or_ndarray = scalar_or_ndarray | input_scalar_or_ndarray
output_index_or_ndarray = index_or_ndarray | input_index_or_ndarray


class BaseLowLevelWCS(LowLevelWCS, metaclass=abc.ABCMeta):
    """
    Abstract base class for the low-level astropy-WCS interface.

    See the `~astropy.wcs.wcsapi.protocols.LowLevelWCS` protocol for the complete
    description of the methods and properties that need to be implemented. This
    class provides default implementations for some properties and specific
    type hints for the methods usage in astropy.
    """

    @abc.abstractmethod
    def pixel_to_world_values(
        self, *pixel_arrays: input_scalar_or_ndarray
    ) -> output_scalar_or_ndarray:
        """
        Convert pixel coordinates to world coordinates.

        This method takes `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim` scalars or arrays as
        input, and pixel coordinates should be zero-based. Returns
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim` scalars or arrays in units given by
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_units`. Note that pixel coordinates are
        assumed to be 0 at the center of the first pixel in each dimension. If a
        pixel is in a region where the WCS is not defined, NaN should be returned.
        The coordinates should be specified in the ``(x, y)`` order, where for
        an image, ``x`` is the horizontal coordinate and ``y`` is the vertical
        coordinate.

        If `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim` is ``1``, this
        method returns a single scalar or array, otherwise a tuple of scalars or
        arrays is returned.
        """

    def array_index_to_world_values(
        self, *index_arrays: input_index_or_ndarray
    ) -> output_scalar_or_ndarray:
        """
        Convert array indices to world coordinates.

        This is the same as `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_to_world_values` except that
        the indices should be given in ``(i, j)`` order, where for an image
        ``i`` is the row and ``j`` is the column (i.e. the opposite order to
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_to_world_values`).

        If `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim` is ``1``, this
        method returns a single scalar or array, otherwise a tuple of scalars or
        arrays is returned.
        """
        return self.pixel_to_world_values(*index_arrays[::-1])

    @abc.abstractmethod
    def world_to_pixel_values(
        self, *world_arrays: input_scalar_or_ndarray
    ) -> output_scalar_or_ndarray:
        """
        Convert world coordinates to pixel coordinates.

        This method takes `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim` scalars or arrays as
        input in units given by `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_units`. Returns
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim` scalars or arrays. Note that pixel
        coordinates are assumed to be 0 at the center of the first pixel in each
        dimension. If a world coordinate does not have a matching pixel
        coordinate, NaN should be returned.  The coordinates should be returned in
        the ``(x, y)`` order, where for an image, ``x`` is the horizontal
        coordinate and ``y`` is the vertical coordinate.

        If `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim` is ``1``, this
        method returns a single scalar or array, otherwise a tuple of scalars or
        arrays is returned.
        """

    def world_to_array_index_values(
        self, *world_arrays: input_scalar_or_ndarray
    ) -> output_index_or_ndarray:
        """
        Convert world coordinates to array indices.

        This is the same as `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_to_pixel_values` except that
        the indices should be returned in ``(i, j)`` order, where for an image
        ``i`` is the row and ``j`` is the column (i.e. the opposite order to
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_to_world_values`). The indices should be
        returned as rounded integers.

        If `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim` is ``1``, this
        method returns a single scalar or array, otherwise a tuple of scalars or
        arrays is returned.
        """
        pixel_arrays = self.world_to_pixel_values(*world_arrays)
        if self.pixel_n_dim == 1:
            pixel_arrays = (pixel_arrays,)
        else:
            pixel_arrays = pixel_arrays[::-1]
        array_indices = tuple(
            np.asarray(np.floor(pixel + 0.5), dtype=int) for pixel in pixel_arrays
        )
        return array_indices[0] if self.pixel_n_dim == 1 else array_indices

    # The following three properties have default fallback implementations, so
    # they are not abstract.

    @property
    def array_shape(self) -> tuple[int, ...]:
        """
        The shape of the data that the WCS applies to as a tuple of length
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim` in ``(row, column)``
        order (the convention for arrays in Python).

        If the WCS is valid in the context of a dataset with a particular
        shape, then this property can be used to store the shape of the
        data. This can be used for example if implementing slicing of WCS
        objects. This is an optional property, and it should return `None`
        if a shape is not known or relevant.
        """
        if self.pixel_shape is None:
            return None
        else:
            return self.pixel_shape[::-1]

    @property
    def pixel_shape(self) -> None:
        """
        The shape of the data that the WCS applies to as a tuple of length
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim` in ``(x, y)``
        order (where for an image, ``x`` is the horizontal coordinate and ``y``
        is the vertical coordinate).

        If the WCS is valid in the context of a dataset with a particular
        shape, then this property can be used to store the shape of the
        data. This can be used for example if implementing slicing of WCS
        objects. This is an optional property, and it should return `None`
        if a shape is not known or relevant.

        If you are interested in getting a shape that is comparable to that of
        a Numpy array, you should use
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.array_shape` instead.
        """
        return None

    @property
    def pixel_bounds(self) -> None:
        """
        The bounds (in pixel coordinates) inside which the WCS is defined,
        as a list with `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim`
        ``(min, max)`` tuples.

        The bounds should be given in ``[(xmin, xmax), (ymin, ymax)]``
        order. WCS solutions are sometimes only guaranteed to be accurate
        within a certain range of pixel values, for example when defining a
        WCS that includes fitted distortions. This is an optional property,
        and it should return `None` if a shape is not known or relevant.

        The bounds can be a mix of values along dimensions where bounds exist,
        and None for other dimensions, e.g. ``[(xmin, xmax), None]``.
        """
        return None

    @property
    def pixel_axis_names(self) -> list[str]:
        """
        A list of strings describing the name for each pixel axis.

        If an axis does not have a name, an empty string should be returned
        (this is the default behavior for all axes if a subclass does not
        override this property). Note that these names are just for display
        purposes and are not standardized.
        """
        return [""] * self.pixel_n_dim

    @property
    def world_axis_names(self) -> list[str]:
        """
        A list of strings describing the name for each world axis.

        If an axis does not have a name, an empty string should be returned
        (this is the default behavior for all axes if a subclass does not
        override this property). Note that these names are just for display
        purposes and are not standardized. For standardized axis types, see
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_physical_types`.
        """
        return [""] * self.world_n_dim

    @property
    def axis_correlation_matrix(self) -> np.typing.NDArray[np.bool_]:
        """
        Returns an (`~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim`,
        `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_n_dim`) matrix that
        indicates using booleans whether a given world coordinate depends on a
        given pixel coordinate.

        This defaults to a matrix where all elements are `True` in the absence
        of any further information. For completely independent axes, the
        diagonal would be `True` and all other entries `False`.
        """
        return np.ones((self.world_n_dim, self.pixel_n_dim), dtype=bool)

    @property
    def serialized_classes(self) -> bool:
        """
        Indicates whether Python objects are given in serialized form or as
        actual Python objects.
        """
        return False

    def _as_mpl_axes(self):
        """Compatibility hook for Matplotlib and WCSAxes.

        With this method, one can do::

            from astropy.wcs import WCS
            import matplotlib.pyplot as plt
            wcs = WCS('filename.fits')
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.8, 0.8], projection=wcs)
            ...

        and this will generate a plot with the correct WCS coordinates on the
        axes.
        """
        from astropy.visualization.wcsaxes import WCSAxes

        return WCSAxes, {"wcs": self}


UCDS_FILE = os.path.join(os.path.dirname(__file__), "data", "ucds.txt")
with open(UCDS_FILE) as f:
    VALID_UCDS = {x.strip() for x in f.read().splitlines()[1:]}


def validate_physical_types(physical_types: str) -> None:
    """
    Validate a list of physical types against the UCD1+ standard.
    """
    for physical_type in physical_types:
        if (
            physical_type is not None
            and physical_type not in VALID_UCDS
            and not physical_type.startswith("custom:")
        ):
            raise ValueError(
                f"'{physical_type}' is not a valid IOVA UCD1+ physical type. It must be"
                " a string specified in the list"
                " (http://www.ivoa.net/documents/latest/UCDlist.html) or if no"
                " matching type exists it can be any string prepended with 'custom:'."
            )
