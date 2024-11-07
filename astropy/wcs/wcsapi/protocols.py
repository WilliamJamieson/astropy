from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .typing import (
        BooleanBuffer,
        HighLevelObject,
        HighLevelOutput,
        IndexArrays,
        Interval,
        OutputCoords,
        OutputIndex,
        ScalarArrays,
        WorldAxisClass,
        WorldAxisComponent,
    )


@runtime_checkable
class LowLevelWCS(Protocol):
    """
    Protocol for low-level WCS interfaces.

    This is described in `APE 14: A shared Python interface for World Coordinate
    Systems <https://doi.org/10.5281/zenodo.1188875>`_.
    """

    @property
    @abstractmethod
    def pixel_n_dim(self) -> int:
        """
        The number of axes in the pixel coordinate system
        """
        ...

    @property
    @abstractmethod
    def world_n_dim(self) -> int:
        """
        The number of axes in the world coordinate system
        """
        ...

    @property
    @abstractmethod
    def pixel_axis_names(self) -> Iterable[str]:
        """
        An iterable of strings describing the name for each pixel axis.

        If an axis does not have a name, an empty string should be returned
        (this is the default behavior for all axes if a subclass does not
        override this property). Note that these names are just for display
        purposes and are not standardized.
        """

    @property
    @abstractmethod
    def array_shape(self) -> tuple[int, ...] | None:
        """
        The shape of the data that the WCS applies to as a tuple of length
        ``pixel_n_dim`` in ``(row, column)`` order (the convention for
        arrays in Python) (optional).

        If the WCS is valid in the context of a dataset with a particular
        shape, then this property can be used to store the shape of the
        data. This can be used for example if implementing slicing of WCS
        objects. This is an optional property, and it should return `None`
        if a shape is neither known nor relevant.
        """
        ...

    @property
    @abstractmethod
    def pixel_shape(self) -> tuple[int, ...] | None:
        """
        The shape of the data that the WCS applies to as a tuple of length
        ``pixel_n_dim`` in ``(x, y)`` order (where for an image, ``x`` is
        the horizontal coordinate and ``y`` is the vertical coordinate)
        (optional).

        If the WCS is valid in the context of a dataset with a particular
        shape, then this property can be used to store the shape of the
        data. This can be used for example if implementing slicing of WCS
        objects. This is an optional property, and it should return `None`
        if a shape is neither known nor relevant.
        """
        ...

    @property
    @abstractmethod
    def pixel_bounds(self) -> tuple[Interval, ...] | list[Interval] | Interval | None:
        """
        The bounds (in pixel coordinates) inside which the WCS is defined,
        as a list with ``pixel_n_dim`` ``(min, max)`` tuples (optional).

        The bounds should be given in ``[(xmin, xmax), (ymin, ymax)]``
        order. WCS solutions are sometimes only guaranteed to be accurate
        within a certain range of pixel values, for example when defining a
        WCS that includes fitted distortions. This is an optional property,
        and it should return `None` if a shape is neither known nor relevant.
        """
        ...

    @property
    @abstractmethod
    def world_axis_physical_types(self) -> Iterable[str]:
        """
        An iterable of strings describing the physical type for each world axis.

        These should be names from the VO UCD1+ controlled Vocabulary
        (http://www.ivoa.net/documents/latest/UCDlist.html). If no matching UCD
        type exists, this can instead be ``"custom:xxx"``, where ``xxx`` is an
        arbitrary string.  Alternatively, if the physical type is
        unknown/undefined, an element can be `None`.
        """
        ...

    @property
    @abstractmethod
    def world_axis_units(self) -> Iterable[str]:
        """
        An iterable of strings given the units of the world coordinates for each
        axis.

        The strings should follow the `IVOA VOUnit standard
        <http://ivoa.net/documents/VOUnits/>`_ (though as noted in the VOUnit
        specification document, units that do not follow this standard are still
        allowed, but just not recommended).
        """
        ...

    @property
    @abstractmethod
    def world_axis_names(self) -> Iterable[str]:
        """
        An iterable of strings describing the name for each world axis.

        If an axis does not have a name, an empty string should be returned
        (this is the default behavior for all axes if a subclass does not
        override this property). Note that these names are just for display
        purposes and are not standardized. For standardized names, use
        ``world_axis_physical_types``.
        """
        ...

    @property
    @abstractmethod
    def axis_correlation_matrix(self) -> BooleanBuffer:
        """
        Returns an ``(world_n_dim, pixel_n_dim)`` matrix that indicates
        using booleans whether a given world coordinate depends on a given
        pixel coordinate. This should default to a matrix where all elements
        are True in the absence of any further information. For completely
        independent axes, the diagonal would be True and all other entries
        False. The pixel axes should be ordered in the ``(x, y)`` order,
        where for an image, ``x`` is the horizontal coordinate and ``y`` is
        the vertical coordinate.
        """
        ...

    @abstractmethod
    def pixel_to_world_values(self, *pixel_values: ScalarArrays) -> OutputCoords:
        """
        Convert pixel coordinates to world coordinates. This method takes
        n_pixel scalars or arrays as input, and pixel coordinates should be
        zero-based. Returns n_world scalars or arrays in units given by
        ``world_axis_units``. Note that pixel coordinates are assumed
        to be 0 at the center of the first pixel in each dimension. If a
        pixel is in a region where the WCS is not defined, NaN can be
        returned. The coordinates should be specified in the ``(x, y)``
        order, where for an image, ``x`` is the horizontal coordinate and
        ``y`` is the vertical coordinate.
        """
        ...

    @abstractmethod
    def array_index_to_world_values(self, *index_arrays: IndexArrays) -> OutputCoords:
        """
        Convert array indices to world coordinates. This is the same as
        ``pixel_to_world_values`` except that the indices should be given
        in ``(i, j)`` order, where for an image ``i`` is the row and ``j``
        is the column (i.e. the opposite order to ``pixel_to_world_values``).
        """
        ...

    @abstractmethod
    def world_to_pixel_values(self, *world_arrays: ScalarArrays) -> OutputCoords:
        """
        Convert world coordinates to pixel coordinates. This method takes
        n_world scalars or arrays as input in units given by ``world_axis_units``.
        Returns n_pixel scalars or arrays. Note that pixel coordinates are
        assumed to be 0 at the center of the first pixel in each dimension.
        to be 0 at the center of the first pixel in each dimension. If a
        world coordinate does not have a matching pixel coordinate, NaN can
        be returned.  The coordinates should be returned in the ``(x, y)``
        order, where for an image, ``x`` is the horizontal coordinate and
        ``y`` is the vertical coordinate.
        """
        ...

    @abstractmethod
    def world_to_array_index_values(self, *world_arrays: ScalarArrays) -> OutputIndex:
        """
        Convert world coordinates to array indices. This is the same as
        ``world_to_pixel_values`` except that the indices should be returned
        in ``(i, j)`` order, where for an image ``i`` is the row and ``j``
        is the column (i.e. the opposite order to ``pixel_to_world_values``).
        The indices should be returned as rounded integers.
        """
        ...

    @property
    @abstractmethod
    def serialized_classes(self) -> bool:
        """
        Indicates whether Python objects are given in serialized form or as
        actual Python objects.
        """
        ...

    @property
    @abstractmethod
    def world_axis_object_components(self) -> list[WorldAxisComponent]:
        """
        A list with n_dim_world elements, where each element is a tuple with
        three items:

        * The first is a name for the world object this world array
          corresponds to, which *must* match the string names used in
          ``world_axis_object_classes``. Note that names might appear twice
          because two world arrays might correspond to a single world object
          (e.g. a celestial coordinate might have both “ra” and “dec”
          arrays, which correspond to a single sky coordinate object).

        * The second element is either a string keyword argument name or a
          positional index for the corresponding class from
          ``world_axis_object_classes``

        * The third argument is a string giving the name of the property
          to access on the corresponding class from
          ``world_axis_object_classes`` in order to get numerical values.

        See the document `APE 14: A shared Python interface for World Coordinate
        Systems <https://doi.org/10.5281/zenodo.1188875>`_ for examples .
        """
        ...

    @property
    @abstractmethod
    def world_axis_object_classes(self) -> dict[str, WorldAxisClass]:
        """
        A dictionary with each key being a string key from
        ``world_axis_object_components``, and each value being a tuple with
        three elements:

        * The first element of the tuple must be a class or a string
          specifying the fully-qualified name of a class, which will specify
          the actual Python object to be created.

        * The second element, should be a tuple specifying the positional
          arguments required to initialize the class. If
          ``world_axis_object_components`` specifies that the world
          coordinates should be passed as a positional argument, this this
          tuple should include ``None`` placeholders for the world
          coordinates.

        * The last tuple element must be a dictionary with the keyword
          arguments required to initialize the class.

        See below for an example of this property. Note that we don't
        require the classes to be Astropy classes since there is no
        guarantee that Astropy will have all the classes to represent all
        kinds of world coordinates. Furthermore, we recommend that the
        output be kept as human-readable as possible.

        The classes used here should have the ability to do conversions by
        passing an instance as the first argument to the same class with
        different arguments (e.g. ``Time(Time(...), scale='tai')``). This is
        a requirement for the implementation of the high-level interface.

        The second and third tuple elements for each value of this
        dictionary can in turn contain either instances of classes, or if
        necessary can contain serialized versions that should take the same
        form as the main classes described above (a tuple with three
        elements with the fully qualified name of the class, then the
        positional arguments and the keyword arguments). For low-level API
        objects implemented in Python, we recommend simply returning the
        actual objects (not the serialized form) for optimal performance.
        Implementations should either always or never use serialized classes
        to represent Python objects, and should indicate which of these they
        follow using the ``serialized_classes`` attribute.

        See the document
        `APE 14: A shared Python interface for World Coordinate Systems
        <https://doi.org/10.5281/zenodo.1188875>`_ for examples .
        """
        ...


@runtime_checkable
class HighLevelWCS(Protocol):
    """
    Protocol for high-level WCS interfaces.

    This is described in `APE 14: A shared Python interface for World Coordinate
    Systems <https://doi.org/10.5281/zenodo.1188875>`_.
    """

    @property
    @abstractmethod
    def low_level_wcs(self) -> LowLevelWCS:
        """
        The low-level WCS object that this high-level WCS object wraps.
        """
        ...

    @abstractmethod
    def pixel_to_world(self, *pixel_arrays: ScalarArrays) -> HighLevelOutput:
        """
        Convert pixel coordinates to world coordinates (represented by Astropy
        objects).

        If a single high-level object is used to represent the world coordinates
        (i.e., if ``len(wcs.world_axis_object_classes) == 1``), it is returned
        as-is (not in a tuple/list), otherwise a tuple of high-level objects is
        returned. See
        `~astropy.wcs.wcsapi.protocols. LowLevelWCS.pixel_to_world_values` for
        pixel indexing and ordering conventions.
        """
        ...

    @abstractmethod
    def array_index_to_world(self, *index_arrays: IndexArrays) -> HighLevelOutput:
        """
        Convert array indices to world coordinates (represented by Astropy
        objects).

        If a single high-level object is used to represent the world coordinates
        (i.e., if ``len(wcs.world_axis_object_classes) == 1``), it is returned
        as-is (not in a tuple/list), otherwise a tuple of high-level objects is
        returned. See
        `~astropy.wcs.wcsapi.protocols. LowLevelWCS.array_index_to_world_values`
        for pixel indexing and ordering conventions.
        """
        ...

    @abstractmethod
    def world_to_pixel(self, *world_objects: tuple[HighLevelOutput]) -> OutputCoords:
        """
        Convert world coordinates (represented by Astropy objects) to pixel
        coordinates. See ``world_to_pixel_values`` for pixel indexing and
        ordering conventions.

        If `~astropy.wcs.wcsapi.protocols.LowLevelWCS.pixel_n_dim` is ``1``,
        this method returns a single scalar or array, otherwise a tuple of
        scalars or arrays is returned. See
        `~astropy.wcs.wcsapi.protocols.LowLevelWCS.world_to_pixel_values` for
        pixel indexing and ordering conventions.
        """
        ...

    @abstractmethod
    def world_to_array_index(
        self, *world_objects: tuple[HighLevelObject]
    ) -> OutputIndex:
        """
        Convert world coordinates (represented by Astropy objects) to array
        indices.

        If `~astropy.wcs.wcsapi.protocols.LowLevelWCS.pixel_n_dim` is ``1``,
        this method returns a single scalar or array, otherwise a tuple of
        scalars or arrays is returned. See
        `~astropy.wcs.wcsapi.protocols.LowLevelWCS.world_to_array_index_values`
        for pixel indexing and ordering conventions. The indices should be
        returned as rounded integers.
        """
        ...
