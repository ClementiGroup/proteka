from warnings import warn
import numpy as np
from .unit import format_unit, is_unit_convertible, unit_conv
from .meta_array import MetaArray


__all__ = ["BaseQuantity", "Quantity"]


class BaseQuantity:
    """Numbers or numpy arrays with a `unit` field. Retrive the value by `in_unit_of`
    method for unit compatibility.
    """

    def __init__(self, value, unit="dimensionless"):
        """Initialize a `BaseQuantity` object. Note that it is only a wrapper around the
        actual array, so changing the array will change the quantity value as well, and
        vise versa.

        Parameters
        ----------
        value : numpy.ndarray or any array-like input
            The raw value of the quantity.
        unit : str, optional
            The unit of the quantity, by default "dimensionless"
        """
        self._value = np.array(value)
        self._unit = format_unit(unit)

    @property
    def raw_value(self):
        """The raw array inside the quantity object.

        Returns
        -------
        numpy.ndarray
        """
        return self._value

    @property
    def unit(self):
        """The unit of the quantity object.

        Returns
        -------
        str
        """
        return self._unit

    def _set_unit(self, new_unit):
        """Private method to directly change the unit of the quantity. Should not be
        used by the user, since the value should be changed accordingly when doing a
        proper unit conversion.

        Parameters
        ----------
        new_unit : str
            The new unit of the Quantity
        """
        self._unit = new_unit

    @property
    def shape(self):
        return self.raw_value.shape

    @property
    def dim(self):
        return len(self.shape)

    def __len__(self):
        return len(self.raw_value)

    def is_unit_convertible_with(self, unit_or_quantity):
        """Check whether it has convertible units with input units or `Quantity` object.

        Parameters
        ----------
        unit_or_quantity : str | (Base)Quantity
            A unit string or a quantity with unit

        Returns
        -------
        bool
            True if the unit is interconvertible.
        """
        return is_unit_convertible(self.unit, unit_or_quantity)

    def in_unit_of(self, target_unit="dimensionless"):
        """Return quantity value in target unit. If not possible, raise `ValueError`.

        Parameters
        ----------
        target_unit : str, optional
            The value of the current quantity will be returned in this unit string, by
            default "dimensionless"

        Returns
        -------
        numpy.ndarray
            The value of the current quantity in `target_unit`.

        Raises
        ------
        ValueError
            When the current quantity is not convertible to `target_unit`.
        """
        target_unit = format_unit(target_unit)
        if not self.is_unit_convertible_with(target_unit):
            raise ValueError(
                f"Quantity unit {self.unit} is not compatible with desired unit {target_unit}."
            )
        return self.raw_value * unit_conv(self.unit, target_unit)

    def to_quantity_with_unit(self, target_unit="dimensionless"):
        """Convert the quantity to target unit. If not possible, raise `ValueError`.
        Comparing to `in_unit_of`, a Quantity object is returned.

        Parameters
        ----------
        target_unit : str, optional
            The unit string to convert the current quantity into, by default
            "dimensionless".

        Returns
        -------
        (Base)Quantity
            A new quantity in `target_unit`.

        Raises
        ------
        ValueError
            When the current quantity is not convertible to `target_unit`.
        """
        new_unit = format_unit(target_unit)
        new_raw_value = self.in_unit_of(target_unit)
        return self.__class__(new_raw_value, new_unit)


class Quantity(MetaArray, BaseQuantity):
    """Quantity (numpy.ndarray + unit str) with metadata (`metadata`). Specialized
    MetaArray with `unit` as a mandatory field in the metadata. Can be converted to and
    from a HDF5 Dataset.
    """

    def __init__(self, value, unit="dimensionless", metadata=None):
        """Initialize a `Quantity` object.

        Parameters
        ----------
        value : numpy.ndarray or any array-like input
            The raw value of the quantity.
        unit : str, optional
            The unit of the quantity, by default "dimensionless"
        metadata : dict, optional
            Metadata to be wrapped, a mapping from str to str or arrays, by default None

        Raises
        ------
        ValueError
            When the `unit` is a key in `metadata` and the value is not the identical as
            the argument. Recommended to double check and either remove the `unit` in
            `metadata` or make it identical with the argument.
        """
        if metadata is None:
            metadata = {}
        if "unit" in metadata and unit != metadata["unit"]:
            raise ValueError(
                'Ambigious `unit` assignment: "unit" should not appear in `metadata`.'
            )
        metadata["unit"] = unit
        super().__init__(np.asarray(value), metadata)

    @property
    def unit(self):
        """The unit of the quantity object.

        Returns
        -------
        str
        """
        return self.metadata["unit"]

    def _set_unit(self, new_unit):
        """Private method to directly change the unit of the quantity. Should not be
        used by the user, since the value should be changed accordingly when doing a
        proper unit conversion.

        Parameters
        ----------
        new_unit : str
            The new unit of the Quantity
        """
        self.metadata["unit"] = new_unit

    @staticmethod
    def from_hdf5(h5dt, suppress_unit_warn=False):
        """Create an instance from the content of HDF5 dataset `h5dt`. If no `unit` is
        present in the HDF5 attributes, then assumed as "dimensionless" with warning.

        Parameters
        ----------
        h5dt : h5py.Dataset
            A HDF5 Dataset.
        suppress_unit_warn : bool, optional
            Whether to allow silently setting unit to "dimensionless", by default False

        Returns
        -------
        Quantity
            The quantity implied by the content in the input HDF5 Dataset.
        """
        meta_array = MetaArray.from_hdf5(h5dt)
        if "unit" not in meta_array.metadata:
            if not suppress_unit_warn:
                warn(
                    f'Input HDF5 dataset {h5dt.name} does not contain a "unit" field, treating as dimensionless.'
                )
            unit = "dimensionless"
        else:
            unit = meta_array.metadata.pop("unit")
        return Quantity(meta_array._value, unit, metadata=meta_array.metadata)

    def __repr__(self):
        return f'<Quantity: shape {self.raw_value.shape}, type "{self.raw_value.dtype}", unit "{self.unit}">'
