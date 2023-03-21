from enum import Enum
from warnings import warn
import mdtraj as md
import numpy as np
from .meta_array import MetaArray

__all__ = [
    "format_unit",
    "unit_conv",
    "is_unit_convertible",
    "Quantity",
    "BaseQuantity",
    "PerFrameQuantity",
    "BUILTIN_QUANTITIES",
]


class PerFrameQuantity(str, Enum):
    """Enum class for some known per-frame quantities. Carrying the shape as a string."""

    SCALAR = "[n_frames]"
    ATOMIC_VECTOR = "[n_frames, n_atoms, 3]"
    BOX_VECTOR = "[n_frames, 3, 3]"
    BOX_QUANTITIES = "[n_frames, 3]"


PFQ = PerFrameQuantity
BUILTIN_QUANTITIES = {
    "coords": (PFQ.ATOMIC_VECTOR, "[L]"),
    "time": (PFQ.SCALAR, "[T]"),
    "forces": (PFQ.ATOMIC_VECTOR, "[E]/[L]"),
    "velocities": (PFQ.ATOMIC_VECTOR, "[L]/[T]"),
    "cell_lengths": (PFQ.BOX_QUANTITIES, "[L]"),
    "cell_angles": (PFQ.BOX_QUANTITIES, "degree"),
    "weights": (PFQ.SCALAR, "dimensionless"),
}


def parse_unit_system(unit_system_str="nm-g/mol-ps-kJ/mol"):
    """Parsing a string that defines a unit system.

    Parameters
    ----------
    unit_system_str : str, optional
        "[L]-[M]-[T]-[E]", by default "nm-g/mol-ps-kJ/mol"

    Returns
    -------
    dict
        A `dict` holding the units for all four input dimensions: [L]ength, [M]ass,
        [T]ime and [E]nergy.

    Raises
    ------
    ValueError
        When the input is not a string made of 4 units joined by '-'.
    """
    units = [u.strip() for u in unit_system_str.split("-")]
    if len(units) != 4:
        raise ValueError(
            'Expecting unit system to be defined as "[L]-[M]-[T]-[E]".'
        )
    rvalue = {}
    for dimension, unit in zip("LMTE", units):
        rvalue["[" + dimension + "]"] = format_unit(unit)
    return rvalue


def unit_system_to_str(unit_dict):
    """Convert the input `unit_dict` to a string for storage.

    Parameters
    ----------
    unit_dict : dict
        Python `dict` (Dict[str, str]) containing unit strings under the name "[L]",
        "[M]", "[T]" and "[E]".

    Returns
    -------
    str
        String "[L]-[M]-[T]-[E]" substituded by the actual unit in `unit_dict`.
    """
    dimensions = [f"[{u}]" for u in "LMTE"]
    assert all(
        [(d in unit_dict) for d in dimensions]
    ), "Input `unit_dict` should contain [L], [M], [T] and [E] units."
    out_str = "-".join([unit_dict[d] for d in dimensions])
    return out_str


def get_preset_unit(quant_name, unit_dict=None):
    """Get the preset unit when `quant_name` is from `BUILTIN_QUANTITIES`, and the [X]
    will be substituted by the corresponding entry in `unit_dict.

    Parameters
    ----------
    quant_name : str
        The name of the quantity, whose unit is queried.
    unit_dict : dict, optional
        A `dict` (Dict[str, str]) between the name of dimension (e.g., [L]ength) and its
        unit (e.g., "nm"), by default None

    Returns
    -------
    str | None
        The unit according to the `unit_dict` and definition in `BUILTIN_QUANTITIES`.
        If `quant_name` is not in `unit_dict`, returns None.
    """
    if quant_name in BUILTIN_QUANTITIES:
        quant_unit = BUILTIN_QUANTITIES[quant_name][-1]
        if unit_dict is not None:
            for dimension, unit in unit_dict.items():
                quant_unit = quant_unit.replace(dimension, unit)
        return quant_unit
    else:
        return None


def convert_to_unit_system(quant_name, quant, unit_dict, verbose=True):
    """Convert the input `quant` to the unit system for the builtin quantities.

    Parameters
    ----------
    quant_name : str
        The name of quantity, which is used to find the unit in the `BUILTIN_QUANTITIES`
        in the unit system.
    quant : Quantity
        The quantity to be converted
    unit_dict : dict
        A `dict` (Dict[str, str]) between the name of dimension (e.g., [L]ength) and its
        unit (e.g., "nm")
    verbose : bool, optional
        Whether print a message about the unit conversion, by default True

    Returns
    -------
    Quantity
        The quantity with converted unit, if `quant_name` is part of the
        `BUILTIN_QUANTITIES`, otherwise the original input
    """
    preset_unit = get_preset_unit(quant_name, unit_dict)
    if preset_unit is not None:
        quant = quant.to_quantity_with_unit(preset_unit)
        if verbose:
            print(f'Quantity "{quant_name}" converted to unit "{preset_unit}".')
    return quant


def format_unit(unit_):
    """Expand some handy abbreviations for compatibility with mdtraj.utils.unit.

    Parameters
    ----------
    unit_ : str
        The unit string with possible abbreviations of unit components.

    Returns
    -------
    str
        The reformatted unit string.
    """
    unit_str = str(unit_)
    if unit_str == "1" or unit_str is None:
        unit_str = "dimensionless"
    unit_str = unit_str.replace("K", "kelvin")
    unit_str = unit_str.replace("ms", "millseconds").replace(
        "us", "microseconds"
    )
    unit_str = unit_str.replace("ns", "nanoseconds").replace(
        "ps", "picoseconds"
    )
    # resolves a bug of recognizing the `ns`
    unit_str = unit_str.replace("dimenanosecondsionless", "dimensionless")
    unit_str = unit_str.replace("fs", "femtoseconds")
    unit_str = unit_str.replace("kJ", "kilojoules").replace(
        "kcal", "kilocalories"
    )
    unit_str = unit_str.replace("mol", "mole").replace("molee", "mole")
    unit_str = unit_str.replace("nm", "nanometers")
    unit_str = unit_str.replace("Angstrom", "angstrom").replace("A", "angstrom")
    return unit_str


def unit_conv(source_unit, target_unit):
    """Returns the multiplicative scaling factor to transform a value in the source unit
    to the target unit.

    Parameters
    ----------
    source_unit : str
        A valid unit string.
    target_unit : str
        A valid unit string. Should have the same dimension as the `source_unit`.

    Returns
    -------
    float | None
        The scaling factor, `None` if the two units are not interconvertible.
    """

    source_unit = format_unit(source_unit)
    target_unit = format_unit(target_unit)
    try:
        return md.utils.in_units_of(1.0, source_unit, target_unit)
    except:
        return None


def is_unit_convertible(unit_or_quantity_1, unit_or_quantity_2):
    """Check whether two units or `Quantity` objects have convertible units.

    Parameters
    ----------
    unit_or_quantity_1 : str | Quantity
        Unit `str` of `Quantity`.
    unit_or_quantity_2 : str | Quantity
        Unit `str` of `Quantity`.

    Returns
    -------
    bool
        True if the two inputs are interconvertible.
    """

    def _get_unit(unit_or_quantity):
        if isinstance(unit_or_quantity, BaseQuantity):
            u = unit_or_quantity.unit
        else:
            u = unit_or_quantity
        return format_unit(u)

    unit1 = _get_unit(unit_or_quantity_1)
    unit2 = _get_unit(unit_or_quantity_2)
    return unit_conv(unit1, unit2) is not None


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
