from enum import Enum
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
}


def parse_unit_system(unit_system_str="nm-g/mol-ps-kJ/mol"):
    """Parsing a string defining a unit system "[L]-[M]-[T]-[E]"."""
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
    """Convert the input `unit_dict` to a string for storage."""
    dimensions = [f"[{u}]" for u in "LMTE"]
    assert all(
        [(d in unit_dict) for d in dimensions]
    ), "Input `unit_dict` should contain [L], [M], [T] and [E] units."
    out_str = "-".join([unit_dict[d] for d in dimensions])
    return out_str


def get_preset_unit(quant_name, unit_dict=None):
    if quant_name in BUILTIN_QUANTITIES:
        quant_unit = BUILTIN_QUANTITIES[quant_name][-1]
        if unit_dict is not None:
            for dimension, unit in unit_dict.items():
                quant_unit = quant_unit.replace(dimension, unit)
        return quant_unit
    else:
        return None


def convert_to_unit_system(quant_name, quant, unit_dict, verbose=True):
    """Convert the input `quant` to the unit system for the builtin quantities."""
    preset_unit = get_preset_unit(quant_name, unit_dict)
    if preset_unit is not None:
        quant.convert_to_unit_(quant_unit)
        if verbose:
            print(f'Quantity "{quant_name}" converted to unit "{unit}".')
    return quant


def format_unit(unit_):
    """Expand some handy abbreviations for compatibility with mdtraj.utils.unit."""
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
    """Find the scaling factor to multiple with the source data in order to output values in the target unit."""

    source_unit = format_unit(source_unit)
    target_unit = format_unit(target_unit)
    try:
        return md.utils.in_units_of(1.0, source_unit, target_unit)
    except:
        return None


def is_unit_convertible(unit_or_quantity_1, unit_or_quantity_2):
    """Check whether two units or `Quantity` objects have convertible units."""

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
    """Numbers or numpy arrays with a `unit` field. Retrive the value by `in_unit_of` method for unit compatibility."""

    def __init__(self, value, unit="dimensionless"):
        self._value = np.array(value)
        self._unit = format_unit(unit)

    @property
    def raw_value(self):
        return self._value

    @property
    def unit(self):
        return self._unit

    def _set_unit(self, new_unit):
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
        """Check whether it has convertible units with input units or `Quantity` object."""
        return is_unit_convertible(self.unit, unit_or_quantity)

    def in_unit_of(self, target_unit="dimensionless"):
        """Return quantity value in target unit. If not possible, raise ValueError."""
        target_unit = format_unit(target_unit)
        if not self.is_unit_convertible_with(target_unit):
            raise ValueError(
                f"Quantity unit {self.unit} is not compatible with desired unit {target_unit}."
            )
        return self.raw_value * unit_conv(self.unit, target_unit)

    def to_quantity_with_unit(self, target_unit="dimensionless"):
        """Convert the quantity to target unit. If not possible, raise ValueError."""
        new_unit = format_unit(target_unit)
        new_raw_value = self.in_unit_of(target_unit)
        return self.__class__(new_raw_value, new_unit)


class Quantity(MetaArray, BaseQuantity):
    """Quantity (ndarray + unit) with metadata (`metadata`). Specialized MetaArray with `unit` as a mandatory field in the metadata. Can be converted to and from a HDF5 Dataset."""

    def __init__(self, value, unit="dimensionless", metadata=None):
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
        return self.metadata["unit"]

    def _set_unit(self, new_unit):
        self.metadata["unit"] = new_unit

    @staticmethod
    def from_hdf5(h5dt, suppress_unit_warn=False):
        """Create an instance from the content of HDF5 dataset `h5dt`. If no `unit` is present in the attributes, then assumed as dimensionless with warning."""
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
