import mdtraj as md
import numpy as np

__all__ = ["format_unit", "unit_conv", "is_unit_convertible", "Quantity"]

def format_unit(unit_):
    """Expand some handy abbreviations for compatibility with mdtraj.utils.unit."""
    unit_str = str(unit_)
    if unit_str == "1" or unit_str is None:
        unit_str = "dimensionless"
    unit_str = unit_str.replace("K", "kelvin")
    unit_str = unit_str.replace("ms", "millseconds").replace("us", "microseconds")
    unit_str = unit_str.replace("ns", "nanoseconds").replace("ps", "picoseconds")
    unit_str = unit_str.replace("fs", "femtoseconds")
    unit_str = unit_str.replace("kJ", "kilojoules").replace("kcal", "kilocalories")
    unit_str = unit_str.replace("mol", "mole").replace("molee", "mole")
    unit_str = unit_str.replace("nm", "nanometers")
    unit_str = unit_str.replace("Angstrom", "angstrom").replace("A", "angstrom")
    return unit_str

def unit_conv(source_unit, target_unit):
    """Find the scaling factor to multiple with the source data in order to output values in the target unit."""
    
    source_unit = format_unit(source_unit)
    target_unit = format_unit(target_unit)
    try:
        return md.utils.in_units_of(1., source_unit, target_unit)
    except:
        return None

def is_unit_convertible(unit_or_quantity_1, unit_or_quantity_2):
    """Check whether two units or `Quantity` objects have convertible units."""
    def _get_unit(unit_or_quantity):
        if isinstance(unit_or_quantity, Quantity):
            return unit_or_quantity.unit
        else:
            return unit_or_quantity
    unit1 = _get_unit(unit_or_quantity_1)
    unit2 = _get_unit(unit_or_quantity_2)
    return (unit_conv(unit1, unit2) is not None)

class Quantity:
    """Numbers or numpy arrays with a `unit` field. Retrive the value by `in_unit_of` method for unit compatibility.
    """
    
    def __init__(self, value, unit="dimensionless"):
        self._value = np.array(value)
        self._unit = format_unit(unit)
    
    @property
    def raw_value(self):
        return self._value
    
    @property
    def unit(self):
        return self._unit
    
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
        """Convert to target unit. If not possible, raise ValueError."""
        target_unit = format_unit(target_unit)
        if not self.is_unit_convertible_with(target_unit):
            raise ValueError(f"Quantity unit {self.unit} is not compatible with desired unit {target_unit}.")
        return self.raw_value * unit_conv(self.unit, target_unit)
