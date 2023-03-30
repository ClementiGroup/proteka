"""Proteka unit handling: a wrapper around package `mdtraj`'s unit system, which in turn
is modeled after OpenMM's unit system.
"""

from enum import Enum
from warnings import warn
import mdtraj as md
import numpy as np


__all__ = [
    "format_unit",
    "unit_conv",
    "is_unit_convertible",
]


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
    unit_str = (
        unit_str.replace("g", "gram")
        .replace("gramram", "gram")
        .replace("degramree", "degree")
        .replace("angramstrom", "angstrom")
    )
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
        if hasattr(unit_or_quantity, "unit"):
            u = unit_or_quantity.unit
        else:
            u = unit_or_quantity
        return format_unit(u)

    unit1 = _get_unit(unit_or_quantity_1)
    unit2 = _get_unit(unit_or_quantity_2)
    return unit_conv(unit1, unit2) is not None
