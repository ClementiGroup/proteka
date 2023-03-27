import pytest
import numpy as np

from .unit_system import UnitSystem
from .quantity import Quantity
from .unit import unit_conv, is_unit_convertible
from .quantity_shapes import PerFrameQuantity as PFQ

us1 = "nm-g/mol-ps-kJ/mol"
us1_json = (
    '{"unit_dict": {"[L]": "nanometers", "[M]": "gram/mole", '
    '"[T]": "picoseconds", "[E]": "kilojoules/mole"}}'
)
us2 = "A-g/mol-fs-kcal/mol"


def test_serialize_deserialize_str():
    us1_from_str = UnitSystem.parse_from_plain_str(us1)
    us1_from_str_2 = UnitSystem.parse_from_str(str(us1_from_str))
    assert (
        unit_conv(us1_from_str.get_preset_unit("coords"), "nanometers") == 1.0
    )
    assert us1_from_str.unit_dict == us1_from_str_2.unit_dict


def test_serialize_deserialize_json():
    us1_from_json = UnitSystem.parse_from_json(us1_json)
    us1_from_json_2 = UnitSystem.parse_from_str(us1_from_json.to_json())
    assert (
        unit_conv(us1_from_json.get_preset_unit("coords"), "nanometers") == 1.0
    )
    assert us1_from_json.unit_dict == us1_from_json_2.unit_dict


def test_unit_conv():
    forces = Quantity(np.random.rand(10, 22, 3), "kcal/mol/nm")
    us1_from_str = UnitSystem.parse_from_str(us1)
    assert np.allclose(
        us1_from_str.convert_quantity("forces", forces, False).raw_value,
        forces.raw_value * 4.184,
    )
    us2_from_str = UnitSystem.parse_from_str(us2)
    assert np.allclose(
        us2_from_str.convert_quantity("forces", forces, False).raw_value,
        forces.raw_value / 10.0,
    )


def test_new_builtin():
    us1_from_str = UnitSystem.parse_from_str(us1)
    us1_from_str.builtin_quantities["pressure"] = (PFQ.SCALAR, "[E]/[L]**3")
    us1_from_json_2 = UnitSystem.parse_from_str(us1_from_str.to_json())
    assert is_unit_convertible(
        us1_from_json_2.get_preset_unit("pressure"), "pascal/mole"
    )
