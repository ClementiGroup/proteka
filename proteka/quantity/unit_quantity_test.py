import pytest
import numpy as np
from .unit import *
from .quantity import *
from tempfile import TemporaryFile
import h5py

test_input_units = [
    "kcal/mol/A",
    "nm/ps",
    "kcal/K",
]

test_formatted_units = [
    "kilocalories/mole/angstrom",
    "nanometers/picoseconds",
    "kilocalories/kelvin",
]

test_output_units = [
    "kJ/mol/nm",
    "A/fs",
    "joules/K",
]

test_output_values = [
    41.84,
    0.01,
    4184.0,
]


@pytest.mark.parametrize(
    "input_, outcome", zip(test_input_units, test_formatted_units)
)
def test_unit_format(input_, outcome):
    assert format_unit(input_) == outcome


@pytest.mark.parametrize(
    "input_unit, output_unit, outcome",
    zip(test_input_units, test_output_units, test_output_values),
)
def test_unit_conv(input_unit, output_unit, outcome):
    assert np.allclose(unit_conv(input_unit, output_unit), outcome)


@pytest.mark.parametrize(
    "input_unit, output_unit, outcome",
    zip(test_input_units, test_output_units, test_output_values),
)
def test_unit_conv(input_unit, output_unit, outcome):
    assert np.allclose(unit_conv(input_unit, output_unit), outcome)


@pytest.mark.parametrize("input_unit", test_input_units)
@pytest.mark.parametrize("output_unit", test_output_units)
def test_unit_convertible(input_unit, output_unit):
    # assuming units to be compatible iff. their line no. match
    input_index = test_input_units.index(input_unit)
    output_index = test_output_units.index(output_unit)
    assert is_unit_convertible(input_unit, output_unit) == (
        input_index == output_index
    )


@pytest.mark.parametrize(
    "input_unit, output_unit, scale_factor",
    zip(test_input_units, test_output_units, test_output_values),
)
def test_base_quantity(input_unit, output_unit, scale_factor):
    # make a quantity with input_unit and convert it to the output unit
    quant = BaseQuantity(5.0, unit=input_unit)
    assert quant.in_unit_of(output_unit) == 5.0 * scale_factor


@pytest.mark.parametrize(
    "input_unit, output_unit, scale_factor",
    zip(test_input_units, test_output_units, test_output_values),
)
def test_quantity(input_unit, output_unit, scale_factor):
    # make a Quantity with input_unit and convert it to the output unit
    mquant = Quantity(
        np.ones(1), unit=input_unit, metadata={"test": "metadata_no1"}
    )
    with TemporaryFile() as fp:
        with h5py.File(fp, "w") as hdf_root:
            mquant.write_to_hdf5(hdf_root, "test_dt")
        with h5py.File(fp, "r") as hdf_root:
            mquant2 = Quantity.from_hdf5(hdf_root["test_dt"])
    assert mquant2.in_unit_of(output_unit) == 1.0 * scale_factor
    assert mquant2.metadata["test"] == "metadata_no1"
