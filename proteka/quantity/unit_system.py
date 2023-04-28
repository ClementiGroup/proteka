from copy import copy
import numpy as np
from .unit import format_unit, is_unit_convertible, unit_conv
from .quantity_shapes import PRESET_BUILTIN_QUANTITIES


class UnitSystem:
    """A class to handle the recognition and conversion of (some) physical units in
    molecular systems. Ensuring the quantities for a system always have the correct and
    desired units for storage and analyses.

    Initialize a `UnitSystem` with given units for base quantities.

    Parameters
    ----------
    length_unit : str, optional
        [L]'s unit, by default "nm"
    mass_unit : str, optional
        [M]'s unit, by default "g/mol"
    time_unit : str, optional
        [T]'s unit, by default "ps"
    energy_unit : str, optional
        Overriding [E]'s unit, when None, uses ([M]'s unit)([L]'s unit)^2/([T]'s
        unit)^2
    extra_builtin_quantities : dict, optional
        A dictionary `name: str` -> `(shape_hint: str|None, unit: str)`, the
        `shape_hint` is a str in the form of a printed list (examples can be found
        in quantity_shapes.PerFrameQuantity; definition of units can carry
        expression with multiple [X] where X = L, M, T or E; overwrites when
        containing pairs that already exists in PRESET_BUILTIN_QUANTITIES; by
        default None

    Raises
    ------
    ValueError
        When input units are not valid strings of corresponding units.

    Notes
    -----
    The unit system is designed as the following:
    The user has to provide the units for base quantities [L], [M], [T] (and optionally
    an overriding [E]), e.g., "[L]": "nm", [T]: "ps". In the meantime, define derived
    quantities as name - expression pairs, e.g., "velocities": "[L]/[T]". Then the unit
    of these quantities can be queried, in this case .get_unit("velocities") -> "nm/ps".
    Special case: the user can specify units to quantities that are not connected with
    the unit system, e.g., "cell_angles": "degree".

    The default units are:
    [L]: nm,
    [M]: g/mol, (~Dalton)
    [T]: ps,
    [E]: kJ/mol,
    which correspond to the internal units used by OpenMM and `mdtraj`.

    Default known quantities (can be accessed as .builtin_quantities) are defined in
    `quantity_shapes.py`.

    Core concepts explained:
    Dimension, as in "dimensional analysis", means how a physical quantity is composed
    by base quantities such as [L]ength, [M]ass and [T]ime. This also determines what
    unit it should bear (or be converted to). For example, a quantity with the dimension
    of velocity [L]/[M] should have a compatible unit with m/s, thus invalid quantity
    with (for example) a unit of kg*m^2 will be refused as an input.

    In the convention of molecular dynamics engines (except those using reduced units,
    e.g., LJ-units), the unit of all physical quantities can usually be directly derived
    from the unit of base quanities. In other word, the unit can be inferred by how the
    quantity itself is derived from the basic ones. For example, a velocity quantity has
    unit ([L]'s unit) / ([M]'s unit) = nm/ps.

    One exception is the [E]nergy, which often use a different unit that might not be
    directly ([M]'s unit)([L]'s unit)^2/([T]'s unit)^2, e.g., when it involves "kcal".
    As a consequence, we allow the user to override the [E]nergy unit, which also
    affects e.g. the unit of forces.
    """

    def __init__(
        self,
        length_unit="nm",
        mass_unit="g/mol",
        time_unit="ps",
        energy_unit=None,
        extra_preset_quantities=None,
    ):
        def check_unit(name, inp_str, unit_convertible_with):
            if not isinstance(inp_str, str):
                raise ValueError(f"Expecting `{name}` to be a string.")
            if not is_unit_convertible(inp_str, unit_convertible_with):
                raise ValueError(
                    f'Expecting `{name}`: "{inp_str}" to be convertible '
                    f'with "{unit_convertible_with}".'
                )

        check_unit("length_unit", length_unit, "nm")
        check_unit("mass_unit", mass_unit, "g/mol")
        check_unit("time_unit", time_unit, "ps")
        if energy_unit is not None:
            check_unit("energy_unit", energy_unit, "kJ/mol")
        if energy_unit is None:
            energy_unit = f"({mass_unit})*({length_unit})**2/({time_unit})**2"
            if np.allclose(unit_conv(energy_unit, "kJ/mol"), 1.0):
                energy_unit = "kJ/mol"
            elif np.allclose(unit_conv(energy_unit, "kcal/mol"), 1.0):
                energy_unit = "kcal/mol"
        self._unit_dict = {
            "[L]": format_unit(length_unit),
            "[M]": format_unit(mass_unit),
            "[T]": format_unit(time_unit),
            "[E]": format_unit(energy_unit),
        }
        self._builtin_quantities = copy(PRESET_BUILTIN_QUANTITIES)
        if extra_preset_quantities is not None:
            for k, v in extra_preset_quantities.items():
                self._builtin_quantities[k] = (v[0], format_unit(v[1]))

    @property
    def unit_dict(self):
        return self._unit_dict

    @property
    def builtin_quantities(self):
        return self._builtin_quantities

    def get_preset_unit(self, quant_name):
        """Get the preset unit when `quant_name` is registered in
        `self.builtin_quantities`, and the [X] will be substituted by the corresponding
        entry in `self.unit_dict`.

        Parameters
        ----------
        quant_name : str
            The name of the quantity, whose unit is queried.

        Returns
        -------
        str | None
            The unit according to the `self.unit_dict` and definition in
            `self.builtin_quantities`, else returns `None`.
        """
        if quant_name in self.builtin_quantities:
            quant_unit = self.builtin_quantities[quant_name][-1]
            for dimension, unit in self.unit_dict.items():
                quant_unit = quant_unit.replace(dimension, unit)
            return quant_unit
        else:
            return None

    @classmethod
    def parse_from_str(cls, unit_system_str="nm-g/mol-ps-kJ/mol"):
        """Parsing a string that defines a unit system either as a simple string with
        format "[L]-[M]-[T](-[E])" or a JSON string.

        Parameters
        ----------
        unit_system_str : str, optional
            "[L]-[M]-[T](-[E])" or JSON string, by default "nm-g/mol-ps-kJ/mol"

        Returns
        -------
        UnitSystem
            A `UnitSystem` holding the units for all four basic dimensions: [L]ength,
            [M]ass, [T]ime and [E]nergy and in case of JSON string containing a

        Raises
        ------
        ValueError
            When the input is not a string made of 3 or 4 units joined by '-'.
        """
        from json import JSONDecodeError

        try:
            unit_system = cls.parse_from_json(unit_system_str)
        except JSONDecodeError:
            try:
                unit_system = cls.parse_from_plain_str(unit_system_str)
            except ValueError as e:
                raise ValueError(
                    "Input `unit_system_str` is neither a valid JSON "
                    'string, nor a "[L]-[M]-[T](-[E])" string.'
                )
        return unit_system

    @classmethod
    def parse_from_plain_str(cls, unit_system_str="nm-g/mol-ps-kJ/mol"):
        """Parsing a string that defines a unit system.

        Parameters
        ----------
        unit_system_str : str, optional
            "[L]-[M]-[T](-[E])", by default "nm-g/mol-ps-kJ/mol"

        Returns
        -------
        UnitSystem
            A `UnitSystem` holding the units for all four input dimensions: [L]ength,
            [M]ass, [T]ime and [E]nergy. If `[E]` was not provided, it will be inferred
            from the rest three as ([M]'s unit)([L]'s unit)^2/([T]'s unit)^2

        Raises
        ------
        ValueError
            When the input is not a string made of 3 or 4 units joined by '-'.
        """
        units = [u.strip() for u in unit_system_str.split("-")]
        if len(units) != 3 and len(units) != 4:
            raise ValueError(
                'Expecting unit system to be defined as "[L]-[M]-[T](-[E])".'
            )
        rvalue = {}
        for dimension, unit in zip("LMTE", units):
            rvalue["[" + dimension + "]"] = format_unit(unit)
        return cls(
            length_unit=rvalue["[L]"],
            mass_unit=rvalue["[M]"],
            time_unit=rvalue["[T]"],
            energy_unit=rvalue.get("[E]"),
        )

    @classmethod
    def parse_from_json(cls, jstr):
        """Parsing a JSON string that defines a unit system.

        Parameters
        ----------
        jstr : JSON string
            Expected to contain a `unit_dict` and optionally `builtin_quantities`.

        Returns
        -------
        UnitSystem
            A `UnitSystem` holding the units for all four basic dimensions: [L]ength,
            [M]ass, [T]ime and [E]nergy and builtin quantities.

        Raises
        ------
        ValueError
            When input units are not valid strings of corresponding units.
        KeyError
            When input JSON string does not contain required fields
        """
        from json import loads

        usys = loads(jstr)
        if "unit_dict" not in usys:
            raise KeyError(
                'Invalid JSON content: should contain a "unit_dict".'
            )
        unit_dict = usys["unit_dict"]
        builtin_quantities = usys.get("builtin_quantities")
        try:
            unit_system = cls(
                length_unit=unit_dict["[L]"],
                mass_unit=unit_dict["[M]"],
                time_unit=unit_dict["[T]"],
                energy_unit=unit_dict.get("[E]"),
                extra_preset_quantities=builtin_quantities,
            )
        except KeyError as e:
            raise KeyError(
                f'Necessary key missing from provided JSON string: "{e}"'
            )
        return unit_system

    def __repr__(self):
        """Convert a `UnitSystem` to a string for storage.

        Returns
        -------
        str
            String "[L]-[M]-[T]-[E]" substituded by the actual unit in `self.unit_dict`.
        """
        dimensions = [f"[{u}]" for u in "LMTE"]
        out_str = "-".join([self.unit_dict[d] for d in dimensions])
        return out_str

    def to_json(self):
        """Convert a `UnitSystem` to a JSON string for storage.

        Returns
        -------
        str
            JSON string containing info from `self.unit_dict` and
            `self.builtin_quantities`.
        """
        from json import dumps

        return dumps(
            {
                "unit_dict": self.unit_dict,
                "builtin_quantities": self.builtin_quantities,
            }
        )

    def convert_quantity(self, quant_name, quant, verbose=True):
        """Convert the input `quant` to the unit system for the builtin quantities.

        Parameters
        ----------
        quant_name : str
            The name of quantity, which is used to find the unit in
            `self.builtin_quantities` in the unit system.
        quant : Quantity
            The quantity to be converted
        verbose : bool, optional
            Whether print a message about the unit conversion, by default True

        Returns
        -------
        Quantity
            The quantity with converted unit, if `quant_name` is part of the
            `BUILTIN_QUANTITIES`, otherwise the original input
        """
        preset_unit = self.get_preset_unit(quant_name)
        if preset_unit is not None:
            quant = quant.to_quantity_with_unit(preset_unit)
            if verbose:
                print(
                    f'Quantity "{quant_name}" converted to unit "{preset_unit}".'
                )
        elif verbose:
            print(
                f'Quantity "{quant_name}"\'s unit is preserved as "{quant.unit}".'
            )
        return quant

    def __eq__(self, other):
        return self._unit_dict == other._unit_dict
