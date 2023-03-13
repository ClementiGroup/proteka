# An `Ensemble` object contains samples and other information for a system at
#   a certain thermodynamic state. The samples usually correspond to a
#   Boltzmann distribution
from warnings import warn
import json
import numpy as np
import mdtraj as md

# from .unit_utils import format_unit, unit_conv, is_unit_compatible
import h5py
from enum import Enum
from .meta_array import MetaArray
from .unit_quantity import (
    BaseQuantity,
    Quantity,
    BUILTIN_QUANTITIES,
    parse_unit_system,
    unit_system_to_str,
    get_preset_unit,
)
from types import MappingProxyType
from .top_utils import json2top, top2json

__all__ = ["Ensemble"]


class HDF5Group:
    """Interface for saving and loading from a HDF5 Group which contains only Datasets (i.e., leaf group)."""

    def __init__(self, data, metadata=None):
        """`data` is a dictionary of `MetaArray`s, while `metadata` will become the attributes."""
        self._data = data
        if metadata is None:
            metadata = {}
        self._attrs = metadata

    @property
    def metadata(self):
        return self._attrs

    def __getitem__(self, key):
        return self._data[key]

    def __delitem__(self, key):
        self._data.pop(key)

    def __contains__(self, key):
        return key in self._data

    def write_to_hdf5(
        self, h5_node, name=None, overwrite_strategy="replace_all"
    ):
        """Write the content to a HDF5 group at `h5_node` or `h5_node[name]` when `name` is not `None`.
        If"""
        if isinstance(h5_node, h5py.Group):
            # h5_node correponds to a Group in HDF5 file
            if name in h5_node:
                # TODO: properly handle the case where the group already exists
                raise ValueError(f"Group {h5_node[name]} already exists.")
                # overwrite(h5_node[name], "h5_node[name]")
            else:
                # create a new group under h5_node
                grp = h5_node.create_group(name)
                for dt_name, dt in self._data.items():
                    # dt is a MetaArray
                    dt.write_to_hdf5(grp, name=dt_name)
                for attr_k, attr_v in self._attrs.items():
                    grp.attrs[attr_k] = attr_v
        else:
            raise ValueError(
                "Input `h5_node` should be an instance of `h5py.Group`."
            )

    @staticmethod
    def from_hdf5(h5grp, skip=None):
        """Create an instance from the content of HDF5 Group `h5grp`. The Datasets under `h5grp`, except for those contained in `skip`, will be read in and interpreted as a `Quantity`. The attributes on `h5grp` will be interpreted as metadata.
        skip (List[str]): the names of entries to skip, e.g., when it is a subgroup or not compatible with `Quantity`.
        """
        if not isinstance(h5grp, h5py.Group):
            raise ValueError(
                f"Input {h5grp}'s type is {type(h5grp)}, expecting a h5py.Group."
            )
        if skip is None:
            skip = []
        # TODO: check there is no sub group
        data = {}
        for dt_name, dt in h5grp.items():
            if not isinstance(dt, h5py.Dataset):
                raise ValueError(
                    f"`{h5grp.name}/{dt_name}` is not a valid Dataset."
                )
            if dt_name not in skip:
                data[dt_name] = Quantity.from_hdf5(dt, suppress_unit_warn=True)
        metadata = {}
        for k, v in h5grp.attrs.items():
            metadata[k] = v
        return HDF5Group(data, metadata)


def toQuantity(array_like, unit="dimensionless"):
    # check and convert `quant` to a Quantity
    try:
        if isinstance(array_like, str):
            quant = np.asarray(array_like, dtype="O")
        else:
            quant = np.asarray(array_like)
    except:
        raise ValueError("Input is not an array.")
    # float? cast to single-precision
    if quant.dtype.kind == "f":
        quant = quant.astype(np.float32, casting="same_kind")
    quant = Quantity(quant, unit=unit)
    return quant


class Ensemble(HDF5Group):
    """An `Ensemble` is an in-memory data structure consisting of sample coordinates and other quantities for a certain system at a certain thermodynamic state.
    The samples usually correspond to a Boltzmann distribution. An `Ensemble` must have `name`, `top` (molecular topology) and `coords` (3D coordinates).
    In addition, an `unit_system` has to be given in format "[L]-[M]-[T]-[E(nergy)]" for units of internal storage, default "nm-g/mol-ps-kJ/mol".

    About `Quantity`:
    A `Quantity` wraps a `numpy.ndarray` and a `unit` (defined in `proteka.dataset.unit_quantity`).
    Assigning a `Quantity` to an `Ensemble` either during initialization or via the dot (.) notation as an attribute:
    - If the input is a plain `numpy.ndarray`, then the unit is assumed as "dimensionless"
    - If the input is a `Quantity`, the input unit will be stored
    Retrieving saved `Quantity`:
    - Via the dot (.) notation: returns a `numpy.ndarray` with the value of the `Quantity` in unit of the stored unit
    - Via index bracket ([]): returns the stored `Quantity` object, allowing flexible automated unit conversion
    List stored quantities:
    - .list_quantities()
    * Special cases are "builtin quantities", whose stored units are dictated by the `unit_system` (also used instead of the default "dimensionless" during assignment):
    - "coords" (ATOMIC_VECTOR): [L]
    - "time" (_per-frame_ SCALAR): [T]
    - "forces" (ATOMIC_VECTOR): [E]/[L]
    - "velocities" (ATOMIC_VECTOR): [L]/[T]
    - "cell_lengths" (BOX_QUANTITIES): [L]
    - "cell_angles": (BOX_QUANTITIES): degree
    In addition, the above quantities are tied to the system molecule via the shape, i.e., each _per-frame_ quantity having the same number of frames as `self.coords`, and correspond to the same number of atoms as indicated by `self.top`, if it is an _ATOMIC_VECTOR_.

    Trjectories:
    Storing the information about which samples contained in the `Ensemble` come from which trajectory.
    Trajectories are sequential. Therefore, samples from different trajectories are expected to be non-overlapping slices.
    Trajectories info is supposed to be stored after Ensemble initialization with `.register_trjs` method.
    Properties:
    - `.n_trjs` (int): number of trajectories
    - `.trj_n_frames` (Dict[str, int]): dictionary of number of frames in each trajectory
    - `.trajectory_slices` or `.trjs` or `.trajectories` (Dict[str, slice]): Python `slice`s for slicing Ensemble quantities according to the `.trjs` records
    - `.trj_indices` (Dict[str, np.ndarray]): indices for different trajectories according to the `.trjs` records

    `mdtraj` interface:
    - .get_mdtraj() (-> Dict[str, mdtraj.Trajectory]): pack an `Ensemble`'s `top` and `coords` (and unitcell + simulation times, if available) into a dictionary of `Trajectory` for analyses according to `self.trjs`
    - .get_all_in_one_mdtraj(): pack all `coords` into one `Trajectory` object (maybe not suitable for kinetic analyses, such as TICA and MSM!)
    """

    def __init__(
        self,
        name,
        top,
        coords,
        quantities=None,
        metadata=None,
        unit_system="nm-g/mol-ps-kJ/mol",
    ):
        """Initialize an Ensemble with following inputs:
        - name (str): a human-readable name of the system. Not necessarily corresponding to the HDF5 group name
        - top (mdtraj.Topology): the molecular topology of the system
        - coords (Quantity or numpy.ndarray): 3D coordinates with shape (n_frames, n_atoms, 3) and dimension [L]
        - quantities (Mapping[str, np.ndarray | Quantity]): optional fields, for example:
            - forces: (n_frames, n_atoms, 3) _ATOMIC_VECTOR_.
            - velocities: (n_frames, n_atoms, 3) _ATOMIC_VECTOR_ with dimension [L]/[T].
            - time: (n_frames,) _per-frame_ scalar indicating the elapsed simulation time with dimension [T].
        - metadata (Mapping[str, str]): metadata to be saved, e.g., simulation temperature, forcefield information,
                                        saving time stride, etc
        - unit_system (str): in format "[L]-[M]-[T]-[E(nergy)]" for units of builtin quantities
        """
        try:
            top_str = top2json(top)
        except:
            raise ValueError(
                "Invalid input `top`, expecting a `mdtraj.Topology` object."
            )
        if (
            len(coords.shape) != 3
            or coords.shape[1] != top.n_atoms
            or coords.shape[2] != 3
        ):
            raise ValueError(
                f"Invalid input `coords`, expecting shape [N_frames, {top.n_atoms}, 3]."
            )
        self._unit_system = parse_unit_system(unit_system)
        super().__init__({}, metadata=metadata)
        self.coords = (
            coords,
            coords.shape,
        )  # overriding shape checks, which depend on coords themselves
        self.save_quantity("top", toQuantity(top_str), None)
        # self.top = top_str
        self.metadata["name"] = name
        self.metadata["unit_system"] = unit_system_to_str(self._unit_system)
        if quantities is not None:
            for k, v in quantities.items():
                if k == "coords":
                    warn('Omitting input `quantities["coords"]`')
                    continue
                elif k == "top":
                    warn('Omitting input `quantities["top"]`')
                    continue
                self.__setattr__(k, v)
        # register a default sequence for the whole length for compatibility
        self.register_trjs({"default": slice(None)})  # same as [:]
        self._data.pop("trjs")

    @property
    def name(self):
        return self.metadata["name"]

    @name.setter
    def name(self, new_name):
        if not isinstance(new_name, str):
            raise ValueError("`name` should be a string")
        self.metadata["name"] = new_name

    @property
    def unit_system(self):
        """Return a read-only dict of the unit system used by the Ensemble."""
        return MappingProxyType(self._unit_system)

    @property
    def top(self):
        """Return the topology of the molecular system."""
        if not hasattr(self, "_top"):
            self._top = json2top(self._data["top"][()])
        return self._top

    @top.setter
    def top(self, v):
        raise NotImplementedError(
            "`Ensemble` is always initialized with a fixed `top`."
        )

    @property
    def topology(self):
        """Return the topology of the molecular system. Alias to self.top."""
        return self.top

    @property
    def n_atoms(self):
        """Return the number of atoms of the molecular system."""
        return self.top.n_atoms

    @property
    def n_frames(self):
        """Number of frames."""
        return self._data["coords"].shape[0]

    @property
    def trajectories(self):
        return self.trajectory_slices

    @property
    def trjs(self):
        return self.trajectory_slices

    @trajectories.setter
    def trajectories(self, v):
        raise NotImplementedError(
            "Please change the trajectory records with `.register_trjs`"
        )

    @trjs.setter
    def trjs(self, v):
        raise NotImplementedError(
            "Please change the trajectory records with `.register_trjs`"
        )

    def register_trjs(self, dict_of_slices):
        """Use input slices to indicate the Ensemble samples for trajectories.
        dict_of_slices (Mapping[str, slice])."""
        trjs = {
            k: slice_.indices(self.n_frames)
            for k, slice_ in dict_of_slices.items()
        }
        self.save_quantity(
            "trjs",
            toQuantity(json.dumps(trjs)),
            shape=None,
            verbose=hasattr(self, "_trjs"),
        )
        self._trjs = trjs

    @property
    def n_trjs(self):
        """Number of trajectories in the Ensemble."""
        if hasattr(self, "_trjs"):
            return len(self._trjs)
        else:
            return 0

    @property
    def trj_n_frames(self):
        """Number of frames contained by each trajectory."""
        if self.n_trjs == 0:
            return None
        else:
            return {
                k: (trj[1] - trj[0]) // trj[2] for k, trj in self._trjs.items()
            }

    @property
    def trajectory_slices(self):
        """Generate python `slice`s for subsetting the _per-frame_ quantities for each Trjectory."""
        if self.n_trjs == 0:
            return None
        else:
            return {k: slice(*trj) for k, trj in self._trjs.items()}

    @property
    def trajectory_indices(self):
        """Indices of frame belonging to each Trjectory."""
        if self.n_trjs == 0:
            return None
        else:
            return {k: slice(*trj) for k, trj in self._trjs.items()}

    def list_quantities(self):
        """List the name of quantities stored in the `Ensemble`."""
        return list(self._data)

    def get_quantity(self, key):
        """Retrieve a Quantity in the `Ensemble`."""
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Quantity `{key}` does not exist")

    def get_unit(self, key):
        """Retrieve the storage unit for a `Quantity` under name `key` in the Ensemble, or alternatively the preset unit of a builtin `Quantity`. If neither is the case, return `None`."""
        if key in self._data:
            return self.get_quantity(key).unit
        elif key in BUILTIN_QUANTITIES:
            preset_unit = get_preset_unit(key, self._unit_system)
            return preset_unit
        else:
            return None

    def __getattr__(self, key):
        return self.get_quantity(key)[...]

    def __setattr__(self, key, quant_w_shape_hint):
        if key.startswith("_"):
            # not to block normal class initializations
            object.__setattr__(self, key, quant_w_shape_hint)
            return
        if isinstance(quant_w_shape_hint, tuple):
            # case 1: quant_w_shape_hint = (quant, shape_hint)
            if len(quant_w_shape_hint) == 2:
                quant, shape_hint = quant_w_shape_hint
            else:
                raise ValueError(
                    "Expecting either a `Quantity` without `shape_hint` or a `Tuple[Quantity, shape_hint(:=str)]`"
                )
        else:
            # case 2: quant_w_shape_hint = quant
            quant = quant_w_shape_hint
            shape_hint = None
        preset_unit = None
        # built-in quantities?
        if key in BUILTIN_QUANTITIES:
            # TODO: check whether the shape hint matches
            if shape_hint is None:
                shape_hint = BUILTIN_QUANTITIES[key][0]
            preset_unit = get_preset_unit(key, self._unit_system)
        # check and convert `quant` to a Quantity
        if not isinstance(quant, BaseQuantity):
            if preset_unit is None:
                preset_unit = "dimensionless"
            print(f'Assuming unit of input "{key}" to be "{preset_unit}".')
            quant = toQuantity(quant, preset_unit)
        else:
            if isinstance(quant, BaseQuantity) and not isinstance(
                quant, Quantity
            ):
                # transform BaseQuantity (without metadata) to a Quantity
                quant = Quantity(quant.raw_value, unit=quant.unit)
            # convert to preset unit
            if preset_unit is not None:
                # print(f"Convert \"{key}\" to internal unit {preset_unit}")
                quant = quant.to_quantity_with_unit(preset_unit)
        self.save_quantity(key, quant, shape_hint)

    def get_all_in_one_mdtraj(self):
        """Pack all `coords` into one `Trajectory` object (maybe not suitable for kinetic analyses, such as TICA and MSM!)"""
        time = self["time"].in_unit_of("ps") if "time" in self else None
        cell_lengths = (
            self["cell_lengths"].in_unit_of("nm")
            if "cell_lengths" in self
            else None
        )
        cell_angles = (
            self["cell_angles"].in_unit_of("degree")
            if "cell_angles" in self
            else None
        )
        return md.Trajectory(
            self["coords"].in_unit_of("nm"),
            self.top,
            time=time,
            unitcell_lengths=cell_lengths,
            unitcell_angles=cell_angles,
        )

    def get_mdtrajs(self):
        """(-> Dict[str, mdtraj.Trajectory]): pack an `Ensemble`'s `top` and `coords` (and unitcell + simulation times, if available) into a dictionary of `Trajectory` for analyses according to `self.trjs`"""
        trj = self.get_all_in_one_mdtraj()
        return {k: trj[slice_] for k, slice_ in self.trajectory_slices.items()}

    def save_quantity(
        self,
        name,
        quantity,
        shape="[n_frames, n_atoms, 3]",
        verbose=True,
    ):
        """
        Save a `quantity` (Quantity) under name `name` (str) with the shape defined by `shape` (str | Tuple | None).
        * In `shape`: n_frames, n_atoms will be interpreted with actual values.
        * Set `shape` to `None` to bypass shape checks.
        """
        # parsing the shape hint
        if shape is not None:
            import ast

            if not isinstance(shape, tuple):
                shape = shape.replace("n_frames", str(self.n_frames))
                shape = shape.replace("n_atoms", str(self.n_atoms))
                shape = tuple(ast.literal_eval(shape))
            # check whether the input shape matches the hint
            if quantity.shape != shape:
                raise ValueError(
                    f"Incompatible input value shape, expecting {shape} but got {quantity.shape}."
                )
        # already exists? check and warn about overwriting and unit compatibility
        if name in self._data:
            old_quant = self._data[name]
            if verbose:
                print(f"Overwriting the previously saved record {name}.")
                if not quantity.is_unit_convertible_with(old_quant):
                    warn(
                        f"Overwriting record {name} with incompatible unit: {old_quant.unit} -> {quantity.unit}."
                    )
        # save data
        self._data[name] = quantity

    @classmethod
    def from_hdf5(cls, h5grp, unit_system="nm-g/mol-ps-kJ/mol"):
        """Create an instance from the content of HDF5 Group `h5grp` (h5py.Group).
        `unit_system` (str "[L]-[M]-[T]-[E(nergy)]"): for units of builtin quantities (see class docstring).
        When given `unit_system` differs from the stored record, units will be converted when reading the `Quantity` into memory.
        Required Datasets under `h5grp`:
        - top: serialized topology
        - coords (with Attribute "unit")
        """
        hdf5grp = HDF5Group.from_hdf5(h5grp)
        dataset_unit_system_str = unit_system_to_str(
            parse_unit_system(hdf5grp.metadata["unit_system"])
        )
        if unit_system is None:
            # no unit system given, fall back to the unit system stored in the h5 file
            formatted_unit_system_str = dataset_unit_system_str
        else:
            formatted_unit_system_str = unit_system_to_str(
                parse_unit_system(unit_system)
            )
            if dataset_unit_system_str != formatted_unit_system_str:
                print(
                    f'Adapting unit system from "{dataset_unit_system_str}" (storage) to "{formatted_unit_system_str}" (memory).'
                )
        try:
            coords = hdf5grp["coords"]
        except:
            raise ValueError(
                f"Missing required Dataset `coords` from input HDF5 Group `{h5grp.name}`."
            )
        try:
            top_str = hdf5grp["top"]
            top = json2top(top_str[()])
        except:
            raise ValueError(
                f"Missing or invalid required Dataset `top` from input HDF5 Group `{h5grp.name}`."
            )
        name = hdf5grp.metadata.get("name", "")
        other_quantities = {}
        for k, v in hdf5grp._data.items():
            if k != "coords" and k != "top" and k != "subsets":
                other_quantities[k] = v
        new_ensemble = cls(
            name,
            top,
            coords,
            quantities=other_quantities,
            metadata=hdf5grp.metadata,
            unit_system=formatted_unit_system_str,
        )
        if "trjs" in hdf5grp:
            try:
                trjs_dict = json.loads(hdf5grp["trjs"][()])
                new_ensemble._trjs = trjs_dict
            except:
                raise ValueError(
                    f"Invalid `subsets` information in the input HDF5 Group `{h5grp.name}`."
                )
        return new_ensemble

    def __repr__(self):
        return f'<Ensemble for molecule "{self.name}" with {self.n_atoms} atoms and {self.n_frames} frames>'
