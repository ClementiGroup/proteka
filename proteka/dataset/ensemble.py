"""An `Ensemble` object contains samples and other information for a system at
a certain thermodynamic state. The samples usually correspond to a
Boltzmann distribution.
"""

from types import MappingProxyType
from typing import Optional, List
from warnings import warn
import json
import numpy as np
import mdtraj as md

# from .unit_utils import format_unit, unit_conv, is_unit_compatible
import h5py
from proteka.quantity import (
    BaseQuantity,
    Quantity,
    UnitSystem,
)
from .top_utils import json2top, top2json

__all__ = ["Ensemble"]


class HDF5Group:
    """Interface for saving and loading from a HDF5 Group which contains only one
    Dataset (i.e., leaf group).

    Parameters
    ----------
    data : Dict[str, Quantity]
        A dictionary of `Quantity`s, which map to HDF5's Datasets under a Group
    metadata : Dict[str, str | ...], optional
        Metadata, which map to HDF5's Group Attributes, by default None
    """

    def __init__(self, data, metadata=None):
        self._data = data
        if metadata is None:
            metadata = {}
        self._attrs = metadata

    @property
    def metadata(self):
        """Accessor of the `metadata` field.

        Returns
        -------
        dict
            The `metadata` field.
        """
        return self._attrs

    def __dir__(self) -> List[str]:
        return [*super().__dir__(), *self._data.keys()]

    def __getitem__(self, key):
        return self._data[key]

    def __delitem__(self, key):
        self._data.pop(key)

    def __contains__(self, key):
        return key in self._data

    def write_to_hdf5(
        self, h5_node, name=None, overwrite_strategy="do_not_replace"
    ):
        """Write the content to a HDF5 group at `h5_node` or `h5_node[name]` when `name`
        is not `None`. If the desired group already exists and `overwrite_strategy` is
        "replace_all", then the original content will be discarded.

        Parameters
        ----------
        h5_node : h5py.Group
            The target of dumping is `h5_node`, or `h5_node[name]` if `name` is not
            `None`.
        name : str, optional
            The subgroup to create or overwrite, by default None
        overwrite_strategy : str, optional
            When equals "replace_all", will overwrite the existing group under
            `h5_node`, by default "do_not_replace"

        Raises
        ------
        ValueError
            When input `h5_node` is not a valid h5py.Group
        ValueError
            When destination group already exists
        """
        if isinstance(h5_node, h5py.Group):
            # h5_node correponds to a Group in HDF5 file
            if name is None:
                name = "."  # path: current group
            if name in h5_node:
                # TODO: more intelligent incremental update
                if overwrite_strategy != "replace_all":
                    raise ValueError(
                        f"Group/Dataset {h5_node[name]} already exists."
                    )
                grp = h5_node[name]
                # delete everything existing
                for child_name in grp:
                    del grp[child_name]
                for attr_name in grp.attrs:
                    del grp.attrs[attr_name]
            else:
                # create a new group under h5_node
                grp = h5_node.create_group(name)
            for dt_name, dt in self._data.items():
                # dt is a Quantity
                dt.write_to_hdf5(grp, name=dt_name)
            for attr_k, attr_v in self._attrs.items():
                grp.attrs[attr_k] = attr_v
        else:
            raise ValueError(
                "Input `h5_node` should be an instance of `h5py.Group`."
            )

    @staticmethod
    def from_hdf5(
        h5grp,
        skip=None,
        offset: Optional[int] = None,
        stride: Optional[int] = None,
        no_offset_stride_quantities: Optional[List[str]] = None,
    ):
        """Create an instance from the content of HDF5 Group `h5grp`. The Datasets under
        `h5grp`, except for those contained in `skip`, will be read in and interpreted
        as a `Quantity`. The attributes on `h5grp` will be interpreted as metadata. Input
        `offset` and `stride` can be set to allow a partial loading of the non-scalar
        datasets with indexing `[offset::stride]`, unless the dataset's name is contained
        in `no_offset_stride_quantities`.

        Parameters
        ----------
        h5grp : h5py.Group
            The HDF5 Group to be read in.
        skip : List[str], optional
            The names of entries to skip, e.g., when it is a subgroup or not compatible
            with `Quantity`, by default None
        offset : None | int, optional
            The offset for loading from the HDF5 file. Default is `None`.
        stride : None | int, optional
            The stride for loading from the HDF5 file. Default is `None`.
        no_offset_stride_quantities : List[str], optional
            The names of entries, for which no offset or stride should be considered
            during loading, by default None

        Returns
        -------
        HDF5Group
            An instance with the content from input `h5grp`.

        Raises
        ------
        ValueError
            When `h5grp` is not a `h5py.Group` or when it has child that is not a valid
            HDF5 Dataset.
        """
        if not isinstance(h5grp, h5py.Group):
            raise ValueError(
                f"Input {h5grp}'s type is {type(h5grp)}, expecting a h5py.Group."
            )
        if skip is None:
            skip = []
        if no_offset_stride_quantities is None:
            no_offset_stride_quantities = []
        # TODO: check there is no sub group
        data = {}
        for dt_name, dt in h5grp.items():
            if not isinstance(dt, h5py.Dataset):
                raise ValueError(
                    f"`{h5grp.name}/{dt_name}` is not a valid Dataset."
                )
            if dt_name not in skip:
                if dt_name not in no_offset_stride_quantities:
                    data[dt_name] = Quantity.from_hdf5(
                        dt,
                        offset=offset,
                        stride=stride,
                        suppress_unit_warn=True,
                    )
                else:
                    data[dt_name] = Quantity.from_hdf5(
                        dt, suppress_unit_warn=True
                    )
        metadata = {}
        for k, v in h5grp.attrs.items():
            metadata[k] = v
        return HDF5Group(data, metadata)


def _to_quantity(array_like, unit="dimensionless"):
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
        quant = quant.astype(np.float32, casting="same_kind", copy=False)
    quant = Quantity(quant, unit=unit)
    return quant


class Ensemble(HDF5Group):
    """An `Ensemble` is an in-memory data structure consisting of sample coordinates and
    other quantities for a certain system at a certain thermodynamic state. The samples
    usually correspond to a Boltzmann distribution.

    An `Ensemble` must have `name`, `top` (molecular topology) and `coords` (3D
    coordinates). In addition, a `unit_system` has to be provided either as a
    pre-defined `UnitSystem` object or a seralized JSON version of `UnitSystem` or a
    string in format of "[L]-[M]-[T]-[E(nergy)]" to specify the units used internally,
    default "nm-g/mol-ps-kJ/mol".

    Parameters
    ----------
    name : str
        a human-readable name of the system. Not necessarily corresponding to the HDF5
        group name
    top : mdtraj.Topology
        the molecular topology of the system
    coords : Quantity or numpy.ndarray
        3D coordinates with shape (n_frames, n_atoms, 3) and dimension [L]
    quantities : Dict[str, np.ndarray | Quantity], optional
        Example key and value pairs for builtin quantities:
        - forces: (n_frames, n_atoms, 3) _ATOMIC_VECTOR_.
        - velocities: (n_frames, n_atoms, 3) _ATOMIC_VECTOR_ with dimension [L]/[T].
        - time: (n_frames,) _per-frame_ scalar indicating the elapsed simulation time
        with dimension [T].
        - weights: (n_frames,) _per-frame_ scalar indicating the Boltzmann weight of
        each frame.
    metadata : dict, optional
        Metadata to be saved, e.g., simulation temperature, forcefield information,
        saving time stride, etc, by default None
    trajectory_slices : Dict[str, slice], optional
        a dictionary for trajectory name and its range expressed as a python slice
        object (similar to the usage of a [start:stop:stride] for indexing.), by default
        None
    unit_system : str | UnitSystem object, optional
        In format "[L]-[M]-[T]-[E(nergy)]" for units of builtin quantities, by default
        "nm-g/mol-ps-kJ/mol"; alternatively, you can provide an existing `UnitSystem` or
        a JSON-serialized such object

    Raises
    ------
    ValueError
        When input `coords` does not correspond to the input `top`.

    Notes
    -----
    Alternative to the default `__init__` method, an `Ensemble` can also be created from
    a `mdtraj.Trajectory` object (see `from_mdtraj_trj`).
    """

    def __init__(
        self,
        name,
        top,
        coords,
        quantities=None,
        metadata=None,
        trajectory_slices=None,
        unit_system="nm-g/mol-ps-kJ/mol",
    ):
        top_str = top2json(top)  # let `top2json` do the type check
        if (
            len(coords.shape) != 3
            or coords.shape[1] != top.n_atoms
            or coords.shape[2] != 3
        ):
            raise ValueError(
                f"Invalid input `coords`, expecting shape [N_frames, {top.n_atoms}, 3]."
            )
        if isinstance(unit_system, UnitSystem):
            # serialize and
            self._unit_system = UnitSystem.parse_from_json(
                unit_system.to_json()
            )
        else:
            self._unit_system = UnitSystem.parse_from_str(unit_system)
        super().__init__({}, metadata=metadata)
        # setting the coords properly
        preset_coords_unit = self.unit_system.get_preset_unit("coords")
        if isinstance(coords, BaseQuantity):
            coords = coords.convert_unit_to_(preset_coords_unit)
        else:
            print(
                f"Assuming the input `coords` to be in unit "
                f'"{preset_coords_unit}"'
            )
            # coords = Quantity(coords, preset_coords_unit)
            coords = _to_quantity(coords, preset_coords_unit)
        self._save_quantity(
            "coords",
            coords,
            shape=coords.shape,
        )  # overriding shape checks, which depend on coords themselves
        self._save_quantity("top", _to_quantity(top_str), shape=None)
        # self.top = top_str
        self.metadata["name"] = name
        self.metadata["unit_system"] = self._unit_system.to_json()
        if quantities is not None:
            for k, v in quantities.items():
                if k == "coords":
                    warn('Omitting input `quantities["coords"]`')
                    continue
                elif k == "top":
                    warn('Omitting input `quantities["top"]`')
                    continue
                self.__setattr__(k, v)
        if trajectory_slices is not None:
            self.register_trjs(trajectory_slices)
        elif "trjs" in self._data:
            # quantity `trjs` (json) has been set via quantities
            # in this case, we simply need to parse it from json
            try:
                self._set_trjs_dict_from_json(self._data["trjs"][()])
            except ValueError:
                raise ValueError(
                    "Invalid `trjs` (trajectories) information in the input quantities."
                )
        else:
            # register a default sequence for the whole length for compatibility
            self.register_trjs({"default": slice(None)})  # same as [:]
            self._data.pop("trjs")

    @staticmethod
    def from_mdtraj_trj(
        name,
        trj,
        quantities=None,
        metadata=None,
        trajectory_slices=None,
        unit_system="nm-g/mol-ps-kJ/mol",
    ):
        """Create an `Ensemble` instance from `mdtraj.Trajectory`.

        Parameters
        ----------
        name : str
            a human-readable name of the system. Not necessarily corresponding to the
            HDF5 group name
        trj : mdtraj.Trajectory
            A trajectory, whose topology and coordinates (and when applicable also the
            unit cell information) will be stored in the created `Ensemble`.
        quantities : Dict[str, np.ndarray | Quantity], optional
            Example key and value pairs for builtin quantities:
            - forces: (n_frames, n_atoms, 3) _ATOMIC_VECTOR_.
            - velocities: (n_frames, n_atoms, 3) _ATOMIC_VECTOR_ with dimension [L]/[T].
            - time: (n_frames,) _per-frame_ scalar indicating the elapsed simulation
                time with dimension [T].
            - weights: (n_frames,) _per-frame_ scalar indicating the Boltzmann weight of
            each frame.
        metadata : dict, optional
            Metadata to be saved, e.g., simulation temperature, forcefield information,
            saving time stride, etc, by default None
        trajectory_slices : Dict[str, slice], optional
            a dictionary for trajectory name and its range expressed as a python slice
            object (similar to the usage of a [start:stop:stride] for indexing.), by
            default None
        unit_system : str | UnitSystem object, optional
            In format "[L]-[M]-[T]-[E(nergy)]" for units of builtin quantities, by
            default "nm-g/mol-ps-kJ/mol"; alternatively, you can provide an existing
            `UnitSystem` or a JSON-serialized such object

        Returns
        -------
        Ensemble
            An instance containing all information from the mdtraj object together with
            the provided `quantities`, `metadata`, `trajectory_slices` and `unit_system`.

        Raises
        ------
        ValueError
        """

        coords = Quantity(trj.xyz, "nm")
        if quantities is None:
            quantities = {}
        if trj.time is not None:
            quantities["time"] = Quantity(trj.time, "ps")
        if trj.unitcell_angles is not None:
            quantities["cell_angles"] = Quantity(trj.unitcell_angles, "degree")
        if trj.unitcell_lengths is not None:
            quantities["cell_lengths"] = Quantity(trj.unitcell_lengths, "nm")
        return Ensemble(
            name,
            trj.top,
            coords,
            quantities=quantities,
            metadata=metadata,
            trajectory_slices=trajectory_slices,
            unit_system=unit_system,
        )

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
        """Return a the unit system used by the `Ensemble`.

        Returns
        -------
        UnitSystem
            The unit system containing units for basic dimension "[X]" (X = L, M, T, E)
            and units for builtin quantities
        """
        return self._unit_system

    @property
    def top(self):
        """Return the topology of the molecular system. Alias to `.top`.

        Returns
        -------
        mdtraj.Topology
        """
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
        """Return the topology of the molecular system. Alias to `.top`.

        Returns
        -------
        mdtraj.Topology
        """
        return self.top

    @property
    def n_atoms(self):
        """Return the number of atoms of the molecule defined in `top`.

        Returns
        -------
        int
            Number of atoms contained in the molecule.
        """
        return self.top.n_atoms

    @property
    def n_frames(self):
        """Return the number of frames.

        Returns
        -------
        int
            Number of frames contained in this `Ensemble`.
        """
        return self._data["coords"].shape[0]

    @property
    def trajectories(self):
        """Get the slices corresponding to each trajectory. These python slice objects
        can be used to retrive the correct portion for the corresponding trajectory for
        each _per-frame_ quantity (e.g., coords, forces, ...) via bracket ([]) operator.
        Alias to `.trajectory_slices`.

        Returns
        -------
        Dict[str, slice]
            Key-slice pairs for each registered trajectory
        """
        return self.trajectory_slices

    @property
    def trjs(self):
        """Get the slices corresponding to each trajectory. These python slice objects
        can be used to retrive the correct portion for the corresponding trajectory for
        each _per-frame_ quantity (e.g., coords, forces, ...) via bracket ([]) operator.
        Alias to `.trajectory_slices`.

        Returns
        -------
        Dict[str, slice]
            Key-slice pairs for each registered trajectory
        """
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
        """The input slices will be used to indicate which slice of the Ensemble samples
        (and other quantities) correspond to which trajectory.

        Parameters
        ----------
        dict_of_slices : Mapping[str, slice]
            Name and range of a trajectory.
        """
        trjs = {
            k: slice_.indices(self.n_frames)
            for k, slice_ in dict_of_slices.items()
        }
        self._save_quantity(
            "trjs",
            _to_quantity(json.dumps(trjs)),
            shape=None,
            verbose=hasattr(self, "_trjs"),
        )
        self._trjs = trjs

    @property
    def n_trjs(self):
        """Number of trajectories in the Ensemble.

        Returns
        -------
        int
        """
        if hasattr(self, "_trjs"):
            return len(self._trjs)
        else:
            return 0

    @property
    def n_frames_per_trj(self):
        """Number of frames contained by each trajectory.

        Returns
        -------
        int
        """
        if self.n_trjs == 0:
            return None
        else:
            # immutable to discourage assignment to the returned value
            return MappingProxyType(
                {
                    k: (trj[1] - trj[0]) // trj[2]
                    for k, trj in self._trjs.items()
                }
            )

    @property
    def trajectory_slices(self):
        """Get the slices corresponding to each trajectory. These python slice objects
        can be used to retrive the correct portion for the corresponding trajectory for
        each _per-frame_ quantity (e.g., coords, forces, ...) via bracket ([]) operator.

        Returns
        -------
        Dict[str, slice]
            Key-slice pairs for each registered trajectory
        """
        if self.n_trjs == 0:
            return None
        else:
            # immutable to discourage assignment to the returned value
            return MappingProxyType(
                {k: slice(*trj) for k, trj in self._trjs.items()}
            )

    @property
    def trajectory_indices(self):
        """Indices of frames belonging to each Trajectory.

        Returns
        -------
        Dict[str, numpy.ndarray]
            Key-(list of int) pairs for each registered trajectory
        """
        if self.n_trjs == 0:
            return None
        else:
            # immutable to discourage assignment to the returned value
            return MappingProxyType(
                {k: np.arange(*trj) for k, trj in self._trjs.items()}
            )

    def list_quantities(self):
        """List the name of quantities stored in the `Ensemble`.

        Returns
        -------
        List[str]
            Quantity names
        """
        return list(self._data)

    def get_quantity(self, key):
        """Retrieve a Quantity in the `Ensemble`.

        Parameters
        ----------
        key : str
            The name of the Quantity

        Returns
        -------
        Quantity
            The Quantity under the name `key` in the `Ensemble`

        Raises
        ------
        KeyError
            When `key` does not correspond to a Quantity existing in the current
            `Ensemble`.
        """
        if key in self:
            return self[key]
        else:
            raise KeyError(f"Quantity `{key}` does not exist")

    def get_unit(self, key):
        """Retrieve the builtin unit for a `Quantity` under name `key` in the Ensemble,
        or alternatively the preset unit of a builtin `Quantity`. If neither is the
        case, return `None`.

        Parameters
        ----------
        key : str
            Name of the Quantity

        Returns
        -------
        str | None
            The bulitin unit of the Quantity under name `key`
        """
        if key in self._data:
            return self.get_quantity(key).unit
        elif key in self.unit_system.builtin_quantities:
            preset_unit = self.unit_system.get_preset_unit(key)
            return preset_unit
        else:
            return None

    def __setattr__(self, key, quant):
        if key.startswith("_"):
            # not to block normal class initializations of `HDF5Group` which has `_data` and `_attrs` attributes
            object.__setattr__(self, key, quant)
            return
        self.set_quantity(key, quant)

    def __delattr__(self, key):
        if key in self._data:
            self._data.pop(key)
        super().__delattr__(key)

    def set_quantity(self, key, quant):
        """Store `quant` (Quantity | numpy.ndarray) under name `key` (str).
        When `quant` is a plain `numpy.ndarray`, the unit is assumed according to
        `.unit_system` if the `key` is one of the `PRESET_BUILTIN_QUANTITIES` (or
        `self.unit_system.builtin_quantities` in case of customized `unit_system` at
        initialization), or `dimensionless` otherwise.
        * When `key` is one of the `PRESET_BUILTIN_QUANTITIES` (or
        `self.unit_system.builtin_quantities`), the unit and shape of `quant`
        need to be compatible.

        Parameters
        ----------
        name : str
            The name/key to store the string.
        quantity : numpy.ndarray | Quantity
            The quantity to be saved. When input is a raw numpy array, the unit is
            assumed to be either the builtin unit (when exists) or "dimensionless".
        """
        # built-in quantities?
        if key in self.unit_system.builtin_quantities:
            shape_hint = self.unit_system.builtin_quantities[key][0]
            preset_unit = self.unit_system.get_preset_unit(key)
        else:
            shape_hint = None
            preset_unit = None
        # check and convert `quant` to a Quantity
        if not isinstance(quant, BaseQuantity):
            if preset_unit is None:
                preset_unit = "dimensionless"
            if preset_unit != "dimensionless":
                print(f'Assuming unit of input "{key}" to be "{preset_unit}".')
            quant = _to_quantity(quant, preset_unit)
        else:
            if isinstance(quant, BaseQuantity) and not isinstance(
                quant, Quantity
            ):
                # transform BaseQuantity (without metadata) to a Quantity
                quant = Quantity(quant.raw_value, unit=quant.unit)
            # convert to preset unit
            if preset_unit is not None:
                # print(f"Convert \"{key}\" to internal unit {preset_unit}")
                quant = quant.convert_unit_to_(preset_unit)
        self._save_quantity(key, quant, shape=shape_hint)

    def get_all_in_one_mdtraj_trj(self):
        """Pack all `coords` into one `Trajectory` object (maybe not suitable for
        kinetic analyses, such as TICA and MSM!)

        Returns
        -------
        mdtraj.Trajectory
            A trajectory, whose topology (and unitcell dimensions, if applicable) come
            from the `Ensemble` object and coordinates from `coords` concatenated
        """
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

    def get_mdtraj_trjs(self):
        """Pack this `Ensemble`'s `top` and `coords` (and unitcell + simulation times,
        if available) into a dictionary of `Trajectory` for analyses, according to
        the slicing given by `self.trjs`.

        Returns
        -------
        Dict[str, mdtraj.Trajectory]
            A dictionary containing all mdtraj Trjectories implied by the `self.trjs`.
        """
        trj = self.get_all_in_one_mdtraj_trj()
        return {k: trj[slice_] for k, slice_ in self.trajectory_slices.items()}

    def _save_quantity(
        self,
        name,
        quantity,
        shape="[n_frames, n_atoms, 3]",
        verbose=True,
    ):
        """Save a `quantity` under name `name` with the shape defined by `shape`.

        Parameters
        ----------
        name : str
            The name/key to store the string.
        quantity : Quantity
            The quantity to be saved.
        shape : str | Tuple, optional
            A shape string/tuple to indicate the allowed shape of input; for string,
            the "n_frames" and "n_atoms" will be substituted automatically to
            `Ensemble`'s property; shape check is bypassed when set to `None`, by
            default "[n_frames, n_atoms, 3]"
        verbose : bool, optional
            Whether to notify about overwriting an existing Quantity, by default True

        Raises
        ------
        ValueError
            When the shape check fails
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
                warn(f"Overwriting the previously saved record {name}.")
                if not quantity.is_unit_convertible_with(old_quant):
                    warn(
                        f"Overwriting record {name} with incompatible unit: "
                        f"{old_quant.unit} -> {quantity.unit}."
                    )
        # save data
        self._data[name] = quantity
        self.__dict__[name] = quantity.raw_value

    @classmethod
    def from_hdf5(
        cls,
        h5grp,
        unit_system: str = "nm-g/mol-ps-kJ/mol",
        offset: Optional[int] = None,
        stride: Optional[int] = None,
        no_offset_stride_quantities: Optional[List[str]] = None,
    ):
        """Create an instance from the content of HDF5 Group `h5grp` (h5py.Group).
        When given `unit_system` differs from the stored record, units will be converted
        when reading the `Quantity` into memory. Input `offset` and `stride` can be set
        to allow a partial loading of the non-scalar datasets with indexing
        `[offset::stride]`, unless the dataset's name is contained in
        `no_offset_stride_quantities`.

        Parameters
        ----------
        h5grp : h5py.Group
            The group should contain all necessary information to initialize an
            `Ensemble`. Notablly, the following Datasets are required:
            - top: serialized topology
            - coords (with Attribute "unit")
            And all Datasets should come with an Attribute `unit` for the physical unit
            used in storage.
        unit_system : str, optional
            Should have the format "[L]-[M]-[T]-[E(nergy)]" for units of builtin
            quantities (see class docstring), by default "nm-g/mol-ps-kJ/mol"
        offset : None | int, optional
            The offset for loading from the HDF5 file. Default is `None`.
        stride : None | int, optional
            The stride for loading from the HDF5 file. Default is `None`.
        no_offset_stride_quantities : List[str], optional
            The names of entries, for which no offset or stride should be considered
            during loading, by default None

        Returns
        -------
        Ensemble
            An instance containing all compatible content from HDF5 Group `h5grp`.

        Raises
        ------
        ValueError
            When the Dataset corresponding to `top`, `coords` or other fields does not
            exist or has invalid format.
        """
        hdf5grp = HDF5Group.from_hdf5(
            h5grp,
            offset=offset,
            stride=stride,
            no_offset_stride_quantities=no_offset_stride_quantities,
        )
        dataset_unit_system_str = str(
            UnitSystem.parse_from_str(hdf5grp.metadata["unit_system"])
        )
        if unit_system is None:
            # no unit system given, fall back to the unit system stored in the h5 file
            formatted_unit_system_str = dataset_unit_system_str
        else:
            formatted_unit_system_str = str(
                UnitSystem.parse_from_str(unit_system)
            )
            if dataset_unit_system_str != formatted_unit_system_str:
                print(
                    f'Adapting unit system from "{dataset_unit_system_str}" (storage) '
                    f'to "{formatted_unit_system_str}" (memory).'
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
            if k != "coords" and k != "top":
                other_quantities[k] = v
        # adapt the `trjs` records to the striding
        # Only implemented for simple cases where the `start` and `stop` are both
        # nonnegative and `step` = 1 for slice `[start:stop:step]`
        if "trjs" in other_quantities and (
            offset is not None or stride is not None
        ):
            if offset is None:
                offset = 0
            if stride is None:
                stride = 1
            from math import ceil

            try:
                trjs_dict = json.loads(other_quantities["trjs"][()])
            except json.JSONDecodeError:
                raise ValueError(
                    "Invalid `trjs` (trajectories) information in the H5 dataset "
                    "(invalid JSON)?"
                )
            new_trjs_dict = {}
            for slice_name in trjs_dict:
                start, stop, step = trjs_dict[slice_name]
                if start < 0 or stop < 0 or step != 1:
                    warn(
                        "Auto adjusting of the `trjs` records is only implemented for "
                        "simple cases where all slices has nonnegative `start`s and "
                        "`stop`s and the `step` is 1. (as for a slice operation "
                        "`[start:stop:step]`) Therefore, the `trjs` records are "
                        "discarded due to inconsistency."
                    )
                    new_trjs_dict = {}
                new_range = [
                    ceil(max(start - offset, 0) / stride),
                    ceil(max(stop - offset, 0) / stride),
                    1,
                ]
                if (new_range[1] - new_range[0]) > 0:
                    # non-empty slice
                    new_trjs_dict[slice_name] = new_range
            if len(new_trjs_dict) == 0:
                other_quantities.pop("trjs")
            else:
                other_quantities["trjs"] = Quantity(json.dumps(new_trjs_dict))
        try:
            new_ensemble = cls(
                name,
                top,
                coords,
                quantities=other_quantities,
                metadata=hdf5grp.metadata,
                unit_system=formatted_unit_system_str,
            )
        except ValueError:
            raise ValueError(
                f"Invalid information in the input HDF5 Group `{h5grp.name}`."
            )
        return new_ensemble

    def _set_trjs_dict_from_json(self, trjs_dict_in_json):
        """Set the `trjs` quantity (as json str) and `_trjs` entry (as Dict[List[int]])
        coherently from input json string.

        Parameters
        ----------
        trjs_dict_in_json : str
            A json string that contains valid trajectory record as a serialized
            Dict[List[int]].

        Raises
        ------
        ValueError
            When input `trjs_dict_in_json` is not a valid json str.
        """
        try:
            trjs_dict = json.loads(trjs_dict_in_json)
        except json.JSONDecodeError:
            raise ValueError(
                "Invalid `trjs` (trajectories) information in the input json."
            )
        self._save_quantity(
            "trjs",
            _to_quantity(trjs_dict_in_json),
            shape=None,
            verbose=False,  # always False, since not called except for init
        )
        self._trjs = trjs_dict

    def __repr__(self):
        return (
            f'<Ensemble for molecule "{self.name}" with {self.n_atoms} atoms and '
            f"{self.n_frames} frames>"
        )
