"""Featurizer takes a proteka.dataset.Ensemble and and extract features from it"""
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
import json
import warnings
import numpy as np
import mdtraj as md
from ..dataset import Ensemble
from ..quantity import Quantity
from ..dataset.top_utils import top2json
from typing import Callable, Dict, List, Optional, Tuple, Union
from itertools import combinations
from .utils import reduce_atom_pairs_by_residue_offset

__all__ = ["Featurizer", "Transform", "TICATransform"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clipped_sigmoid(x: np.ndarray, x_min: float = -80, x_max: float = 80):
    """Returns the sigmoid, but input values are capped and output is scaled.

    Input values (x) are capped at x_min and x_max, and the output is linearly
    scaled to have a range of [0,1]. Continuous, but not smooth; useful to
    avoid overflow errors.

    Parameters
    ----------
    x: np.ndarray
        Values to transform via modified sigmoid
    x_min: float
        Minimum to cap values at.
    x_max: float
        Maximum to cap values at.


    Returns
    -------
    np.ndarray of transformed values.
    """

    min_sig = sigmoid(x_min)
    max_sig = sigmoid(x_max)
    val = sigmoid(np.clip(x, a_min=x_min, a_max=x_max))
    return (val - min_sig) / (max_sig - min_sig)


class Transform(ABC):
    """Abstract transformer class that defines a transformation of the
    data. The class should be serializable, so it can be stored in Ensemble
    """

    @abstractmethod
    def transform(self, Ensemble):
        """Transform ensemble into a new set of features.
        The result should be returned as a numpy array
        """

    @abstractmethod
    def to_json(self):
        """
        Serialize object as a json string
        """

    @abstractmethod
    def from_json(self, string):
        """
        Instantiate Transformer from a json string
        """


class TICATransform(Transform):

    """Get TICA transform of the data.
    A feature vector X is transformed as
    (X - bias)@transform_matrix
    """

    def __init__(
        self,
        features: List[Tuple],
        bias: Optional[np.ndarray] = None,
        transform_matrix: Optional[np.ndarray] = None,
        estimation_params: Optional[Dict] = None,
    ):
        """

        Parameters
        ----------
        features : List[Dict]
            List of features to be used for TICA.
            Each element is a Tuple whose first element is a string representing feature
            name and whose second element is a dictionary of parameters used to compute corresponding feature
            (see Featurizer.get_feature for details). The order of features input
            for TICA model parametrization/transforming follows the same order in
            the supplied List.
        bias : np.ndarray, Optional
            Bias used to compute the TICA transformation. If not provided, it will be infrred from data
            during the transformation.
        transform_matrix : np.ndarray, Optional
            Transformation matrix used to compute the TICA transformation.
            If not provided, it will be inferred from data.
        estimation_params: Optional[Dict]
            Parameters used to estimate TICA model. See `deeptime.decomposition.TICA` for details
            If bias and transform_matrix are provided, estimation_params are ignored

        """
        if not isinstance(features, List):
            raise ValueError(
                "Input features must be an List to preserve TICA feature order"
            )
        if not all([isinstance(f, Tuple) for f in features]) or not all(
            [len(f) == 2 for f in features]
        ):
            raise ValueError(
                f"All elements in `features` must be length-2 Tuples, but {features} was received"
            )
        else:
            self.features = features
        self.bias = bias
        self.transform_matrix = transform_matrix
        self.estimation_params = estimation_params

    def fit_from_data(self, ensemble: Ensemble):
        """
        Fit TICA model from data, using Deeptime library
        """
        from deeptime.decomposition import TICA

        features = []
        for feature, params in self.features:
            features.append(Featurizer.get_feature(ensemble, feature, **params))
        features = np.concatenate(features, axis=1)
        estimator = TICA(**self.estimation_params)
        # Loop over trajectories in ensemble, get corresponding slice and perform
        # partial fit
        for slice in ensemble.trajectory_indices.values():
            estimator.partial_fit(features[slice])
        model = estimator.fetch_model()

        # Set values that are required for further transformation
        self.bias = model.instantaneous_obs.obs1.mean
        self.transform_matrix = model.instantaneous_obs.obs1.sqrt_inv_cov

    def transform(self, ensemble: Ensemble) -> np.ndarray:
        """
        Extract TICA features from ensemble
        """
        if self.transform_matrix is None or self.bias is None:
            self.fit_from_data(ensemble)
        elif self.estimation_params is not None:
            warnings.warn(
                "Transform matrix and bias are provided, ignoring estimation_params"
            )
        features = []
        for feature_tuple in self.features:
            feature, params = feature_tuple[0], feature_tuple[1]
            features.append(Featurizer.get_feature(ensemble, feature, **params))
        features = np.concatenate(features, axis=1)
        return (features - self.bias) @ self.transform_matrix

    def to_dict(self, arrays2list: bool = True) -> Dict:
        """Generate dictionary from the class instance"""
        if arrays2list:
            bias = self.bias.tolist()
            transform_matrix = self.transform_matrix.tolist()
        else:
            bias = self.bias
            transform_matrix = self.transform_matrix
        return {
            "features": self.features,
            "bias": bias,
            "transform_matrix": transform_matrix,
            "estimation_params": self.estimation_params,
        }

    def to_json(self):
        """Serialize transformer to json string"""
        return json.dumps(self.to_dict(arrays2list=True))

    @classmethod
    def from_dict(cls, input_dict: Dict) -> Transform:
        """Instantiate transformer from a dictionary"""
        return cls(
            features=input_dict["features"],
            bias=np.array(input_dict["bias"]),
            transform_matrix=np.array(input_dict["transform_matrix"]),
            estimation_params=input_dict["estimation_params"],
        )

    @classmethod
    def from_json(cls, input_string: str) -> Transform:
        """
        Instantiate Transformer from a json string
        """
        return cls.from_dict(json.loads(input_string))

    @staticmethod
    def from_hdf5(self, h5file: str) -> Transform:
        """
        Instantiate Transformer from a hdf5 file
        """
        raise NotImplementedError


class Featurizer:
    """Class for computing features from Ensembles.
    The class has no state, all the details of featurization
    should be passed directly to"""

    simple_dssp_lookup = {
        "NA": 0,
        "H": 1,
        "E": 2,
        "C": 3,
    }

    full_dssp_lookup = {
        "NA": 0,
        "H": 1,
        "B": 2,
        "E": 3,
        "G": 4,
        "I": 5,
        "T": 6,
        "S": 7,
        " ": 8,
    }

    def __init__(self):
        pass

    @staticmethod
    def validate_c_alpha(ensemble: Ensemble) -> bool:
        """Check if C-alpha-based metrics can be computed"""
        ca_atoms = ensemble.top.select("name CA")
        # Should have at least 2 CA atoms in total to compute CA-based statistics
        assert len(ca_atoms) > 1, "Number of CA atoms is less than 2"

        # Should have one CA atom per each residue
        assert (
            len(ca_atoms) == ensemble.top.n_residues
        ), "Number of CA atoms does not match the number of residues"

        # Should have only one chain
        if ensemble.top.n_chains > 1:
            raise NotImplementedError(
                "More than one chain is not supported yet"
            )

        # Should have no breaks in chain (i.e. no missing residues):
        for chain in ensemble.top.chains:
            assert (
                chain.n_residues
                == chain.residue(chain.n_residues - 1).resSeq
                - chain.residue(0).resSeq
                + 1
            ), "Chain has missing residues"

    @staticmethod
    def _get_consecutive_ca(topology: md.Topology, order: int = 2):
        """Get all subsequences of consecutive CA atoms of length `order`
        (pairs, triplets, etc.)

        Each subsequence comes from a single chain. There are no breaks within subsequences.
        It is assumed that residues in the topology are sorted in ascending order.

        Parameters
        ----------
        topology : md.Topology
            Topology object
        order : int, optional
            Number of consecutive atoms, by default 2
        """
        consecutives = []
        for chain in topology.chains:
            # In a chain, a maximum of chain.n_residues - order + 1 subsequences
            # of length order can be possible
            # For example, if order = 2, then the maximum number of pairs is
            # chain.n_residues - 2 + 1 = chain.n_residues - 1
            for i in range(chain.n_residues - order + 1):
                atom_list = []
                # The inner loop here is needed for testing for breaks.
                # If there is a break, then the `resSeq` of the current residue and the
                # previous residue differ by more than 1. In this case, we break out of
                # the inner loop and move on to the next residue.
                # Of course, it only makes sense to do comparison with the previous residue
                # if the current residue is not the first residue in the subsequence,
                # i.e. if `j != 0`.
                for j in range(order):
                    res = chain.residue(i + j)
                    if (j != 0) and (
                        res.resSeq - chain.residue(i + j - 1).resSeq != 1
                    ):
                        break
                    atom = res.atom("CA")
                    atom_list.append(atom.index)
                consecutives.append(atom_list)
        return consecutives

    def add(self, ensemble: Ensemble, feature: str, *args, **kwargs):
        """Add a new feature to the Ensemble object"""
        if hasattr(self, "add_" + feature):
            getattr(self, "add_" + feature)(ensemble, *args, **kwargs)
        else:
            allowed_features = [
                attr for attr in dir(self) if attr.startswith("add_")
            ]
            raise ValueError(
                f"Feature {feature} is not supported. Supported features are: {allowed_features}"
            )

    def add_ca_bonds(self, ensemble: Ensemble):
        """
        Create a Quantity object that contains length of pseudobonds
        between consecutive CA atoms.
        """
        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        self.validate_c_alpha(ensemble)

        # Get the pairs of consecutive CA atoms
        ca_pairs = self._get_consecutive_ca(ensemble.top, order=2)
        ca_bonds = md.compute_distances(trajectory, ca_pairs, periodic=False)
        quantity = Quantity(
            ca_bonds, "nanometers", metadata={"feature": "ca_bonds"}
        )
        ensemble.set_quantity("ca_bonds", quantity)

    def add_ca_distances(
        self, ensemble: Ensemble, offset: int = 1, subset_selection: str = None
    ):
        """Get distances between CA atoms.

        Parameters:
        ensemble : Ensemble
            Ensemble object
        offset: int, optional
            Distances between CA atoms in residues i and j are included if
            | i -j | >  offset. The default value is 1, which means that
            all nonbonded Calpha-Calpha distances are included. If offset is 0,
            then all the distances are included.
        subset_selection: str, optional
            A string that is used to select a subset of CA atoms, according to mdtraj
            selection language, e.g. "chainid 0 and resid 0 to 10". If `subset_selection` is None,
            then all CA atoms compatible with the specified offset are used.

        """
        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        if subset_selection is not None:
            selection = "name CA and " + subset_selection
        else:
            selection = "name CA"
        ca_atoms = trajectory.top.select(selection)
        self.validate_c_alpha(ensemble)
        # get all the pairs
        ca_pairs_all = list(combinations(ca_atoms, 2))
        # select pairs compatible with offset
        ca_pairs = list(
            filter(
                lambda x: abs(
                    trajectory.top.atom(x[0]).residue.resSeq
                    - trajectory.top.atom(x[1]).residue.resSeq
                )
                > offset,
                ca_pairs_all,
            )
        )
        # Compute distances
        ca_distances = md.compute_distances(
            trajectory, ca_pairs, periodic=False
        )
        quantity = Quantity(
            ca_distances,
            "nanometers",
            metadata={
                "feature": "ca_distances",
                "offset": offset,
                "subset_selection": subset_selection,
            },
        )
        ensemble.set_quantity("ca_distances", quantity)

    def add_ca_angles(self, ensemble: Ensemble):
        """Get angles between consecutive CA atoms"""
        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        self.validate_c_alpha(ensemble)

        # Get the triplets of consecutive CA atoms
        ca_triplets = self._get_consecutive_ca(ensemble.top, order=3)
        ca_angles = md.compute_angles(trajectory, ca_triplets, periodic=False)
        quantity = Quantity(
            ca_angles,
            "radians",
            metadata={"feature": "ca_angles"},
        )
        ensemble.set_quantity("ca_angles", quantity)

    def add_ca_dihedrals(self, ensemble: Ensemble):
        """Get dihedral angles between consecutive CA atoms"""
        self.validate_c_alpha(ensemble)
        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        # Get the quadruplets of consecutive CA atoms
        ca_quadruplets = self._get_consecutive_ca(ensemble.top, order=4)
        ca_dihedrals = md.compute_dihedrals(
            trajectory, ca_quadruplets, periodic=False
        )
        quantity = Quantity(
            ca_dihedrals, "radians", metadata={"feature": "ca_dihedrals"}
        )
        ensemble.set_quantity("ca_dihedrals", quantity)

    def add_phi(self, ensemble: Ensemble):
        """Get protein backbone phi torsions"""
        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        _, phi = md.compute_phi(trajectory)
        quantity = Quantity(phi, "radians", metadata={"feature": "phi"})
        ensemble.set_quantity("phi", quantity)

    def add_psi(self, ensemble: Ensemble):
        """Get protein backbone psi torsions"""
        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        _, psi = md.compute_psi(trajectory)
        quantity = Quantity(psi, "radians", metadata={"feature": "psi"})
        ensemble.set_quantity("psi", quantity)

    def add_rmsd(
        self,
        ensemble: Ensemble,
        reference_structure: md.Trajectory,
        atom_selection: Optional[str] = None,
        **kwargs,
    ):
        """Get RMSD of a subset of atoms
        reference: Reference mdtraj.Trajectory object
        Wrapper of mdtraj.rmsd

        Parameters
        ----------
        reference_structure:
            Reference structure from which RMSD calculations are made
        atom_selection:
            MDTraj atom selection phrase to specify which atoms should contribute
            to the RMSD calculation for both the target AND the reference structure.
            This option can be used to ensure that consitent atom sets are used for
            targets and references at different resolutions.
        kwargs:
            kwarg options for `mdtraj.rmsd()`,
            for example `{"frame": 0, "atom_indices": np.arange(10), "parallel": True,
            "precentered": False, "ref_atom_indices": np.arange(10,20)}`. See
            help(mdtraj.rmsd) for more information. Note that if `atom_selection` is
            specified, kwargs cannot contain atom_indices or ref_atom_indices entries
            or a TypeError will be raised by mdtraj.rmsd.
        """

        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        # store for serialization
        ref_coords = reference_structure.xyz.tolist()
        ref_top = top2json(reference_structure.topology)

        if atom_selection is not None:
            target_inds = trajectory.topology.select(atom_selection)
            ref_inds = reference_structure.topology.select(atom_selection)
            assert len(target_inds) == len(ref_inds)

            # If atom_indices/ref_atom_indices is accidently repeated in
            # **kwargs, it will be caught by a SyntaxError :)
            rmsd = md.rmsd(
                trajectory,
                reference_structure,
                atom_indices=target_inds,
                ref_atom_indices=ref_inds,
                **kwargs,
            )
        else:
            rmsd = md.rmsd(
                trajectory,
                reference_structure,
                **kwargs,
            )

        metadata = {
            "feature": "rmsd",
            "reference_structure_coords": ref_coords,
            "reference_structure_top": ref_top,
            "atom_selection": atom_selection,
        }

        # Cannot use `update` method if kwargs is None/{}
        if kwargs is not None:
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    metadata[k] = v.tolist()
                else:
                    metadata[k] = v

        quantity = Quantity(rmsd, "nanometers", metadata=metadata)
        ensemble.set_quantity("rmsd", quantity)

    def add_helicity(
        self,
        ensemble: Ensemble,
        sigma_squared: float = 0.02,
        r_0: float = 0.5,
        residue_indices: Optional[Iterable[int]] = None,
    ):
        """Gets helicity for specified ensemble, as described by Rudzinski and Noid
        (2015, https://pubs.acs.org/doi/pdf/10.1021/ct5009922) :

            Q_helix = (1.0/N) * \\sum_{ij in 1,4 pairs} exp(-((1.0)/(2.0*sigma_squared))*(r_{ij}-r_0)^2)

        where `N` is the number of possible 1-4 carbon alpha pairs in the molecule
        (up to the specified residues indices) and the constants `sigma_squared`
        and `r_0` are 0.02 nm^2 and 0.5 nm respectively by default. If `residue_indices`
        is specified, only the carbon alpha atoms in those specified residues will be
        included in the calculation of the 1-4 distances used in the helicity computation.

        Parameters
        ----------
        ensemble:
            Ensemble for which helicity should be computed
        sigma_squared:
            Helical variance, 0.02 nm^2 by default.
        r_0:
            Helical 1-4 average distance, 0.5 nm by default
        residue_indices:
            List of zero-based, integer indices that specify which residues should
            be considered for computing 1-4 carbon alpha pairs
        """

        traj = ensemble.get_all_in_one_mdtraj_trj()
        ca_idx = traj.topology.select("name CA")
        ca_atoms = np.array(list(traj.topology.atoms))[ca_idx]
        if residue_indices is not None:
            ca_idx = np.array(
                [
                    i
                    for i in ca_idx
                    if ca_atoms[i].residue.index in residue_indices
                ]
            )
        # 1-4 pair formation
        pairs_1_4 = []
        pairs = list(combinations(np.arange(len(ca_idx)), 2))
        for pair in pairs:
            a1, a2 = ca_atoms[pair[0]], ca_atoms[pair[1]]
            if np.abs(a1.residue.index - a2.residue.index) == 3:
                pairs_1_4.append((pair[0], pair[1]))

        if len(pairs_1_4) == 0:
            raise RuntimeError(
                "No 1-4 carbon alpha pairs found in the topology and for the residue selection"
            )

        n_1_4 = len(pairs_1_4)
        distances_1_4 = md.compute_distances(traj, pairs_1_4, periodic=False)
        helicity = (1.0 / n_1_4) * np.exp(
            -((1.0) / (sigma_squared)) * ((distances_1_4 - r_0) ** 2)
        )
        helicity = np.sum(helicity, axis=-1)

        quantity = Quantity(
            helicity,
            "dimensionless",
            metadata={
                "feature": "helicity",
                "sigma_squared": sigma_squared,
                "r_0": r_0,
                "residue_indices": residue_indices,
            },
        )
        ensemble.set_quantity("helicity", quantity)

    def add_rg(
        self, ensemble: Ensemble, atom_selection: Optional[str] = None, **kwargs
    ):
        """Get radius of gyration for each structure in an ensemble. Additional
        kwargs are passed to `mdtraj.compute_rg` - see
        https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_rg.html for
        more details.

        Parameters
        ----------
        ensemble:
            Ensemble for which the radius of gyration should be computed
        atom_selection:
            MDTraj atom selection string specifying certain subsets of atoms
            to contribute to the radius of gyration calculation (eg, "name CA" for
            only using carbon alpha atoms)
        """

        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        if atom_selection != None:
            trajectory = trajectory.atom_slice(
                trajectory.topology.select(atom_selection)
            )
        rg = md.compute_rg(trajectory, **kwargs)

        metadata = {"feature": "rg", "atom_selection": atom_selection}

        # Cannot use `update` method if kwargs is None/{}
        if kwargs is not None:
            for k, v in kwargs.items():
                if isinstance(v, np.ndarray):
                    metadata[k] = v.tolist()
                else:
                    metadata[k] = v

        quantity = Quantity(rg, "nanometers", metadata=metadata)
        ensemble.set_quantity("rg", quantity)

    def add_fraction_native_contacts(
        self,
        ensemble: Ensemble,
        reference_structure: md.Trajectory,
        beta: float = 50,
        lam: float = 1.8,
        native_cutoff: float = 0.45,
        res_offset: int = 3,
        atom_selection: str = "all and not element H",
        use_atomistic_reference: bool = True,
        rep_atoms: Optional[List[str]] = None,
        return_pairs: bool = False,
    ) -> Union[None, Dict]:
        """Gets fraction of native contacts according to the method
        defined in Best, Hummer, and Eaton (2013). This method allows for two
        ways to compute contacts, controlled by the `use_atomistic_reference` keyword
        argument (see below). If `use_atomistic_reference=True`, the contacting residues
        are determined at the all atom resolution, and equilibrium contact distances are
        determined for a set of representative atoms (thereby allowing native contacts to
        be defined for coarse grain representations. Else, the reference and the supplied
        ensemble are assumed to be at the same resolution, and the contacts are defined and
        computed based on the suppled atom selection.

        Parameters
        ----------
        ensemble:
            Ensemble object for which the fraction of native contacts should be computed
        refernce_structure:
            Single frame reference md.Trajectory defining the native state
        beta:
            Contact smoothing parameter. Set to 50 nm^-1 be default
        lam:
            Contact fluctuation allowance factor. Set to 1.8 by default. For coarse grain
            models, consider using a smaller factor, between 1.2-1.5.
        native_cutoff:
            Float specifying contact distance threshold. Set to 0.45 nm by default. If
        res_offset:
            Minimum residue |i-j| distance for two atoms to be considered for contact
            set elegibility. By default, only atoms more than three residues apart are
            considered.
        atom_selection:
            MDTraj atom selection string specifying which atoms should be used to define
            contacts in the reference. In the case of `use_atomistic_reference=True`, this
            controls the selection for defining reference structure contacts only (for example, to get all
            heavy atoms, supply "all and not element H"), and the atom selection for the ensemble
            is governed by `rep_atoms`. In the case that `use_atomistic_reference=False`, this atom
            selection is applied mutually to both the reference structure and the ensemble.
        use_atomistic_reference:
            If True, `reference_structure` is assumed to be an atomistic structure, and
            residues in contact will be determined using the `atom_selection` kwarg applied
            to `reference_structure`. The *final* contact distances and pairs will be defined by the
            `rep_atoms` kwarg, and the ensemble distances will also be calculated for these representative
            atoms. This option is useful for defining contacts for Go or other CG models. Eg, the residues
            in contact can be found by the `atom_selection="heavy"` strategy for the all-atom reference, and
            the native distances can be computed for the corresponding carbon alpha pairs with
            `rep_atoms=["CA"]`, and the same carbon alpha distances will be thusly computed for the model
            ensemble for the same residue pairs.
        rep_atoms:
            List of MDTraj atom names to define *final* contact distances for both the reference structure
            and the model ensemble. Only used if `use_atomistic_reference=True`. If not set, only CA atoms
            will be used.
        return_pairs;
            If true, a dictionary keyed by `"ref_atom_pairs"` and `"model_atom_pairs"`, containing the
            reference contact atom index pairs and the model contact atom index pairs respectively
            is returned.

        Returns
        -------
        dictionary:
            If `return_pairs=True`, the dictionary specified by the `return_pairs` parameter above
            is returned.
        """

        if reference_structure.n_frames != 1:
            raise ValueError(
                f"Native structure has {reference_structure.n_frames} frames, but should only have 1."
            )

        # get reference coordinates and ref/ens topologies
        traj = ensemble.get_all_in_one_mdtraj_trj()
        native_coords = reference_structure.xyz  # for serialization
        native_top = reference_structure.topology
        traj_top = traj.topology

        # protein length check
        assert len(list(native_top.residues)) == len(list(traj_top.residues))

        # get atom/residue lists for reference and ensemble
        model_atoms = list(traj_top.atoms)
        model_residues = list(traj_top.residues)
        ref_atoms = list(native_top.atoms)
        ref_residues = list(native_top.residues)

        # rep_atom checks
        if rep_atoms is None:
            rep_atoms = ["CA"]
        for ra in rep_atoms:
            assert ra in set([atom.name for atom in ref_atoms])
            assert ra in set([atom.name for atom in model_atoms])

        if use_atomistic_reference == True:
            # get types of atoms used for defining contacts
            selection_all = native_top.select(atom_selection)

            # residue offset filter
            all_ref_pairs = list(combinations(selection_all, 2))
            ref_res_filtered_atom_pairs = np.array(
                reduce_atom_pairs_by_residue_offset(
                    ref_atoms, all_ref_pairs, res_offset
                )
            )

            # compute the distances between filtered pairs and get the pairs within cutoff
            ref_distances = md.compute_distances(
                reference_structure, ref_res_filtered_atom_pairs, periodic=False
            )
            contacts_all = ref_res_filtered_atom_pairs[
                ref_distances.squeeze() < native_cutoff
            ].squeeze()

            # Get residue indices in contact, store them
            # And store the chosen representative atom pairs for each residue pair
            residue_pairs = []
            native_contacts = []
            for atom_pair in contacts_all:
                res_1, res_2 = (
                    ref_atoms[atom_pair[0]].residue.index,
                    ref_atoms[atom_pair[1]].residue.index,
                )
                if sorted([res_1, res_2]) not in residue_pairs:
                    residue_pairs.append(sorted([res_1, res_2]))
                    # store requested representative atoms
                    res_1_rep_atoms = []
                    res_2_rep_atoms = []
                    for ra in rep_atoms:
                        res1_atoms = list(ref_residues[res_1].atoms_by_name(ra))
                        res2_atoms = list(ref_residues[res_2].atoms_by_name(ra))
                        # account for cases where the requested atom is missing
                        if not (len(res1_atoms) == len(res2_atoms) == 1):
                            if len(res1_atoms) == 0 or len(res2_atoms) == 0:
                                continue  # skip because pair cannot be formed
                            else:
                                raise RuntimeError(
                                    f"Residue {ref_residues[res_1]} or residue {ref_residues[res_2]} has a repeated {ra} atom"
                                )
                        res_1_rep_atoms.append(res1_atoms[0])
                        res_2_rep_atoms.append(res2_atoms[0])

                    all_ref_rep_atom_pairs = []
                    for atom1 in res_1_rep_atoms:
                        for atom2 in res_2_rep_atoms:
                            if [
                                atom1.index,
                                atom2.index,
                            ] not in all_ref_rep_atom_pairs:
                                all_ref_rep_atom_pairs.append(
                                    [atom1.index, atom2.index]
                                )
                    native_contacts.extend(all_ref_rep_atom_pairs)

            # Now we do the exact same for the ensemble atoms, for the same residues in contact
            traj_native_pairs = []
            for res_pair in residue_pairs:
                res_1, res_2 = res_pair[0], res_pair[1]
                res_1_rep_atoms = []
                res_2_rep_atoms = []
                for ra in rep_atoms:
                    res1_atoms = list(model_residues[res_1].atoms_by_name(ra))
                    res2_atoms = list(model_residues[res_2].atoms_by_name(ra))
                    # account for cases where the requested atom is missing
                    if not (len(res1_atoms) == len(res2_atoms) == 1):
                        if len(res1_atoms) == 0 or len(res2_atoms) == 0:
                            continue  # skip because pair cannot be formed
                        else:
                            raise RuntimeError(
                                f"Residue {ref_residues[res_1]} or residue {ref_residues[res_2]} has a repeated {ra} atom"
                            )
                    res_1_rep_atoms.append(res1_atoms[0])
                    res_2_rep_atoms.append(res2_atoms[0])

                all_model_rep_atom_pairs = []
                for atom1 in res_1_rep_atoms:
                    for atom2 in res_2_rep_atoms:
                        if (
                            sorted([atom1.index, atom2.index])
                            not in all_model_rep_atom_pairs
                        ):
                            all_model_rep_atom_pairs.append(
                                sorted([atom1.index, atom2.index])
                            )
                traj_native_pairs.extend(all_model_rep_atom_pairs)

            # checks: make sure atom pairs are the same atoms
            # and come from the same residues
            assert len(native_contacts) == len(traj_native_pairs)
            for rp, mp in zip(native_contacts, traj_native_pairs):
                assert (
                    ref_atoms[rp[0]].name == model_atoms[mp[0]].name
                ), f"mismatch between ref atom {ref_atoms[rp[0]]} and model atom {model_atoms[mp[0]]}"
                assert (
                    ref_atoms[rp[1]].name == model_atoms[mp[1]].name
                ), f"mismatch between ref atom {ref_atoms[rp[1]]} and model atom {model_atoms[mp[1]]}"

                assert (
                    ref_atoms[rp[0]].residue.index
                    == model_atoms[mp[0]].residue.index
                ), f"mismatch between ref residue {ref_atoms[rp[0]].residue} and model residue {model_atoms[mp[0]].residue}"
                assert (
                    ref_atoms[rp[1]].residue.index
                    == model_atoms[mp[1]].residue.index
                ), f"mismatch between ref residue {ref_atoms[rp[1]].residue} and model residue {model_atoms[mp[1]].residue}"

        else:
            # Don't assume atomistic reference for defining residues in contact
            # Instead use mutual atom selection chosen for reference and ensemble
            reference_idx = reference_structure.top.select(atom_selection)
            traj_idx = traj.top.select(atom_selection)
            assert len(reference_idx) == len(traj_idx)

            ref_atoms = np.array(list(reference_structure.top.atoms))
            traj_atoms = np.array(list(traj.top.atoms))

            # check to see if atom lists are the same atoms and and come from the same residues
            for i1, i2 in zip(reference_idx, traj_idx):
                assert ref_atoms[i1].name == traj_atoms[i2].name
                assert (
                    ref_atoms[i1].residue.index == traj_atoms[i2].residue.index
                )

            all_ref_pairs = np.array(list(combinations(reference_idx, 2)))
            all_traj_pairs = np.array(list(combinations(traj_idx, 2)))
            assert len(all_ref_pairs) == len(all_traj_pairs)

            # Filter based on residue_offset
            filtered_ref_pairs = np.array(
                reduce_atom_pairs_by_residue_offset(
                    ref_atoms, all_ref_pairs, res_offset
                )
            )
            filtered_traj_pairs = np.array(
                reduce_atom_pairs_by_residue_offset(
                    traj_atoms, all_traj_pairs, res_offset
                )
            )

            nat_dist = md.compute_distances(
                reference_structure, filtered_ref_pairs, periodic=False
            )
            nat_idx = np.argwhere(nat_dist.squeeze() < native_cutoff).squeeze()
            native_contacts = filtered_ref_pairs[nat_idx]
            traj_native_pairs = filtered_traj_pairs[nat_idx]

        # now compute distances
        r_0 = md.compute_distances(
            reference_structure, native_contacts, periodic=False
        ).astype("float64")
        traj_nat_dist = md.compute_distances(
            traj, traj_native_pairs, periodic=False
        ).astype("float64")

        # compute native contacts for the same pairs
        q = np.mean(
            1.0 / (1.0 + np.exp(beta * (traj_nat_dist - lam * r_0))), axis=1
        )

        metadata = {
            "feature": "fraction_native_contacts",
            "reference_structure_coords": native_coords,
            "reference_structure_top": top2json(native_top),
            "reference_structure_is_atomistic": use_atomistic_reference,
            "atom_selection": atom_selection,
            "beta": beta,
            "lam": lam,
            "native_cutoff": native_cutoff,
            "res_offset": res_offset,
            "use_atomistic_reference": use_atomistic_reference,
            "rep_atoms": rep_atoms,
            "return_pairs": return_pairs,
        }

        quantity = Quantity(q, "dimensionless", metadata=metadata)
        ensemble.set_quantity("fraction_native_contacts", quantity)
        if return_pairs:
            return {
                "ref_atom_pairs": native_contacts,
                "model_atom_pairs": traj_native_pairs,
            }

    def add_end2end_distance(self, ensemble: Ensemble):
        """Get distance between CA atoms of the first and last residue in the protein
        for each structure in the ensemble
        """
        self.validate_c_alpha(ensemble)
        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        ca_atoms = trajectory.top.select("name CA")
        # Get the pair of the first and last CA atoms
        ca_pair = [[ca_atoms[0], ca_atoms[-1]]]
        distance = md.compute_distances(trajectory, ca_pair, periodic=False)
        # reshape to have the samme shape than frames
        distance = distance.reshape(-1)
        quantity = Quantity(
            distance,
            "nanometers",
            metadata={"feature": "end2end_distance"},
        )
        ensemble.set_quantity("end2end_distance", quantity)

    def add_tica(self, ensemble: Ensemble, transform: TICATransform):
        tica_coordinates = transform.transform(ensemble)
        quantity = Quantity(
            tica_coordinates,
            "mixed",
            metadata={"feature": "tica", "transform": transform},
        )
        ensemble.set_quantity("tica", quantity)

    def add_dssp(
        self,
        ensemble: Ensemble,
        simplified: bool = True,
        digitize: bool = False,
    ):
        """Adds DSSP secondary codes to each amino acid. Requires high backbone resolution
        (eg, N, C, O) in topology. DSSP codes are categorically digitized according to the
        following schemes if specified:

            Simplified:
                'NA' -> 0
                'H' -> 1
                'E' -> 2
                'C' -> 3

            Full:
                'NA' -> 0
                'H' -> 1
                'B' -> 2
                'E' -> 3
                'G' -> 4
                'I' -> 5
                'T' -> 6
                'S' -> 7
                ' ' -> 8

        Parameters
        ----------
        simplified:
            If True, only simplified DSSP codes are reported. See help(mdtraj.compute_dssp)
        digitize:
            If True, the DSSP codes with be digitized according to the mappings above
        """

        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        dssp_codes = md.compute_dssp(trajectory, simplified=simplified)

        if digitize:
            # use np.unique array reconstruction, with an intermediate lookup transform
            lookup = (
                Featurizer.simple_dssp_lookup
                if simplified
                else Featurizer.full_dssp_lookup
            )
            unique_codes, inverse_idx = np.unique(
                dssp_codes, return_inverse=True
            )
            found_digits = np.array([lookup[code] for code in unique_codes])
            dssp_codes = found_digits[inverse_idx].reshape(dssp_codes.shape)

        quantity = Quantity(
            dssp_codes,
            "dimensionless",
            metadata={
                "feature": "dssp",
                "simpflified": simplified,
                "digitize": digitize,
            },
        )
        ensemble.set_quantity("dssp", quantity)
        return

    def add_local_contact_number(
        self,
        ensemble: Ensemble,
        atom_type: str = "CA",
        min_res_dist: int = 3,
        cut: float = 1,
        beta: float = 50,
    ):
        """Adds PROTHON local contact number trajectory features for either CA
        or CB atoms. Based on the implementation in
        https://www.biorxiv.org/content/10.1101/2023.04.11.536474v1.full

        Parameter
        ---------
        Ensemble:
            Ensemble for which local contact numbers should be computed
        atom_type:
            Either "CA" or "CB". Determines the heavy atoms used to determine
            contacts. If "CB" is used, all contacts for involving GLY will be
            computed using GLY CA
        min_res_dist:
            Specifies the minumum residue separation to be considered as part
            set of non-bonded distances for contact calculations.
        cut:
            Contact distance cutoff
        beta:
            Smoothing parameter for contact discriminator with a default of
            50 nm^-1
        """

        if atom_type not in ["CA", "CB"]:
            raise ValueError(
                "`atom_type` must be 'CA' or 'CB', but '{}' was supplied".format(
                    atom_type
                )
            )

        # prepare atom/residue index arrays
        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        atoms = np.array(list(trajectory.topology.atoms))
        residues = np.array(list(trajectory.topology.residues))
        if atom_type == "CA":
            atom_inds = trajectory.topology.select("name {}".format(atom_type))
        if atom_type == "CB":
            atom_inds = trajectory.topology.select(
                "name {} or (name CA and resname GLY) or (name CA and resname IGL)".format(
                    atom_type
                )
            )
        assert all(np.diff(atom_inds) > 0)

        residue_inds = np.array([res.index for res in residues])

        assert len(residue_inds) == len(atom_inds)

        # grab fully connected pairs
        ind1, ind2 = np.triu_indices(len(atom_inds), 1)
        # apply residue neighbor restriction
        pairs = np.array([atom_inds[ind1], atom_inds[ind2]]).T
        res_pairs = np.array([residue_inds[ind1], residue_inds[ind2]]).T
        idx_to_del = []
        for i, pair in enumerate(pairs):
            a1, a2 = atoms[pair]
            if np.abs(a1.residue.index - a2.residue.index) < min_res_dist:
                idx_to_del.append(i)
        pairs = np.delete(pairs, idx_to_del, axis=0)
        res_pairs = np.delete(res_pairs, idx_to_del, axis=0)
        distances = md.compute_distances(trajectory, pairs, periodic=False)

        # compute local contacts
        contacts = clipped_sigmoid(beta * (cut - distances))
        contacts = md.geometry.squareform(contacts, res_pairs)

        contact_per_atom = np.sum(contacts, axis=-1)

        assert contact_per_atom.shape[-1] == len(atom_inds)

        quantity = Quantity(
            contact_per_atom,
            None,
            metadata={
                "feature": "local_contact_number",
                "atom_type": atom_type,
                "min_res_dist": min_res_dist,
                "cut": cut,
                "beta": beta,
            },
        )
        ensemble.set_quantity("local_contact_number", quantity)
        return

    def add_temporary_feature(
        self,
        ensemble: Ensemble,
        name: str,
        feat_func: Callable,
        *args,
        units: Optional[str] = None,
        warn_user=True,
        **kwargs,
    ):
        """Generates temporary feautures according to a user defined transform, `feat_func`, with args
        `feat_args`, and kwargs `feat_kwargs` saved as a Quantity with name `name`.

        WARNING: This method adds temporary, non-serializable features. They will NOT be included
        in serialized versions of the ensemble.

        Parameters
        ----------
        ensemble:
            Ensemble to which the feature should be computed for and added to
        name:
            String representing the name of the feature
        feat_func:
            Callable representing the function that generates the feature
        units:
            If supplied, specifies the units of the feature
        """

        if warn_user:
            warnings.warn(
                "This functionality does not support serialization. Please "
                "use for temporary analyses only."
            )

        feature = feat_func(*args, **kwargs)

        quantity = Quantity(
            feature,
            units,
            metadata={"feature": name},
        )
        ensemble.set_quantity(name, quantity)

    @staticmethod
    def _reference_structure_equality(
        input_structure: md.Trajectory,
        stored_coords: Union[List, np.ndarray],
        stored_top: str,
    ) -> bool:
        """Helper method for testing reference structure serialized equality for RMSD recomputation

        Parameters
        ----------
        input_structures:
            input MDTraj single frame Trajectory for proposed RMSD calculations
        stored_coords:
            Saved reference structure coordinates
        stored_top:
            Saved reference structure topology

        Returns
        -------
        bool:
            If the coordinates and topologies between the proposed and stored structures are
            the same, True is returned. Else, False is returned.
        """

        ref_coords = np.array(input_structure.xyz)
        stored_coords = np.array(stored_coords)
        ref_top = top2json(input_structure.topology)

        equals = []
        equals.append(stored_top == ref_top)
        equals.append(np.allclose(stored_coords, ref_coords))
        return all(equals)

    @staticmethod
    def compose_2d_feature(ens: Ensemble, compose_str: str) -> np.ndarray:
        """Composes two 1-D features into a 2-D features. Features must be
        already computed and stored in the supplied ensemble to preserve metadata.
        Since these features are composed of already stored quantities, the 2-D feature
        will not be registered as a quaitity.

        Parameters
        ----------
        ens:
            The ensemble for which the features should be composed
        compose_str:
            Composition string of two features with a single delimeter "_AND_".
            Eg, "rmsd_AND_fraction_native_contacts", "rg_AND_helicity", etc

        Returns
        -------
        composed_feature:
            numpy.ndarray of the composed 2-D feature, of shape (n_frames, 2), with
            the order of the features on the last axis the same as the order
            specified by `compose_str`
        """

        allowed_features = [
            attr[len("add_") :]
            for attr in dir(Featurizer)
            if attr.startswith("add_")
        ]
        # parse/check composition
        assert "_AND_" in compose_str
        features = compose_str.split("_AND_")
        assert len(features) == 2

        feature_data = []
        for feature in features:
            if not hasattr(ens, feature):
                raise RuntimeError(
                    f"Feature composition should only be performed with already computed features, and '{feature}' was not found in the supplied ensemble."
                )
            else:
                feature_data.append(ens.get_quantity(feature).raw_value)
        assert len(feature_data) == 2
        assert feature_data[0].shape == feature_data[1].shape
        n_frames = ens.n_frames
        composed_feature = np.hstack(
            [
                feature_data[0].reshape(n_frames, 1),
                feature_data[1].reshape(n_frames, 1),
            ]
        )
        return composed_feature

    @staticmethod
    def get_feature(
        ensemble: Ensemble, feature: str, *args, recompute=False, **kwargs
    ):
        """Get feature from an Ensemble object. If it is not there,
        compute it and store it in the Ensemble object

        Parameters
        ----------
        ensemble : Ensemble
            Target : str
            feature name
        """
        if not hasattr(ensemble, feature):
            recompute = True
        else:
            # Need to check, that current feature has the same
            # parameters as the requested one
            if (
                feature in ["rmsd", "fraction_native_contacts"]
                and len(args) == 1
            ):
                # Handle RMSD structure serialization (as arg)
                reference_structure = args[0]
                if not Featurizer._reference_structure_equality(
                    reference_structure,
                    ensemble[feature].metadata["reference_structure_coords"],
                    ensemble[feature].metadata["reference_structure_top"],
                ):
                    recompute = True

            for key, value in kwargs.items():
                if (
                    feature in ["rmsd", "fraction_native_contacts"]
                    and key == "reference_structure"
                ):
                    # Handle RMSD structure serialization (as kwarg)
                    reference_structure = kwargs[key]
                    if not Featurizer._reference_structure_equality(
                        reference_structure,
                        ensemble[feature].metadata[
                            "reference_structure_coords"
                        ],
                        ensemble[feature].metadata["reference_structure_top"],
                    ):
                        recompute = True
                        break
                else:
                    if not ensemble[feature].metadata.get(key) == (
                        value.tolist()
                        if isinstance(value, np.ndarray)
                        else value
                    ):
                        recompute = True
                        break
        if recompute:
            featurizer = Featurizer()
            featurizer.add(ensemble, feature, *args, **kwargs)
        return getattr(ensemble, feature)
