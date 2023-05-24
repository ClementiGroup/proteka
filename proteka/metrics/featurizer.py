"""Featurizer takes a proteka.dataset.Ensemble and and extract features from it
    """
from abc import ABC, abstractmethod
from collections.abc import Iterable
import json
import warnings
import numpy as np
import mdtraj as md
from typing import Dict, Optional, List, Tuple
from ..dataset import Ensemble
from ..quantity import Quantity
from typing import Callable, Dict, List, Optional, Tuple
from itertools import combinations

__all__ = ["Featurizer", "Transform", "TICATransform"]


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
        for feature_tuple in self.features:
            feature, params = feature_tuple[0], feature_tuple[1]
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
        return cls.from_dict(json.loads(string))

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

    def add(self, ensemble: Ensemble, feature: str, **kwargs):
        """Add a new feature to the Ensemble object"""
        if hasattr(self, "add_" + feature):
            getattr(self, "add_" + feature)(ensemble, **kwargs)
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
        print(ca_pairs_all)
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
        print(ca_pairs)
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
        self, ensemble: Ensemble, reference_structure: md.Trajectory, **kwargs
    ):
        """Get RMSD of a subset of atoms
        reference: Reference mdtraj.Trajectory object
        Wrapper of mdtraj.rmsd

        Parameters
        ----------
        reference_structure:
            Reference structure from which RMSD calculations are made
        kwargs:
            kwarg options for `mdtraj.rmsd()`,
            for example `{"frame": 0, "atom_indices": np.arange(10), "parallel": True,
            "precentered": False}`. See
            help(mdtraj.rmsd) for more information.
        """

        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        rmsd = md.rmsd(trajectory, reference_structure, **kwargs)
        quantity = Quantity(rmsd, "nanometers", metadata={"feature": "rmsd"})
        ensemble.set_quantity("rmsd", quantity)

    def add_rg(self, ensemble: Ensemble):
        """Get radius of gyration for each structure in an ensemble"""
        trajectory = ensemble.get_all_in_one_mdtraj_trj()
        rg = md.compute_rg(trajectory)
        quantity = Quantity(rg, "nanometers", metadata={"feature": "rg"})
        ensemble.set_quantity("rg", quantity)

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
            None,
            metadata={"feature": "dssp"},
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
                "name {} or (name CA and resname GLY)".format(atom_type)
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
        contacts = 1.0 / (1.0 + np.exp(beta * (distances - cut)))
        contacts = md.geometry.squareform(contacts, res_pairs)
        contact_per_atom = np.sum(contacts, axis=-1)
        assert contact_per_atom.shape[-1] == len(atom_inds)

        quantity = Quantity(
            contact_per_atom,
            None,
            metadata={"feature": "local_contact_number"},
        )
        ensemble.set_quantity("local_contact_number", quantity)
        return

    @staticmethod
    def get_feature(
        ensemble: Ensemble, feature: str, recompute=False, **kwargs
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
            for key, value in kwargs.items():
                if not ensemble[feature].metadata.get(key) == value:
                    recompute = True
        if recompute:
            featurizer = Featurizer()
            featurizer.add(ensemble, feature, **kwargs)
        return getattr(ensemble, feature)
