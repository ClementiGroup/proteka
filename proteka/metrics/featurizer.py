"""Featurizer takes a proteka.dataset.Ensemble and and extract features from it
    """
from collections.abc import Iterable
from itertools import combinations
import numpy as np
import mdtraj as md
from ..dataset import Ensemble
from ..quantity import Quantity
from typing import Callable, Dict, List, Optional, Tuple


__all__ = ["Featurizer"]


class Featurizer:
    """Extract features from an Ensemble entity and
    return them as Quantity objects"""

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

    def __init__(self, ensemble: Ensemble):
        self.ensemble = ensemble

    def validate_c_alpha(self) -> bool:
        """Check if C-alpha-based metrics can be computed"""
        ca_atoms = self.ensemble.top.select("name CA")
        # Should have at least 2 CA atoms in total to compute CA-based statistics
        assert len(ca_atoms) > 1, "Number of CA atoms is less than 2"

        # Should have one CA atom per each residue
        assert (
            len(ca_atoms) == self.ensemble.top.n_residues
        ), "Number of CA atoms does not match the number of residues"

        # Should have only one chain
        if self.ensemble.top.n_chains > 1:
            raise NotImplementedError(
                "More than one chain is not supported yet"
            )

        # Should have no breaks in chain (i.e. no missing residues):
        for chain in self.ensemble.top.chains:
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

    def add(self, feature: str, **kwargs):
        """Add a new feature to the Ensemble object"""
        if hasattr(self, "add_" + feature):
            getattr(self, "add_" + feature)(**kwargs)
        else:
            allowed_features = [
                attr for attr in dir(self) if attr.startswith("add_")
            ]
            raise ValueError(
                f"Feature {feature} is not supported. Supported features are: {allowed_features}"
            )
        return

    def add_ca_bonds(self):
        """
        Returns a Quantity object that contains length of pseudobonds
        between consecutive CA atoms.
        """
        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        self.validate_c_alpha()

        # Get the pairs of consecutive CA atoms
        ca_pairs = self._get_consecutive_ca(self.ensemble.top, order=2)
        ca_bonds = md.compute_distances(trajectory, ca_pairs, periodic=False)
        quantity = Quantity(
            ca_bonds, "nanometers", metadata={"feature": "ca_bonds"}
        )
        self.ensemble.set_quantity("ca_bonds", quantity)
        return

    def add_ca_distances(self, offset: int = 1):
        """Get distances between CA atoms.

        Parameters:
        offset: int, optional
            Distances between CA atoms in residues i and j are included if
            | i -j | >  offset. The default value is 1, which means that
            all nonbonded Calpha-Calpha distances are included. If offset is 0,
            then all the distances are included.

        """
        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        ca_atoms = trajectory.top.select("name CA")
        self.validate_c_alpha()

        # Get indices of pairs of atoms
        ind1, ind2 = np.triu_indices(len(ca_atoms), offset + 1)
        ca_pairs = np.array([ca_atoms[ind1], ca_atoms[ind2]]).T
        ca_distances = md.compute_distances(
            trajectory, ca_pairs, periodic=False
        )
        quantity = Quantity(
            ca_distances,
            "nanometers",
            metadata={"feature": "ca_distances", "offset": offset},
        )
        self.ensemble.set_quantity("ca_distances", quantity)

    @staticmethod
    def get_general_distances(
        ensemble: Ensemble,
        atom_types: Tuple[str],
        threshold: float,
        res_offset: int = 1,
        stride: Optional[int] = None,
    ) -> np.ndarray:
        """ "Compute clashes between atoms of types `atom_type_1/2` according
        to the supplied `distance` threshold.

        Parameters
        ----------
        ensemble:
            `Ensemble` over which clashes should be detected
        atom_types:
            Tuple of strings specifying for which two atom types distances should be
            computed. Uses MDTraj atom type names.
        threshold:
            Clash threshold for type pairs
        res_offset:
            `int` that determines the minimum residue separation for inclusion in distance
            calculations.
        stride:
            If specified, this stride is applied to the trajectory before the distance
            calculations

        Returns
        -------
        distances:
            `np.ndarray` of distances for the requested atom type pairs
        """

        if not isinstance(atom_types, tuple):
            raise ValueError(
                "atom_type_pairs must be a list of tuples of strings"
            )
        if len(atom_types) != 2:
            raise ValueError(
                f"Only 2 atom types may be specified but {atom_types} was supplied"
            )

        atom_type_1, atom_type_2 = atom_types[0], atom_types[1]
        if len(ensemble.top.select(f"name {atom_type_1}")) == 0:
            raise RuntimeError(
                f"atom type {atom_type_1} not found in ensemble topology"
            )
        if len(ensemble.top.select(f"name {atom_type_2}")) == 0:
            raise RuntimeError(
                f"atom type {atom_type_2} not found in ensemble topology"
            )

        all_atoms = list(ensemble.top.atoms)
        atom_indices = ensemble.top.select(
            f"name {atom_type_1} or name {atom_type_2}"
        )
        all_pairs = list(combinations(atom_indices, 2))

        pruned_pairs = []
        # res exclusion filtering
        for pair in all_pairs:
            a1, a2 = all_atoms[pair[0]], all_atoms[pair[1]]
            if np.abs(a1.residue.index - a2.residue.index) > res_offset:
                pruned_pairs.append((pair[0], pair[1]))

        traj = ensemble.get_all_in_one_mdtraj_trj()
        if stride != None:
            traj.xyz = traj.xyz[::stride]
        distances = md.compute_distances(traj, pruned_pairs)

        return distances

    def add_ca_angles(self):
        """Get angles between consecutive CA atoms"""
        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        self.validate_c_alpha()

        # Get the triplets of consecutive CA atoms
        ca_triplets = self._get_consecutive_ca(self.ensemble.top, order=3)
        ca_angles = md.compute_angles(trajectory, ca_triplets, periodic=False)
        quantity = Quantity(
            ca_angles,
            "radians",
            metadata={"feature": "ca_angles"},
        )
        self.ensemble.set_quantity("ca_angles", quantity)
        return

    def add_ca_dihedrals(self):
        """Get dihedral angles between consecutive CA atoms"""
        self.validate_c_alpha()
        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        # Get the quadruplets of consecutive CA atoms
        ca_quadruplets = self._get_consecutive_ca(self.ensemble.top, order=4)
        ca_dihedrals = md.compute_dihedrals(
            trajectory, ca_quadruplets, periodic=False
        )
        quantity = Quantity(
            ca_dihedrals, "radians", metadata={"feature": "ca_dihedrals"}
        )
        self.ensemble.set_quantity("ca_dihedrals", quantity)
        return

    def add_phi(self):
        """Get protein backbone phi torsions"""
        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        _, phi = md.compute_phi(trajectory)
        quantity = Quantity(phi, "radians", metadata={"feature": "phi"})
        self.ensemble.set_quantity("phi", quantity)
        return

    def add_psi(self):
        """Get protein backbone psi torsions"""
        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        _, psi = md.compute_psi(trajectory)
        quantity = Quantity(psi, "radians", metadata={"feature": "psi"})
        self.ensemble.set_quantity("psi", quantity)
        return

    def add_rmsd(self, reference_structure: md.Trajectory, **kwargs):
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

        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        rmsd = md.rmsd(trajectory, reference_structure, **kwargs)
        quantity = Quantity(rmsd, "nanometers", metadata={"feature": "rmsd"})
        self.ensemble.set_quantity("rmsd", quantity)
        return

    def add_rg(self):
        """Get radius of gyration for each structure in an ensemble"""
        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        rg = md.compute_rg(trajectory)
        quantity = Quantity(rg, "nanometers", metadata={"feature": "rg"})
        self.ensemble.set_quantity("rg", quantity)
        return

    def add_end2end_distance(self):
        """Get distance between CA atoms of the first and last residue in the protein
        for each structure in the ensemble
        """
        self.validate_c_alpha()
        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        ca_atoms = trajectory.top.select("name CA")

        # Get the pair of the first and last CA atoms
        ca_pair = [[ca_atoms[0], ca_atoms[-1]]]
        distance = md.compute_distances(trajectory, ca_pair, periodic=False)
        quantity = Quantity(
            distance,
            "nanometers",
            metadata={"feature": "end2end_distance"},
        )
        self.ensemble.set_quantity("end2end_distance", quantity)
        return

    def add_dssp(self, simplified: bool = True, digitize: bool = False):
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

        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
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
        self.ensemble.set_quantity("dssp", quantity)
        return

    def add_local_contact_number(
        self,
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
        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
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
        self.ensemble.set_quantity("local_contact_number", quantity)
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
        if hasattr(ensemble, feature) and (not recompute):
            return getattr(ensemble, feature)
        else:
            featurizer = Featurizer(ensemble)
            featurizer.add(feature, **kwargs)
            return getattr(ensemble, feature)
