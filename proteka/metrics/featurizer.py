"""Featurizer takes a proteka.dataset.Ensemble and and extract features from it
    """
from collections.abc import Iterable
import numpy as np
import mdtraj as md
from ..dataset import Ensemble
from ..quantity import Quantity
from typing import Callable, Dict, List, Optional

__all__ = ["Featurizer"]


class Featurizer:
    """Extract features from an Ensemble entity and
    return them as Quantity objects"""

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

    def add_rmsd(
        self,
        reference_structure: md.Trajectory,
        rmsd_kwargs: Optional[Dict] = None,
    ):
        """Get RMSD of a subset of atoms
        reference: Reference mdtraj.Trajectory object
        Wrapper of mdtraj.rmsd

        Parameters
        ----------
        reference_structure:
            MDTraj single-frame Trajectory which serves as the
            reference structure for RMSD calculations.
        rmsd_kwargs:
            Dictionary of kwarg options for `mdtraj.rmsd()`. See
            help(mdtraj.rmsd) for more information.
        """

        # RMSD kwarg sanity checks
        valid_rmsd_opts = set(
            "frame", "atom_indices", "parallel", "precentered"
        )
        assert all([opt in valid_rmsd_opts for opt in rmsd_kwargs.keys()])

        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        rmsd = md.rmsd(trajectory, reference, **rmsd_kwargs)
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

    def add_local_contact_number(
        self,
        atom_type: str = "CA",
        min_res_dist: int = 3,
        cut: float = 1,
        beta: float = 0.02,
    ):
        """Adds PROTHON local contact number trajectory features for either CA
        or CB atoms. Based on the implementation in
        https://www.biorxiv.org/content/10.1101/2023.04.11.536474v1.full

        Parameter
        ---------
        atom_type:
            Either "CA" or "CB". Determines the heavy atoms used to determine
            contacts
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

        # compute distances
        trajectory = self.ensemble.get_all_in_one_mdtraj_trj()
        atom_inds = trajectory.topology.select("name {}".format(atom_type))
        ind1, ind2 = np.triu_indices(len(atom_inds), min_res_dist + 1)
        pairs = np.array([atom_inds[ind1], atom_inds[ind2]]).T
        distances = md.compute_distances(trajectory, pairs, periodic=False)

        # compute local contacts
        contacts = 1.0 / (1.0 + np.exp(beta * (distances - cut)))
        contacts = md.geometry.squareform(contacts, pairs)
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
