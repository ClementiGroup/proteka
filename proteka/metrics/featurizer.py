"""Featurizer takes a proteka.dataset.Ensemble and and extract features from it
    """
import numpy as np
import mdtraj as md
from ..dataset import Ensemble, Quantity


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

    def get_ca_bonds(self) -> Quantity:
        """
        Returns a Quantity object that contains length of pseudobonds
        between consecutive CA atoms.
        """
        trajectory = self.ensemble.get_all_in_one_mdtraj()
        ca_atoms = trajectory.top.select("name CA")
        self.validate_c_alpha()

        # Get the pairs of consecutive CA atoms
        ca_pairs = [
            [ca_atoms[i], ca_atoms[i + 1]] for i in range(len(ca_atoms) - 1)
        ]
        ca_bonds = md.compute_distances(trajectory, ca_pairs, periodic=False)
        return Quantity(
            ca_bonds, "nanometers", metadata={"feature": "CA-CA pseudobonds"}
        )

    def get_ca_distances(self, offset=1) -> Quantity:
        """Get distances between CA atoms.

        Parameters:
        offset: int, optional
            Distances between CA atoms in residues i and j are included if
            | i -j | >=  offset. The default value is 1, which means that
            all Calpha-Calpha distances are included.

        """
        trajectory = self.ensemble.get_all_in_one_mdtraj()
        ca_atoms = trajectory.top.select("name CA")
        self.validate_c_alpha()

        # Get indices of pairs of atoms
        ind1, ind2 = np.triu_indices(len(ca_atoms), offset)
        ca_pairs = np.array([ca_atoms[ind1], ca_atoms[ind2]]).T
        ca_distances = md.compute_distances(
            trajectory, ca_pairs, periodic=False
        )
        return Quantity(
            ca_distances,
            "nanometers",
            metadata={"feature": "CA-CA distances", "offset": offset},
        )

    def get_ca_angles(self) -> Quantity:
        """Get angles between consecutive CA atoms"""
        trajectory = self.ensemble.get_all_in_one_mdtraj()
        ca_atoms = trajectory.top.select("name CA")
        self.validate_c_alpha()

        # Get the triplets of consecutive CA atoms
        ca_triplets = [
            [ca_atoms[i], ca_atoms[i + 1], ca_atoms[i + 2]]
            for i in range(len(ca_atoms) - 2)
        ]
        ca_angles = md.compute_distances(
            trajectory, ca_triplets, periodic=False
        )
        return Quantity(
            ca_angles,
            "radians",
            metadata={"feature": "consecutive CA-CA-CA angles"},
        )

    def get_ca_bond_vectors(self) -> Quantity:
        """Get vectors between consecutive CA atoms"""
        raise NotImplementedError

    def get_ca_dihedrals(self) -> Quantity:
        """Get dihedral angles between consecutive CA atoms"""
        trajectory = self.ensemble.get_all_in_one_mdtraj()
        ca_atoms = trajectory.top.select("name CA")
        print(ca_atoms)
        self.validate_c_alpha()

        # Get the quadruplets of consecutive CA atoms
        ca_quadruplets = [
            [ca_atoms[i], ca_atoms[i + 1], ca_atoms[i + 2], ca_atoms[i + 2]]
            for i in range(len(ca_atoms) - 2)
        ]
        ca_dihedrals = md.compute_dihedrals(
            trajectory, ca_quadruplets, periodic=False
        )
        return Quantity(
            ca_dihedrals,
            "radians",
            metadata={"feature": "consecutive CA-CA-CA_CA dihedrals"},
        )

    def get_backbone_torsions(self) -> Quantity:
        """Get protein backbone torsions"""
        raise NotImplementedError

    def get_rmsd(self, reference=None, frame=0, atom_indices=None) -> Quantity:
        """Get RMSD of a subset of atoms.
        Wrapper of mdtraj.rmsd
        """
        raise NotImplementedError

    def get_rg(self) -> Quantity:
        """Get radius of gyration for each structure in an ensemble"""
        raise NotImplementedError

    def get_end2end_distance(self) -> Quantity:
        """Get distance between CA atoms of the first and last residue in the protein
        for each structure in the ensemble
        """
        trajectory = self.ensemble.get_all_in_one_mdtraj()
        ca_atoms = trajectory.top.select("name CA")
        self.validate_c_alpha()

        # Get the pair of the first and last CA atoms
        ca_pair = [[ca_atoms[0], ca_atoms[-1]]]
        distance = md.compute_distances(trajectory, ca_pair, periodic=False)
        return Quantity(
            distance,
            "nanometers",
            metadata={"feature": "CA-CA end-to-end distance"},
        )
