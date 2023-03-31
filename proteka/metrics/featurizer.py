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
        """Get pairs of consecutive CA atoms such
        as atoms in a pair come from the same chain and consecutive residues

        Parameters
        ----------
        topology : _type_
            Topology object
        order : int, optional
            Number of consecutive atoms, by default 2
        """
        for chain in topology.chains:
            for i in range(chain.n_residues - order + 1):
                atom_list = []
                start_group = True
                for j in range(order):
                    res = chain.residue(i + j)
                    if not start_group and (
                        res.resSeq - chain.residue(i).resSeq != 1
                    ):
                        break
                    start_group = False
                    atom = res.atom("CA")
                    atom_list.append(atom.index)
                yield atom_list

    def add(self, feature: str, **kwargs):
        """Add a new feature to the ensemble"""
        feature_dict = {
            "end2end_distance": self.add_end2end_distance,
            "ca_bonds": self.add_ca_bonds,
            "ca_distances": self.add_ca_distances,
            "rmsd": self.add_rmsd,
            "rg": self.add_rg,
            "ca_angles": self.add_ca_angles,
            "ca_dihedrals": self.add_ca_dihedrals,
            "phi": self.add_phi,
            "psi": self.add_psi,
        }

        if feature in feature_dict.keys():
            feature_dict[feature](**kwargs)
        else:
            raise ValueError(f"Feature {feature} is not supported. Supported features are: {feature_dict.keys()}")
        return

    def add_ca_bonds(self) -> Quantity:
        """
        Returns a Quantity object that contains length of pseudobonds
        between consecutive CA atoms.
        """
        trajectory = self.ensemble.get_all_in_one_mdtraj()
        ca_atoms = trajectory.top.select("name CA")
        self.validate_c_alpha()

        # Get the pairs of consecutive CA atoms
        # TODO support for wrong order/multiple chains(NetworkX)
        # TODO register Quantity with an ensemble instead of returning
        # it 
        ca_pairs = [
            [ca_atoms[i], ca_atoms[i + 1]] for i in range(len(ca_atoms) - 1)
        ]
        ca_bonds = md.compute_distances(trajectory, ca_pairs, periodic=False)
        quantity = Quantity(ca_bonds, "nanometers", metadata={"feature": "ca_ca_pseudobonds"})
        self.ensemble.set_quantity("ca_ca_pseudobonds", quantity)
        return 
        

    def add_ca_distances(self, offset=1) -> Quantity:
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
        quantity = Quantity(
            ca_distances,
            "nanometers",
            metadata={"feature": "ca_ca_distances", "offset": offset},
        )
        self.ensemble.set_quantity("ca_ca_distances", quantity)

    def add_ca_angles(self) -> Quantity:
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
        quantity = Quantity(
            ca_angles,
            "radians",
            metadata={"feature": "consecutive_ca_ca_ca_angles"},
        )
        self.ensemble.set_quantity("consecutive_ca_ca_ca_angles", quantity)
        return

    def add_ca_bond_vectors(self) -> Quantity:
        """Get vectors between consecutive CA atoms"""
        raise NotImplementedError

    def add_ca_dihedrals(self) -> Quantity:
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
        quantity = Quantity(
            ca_dihedrals,
            "radians",
            metadata={"feature": "consecutive_ca_ca_ca_ca_dihedrals"},
        )
        self.ensemble.set_quantity("consecutive_ca_ca_ca_ca_dihedrals", quantity)
        return

    def add_phi(self) -> Quantity:
        """Get protein backbone phi torsions"""
        trajectory = self.ensemble.get_all_in_one_mdtraj()
        _, phi = md.compute_phi(trajectory)
        quantity = Quantity(phi, "radians", metadata={"feature": "phi"})
        self.ensemble.set_quantity("phi", quantity)
        return

    def add_psi(self) -> Quantity:
        """Get protein backbone psi torsions"""
        trajectory = self.ensemble.get_all_in_one_mdtraj()
        _, psi = md.compute_psi(trajectory)
        quantity = Quantity(psi, "radians", metadata={"feature": "psi"})
        self.ensemble.set_quantity("psi", quantity)
        return

    def add_rmsd(
        self,
        reference: md.Trajectory = None,
        frame: int = 0,
        atom_indices: Iterable[int] = None,
    ) -> Quantity:
        """Get RMSD of a subset of atoms
        reference: Reference mdtraj.Trajectory object
        Wrapper of mdtraj.rmsd
        """
        trajectory = self.ensemble.get_all_in_one_mdtraj()
        rmsd = md.rmsd(trajectory, reference, frame, atom_indices=atom_indices)
        quantity = Quantity(rmsd, "nanometers", metadata={"feature": "rmsd"})
        self.ensemble.set_quantity("rmsd", quantity)
        return

    def add_rg(self) -> Quantity:
        """Get radius of gyration for each structure in an ensemble"""
        raise NotImplementedError

    def add_end2end_distance(self) -> Quantity:
        """Get distance between CA atoms of the first and last residue in the protein
        for each structure in the ensemble
        """
        trajectory = self.ensemble.get_all_in_one_mdtraj()
        ca_atoms = trajectory.top.select("name CA")
        self.validate_c_alpha()

        # Get the pair of the first and last CA atoms
        ca_pair = [[ca_atoms[0], ca_atoms[-1]]]
        distance = md.compute_distances(trajectory, ca_pair, periodic=False)
        quantity = Quantity(
            distance,
            "nanometers",
            metadata={"feature": "ca_ca_end2end_distance"},
        )
        self.ensemble.set_quantity("ca_ca_end2end_distance", quantity)
        return
    
    
