
import numpy as np
import mdtraj as md

from proteka.dataset import Emsemble
from proteka.metrics import Featurizer


def _get_grid_configuration(n_atoms, grid_size=0.4, ndim=3):
    """
    Position n atoms on a 3D grid with a specified grid size.
    Self-crossings and overlaps are allowed.
    The first atom is always placed at the origin.
    """
    xyz = np.zeros((n_atoms, ndim))
    for i in range(1, n_atoms):
        # Randomly select  one coordinate to change
        dim = np.random.choice(range(ndim), size=1) 
        new_coordinate = xyz[i-1,:].copy()
        new_coordinate[dim] += np.random.choice([-1, 1]) * grid_size
        xyz[i,:] = new_coordinate
    return xyz


def _generate_grid_polymer(n_frames, n_atoms, grid_size=0.4):
    """Generate a set of n_frames configuration of a CA polymer on a grid"""
    xyz = np.zeros((n_frames, n_atoms, 3))
    for i in range(n_frames):
        xyz[i] = _get_grid_configuration(n_atoms, grid_size=grid_size)
    
    top = md.Topology()
    chain = top.add_chain()
    for i in range(n_atoms):
        res = top.add_residue('ALA', chain)
        top.add_atom("CA", md.element.carbon, res)
    return md.Trajectory(xyz, top)


def test_get_ca_bonds():
    grid_size = 0.4
    traj = _generate_grid_polymer(n_frames=10, n_atoms=5, grid_size=grid_size)
    ens = Ensemble("CAgrid", traj.top, Quantity(traj.xyz, "nm"))
    featurizer = Featurizer(ens)
    ca_bonds = featurizer.get_ca_bonds()
    assert np.all(ca_bonds.in_unit_of('nm')) == grid_size
    
if __name__ == "__main__":
    test_get_ca_bonds()