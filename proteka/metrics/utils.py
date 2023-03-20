import numpy as np
import mdtraj as md

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


def generate_grid_polymer(n_frames, n_atoms, grid_size=0.4):
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