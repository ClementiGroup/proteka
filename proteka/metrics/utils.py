import numpy as np
import mdtraj as md
from ..dataset import Ensemble


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


def histogram_features(
    target: Ensemble, reference: Ensemble, bins: int = 100
):
    """Take a two Ensemble objects, and compute histograms of target
    and reference. Histogram of the target is computed over the range,
    defined by reference. The function returns the histograms of the target and
    reference

    Parameters
    ----------
    target, reference : Ensemble
        Target and reference Ensemble objects
    n_bins : int, optional
        Number of histograms to use, by default 100
    """

    try:
        weights = reference.weights
    except AttributeError:
        weights = None
    hist_reference, bin_edges = np.histogram(
        reference, bins=bins, weights=weights
    )
    try:
        weights = target.weights
    except AttributeError:
        weights = None
    hist_target, _ = np.histogram(target, bins=bin_edges, weights=weights)

    return hist_reference, hist_target
