import numpy as np
import mdtraj as md
from .featurizer import Featurizer
from ..dataset import Ensemble
from typing import Tuple
from deeptime.decomposition import TICA

__all__ = [
    "generate_grid_polymer",
    "get_6_bead_frame",
    "histogram_features",
    "histogram_features2d",
]


def _get_grid_configuration(n_atoms, grid_size=0.4, ndim=3):
    """distributions
    Position n atoms on a 3D grid with a specified grid size.
    Self-crossings and overlaps are allowed.
    The first atom is always placed at the origin.
    """
    xyz = np.zeros((n_atoms, ndim))
    for i in range(1, n_atoms):
        # Randomly select  one coordinate to change
        dim = np.random.choice(range(ndim), size=1)
        new_coordinate = xyz[i - 1, :].copy()
        new_coordinate[dim] += np.random.choice([-1, 1]) * grid_size
        xyz[i, :] = new_coordinate
    return xyz


def generate_grid_polymer(n_frames, n_atoms, grid_size=0.4):
    """Generate a set of n_frames configuration of a CA polymer on a grid"""
    xyz = np.zeros((n_frames, n_atoms, 3))
    for i in range(n_frames):
        xyz[i] = _get_grid_configuration(n_atoms, grid_size=grid_size)

    top = md.Topology()
    chain = top.add_chain()
    for i in range(n_atoms):
        res = top.add_residue("ALA", chain)
        top.add_atom("CA", md.element.carbon, res)
    return md.Trajectory(xyz, top)


def get_6_bead_frame():
    """Generate a frame that contains
      6 beads with a predefined geometry.
      Distance between consecutive beads is 0.38 nm
      
      0            5   
       \\          /   
        1-0.2nm- 4   
       /          \\
      2___0.38 nm__3
      
      Atoms  1, 2, 3 and 4 are in plane, atoms 0 and 5 are out of plane, 90 degrees 
      
      
      
    """
    n_atoms = 6
    d = 0.3800e0
    d_clash = 0.2000e0
    xyz = np.zeros((n_atoms, 3))

    # position atoms 2 and 3
    # x axis is defined by 2-3 vector
    # y axis crosses 2-3 vector in the middle
    xyz[2, :] = [-d / 2, 0, 0]
    xyz[3, :] = [d / 2, 0, 0]

    # Find positions of the clashing atoms
    y_position = np.sqrt(d**2 - ((d - d_clash) / 2) ** 2)
    xyz[1, :] = [-d_clash / 2, y_position, 0]
    xyz[4, :] = [d_clash / 2, y_position, 0]

    # Add atoms 0 and 5: same x,y position as atoms 1 and 4,
    # but are above or below the plane by d
    xyz[0, :] = [-d_clash / 2, y_position, d]
    xyz[5, :] = [d_clash / 2, y_position, -d]

    topology = md.Topology()
    chain = topology.add_chain()
    for i in range(n_atoms):
        residue = topology.add_residue("ALA", chain)
        topology.add_atom("CA", md.element.carbon, residue)
    return md.Trajectory(xyz, topology)


def histogram_features(
    target: np.array,
    reference: np.array,
    target_weights: np.array = None,
    reference_weights: np.array = None,
    bins: int = 100,
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
    hist_reference, bin_edges = np.histogram(
        reference, bins=bins, weights=reference_weights
    )
    hist_target, _ = np.histogram(
        target, bins=bin_edges, weights=target_weights
    )

    return hist_target, hist_reference


def histogram_features2d(
    target: np.array,
    reference: np.array,
    target_weights: np.array = None,
    reference_weights: np.array = None,
    bins: int = 100,
):
    """Take a two Ensemble objects, and compute 2d histograms of target
    and reference. Histogram of the target is computed over the range,
    defined by reference. The function returns the 2d histograms of the target and
    reference

    Parameters
    ----------
    target, reference : Ensemble
        Target and reference Ensemble objects
    n_bins : int, optional
        Number of histograms to use, by default 100
    """

    hist_reference, xedges, yedges = np.histogram2d(
        reference[:, 0], reference[:, 1], bins=bins, weights=reference_weights
    )
    hist_target, _, _ = np.histogram2d(
        target[:, 0],
        target[:, 1],
        bins=[xedges, yedges],
        weights=target_weights,
    )
    return hist_target, hist_reference

def get_tica_features(target: Ensemble, reference: Ensemble, **kwargs) -> Tuple[np.array, np.array]:
    """Perform TICA on the reference ensemble and use it to transform target ensemble.
    returns the first 2 TICA components both for the target and the reference ensemble
    
    Parameters
    ----------
    target : Ensemble
        target ensemble
    reference : Ensemble
        reference ensemble, will be used for TICA model fitting

    Returns
    -------
    tica_target: np.array
        2-dimensional array containing the first 2 tica features for the target ensemble
    tica_reference: np.array
        2-dimensional array containing the first 2 tica features for the reference ensemble
    """
    
    # Fit TICA model on the reference ensemble
    estimator = TICA(dim=2, **kwargs)
    # will fit on the CA distances of the reference ensemble.
    ca_reference = Featurizer.get_feature(reference, "ca_distances")
    estimator.fit(ca_reference)
    model = estimator.fetch_model()
    # Transform the reference ensemble
    tica_reference = model.transform(ca_reference)
    # Transform the target ensemble
    ca_target = Featurizer.get_feature(target, "ca_distances")
    tica_target = model.transform(ca_target)

    return tica_target, tica_reference