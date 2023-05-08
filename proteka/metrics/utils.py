import numpy as np
import mdtraj as md
from ..dataset import Ensemble
from typing import Union, Tuple

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


def get_CLN_trajectory(single_frame=False) -> md.Trajectory:
    """Get a random 49 atom CG backbonde + CB model of CLN025 (nanometers),
    with 100 noise-perturbed frames.
    """
    nframes = 100
    coords = np.array(
        [
            [-15.65, 3.208, 5.655],
            [-15.765, 3.16, 5.722],
            [-15.894, 3.197, 5.652],
            [-15.749, 3.013, 5.703],
            [-15.664, 2.956, 5.645],
            [-15.839, 2.945, 5.765],
            [-15.849, 2.797, 5.773],
            [-15.81, 2.751, 5.909],
            [-15.988, 2.741, 5.736],
            [-16.093, 2.762, 5.796],
            [-15.998, 2.663, 5.625],
            [-16.127, 2.634, 5.57],
            [-16.111, 2.605, 5.42],
            [-16.18, 2.509, 5.646],
            [-16.116, 2.404, 5.655],
            [-16.304, 2.514, 5.71],
            [-16.362, 2.401, 5.791],
            [-16.473, 2.473, 5.875],
            [-16.409, 2.282, 5.701],
            [-16.42, 2.177, 5.763],
            [-16.443, 2.294, 5.567],
            [-16.49, 2.186, 5.487],
            [-16.6, 2.245, 5.386],
            [-16.37, 2.126, 5.419],
            [-16.352, 2.004, 5.424],
            [-16.269, 2.202, 5.366],
            [-16.15, 2.139, 5.292],
            [-16.1, 2.227, 5.174],
            [-16.034, 2.12, 5.382],
            [-15.956, 2.028, 5.357],
            [-16.015, 2.201, 5.492],
            [-15.893, 2.203, 5.57],
            [-15.77, 2.267, 5.51],
            [-15.659, 2.238, 5.549],
            [-15.795, 2.353, 5.408],
            [-15.691, 2.422, 5.321],
            [-15.738, 2.434, 5.182],
            [-15.654, 2.553, 5.38],
            [-15.74, 2.632, 5.426],
            [-15.52, 2.596, 5.394],
            [-15.484, 2.729, 5.439],
            [-15.33, 2.729, 5.491],
            [-15.511, 2.835, 5.332],
            [-15.456, 2.822, 5.223],
            [-15.597, 2.932, 5.351],
            [-15.625, 3.026, 5.246],
            [-15.763, 3.007, 5.175],
            [-15.601, 3.176, 5.289],
            [-15.675, 3.226, 5.364],
        ]
    )
    if single_frame == False:
        coords = coords + 0.01 * np.random.randn(nframes, 49, 3)
    topology = md.Topology()
    chain = topology.add_chain()
    resnames = [
        "TYR",
        "TYR",
        "ASP",
        "PRO",
        "GLU",
        "THR",
        "GLY",
        "THR",
        "TRP",
        "TYR",
    ]
    for r in resnames:
        residue = topology.add_residue(r, chain)
        topology.add_atom("N", md.element.carbon, residue)
        topology.add_atom("CA", md.element.carbon, residue)
        if r != "GLY":
            topology.add_atom("CB", md.element.carbon, residue)
        topology.add_atom("C", md.element.carbon, residue)
        topology.add_atom("O", md.element.carbon, residue)
    return md.Trajectory(coords, topology)


def histogram_features(
    target: np.ndarray,
    reference: np.ndarray,
    target_weights: np.ndarray = None,
    reference_weights: np.ndarray = None,
    bins: Union[int, np.ndarray] = 100,
):
    """Take a two arrays, and compute vector histograms of target
    and reference. Histogram of the target is computed over the range,
    defined by reference. The function returns the histograms of the target and
    reference. Marginal histograms
    will be returned by accumulating indepentdly over the last array axis.

    Parameters
    ----------
    target, reference : np.ndarray
        Target and reference np.ndarrays
    target_weights, reference_weights: np.ndarray
        Frame weights for the target and reference probabilities
    bins : int or np.ndarray, optional
        Number of bins to use, by default 100 over the support specified
        by the reference. If np.ndarray, those bins will be used instead
    """

    hist_reference, bin_edges = np.histogram(
        reference, bins=bins, weights=reference_weights
    )
    hist_target, _ = np.histogram(
        target, bins=bin_edges, weights=target_weights
    )
    return hist_target, hist_reference


def histogram_vector_features(
    target: np.ndarray,
    reference: np.ndarray,
    target_weights: np.ndarray = None,
    reference_weights: np.ndarray = None,
    bins: Union[int, np.ndarray] = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Take a two multi-feature arrays, and compute vector histograms of target
    and reference. Histogram of the target is computed over the range,
    defined by reference. The function returns the histograms of the target and
    reference. Marginal histograms will be returned by accumulating indepentdly
    over the last array axis.

    Parameters
    ----------
    target, reference : np.ndarray
        Target and reference np.ndarrays
    target_weights, reference_weights: np.ndarray
        Frame weights for the target and reference probabilities
    bins : int or np.ndarray, optional
        Number of bins to use, by default 100 over the support specified
        by the reference. If np.ndarray, those bins will be used instead
    """

    assert target.shape[-1] == reference.shape[-1]

    # slow implementation, I know.
    num_feats = target.shape[-1]
    num_bins = len(bins) if isinstance(bins, np.ndarray) else bins
    hist_reference = np.zeros((num_bins, num_feats))
    hist_target = np.zeros((num_bins, num_feats))

    for i in range(num_feats):
        hist_reference[:, i], hist_target[:, i] = histogram_features(
            target[:, i],
            reference[:, i],
            target_weights=target_weights,
            reference_weights=reference_weights,
            bins=bins,
        )

    return hist_target, hist_reference


def histogram_features2d(
    target: np.ndarray,
    reference: np.ndarray,
    target_weights: np.ndarray = None,
    reference_weights: np.ndarray = None,
    bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Take a two 2 feature arrays, and compute 2D histograms of target
    and reference. Histogram of the target is computed over the range,
    defined by reference. The function returns the histograms of the target and
    reference. Marginal histograms will be returned by accumulating indepentdly
    over the last array axis.

    Parameters
    ----------
    target, reference : np.ndarray
        Target and reference np.ndarrays
    target_weights, reference_weights: np.ndarray
        Frame weights for the target and reference probabilities
    bins : int or np.ndarray, optional
        Number of bins to use, by default 100 over the support specified
        by the reference. If np.ndarray, those bins will be used instead
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
    return hist_reference, hist_target
