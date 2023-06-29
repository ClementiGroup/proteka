import numpy as np
import mdtraj as md
from ..dataset import Ensemble
from typing import Union, Tuple, Optional, Dict, List
from itertools import combinations
from collections.abc import Iterable

__all__ = [
    "generate_grid_polymer",
    "get_6_bead_frame",
    "histogram_features",
    "histogram_features2d",
    "histogram_vector_features",
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


def get_general_distances(
    ensemble: Ensemble,
    atom_names: Tuple[str],
    res_offset: int = 1,
    stride: Optional[int] = None,
    periodic: bool = False,
) -> np.ndarray:
    """Compute all distances between two all atom of two specified names. If atom names are different,
    (e.g. ("N", "O")), only name1-name2 distances will be computed, and not name1-name1 nor name2-name2.
    If atom names are the same (e.g. ("CB", "CB")), name1-name2 distances will be computed.

    Parameters
    ----------
    ensemble:
        `Ensemble` from which distances should be computed
    atom_names:
        Tuple of strings specifying for which two atom names distances should be
        computed. Uses MDTraj atom type names (e.g., "CA" or "CB").
    res_offset:
                `int` that determines the minimum residue separation for inclusion in distance
                calculations; two atoms that belong to residues i and j are included in the
                calculations if |i-j| > res_offset
    stride:
        If specified, this stride is applied to the trajectory before the distance
        calculations
    periodic:
        If true, minimum-image conventions are used when calculating distances using
        MDTraj.

    Returns
    -------
    distances:
        `np.ndarray` of distances for the requested atom type pairs
    """

    if not isinstance(atom_names, tuple):
        raise ValueError("atom_name_pairs must be a list of tuples of strings")
    if len(atom_names) != 2:
        raise ValueError(
            f"Only 2 atom types may be specified but {atom_names} was supplied"
        )

    atom_name_1, atom_name_2 = atom_names[0], atom_names[1]
    if len(ensemble.top.select(f"name {atom_name_1}")) == 0:
        raise RuntimeError(
            f"atom type {atom_name_1} not found in ensemble topology"
        )
    if len(ensemble.top.select(f"name {atom_name_2}")) == 0:
        raise RuntimeError(
            f"atom type {atom_name_2} not found in ensemble topology"
        )

    all_atoms = list(ensemble.top.atoms)
    atom_indices = ensemble.top.select(
        f"name {atom_name_1} or name {atom_name_2}"
    )
    all_pairs = list(combinations(atom_indices, 2))

    pruned_pairs = []
    # res exclusion filtering
    for pair in all_pairs:
        a1, a2 = all_atoms[pair[0]], all_atoms[pair[1]]
        if np.abs(a1.residue.index - a2.residue.index) > res_offset:
            # Eliminate extra same name pairs for case of two different atom types
            if atom_name_1 != atom_name_2:
                if a1.name != a2.name:
                    pruned_pairs.append((pair[0], pair[1]))
            else:
                pruned_pairs.append((pair[0], pair[1]))

    traj = ensemble.get_all_in_one_mdtraj_trj()
    if stride != None:
        traj.xyz = traj.xyz[::stride]
    distances = md.compute_distances(traj, pruned_pairs, periodic=periodic)

    return distances


def get_ALA_10_helix() -> md.Trajectory:
    """Get a perfect helical structure of ALA-10
    at the carbon alpha resolution in nm
    """

    coords = np.array(
        [
            [
                [-0.3274, -0.6615, -0.468],
                [-0.29, -0.2958, -0.5651],
                [-0.4475, -0.1869, -0.2367],
                [-0.2079, -0.4088, -0.042],
                [0.089, -0.2591, -0.2264],
                [-0.0311, 0.0924, -0.1457],
                [-0.0639, 0.0025, 0.2223],
                [0.29, -0.1362, 0.2254],
                [0.4225, 0.1834, 0.0679],
                [0.245, 0.3929, 0.3308],
            ]
        ]
    )

    topology = md.Topology()
    chain = topology.add_chain()
    resnames = ["ALA" for _ in range(10)]
    for r in resnames:
        residue = topology.add_residue(r, chain)
        topology.add_atom("CA", md.element.carbon, residue)
    return md.Trajectory(coords, topology)


def get_CLN_trajectory(
    single_frame=False, seed=1678543, unfolded=False, pro_ca_cb_swap=False
) -> md.Trajectory:
    """Get a random 49 atom CG backbone + CB model of CLN025 (nanometers),
    with 100 noise-perturbed frames.
    """
    nframes = 100
    if unfolded:
        coords = np.array(
            [
                [-1.296, 5.911, 12.133],
                [-1.276, 6.055, 12.113],
                [-1.133, 6.094, 12.077],
                [-1.377, 6.125, 12.024],
                [-1.365, 6.145, 11.902],
                [-1.493, 6.166, 12.08],
                [-1.606, 6.227, 12.009],
                [-1.725, 6.128, 12.024],
                [-1.634, 6.372, 12.059],
                [-1.671, 6.39, 12.175],
                [-1.611, 6.472, 11.967],
                [-1.635, 6.611, 12.0],
                [-1.562, 6.71, 11.898],
                [-1.774, 6.65, 12.023],
                [-1.866, 6.59, 11.964],
                [-1.803, 6.751, 12.104],
                [-1.942, 6.799, 12.122],
                [-1.954, 6.797, 12.279],
                [-1.96, 6.939, 12.072],
                [-2.023, 7.019, 12.14],
                [-1.919, 6.958, 11.939],
                [-1.899, 7.086, 11.889],
                [-1.758, 7.088, 11.824],
                [-1.994, 7.124, 11.779],
                [-1.989, 7.24, 11.726],
                [-2.091, 7.036, 11.737],
                [-2.179, 7.044, 11.621],
                [-2.234, 6.918, 11.557],
                [-2.305, 7.119, 11.657],
                [-2.346, 7.134, 11.772],
                [-2.377, 7.175, 11.562],
                [-2.503, 7.257, 11.561],
                [-2.617, 7.171, 11.611],
                [-2.696, 7.21, 11.701],
                [-2.616, 7.05, 11.57],
                [-2.684, 6.938, 11.626],
                [-2.657, 6.801, 11.559],
                [-2.661, 6.911, 11.778],
                [-2.75, 6.858, 11.847],
                [-2.547, 6.943, 11.841],
                [-2.522, 6.893, 11.979],
                [-2.374, 6.897, 12.017],
                [-2.595, 6.956, 12.092],
                [-2.599, 7.076, 12.107],
                [-2.671, 6.875, 12.173],
                [-2.742, 6.916, 12.296],
                [-2.9, 6.918, 12.268],
                [-2.707, 6.838, 12.421],
                [-2.729, 6.713, 12.422],
            ],
            dtype="float64",
        )
    else:
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
            ],
            dtype="float64",
        )
    if single_frame == False:
        np.random.seed(seed)
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
        topology.add_atom("N", md.element.nitrogen, residue)
        if r != "GLY":
            if pro_ca_cb_swap and r == "PRO":
                topology.add_atom("CB", md.element.carbon, residue)
                topology.add_atom("CA", md.element.carbon, residue)
            else:
                topology.add_atom("CA", md.element.carbon, residue)
                topology.add_atom("CB", md.element.carbon, residue)
        else:
            topology.add_atom("CA", md.element.carbon, residue)
        topology.add_atom("C", md.element.carbon, residue)
        topology.add_atom("O", md.element.oxygen, residue)
    return md.Trajectory(coords, topology)


def histogram_features(
    target: np.ndarray,
    reference: np.ndarray,
    target_weights: np.ndarray = None,
    reference_weights: np.ndarray = None,
    bins: Union[int, np.ndarray] = 100,
    open_edges: bool = False,
):
    """Take a two arrays, and compute vector histograms of target
    and reference. Histogram of the target is computed over the range,
    defined by reference. The function returns the histograms of the target and
    reference. Marginal histograms
    will be returned by accumulating independently over the last array axis.

    Parameters
    ----------
    target, reference : np.ndarray
        Target and reference np.ndarrays
    target_weights, reference_weights: np.ndarray
        Frame weights for the target and reference probabilities
    bins : int or np.ndarray, optional
        Number of bins to use, by default 100 over the support specified
        by the reference. If np.ndarray, those bins will be used instead
    open_edges : bool, optional
        If True, the leftmost edge of the first bin for the target array
        is assigned to -inf, and the rightmost edge of the last bin for the
        target array is assigned to +inf. If False, the first bin includes
        the left edge and the last bin includes the right edge. Default is False.
    """

    hist_reference, bin_edges = np.histogram(
        reference, bins=bins, weights=reference_weights
    )
    if open_edges:
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
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
    open_edges: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Take a two multi-feature arrays, and compute vector histograms of target
    and reference. Histogram of the target is computed over the range,
    defined by reference. The function returns the histograms of the target and
    reference. Marginal histograms will be returned by accumulating independently
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
    open_edges: bool = False,
        If True, the leftmost edge of the first bin for the target array
        is assigned to -inf, and the rightmost edge of the last bin for the
        target array is assigned to +inf. If False, the first bin includes
        the left edge and the last bin includes the right edge. Default is False.
    """

    assert target.shape[-1] == reference.shape[-1]

    # slow implementation, I know.
    num_feats = target.shape[-1]
    num_bins = (len(bins) - 1) if isinstance(bins, np.ndarray) else bins
    hist_reference = np.zeros((num_bins, num_feats))
    hist_target = np.zeros((num_bins, num_feats))

    for i in range(num_feats):
        hist_target[:, i], hist_reference[:, i] = histogram_features(
            target[:, i],
            reference[:, i],
            target_weights=target_weights,
            reference_weights=reference_weights,
            bins=bins,
            open_edges=open_edges,
        )

    return hist_target, hist_reference


def histogram_features2d(
    target: np.ndarray,
    reference: np.ndarray,
    target_weights: np.ndarray = None,
    reference_weights: np.ndarray = None,
    bins: int = 100,
    open_edges: bool = False,
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
    open_edges: bool = False,
        If True, the leftmost edge of the first bin for the target array
        is assigned to -inf, and the rightmost edge of the last bin for the
        target array is assigned to +inf. If False, the first bin includes
        the left edge and the last bin includes the right edge. Default is False.
    """
    assert target.shape[1] == 2, "Target should be 2d with shape (n, 2)"
    assert reference.shape[1] == 2, "Reference should be 2d with shape (n, 2)"

    hist_reference, xedges, yedges = np.histogram2d(
        reference[:, 0], reference[:, 1], bins=bins, weights=reference_weights
    )
    if open_edges:
        xedges[0] = -np.inf
        xedges[-1] = np.inf
        yedges[0] = -np.inf
        yedges[-1] = np.inf

    hist_target, _, _ = np.histogram2d(
        target[:, 0],
        target[:, 1],
        bins=[xedges, yedges],
        weights=target_weights,
    )
    return hist_target, hist_reference


def reduce_atom_pairs_by_residue_offset(
    atom_list: List[md.core.topology.Atom],
    atom_idx_pairs,
    res_offset: int = 3,
) -> Iterable:
    """Calculates all contacts under certain cutoff given a list of atom indices

    Parameters
    ----------
    atom_list:
        List of MDTraj `Atom` instances
    atom_idx:
        list of atom index pairs for computing distances
    res_offset:
        minimum distance between residues to be considered pairs

    Returns
    -------
    filtered_pairs:
       List of atom index pairs satisyfing the minimum residue offset specified.
    """

    filtered_pairs = []
    for p in atom_idx_pairs:
        p1_a, p1_b = p[0], p[1]
        if (
            np.abs(
                atom_list[p1_a].residue.index - atom_list[p1_b].residue.index
            )
            > res_offset
        ):
            filtered_pairs.append([p1_a, p1_b])

    return filtered_pairs
