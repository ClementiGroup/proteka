import numpy as np
import mdtraj as md
import pytest

from proteka.dataset import Ensemble
from proteka.quantity import Quantity
from proteka.metrics import Featurizer
from proteka.metrics.utils import generate_grid_polymer, get_6_bead_frame


@pytest.fixture
def single_frame():
    traj = get_6_bead_frame()
    ensemble = Ensemble("6bead", traj.top, Quantity(traj.xyz, "nm"))
    return ensemble


@pytest.fixture
def grid_polymer():
    traj = generate_grid_polymer(n_frames=10, n_atoms=5)
    ensemble = Ensemble("CAgrid", traj.top, Quantity(traj.xyz, "nm"))
    return ensemble


@pytest.mark.parametrize(
    "order,result",
    [
        (2, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
        (3, [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]),
        (4, [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]),
    ],
)
def test_get_consecutive_ca(single_frame, order, result):
    ca = Featurizer._get_consecutive_ca(single_frame.top, order=order)
    print(ca)
    assert ca == result


def test_get_ca_bonds(grid_polymer):
    grid_size = 0.4
    traj = generate_grid_polymer(n_frames=10, n_atoms=5, grid_size=grid_size)
    ens = Ensemble("CAgrid", traj.top, Quantity(traj.xyz, "nm"))
    featurizer = Featurizer(ens)
    featurizer.add_ca_bonds()
    print(ens.ca_bonds)
    assert np.all(np.isclose(ens.ca_bonds, grid_size))


def test_ca_bonds(single_frame):
    bonds = Featurizer.get_feature(single_frame, "ca_bonds")
    assert np.all(np.isclose(bonds, 0.38e0))


def test_ca_angles(single_frame):
    angles = Featurizer.get_feature(single_frame, "ca_angles")
    reference_angles = np.array(
        [np.pi / 2, np.arccos(9 / 38), np.arccos(9 / 38), np.pi / 2]
    )
    assert np.all(np.isclose(angles, reference_angles))


def test_ca_dihedrals(single_frame):
    dihedrals = Featurizer.get_feature(single_frame, "ca_dihedrals")
    print(dihedrals)
    reference_dihedrals = np.array([-np.pi / 2, 0, -np.pi / 2])
    print(reference_dihedrals)
    assert np.all(np.isclose(dihedrals, reference_dihedrals))
