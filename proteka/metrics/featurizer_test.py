import numpy as np
import mdtraj as md
import pytest
from itertools import combinations
import deeptime as dt

from proteka.dataset import Ensemble
from proteka.quantity import Quantity
from proteka.metrics import Featurizer, TICATransform
from proteka.metrics.utils import (
    generate_grid_polymer,
    get_6_bead_frame,
    get_CLN_trajectory,
)


@pytest.fixture
def single_frame():
    traj = get_6_bead_frame()
    ensemble = Ensemble("6bead", traj.top, Quantity(traj.xyz, "nm"))
    return ensemble


@pytest.fixture
def get_CLN_frame():
    traj = get_CLN_trajectory(single_frame=True)
    ensemble = Ensemble.from_mdtraj_trj("ref", traj)
    return ensemble


@pytest.fixture
def grid_polymer():
    traj = generate_grid_polymer(n_frames=10, n_atoms=5)
    ensemble = Ensemble("CAgrid", traj.top, Quantity(traj.xyz, "nm"))
    return ensemble


ref_dssp_simple = np.array(["C", "E", "C", "C", "C", "C", "C", "C", "E", "C"])
ref_dssp_simple_digit = np.array([3, 2, 3, 3, 3, 3, 3, 3, 2, 3])
ref_dssp_full = np.array([" ", "B", " ", "T", "T", "T", " ", " ", "B", " "])
ref_dssp_full_digit = np.array([8, 2, 8, 6, 6, 6, 8, 8, 2, 8])

# manually calculated LCN for single CLN frame
ref_local_contact_number = np.array(
    [
        [
            4.5867076,
            5.9684186,
            4.9999995,
            3.3204648,
            1.9700541,
            3.1611652,
            4.733629,
            4.9989476,
            4.4786115,
            4.151842,
        ]
    ]
)


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
    assert ca == result


def test_get_ca_bonds(grid_polymer):
    grid_size = 0.4
    traj = generate_grid_polymer(n_frames=10, n_atoms=5, grid_size=grid_size)
    ens = Ensemble("CAgrid", traj.top, Quantity(traj.xyz, "nm"))
    featurizer = Featurizer()
    featurizer.add_ca_bonds(ens)
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
    reference_dihedrals = np.array([-np.pi / 2, 0, -np.pi / 2])
    assert np.all(np.isclose(dihedrals, reference_dihedrals))


def test_ca_distances(single_frame):
    distances = Featurizer.get_feature(single_frame, "ca_distances", offset=0)
    reference_distances = md.compute_distances(
        get_6_bead_frame(), combinations(range(6), 2)
    )
    assert np.all(np.isclose(distances, reference_distances))


def test_tica(grid_polymer):
    """Test that tics are correctly calculated"""
    features = {'ca_distances': {'offset':1},
                'ca_angles': {},
                'ca_dihedrals': {} ,
                }
    dim=3
    lagtime=1
    # Get TICA from the proteka featurizer
    transform = TICATransform(features, estimation_params={'lagtime':lagtime, 'dim':dim})
    tica_proteka = Featurizer.get_feature(grid_polymer, 'tica', transform=transform)
    # Get TICA from deeptime
    input_features = []
    for feature, params in features.items():
        input_features.append(Featurizer.get_feature(grid_polymer, feature, **params))
    input_features = np.concatenate(input_features, axis=1)
    print(input_features.shape)
    tica_deeptime = dt.decomposition.TICA(lagtime=lagtime, dim=dim).fit_transform(input_features)
    for i in range(dim):
        assert np.all(np.isclose(tica_proteka[:,i], tica_deeptime[:,i]))
        
def test_feature_rewriting(grid_polymer):
    """Check that if a feature is already in the ensemble, but has different parameters, it is 
    rewritten"""
    
    distances = Featurizer.get_feature(grid_polymer, 'ca_distances', offset=1)
    new_distances = Featurizer.get_feature(grid_polymer, 'ca_distances', offset=2)
    assert distances.shape != new_distances.shape
    

def test_local_contact_number(get_CLN_frame):
    """Tests local contact number calculation"""
    ens = get_CLN_frame
    feat = Featurizer(ens)
    feat.add_local_contact_number()
    local_contact_number = ens.get_quantity("local_contact_number")
    np.allclose(ref_local_contact_number, local_contact_number)


def test_dssp(get_CLN_frame):
    """Tests DSSP recording"""
    ens = get_CLN_frame
    feat = Featurizer(ens)
    feat.add_dssp(digitize=False)
    simple_dssp = ens.get_quantity("dssp").raw_value

    feat = Featurizer(ens)
    feat.add_dssp(digitize=True)
    simple_dssp_digit = ens.get_quantity("dssp").raw_value

    feat = Featurizer(ens)
    feat.add_dssp(digitize=False, simplified=False)
    full_dssp = ens.get_quantity("dssp").raw_value

    feat = Featurizer(ens)
    feat.add_dssp(digitize=True, simplified=False)
    full_dssp_digit = ens.get_quantity("dssp").raw_value

    np.testing.assert_array_equal(simple_dssp.flatten(), ref_dssp_simple)
    np.testing.assert_array_equal(
        simple_dssp_digit.flatten(), ref_dssp_simple_digit
    )
    np.testing.assert_array_equal(full_dssp.flatten(), ref_dssp_full)
    np.testing.assert_array_equal(
        full_dssp_digit.flatten(), ref_dssp_full_digit
    )
