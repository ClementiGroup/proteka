import numpy as np
import mdtraj as md
import pytest
from itertools import combinations
import deeptime as dt
from proteka.dataset import Ensemble
from proteka.quantity import Quantity
from proteka.metrics import Featurizer, TICATransform
from proteka.metrics.utils import (
    get_general_distances,
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
def get_CLN_traj():
    class ens_factory:
        def make_ens(self, ca_only=False):
            traj = get_CLN_trajectory()
            if ca_only:
                traj = traj.atom_slice(traj.topology.select("name CA"))
            ensemble = Ensemble.from_mdtraj_trj("ref", traj)
            return ensemble

    return ens_factory()


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
    featurizer = Featurizer()
    featurizer.add_ca_bonds(grid_polymer)
    assert np.all(np.isclose(grid_polymer.ca_bonds, grid_size))


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


def test_ca_distances_offset(single_frame):
    distances = Featurizer.get_feature(single_frame, "ca_distances", offset=2)
    reference_distances = md.compute_distances(
        get_6_bead_frame(), [[0, 3], [0, 4], [0, 5], [1, 4], [1, 5], [2, 5]]
    )
    assert np.all(np.isclose(distances, reference_distances))


def test_ca_distances_custom_selection(single_frame):
    distances = Featurizer.get_feature(
        single_frame, "ca_distances", subset_selection="resid 0 to 2", offset=1
    )
    reference_distances = md.compute_distances(get_6_bead_frame(), [[0, 2]])
    assert np.all(np.isclose(distances, reference_distances))


def test_general_distances(get_CLN_frame):
    """Runs general distance test on CB-CB pairs more than 2 res apart"""
    ens = get_CLN_frame
    cb_idx = ens.top.select("name CB")
    atoms = list(ens.top.atoms)
    cb_pairs = list(combinations(cb_idx, 2))
    pruned_pairs = []
    for pair in cb_pairs:
        a1, a2 = atoms[pair[0]], atoms[pair[1]]
        if np.abs(a1.residue.index - a2.residue.index) > 2:
            pruned_pairs.append((pair[0], pair[1]))
    traj = ens.get_all_in_one_mdtraj_trj()
    manual_distances = md.compute_distances(traj, pruned_pairs)

    distances = get_general_distances(ens, ("CB", "CB"), 2)
    np.testing.assert_array_equal(manual_distances, distances)


def test_general_clashes_atom_input_raises(get_CLN_frame):
    """Tests raises for improper atom_name inputs for get_general_distances"""
    with pytest.raises(ValueError):
        distances = get_general_distances(
            get_CLN_frame,
            [("CA", "O", "C")],
            res_offset=2,
        )

    with pytest.raises(ValueError):
        distances = get_general_distances(
            get_CLN_frame,
            ["CA", "CB"],
            res_offset=2,
        )

    with pytest.raises(RuntimeError):
        distances = get_general_distances(
            get_CLN_frame,
            ("silly atom", "O"),
            res_offset=2,
        )


def test_tica_order(grid_polymer):
    """Tests raises for input features"""
    features = {
        "ca_distances": {"offset": 1},
        "ca_angles": {},
        "ca_dihedrals": {},
    }
    dim = 3
    lagtime = 1
    with pytest.raises(ValueError):
        transform = TICATransform(
            features, estimation_params={"lagtime": lagtime, "dim": dim}
        )
    with pytest.raises(ValueError):
        transform = TICATransform(
            [("A", {}), ("Silly", {}), ("Mistake", "Oops", {})],
            estimation_params={"lagtime": lagtime, "dim": dim},
        )


def test_tica(grid_polymer):
    """Test that tics are correctly calculated"""
    features = [
        ("ca_distances", {"offset": 1}),
        ("ca_angles", {}),
        ("ca_dihedrals", {}),
    ]

    dim = 3
    lagtime = 1
    # Get TICA from the proteka featurizer
    transform = TICATransform(
        features, estimation_params={"lagtime": lagtime, "dim": dim}
    )
    tica_proteka = Featurizer.get_feature(
        grid_polymer, "tica", transform=transform
    )
    # Get TICA from deeptime
    input_features = []
    for feature, params in features:
        input_features.append(
            Featurizer.get_feature(grid_polymer, feature, **params)
        )
    input_features = np.concatenate(input_features, axis=1)
    print(input_features.shape)
    tica_deeptime = dt.decomposition.TICA(
        lagtime=lagtime, dim=dim
    ).fit_transform(input_features)
    for i in range(dim):
        assert np.all(np.isclose(tica_proteka[:, i], tica_deeptime[:, i]))


def test_feature_rewriting(grid_polymer):
    """Check that if a feature is already in the ensemble, but has different parameters, it is
    rewritten"""

    distances = Featurizer.get_feature(grid_polymer, "ca_distances", offset=1)
    new_distances = Featurizer.get_feature(
        grid_polymer, "ca_distances", offset=2
    )
    assert distances.shape != new_distances.shape


def test_local_contact_number(get_CLN_frame):
    """Tests local contact number calculation"""
    ens = get_CLN_frame
    feat = Featurizer()
    feat.add_local_contact_number(ens)
    local_contact_number = ens.get_quantity("local_contact_number")
    np.allclose(ref_local_contact_number, local_contact_number)


def test_dssp(get_CLN_frame):
    """Tests DSSP recording"""
    ens = get_CLN_frame
    feat = Featurizer()
    feat.add_dssp(ens, digitize=False)
    simple_dssp = ens.get_quantity("dssp").raw_value

    feat = Featurizer()
    feat.add_dssp(ens, digitize=True)
    simple_dssp_digit = ens.get_quantity("dssp").raw_value

    feat = Featurizer()
    feat.add_dssp(ens, digitize=False, simplified=False)
    full_dssp = ens.get_quantity("dssp").raw_value

    feat = Featurizer()
    feat.add_dssp(ens, digitize=True, simplified=False)
    full_dssp_digit = ens.get_quantity("dssp").raw_value

    np.testing.assert_array_equal(simple_dssp.flatten(), ref_dssp_simple)
    np.testing.assert_array_equal(
        simple_dssp_digit.flatten(), ref_dssp_simple_digit
    )
    np.testing.assert_array_equal(full_dssp.flatten(), ref_dssp_full)
    np.testing.assert_array_equal(
        full_dssp_digit.flatten(), ref_dssp_full_digit
    )


def test_rmsd_syntax_raise(get_CLN_traj):
    # Check to see if only one indexing choice can be used at a time
    ens = get_CLN_traj.make_ens()
    ref_structure = ens.get_all_in_one_mdtraj_trj()[0]
    feat = Featurizer()
    with pytest.raises(TypeError):
        feat.add_rmsd(
            ens,
            ref_structure,
            atom_selection="name CA",
            atom_indices=np.arange(10),
            ref_atom_indices=np.arange(10),
        )


def test_rmsd(get_CLN_traj):
    # Check to see if only one indexing choice can be used at a time
    ens = get_CLN_traj.make_ens()
    traj = ens.get_all_in_one_mdtraj_trj()
    ref_structure = traj[0]

    manual_rmsd = md.rmsd(traj, ref_structure)

    feat = Featurizer()
    feat.add_rmsd(ens, ref_structure)
    rmsd = ens.get_quantity("rmsd").raw_value
    # we need to add some tolerance to the test or it breaks
    np.testing.assert_array_almost_equal_nulp(rmsd, manual_rmsd, nulp=1e3)


def test_rmsd_atom_selection(get_CLN_traj):
    # Check to see if only one indexing choice can be used at a time
    ens = get_CLN_traj.make_ens()
    ca_ens = get_CLN_traj.make_ens(ca_only=True)
    ref_structure = ens.get_all_in_one_mdtraj_trj()[0]
    ca_traj = ca_ens.get_all_in_one_mdtraj_trj()

    manual_rmsd = md.rmsd(
        ca_traj,
        ref_structure,
        atom_indices=ca_traj.topology.select("name CA"),
        ref_atom_indices=ref_structure.topology.select("name CA"),
    )

    feat = Featurizer()
    feat.add_rmsd(
        ca_ens,
        ref_structure,
        atom_selection="name CA",
    )
    rmsd = ca_ens.get_quantity("rmsd").raw_value
    np.testing.assert_array_equal(rmsd, manual_rmsd)
