"""Test calculator
"""

import numpy as np
import mdtraj as md
import pytest
import tempfile
from pathlib import Path
import os.path as osp
from ruamel.yaml import YAML
from collections.abc import Iterable, Mapping
from proteka.dataset import Ensemble
from proteka.quantity import Quantity
from proteka.metrics import Featurizer
from proteka.metrics import (
    StructuralIntegrityMetrics,
    StructuralQualityMetrics,
    EnsembleQualityMetrics,
)
from proteka.metrics.utils import get_6_bead_frame, get_CLN_trajectory


@pytest.fixture
def single_frame():
    traj = get_6_bead_frame()
    ensemble = Ensemble("6bead", traj.top, Quantity(traj.xyz, "nm"))
    return ensemble


@pytest.fixture
def get_two_ensembles():
    ref_traj = get_CLN_trajectory()
    target_traj = get_CLN_trajectory()
    ref_ensemble = Ensemble.from_mdtraj_trj("ref", ref_traj)
    target_ensemble = Ensemble.from_mdtraj_trj("target", target_traj)
    return target_ensemble, ref_ensemble


@pytest.fixture
def cln_single_frame():
    traj = get_CLN_trajectory(single_frame=True)
    ensemble = Ensemble("6bead", traj.top, Quantity(traj.xyz, "nm"))
    return ensemble


def test_ca_clashes(single_frame):
    clashes = StructuralIntegrityMetrics.ca_clashes(single_frame)
    assert clashes["CA-CA clashes"] == 1


def test_structural_metric_run(get_two_ensembles):
    target_ensemble, _ = get_two_ensembles
    reference_structure = target_ensemble.get_all_in_one_mdtraj_trj()[0]
    metrics = {
        "reference_structure": reference_structure,
        "features": {
            "rmsd": {
                "feature_params": {"atom_selection": "name CA"},
                "metric_params": {"fraction_smaller": {"threshold": 0.5}},
            }
        },
    }

    sqm = StructuralQualityMetrics(metrics)
    results = sqm(target_ensemble)
    assert len(results) == 1


def test_ensemble_metric_run(get_two_ensembles):
    target_ensemble, ref_ensemble = get_two_ensembles
    reference_structure = target_ensemble.get_all_in_one_mdtraj_trj()[0]

    # precompute features needed for compound features
    feat = Featurizer()
    feat.add_rg(target_ensemble)
    feat.add_rg(ref_ensemble)
    feat.add_end2end_distance(target_ensemble)
    feat.add_end2end_distance(ref_ensemble)

    metrics = {
        "features": {
            "ca_distances": {
                "feature_params": {},
                "metric_params": {"js_div": {"bins": np.linspace(0, 1.6, 100)}},
            },
            "rg": {
                "metric_params": {"js_div": {"bins": np.linspace(0, 1.6, 100)}},
            },
            "helicity": {
                "metric_params": {"js_div": {"bins": np.linspace(0, 1.0, 100)}},
            },
            "end2end_distance": {
                "metric_params": {"js_div": {"bins": np.linspace(0, 1.6, 100)}},
            },
            "rmsd": {
                "feature_params": {
                    "reference_structure": reference_structure,
                    "atom_selection": "name CA",
                },
                "metric_params": {"js_div": {"bins": np.linspace(0, 1.6, 100)}},
            },
            "fraction_native_contacts": {
                "feature_params": {
                    "reference_structure": reference_structure,
                    "atom_selection": "name CA",
                    "use_atomistic_reference": False,
                },
                "metric_params": {"js_div": {"bins": np.linspace(0, 1.0, 100)}},
            },
            "rg_AND_helicity": {
                "metric_params": {"js_div": {"bins": 100}},
            },
            "dssp": {
                "feature_params": {"digitize": True},
                "metric_params": {
                    "mse_ldist": {"bins": np.array([0, 1, 2, 3, 4])},
                    "js_div": {"bins": np.array([0, 1, 2, 3, 4])},
                },
            },
            "local_contact_number": {
                "feature_params": {"atom_type": "CB"},
                "metric_params": {
                    "mse_ldist": {
                        "bins": np.linspace(
                            0, reference_structure.topology.n_residues, 100
                        )
                    },
                    "js_div": {
                        "bins": np.linspace(
                            0, reference_structure.topology.n_residues, 100
                        )
                    },
                },
            },
        },
    }
    eqm = EnsembleQualityMetrics(metrics)
    results = eqm(target_ensemble, ref_ensemble)
    assert len(results) == 11


def test_calculator_config_bin_conversion():
    # Tests to make sure non-int binopts are converted correctly
    # for EnsembleQualityMetrics instanced from configs
    metrics = {
        "EnsembleQualityMetrics": {
            "features": {
                "rg": {
                    "feature_params": {"atom_selection": "name CA"},
                    "metric_params": {
                        "js_div": {
                            "bins": {"start": 0, "stop": 100, "num": 101}
                        },
                    },
                },
                "ca_distances": {
                    "feature_params": {"offset": 1},
                    "metric_params": {
                        "mse_ldist": {"bins": 101},
                    },
                },
                "feature3_AND_feature4": {
                    "feature_params": {"offset": 1},
                    "metric_params": {
                        "kl_div": {"bins": [101, 101]},
                    },
                },
                "feature1_AND_feature2": {
                    "feature_params": {"offset": 1},
                    "metric_params": {
                        "mse_dist": {
                            "bins": [
                                {"start": 0, "stop": 1, "num": 101},
                                {"start": 0, "stop": 2, "num": 101},
                            ]
                        },
                    },
                },
            },
        },
    }
    expected_bins1 = np.linspace(0, 100, 101)
    expected_bins2 = 101
    expected_bins3 = [101, 101]
    expected_bins4 = [np.linspace(0, 1, 101), np.linspace(0, 2, 101)]

    with tempfile.TemporaryDirectory() as tmp:
        yaml = YAML()
        yaml.dump(metrics, open(osp.join(tmp, "test.yaml"), "w"))
        eqm1 = EnsembleQualityMetrics.from_config(
            osp.join(tmp, "test.yaml")
        )  # from YAML
        eqm2 = EnsembleQualityMetrics.from_config(metrics)  # from dictionary
        for eqm in [eqm1, eqm2]:
            np.testing.assert_array_equal(
                eqm.metrics["features"]["rg"]["metric_params"]["js_div"][
                    "bins"
                ],
                expected_bins1,
            )
            assert (
                eqm.metrics["features"]["ca_distances"]["metric_params"][
                    "mse_ldist"
                ]["bins"]
                == expected_bins2
            )
            assert (
                eqm.metrics["features"]["feature3_AND_feature4"][
                    "metric_params"
                ]["kl_div"]["bins"]
                == expected_bins3
            )
            assert isinstance(
                eqm.metrics["features"]["feature1_AND_feature2"][
                    "metric_params"
                ]["mse_dist"]["bins"],
                list,
            )
            for bins, ebins in zip(
                eqm.metrics["features"]["feature1_AND_feature2"][
                    "metric_params"
                ]["mse_dist"]["bins"],
                expected_bins4,
            ):
                np.testing.assert_array_equal(bins, ebins)


def test_eqm_input_dict_copy():
    # Tests whether the dictionnary used to intialize EnsembleQualityMetrics is modified when calling from_config()
    metrics = {
        "EnsembleQualityMetrics": {
            "features": {
                "feature1_AND_feature2": {
                    "feature_params": {"offset": 1},
                    "metric_params": {
                        "mse_dist": {
                            "bins": [
                                {"start": 0, "stop": 1, "num": 101},
                                {"start": 0, "stop": 2, "num": 101},
                            ]
                        },
                    },
                },
            },
        },
    }

    eqm = EnsembleQualityMetrics.from_config(metrics)
    for feature in metrics["EnsembleQualityMetrics"]["features"].keys():
        feature_dict = metrics["EnsembleQualityMetrics"]["features"][feature]
        for metric in feature_dict["metric_params"].keys():
            if "bins" in list(feature_dict["metric_params"][metric].keys()):
                binopts = feature_dict["metric_params"][metric]["bins"]
                assert isinstance(binopts, list)
                assert all([isinstance(opt, Mapping) for opt in binopts])

    parsed_metrics = EnsembleQualityMetrics.parse_config(
        metrics["EnsembleQualityMetrics"]
    )
    for feature in metrics["EnsembleQualityMetrics"]["features"].keys():
        feature_dict = metrics["EnsembleQualityMetrics"]["features"][feature]
        for metric in feature_dict["metric_params"].keys():
            if "bins" in list(feature_dict["metric_params"][metric].keys()):
                binopts = feature_dict["metric_params"][metric]["bins"]
                assert isinstance(binopts, list)
                assert all([isinstance(opt, Mapping) for opt in binopts])


def test_calculator_config_mdtraj_conversion():
    # Tests to make sure paths to pdb files are converted correctly
    # for EnsembleQualityMetrics instanced from configs
    root_dir = Path(__file__).parent.parent.parent
    cln_path = osp.join(
        root_dir, "examples", "example_dataset_files", "cln_folded.pdb"
    )
    metrics = {
        "EnsembleQualityMetrics": {
            "features": {
                "rmsd": {
                    "feature_params": {
                        "atom_selection": "name CA",
                        "reference_structure": cln_path,
                    },
                    "metric_params": {
                        "js_div": {
                            "bins": {"start": 0, "stop": 100, "num": 101}
                        },
                    },
                },
            },
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        yaml = YAML()
        yaml.dump(metrics, open(osp.join(tmp, "test.yaml"), "w"))
        eqm1 = EnsembleQualityMetrics.from_config(
            osp.join(tmp, "test.yaml")
        )  # from YAML
        eqm2 = EnsembleQualityMetrics.from_config(metrics)  # from dictionary
        for eqm in [eqm1, eqm2]:
            assert isinstance(
                eqm.metrics["features"]["rmsd"]["feature_params"][
                    "reference_structure"
                ],
                md.Trajectory,
            )


def test_structural_calculator_config_bin_conversion():
    # Tests to make sure metric params are stored properly for StructuralQualityMetrics
    root_dir = Path(__file__).parent.parent.parent
    cln_path = osp.join(
        root_dir, "examples", "example_dataset_files", "cln_folded.pdb"
    )
    metrics = {
        "StructuralQualityMetrics": {
            "reference_structure": cln_path,
            "features": {
                "rmsd": {
                    "feature_params": {"atom_selection": "name CA"},
                    "metric_params": {"fraction_smaller": {"threshold": 0.25}},
                }
            },
        }
    }

    expected_thres = 0.25

    with tempfile.TemporaryDirectory() as tmp:
        yaml = YAML()
        yaml.dump(metrics, open(osp.join(tmp, "test.yaml"), "w"))
        eqm = StructuralQualityMetrics.from_config(osp.join(tmp, "test.yaml"))
        np.testing.assert_array_equal(
            eqm.metrics["features"]["rmsd"]["metric_params"][
                "fraction_smaller"
            ]["threshold"],
            expected_thres,
        )


def test_general_default_clashes(cln_single_frame):
    # Checks to makes sure a "shrunken" CLN trajecory violates certain clashes
    # between all atoms of the two terminal residues when using the default
    # VDW/allowance based clashes

    ens = cln_single_frame
    original_coords = ens.get_quantity("coords").raw_value
    with pytest.warns(UserWarning):
        ens.set_quantity("coords", 2.0 * original_coords / 3.0)
    clashes = StructuralIntegrityMetrics.general_clashes(
        ens,
        [
            ("N", "N"),
            ("N", "CA"),
            ("N", "CB"),
            ("N", "C"),
            ("N", "O"),
            ("CB", "CB"),
            ("CB", "CA"),
            ("CB", "C"),
            ("CB", "O"),
            ("CA", "CA"),
            ("CA", "C"),
            ("CA", "O"),
            ("C", "C"),
            ("C", "O"),
            ("O", "O"),
        ],
        res_offset=8,
    )
    assert clashes["N-N clashes"] == 0
    assert clashes["N-CA clashes"] == 2
    assert clashes["N-CB clashes"] == 0
    assert clashes["N-C clashes"] == 2
    assert clashes["N-O clashes"] == 2

    assert clashes["CB-CB clashes"] == 0
    assert clashes["CB-CA clashes"] == 0
    assert clashes["CB-C clashes"] == 1
    assert clashes["CB-O clashes"] == 1

    assert clashes["CA-CA clashes"] == 0
    assert clashes["CA-C clashes"] == 2
    assert clashes["CA-O clashes"] == 2

    assert clashes["C-C clashes"] == 1
    assert clashes["C-O clashes"] == 2

    assert clashes["O-O clashes"] == 0


def test_general_clashes(cln_single_frame):
    # Test to make sure that conservative clash threshold (0.35 nm) for N-O
    # detects clashes properly for residues at least 3 pairs appart and for
    # the CB-CB pairs with a threshold of 0.5 nm
    clashes = StructuralIntegrityMetrics.general_clashes(
        cln_single_frame,
        [("N", "O"), ("CB", "CB")],
        thresholds=[0.35, 0.5],
        res_offset=2,
    )
    assert clashes["N-O clashes"] == 4
    assert clashes["CB-CB clashes"] == 2


def test_general_clashes_raises(cln_single_frame):
    """Test raises for improper inputs"""
    with pytest.raises(RuntimeError):
        clashes = StructuralIntegrityMetrics.general_clashes(
            cln_single_frame,
            [("N", "O"), ("CB", "CB"), ("CA", "CA")],
            thresholds=[0.35, 0.5],
            res_offset=2,
        )

    with pytest.raises(ValueError):
        clashes = StructuralIntegrityMetrics.general_clashes(
            cln_single_frame,
            "silly_input",
            thresholds=[0.35, 0.5],
            res_offset=2,
        )
    with pytest.raises(ValueError):
        clashes = StructuralIntegrityMetrics.general_clashes(
            cln_single_frame,
            [("N", "O")],
            thresholds="silly_input",
            res_offset=2,
        )
