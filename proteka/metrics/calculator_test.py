"""Test calculator
"""
import numpy as np
import mdtraj as md
import pytest

from proteka.dataset import Ensemble
from proteka.quantity import Quantity
from proteka.metrics import Featurizer
from proteka.metrics import StructuralIntegrityMetrics, EnsembleQualityMetrics
from proteka.metrics.utils import get_6_bead_frame, get_CA_CLN_trajectory


@pytest.fixture
def single_frame():
    traj = get_6_bead_frame()
    ensemble = Ensemble("6bead", traj.top, Quantity(traj.xyz, "nm"))
    return ensemble


@pytest.fixture
def get_two_ensembles():
    ref_traj = get_CA_CLN_trajectory()
    target_traj = get_CA_CLN_trajectory()
    ref_ensemble = Ensemble.from_mdtraj_trj("ref", ref_traj)
    target_ensemble = Ensemble.from_mdtraj_trj("target", target_traj)
    return target_ensemble, ref_ensemble


def test_ca_clashes(single_frame):
    clashes = StructuralIntegrityMetrics.ca_clashes(single_frame)
    assert clashes["N clashes"] == 1


def test_basic_metric_run(get_two_ensembles):
    target_ensemble, ref_ensemble = get_two_ensembles

    target_feat = Featurizer(target_ensemble)
    ref_feat = Featurizer(ref_ensemble)

    # target_feat.add_ca_distances()
    target_feat.add_rg()
    # ref_feat.add_ca_distances()
    ref_feat.add_rg()

    print(target_ensemble.list_quantities())
    print(ref_ensemble.list_quantities())

    eqm = EnsembleQualityMetrics()
    results = eqm(target_ensemble, ref_ensemble, "rg", "all")
