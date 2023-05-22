"""Test calculator
"""
import numpy as np
import mdtraj as md
import pytest

from proteka.dataset import Ensemble
from proteka.quantity import Quantity
from proteka.metrics import Featurizer
from proteka.metrics import StructuralIntegrityMetrics
from proteka.metrics.utils import get_6_bead_frame, get_CLN_trajectory


@pytest.fixture
def single_frame():
    traj = get_6_bead_frame()
    ensemble = Ensemble("6bead", traj.top, Quantity(traj.xyz, "nm"))
    return ensemble


@pytest.fixture
def cln_single_frame():
    traj = get_CLN_trajectory(single_frame=True)
    ensemble = Ensemble("6bead", traj.top, Quantity(traj.xyz, "nm"))
    return ensemble


def test_ca_clashes(single_frame):
    clashes = StructuralIntegrityMetrics.ca_clashes(single_frame)
    assert clashes["CA-CA clashes"] == 1


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


def test_general_clashes_len_raises(cln_single_frame):
    with pytest.raises(RuntimeError):
        clashes = StructuralIntegrityMetrics.general_clashes(
            cln_single_frame,
            [("N", "O"), ("CB", "CB"), ("CA", "CA")],
            thresholds=[0.35, 0.5],
            res_offset=2,
        )


def test_general_clashes_type_raises(cln_single_frame):
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


def test_general_clashes_atom_type_raises(cln_single_frame):
    with pytest.raises(RuntimeError):
        clashes = StructuralIntegrityMetrics.general_clashes(
            cln_single_frame,
            [("silly_atom", "O")],
            thresholds=[0.35, 0.5],
            res_offset=2,
        )
