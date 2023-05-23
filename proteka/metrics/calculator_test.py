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


def test_general_default_clashes(cln_single_frame):
    # Checks to makes sure a "shrunken" CLN trajecory violates certain clashes
    # between all atoms of the two terminal residues when using the default
    # VDW/allowance based clashes

    ens = cln_single_frame
    original_coords = ens.get_quantity("coords").raw_value
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
    assert clashes["N-N clashes"] == 1
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

    assert clashes["O-O clashes"] == 1


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
