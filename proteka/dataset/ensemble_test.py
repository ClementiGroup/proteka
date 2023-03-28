
from pathlib import Path
from proteka import Ensemble
import pytest
import numpy as np

from .top_utils_test import example_json
from .top_utils import json2top, top2json



@pytest.fixture
def example_ensemble():
    """Create a dataset from a json topology."""
    top = json2top(example_json)
    ensemble = Ensemble(name="example_ensemble", top=top, coords=np.zeros((10, top.n_atoms, 3)))
    return ensemble

def test_ensemble_attributes(example_ensemble):
    """Test ensemble creation."""
    assert example_ensemble.name == "example_ensemble"
    assert example_ensemble.n_atoms == 22
    assert example_ensemble.n_frames == 10
    assert top2json(example_ensemble.top) == example_json
    assert example_ensemble.n_trjs == 1
    assert len(example_ensemble.trajectory_slices) == 1
    assert len(example_ensemble.trajectories) == 1
    assert example_ensemble.trajectory_slices["default"] == slice(0, 10, 1)


def test_trajectory_slices(example_ensemble):
    example_ensemble.trajectory_slices["test_slice"] = slice(0,1000,1)
    example_ensemble.get_mdtrajs()


