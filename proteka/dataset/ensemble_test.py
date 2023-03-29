import h5py

from pathlib import Path
from proteka import Ensemble, UnitSystem, Quantity
import pytest
import numpy as np

from .top_utils import json2top, top2json


@pytest.fixture
def example_ensemble(example_json_topology):
    """Create a dataset from a json topology."""
    top = json2top(example_json_topology)
    ensemble = Ensemble(
        name="example_ensemble", top=top, coords=np.zeros((10, top.n_atoms, 3))
    )
    assert top2json(ensemble.top) == example_json_topology
    return ensemble


def test_casting(example_ensemble):
    assert example_ensemble["coords"].unit == "nanometers"
    example_ensemble.forces = np.zeros((10, example_ensemble.n_atoms, 3))
    assert isinstance(example_ensemble.forces, np.ndarray)
    assert isinstance(example_ensemble["forces"], Quantity)
    assert example_ensemble["forces"].unit == "kilojoules/mole/nanometers"
    with pytest.raises(ValueError):
        # incompatible unit
        example_ensemble.forces = Quantity(
            np.zeros((10, example_ensemble.n_atoms, 3)), "nanometers"
        )
    with pytest.raises(ValueError):
        # incompatible shape
        example_ensemble.forces = np.zeros((10, example_ensemble.n_atoms, 2))
    # assert isinstance(example_ensemble["forces"], np.ndarray)
    example_ensemble.velocities = np.zeros((10, example_ensemble.n_atoms, 3))
    assert example_ensemble["velocities"].unit == "nanometers/picoseconds"
    example_ensemble.weights = np.ones((10,))
    assert example_ensemble["weights"].unit == "dimensionless"

    example_ensemble.velocities = Quantity(
        np.ones((10, example_ensemble.n_atoms, 3)), "Angstrom/picoseconds"
    )
    assert example_ensemble["velocities"].unit == "nanometers/picoseconds"
    assert example_ensemble["velocities"][0, 0, 0] == 0.1


def test_ensemble_attributes(example_ensemble):
    """Test ensemble creation."""
    assert example_ensemble.name == "example_ensemble"
    assert example_ensemble.n_atoms == 22
    assert example_ensemble.n_frames == 10
    assert example_ensemble.n_trjs == 1
    assert len(example_ensemble.trajectory_slices) == 1
    assert len(example_ensemble.trajectories) == 1
    assert example_ensemble.trajectory_slices["default"] == slice(0, 10, 1)
    assert example_ensemble.unit_system == UnitSystem()


def test_ensemble_trajectories(example_ensemble):
    """Test trajectory utilities."""
    with pytest.raises(TypeError):
        example_ensemble.trajectory_slices["test_slice"] = slice(0, 10, 1)

    assert ("default",) == tuple(example_ensemble.trajectory_slices.keys())
    assert ("default",) == tuple(example_ensemble.trajectories.keys())
    assert ("default",) == tuple(example_ensemble.trajectory_indices.keys())

    example_ensemble.register_trjs(
        {"part1": slice(0, 5, 1), "part2": slice(5, 10, 1)}
    )
    assert "default" not in example_ensemble.trajectory_slices
    assert ("part1", "part2") == tuple(
        example_ensemble.trajectory_slices.keys()
    )
    assert ("part1", "part2") == tuple(example_ensemble.trajectories.keys())
    assert ("part1", "part2") == tuple(
        example_ensemble.trajectory_indices.keys()
    )

    trajectories = example_ensemble.get_mdtraj_trjs()
    assert len(trajectories) == 2
    assert trajectories["part1"].n_frames == 5
    assert trajectories["part2"].n_frames == 5

    trajectory = example_ensemble.get_all_in_one_mdtraj_trj()
    assert trajectory.n_frames == 10


def test_ensemble_from_trajectory(example_ensemble):
    """Test ensemble creation from an mdtraj.Trajectory."""
    trajectory = example_ensemble.get_all_in_one_mdtraj_trj()
    ensemble2 = Ensemble.from_mdtraj_trj(trj=trajectory, name="ensemble2")
    assert ensemble2.n_frames == 10
    assert np.allclose(ensemble2.coords, example_ensemble.coords)


def test_ensemble_attributes(example_ensemble):
    """Test builtin and custom fields"""
    example_ensemble.forces = np.zeros((10, example_ensemble.n_atoms, 3))
    assert "forces" in example_ensemble
    assert "velocities" not in example_ensemble
    # a custom field
    assert "custom_field" not in example_ensemble
    example_ensemble.custom_field = np.zeros((5, example_ensemble.n_atoms, 3))
    assert "custom_field" in example_ensemble
    assert example_ensemble["custom_field"].unit == "dimensionless"


def test_ensemble_to_h5(example_ensemble, tmpdir):
    """Test saving and loading ensembles."""
    ensemble = example_ensemble
    ensemble.forces = np.zeros((10, ensemble.n_atoms, 3))
    ensemble.custom_field = np.zeros((5, ensemble.n_atoms, 3))
    with h5py.File(tmpdir / "test.h5", "w") as f:
        ensemble.write_to_hdf5(f, name="example_ensemble")

    with h5py.File(tmpdir / "test.h5", "r") as f:
        group = f["example_ensemble"]
        ensemble2 = Ensemble.from_hdf5(group)

    assert ensemble2.n_frames == 10
    assert np.allclose(ensemble2.coords, ensemble.coords)
    assert np.allclose(ensemble2.forces, ensemble.forces)
    assert np.allclose(ensemble2.custom_field, ensemble.custom_field)
    assert ensemble2.top == ensemble.top
    assert ensemble2.name == ensemble.name
    assert ensemble2["custom_field"].unit == "dimensionless"
