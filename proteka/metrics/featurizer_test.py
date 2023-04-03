import numpy as np

from proteka.dataset import Ensemble
from proteka.quantity import Quantity
from proteka.metrics import Featurizer
from proteka.metrics.utils import generate_grid_polymer


def test_get_ca_bonds():
    grid_size = 0.4
    traj = generate_grid_polymer(n_frames=10, n_atoms=5, grid_size=grid_size)
    ens = Ensemble("CAgrid", traj.top, Quantity(traj.xyz, "nm"))
    featurizer = Featurizer(ens)
    featurizer.add_ca_bonds()
    print(ens.ca_bonds)
    assert np.all(np.isclose(ens.ca_bonds, grid_size))


if __name__ == "__main__":
    test_get_ca_bonds()
