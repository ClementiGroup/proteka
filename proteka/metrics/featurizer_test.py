
import numpy as np
import mdtraj as md

from proteka.dataset import Emsemble
from proteka.metrics import Featurizer
from proteka.metrics.utils import generate_grid_polymer


def test_get_ca_bonds():
    grid_size = 0.4
    traj = _generate_grid_polymer(n_frames=10, n_atoms=5, grid_size=grid_size)
    ens = Ensemble("CAgrid", traj.top, Quantity(traj.xyz, "nm"))
    featurizer = Featurizer(ens)
    ca_bonds = featurizer.get_ca_bonds()
    assert np.all(ca_bonds.in_unit_of('nm')) == grid_size
    
if __name__ == "__main__":
    test_get_ca_bonds()