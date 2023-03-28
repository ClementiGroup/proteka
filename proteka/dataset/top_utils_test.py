import pytest
from .top_utils import *
import mdtraj as md


@pytest.mark.parametrize(
    "top_, dict_", [(md.Topology(), {"chains": [], "bonds": []})]
)
def test_top_dict(top_, dict_):
    assert top2dict(top_) == dict_
    new_top = dict2top(dict_)
    assert isinstance(new_top, md.Topology)
    assert new_top.n_atoms == top_.n_atoms


def test_top_json(example_json_topology):
    top = json2top(example_json_topology)
    top_json = top2json(top)
    assert top_json == example_json_topology
    assert top.n_residues == 3
    assert top.n_chains == 1
    assert top.n_atoms == 22
    assert top.n_bonds == 21
