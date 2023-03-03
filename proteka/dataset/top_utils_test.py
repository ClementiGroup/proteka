import pytest
from .top_utils import *
import mdtraj as md

example_json = '{"chains": [{"index": 0, "residues": [{"index": 0, "resSeq": 1, "name": "ACE", "segment_id": "", "atoms": [{"index": 0, "name": "H1", "element": "H"}, {"index": 1, "name": "CH3", "element": "C"}, {"index": 2, "name": "H2", "element": "H"}, {"index": 3, "name": "H3", "element": "H"}, {"index": 4, "name": "C", "element": "C"}, {"index": 5, "name": "O", "element": "O"}]}, {"index": 1, "resSeq": 2, "name": "ALA", "segment_id": "", "atoms": [{"index": 6, "name": "N", "element": "N"}, {"index": 7, "name": "H", "element": "H"}, {"index": 8, "name": "CA", "element": "C"}, {"index": 9, "name": "HA", "element": "H"}, {"index": 10, "name": "CB", "element": "C"}, {"index": 11, "name": "HB1", "element": "H"}, {"index": 12, "name": "HB2", "element": "H"}, {"index": 13, "name": "HB3", "element": "H"}, {"index": 14, "name": "C", "element": "C"}, {"index": 15, "name": "O", "element": "O"}]}, {"index": 2, "resSeq": 3, "name": "NME", "segment_id": "", "atoms": [{"index": 16, "name": "N", "element": "N"}, {"index": 17, "name": "H", "element": "H"}, {"index": 18, "name": "C", "element": "C"}, {"index": 19, "name": "H1", "element": "H"}, {"index": 20, "name": "H2", "element": "H"}, {"index": 21, "name": "H3", "element": "H"}]}]}], "bonds": [[1, 4], [4, 5], [0, 1], [1, 2], [1, 3], [4, 6], [8, 14], [14, 15], [8, 10], [8, 9], [6, 8], [10, 11], [10, 12], [10, 13], [6, 7], [14, 16], [18, 19], [18, 20], [18, 21], [16, 18], [16, 17]]}'


@pytest.mark.parametrize(
    "top_, dict_", [(md.Topology(), {"chains": [], "bonds": []})]
)
def test_top_dict(top_, dict_):
    assert top2dict(top_) == dict_
    new_top = dict2top(dict_)
    assert isinstance(new_top, md.Topology)
    assert new_top.n_atoms == top_.n_atoms


def test_top_json():
    top = json2top(example_json)
    top_json = top2json(top)
    assert top_json == example_json
    assert top.n_residues == 3
    assert top.n_chains == 1
    assert top.n_atoms == 22
    assert top.n_bonds == 21
