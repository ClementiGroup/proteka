"""top_utils: (de-)serialization of `mdtraj.Topology` objects to(/from) JSON string via
python `dict`, rewritten to be compatible with mdtraj's HDF5 interface."""

__all__ = ["dict2top", "top2dict", "json2top", "top2json"]

import mdtraj as md
import json
from warnings import warn


def dict2top(top_dict):
    """Transforms a `dict` back to `mdtraj.Topology` in a lossless manner.
    Raises `ValueError` when the input does not contain keys required by a topology
    or the value is in a different format or order in the transformation `top2dict`.

    Parameters
    ----------
    top_dict : dict
        A Python `dict` that matches the output format of `top2dict` saving routine.

    Returns
    -------
    mdtraj.Topology
        The topology encoded by the dictionary.

    Raises
    ------
    ValueError
        When the format of input `dict` dieviates from a possible output of `top2dict`.
    RuntimeError
        When an Error is thrown during parsing input for a certain residue/chain/bond.
    """
    if "chains" not in top_dict or not isinstance(top_dict["chains"], list):
        raise ValueError(
            "Incompatible format. `chains` should be a list in the input `top_dict`."
        )
    if "bonds" not in top_dict or not isinstance(top_dict["chains"], list):
        raise ValueError(
            "Incompatible format. `bonds` should appear in the input `top_dict`."
        )
    top = md.Topology()
    # parsing the chain list
    res_id = 0
    atom_id = 0
    use_resSeq = True
    for chain_id, chain_dict in enumerate(top_dict["chains"]):
        chain = top.add_chain()
        if "index" in chain_dict:
            assert (
                int(chain_dict["index"]) == chain_id
            ), "Invalid chain order in `top_dict`."
        if "residues" not in chain_dict or not isinstance(
            chain_dict["residues"], list
        ):
            raise ValueError(
                f"Incompatible format. `resiudes` should be a list in the input chain #{chain_id}."
            )
        for res in chain_dict["residues"]:
            try:
                # parsing the residue dict
                assert (
                    int(res["index"]) == res_id
                ), "Invalid residue order in `top_dict`."
                res_id += 1
                name = res["name"]
                if use_resSeq and "resSeq" not in res:
                    warn(
                        "`resSeq` does not exist in input `top_dict`, falling back to mdtraj's convention"
                    )
                resSeq = res.get("resSeq", None)
                segment_id = res.get("segment_id", "")
                residue = top.add_residue(
                    name, chain, resSeq=resSeq, segment_id=segment_id
                )
                atoms = res["atoms"]
                for atom in atoms:
                    # parsing the atom dict
                    try:
                        assert (
                            int(atom["index"]) == atom_id
                        ), "Invalid atom order in `top_dict`."
                        atom_id += 1
                        name = atom["name"]
                        elem_sym = atom["element"]
                        try:
                            element = md.element.get_by_symbol(elem_sym)
                        except KeyError as e:
                            # unknown element or virtual sites?
                            element = md.element.virtual
                            warn(
                                f'Unknown element or virtual site "{elem_sym}" for atom #{atom_id}.'
                            )
                        top.add_atom(name, element, residue)
                    except:
                        raise RuntimeError(f"Error in parsing atom #{atom_id}")
            except:
                raise RuntimeError(
                    f"Error in parsing residue #{res_id} in chain #{chain_id}:"
                )
    # parsing the bond list
    for bond_id, bond in enumerate(top_dict["bonds"]):
        try:
            top.add_bond(top.atom(int(bond[0])), top.atom(int(bond[1])))
        except IndexError as e:
            raise ValueError(
                f"Error in parsing bond #{bond_id}: {bond}, atom index out of range."
            )
    return top


def top2dict(top):
    """Transforms a `mdtraj.Topology` object to python `dict` in a lossless manner.

    Parameters
    ----------
    top : mdtraj.Topology
        The molecular topology to be transformed.

    Returns
    -------
    dict
        The dictionary containing the information of the input `top`.

    Raises
    ------
    ValueError
        When input `top` is not a valid `mdtraj.Topology` object.
    """
    if not isinstance(top, md.Topology):
        raise ValueError(f"Input {top} is not a valid mdtraj.Topology object.")
    out_dict = {}
    chains = []
    for c in top.chains:
        chain = {
            "index": c.index,
            "residues": [],
        }
        for res in c.residues:
            res_dict = {
                "index": res.index,
                "resSeq": res.resSeq,
                "name": res.name,
                "segment_id": res.segment_id,
            }
            atoms = []
            for atom in res.atoms:
                atom_dict = {
                    "index": atom.index,
                    "name": atom.name,
                    "element": atom.element.symbol,
                }
                atoms.append(atom_dict)
            res_dict["atoms"] = atoms
            chain["residues"].append(res_dict)
        chains.append(chain)
    out_dict["chains"] = chains
    bonds = [[b.atom1.index, b.atom2.index] for b in top.bonds]
    out_dict["bonds"] = bonds
    return out_dict


def json2top(top_json_string):
    """Transforms a JSON string back to `mdtraj.Topology` in a lossless manner.

    Parameters
    ----------
    top_json_string : str
        A serialized JSON string containing a topology information, most likely an
        output of `top2json`.

    Returns
    -------
    mdtraj.Topology
        The topology defined by the input `top_json_string`.

    Raises
    ------
    ValueError
        When the input does not contain keys required by a topology or the value is in a
        different format or order from the possible output of `top2json`.
    """
    top_dict = json.loads(top_json_string)
    top = dict2top(top_dict)
    return top


def top2json(top):
    """Transforms a `mdtraj.Topology` object to a JSON string in a lossless manner.

    Parameters
    ----------
    top : mdtraj.Topology
        The topology to be serialized.

    Returns
    -------
    str
        A JSON string containing the information of the `top`.
    """
    top_dict = top2dict(top)
    return json.dumps(top_dict)
