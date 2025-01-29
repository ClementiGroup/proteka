Ensemble
=========

Introduction
------------

About ``Quantity``

A ``Quantity`` wraps a ``numpy.ndarray`` and a ``unit`` (defined in ``proteka.dataset.unit_quantity``). Assigning a ``Quantity`` to an ``Ensemble`` either during initialization or via the dot (.) notation as an attribute:

- If the input is a plain ``numpy.ndarray``, then the unit is assumed as"dimensionless"
- If the input is a ``Quantity``, the input unit will be stored

Retrieving saved ``Quantity``:

- Accessing as an attribute (via dot (.)): returns a ``numpy.ndarray`` with the value of the ``Quantity`` in unit of the stored unit
- Via index bracket ([]): returns the stored ``Quantity`` object, allowing flexible automated unit conversion

List stored quantities: ``.list_quantities()``

* Special cases are "builtin quantities", whose stored units are dictated by the ``unit_system`` (also used instead of the default "dimensionless" during assignment):
* "coords" (ATOMIC_VECTOR): [L]
* "time" (*per-frame* SCALAR): [T]
* "forces" (ATOMIC_VECTOR): [E]/[L]
* "velocities" (ATOMIC_VECTOR): [L]/[T]
* "cell_lengths" (BOX_QUANTITIES): [L]
* "cell_angles": (BOX_QUANTITIES): degree
In addition, the above quantities are tied to the system molecule via the shape,
i.e., each *per-frame* quantity having the same number of frames as ``self.coords``,
and correspond to the same number of atoms as indicated by ``self.top``, if it is an
_ATOMIC_VECTOR_.

Trajectories:
-------------

Storing the information about which samples contained in the ``Ensemble`` come from
which trajectory.
Trajectories are sequential. Therefore, samples from different trajectories are
expected to be non-overlapping slices.
Trajectories info is supposed to be stored either during the Ensemble initialization
or after with ``.register_trjs`` method.

Properties
---------- 

- ``.n_trjs`` (int): number of trajectories
- ``.n_frames_per_trj`` (Dict[str, int]): dictionary of number of frames in each trajectory
- ``.trajectory_slices`` or ``.trjs`` or ``.trajectories`` (Dict[str, slice]): Python ``slice``s for slicing Ensemble quantities according to the ``.trjs`` records
- ``.trj_indices`` (Dict[str, np.ndarray]): indices for different trajectoriesaccording to the ``.trjs`` records

``mdtraj`` interface:
--------------------

- ``.get_mdtraj_trjs()`` (-> Dict[str, mdtraj.Trajectory]): pack an ``Ensemble``'s ``top``and ``coords`` (and unitcell + simulation times, if available) into a dictionary of ``mdtraj.Trajectory`` for analyses according to ``self.trjs``
- ``.get_all_in_one_mdtraj_trj()``: pack all ``coords`` into one ``Trajectory`` object (maybe not suitable for kinetic analyses, such as TICA and MSM!)


Implementations
----------

.. autoclass:: proteka.dataset.Ensemble
    :members: