"""Utilities to move forces and displacements in and out of an fc4 dataset.

The fc4 type-1 displacement dataset (see
:mod:`phono3py.phonon4.displacement_fc4`) is nested three levels deep and is
self-contained: every node -- first atoms, second atoms, and third atoms -- is a
supercell whose forces are computed, namely ``F(a1)``, ``F(a1+a2)``, and
``F(a1+a2+a3)`` respectively. Each node carries a running ``id`` assigned level
by level (all first atoms first, then all second atoms, then all third atoms).

These helpers use ``id`` as the canonical flat index (supercell ``id - 1``), so
the flat ordering of supercells, displacements, and forces is level-grouped and
consistent with the ids stored in the dataset.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms

from phono3py.phonon4.displacement_fc4 import Fc4Type1DisplacementDataset


def count_supercells_fc4(disp_dataset: Fc4Type1DisplacementDataset) -> int:
    """Return the number of supercells (nodes) in the fc4 dataset."""
    count = 0
    for first_atom in disp_dataset["first_atoms"]:
        count += 1
        for second_atom in first_atom["second_atoms"]:
            count += 1
            count += len(second_atom["third_atoms"])
    return count


def get_displacements_and_forces_fc4(
    disp_dataset: Fc4Type1DisplacementDataset,
) -> tuple[NDArray[np.double], NDArray[np.double] | None]:
    """Return cumulative displacements and forces of all fc4 supercells.

    Each supercell is indexed by ``id - 1``. The displacement of a node is
    cumulative: a first-atom node displaces Atom 1 only, a second-atom node
    displaces Atoms 1 and 2, and a third-atom node displaces Atoms 1, 2, and 3.

    Parameters
    ----------
    disp_dataset : Fc4Type1DisplacementDataset
        Nested type-1 fc4 displacement dataset.

    Returns
    -------
    displacements : ndarray
        Cumulative displacements of all atoms in all supercells.
        shape=(supercells, supercell atoms, 3), dtype='double', order='C'
    forces : ndarray or None
        Forces of all atoms in all supercells, or ``None`` when no node
        carries forces.
        shape=(supercells, supercell atoms, 3), dtype='double', order='C'

    """
    natom = disp_dataset["natom"]
    ncells = count_supercells_fc4(disp_dataset)
    displacements = np.zeros((ncells, natom, 3), dtype="double", order="C")
    forces = np.zeros_like(displacements)
    forces_count = 0

    for first_atom in disp_dataset["first_atoms"]:
        disp1 = np.zeros((natom, 3), dtype="double")
        disp1[first_atom["number"]] += first_atom["displacement"]
        displacements[first_atom["id"] - 1] = disp1
        if "forces" in first_atom:
            forces[first_atom["id"] - 1] = first_atom["forces"]
            forces_count += 1
        for second_atom in first_atom["second_atoms"]:
            disp2 = disp1.copy()
            disp2[second_atom["number"]] += second_atom["displacement"]
            displacements[second_atom["id"] - 1] = disp2
            if "forces" in second_atom:
                forces[second_atom["id"] - 1] = second_atom["forces"]
                forces_count += 1
            for third_atom in second_atom["third_atoms"]:
                disp3 = disp2.copy()
                disp3[third_atom["number"]] += third_atom["displacement"]
                displacements[third_atom["id"] - 1] = disp3
                if "forces" in third_atom:
                    forces[third_atom["id"] - 1] = third_atom["forces"]
                    forces_count += 1

    if forces_count == 0:
        return displacements, None
    if forces_count != ncells:
        raise RuntimeError(
            f"fc4 dataset has partial forces ({forces_count} of {ncells} "
            "supercells); all or none of the supercells must carry forces."
        )
    return displacements, forces


def get_supercells_with_displacements_fc4(
    supercell: PhonopyAtoms, disp_dataset: Fc4Type1DisplacementDataset
) -> list[PhonopyAtoms]:
    """Return the fc4 supercells with displacements applied, ordered by id.

    The returned list is indexed by ``id - 1`` and is suitable for batched
    force evaluation (e.g. with a calculator or an MLP). Pass the resulting
    forces back with :func:`set_forces_in_dataset_fc4` using the same order.

    """
    displacements, _ = get_displacements_and_forces_fc4(disp_dataset)
    base = supercell.positions
    return [
        PhonopyAtoms(
            cell=supercell.cell,
            symbols=supercell.symbols,
            positions=base + disp,
        )
        for disp in displacements
    ]


def set_forces_in_dataset_fc4(
    disp_dataset: Fc4Type1DisplacementDataset,
    forces: NDArray[np.double] | list[NDArray[np.double]],
) -> None:
    """Store supercell forces into the fc4 dataset in place, ordered by id.

    ``forces`` must be ordered to match
    :func:`get_supercells_with_displacements_fc4`, i.e. supercell ``id - 1``.

    """
    ncells = count_supercells_fc4(disp_dataset)
    forces = np.asarray(forces, dtype="double")
    if forces.shape[0] != ncells:
        raise RuntimeError(
            f"Number of force sets ({forces.shape[0]}) does not match the "
            f"number of fc4 supercells ({ncells})."
        )
    for first_atom in disp_dataset["first_atoms"]:
        first_atom["forces"] = np.array(
            forces[first_atom["id"] - 1], dtype="double", order="C"
        )
        for second_atom in first_atom["second_atoms"]:
            second_atom["forces"] = np.array(
                forces[second_atom["id"] - 1], dtype="double", order="C"
            )
            for third_atom in second_atom["third_atoms"]:
                third_atom["forces"] = np.array(
                    forces[third_atom["id"] - 1], dtype="double", order="C"
                )


def forces_in_dataset_fc4(disp_dataset: Fc4Type1DisplacementDataset | None) -> bool:
    """Return whether the fc4 dataset already carries forces."""
    if disp_dataset is None or not disp_dataset["first_atoms"]:
        return False
    return "forces" in disp_dataset["first_atoms"][0]
