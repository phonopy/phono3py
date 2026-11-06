# SPDX-License-Identifier: BSD-3-Clause
"""Parse displacement dataset."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.displacement import DisplacementDataset

from phono3py.phonon3.displacement_fc3 import Fc3DisplacementDataset


def get_displacements_and_forces_fc3(
    disp_dataset: Fc3DisplacementDataset,
) -> tuple[NDArray, NDArray | None]:
    """Return displacements and forces from disp_dataset.

    Note
    ----
    Dipslacements and forces of all atoms in supercells are returned.

    Parameters
    ----------
    disp_dataset : Fc3DisplacementDataset
        Displacement dataset.

    Returns
    -------
    displacements : ndarray
        Displacements of all atoms in all supercells.
        shape=(snapshots, supercell atoms, 3), dtype='double', order='C'
    forces : ndarray or None
        Forces of all atoms in all supercells.
        shape=(snapshots, supercell atoms, 3), dtype='double', order='C'
        None is returned when forces don't exist.

    """
    if "first_atoms" in disp_dataset:
        natom = disp_dataset["natom"]
        ndisp = len(disp_dataset["first_atoms"])
        for disp1 in disp_dataset["first_atoms"]:
            ndisp += len(disp1["second_atoms"])
        displacements = np.zeros((ndisp, natom, 3), dtype="double", order="C")
        forces = np.zeros_like(displacements)
        indices = []
        count = 0
        forces_count = 0
        for disp1 in disp_dataset["first_atoms"]:
            indices.append(count)
            displacements[count, disp1["number"]] = disp1["displacement"]
            if "forces" in disp1:
                forces_count += 1
                forces[count] = disp1["forces"]
            count += 1

        for disp1 in disp_dataset["first_atoms"]:
            for disp2 in disp1["second_atoms"]:
                if "included" in disp2:
                    if disp2["included"]:
                        indices.append(count)
                else:
                    indices.append(count)
                displacements[count, disp1["number"]] = disp1["displacement"]
                displacements[count, disp2["number"]] += disp2["displacement"]
                if "forces" in disp2:
                    forces_count += 1
                    forces[count] = disp2["forces"]
                count += 1

        if forces_count == 0:
            forces = None  # type: ignore[assignment]
        else:
            forces = np.array(forces[indices], dtype="double", order="C")
            assert forces_count == count

        displacements = np.array(displacements[indices], dtype="double", order="C")
        return displacements, forces
    elif "displacements" in disp_dataset:
        displacements = disp_dataset["displacements"]
        if "forces" in disp_dataset:
            forces = disp_dataset["forces"]
        else:
            forces = None
        return displacements, forces
    else:
        raise RuntimeError("disp_dataset doesn't contain correct information.")


def forces_in_dataset(
    dataset: Fc3DisplacementDataset | DisplacementDataset | None,
) -> bool:
    """Return whether forces in dataset or not."""
    if dataset is None:
        return False
    return "forces" in dataset or (
        "first_atoms" in dataset and "forces" in dataset["first_atoms"][0]
    )
