# SPDX-License-Identifier: BSD-3-Clause
"""ALM interface for force constants calculation."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms

from phono3py.phonon3.dataset import get_displacements_and_forces_fc3
from phono3py.phonon3.displacement_fc3 import Fc3DisplacementDataset


def get_fc3_calc_dataset_wien2k(
    force_filenames: Sequence[str | os.PathLike],
    supercell: PhonopyAtoms,
    disp_dataset: Fc3DisplacementDataset,
    wien2k_P1_mode: bool = False,
    symmetry_tolerance: float | None = None,
    verbose: bool = False,
) -> dict[str, list[NDArray[np.double]]]:
    """Read Wien2k output files and parse force sets."""
    from phonopy.interface.wien2k import parse_set_of_forces

    disps, _ = get_displacements_and_forces_fc3(disp_dataset)
    force_sets = parse_set_of_forces(
        disps,  # type: ignore[arg-type]
        force_filenames,
        supercell,
        wien2k_P1_mode=wien2k_P1_mode,
        symmetry_tolerance=symmetry_tolerance,
        verbose=verbose,
    )
    return {"forces": force_sets}
