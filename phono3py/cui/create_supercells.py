"""Command line user interface to create supercells."""

# Copyright (C) 2015 Atsushi Togo
# All rights reserved.
#
# This file is part of phono3py.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

import dataclasses
import os
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from phonopy.cui.collect_cell_info import CellInfoResult
from phonopy.cui.collect_cell_info import get_cell_info as phonopy_get_cell_info
from phonopy.interface.calculator import write_supercells_with_displacements
from phonopy.structure.cells import print_cell

from phono3py import Phono3py
from phono3py.cui.settings import Phono3pySettings
from phono3py.cui.show_log import print_supercell_matrix
from phono3py.interface.calculator import (
    get_additional_info_to_write_fc2_supercells,
    get_additional_info_to_write_supercells,
    get_default_displacement_distance,
)
from phono3py.interface.fc_calculator import determine_cutoff_pair_distance
from phono3py.interface.phono3py_yaml import Phono3pyYaml


@dataclasses.dataclass
class Phono3pyCellInfoResult(CellInfoResult):
    """Phono3py cell info result.

    This is a subclass of CellInfoResult to add phonon supercell matrix.

    """

    phono3py_yaml: Phono3pyYaml | None = None
    phonon_supercell_matrix: Sequence[Sequence[int]] | NDArray | None = None


def get_cell_info(
    settings: Phono3pySettings,
    cell_filename: str | os.PathLike | None,
    log_level: int,
    load_phonopy_yaml: bool = True,
) -> Phono3pyCellInfoResult:
    """Return calculator interface and crystal structure information."""
    cell_info = phonopy_get_cell_info(
        settings,
        cell_filename,
        log_level=log_level,
        load_phonopy_yaml=load_phonopy_yaml,
        phonopy_yaml_cls=Phono3pyYaml,
    )

    cell_info_dict = dataclasses.asdict(cell_info)
    cell_info_dict["phono3py_yaml"] = cell_info_dict.pop("phonopy_yaml")
    cell_info = Phono3pyCellInfoResult(
        **cell_info_dict,
        phonon_supercell_matrix=settings.phonon_supercell_matrix,
    )

    ph3py_yaml = cell_info.phono3py_yaml
    if cell_info.phonon_supercell_matrix is None and ph3py_yaml:
        cell_info.phonon_supercell_matrix = ph3py_yaml.phonon_supercell_matrix

    return cell_info


def create_phono3py_supercells(
    cell_info: Phono3pyCellInfoResult,
    settings: Phono3pySettings,
    symprec: float,
    interface_mode: str | None = "vasp",
    log_level: int = 1,
):
    """Create displacements and supercells.

    Distance unit used is that for the calculator interface.
    The default unit is Angstrom.

    """
    optional_structure_info = cell_info.optional_structure_info

    if settings.displacement_distance is None:
        distance = get_default_displacement_distance(interface_mode)
    else:
        distance = settings.displacement_distance
    ph3 = Phono3py(
        cell_info.unitcell,
        cell_info.supercell_matrix,
        primitive_matrix=cell_info.primitive_matrix,
        phonon_supercell_matrix=cell_info.phonon_supercell_matrix,
        is_symmetry=settings.is_symmetry,
        symprec=symprec,
        calculator=interface_mode,
    )

    if log_level:
        print("")
        print('Unit cell was read from "%s".' % optional_structure_info[0])
        print("-" * 32 + " unit cell " + "-" * 33)  # 32 + 11 + 33 = 76
        print_cell(ph3.unitcell)
        print("-" * 76)
        print_supercell_matrix(ph3.supercell_matrix, ph3.phonon_supercell_matrix)
        if ph3.primitive_matrix is not None:
            print("Primitive matrix:")
            for v in ph3.primitive_matrix:
                print("  %s" % v)
        print("Displacement distance: %s" % distance)

    cutoff_pair_distance = determine_cutoff_pair_distance(
        fc_calculator=settings.fc_calculator,
        fc_calculator_options=settings.fc_calculator_options,
        cutoff_pair_distance=settings.cutoff_pair_distance,
        symfc_memory_size=settings.symfc_memory_size,
        random_displacements=settings.random_displacements,
        supercell=ph3.supercell,
        primitive=ph3.primitive,
        symmetry=ph3.symmetry,
        log_level=log_level,
    )
    ph3.generate_displacements(
        distance=distance,
        cutoff_pair_distance=cutoff_pair_distance,
        is_plusminus=settings.is_plusminus_displacement,
        is_diagonal=settings.is_diagonal_displacement,
        number_of_snapshots=settings.random_displacements,
        random_seed=settings.random_seed,
        number_estimation_factor=settings.rd_number_estimation_factor,
    )

    if (
        settings.random_displacements_fc2
        or settings.phonon_supercell_matrix is not None
    ):
        ph3.generate_fc2_displacements(
            distance=distance,
            is_plusminus=settings.is_plusminus_displacement_fc2,
            number_of_snapshots=settings.random_displacements_fc2,
            random_seed=settings.random_seed,
        )

    ids = []
    disp_cells = []
    for i, cell in enumerate(ph3.supercells_with_displacements):
        if cell is not None:
            ids.append(i + 1)
            disp_cells.append(cell)

    additional_info = get_additional_info_to_write_supercells(
        interface_mode, ph3.supercell_matrix
    )

    additional_info["supercell_matrix"] = ph3.supercell_matrix

    write_supercells_with_displacements(
        interface_mode,
        ph3.supercell,
        disp_cells,
        optional_structure_info,
        displacement_ids=ids,
        zfill_width=5,
        additional_info=additional_info,
    )

    if log_level:
        num_disps = len(ph3.supercells_with_displacements)
        num_disp_files = len(disp_cells)
        print(f"Number of displacements: {num_disps}")
        if cutoff_pair_distance is not None:
            print(f"Cutoff distance for displacements: {cutoff_pair_distance}")
            print(f"Number of displacement supercell files created: {num_disp_files}")

    if (
        ph3.phonon_supercell_matrix is not None
        and ph3.phonon_supercells_with_displacements is not None
    ):
        num_disps = len(ph3.phonon_supercells_with_displacements)
        additional_info = get_additional_info_to_write_fc2_supercells(
            interface_mode, ph3.phonon_supercell_matrix
        )
        write_supercells_with_displacements(
            interface_mode,
            ph3.phonon_supercell,
            ph3.phonon_supercells_with_displacements,
            optional_structure_info,
            zfill_width=5,
            additional_info=additional_info,
        )

        if log_level:
            print("Number of displacements for special fc2: %d" % num_disps)

    if log_level:
        identity = np.eye(3, dtype=int)
        n_pure_trans = sum(
            [
                (r == identity).all()
                for r in ph3.symmetry.symmetry_operations["rotations"]
            ]
        )

        if len(ph3.supercell) // len(ph3.primitive) != n_pure_trans:
            print("*" * 72)
            print(
                "Note: "
                'A better primitive cell can be chosen by using "--pa auto" option.'
            )
            print("*" * 72)

    return ph3
