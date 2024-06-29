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

from phonopy.interface.calculator import write_supercells_with_displacements

from phono3py import Phono3py
from phono3py.interface.calculator import (
    get_additional_info_to_write_fc2_supercells,
    get_additional_info_to_write_supercells,
    get_default_displacement_distance,
)


def create_phono3py_supercells(
    cell_info,
    settings,
    symprec,
    interface_mode="vasp",
    log_level=1,
):
    """Create displacements and supercells.

    Distance unit used is that for the calculator interface.
    The default unit is Angstron.

    """
    optional_structure_info = cell_info["optional_structure_info"]

    if settings.displacement_distance is None:
        distance = get_default_displacement_distance(interface_mode)
    else:
        distance = settings.displacement_distance
    phono3py = Phono3py(
        cell_info["unitcell"],
        cell_info["supercell_matrix"],
        primitive_matrix=cell_info["primitive_matrix"],
        phonon_supercell_matrix=cell_info["phonon_supercell_matrix"],
        is_symmetry=settings.is_symmetry,
        symprec=symprec,
        calculator=interface_mode,
    )
    phono3py.generate_displacements(
        distance=distance,
        cutoff_pair_distance=settings.cutoff_pair_distance,
        is_plusminus=settings.is_plusminus_displacement,
        is_diagonal=settings.is_diagonal_displacement,
        number_of_snapshots=settings.random_displacements,
        random_seed=settings.random_seed,
    )

    if settings.random_displacements_fc2:
        phono3py.generate_fc2_displacements(
            distance=distance,
            is_plusminus=settings.is_plusminus_displacement,
            number_of_snapshots=settings.random_displacements_fc2,
            random_seed=settings.random_seed,
        )

    if log_level:
        print("")
        print('Unit cell was read from "%s".' % optional_structure_info[0])
        print("Displacement distance: %s" % distance)

    ids = []
    disp_cells = []
    for i, cell in enumerate(phono3py.supercells_with_displacements):
        if cell is not None:
            ids.append(i + 1)
            disp_cells.append(cell)

    additional_info = get_additional_info_to_write_supercells(
        interface_mode, phono3py.supercell_matrix
    )
    write_supercells_with_displacements(
        interface_mode,
        phono3py.supercell,
        disp_cells,
        optional_structure_info,
        displacement_ids=ids,
        zfill_width=5,
        additional_info=additional_info,
    )

    if log_level:
        num_disps = len(phono3py.supercells_with_displacements)
        num_disp_files = len(disp_cells)
        print("Number of displacements: %d" % num_disps)
        if settings.cutoff_pair_distance is not None:
            print(
                "Cutoff distance for displacements: %s" % settings.cutoff_pair_distance
            )
            print("Number of displacement supercell files created: %d" % num_disp_files)

    if phono3py.phonon_supercell_matrix is not None:
        additional_info = get_additional_info_to_write_fc2_supercells(
            interface_mode, phono3py.phonon_supercell_matrix
        )
        write_supercells_with_displacements(
            interface_mode,
            phono3py.phonon_supercell,
            phono3py.phonon_supercells_with_displacements,
            optional_structure_info,
            zfill_width=5,
            additional_info=additional_info,
        )

        if log_level:
            print("Number of displacements for special fc2: %d" % num_disps)

    return phono3py
