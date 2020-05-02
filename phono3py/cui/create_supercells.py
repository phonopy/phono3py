# Copyright (C) 2015 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
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

from phono3py import Phono3py
from phono3py.file_IO import write_disp_fc3_yaml, write_disp_fc2_yaml
from phono3py.interface.calculator import (
    get_additional_info_to_write_fc2_supercells,
    get_additional_info_to_write_supercells,
    get_default_displacement_distance)
from phonopy.interface.calculator import write_supercells_with_displacements


def create_phono3py_supercells(unitcell,
                               supercell_matrix,
                               phonon_supercell_matrix,
                               displacement_distance,
                               is_plusminus,
                               is_diagonal,
                               cutoff_pair_distance,
                               optional_structure_info,
                               is_symmetry,
                               symprec,
                               interface_mode='vasp',
                               output_filename=None,
                               log_level=1):
    if displacement_distance is None:
        distance = get_default_displacement_distance(interface_mode)
    else:
        distance = displacement_distance
    phono3py = Phono3py(
        unitcell,
        supercell_matrix,
        phonon_supercell_matrix=phonon_supercell_matrix,
        is_symmetry=is_symmetry,
        symprec=symprec)
    supercell = phono3py.get_supercell()
    phono3py.generate_displacements(
        distance=distance,
        cutoff_pair_distance=cutoff_pair_distance,
        is_plusminus=is_plusminus,
        is_diagonal=is_diagonal)
    dds = phono3py.get_displacement_dataset()

    if log_level:
        print('')
        print("Displacement distance: %s" % distance)

    if output_filename is None:
        filename = 'disp_fc3.yaml'
    else:
        filename = 'disp_fc3.' + output_filename + '.yaml'

    num_disps, num_disp_files = write_disp_fc3_yaml(dds,
                                                    supercell,
                                                    filename=filename)
    cells_with_disps = phono3py.supercells_with_displacements
    ids = []
    disp_cells = []
    for i, cell in enumerate(cells_with_disps):
        if cell is not None:
            ids.append(i + 1)
            disp_cells.append(cell)

    additional_info = get_additional_info_to_write_supercells(
        interface_mode, phono3py)
    write_supercells_with_displacements(interface_mode,
                                        supercell,
                                        disp_cells,
                                        optional_structure_info,
                                        displacement_ids=ids,
                                        zfill_width=5,
                                        additional_info=additional_info)

    if log_level:
        print("Number of displacements: %d" % num_disps)
        if cutoff_pair_distance is not None:
            print("Cutoff distance for displacements: %s" %
                  cutoff_pair_distance)
            print("Number of displacement supercell files created: %d" %
                  num_disp_files)

    if phonon_supercell_matrix is not None:
        phonon_dds = phono3py.phonon_dataset
        phonon_supercell = phono3py.phonon_supercell
        phonon_supercell_matrix = phono3py.phonon_supercell_matrix
        if output_filename is None:
            filename = 'disp_fc2.yaml'
        else:
            filename = 'disp_fc2.' + output_filename + '.yaml'

        num_disps = write_disp_fc2_yaml(phonon_dds,
                                        phonon_supercell,
                                        filename=filename)
        cells_with_disps = phono3py.phonon_supercells_with_displacements

        additional_info = get_additional_info_to_write_fc2_supercells(
            interface_mode, phono3py)
        write_supercells_with_displacements(interface_mode,
                                            supercell,
                                            cells_with_disps,
                                            optional_structure_info,
                                            zfill_width=5,
                                            additional_info=additional_info)

        if log_level:
            print("Number of displacements for special fc2: %d" % num_disps)

    return phono3py
