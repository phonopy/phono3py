"""Command line user interface to create force sets files."""

# Copyright (C) 2024 Atsushi Togo
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

import copy
import sys
from typing import Optional, Union

from phonopy.cui.create_force_sets import check_number_of_force_files
from phonopy.cui.phonopy_script import file_exists, files_exist, print_error
from phonopy.file_IO import parse_FORCE_SETS, write_FORCE_SETS
from phonopy.interface.calculator import get_calc_dataset

from phono3py.file_IO import (
    get_length_of_first_line,
    parse_FORCES_FC2,
    write_FORCES_FC2,
    write_FORCES_FC3,
)
from phono3py.interface.phono3py_yaml import (
    Phono3pyYaml,
    displacements_yaml_lines_type1,
)


def create_FORCES_FC3_and_FORCES_FC2(
    settings,
    cell_filename: Optional[str],
    log_level: Union[bool, int],
):
    """Create FORCES_FC3 and FORCES_FC2 from files."""
    interface_mode = settings.calculator
    ph3py_yaml = None

    #####################
    # Create FORCES_FC3 #
    #####################
    if settings.create_forces_fc3 or settings.create_forces_fc3_file:
        disp_filename_candidates = [
            "phono3py_disp.yaml",
        ]
        if cell_filename is not None:
            disp_filename_candidates.insert(0, cell_filename)
        disp_filenames = files_exist(disp_filename_candidates, log_level, is_any=True)
        disp_filename = disp_filenames[0]
        ph3py_yaml = Phono3pyYaml()
        ph3py_yaml.read(disp_filename)
        if ph3py_yaml.calculator is not None:
            interface_mode = ph3py_yaml.calculator  # overwrite
        disp_dataset = ph3py_yaml.dataset

        if log_level:
            print("")
            print('Displacement dataset was read from "%s".' % disp_filename)

        num_atoms = disp_dataset["natom"]
        num_disps = len(disp_dataset["first_atoms"])
        for d1 in disp_dataset["first_atoms"]:
            for d2 in d1["second_atoms"]:
                if "included" not in d2 or d2["included"]:
                    num_disps += 1

        if settings.create_forces_fc3_file:
            file_exists(settings.create_forces_fc3_file, log_level)
            force_filenames = [x.strip() for x in open(settings.create_forces_fc3_file)]
        else:
            force_filenames = settings.create_forces_fc3

        for filename in force_filenames:
            file_exists(filename, log_level)

        if log_level > 0:
            print("Number of displacements: %d" % num_disps)
            print("Number of supercell files: %d" % len(force_filenames))

        if not check_number_of_force_files(num_disps, force_filenames, disp_filename):
            force_sets = []
        else:
            calc_dataset = get_calc_dataset(
                interface_mode,
                num_atoms,
                force_filenames,
                verbose=(log_level > 0),
            )
            force_sets = calc_dataset["forces"]

        if settings.subtract_forces:
            force_filename = settings.subtract_forces
            file_exists(force_filename, log_level)
            calc_dataset = get_calc_dataset(
                interface_mode,
                num_atoms,
                [
                    force_filename,
                ],
                verbose=(log_level > 0),
            )
            force_set_zero = calc_dataset["forces"][0]
            for fs in force_sets:
                fs -= force_set_zero

            if log_level > 0:
                print(
                    "Forces in '%s' were subtracted from supercell forces."
                    % force_filename
                )

        if force_sets:
            write_FORCES_FC3(disp_dataset, forces_fc3=force_sets, filename="FORCES_FC3")
            if log_level:
                print("")
                print("%s has been created." % "FORCES_FC3")
                print("")
        else:
            if log_level:
                print("")
                print("%s could not be created." % "FORCES_FC3")
                print_error()
            sys.exit(1)

    #####################
    # Create FORCES_FC2 #
    #####################
    if settings.create_forces_fc2:
        disp_filename_candidates = [
            "phono3py_disp.yaml",
        ]
        if cell_filename is not None:
            disp_filename_candidates.insert(0, cell_filename)
        disp_filenames = files_exist(disp_filename_candidates, log_level, is_any=True)
        disp_filename = disp_filenames[0]

        # ph3py_yaml is not None, phono3py_disp.yaml is already read.
        if ph3py_yaml is None:
            ph3py_yaml = Phono3pyYaml()
            ph3py_yaml.read(disp_filename)
            if ph3py_yaml.calculator is not None:
                interface_mode = ph3py_yaml.calculator  # overwrite
        disp_dataset = ph3py_yaml.phonon_dataset

        if log_level:
            print('Displacement dataset was read from "%s".' % disp_filename)
        num_atoms = disp_dataset["natom"]
        num_disps = len(disp_dataset["first_atoms"])
        force_filenames = settings.create_forces_fc2
        for filename in force_filenames:
            file_exists(filename, log_level)

        if log_level > 0:
            print("Number of displacements: %d" % num_disps)
            print("Number of supercell files: %d" % len(force_filenames))

        calc_dataset = get_calc_dataset(
            interface_mode,
            num_atoms,
            force_filenames,
            verbose=(log_level > 0),
        )
        force_sets = calc_dataset["forces"]

        if settings.subtract_forces:
            force_filename = settings.subtract_forces
            file_exists(force_filename, log_level)
            calc_dataset_zero = get_calc_dataset(
                interface_mode,
                num_atoms,
                [
                    force_filename,
                ],
                verbose=(log_level > 0),
            )
            force_set_zero = calc_dataset_zero["forces"][0]
            for fs in force_sets:
                fs -= force_set_zero

            if log_level > 0:
                print(
                    "Forces in '%s' were subtracted from supercell forces."
                    % force_filename
                )

        if force_sets:
            write_FORCES_FC2(disp_dataset, forces_fc2=force_sets, filename="FORCES_FC2")
            if log_level:
                print("")
                print("%s has been created." % "FORCES_FC2")
                print("")
        else:
            if log_level:
                print("")
                print("%s could not be created." % "FORCES_FC2")
                print_error()
            sys.exit(1)


def create_FORCES_FC2_from_FORCE_SETS(log_level):
    """Convert FORCE_SETS to FORCES_FC2."""
    filename = "FORCE_SETS"
    file_exists(filename, log_level)
    disp_dataset = parse_FORCE_SETS(filename=filename)
    write_FORCES_FC2(disp_dataset)

    if log_level:
        print("")
        print("FORCES_FC2 has been created from FORCE_SETS.")
        print("The following yaml lines should replace respective part of")
        print("phono3py_disp.yaml made with --dim-fc2=dim_of_FORCE_SETS.")

        print("")
        print("\n".join(displacements_yaml_lines_type1(disp_dataset)))


def create_FORCE_SETS_from_FORCES_FCx(
    phonon_smat, input_filename: Optional[str], cell_filename: Optional[str], log_level
):
    """Convert FORCES_FC3 or FORCES_FC2 to FORCE_SETS."""
    if cell_filename is not None:
        disp_filename = cell_filename
    elif input_filename is None:
        disp_filename = "phono3py_disp.yaml"
    else:
        disp_filename = f"phono3py_disp.{input_filename}.yaml"
    if phonon_smat is not None:
        forces_filename = "FORCES_FC2"
    else:
        forces_filename = "FORCES_FC3"

    if log_level:
        print(f'Displacement dataset is read from "{disp_filename}".')
        print(f'Forces are read from "{forces_filename}"')

    with open(forces_filename, "r") as f:
        len_first_line = get_length_of_first_line(f)

    if len_first_line == 3:
        file_exists(disp_filename, log_level)
        file_exists(forces_filename, log_level)
        ph3yml = Phono3pyYaml()
        ph3yml.read(disp_filename)
        if phonon_smat is None:
            dataset = copy.deepcopy(ph3yml.dataset)
            smat = ph3yml.supercell_matrix
        else:
            dataset = copy.deepcopy(ph3yml.phonon_dataset)
            smat = ph3yml.phonon_supercell_matrix

        if smat is None or (phonon_smat is not None and (phonon_smat != smat).any()):
            if log_level:
                print("")
                print("Supercell matrix is inconsistent.")
                print(f'Supercell matrix read from "{disp_filename}":')
                print(smat)
                print("Supercell matrix given by --dim-fc2:")
                print(phonon_smat)
                print_error()
            sys.exit(1)

        parse_FORCES_FC2(dataset, filename=forces_filename)
        write_FORCE_SETS(dataset)

        if log_level:
            print("FORCE_SETS has been created.")
    else:
        if log_level:
            print(
                "The file format of %s is already readable by phonopy."
                % forces_filename
            )
