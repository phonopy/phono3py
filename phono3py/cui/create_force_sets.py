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
from typing import Optional

import numpy as np
from phonopy.cui.create_force_sets import check_number_of_force_files
from phonopy.cui.load_helper import get_nac_params
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
    log_level: int = 0,
):
    """Create FORCES_FC3 and FORCES_FC2 or phono3py_params.yaml from files.

    With settings.save_params=True, phono3py_params.yaml is created instead of
    FORCES_FC3 and FORCES_FC2. To create phono3py_params.yaml, at least fc3
    forces have to be collected.

    """
    interface_mode = settings.calculator
    disp_filename_candidates = [
        "phono3py_disp.yaml",
    ]
    if cell_filename is not None:
        disp_filename_candidates.insert(0, cell_filename)
    disp_filenames = files_exist(disp_filename_candidates, log_level, is_any=True)
    disp_filename = disp_filenames[0]
    ph3py_yaml = Phono3pyYaml(settings={"force_sets": True})
    ph3py_yaml.read(disp_filename)
    if ph3py_yaml.calculator is not None:
        interface_mode = ph3py_yaml.calculator  # overwrite

    if settings.create_forces_fc3 or settings.create_forces_fc3_file:
        calc_dataset_fc3 = _get_force_sets_fc3(
            settings, ph3py_yaml.dataset, disp_filename, interface_mode, log_level
        )
        if not calc_dataset_fc3["forces"]:
            if log_level:
                print("%s could not be created." % "FORCES_FC3")
                print_error()
            sys.exit(1)

    if settings.create_forces_fc2:
        calc_dataset_fc2 = _get_force_sets_fc2(
            settings,
            ph3py_yaml.phonon_dataset,
            disp_filename,
            interface_mode,
            log_level,
        )
        if not calc_dataset_fc2["forces"]:
            if log_level:
                print("%s could not be created." % "FORCES_FC2")
                print_error()
            sys.exit(1)

    if settings.save_params:
        fc3_yaml_filename = "phono3py_params.yaml"
        if not (settings.create_forces_fc3 or settings.create_forces_fc3_file):
            if log_level:
                print(f'When creating "{fc3_yaml_filename}" with force sets for fc2, ')
                print("force sets for fc3 have to be collected simultaneously.")
                print(f'"{fc3_yaml_filename}" could not be created.')
                print("")
                print_error()
            sys.exit(1)

        _set_forces_and_nac_params(
            ph3py_yaml, settings, calc_dataset_fc3, calc_dataset_fc2
        )

        with open(fc3_yaml_filename, "w") as w:
            w.write(str(ph3py_yaml))
            if log_level:
                print(f'"{fc3_yaml_filename}" has been created.')
    else:
        if settings.create_forces_fc3 or settings.create_forces_fc3_file:
            write_FORCES_FC3(
                ph3py_yaml.dataset,
                forces_fc3=calc_dataset_fc3["forces"],
                filename="FORCES_FC3",
            )
            if log_level:
                print("%s has been created." % "FORCES_FC3")

        if settings.create_forces_fc2:
            write_FORCES_FC2(
                ph3py_yaml.phonon_dataset,
                forces_fc2=calc_dataset_fc2["forces"],
                filename="FORCES_FC2",
            )
            if log_level:
                print("%s has been created." % "FORCES_FC2")


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


def _get_force_sets_fc2(
    settings, disp_dataset, disp_filename, interface_mode, log_level
) -> dict:
    interface_mode = settings.calculator
    if log_level:
        print(f'FC2 displacement dataset was read from "{disp_filename}".')

    if "first_atoms" in disp_dataset:  # type-1
        num_atoms = disp_dataset["natom"]
        num_disps = len(disp_dataset["first_atoms"])
    elif "displacements" in disp_dataset:  # type-2:
        num_disps = len(disp_dataset["displacements"])
        num_atoms = len(disp_dataset["displacements"][0])
    else:
        raise RuntimeError("FC2 displacement dataset is broken.")

    force_filenames = settings.create_forces_fc2
    for filename in force_filenames:
        file_exists(filename, log_level)

    if log_level > 0:
        print(f"  Number of displacements: {num_disps}")
        print(f"  Number of supercell files: {len(force_filenames)}")

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
            print("Forces in '{force_filename}' were subtracted from supercell forces.")

    if log_level > 0:
        print("")

    return calc_dataset


def _get_force_sets_fc3(
    settings, disp_dataset, disp_filename, interface_mode, log_level
) -> dict:
    if log_level:
        print("")
        print(f'FC3 Displacement dataset was read from "{disp_filename}".')

    if "first_atoms" in disp_dataset:  # type-1
        num_atoms = disp_dataset["natom"]
        num_disps = len(disp_dataset["first_atoms"])
        for d1 in disp_dataset["first_atoms"]:
            for d2 in d1["second_atoms"]:
                if "included" not in d2 or d2["included"]:
                    num_disps += 1
    elif "displacements" in disp_dataset:  # type-2:
        num_disps = len(disp_dataset["displacements"])
        num_atoms = len(disp_dataset["displacements"][0])
    else:
        raise RuntimeError("FC3 displacement dataset is broken.")

    if settings.create_forces_fc3_file:
        file_exists(settings.create_forces_fc3_file, log_level)
        force_filenames = [x.strip() for x in open(settings.create_forces_fc3_file)]
    else:
        force_filenames = settings.create_forces_fc3

    for filename in force_filenames:
        file_exists(filename, log_level)

    if log_level > 0:
        print(f"  Number of displacements: {num_disps}")
        print(f"  Number of supercell files: {len(force_filenames)}")

    if not check_number_of_force_files(num_disps, force_filenames, disp_filename):
        calc_dataset = {"forces": []}
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
                f"Forces in '{force_filename}' were subtracted from supercell forces."
            )

    if log_level > 0:
        print("")

    return calc_dataset


def _set_forces_and_nac_params(
    ph3py_yaml: Phono3pyYaml, settings, calc_dataset_fc3: dict, calc_dataset_fc2: dict
):
    if "first_atoms" in ph3py_yaml.dataset:
        count = len(ph3py_yaml.dataset["first_atoms"])
        for i, d1 in enumerate(ph3py_yaml.dataset["first_atoms"]):
            d1["forces"] = calc_dataset_fc3["forces"][i]
            if "supercell_energies" in calc_dataset_fc3:
                d1["supercell_energy"] = float(
                    calc_dataset_fc3["supercell_energies"][i]
                )
            for d2 in d1["second_atoms"]:
                if "included" not in d2 or d2["included"]:
                    d2["forces"] = calc_dataset_fc3["forces"][count]
                    if "supercell_energies" in calc_dataset_fc3:
                        d2["supercell_energy"] = float(
                            calc_dataset_fc3["supercell_energies"][count]
                        )
                    count += 1
    else:
        ph3py_yaml.dataset["forces"] = np.array(
            calc_dataset_fc3["forces"], dtype="double", order="C"
        )
        ph3py_yaml.dataset["supercell_energies"] = np.array(
            calc_dataset_fc3["supercell_energies"], dtype="double"
        )

    if settings.create_forces_fc2:
        if "first_atoms" in ph3py_yaml.phonon_dataset:
            for i, d in enumerate(ph3py_yaml.phonon_dataset["first_atoms"]):
                d["forces"] = calc_dataset_fc2["forces"][i]
                if "supercell_energies" in calc_dataset_fc2:
                    d["supercell_energy"] = float(
                        calc_dataset_fc2["supercell_energies"][i]
                    )
        else:
            ph3py_yaml.phonon_dataset["forces"] = np.array(
                calc_dataset_fc2["forces"], dtype="double", order="C"
            )
            ph3py_yaml.phonon_dataset["supercell_energies"] = np.array(
                calc_dataset_fc2["supercell_energies"], dtype="double"
            )

    nac_params = get_nac_params(primitive=ph3py_yaml.primitive)
    if nac_params:
        ph3py_yaml.nac_params = nac_params
