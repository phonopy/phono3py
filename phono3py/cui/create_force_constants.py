"""Force constants calculation utilities for command line user interface."""

# Copyright (C) 2020 Atsushi Togo
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
import dataclasses
import os
import pathlib
from typing import Literal, cast

import numpy as np
from phonopy import Phonopy
from phonopy.cui.load_helper import (
    develop_or_load_pypolymlp as develop_or_load_pypolymlp_phonopy,
)
from phonopy.file_IO import get_dataset_type2, get_io_module_to_decompress
from phonopy.harmonic.displacement import (
    DisplacementDataset,
    Type1DisplacementDataset,
    Type2DisplacementDataset,
)
from phonopy.interface.calculator import get_calculator_physical_units
from phonopy.interface.pypolymlp import PypolymlpParams, parse_mlp_params

from phono3py import Phono3py
from phono3py.file_IO import (
    get_length_of_first_line,
    parse_FORCES_FC2,
    parse_FORCES_FC3,
)
from phono3py.interface.fc_calculator import determine_cutoff_pair_distance
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phono3py.phonon3.dataset import forces_in_dataset
from phono3py.phonon3.displacement_fc3 import (
    Fc3DisplacementDataset,
    Fc3Type1DisplacementDataset,
)


def parse_forces(
    phono3py: Phono3py,
    ph3py_yaml: Phono3pyYaml | None = None,
    cutoff_pair_distance: float | None = None,
    force_filename: str | os.PathLike = "FORCES_FC3",
    phono3py_yaml_filename: str | os.PathLike | None = None,
    fc_type: Literal["fc3", "phonon_fc2"] = "fc3",
    calculator: str | None = None,
    log_level: int = 0,
) -> Fc3DisplacementDataset | DisplacementDataset:
    """Read displacements and forces.

    Physical units of displacements and forces are converted following the
    calculator name. The calculator name may be given as the user input or found
    in phono3py-yaml file. When dumping to phono3py-yaml file, it is assumed
    that displacements and forces are written in the default units (A and eV/A)
    without writing calculator name in it.

    """
    dataset_filename_read_from = None
    dataset: Fc3DisplacementDataset | DisplacementDataset | None = None

    if phono3py.phonon_supercell is None or fc_type == "fc3":
        natom = len(phono3py.supercell)
    else:
        natom = len(phono3py.phonon_supercell)

    # Get dataset from ph3py_yaml. dataset can be None.
    # physical_units can be overwritten if calculator is found in ph3py_yaml.
    if ph3py_yaml:
        dataset = _extract_dataset_from_ph3py_yaml(ph3py_yaml, fc_type)
        if dataset is not None:
            dataset_filename_read_from = phono3py_yaml_filename

    # Forces are not yet found in dataset. Then try to read from FORCES_FC3 or
    # FORCES_FC2.
    if force_filename is not None and pathlib.Path(force_filename).is_file():
        if not forces_in_dataset(dataset):
            # Try reading type2 dataset with forces.
            type2_dataset = _read_type2_dataset_from_FORCES_FC3_or_FC2(
                natom, filename=force_filename, log_level=log_level
            )
            if type2_dataset is not None:
                dataset = type2_dataset
                dataset_filename_read_from = force_filename

        assert dataset is not None

        # Try reading forces with type1 dataset.
        if not forces_in_dataset(dataset):
            if fc_type == "fc3":
                parse_FORCES_FC3(
                    cast(Fc3Type1DisplacementDataset, dataset), force_filename
                )
            else:
                parse_FORCES_FC2(
                    cast(Type1DisplacementDataset, dataset), force_filename
                )
            if log_level:
                print(
                    f'Sets of supercell forces were read from "{force_filename}".',
                    flush=True,
                )

    if dataset is None:
        raise RuntimeError("Dataset is not found.")

    physical_units = get_calculator_physical_units(calculator)

    # Overwrite dataset['cutoff_distance'] when necessary.
    # The distance unit is assumed Angstrom.
    if fc_type == "fc3" and cutoff_pair_distance:
        if "cutoff_distance" not in dataset or (
            "cutoff_distance" in dataset
            and cutoff_pair_distance < dataset["cutoff_distance"]
        ):
            dataset["cutoff_distance"] = cutoff_pair_distance
            if log_level:
                print("Cutoff-pair-distance: %f" % cutoff_pair_distance)

    # Units of displacements and forces are converted. If forces don't
    # exist, the conversion will not be performed for forces.
    if calculator is not None:
        _convert_unit_in_dataset(
            dataset,
            distance_to_A=physical_units.distance_to_A,
            force_to_eVperA=physical_units.force_to_eVperA,
        )

    if "natom" in dataset and dataset["natom"] != natom:
        raise RuntimeError(
            "Number of atoms in supercell is not consistent with "
            f'"{dataset_filename_read_from}".'
        )

    if log_level and dataset_filename_read_from is not None:
        print(
            f"Displacement dataset for {fc_type} was read from "
            f'"{dataset_filename_read_from}".'
        )

    if calculator is not None and log_level:
        print(
            f"Displacements and forces were converted from {calculator} "
            "unit to A and eV/A."
        )

    return dataset


def _read_type2_dataset_from_FORCES_FC3_or_FC2(
    natom: int,
    filename: str | os.PathLike = "FORCES_FC3",
    log_level: int = 0,
) -> Type2DisplacementDataset | None:
    myio = get_io_module_to_decompress(filename)
    with myio.open(filename, "rt") as f:
        len_first_line = get_length_of_first_line(f)
        if len_first_line == 6:  # Type-2
            _dataset = get_dataset_type2(f, natom)
            if log_level:
                n_disp = len(_dataset["displacements"])
                print(f'{n_disp} snapshots were found in "{filename}".')
            return _dataset
    return None


def develop_or_load_pypolymlp(
    ph3py: Phono3py,
    mlp_params: str | dict | PypolymlpParams | None = None,
    mlp_filename: str | os.PathLike | None = None,
    log_level: int = 0,
) -> None:
    """Run pypolymlp to compute forces."""
    develop_or_load_pypolymlp_phonopy(
        cast(Phonopy, ph3py),
        mlp_params=mlp_params,
        mlp_filename=mlp_filename,
        log_level=log_level,
    )


def generate_displacements_and_evaluate_pypolymlp(
    ph3py: Phono3py,
    displacement_distance: float | None = None,
    number_of_snapshots: int | Literal["auto"] | None = None,
    number_estimation_factor: int | None = None,
    random_seed: int | None = None,
    fc_calculator: Literal["traditional", "symfc", "alm"] | str | None = None,
    fc_calculator_options: str | None = None,
    cutoff_pair_distance: float | None = None,
    symfc_memory_size: float | None = None,
    log_level: int = 0,
) -> None:
    """Generate displacements and evaluate forces by pypolymlp."""
    if displacement_distance is None:
        _displacement_distance = 0.01
    else:
        _displacement_distance = displacement_distance

    if log_level:
        if number_of_snapshots:
            print("Generate random displacements")
            print(
                "  Twice of number of snapshots will be generated "
                "for plus-minus displacements."
            )
        else:
            print("Generate displacements")
        print(
            f"  Displacement distance: {_displacement_distance:.5f}".rstrip("0").rstrip(
                "."
            )
        )

    cutoff_pair_distance = determine_cutoff_pair_distance(
        fc_calculator=fc_calculator,
        fc_calculator_options=fc_calculator_options,
        cutoff_pair_distance=cutoff_pair_distance,
        symfc_memory_size=symfc_memory_size,
        random_displacements=number_of_snapshots,
        supercell=ph3py.supercell,
        primitive=ph3py.primitive,
        symmetry=ph3py.symmetry,
        log_level=log_level,
    )
    ph3py.generate_displacements(
        distance=_displacement_distance,
        cutoff_pair_distance=cutoff_pair_distance,
        is_plusminus=True,
        number_of_snapshots=number_of_snapshots,
        random_seed=random_seed,
        number_estimation_factor=number_estimation_factor,
    )

    if log_level:
        print(
            f"Evaluate forces in {ph3py.displacements.shape[0]} supercells "
            "by pypolymlp",
            flush=True,
        )

    if ph3py.supercells_with_displacements is None:
        raise RuntimeError("Displacements are not set. Run generate_displacements.")

    ph3py.evaluate_mlp()


def run_pypolymlp_to_compute_phonon_forces(
    ph3py: Phono3py,
    mlp_params: str | dict | PypolymlpParams | None = None,
    displacement_distance: float | None = None,
    number_of_snapshots: int | None = None,
    random_seed: int | None = None,
    log_level: int = 0,
) -> None:
    """Run pypolymlp to compute phonon forces."""
    if ph3py.phonon_mlp_dataset is not None:
        if log_level:
            import pypolymlp

            print("-" * 29 + " pypolymlp start " + "-" * 30)
            print(f"Pypolymlp version {pypolymlp.__version__}")
            print("Pypolymlp is a generator of polynomial machine learning potentials.")
            print("Please cite the paper: A. Seko, J. Appl. Phys. 133, 011101 (2023).")
            print("Pypolymlp is developed at https://github.com/sekocha/pypolymlp.")
            if mlp_params:
                print("Parameters:")
                for k, v in dataclasses.asdict(parse_mlp_params(mlp_params)).items():
                    if v is not None:
                        print(f"  {k}: {v}")
        if log_level:
            print("Developing MLPs by pypolymlp...", flush=True)

        ph3py.develop_phonon_mlp(params=mlp_params)

        if log_level:
            print("-" * 30 + " pypolymlp end " + "-" * 31, flush=True)

    if displacement_distance is None:
        _displacement_distance = 0.01
    else:
        _displacement_distance = displacement_distance
    if log_level:
        print("Generate random displacements for fc2")
        print(
            f"  Displacement distance: {_displacement_distance:.5f}".rstrip("0").rstrip(
                "."
            )
        )
    ph3py.generate_fc2_displacements(
        distance=_displacement_distance,
        is_plusminus=True,
        number_of_snapshots=number_of_snapshots,
        random_seed=random_seed,
    )
    if log_level:
        print(
            f"Evaluate forces in {ph3py.phonon_displacements.shape[0]} "
            "supercells by pypolymlp",
            flush=True,
        )
    ph3py.evaluate_phonon_mlp()


def _convert_unit_in_dataset(
    dataset: Fc3DisplacementDataset | DisplacementDataset,
    distance_to_A: float | None = None,
    force_to_eVperA: float | None = None,
    energy_to_eV: float | None = None,
) -> None:
    """Convert physical units of displacements and forces in dataset.

    dataset is overwritten.

    cutoff_distance in dataset is also converted.

    """
    if "first_atoms" in dataset:
        for d1 in dataset["first_atoms"]:
            if distance_to_A is not None:
                d1["displacement"] = (
                    np.asarray(d1["displacement"], dtype="double", order="C")
                    * distance_to_A
                )
            if force_to_eVperA is not None and "forces" in d1:
                d1["forces"] = (
                    np.asarray(d1["forces"], dtype="double", order="C")
                    * force_to_eVperA
                )
            if energy_to_eV is not None and "supercell_energy" in d1:
                d1["supercell_energy"] = d1["supercell_energy"] * energy_to_eV
            if "second_atoms" in d1:
                for d2 in d1["second_atoms"]:
                    if distance_to_A is not None:
                        d2["displacement"] = (
                            np.asarray(d2["displacement"], dtype="double", order="C")
                            * distance_to_A
                        )
                    if force_to_eVperA is not None and "forces" in d2:
                        d2["forces"] = (
                            np.asarray(d2["forces"], dtype="double", order="C")
                            * force_to_eVperA
                        )
                    if energy_to_eV is not None and "supercell_energy" in d2:
                        d2["supercell_energy"] = d2["supercell_energy"] * energy_to_eV
    else:
        if distance_to_A is not None and "displacements" in dataset:
            dataset["displacements"] = (
                np.asarray(dataset["displacements"], dtype="double", order="C")
                * distance_to_A
            )
        if force_to_eVperA is not None and "forces" in dataset:
            dataset["forces"] = (
                np.asarray(dataset["forces"], dtype="double", order="C")
                * force_to_eVperA
            )
        if energy_to_eV is not None and "supercell_energies" in dataset:
            dataset["supercell_energies"] = (
                np.asarray(dataset["supercell_energies"], dtype="double", order="C")
                * energy_to_eV
            )

    if distance_to_A is not None and "cutoff_distance" in dataset:
        dataset["cutoff_distance"] = dataset["cutoff_distance"] * distance_to_A


def _extract_dataset_from_ph3py_yaml(
    ph3py_yaml: Phono3pyYaml, fc_type: Literal["fc3", "phonon_fc2"]
) -> Fc3DisplacementDataset | DisplacementDataset | None:
    if ph3py_yaml.phonon_supercell is None or fc_type == "fc3":
        if ph3py_yaml.dataset is not None:
            return copy.deepcopy(ph3py_yaml.dataset)  # type: ignore[return-value]
    else:
        if ph3py_yaml.phonon_dataset is not None:
            return copy.deepcopy(ph3py_yaml.phonon_dataset)
    return None
