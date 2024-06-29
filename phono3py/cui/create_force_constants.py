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
import os
import pathlib
import sys
from dataclasses import asdict
from typing import Optional, Union

import numpy as np
from phonopy.cui.phonopy_script import file_exists, print_error
from phonopy.file_IO import get_dataset_type2
from phonopy.harmonic.force_constants import (
    show_drift_force_constants,
    symmetrize_compact_force_constants,
    symmetrize_force_constants,
)
from phonopy.interface.calculator import get_default_physical_units
from phonopy.interface.fc_calculator import fc_calculator_names
from phonopy.interface.pypolymlp import PypolymlpParams, parse_mlp_params

from phono3py import Phono3py
from phono3py.cui.show_log import show_phono3py_force_constants_settings
from phono3py.file_IO import (
    get_length_of_first_line,
    parse_FORCES_FC2,
    parse_FORCES_FC3,
    read_fc2_from_hdf5,
    read_fc3_from_hdf5,
    write_fc2_to_hdf5,
    write_fc3_to_hdf5,
)
from phono3py.interface.fc_calculator import extract_fc2_fc3_calculators
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phono3py.phonon3.fc3 import (
    set_permutation_symmetry_fc3,
    set_translational_invariance_fc3,
    show_drift_fc3,
)


def create_phono3py_force_constants(
    phono3py: Phono3py,
    settings,
    ph3py_yaml: Optional[Phono3pyYaml] = None,
    phono3py_yaml_filename: Optional[str] = None,
    input_filename: Optional[str] = None,
    output_filename: Optional[str] = None,
    log_level=1,
):
    """Read or calculate force constants.

    This function is for the 'phonopy' command only and not for the
    'phonopy-load' command.

    """
    # Only for build-in fc calculator.
    # These are not applied to external fc calculators.
    symmetrize_fc3r = settings.is_symmetrize_fc3_r or settings.fc_symmetry
    symmetrize_fc2 = settings.is_symmetrize_fc2 or settings.fc_symmetry

    (fc_calculator, fc_calculator_options) = get_fc_calculator_params(settings)

    if log_level:
        show_phono3py_force_constants_settings(settings)

    #######
    # fc3 #
    #######
    if (
        settings.is_joint_dos
        or (settings.is_isotope and not (settings.is_bterta or settings.is_lbte))
        or settings.read_gamma
        or settings.read_pp
        or (not settings.is_bterta and settings.write_phonon)
        or settings.constant_averaged_pp_interaction is not None
    ):
        pass
    else:
        if settings.read_fc3:
            _read_phono3py_fc3(phono3py, symmetrize_fc3r, input_filename, log_level)
        else:  # fc3 from FORCES_FC3 or ph3py_yaml
            _create_phono3py_fc3(
                phono3py,
                ph3py_yaml,
                phono3py_yaml_filename,
                symmetrize_fc3r,
                settings.is_compact_fc,
                settings.cutoff_pair_distance,
                fc_calculator,
                fc_calculator_options,
                settings.use_pypolymlp,
                settings.mlp_params,
                settings.displacement_distance,
                settings.random_displacements,
                settings.random_seed,
                log_level,
            )

        cutoff_distance = settings.cutoff_fc3_distance
        if cutoff_distance is not None and cutoff_distance > 0:
            if log_level:
                print(
                    "Cutting-off fc3 by zero (cut-off distance: %f)" % cutoff_distance
                )
            phono3py.cutoff_fc3_by_zero(cutoff_distance)

        if not settings.read_fc3:
            if output_filename is None:
                filename = "fc3.hdf5"
            else:
                filename = "fc3." + output_filename + ".hdf5"
            if log_level:
                print('Writing fc3 to "%s".' % filename)
            write_fc3_to_hdf5(
                phono3py.fc3,
                filename=filename,
                p2s_map=phono3py.primitive.p2s_map,
                compression=settings.hdf5_compression,
            )

        if log_level:
            show_drift_fc3(phono3py.fc3, primitive=phono3py.primitive)

    #######
    # fc2 #
    #######
    phonon_primitive = phono3py.phonon_primitive
    p2s_map = phonon_primitive.p2s_map
    if settings.read_fc2:
        _read_phono3py_fc2(phono3py, symmetrize_fc2, input_filename, log_level)
    else:
        if phono3py.phonon_supercell_matrix is None:
            force_filename = "FORCES_FC3"
        else:
            force_filename = "FORCES_FC2"
        _create_phono3py_fc2(
            phono3py,
            ph3py_yaml,
            force_filename,
            symmetrize_fc2,
            settings.is_compact_fc,
            fc_calculator,
            fc_calculator_options,
            log_level,
        )
        if output_filename is None:
            filename = "fc2.hdf5"
        else:
            filename = "fc2." + output_filename + ".hdf5"
        if log_level:
            print('Writing fc2 to "%s".' % filename)
        write_fc2_to_hdf5(
            phono3py.fc2,
            filename=filename,
            p2s_map=p2s_map,
            physical_unit="eV/angstrom^2",
            compression=settings.hdf5_compression,
        )

    if log_level:
        show_drift_force_constants(phono3py.fc2, primitive=phonon_primitive, name="fc2")


def parse_forces(
    phono3py: Phono3py,
    ph3py_yaml: Optional[Phono3pyYaml] = None,
    cutoff_pair_distance=None,
    force_filename: str = "FORCES_FC3",
    phono3py_yaml_filename: Optional[str] = None,
    fc_type=None,
    log_level=0,
):
    """Read displacements and forces.

    Physical units of displacements and forces are converted following the
    calculator name. The calculator name may be given as the user input or found
    in phono3py-yaml file. When dumping to phono3py-yaml file, it is assumed
    that displacements and forces are written in the default units (A and eV/A)
    without writing calculator name in it.

    """
    filename_read_from: Optional[str] = None

    calculator = phono3py.calculator
    # Get dataset from ph3py_yaml. dataset can be None.
    # physical_units can be overwritten if calculator is found in ph3py_yaml.
    dataset = _extract_dataset_from_ph3py_yaml(ph3py_yaml, fc_type)
    if dataset and ph3py_yaml.calculator:
        calculator = ph3py_yaml.calculator

    physical_units = get_default_physical_units(calculator)

    if fc_type == "phonon_fc2":
        natom = len(phono3py.phonon_supercell)
    else:
        natom = len(phono3py.supercell)

    if dataset:
        filename_read_from = phono3py_yaml_filename

        # Units of displacements and forces are converted. If forces don't
        # exist, the convesion will not be performed for forces.
        if calculator is not None:
            _convert_unit_in_dataset(
                dataset,
                distance_to_A=physical_units["distance_to_A"],
                force_to_eVperA=physical_units["force_to_eVperA"],
            )

    # Try to read FORCES_FC* if type-2 and return dataset.
    # None is returned unless type-2.
    # can emit FileNotFoundError.
    if dataset is None or (dataset is not None and not forces_in_dataset(dataset)):
        _dataset = read_type2_dataset(
            natom, filename=force_filename, log_level=log_level
        )
        # Do not overwrite dataset when _dataset is None.
        if _dataset:
            filename_read_from = force_filename
            dataset = _dataset

            # Units of displacements and forces are converted.
            if calculator is not None:
                _convert_unit_in_dataset(
                    dataset,
                    distance_to_A=physical_units["distance_to_A"],
                    force_to_eVperA=physical_units["force_to_eVperA"],
                )

    assert dataset is not None

    if "natom" in dataset and dataset["natom"] != natom:
        msg = (
            "Number of atoms in supercell is not consistent with "
            '"%s".' % filename_read_from
        )
        raise RuntimeError(msg)

    if log_level and filename_read_from is not None:
        print(
            'Displacement dataset for %s was read from "%s".'
            % (fc_type, filename_read_from)
        )

    # Overwrite dataset['cutoff_distance'] when necessary.
    if cutoff_pair_distance:
        if "cutoff_distance" not in dataset or (
            "cutoff_distance" in dataset
            and cutoff_pair_distance < dataset["cutoff_distance"]
        ):
            dataset["cutoff_distance"] = cutoff_pair_distance
            if log_level:
                print("Cutoff-pair-distance: %f" % cutoff_pair_distance)

    # Type-1 FORCES_FC*.
    # dataset comes either from disp_fc*.yaml or phono3py*.yaml.
    if not forces_in_dataset(dataset):
        if fc_type == "phonon_fc2":
            parse_FORCES_FC2(dataset, filename=force_filename)
        else:
            parse_FORCES_FC3(dataset, filename=force_filename)

        # Unit of displacements is already converted.
        # Therefore, only unit of forces is converted.
        if calculator is not None:
            _convert_unit_in_dataset(
                dataset,
                force_to_eVperA=physical_units["force_to_eVperA"],
            )

        if log_level:
            print('Sets of supercell forces were read from "%s".' % force_filename)
            sys.stdout.flush()

    return dataset


def forces_in_dataset(dataset: dict) -> bool:
    """Return whether forces in dataset or not."""
    return "forces" in dataset or (
        "first_atoms" in dataset and "forces" in dataset["first_atoms"][0]
    )


def displacements_in_dataset(dataset: Optional[dict]) -> bool:
    """Return whether displacements in dataset or not."""
    if dataset is None:
        return False
    return "displacements" in dataset or "first_atoms" in dataset


def get_fc_calculator_params(settings):
    """Return fc_calculator and fc_calculator_params from settings."""
    fc_calculator = None
    fc_calculator_list = []
    if settings.fc_calculator is not None:
        for fc_calculatr_str in settings.fc_calculator.split("|"):
            if fc_calculatr_str == "":  # No external calculator
                fc_calculator_list.append(fc_calculatr_str.lower())
            elif fc_calculatr_str.lower() in fc_calculator_names:
                fc_calculator_list.append(fc_calculatr_str.lower())
        if fc_calculator_list:
            fc_calculator = "|".join(fc_calculator_list)

    fc_calculator_options = None
    if settings.fc_calculator_options is not None:
        fc_calculator_options = settings.fc_calculator_options

    return fc_calculator, fc_calculator_options


def _read_phono3py_fc3(phono3py: Phono3py, symmetrize_fc3r, input_filename, log_level):
    if input_filename is None:
        filename = "fc3.hdf5"
    else:
        filename = "fc3." + input_filename + ".hdf5"
    file_exists(filename, log_level)
    if log_level:
        print('Reading fc3 from "%s".' % filename)

    p2s_map = phono3py.primitive.p2s_map
    try:
        fc3 = read_fc3_from_hdf5(filename=filename, p2s_map=p2s_map)
    except RuntimeError:
        import traceback

        traceback.print_exc()
        if log_level:
            print_error()
        sys.exit(1)
    num_atom = len(phono3py.supercell)
    if fc3.shape[1] != num_atom:
        print("Matrix shape of fc3 doesn't agree with supercell size.")
        if log_level:
            print_error()
        sys.exit(1)

    if symmetrize_fc3r:
        set_translational_invariance_fc3(fc3)
        set_permutation_symmetry_fc3(fc3)

    phono3py.fc3 = fc3


def _read_phono3py_fc2(phono3py, symmetrize_fc2, input_filename, log_level):
    if input_filename is None:
        filename = "fc2.hdf5"
    else:
        filename = "fc2." + input_filename + ".hdf5"
    file_exists(filename, log_level)
    if log_level:
        print('Reading fc2 from "%s".' % filename)

    num_atom = len(phono3py.phonon_supercell)
    p2s_map = phono3py.phonon_primitive.p2s_map
    try:
        phonon_fc2 = read_fc2_from_hdf5(filename=filename, p2s_map=p2s_map)
    except RuntimeError:
        import traceback

        traceback.print_exc()
        if log_level:
            print_error()
        sys.exit(1)

    if phonon_fc2.shape[1] != num_atom:
        print("Matrix shape of fc2 doesn't agree with supercell size.")
        if log_level:
            print_error()
        sys.exit(1)

    if symmetrize_fc2:
        if phonon_fc2.shape[0] == phonon_fc2.shape[1]:
            symmetrize_force_constants(phonon_fc2)
        else:
            symmetrize_compact_force_constants(phonon_fc2, phono3py.phonon_primitive)

    phono3py.fc2 = phonon_fc2


def read_type2_dataset(natom, filename="FORCES_FC3", log_level=0) -> Optional[dict]:
    """Read type-2 FORCES_FC3."""
    if not pathlib.Path(filename).exists():
        return None

    with open(filename, "r") as f:
        len_first_line = get_length_of_first_line(f)
        if len_first_line == 6:
            dataset = get_dataset_type2(f, natom)
            if log_level:
                print(
                    "%d snapshots were found in %s."
                    % (len(dataset["displacements"]), "FORCES_FC3")
                )
        else:
            dataset = None
    return dataset


def _create_phono3py_fc3(
    phono3py: Phono3py,
    ph3py_yaml: Optional[Phono3pyYaml],
    phono3py_yaml_filename: Optional[str],
    symmetrize_fc3r: bool,
    is_compact_fc: bool,
    cutoff_pair_distance: Optional[float],
    fc_calculator: Optional[str],
    fc_calculator_options: Optional[str],
    use_pypolymlp: bool,
    mlp_params: Union[str, dict, PypolymlpParams],
    displacement_distance: Optional[float],
    number_of_snapshots: Optional[int],
    random_seed: Optional[int],
    log_level: int,
):
    """Read or calculate fc3.

    Note
    ----
    cutoff_pair_distance is the parameter to determine each displaced
    supercell is included to the computation of fc3. It is assumed that
    cutoff_pair_distance is stored in the step to create sets of
    displacements and the value is stored n the displacement dataset and
    also as the parameter 'included': True or False for each displacement.
    The parameter cutoff_pair_distance here can be used in the step to
    create fc3 by overwriting original cutoff_pair_distance value only
    when the former value is smaller than the later.

    """
    _ph3py_yaml = _get_default_ph3py_yaml(ph3py_yaml)

    try:
        dataset = parse_forces(
            phono3py,
            ph3py_yaml=_ph3py_yaml,
            cutoff_pair_distance=cutoff_pair_distance,
            force_filename="FORCES_FC3",
            phono3py_yaml_filename=phono3py_yaml_filename,
            fc_type="fc3",
            log_level=log_level,
        )
    except RuntimeError as e:
        # from _parse_forces_type1
        if log_level:
            print(str(e))
            print_error()
        sys.exit(1)
    except FileNotFoundError as e:
        # from _get_type2_dataset
        file_exists(e.filename, log_level)

    if use_pypolymlp:
        phono3py.mlp_dataset = dataset
        run_pypolymlp_to_compute_forces(
            phono3py,
            mlp_params,
            displacement_distance=displacement_distance,
            number_of_snapshots=number_of_snapshots,
            random_seed=random_seed,
            log_level=log_level,
        )
    else:
        phono3py.dataset = dataset
    phono3py.produce_fc3(
        symmetrize_fc3r=symmetrize_fc3r,
        is_compact_fc=is_compact_fc,
        fc_calculator=extract_fc2_fc3_calculators(fc_calculator, 3),
        fc_calculator_options=extract_fc2_fc3_calculators(fc_calculator_options, 3),
    )


def run_pypolymlp_to_compute_forces(
    ph3py: Phono3py,
    mlp_params: Union[str, dict, PypolymlpParams],
    displacement_distance: Optional[float] = None,
    number_of_snapshots: Optional[int] = None,
    random_seed: Optional[int] = None,
    log_level: int = 0,
):
    """Run pypolymlp to compute forces."""
    if log_level:
        print("-" * 29 + " pypolymlp start " + "-" * 30)
        print("Pypolymlp is a generator of polynomial machine learning potentials.")
        print("Please cite the paper: A. Seko, J. Appl. Phys. 133, 011101 (2023).")
        print("Pypolymlp is developed at https://github.com/sekocha/pypolymlp.")
        if mlp_params:
            print("Parameters:")
            for k, v in asdict(parse_mlp_params(mlp_params)).items():
                if v is not None:
                    print(f"  {k}: {v}")
    if log_level > 1:
        print("")
    if log_level:
        print("Developing MLPs by pypolymlp...", flush=True)

    ph3py.develop_mlp(params=mlp_params)

    if log_level:
        print("-" * 30 + " pypolymlp end " + "-" * 31, flush=True)

    if displacement_distance is None:
        _displacement_distance = 0.001
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
    ph3py.generate_displacements(
        distance=_displacement_distance,
        is_plusminus=True,
        number_of_snapshots=number_of_snapshots,
        random_seed=random_seed,
    )

    if log_level:
        print(
            f"Evaluate forces in {ph3py.displacements.shape[0]} supercells "
            "by pypolymlp",
            flush=True,
        )

    if ph3py.mlp_dataset is None:
        msg = "mlp_dataset has to be set before calling this method."
        raise RuntimeError(msg)
    if ph3py.supercells_with_displacements is None:
        raise RuntimeError("Displacements are not set. Run generate_displacements.")

    ph3py.evaluate_mlp()


def run_pypolymlp_to_compute_phonon_forces(
    ph3py: Phono3py,
    mlp_params: Optional[Union[str, dict, PypolymlpParams]] = None,
    displacement_distance: Optional[float] = None,
    number_of_snapshots: Optional[int] = None,
    random_seed: Optional[int] = None,
    log_level: int = 0,
):
    """Run pypolymlp to compute phonon forces."""
    if ph3py.phonon_mlp_dataset is not None:
        if log_level:
            print("-" * 29 + " pypolymlp start " + "-" * 30)
            print("Pypolymlp is a generator of polynomial machine learning potentials.")
            print("Please cite the paper: A. Seko, J. Appl. Phys. 133, 011101 (2023).")
            print("Pypolymlp is developed at https://github.com/sekocha/pypolymlp.")
            if mlp_params:
                print("Parameters:")
                for k, v in asdict(parse_mlp_params(mlp_params)).items():
                    if v is not None:
                        print(f"  {k}: {v}")
        if log_level > 1:
            print("")
        if log_level:
            print("Developing MLPs by pypolymlp...", flush=True)

        ph3py.develop_phonon_mlp(params=mlp_params)

        if log_level:
            print("-" * 30 + " pypolymlp end " + "-" * 31, flush=True)

    if displacement_distance is None:
        _displacement_distance = 0.001
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


def _create_phono3py_fc2(
    phono3py: Phono3py,
    ph3py_yaml: Optional[Phono3pyYaml],
    force_filename,
    symmetrize_fc2,
    is_compact_fc,
    fc_calculator,
    fc_calculator_options,
    log_level,
):
    """Read forces and produce fc2.

    force_filename is either "FORCES_FC2" or "FORCES_FC3".

    """
    _ph3py_yaml = _get_default_ph3py_yaml(ph3py_yaml)

    try:
        dataset = parse_forces(
            phono3py,
            ph3py_yaml=_ph3py_yaml,
            force_filename=force_filename,
            fc_type="fc2",
            log_level=log_level,
        )
    except RuntimeError as e:
        if log_level:
            print(str(e))
            print_error()
        sys.exit(1)
    except FileNotFoundError as e:
        file_exists(e.filename, log_level)

    phono3py.phonon_dataset = dataset
    phono3py.produce_fc2(
        symmetrize_fc2=symmetrize_fc2,
        is_compact_fc=is_compact_fc,
        fc_calculator=extract_fc2_fc3_calculators(fc_calculator, 2),
        fc_calculator_options=extract_fc2_fc3_calculators(fc_calculator_options, 2),
    )


def _get_default_ph3py_yaml(ph3py_yaml: Optional[Phono3pyYaml]):
    _ph3py_yaml = ph3py_yaml
    if _ph3py_yaml is None and os.path.isfile("phono3py_disp.yaml"):
        _ph3py_yaml = Phono3pyYaml()
        _ph3py_yaml.read("phono3py_disp.yaml")
    return _ph3py_yaml


def _convert_unit_in_dataset(
    dataset: dict,
    distance_to_A: Optional[float] = None,
    force_to_eVperA: Optional[float] = None,
) -> None:
    """Convert physical units of displacements and forces in dataset.

    dataset is overwritten.

    """
    if "first_atoms" in dataset:
        for d1 in dataset["first_atoms"]:
            if distance_to_A is not None:
                disp = _to_ndarray(d1["displacement"])
                d1["displacement"] = disp * distance_to_A
            if force_to_eVperA is not None and "forces" in d1:
                forces = _to_ndarray(d1["forces"])
                d1["forces"] = forces * force_to_eVperA
            if "second_atoms" in d1:
                for d2 in d1["second_atoms"]:
                    if distance_to_A is not None:
                        disp = _to_ndarray(d2["displacement"])
                        d2["displacement"] = disp * distance_to_A
                    if force_to_eVperA is not None and "forces" in d2:
                        forces = _to_ndarray(d2["forces"])
                        d2["forces"] = forces * force_to_eVperA
    else:
        if distance_to_A is not None and "displacements" in dataset:
            disp = _to_ndarray(dataset["displacements"])
            dataset["displacements"] = disp * distance_to_A
        if force_to_eVperA is not None and "forces" in dataset:
            forces = _to_ndarray(dataset["forces"])
            dataset["forces"] = forces * force_to_eVperA


def _to_ndarray(array, dtype="double"):
    if type(array) is not np.ndarray:
        return np.array(array, dtype=dtype, order="C")
    else:
        return array


def _extract_dataset_from_ph3py_yaml(ph3py_yaml: Optional[Phono3pyYaml], fc_type):
    dataset = None
    if fc_type == "phonon_fc2":
        if ph3py_yaml and ph3py_yaml.phonon_dataset is not None:
            dataset = copy.deepcopy(ph3py_yaml.phonon_dataset)
    else:
        if ph3py_yaml and ph3py_yaml.dataset is not None:
            dataset = copy.deepcopy(ph3py_yaml.dataset)
    return dataset
