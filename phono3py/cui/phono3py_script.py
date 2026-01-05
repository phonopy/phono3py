"""Phono3py main command line script."""

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

import argparse
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from phonopy.api_phonopy import Phonopy
from phonopy.cui.phonopy_argparse import show_deprecated_option_warnings
from phonopy.cui.phonopy_script import (
    file_exists,
    print_end,
    print_error,
    print_error_message,
    print_time,
    print_version,
    store_nac_params,
)
from phonopy.cui.settings import PhonopySettings
from phonopy.exception import (
    CellNotFoundError,
    ForceCalculatorRequiredError,
    PypolymlpDevelopmentError,
    PypolymlpFileNotFoundError,
    PypolymlpRelaxationError,
)
from phonopy.file_IO import is_file_phonopy_yaml
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixGL
from phonopy.harmonic.force_constants import show_drift_force_constants
from phonopy.interface.calculator import get_calculator_physical_units
from phonopy.interface.pypolymlp import get_change_in_positions, relax_atomic_positions
from phonopy.interface.symfc import estimate_symfc_cutoff_from_memsize
from phonopy.phonon.band_structure import get_band_qpoints
from phonopy.physical_units import get_physical_units
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import isclose as cells_isclose

from phono3py import Phono3py, Phono3pyIsotope, Phono3pyJointDos
from phono3py.cui.create_force_constants import (
    develop_or_load_pypolymlp,
    generate_displacements_and_evaluate_pypolymlp,
)
from phono3py.cui.create_force_sets import (
    create_FORCE_SETS_from_FORCES_FCx,
    create_FORCES_FC2_from_FORCE_SETS,
    create_FORCES_FC3_and_FORCES_FC2,
)
from phono3py.cui.create_supercells import (
    Phono3pyCellInfoResult,
    create_phono3py_supercells,
    get_cell_info,
)
from phono3py.cui.load import (
    compute_force_constants_from_datasets,
    load_fc2_and_fc3,
    select_and_load_dataset,
    select_and_load_phonon_dataset,
)
from phono3py.cui.phono3py_argparse import get_parser
from phono3py.cui.settings import Phono3pyConfParser, Phono3pySettings
from phono3py.cui.show_log import (
    show_general_settings,
    show_phono3py_cells,
    show_phono3py_settings,
)
from phono3py.cui.triplets_info import show_num_triplets, write_grid_points
from phono3py.file_IO import (
    read_phonon_from_hdf5,
    write_fc2_to_hdf5,
    write_fc3_to_hdf5,
    write_phonon_to_hdf5,
)
from phono3py.interface.fc_calculator import (
    determine_cutoff_pair_distance,
    get_fc_calculator_params,
)
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phono3py.phonon.grid import BZGrid, get_grid_point_from_address, get_ir_grid_points
from phono3py.phonon3.dataset import forces_in_dataset
from phono3py.phonon3.fc3 import show_drift_fc3
from phono3py.phonon3.gruneisen import run_gruneisen_parameters
from phono3py.version import __version__

# import logging
# logging.basicConfig()
# logging.getLogger("phono3py.phonon3.fc3").setLevel(level=logging.DEBUG)


# AA is created at http://www.network-science.de/ascii/.
def print_phono3py():
    """Show phono3py logo."""
    print(
        r"""        _                      _____
  _ __ | |__   ___  _ __   ___|___ / _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ |_ \| '_ \| | | |
 | |_) | | | | (_) | | | | (_) |__) | |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___/____/| .__/ \__, |
 |_|                                |_|    |___/ """
    )
    print_version(__version__, package_name="phono3py")
    print_time()


def print_end_phono3py():
    """Print END logo."""
    print_time()
    print_end()


def _finalize_phono3py(
    phono3py: Phono3py,
    confs_dict: dict,
    log_level: int,
    write_displacements: bool = False,
    filename: str | None = None,
):
    """Write phono3py.yaml and then exit.

    Parameters
    ----------
    phono3py : Phono3py
        Phono3py instance.
    confs_dict : dict
        This contains the settings and command options that the user set.
    log_level : int
        Log level. 0 means quiet.
    write_displacements : Bool
        When True, crystal structure is written in the length unit of
        calculator interface in phono3py_disp.yaml. Otherwise, the
        default unit (angstrom) is used.
    filename : str, optional
        phono3py.yaml is written in this filename.

    """
    if filename is None:
        yaml_filename = "phono3py.yaml"
    else:
        yaml_filename = filename

    _physical_units = get_calculator_physical_units(phono3py.calculator)

    ph3py_yaml = Phono3pyYaml(
        configuration=confs_dict,
        calculator=phono3py.calculator,
        physical_units=_physical_units,
        settings={
            "force_sets": False,
            "displacements": write_displacements,
        },
    )
    ph3py_yaml.set_phonon_info(phono3py)
    with open(yaml_filename, "w") as w:
        w.write(str(ph3py_yaml))

    if log_level > 0:
        if write_displacements:
            print(f'Displacement dataset was written in "{yaml_filename}".')
        else:
            print(f'Summary of calculation was written in "{yaml_filename}".')
        print_end_phono3py()
    sys.exit(0)


def _get_run_mode(settings: Phono3pySettings):
    """Extract run mode from settings."""
    if settings.is_gruneisen:
        run_mode = "gruneisen"
    elif settings.is_joint_dos:
        run_mode = "jdos"
    elif settings.is_isotope and not (settings.is_bterta or settings.is_lbte):
        run_mode = "isotope"
    elif settings.is_imag_self_energy:
        run_mode = "imag_self_energy"
    elif settings.is_real_self_energy:
        run_mode = "real_self_energy"
    elif settings.is_spectral_function:
        run_mode = "spectral_function"
    elif settings.is_bterta:
        run_mode = "conductivity-RTA"
    elif settings.is_lbte:
        run_mode = "conductivity-LBTE"
    elif (
        settings.create_displacements or settings.random_displacements is not None
    ) and not settings.use_pypolymlp:
        run_mode = "displacements"
    elif settings.write_phonon:
        run_mode = "phonon"
    else:
        run_mode = "force constants"
    return run_mode


def _start_phono3py(**argparse_control) -> tuple[argparse.Namespace, int]:
    """Parse arguments and set some basic parameters."""
    parser, deprecated = get_parser(argparse_control.get("load_phono3py_yaml", False))
    args = parser.parse_args()

    # Log level
    log_level = 1
    if args.verbose:
        log_level = 2
    if args.quiet:
        log_level = 0
    if args.log_level is not None:
        log_level = args.log_level

    # Title
    if log_level:
        print_phono3py()
        import phono3py._phono3py as phono3c  # type: ignore[import]

        max_threads = phono3c.omp_max_threads()
        if max_threads > 0:
            print(f"Compiled with OpenMP support (max {max_threads} threads).")
        if phono3c.include_lapacke():
            print("Compiled with LAPACKE.")

        if argparse_control.get("load_phono3py_yaml", False):
            print("Running in phono3py.load mode.")
        print("Python version %d.%d.%d" % sys.version_info[:3])
        import spglib

        try:  # spglib.get_version() is deprecated.
            print(f"Spglib version {spglib.spg_get_version()}")  # type: ignore
        except AttributeError:
            print("Spglib version %d.%d.%d" % spglib.get_version())  # type: ignore

        if deprecated:
            show_deprecated_option_warnings(deprecated)

    return args, log_level


def _read_phono3py_settings(
    args: argparse.Namespace, argparse_control: dict, log_level: int
):
    """Read phono3py settings.

    From:
    * Traditional configuration file.
    * phono3py.yaml type file
    * Command line options

    """
    load_phono3py_yaml = argparse_control.get("load_phono3py_yaml", False)

    if len(args.filename) > 0:
        file_exists(args.filename[0], log_level=log_level)
        if load_phono3py_yaml:
            phono3py_conf_parser = Phono3pyConfParser(
                filename=args.conf_filename,
                args=args,
                load_phono3py_yaml=load_phono3py_yaml,
            )
            cell_filename = args.filename[0]
        else:
            if is_file_phonopy_yaml(args.filename[0], keyword="phono3py"):
                phono3py_conf_parser = Phono3pyConfParser(
                    args=args, load_phono3py_yaml=load_phono3py_yaml
                )
                cell_filename = args.filename[0]
            else:  # args.filename[0] is assumed to be phono3py-conf file.
                phono3py_conf_parser = Phono3pyConfParser(
                    filename=args.filename[0],
                    args=args,
                    load_phono3py_yaml=load_phono3py_yaml,
                )
                cell_filename = phono3py_conf_parser.settings.cell_filename
    else:
        if load_phono3py_yaml:
            phono3py_conf_parser = Phono3pyConfParser(
                args=args,
                filename=args.conf_filename,
                load_phono3py_yaml=load_phono3py_yaml,
            )
        else:
            phono3py_conf_parser = Phono3pyConfParser(
                args=args, load_phono3py_yaml=load_phono3py_yaml
            )
        cell_filename = phono3py_conf_parser.settings.cell_filename

    confs_dict = phono3py_conf_parser.confs.copy()
    settings = phono3py_conf_parser.settings

    return settings, confs_dict, cell_filename


def _get_default_values(settings: Phono3pySettings):
    """Set default values."""
    # Brillouin zone integration: Tetrahedron (default) or smearing method
    sigma = settings.sigma
    if sigma is None:
        sigmas = []
    elif isinstance(sigma, float):
        sigmas = [sigma]
    else:
        sigmas = sigma
    if settings.is_tetrahedron_method:
        sigmas = [None] + sigmas
    if len(sigmas) == 0:
        sigmas = [None]

    if settings.temperatures is None:
        if settings.is_joint_dos:
            temperature_points = None
            temperatures = None
        else:
            t_max = settings.max_temperature
            t_min = settings.min_temperature
            t_step = settings.temperature_step
            temperature_points = [0.0, 300.0]  # For spectra
            temperatures = np.arange(t_min, t_max + float(t_step) / 10, t_step)
    else:
        temperature_points = settings.temperatures  # For spectra
        temperatures = settings.temperatures  # For others

    if settings.frequency_conversion_factor is None:
        frequency_factor_to_THz = get_physical_units().DefaultToTHz
    else:
        frequency_factor_to_THz = settings.frequency_conversion_factor

    if settings.num_frequency_points is None:
        if settings.frequency_pitch is None:
            num_frequency_points = 201
            frequency_step = None
        else:
            num_frequency_points = None
            frequency_step = settings.frequency_pitch
    else:
        num_frequency_points = settings.num_frequency_points
        frequency_step = None

    if settings.num_points_in_batch is None:
        num_points_in_batch = 10
    else:
        num_points_in_batch = settings.num_points_in_batch

    if settings.frequency_scale_factor is None:
        frequency_scale_factor = None
    else:
        frequency_scale_factor = settings.frequency_scale_factor

    if settings.cutoff_frequency is None:
        cutoff_frequency = 1e-2
    else:
        cutoff_frequency = settings.cutoff_frequency

    params = {}
    params["sigmas"] = sigmas
    params["temperature_points"] = temperature_points
    params["temperatures"] = temperatures
    params["frequency_factor_to_THz"] = frequency_factor_to_THz
    params["num_frequency_points"] = num_frequency_points
    params["num_points_in_batch"] = num_points_in_batch
    params["frequency_step"] = frequency_step
    params["frequency_scale_factor"] = frequency_scale_factor
    params["cutoff_frequency"] = cutoff_frequency

    return params


def _check_supercell_in_yaml(
    cell_info: Phono3pyCellInfoResult,
    ph3: Phono3py,
    distance_to_A: float | None,
    log_level: int,
):
    """Check consistency between generated cells and cells in yaml."""
    if cell_info.phono3py_yaml is not None:
        if distance_to_A is None:
            d2A = 1.0
        else:
            d2A = distance_to_A
        phono3py_yaml = cell_info.phono3py_yaml
        if phono3py_yaml.supercell is not None and ph3.supercell is not None:  # noqa E129
            yaml_cell = phono3py_yaml.supercell.copy()
            yaml_cell.cell = yaml_cell.cell * d2A
            if not cells_isclose(yaml_cell, ph3.supercell):
                if log_level:
                    print(
                        "Generated supercell is inconsistent with "
                        f'that in "{cell_info.optional_structure_info[0]}".'
                    )
                    print_error()
                sys.exit(1)
        if (
            phono3py_yaml.phonon_supercell is not None
            and ph3.phonon_supercell is not None
        ):  # noqa E129
            yaml_cell = phono3py_yaml.phonon_supercell.copy()
            yaml_cell.cell = yaml_cell.cell * d2A
            if not cells_isclose(yaml_cell, ph3.phonon_supercell):
                if log_level:
                    print(
                        "Generated phonon supercell is inconsistent with "
                        f'that in "{cell_info.optional_structure_info[0]}".'
                    )
                    print_error()
                sys.exit(1)


def _init_phono3py_with_cell_info(
    settings: Phono3pySettings,
    cell_info: Phono3pyCellInfoResult,
    interface_mode: str | None,
    symprec: float,
    log_level: int,
) -> tuple[Phono3py, dict]:
    """Initialize phono3py and update settings by default values."""
    # updated_settings keys
    # ('sigmas', 'temperature_points', 'temperatures',
    #  'frequency_factor_to_THz', 'num_frequency_points',
    #  'frequency_step', 'frequency_scale_factor',
    #  'cutoff_frequency')
    phono3py, updated_settings, distance_to_A = _init_phono3py(
        settings,
        cell_info.unitcell.copy(),
        supercell_matrix=cell_info.supercell_matrix,
        primitive_matrix=cell_info.primitive_matrix,
        phonon_supercell_matrix=cell_info.phonon_supercell_matrix,
        interface_mode=interface_mode,
        symprec=symprec,
        log_level=log_level,
    )
    _check_supercell_in_yaml(cell_info, phono3py, distance_to_A, log_level)

    return phono3py, updated_settings


def _init_phono3py(
    settings: Phono3pySettings,
    unitcell: PhonopyAtoms,
    supercell_matrix: Sequence[Sequence[int]] | NDArray | None = None,
    primitive_matrix: Literal["P", "F", "I", "A", "C", "R", "auto"]
    | Sequence[Sequence[float]]
    | NDArray
    | None = None,
    phonon_supercell_matrix: Sequence[Sequence[int]] | NDArray | None = None,
    interface_mode: str | None = None,
    symprec: float = 1e-5,
    log_level: int = 0,
) -> tuple[Phono3py, dict, float | None]:
    """Initialize phono3py and update settings by default values."""
    if interface_mode is not None:
        physical_units = get_calculator_physical_units(interface_mode)
        distance_to_A = physical_units["distance_to_A"]
        assert distance_to_A is not None
        lattice = unitcell.cell
        lattice *= distance_to_A
        unitcell.cell = lattice
    else:
        distance_to_A = None

    updated_settings = _get_default_values(settings)

    phono3py = Phono3py(
        unitcell,
        supercell_matrix,
        primitive_matrix=primitive_matrix,
        phonon_supercell_matrix=phonon_supercell_matrix,
        cutoff_frequency=updated_settings["cutoff_frequency"],
        frequency_factor_to_THz=updated_settings["frequency_factor_to_THz"],
        is_symmetry=settings.is_symmetry,
        is_mesh_symmetry=settings.is_mesh_symmetry,
        use_grg=settings.use_grg,
        make_r0_average=settings.is_fc3_r0_average,
        symprec=symprec,
        log_level=log_level,
    )
    phono3py.masses = settings.masses
    phono3py.band_indices = settings.band_indices
    phono3py.sigmas = updated_settings["sigmas"]
    phono3py.sigma_cutoff = settings.sigma_cutoff_width

    return phono3py, updated_settings, distance_to_A


def _settings_to_grid_points(
    settings: Phono3pySettings, bz_grid: BZGrid
) -> ArrayLike | None:
    """Read or set grid point indices."""
    if settings.grid_addresses is not None:
        grid_points = _grid_addresses_to_grid_points(settings.grid_addresses, bz_grid)
    elif settings.grid_points is not None:
        grid_points = settings.grid_points
    else:
        grid_points = None
    return grid_points


def _grid_addresses_to_grid_points(grid_addresses: NDArray, bz_grid: BZGrid) -> NDArray:
    """Return grid point indices from grid addresses."""
    grid_points = [
        get_grid_point_from_address(ga, bz_grid.D_diag) for ga in grid_addresses
    ]
    return bz_grid.grg2bzg[grid_points]


def _create_supercells_with_displacements(
    settings: Phono3pySettings,
    cell_info: Phono3pyCellInfoResult,
    confs_dict: dict,
    unitcell_filename: str,
    interface_mode: str | None,
    symprec: float = 1e-5,
    output_yaml_filename: str | None = None,
    log_level: int = 0,
):
    """Create supercells and write displacements."""
    if (
        settings.create_displacements
        or settings.random_displacements is not None
        or settings.random_displacements_fc2 is not None
    ):
        ph3py = create_phono3py_supercells(
            cell_info,
            settings,
            symprec,
            interface_mode=interface_mode,
            log_level=log_level,
        )

        if pathlib.Path("BORN").exists():
            store_nac_params(
                cast(Phonopy, ph3py),
                cast(PhonopySettings, settings),
                cell_info.phono3py_yaml,
                unitcell_filename,
                log_level,
                nac_factor=get_physical_units().Hartree * get_physical_units().Bohr,
            )

        if log_level:
            if ph3py.supercell.magnetic_moments is None:
                print("Spacegroup: %s" % ph3py.symmetry.get_international_table())
            else:
                print(
                    "Number of symmetry operations in supercell: %d"
                    % len(ph3py.symmetry.symmetry_operations["rotations"])
                )

        ph3py.save("phono3py_disp.yaml")
        if log_level > 0:
            print('Displacement dataset was written in "phono3py_disp.yaml".')

        _finalize_phono3py(
            ph3py,
            confs_dict,
            log_level,
            filename=output_yaml_filename,
        )


def _run_pypolymlp(
    ph3py: Phono3py,
    settings: Phono3pySettings,
    confs_dict: dict,
    output_yaml_filename: str | None = None,
    mlp_eval_filename: str = "phono3py_mlp_eval_dataset.yaml",
    log_level: int = 0,
) -> Phono3py:
    assert ph3py.mlp_dataset is None
    if ph3py.dataset is not None:  # If None, load mlp from polymlp.yaml.
        ph3py.mlp_dataset = ph3py.dataset
        ph3py.dataset = None

    try:
        develop_or_load_pypolymlp(
            ph3py,
            mlp_params=settings.mlp_params,
            log_level=log_level,
        )
    except (PypolymlpDevelopmentError, PypolymlpFileNotFoundError) as e:
        print_error_message(str(e))
        if log_level:
            print_error()
        sys.exit(1)

    _ph3py = ph3py
    if settings.relax_atomic_positions:
        if log_level:
            print("Relaxing atomic positions using polynomial MLPs...")

        try:
            assert ph3py.mlp is not None
            relaxed_unitcell = relax_atomic_positions(
                ph3py.unitcell,
                ph3py.mlp.mlp,
                verbose=log_level > 1,
            )
        except (PypolymlpRelaxationError, ValueError) as e:
            # ValueError can come from pypolymlp directly.
            print_error_message(str(e))
            if log_level:
                print_error()
            sys.exit(1)

        if log_level:
            if relaxed_unitcell is None:
                print("No relaxation was performed due to symmetry constraints.")
            else:
                get_change_in_positions(
                    relaxed_unitcell, ph3py.unitcell, verbose=log_level > 0
                )
            print("-" * 76)

        if relaxed_unitcell is not None:
            _ph3py, _, _ = _init_phono3py(
                settings,
                relaxed_unitcell.copy(),
                supercell_matrix=ph3py.supercell_matrix,
                primitive_matrix=ph3py.primitive_matrix,
                phonon_supercell_matrix=ph3py.phonon_supercell_matrix,
                symprec=ph3py.symmetry.tolerance,
                log_level=log_level,
            )
            if ph3py.mesh_numbers is not None:
                assert settings.mesh_numbers is not None
                _ph3py.mesh_numbers = settings.mesh_numbers
            _ph3py.nac_params = ph3py.nac_params
            _ph3py.mlp = ph3py.mlp

    if settings.create_displacements or settings.random_displacements is not None:
        generate_displacements_and_evaluate_pypolymlp(
            _ph3py,
            displacement_distance=settings.displacement_distance,
            number_of_snapshots=settings.random_displacements,
            number_estimation_factor=settings.rd_number_estimation_factor,
            random_seed=settings.random_seed,
            fc_calculator=settings.fc_calculator,
            fc_calculator_options=settings.fc_calculator_options,
            cutoff_pair_distance=settings.cutoff_pair_distance,
            symfc_memory_size=settings.symfc_memory_size,
            log_level=log_level,
        )
        if log_level:
            print(f'Dataset generated using MLPs was written in "{mlp_eval_filename}".')
        _ph3py.save(mlp_eval_filename)
    else:
        if log_level:
            print(
                "Generate displacements (--rd or -d) for proceeding to phonon "
                "calculations."
            )
        _finalize_phono3py(
            _ph3py, confs_dict, log_level=log_level, filename=output_yaml_filename
        )

    return _ph3py


def _produce_force_constants(
    ph3py: Phono3py,
    settings: Phono3pySettings,
    log_level: int,
    load_phono3py_yaml: bool,
):
    """Calculate, read, and write force constants."""
    if log_level:
        print("-" * 29 + " Force constants " + "-" * 30)

    read_fc3 = ph3py.fc3 is not None
    read_fc2 = ph3py.fc2 is not None

    cutoff_pair_distance = None
    if settings.use_pypolymlp and ph3py.dataset is not None:
        cutoff_pair_distance = ph3py.dataset.get("cutoff_distance")
    if cutoff_pair_distance is None:
        cutoff_pair_distance = determine_cutoff_pair_distance(
            fc_calculator=settings.fc_calculator,
            fc_calculator_options=settings.fc_calculator_options,
            cutoff_pair_distance=settings.cutoff_pair_distance,
            symfc_memory_size=settings.symfc_memory_size,
            random_displacements=settings.random_displacements,
            supercell=ph3py.supercell,
            primitive=ph3py.primitive,
            log_level=log_level,
        )
    if cutoff_pair_distance is None and ph3py.dataset is not None:
        cutoff_pair_distance = ph3py.dataset.get("cutoff_distance")

    (fc_calculator, fc_calculator_options) = get_fc_calculator_params(
        settings.fc_calculator,
        settings.fc_calculator_options,
        cutoff_pair_distance,
        log_level=(not read_fc3) * 1,
    )
    try:
        compute_force_constants_from_datasets(
            ph3py,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            cutoff_pair_distance=cutoff_pair_distance,
            symmetrize_fc=settings.fc_symmetry,
            is_compact_fc=settings.is_compact_fc,
            load_phono3py_yaml=load_phono3py_yaml,
        )
    except ForceCalculatorRequiredError as e:
        if load_phono3py_yaml:
            if log_level:
                print(str(e))
                print("Try symfc to handle general (or random) displacements.")
        else:
            print_error_message(str(e))
            if log_level:
                print_error()
            sys.exit(1)

        compute_force_constants_from_datasets(
            ph3py,
            fc_calculator="symfc",
            fc_calculator_options=fc_calculator_options,
            cutoff_pair_distance=cutoff_pair_distance,
            symmetrize_fc=settings.fc_symmetry,
            is_compact_fc=settings.is_compact_fc,
        )

    if ph3py.fc3 is None and not settings.write_phonon and not settings.read_gamma:
        if log_level:
            print("fc3 could not be obtained.")
            if not forces_in_dataset(ph3py.dataset):
                print("Forces were not found.")

        print_error()
        sys.exit(1)

    # When settings.write_phonon=True, fc3 can be None.
    if ph3py.fc3 is None:
        assert settings.write_phonon or settings.read_gamma
    else:
        if log_level:
            show_drift_fc3(ph3py.fc3, primitive=ph3py.primitive)

        cutoff_distance = settings.cutoff_fc3_distance
        if cutoff_distance is not None and cutoff_distance > 0:
            if log_level:
                print(
                    "Cutting-off fc3 by zero (cut-off distance: %f)" % cutoff_distance
                )
            ph3py.cutoff_fc3_by_zero(cutoff_distance)

        if not read_fc3:
            write_fc3_to_hdf5(
                ph3py.fc3,
                fc3_nonzero_indices=ph3py.fc3_nonzero_indices,
                p2s_map=ph3py.primitive.p2s_map,
                fc3_cutoff=ph3py.fc3_cutoff,
                compression=settings.hdf5_compression,
            )
            if log_level:
                print('fc3 was written into "fc3.hdf5".')

    if ph3py.fc2 is None:
        if log_level:
            print("fc2 could not be obtained.")
            if ph3py.phonon_supercell_matrix is None:
                if not forces_in_dataset(ph3py.dataset):
                    print("Forces were not found.")
            else:
                if not forces_in_dataset(ph3py.phonon_dataset):
                    print("Forces for dim-fc2 were not found.")

        print_error()
        sys.exit(1)

    if log_level:
        show_drift_force_constants(
            ph3py.fc2, primitive=ph3py.phonon_primitive, name="fc2"
        )

    if not read_fc2:
        write_fc2_to_hdf5(
            ph3py.fc2,
            p2s_map=ph3py.phonon_primitive.p2s_map,
            physical_unit="eV/angstrom^2",
            cutoff=ph3py.fc2_cutoff,
            compression=settings.hdf5_compression,
        )
        if log_level:
            print('fc2 was written into "fc2.hdf5".')


def _run_gruneisen_then_exit(
    phono3py: Phono3py,
    settings: Phono3pySettings,
    log_level: int,
):
    """Run mode Grueneisen parameter calculation from fc3."""
    if (
        settings.mesh_numbers is None
        and settings.band_paths is None
        and settings.qpoints is None
    ):
        print("An option of --mesh, --band, or --qpoints has to be specified.")
        if log_level:
            print_error()
        sys.exit(1)

    assert phono3py.fc2 is not None
    assert phono3py.fc3 is not None
    if len(phono3py.fc2) != len(phono3py.fc3):
        print("Supercells used for fc2 and fc3 have to be same.")
        if log_level:
            print_error()
        sys.exit(1)

    if settings.band_paths is not None:
        if settings.band_points is None:
            npoints = 51
        else:
            npoints = settings.band_points
        band_paths = settings.band_paths
        bands = get_band_qpoints(band_paths, npoints=npoints)
    else:
        bands = None

    rotations = phono3py.primitive_symmetry.pointgroup_operations
    run_gruneisen_parameters(
        phono3py.fc2,
        phono3py.fc3,
        phono3py.supercell,
        phono3py.primitive,
        bands,
        settings.mesh_numbers,
        rotations,
        settings.qpoints,
        nac_params=phono3py.nac_params,
        nac_q_direction=settings.nac_q_direction,
        ion_clamped=settings.ion_clamped,
        factor=get_physical_units().DefaultToTHz,
        symprec=phono3py.symmetry.tolerance,
        log_level=log_level,
    )

    if log_level:
        print_end_phono3py()
    sys.exit(0)


def _run_jdos_then_exit(
    phono3py: Phono3py,
    settings: Phono3pySettings,
    updated_settings: dict,
    log_level: int,
):
    """Run joint-DOS calculation."""
    joint_dos = Phono3pyJointDos(
        phono3py.phonon_supercell,
        phono3py.phonon_primitive,
        phono3py.fc2,
        mesh=settings.mesh_numbers,
        nac_params=phono3py.nac_params,
        nac_q_direction=settings.nac_q_direction,
        sigmas=updated_settings["sigmas"],
        cutoff_frequency=updated_settings["cutoff_frequency"],
        frequency_step=updated_settings["frequency_step"],
        num_frequency_points=updated_settings["num_frequency_points"],
        num_points_in_batch=updated_settings["num_points_in_batch"],
        temperatures=updated_settings["temperature_points"],
        frequency_factor_to_THz=updated_settings["frequency_factor_to_THz"],
        frequency_scale_factor=updated_settings["frequency_scale_factor"],
        use_grg=settings.use_grg,
        is_mesh_symmetry=settings.is_mesh_symmetry,
        symprec=phono3py.symmetry.tolerance,
        log_level=log_level,
    )

    if log_level > 0:
        dm = joint_dos.dynamical_matrix
        if isinstance(dm, DynamicalMatrixGL):
            dm.show_nac_message()

    assert joint_dos.grid is not None
    grid_points = _settings_to_grid_points(settings, joint_dos.grid)
    joint_dos.run(grid_points, write_jdos=True)

    if log_level:
        print_end_phono3py()
    sys.exit(0)


def _run_isotope_then_exit(
    phono3py: Phono3py,
    settings: Phono3pySettings,
    updated_settings: dict,
    log_level: int,
):
    """Run isotope scattering calculation."""
    mass_variances = settings.mass_variances
    if settings.band_indices is not None:
        band_indices = np.hstack(settings.band_indices).astype("intc")
    else:
        band_indices = None
    iso = Phono3pyIsotope(
        settings.mesh_numbers,
        phono3py.phonon_primitive,
        mass_variances=mass_variances,
        band_indices=band_indices,
        sigmas=updated_settings["sigmas"],
        frequency_factor_to_THz=updated_settings["frequency_factor_to_THz"],
        use_grg=settings.use_grg,
        symprec=phono3py.symmetry.tolerance,
        cutoff_frequency=settings.cutoff_frequency,
        lapack_zheev_uplo=settings.lapack_zheev_uplo,
    )
    iso.init_dynamical_matrix(
        phono3py.fc2,
        phono3py.phonon_supercell,
        phono3py.phonon_primitive,
        nac_params=phono3py.nac_params,
        frequency_scale_factor=updated_settings["frequency_scale_factor"],
    )
    if log_level > 0:
        dm = iso.dynamical_matrix
        if isinstance(dm, DynamicalMatrixGL):
            dm.show_nac_message()

    grid_points = _settings_to_grid_points(settings, iso.grid)
    iso.run(grid_points)

    if log_level:
        print_end_phono3py()
    sys.exit(0)


def _init_phph_interaction(
    phono3py: Phono3py,
    settings: Phono3pySettings,
    updated_settings: dict,
    log_level: int,
):
    """Initialize ph-ph interaction and phonons on grid."""
    if log_level:
        print("Generating grid system ... ", end="", flush=True)
    assert phono3py.grid is not None
    assert phono3py.mesh_numbers is not None
    bz_grid = phono3py.grid
    if log_level:
        if bz_grid.grid_matrix is None:
            print("[ %d %d %d ]" % tuple(phono3py.mesh_numbers))
        else:
            print("")
            print("Generalized regular grid: [ %d %d %d ]" % tuple(bz_grid.D_diag))
            print("Grid generation matrix:")
            print("  [ %d %d %d ]" % tuple(bz_grid.grid_matrix[0]))
            print("  [ %d %d %d ]" % tuple(bz_grid.grid_matrix[1]))
            print("  [ %d %d %d ]" % tuple(bz_grid.grid_matrix[2]))

        if settings.is_symmetrize_fc3_q:
            print("Permutation symmetry of ph-ph interaction strengths: True")
        if settings.is_fc3_r0_average:
            print("fc3-r2q-transformation over three atoms: True")
        else:
            print("fc3-r2q-transformation over three atoms: False")

    ave_pp = settings.constant_averaged_pp_interaction
    phono3py.init_phph_interaction(
        nac_q_direction=settings.nac_q_direction,
        constant_averaged_interaction=ave_pp,
        frequency_scale_factor=updated_settings["frequency_scale_factor"],
        symmetrize_fc3q=settings.is_symmetrize_fc3_q,
        lapack_zheev_uplo=settings.lapack_zheev_uplo,
    )

    if not settings.read_phonon:
        if log_level:
            print("-" * 27 + " Phonon calculations " + "-" * 28)
            dm = phono3py.dynamical_matrix
            if isinstance(dm, DynamicalMatrixGL):
                dm.show_nac_message()
            print("Running harmonic phonon calculations...")
            sys.stdout.flush()
        phono3py.run_phonon_solver()

    if settings.write_phonon:
        freqs, eigvecs, grid_address = phono3py.get_phonon_data()
        ir_grid_points, ir_grid_weights, _ = get_ir_grid_points(bz_grid)
        ir_grid_points = np.array(bz_grid.grg2bzg[ir_grid_points], dtype="int64")
        filename = write_phonon_to_hdf5(
            freqs,
            eigvecs,
            grid_address,
            phono3py.mesh_numbers,
            bz_grid=bz_grid,
            ir_grid_points=ir_grid_points,
            ir_grid_weights=ir_grid_weights,
            compression=settings.hdf5_compression,
        )
        if filename:
            if log_level:
                print('Phonons are written into "%s".' % filename)
        else:
            print("Writing phonons failed.")
            if log_level:
                print_error()
            sys.exit(1)

    if settings.read_phonon:
        phonons = read_phonon_from_hdf5(phono3py.mesh_numbers, verbose=(log_level > 0))
        if phonons is None:
            print("Reading phonons failed.")
            if log_level:
                print_error()
            sys.exit(1)

        try:
            phono3py.set_phonon_data(*phonons)
        except RuntimeError:
            if log_level:
                print_error()
            sys.exit(1)


def _load_dataset_and_phonon_dataset(
    ph3py: Phono3py,
    ph3py_yaml: Phono3pyYaml | None = None,
    forces_fc3_filename: str | os.PathLike | Sequence | None = None,
    forces_fc2_filename: str | os.PathLike | Sequence | None = None,
    phono3py_yaml_filename: str | os.PathLike | None = None,
    cutoff_pair_distance: float | None = None,
    calculator: str | None = None,
    log_level: int = 0,
):
    """Set displacements, forces, and create force constants."""
    if ph3py.fc3 is None or (
        ph3py.fc2 is None and ph3py.phonon_supercell_matrix is None
    ):
        dataset = select_and_load_dataset(
            ph3py,
            ph3py_yaml=ph3py_yaml,
            forces_fc3_filename=forces_fc3_filename,
            phono3py_yaml_filename=phono3py_yaml_filename,
            cutoff_pair_distance=cutoff_pair_distance,
            calculator=calculator,
            log_level=log_level,
        )
        if dataset is not None:
            ph3py.dataset = dataset

    if ph3py.fc2 is None and ph3py.phonon_supercell_matrix is not None:
        phonon_dataset = select_and_load_phonon_dataset(
            ph3py,
            ph3py_yaml=ph3py_yaml,
            forces_fc2_filename=forces_fc2_filename,
            calculator=calculator,
            log_level=log_level,
        )
        if phonon_dataset is not None:
            ph3py.phonon_dataset = phonon_dataset


def main(**argparse_control):
    """Phono3py main part of command line interface."""
    # import warnings

    # warnings.simplefilter("error")
    load_phono3py_yaml = argparse_control.get("load_phono3py_yaml", False)

    if "args" in argparse_control:  # This is for pytest.
        args = argparse_control["args"]
        log_level = args.log_level
    else:
        args, log_level = _start_phono3py(**argparse_control)

    output_yaml_filename: str | None
    if load_phono3py_yaml:
        output_yaml_filename = args.output_yaml_filename
    else:
        output_yaml_filename = None

    settings, confs_dict, cell_filename = _read_phono3py_settings(
        args, argparse_control, log_level
    )

    if args.force_sets_to_forces_fc2_mode:
        create_FORCES_FC2_from_FORCE_SETS(log_level)
        if log_level:
            print_end_phono3py()
        sys.exit(0)
    if args.force_sets_mode:
        create_FORCE_SETS_from_FORCES_FCx(
            settings.phonon_supercell_matrix, cell_filename, log_level
        )
        if log_level:
            print_end_phono3py()
        sys.exit(0)
    if args.write_grid_points:
        run_mode = "write_grid_info"
    elif args.show_num_triplets:
        run_mode = "show_triplets_info"
    else:
        run_mode = None

    # -----------------------------------------------------------------------
    # ----------------- 'args' should not be used below. --------------------
    # -----------------------------------------------------------------------

    ####################################
    # Create FORCES_FC3 and FORCES_FC2 #
    ####################################
    if (
        settings.create_forces_fc3
        or settings.create_forces_fc3_file
        or settings.create_forces_fc2
    ):
        create_FORCES_FC3_and_FORCES_FC2(settings, cell_filename, log_level=log_level)
        if log_level:
            print_end_phono3py()
        sys.exit(0)

    ###########################################################
    # Symmetry tolerance. Distance unit depends on interface. #
    ###########################################################
    if settings.symmetry_tolerance is None:
        symprec = 1e-5
    else:
        symprec = settings.symmetry_tolerance

    try:
        cell_info = get_cell_info(
            settings,
            cell_filename,
            log_level=log_level,
            load_phonopy_yaml=load_phono3py_yaml,
        )
    except CellNotFoundError as e:
        print_error_message(str(e))
        if log_level:
            print_error()
        sys.exit(1)

    unitcell_filename = cell_info.optional_structure_info[0]
    interface_mode = cell_info.interface_mode

    if run_mode is None:
        run_mode = _get_run_mode(settings)
    assert run_mode is not None

    ######################################################
    # Create supercells with displacements and then exit #
    ######################################################
    if not settings.use_pypolymlp:
        _create_supercells_with_displacements(
            settings,
            cell_info,
            confs_dict,
            unitcell_filename,
            interface_mode,
            symprec=symprec,
            output_yaml_filename=output_yaml_filename,
            log_level=log_level,
        )

    #######################
    # Initialize phono3py #
    #######################
    # updated_settings keys
    # ('sigmas', 'temperature_points', 'temperatures',
    #  'frequency_factor_to_THz', 'num_frequency_points',
    #  'frequency_step', 'frequency_scale_factor',
    #  'cutoff_frequency')
    ph3py, updated_settings = _init_phono3py_with_cell_info(
        settings, cell_info, interface_mode, symprec, log_level
    )

    #################################################
    # Show phono3py settings and crystal structures #
    #################################################
    if log_level:
        show_general_settings(
            settings,
            run_mode,
            ph3py,
            unitcell_filename,
            interface_mode,
        )

        if ph3py.supercell.magnetic_moments is None:
            print("Spacegroup: %s" % ph3py.symmetry.get_international_table())
        else:
            print(
                "Number of symmetry operations in supercell: %d"
                % len(ph3py.symmetry.symmetry_operations["rotations"])
            )

    if log_level > 1:
        show_phono3py_cells(ph3py)
    elif log_level:
        print(
            "Use -v option to watch primitive cell, unit cell, "
            "and supercell structures."
        )

    ###############################
    # Memory estimation for symfc #
    ###############################
    if settings.show_symfc_memory_usage and load_phono3py_yaml:
        print("Quick estimation of memory size required for solving fc3 by symfc")
        print("cutoff   memsize")
        print("------   -------")
        estimate_symfc_cutoff_from_memsize(
            ph3py.supercell, ph3py.primitive, ph3py.symmetry, 3, verbose=True
        )

        if log_level:
            print_end_phono3py()
        sys.exit(0)

    ##################
    # Check settings #
    ##################
    run_modes_with_mesh = (
        "conductivity-RTA",
        "conductivity-LBTE",
        "imag_self_energy",
        "real_self_energy",
        "jdos",
        "isotope",
        "phonon",
        "write_grid_info",
        "show_triplets_info",
    )
    run_modes_with_gp = ("imag_self_energy", "real_self_energy", "jdos", "isotope")

    if settings.mesh_numbers is None and run_mode in run_modes_with_mesh:
        print("")
        print("Mesh numbers have to be specified.")
        print("")
        if log_level:
            print_error()
        sys.exit(1)

    if (
        run_mode in run_modes_with_gp
        and settings.grid_points is None
        and settings.grid_addresses is None
    ):  # noqa E129
        print("")
        print("Grid point(s) has to be specified.")
        print("")
        if log_level:
            print_error()
        sys.exit(1)

    ####################
    # Set mesh numbers #
    ####################
    if run_mode in run_modes_with_mesh:
        assert settings.mesh_numbers is not None
        # jdos and isotope modes need to set mesh numbers differently.
        if run_mode not in ("jdos", "isotope"):
            ph3py.mesh_numbers = settings.mesh_numbers

    #########################################################
    # Write ir-grid points and grid addresses and then exit #
    #########################################################
    if run_mode == "write_grid_info":
        assert ph3py.grid is not None
        write_grid_points(
            ph3py.primitive,
            ph3py.grid,
            band_indices=settings.band_indices,
            sigmas=updated_settings["sigmas"],
            temperatures=updated_settings["temperatures"],
            is_kappa_star=settings.is_kappa_star,
            is_lbte=(settings.write_collision or settings.is_lbte),
            compression=settings.hdf5_compression,
        )

        if log_level:
            print_end_phono3py()
        sys.exit(0)

    ################################################################
    # Show reduced number of triplets at grid points and then exit #
    ################################################################
    if run_mode == "show_triplets_info":
        assert ph3py.grid is not None
        grid_points = _settings_to_grid_points(settings, ph3py.grid)
        show_num_triplets(
            ph3py.primitive,
            ph3py.grid,
            band_indices=settings.band_indices,
            grid_points=grid_points,
            is_kappa_star=settings.is_kappa_star,
        )

        if log_level:
            print_end_phono3py()
        sys.exit(0)

    ##################################
    # Non-analytical term correction #
    ##################################
    if settings.is_nac:
        store_nac_params(
            cast(Phonopy, ph3py),
            settings,
            cell_info.phono3py_yaml,
            unitcell_filename,
            log_level,
            nac_factor=get_physical_units().Hartree * get_physical_units().Bohr,
        )

    ########################
    # Read force constants #
    ########################
    load_fc2_and_fc3(
        ph3py,
        read_fc3=settings.read_fc3,
        read_fc2=settings.read_fc2,
        log_level=log_level,
    )

    ############
    # Datasets #
    ############
    _load_dataset_and_phonon_dataset(
        ph3py,
        ph3py_yaml=cell_info.phono3py_yaml,
        phono3py_yaml_filename=unitcell_filename,
        cutoff_pair_distance=settings.cutoff_pair_distance,
        calculator=interface_mode,
        log_level=log_level,
    )

    ###################
    # polynomial MLPs #
    ###################
    if settings.use_pypolymlp:
        if ph3py.fc3 is None or (
            ph3py.fc2 is None and ph3py.phonon_supercell_matrix is None
        ):
            # Note that ph3py is replaced when relax_atomic_positions is True.
            ph3py = _run_pypolymlp(
                ph3py,
                settings,
                confs_dict,
                output_yaml_filename=output_yaml_filename,
                log_level=log_level,
            )
        else:
            if log_level:
                print(
                    "Pypolymlp was not developed or used because fc2 and fc3 "
                    "are available."
                )

    ###########################
    # Produce force constants #
    ###########################
    _produce_force_constants(ph3py, settings, log_level, load_phono3py_yaml)

    ############################################
    # Phonon Gruneisen parameter and then exit #
    ############################################
    if settings.is_gruneisen:
        _run_gruneisen_then_exit(ph3py, settings, log_level)

    #################
    # Show settings #
    #################
    if log_level:
        show_phono3py_settings(ph3py, settings, updated_settings, log_level)

    ###########################
    # Joint DOS and then exit #
    ###########################
    if run_mode == "jdos":
        _run_jdos_then_exit(ph3py, settings, updated_settings, log_level)

    ################################################
    # Mass variances for phonon-isotope scattering #
    ################################################
    if settings.is_isotope and settings.mass_variances is None:
        from phonopy.structure.atoms import isotope_data

        symbols = ph3py.phonon_primitive.symbols
        in_database = True
        for s in set(symbols):
            if s not in isotope_data:
                print("%s is not in the list of isotope databese" % s)
                print("(not implemented).")
                print("Use --mass_variances option.")
                in_database = False
        if not in_database:
            if log_level:
                print_end_phono3py()
            sys.exit(0)

    #########################################
    # Phonon-isotope lifetime and then exit #
    #########################################
    if run_mode == "isotope":
        _run_isotope_then_exit(ph3py, settings, updated_settings, log_level)

    ########################################
    # Initialize phonon-phonon interaction #
    ########################################
    if run_mode in run_modes_with_mesh:
        _init_phph_interaction(
            ph3py,
            settings,
            updated_settings,
            log_level,
        )

    #######################################################
    # Run imaginary part of self energy of bubble diagram #
    #######################################################
    if run_mode == "imag_self_energy":
        assert ph3py.grid is not None
        ph3py.run_imag_self_energy(
            _settings_to_grid_points(settings, ph3py.grid),
            updated_settings["temperature_points"],
            frequency_step=updated_settings["frequency_step"],
            num_frequency_points=updated_settings["num_frequency_points"],
            num_points_in_batch=updated_settings["num_points_in_batch"],
            scattering_event_class=settings.scattering_event_class,
            write_txt=True,
            write_gamma_detail=settings.write_gamma_detail,
        )

    #####################################################
    # Run frequency shift calculation of bubble diagram #
    #####################################################
    elif run_mode == "real_self_energy":
        assert ph3py.grid is not None
        ph3py.run_real_self_energy(
            _settings_to_grid_points(settings, ph3py.grid),
            updated_settings["temperature_points"],
            frequency_step=updated_settings["frequency_step"],
            num_frequency_points=updated_settings["num_frequency_points"],
            write_txt=True,
            write_hdf5=True,
        )

    #######################################################
    # Run spectral function calculation of bubble diagram #
    #######################################################
    elif run_mode == "spectral_function":
        assert ph3py.grid is not None
        ph3py.run_spectral_function(
            _settings_to_grid_points(settings, ph3py.grid),
            updated_settings["temperature_points"],
            frequency_step=updated_settings["frequency_step"],
            num_frequency_points=updated_settings["num_frequency_points"],
            num_points_in_batch=updated_settings["num_points_in_batch"],
            write_txt=True,
            write_hdf5=True,
        )

    ####################################
    # Run lattice thermal conductivity #
    ####################################
    elif run_mode == "conductivity-RTA" or run_mode == "conductivity-LBTE":
        assert ph3py.grid is not None
        grid_points = _settings_to_grid_points(settings, ph3py.grid)
        ph3py.run_thermal_conductivity(
            is_LBTE=settings.is_lbte,
            temperatures=updated_settings["temperatures"],
            is_isotope=settings.is_isotope,
            mass_variances=settings.mass_variances,
            grid_points=grid_points,
            boundary_mfp=settings.boundary_mfp,
            solve_collective_phonon=settings.solve_collective_phonon,
            use_ave_pp=settings.use_ave_pp,
            is_reducible_collision_matrix=settings.is_reducible_collision_matrix,  # noqa E501
            is_kappa_star=settings.is_kappa_star,
            gv_delta_q=settings.group_velocity_delta_q,
            is_full_pp=(settings.is_full_pp or settings.is_symmetrize_fc3_q),
            pinv_cutoff=settings.pinv_cutoff,
            pinv_solver=settings.pinv_solver,
            pinv_method=settings.pinv_method,
            write_gamma=settings.write_gamma,
            read_gamma=settings.read_gamma,
            write_kappa=True,
            is_N_U=settings.is_N_U,
            conductivity_type=settings.conductivity_type,
            write_gamma_detail=settings.write_gamma_detail,
            write_collision=settings.write_collision,
            read_collision=settings.read_collision,
            write_pp=settings.write_pp,
            read_pp=settings.read_pp,
            write_LBTE_solution=settings.write_LBTE_solution,
            compression=settings.hdf5_compression,
        )
    else:
        if log_level:
            print(
                "-" * 11
                + " None of ph-ph interaction calculation was performed. "
                + "-" * 11
            )

    _finalize_phono3py(ph3py, confs_dict, log_level, filename=output_yaml_filename)
