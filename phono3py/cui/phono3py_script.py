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
import pathlib
import sys
import warnings
from typing import Optional

import numpy as np
from phonopy.cui.collect_cell_info import collect_cell_info
from phonopy.cui.phonopy_argparse import show_deprecated_option_warnings
from phonopy.cui.phonopy_script import (
    file_exists,
    print_end,
    print_error,
    print_error_message,
    print_time,
    print_version,
    set_magnetic_moments,
    store_nac_params,
)
from phonopy.exception import ForceCalculatorRequiredError
from phonopy.file_IO import is_file_phonopy_yaml
from phonopy.harmonic.force_constants import show_drift_force_constants
from phonopy.interface.calculator import get_default_physical_units
from phonopy.phonon.band_structure import get_band_qpoints
from phonopy.structure.cells import isclose as cells_isclose
from phonopy.units import Bohr, Hartree, VaspToTHz

from phono3py import Phono3py, Phono3pyIsotope, Phono3pyJointDos
from phono3py.cui.create_force_constants import (
    create_phono3py_force_constants,
    get_cutoff_pair_distance,
    get_fc_calculator_params,
    run_pypolymlp_to_compute_forces,
)
from phono3py.cui.create_force_sets import (
    create_FORCE_SETS_from_FORCES_FCx,
    create_FORCES_FC2_from_FORCE_SETS,
    create_FORCES_FC3_and_FORCES_FC2,
)
from phono3py.cui.create_supercells import create_phono3py_supercells
from phono3py.cui.load import (
    compute_force_constants_from_datasets,
    load_dataset_and_phonon_dataset,
    load_fc2_and_fc3,
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
from phono3py.interface.fc_calculator import estimate_symfc_memory_usage
from phono3py.interface.phono3py_yaml import Phono3pyYaml
from phono3py.phonon.grid import get_grid_point_from_address, get_ir_grid_points
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


def finalize_phono3py(
    phono3py: Phono3py,
    confs_dict,
    log_level,
    write_displacements=False,
    filename=None,
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

    _physical_units = get_default_physical_units(phono3py.calculator)

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


def get_run_mode(settings):
    """Extract run mode from settings."""
    run_mode = None
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
    return run_mode


def start_phono3py(**argparse_control) -> tuple[argparse.Namespace, int]:
    """Parse arguments and set some basic parameters."""
    parser, deprecated = get_parser(**argparse_control)
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
        import phono3py._phono3py as phono3c

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
            print(f"Spglib version {spglib.spg_get_version()}")
        except AttributeError:
            print("Spglib version %d.%d.%d" % spglib.get_version())

        if deprecated:
            show_deprecated_option_warnings(deprecated)

    return args, log_level


def read_phono3py_settings(args, argparse_control, log_level):
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
                default_settings=argparse_control,
            )
            cell_filename = args.filename[0]
        else:
            if is_file_phonopy_yaml(args.filename[0], keyword="phono3py"):
                phono3py_conf_parser = Phono3pyConfParser(
                    args=args, default_settings=argparse_control
                )
                cell_filename = args.filename[0]
            else:  # args.filename[0] is assumed to be phono3py-conf file.
                phono3py_conf_parser = Phono3pyConfParser(
                    filename=args.filename[0],
                    args=args,
                    default_settings=argparse_control,
                )
                cell_filename = phono3py_conf_parser.settings.cell_filename
    else:
        if load_phono3py_yaml:
            phono3py_conf_parser = Phono3pyConfParser(
                args=args,
                filename=args.conf_filename,
                default_settings=argparse_control,
            )
        else:
            phono3py_conf_parser = Phono3pyConfParser(
                args=args, default_settings=argparse_control
            )
        cell_filename = phono3py_conf_parser.settings.cell_filename

    confs_dict = phono3py_conf_parser.confs.copy()
    settings = phono3py_conf_parser.settings

    return settings, confs_dict, cell_filename


def get_input_output_filenames_from_args(args):
    """Return strings inserted to input and output filenames."""
    if args.input_filename is not None:
        warnings.warn(
            "-i option is deprecated and will be removed soon.",
            DeprecationWarning,
            stacklevel=2,
        )
    input_filename = args.input_filename
    if args.output_filename is not None:
        warnings.warn(
            "-o option is deprecated and will be removed soon.",
            DeprecationWarning,
            stacklevel=2,
        )
    output_filename = args.output_filename
    if args.input_output_filename is not None:
        warnings.warn(
            "--io option is deprecated and will be removed soon.",
            DeprecationWarning,
            stacklevel=2,
        )
        input_filename = args.input_output_filename
        output_filename = args.input_output_filename
    return input_filename, output_filename


def get_cell_info(
    settings: Phono3pySettings, cell_filename: str, log_level: int
) -> dict:
    """Return calculator interface and crystal structure information."""
    cell_info = collect_cell_info(
        supercell_matrix=settings.supercell_matrix,
        primitive_matrix=settings.primitive_matrix,
        interface_mode=settings.calculator,
        cell_filename=cell_filename,
        chemical_symbols=settings.chemical_symbols,
        phonopy_yaml_cls=Phono3pyYaml,
    )
    if "error_message" in cell_info:
        print_error_message(cell_info["error_message"])
        if log_level > 0:
            print_error()
        sys.exit(1)

    set_magnetic_moments(cell_info, settings, log_level)

    cell_info["phonon_supercell_matrix"] = settings.phonon_supercell_matrix
    ph3py_yaml: Phono3pyYaml = cell_info["phonopy_yaml"]
    if cell_info["phonon_supercell_matrix"] is None and ph3py_yaml:
        ph_smat = ph3py_yaml.phonon_supercell_matrix
        cell_info["phonon_supercell_matrix"] = ph_smat

    return cell_info


def get_default_values(settings):
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
        frequency_factor_to_THz = VaspToTHz
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


def check_supercell_in_yaml(cell_info, ph3, distance_to_A, log_level):
    """Check consistency between generated cells and cells in yaml."""
    if cell_info["phonopy_yaml"] is not None:
        if distance_to_A is None:
            d2A = 1.0
        else:
            d2A = distance_to_A
        if (
            cell_info["phonopy_yaml"].supercell is not None
            and ph3.supercell is not None
        ):  # noqa E129
            yaml_cell = cell_info["phonopy_yaml"].supercell.copy()
            yaml_cell.cell = yaml_cell.cell * d2A
            if not cells_isclose(yaml_cell, ph3.supercell):
                if log_level:
                    print(
                        "Generated supercell is inconsistent with "
                        'that in "%s".' % cell_info["optional_structure_info"][0]
                    )
                    print_error()
                sys.exit(1)
        if (
            cell_info["phonopy_yaml"].phonon_supercell is not None
            and ph3.phonon_supercell is not None
        ):  # noqa E129
            yaml_cell = cell_info["phonopy_yaml"].phonon_supercell.copy()
            yaml_cell.cell = yaml_cell.cell * d2A
            if not cells_isclose(yaml_cell, ph3.phonon_supercell):
                if log_level:
                    print(
                        "Generated phonon supercell is inconsistent with "
                        'that in "%s".' % cell_info["optional_structure_info"][0]
                    )
                    print_error()
                sys.exit(1)


def init_phono3py(
    settings, cell_info, interface_mode, symprec, log_level
) -> tuple[Phono3py, dict]:
    """Initialize phono3py and update settings by default values."""
    physical_units = get_default_physical_units(interface_mode)
    distance_to_A = physical_units["distance_to_A"]

    # Change unit of lattice parameters to angstrom
    unitcell = cell_info["unitcell"].copy()
    if distance_to_A is not None:
        lattice = unitcell.cell
        lattice *= distance_to_A
        unitcell.cell = lattice

    # updated_settings keys
    # ('sigmas', 'temperature_points', 'temperatures',
    #  'frequency_factor_to_THz', 'num_frequency_points',
    #  'frequency_step', 'frequency_scale_factor',
    #  'cutoff_frequency')
    updated_settings = get_default_values(settings)

    phono3py = Phono3py(
        unitcell,
        cell_info["supercell_matrix"],
        primitive_matrix=cell_info["primitive_matrix"],
        phonon_supercell_matrix=cell_info["phonon_supercell_matrix"],
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

    check_supercell_in_yaml(cell_info, phono3py, distance_to_A, log_level)

    return phono3py, updated_settings


def settings_to_grid_points(settings, bz_grid):
    """Read or set grid point indices."""
    if settings.grid_addresses is not None:
        grid_points = grid_addresses_to_grid_points(settings.grid_addresses, bz_grid)
    elif settings.grid_points is not None:
        grid_points = settings.grid_points
    else:
        grid_points = None
    return grid_points


def grid_addresses_to_grid_points(grid_addresses, bz_grid):
    """Return grid point indices from grid addresses."""
    grid_points = [
        get_grid_point_from_address(ga, bz_grid.D_diag) for ga in grid_addresses
    ]
    return bz_grid.grg2bzg[grid_points]


def create_supercells_with_displacements(
    settings: Phono3pySettings,
    cell_info: dict,
    confs_dict: dict,
    unitcell_filename: str,
    interface_mode: Optional[str],
    load_phono3py_yaml: bool,
    symprec: float,
    log_level: int,
):
    """Create supercells and write displacements."""
    if (
        settings.create_displacements
        or settings.random_displacements is not None
        or settings.random_displacements_fc2 is not None
    ):
        phono3py = create_phono3py_supercells(
            cell_info,
            settings,
            symprec,
            interface_mode=interface_mode,
            log_level=log_level,
        )

        if pathlib.Path("BORN").exists():
            store_nac_params(
                phono3py,
                settings,
                cell_info["phonopy_yaml"],
                unitcell_filename,
                log_level,
                nac_factor=Hartree * Bohr,
                load_phonopy_yaml=load_phono3py_yaml,
            )

        if log_level:
            if phono3py.supercell.magnetic_moments is None:
                print("Spacegroup: %s" % phono3py.symmetry.get_international_table())
            else:
                print(
                    "Number of symmetry operations in supercell: %d"
                    % len(phono3py.symmetry.symmetry_operations["rotations"])
                )

        finalize_phono3py(
            phono3py,
            confs_dict,
            log_level,
            write_displacements=True,
            filename="phono3py_disp.yaml",
        )


def _store_force_constants(ph3py: Phono3py, settings: Phono3pySettings, log_level: int):
    """Calculate, read, and write force constants."""
    if log_level:
        print("-" * 29 + " Force constants " + "-" * 30)

    read_fc3 = ph3py.fc3 is not None
    read_fc2 = ph3py.fc2 is not None

    load_fc2_and_fc3(ph3py, log_level=log_level)

    cutoff_pair_distance = get_cutoff_pair_distance(settings)
    (fc_calculator, fc_calculator_options) = get_fc_calculator_params(
        settings, log_level=(not read_fc3) * 1
    )
    try:
        compute_force_constants_from_datasets(
            ph3py,
            fc_calculator=fc_calculator,
            fc_calculator_options=fc_calculator_options,
            cutoff_pair_distance=cutoff_pair_distance,
            symmetrize_fc=settings.fc_symmetry,
            is_compact_fc=settings.is_compact_fc,
            log_level=log_level,
        )
    except ForceCalculatorRequiredError:
        if log_level:
            print("Symfc will be used to handle general (or random) displacements.")

        compute_force_constants_from_datasets(
            ph3py,
            fc_calculator="symfc",
            fc_calculator_options=fc_calculator_options,
            cutoff_pair_distance=cutoff_pair_distance,
            symmetrize_fc=settings.fc_symmetry,
            is_compact_fc=settings.is_compact_fc,
            log_level=log_level,
        )
        # _show_fc_calculator_not_found(log_level)

    if log_level:
        if ph3py.fc3 is None:
            print("fc3 could not be obtained.")
            if not forces_in_dataset(ph3py.dataset):
                print("Forces were not found.")
        else:
            show_drift_fc3(ph3py.fc3, primitive=ph3py.primitive)
        if ph3py.fc2 is None:
            print("fc2 could not be obtained.")
            if ph3py.phonon_supercell_matrix is None:
                if not forces_in_dataset(ph3py.dataset):
                    print("Forces were not found.")
            else:
                if not forces_in_dataset(ph3py.phonon_dataset):
                    print("Forces for dim-fc2 were not found.")
        else:
            show_drift_force_constants(
                ph3py.fc2, primitive=ph3py.phonon_primitive, name="fc2"
            )

    if ph3py.fc3 is None or ph3py.fc2 is None:
        print_error()
        sys.exit(1)

    cutoff_distance = settings.cutoff_fc3_distance
    if cutoff_distance is not None and cutoff_distance > 0:
        if log_level:
            print("Cutting-off fc3 by zero (cut-off distance: %f)" % cutoff_distance)
        ph3py.cutoff_fc3_by_zero(cutoff_distance)

    if not read_fc3:
        write_fc3_to_hdf5(
            ph3py.fc3,
            p2s_map=ph3py.primitive.p2s_map,
            compression=settings.hdf5_compression,
        )
        if log_level:
            print('fc3 was written into "fc3.hdf5".')
    if not read_fc2:
        write_fc2_to_hdf5(
            ph3py.fc2,
            p2s_map=ph3py.primitive.p2s_map,
            physical_unit="eV/angstrom^2",
            compression=settings.hdf5_compression,
        )
        if log_level:
            print('fc2 was written into "fc2.hdf5".')


def _show_fc_calculator_not_found(log_level):
    if log_level:
        print("")
        print(
            "Built-in force constants calculator doesn't support the "
            "displacements-forces dataset. "
            "An external force calculator, e.g., symfc (--symfc_ or ALM (--alm), "
            "has to be used to compute force constants."
        )
        print_error()
    sys.exit(1)


def run_gruneisen_then_exit(phono3py, settings, output_filename, log_level):
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
        factor=VaspToTHz,
        symprec=phono3py.symmetry.tolerance,
        output_filename=output_filename,
        log_level=log_level,
    )

    if log_level:
        print_end_phono3py()
    sys.exit(0)


def run_jdos_then_exit(
    phono3py: Phono3py, settings, updated_settings, output_filename, log_level
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
        output_filename=output_filename,
        log_level=log_level,
    )

    if log_level > 0:
        dm = joint_dos.dynamical_matrix
        if dm.is_nac() and dm.nac_method == "gonze":
            dm.show_Gonze_nac_message()

    grid_points = settings_to_grid_points(settings, joint_dos.grid)
    joint_dos.run(grid_points, write_jdos=True)

    if log_level:
        print_end_phono3py()
    sys.exit(0)


def run_isotope_then_exit(phono3py, settings, updated_settings, log_level):
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
        if dm.is_nac() and dm.nac_method == "gonze":
            dm.show_Gonze_nac_message()

    grid_points = settings_to_grid_points(settings, iso.grid)
    iso.run(grid_points)

    if log_level:
        print_end_phono3py()
    sys.exit(0)


def init_phph_interaction(
    phono3py: Phono3py,
    settings,
    updated_settings,
    input_filename,
    output_filename,
    log_level,
):
    """Initialize ph-ph interaction and phonons on grid."""
    if log_level:
        print("Generating grid system ... ", end="", flush=True)
    phono3py.mesh_numbers = settings.mesh_numbers
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
            print("fc3 r2q transformation over three atoms: True")

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
            if dm.is_nac() and dm.nac_method == "gonze":
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
            filename=output_filename,
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
        phonons = read_phonon_from_hdf5(
            phono3py.mesh_numbers, filename=input_filename, verbose=(log_level > 0)
        )
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


def main(**argparse_control):
    """Phono3py main part of command line interface."""
    # import warnings

    # warnings.simplefilter("error")
    load_phono3py_yaml = argparse_control.get("load_phono3py_yaml", False)

    if "args" in argparse_control:  # This is for pytest.
        args = argparse_control["args"]
        log_level = args.log_level
    else:
        args, log_level = start_phono3py(**argparse_control)

    if load_phono3py_yaml:
        input_filename = None
        output_filename = None
        output_yaml_filename = args.output_yaml_filename
    else:
        (input_filename, output_filename) = get_input_output_filenames_from_args(args)
        output_yaml_filename = None

    settings, confs_dict, cell_filename = read_phono3py_settings(
        args, argparse_control, log_level
    )

    if args.force_sets_to_forces_fc2_mode:
        create_FORCES_FC2_from_FORCE_SETS(log_level)
        if log_level:
            print_end_phono3py()
        sys.exit(0)
    if args.force_sets_mode:
        create_FORCE_SETS_from_FORCES_FCx(
            settings.phonon_supercell_matrix, input_filename, cell_filename, log_level
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

    cell_info = get_cell_info(settings, cell_filename, log_level)
    unitcell_filename = cell_info["optional_structure_info"][0]
    interface_mode = cell_info["interface_mode"]
    # ph3py_yaml = cell_info['phonopy_yaml']

    if run_mode is None:
        run_mode = get_run_mode(settings)

    ######################################################
    # Create supercells with displacements and then exit #
    ######################################################
    if not settings.use_pypolymlp:
        create_supercells_with_displacements(
            settings,
            cell_info,
            confs_dict,
            unitcell_filename,
            interface_mode,
            load_phono3py_yaml,
            symprec,
            log_level,
        )

    #######################
    # Initialize phono3py #
    #######################
    # updated_settings keys
    # ('sigmas', 'temperature_points', 'temperatures',
    #  'frequency_factor_to_THz', 'num_frequency_points',
    #  'frequency_step', 'frequency_scale_factor',
    #  'cutoff_frequency')
    ph3py, updated_settings = init_phono3py(
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
            input_filename,
            output_filename,
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
        vecs, _ = ph3py.primitive.get_smallest_vectors()
        dists = np.unique(
            np.round(np.linalg.norm(vecs @ ph3py.primitive.cell, axis=-1), decimals=1)
        )
        print("cutoff   memsize")
        print("------   -------")
        for cutoff in dists[1:] + 0.1:
            memsize, memsize2 = estimate_symfc_memory_usage(
                ph3py.supercell, ph3py.symmetry, cutoff
            )
            print(
                f"{cutoff:5.1f}  {memsize + memsize2:6.2f} GB "
                f"({memsize:.2f}+{memsize2:.2f})"
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

    #########################################################
    # Write ir-grid points and grid addresses and then exit #
    #########################################################
    if run_mode == "write_grid_info":
        ph3py.mesh_numbers = settings.mesh_numbers
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
        ph3py.mesh_numbers = settings.mesh_numbers
        grid_points = settings_to_grid_points(settings, ph3py.grid)
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
            ph3py,
            settings,
            cell_info["phonopy_yaml"],
            unitcell_filename,
            log_level,
            nac_factor=Hartree * Bohr,
            load_phonopy_yaml=load_phono3py_yaml,
        )

    ############
    # Datasets #
    ############
    if load_phono3py_yaml:
        assert ph3py.dataset is None
        assert ph3py.phonon_dataset is None
        load_dataset_and_phonon_dataset(
            ph3py,
            ph3py_yaml=cell_info["phonopy_yaml"],
            phono3py_yaml_filename=unitcell_filename,
            cutoff_pair_distance=settings.cutoff_pair_distance,
            calculator=interface_mode,
            log_level=log_level,
        )

    ###################
    # polynomial MLPs #
    ###################
    if load_phono3py_yaml and settings.use_pypolymlp:
        assert ph3py.mlp_dataset is None
        if ph3py.dataset is not None:
            ph3py.mlp_dataset = ph3py.dataset
            ph3py.dataset = None
        prepare_dataset = (
            settings.create_displacements or settings.random_displacements is not None
        )
        cutoff_pair_distance = get_cutoff_pair_distance(settings)
        run_pypolymlp_to_compute_forces(
            ph3py,
            mlp_params=settings.mlp_params,
            displacement_distance=settings.displacement_distance,
            number_of_snapshots=settings.random_displacements,
            random_seed=settings.random_seed,
            cutoff_pair_distance=cutoff_pair_distance,
            prepare_dataset=prepare_dataset,
            log_level=log_level,
        )

        if ph3py.dataset is not None:
            mlp_eval_filename = "phono3py_mlp_eval_dataset.yaml"
            if log_level:
                print(
                    "Dataset generated using MLPs was written in "
                    f'"{mlp_eval_filename}".'
                )
            ph3py.save(mlp_eval_filename)

        # pypolymlp dataset is stored in "polymlp.yaml" and stop here.
        if not prepare_dataset:
            if log_level:
                print(
                    "Generate displacements (--rd or -d) for proceeding to phonon "
                    "calculations."
                )
            finalize_phono3py(
                ph3py, confs_dict, log_level, filename=output_yaml_filename
            )

    ###################
    # Force constants #
    ###################
    if load_phono3py_yaml:
        assert ph3py.fc2 is None
        assert ph3py.fc3 is None
        _store_force_constants(ph3py, settings, log_level)
    else:
        try:
            create_phono3py_force_constants(
                ph3py,
                settings,
                ph3py_yaml=cell_info["phonopy_yaml"],
                phono3py_yaml_filename=unitcell_filename,
                calculator=interface_mode,
                input_filename=input_filename,
                output_filename=output_filename,
                log_level=log_level,
            )
        except ForceCalculatorRequiredError:
            _show_fc_calculator_not_found(log_level)

    ############################################
    # Phonon Gruneisen parameter and then exit #
    ############################################
    if settings.is_gruneisen:
        run_gruneisen_then_exit(ph3py, settings, output_filename, log_level)

    #################
    # Show settings #
    #################
    if log_level and run_mode is not None:
        show_phono3py_settings(ph3py, settings, updated_settings, log_level)

    ###########################
    # Joint DOS and then exit #
    ###########################
    if run_mode == "jdos":
        run_jdos_then_exit(
            ph3py, settings, updated_settings, output_filename, log_level
        )

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
        run_isotope_then_exit(ph3py, settings, updated_settings, log_level)

    ########################################
    # Initialize phonon-phonon interaction #
    ########################################
    if run_mode is not None:
        init_phph_interaction(
            ph3py,
            settings,
            updated_settings,
            input_filename,
            output_filename,
            log_level,
        )

    #######################################################
    # Run imaginary part of self energy of bubble diagram #
    #######################################################
    if run_mode == "imag_self_energy":
        ph3py.run_imag_self_energy(
            settings_to_grid_points(settings, ph3py.grid),
            updated_settings["temperature_points"],
            frequency_step=updated_settings["frequency_step"],
            num_frequency_points=updated_settings["num_frequency_points"],
            num_points_in_batch=updated_settings["num_points_in_batch"],
            scattering_event_class=settings.scattering_event_class,
            write_txt=True,
            write_gamma_detail=settings.write_gamma_detail,
            output_filename=output_filename,
        )

    #####################################################
    # Run frequency shift calculation of bubble diagram #
    #####################################################
    elif run_mode == "real_self_energy":
        ph3py.run_real_self_energy(
            settings_to_grid_points(settings, ph3py.grid),
            updated_settings["temperature_points"],
            frequency_step=updated_settings["frequency_step"],
            num_frequency_points=updated_settings["num_frequency_points"],
            write_txt=True,
            write_hdf5=True,
            output_filename=output_filename,
        )

    #######################################################
    # Run spectral function calculation of bubble diagram #
    #######################################################
    elif run_mode == "spectral_function":
        ph3py.run_spectral_function(
            settings_to_grid_points(settings, ph3py.grid),
            updated_settings["temperature_points"],
            frequency_step=updated_settings["frequency_step"],
            num_frequency_points=updated_settings["num_frequency_points"],
            num_points_in_batch=updated_settings["num_points_in_batch"],
            write_txt=True,
            write_hdf5=True,
            output_filename=output_filename,
        )

    ####################################
    # Run lattice thermal conductivity #
    ####################################
    elif run_mode == "conductivity-RTA" or run_mode == "conductivity-LBTE":
        grid_points = settings_to_grid_points(settings, ph3py.grid)
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
            input_filename=input_filename,
            output_filename=output_filename,
        )
    else:
        if log_level:
            print(
                "-" * 11
                + " None of ph-ph interaction calculation was performed. "
                + "-" * 11
            )

    finalize_phono3py(ph3py, confs_dict, log_level, filename=output_yaml_filename)
