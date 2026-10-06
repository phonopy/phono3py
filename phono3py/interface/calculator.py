# SPDX-License-Identifier: BSD-3-Clause
"""Utilities of calculator interfaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

calculator_info: dict[str, dict[str, dict[str, str]]] = {
    "abinit": {"option": {"name": "--abinit", "help": "Invoke Abinit mode"}},
    # 'aims': {'option': {'name': "--aims",
    #                     'help': "Invoke FHI-aims mode"}},
    # 'cp2k': {'option': {'name': "--cp2k",
    #                     'help': "Invoke CP2K mode"}},
    "crystal": {"option": {"name": "--crystal", "help": "Invoke CRYSTAL mode"}},
    # 'dftbp': {'option': {'name': "--dftb+",
    #                      'help': "Invoke dftb+ mode"}},
    # 'elk': {'option': {'name': "--elk",
    #                    'help': "Invoke elk mode"}},
    "lammps": {"option": {"name": "--lammps", "help": "Invoke Lammps mode"}},
    "qe": {"option": {"name": "--qe", "help": "Invoke Quantum espresso (QE) mode"}},
    # 'siesta': {'option': {'name': "--siesta",
    #                       'help': "Invoke Siesta mode"}},
    "turbomole": {"option": {"name": "--turbomole", "help": "Invoke TURBOMOLE mode"}},
    "vasp": {"option": {"name": "--vasp", "help": "Invoke Vasp mode"}},
    "wien2k": {"option": {"name": "--wien2k", "help": "Invoke Wien2k mode"}},
}


def get_default_displacement_distance(interface_mode: str | None) -> float:
    """Return default displacement distances for calculators."""
    if interface_mode in ("qe", "abinit", "turbomole"):
        displacement_distance = 0.06
    elif interface_mode == "crystal":
        displacement_distance = 0.03
    else:
        displacement_distance = 0.03
    return displacement_distance


def get_additional_info_to_write_supercells(
    interface_mode: str | None,
    supercell_matrix: NDArray[np.int64],
) -> dict[str, Any]:
    """Return additional information to write fc3-supercells for calculators.

    The fc3 supercell filename is left to each calculator default (as phonopy
    does), so "pre_filename" is not set here. Only crystal needs extra template
    information. See get_additional_info_to_write_fc2_supercells for why the fc2
    supercells override the filename instead.

    """
    additional_info: dict[str, Any] = {}
    if interface_mode == "crystal":
        additional_info["template_file"] = "TEMPLATE3"
        additional_info["supercell_matrix"] = supercell_matrix
    return additional_info


def get_additional_info_to_write_fc2_supercells(
    interface_mode: str | None,
    phonon_supercell_matrix: NDArray[np.int64],
    suffix: str = "fc2",
) -> dict[str, Any]:
    """Return additional information to write fc2-supercells for calculators.

    The fc2 supercells share the directory with the fc3 supercells, so
    "pre_filename" is overwritten here with a suffix to avoid filename
    collisions. The base name follows the calculator default convention:
    "POSCAR" for VASP and "supercell" for the others.

    """
    additional_info: dict[str, Any] = {}
    if interface_mode is None or interface_mode == "vasp":
        additional_info["pre_filename"] = "POSCAR_%s" % suffix.upper()
    else:
        additional_info["pre_filename"] = "supercell_%s" % suffix
    if interface_mode == "crystal":
        additional_info["template_file"] = "TEMPLATE"
        additional_info["supercell_matrix"] = phonon_supercell_matrix
    return additional_info
