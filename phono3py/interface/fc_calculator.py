"""Interfaces for force constants calculators."""

# Copyright (C) 2019 Atsushi Togo
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

from typing import Optional, Union

import numpy as np
from phonopy.interface.fc_calculator import FCSolver
from phonopy.interface.symfc import SymfcFCSolver
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry

from phono3py.phonon3.dataset import get_displacements_and_forces_fc3
from phono3py.phonon3.fc3 import get_fc3


def get_fc3_solver(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    dataset: dict,
    fc_calculator: Optional[str] = None,
    fc_calculator_options: Optional[str] = None,
    is_compact_fc: bool = False,
    symmetry: Optional[Symmetry] = None,
    log_level: int = 0,
) -> FC3Solver:
    """Return force constants solver for fc3.

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell
    primitive : Primitive
        Primitive cell
    dataset : dict, optional
        Dataset that contains displacements, forces, and optionally
        energies. Default is None.
    fc_calculator : str, optional
        Currently only 'alm' is supported. Default is None, meaning invoking
        'alm'.
    fc_calculator_options : str, optional
        This is arbitrary string.
    is_compact_fc : bool, optional
        If True, force constants are returned in the compact form.
    symmetry : Symmetry, optional
        Symmetry of supercell. This is used for the traditional and symfc FC
        solver. Default is None.
    log_level : integer or bool, optional
        Verbosity level. False or 0 means quiet. True or 1 means normal level
        of log to stdout. 2 gives verbose mode.

    Returns
    -------
    FC3Solver
        Force constants solver for fc3.

    """
    fc_solver_name = fc_calculator if fc_calculator is not None else "traditional"
    fc_solver = FC3Solver(
        fc_solver_name,
        supercell,
        symmetry=symmetry,
        dataset=dataset,
        is_compact_fc=is_compact_fc,
        primitive=primitive,
        orders=[2, 3],
        options=fc_calculator_options,
        log_level=log_level,
    )
    return fc_solver


def extract_fc2_fc3_calculators(
    fc_calculator: Optional[Union[str, dict]],
    order: int,
) -> Optional[Union[str, dict]]:
    """Extract fc_calculator and fc_calculator_options for fc2 and fc3.

    fc_calculator : str
        FC calculator. "|" separates fc2 and fc3. First and last
        parts separated correspond to fc2 and fc3 calculators, respectively.
    order : int = 2 or 3
        2 and 3 indicate fc2 and fc3, respectively.

    """
    if isinstance(fc_calculator, dict) or fc_calculator is None:
        return fc_calculator
    elif isinstance(fc_calculator, str):
        if "|" in fc_calculator:
            _fc_calculator = fc_calculator.split("|")[order - 2]
            if _fc_calculator == "":
                return None
            return _fc_calculator
        else:
            if fc_calculator.strip() == "":
                return None
            return fc_calculator
    else:
        raise RuntimeError("fc_calculator should be str, dict, or None.")


def update_cutoff_fc_calculator_options(
    fc_calc_opts: Optional[Union[str, dict]],
    cutoff_pair_distance: Optional[float],
) -> Optional[Union[str, dict]]:
    """Update fc_calculator_options with cutoff distances.

    Parameters
    ----------
    fc_calc_opts : str or dict
        FC calculator options.
    cutoff_pair_distance : float, optional
        Cutoff distance for pair interaction.

    """
    if cutoff_pair_distance is not None:
        if not isinstance(fc_calc_opts, (str, dict)) and fc_calc_opts is not None:
            raise RuntimeError("fc_calculator_options should be str, dict, or None.")

        if isinstance(fc_calc_opts, dict) and "cutoff" not in fc_calc_opts:
            fc_calc_opts["cutoff"] = float(cutoff_pair_distance)
        elif isinstance(fc_calc_opts, str) and "cutoff" not in fc_calc_opts:
            fc_calc_opts = f"{fc_calc_opts}, cutoff = {cutoff_pair_distance}"
        elif fc_calc_opts is None:
            fc_calc_opts = f"cutoff = {cutoff_pair_distance}"

    return fc_calc_opts


def estimate_symfc_memory_usage(
    supercell: PhonopyAtoms, symmetry: Symmetry, cutoff: float, batch_size: int = 100
):
    """Estimate memory usage to run symfc for fc3 with cutoff.

    Total memory usage is memsize + memsize2. These are separated because
    they behave differently with respect to cutoff distance.

    batch_size is hardcoded to 100 because it is so in symfc.

    """
    symfc_solver = SymfcFCSolver(supercell, symmetry, options={"cutoff": {3: cutoff}})
    basis_size = symfc_solver.estimate_basis_size(orders=[3])[3]
    memsize = basis_size**2 * 3 * 8 / 10**9
    memsize2 = len(supercell) * 3 * batch_size * basis_size * 8 / 10**9
    return memsize, memsize2


class FDFC3Solver:
    """Finite difference type force constants calculator.

    This is phono3py's traditional force constants calculator.

    """

    def __init__(
        self,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        symmetry: Symmetry,
        dataset: dict,
        is_compact_fc: bool = False,
        log_level: int = 0,  # currently not used
    ):
        self._fc2, self._fc3 = self._run(
            supercell,
            primitive,
            symmetry,
            dataset,
            is_compact_fc,
            log_level,
        )

    @property
    def force_constants(self) -> dict[int, np.ndarray]:
        """Return force constants.

        Returns
        -------
        dict[int, np.ndarray]
            Force constants with order as key.

        """
        return {2: self._fc2, 3: self._fc3}

    def _run(
        self,
        supercell: PhonopyAtoms,
        primitive: Primitive,
        symmetry: Symmetry,
        dataset: dict,
        is_compact_fc: bool,
        log_level: int,
    ):
        return get_fc3(
            supercell,
            primitive,
            dataset,
            symmetry,
            is_compact_fc=is_compact_fc,
            verbose=log_level > 0,
        )


class FC3Solver(FCSolver):
    """Force constants solver for fc3."""

    def _set_traditional_solver(self, solver_class: Optional[type] = FDFC3Solver):
        return super()._set_traditional_solver(solver_class=solver_class)

    def _get_displacements_and_forces(self):
        """Return displacements and forces for fc3."""
        return get_displacements_and_forces_fc3(self._dataset)
