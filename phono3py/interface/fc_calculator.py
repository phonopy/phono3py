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

from typing import Literal

import numpy as np
from phonopy.interface.fc_calculator import FCSolver, fc_calculator_names
from phonopy.interface.symfc import parse_symfc_options, update_symfc_cutoff_by_memsize
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry

from phono3py.phonon3.dataset import get_displacements_and_forces_fc3
from phono3py.phonon3.fc3 import get_fc3


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

    def _set_traditional_solver(self, solver_class: type | None = FDFC3Solver):
        return super()._set_traditional_solver(solver_class=solver_class)

    def _set_symfc_solver(self):
        return super()._set_symfc_solver(order=3)

    def _get_displacements_and_forces(self):
        """Return displacements and forces for fc3."""
        assert self._dataset is not None
        return get_displacements_and_forces_fc3(self._dataset)


def extract_fc2_fc3_calculators(
    fc_calculator: Literal["traditional", "symfc", "alm"] | str | None,
    order: int,
) -> Literal["traditional", "symfc", "alm"] | None:
    """Extract fc_calculator for fc2 and fc3.

    fc_calculator : str
        FC calculator. "|" separates fc2 and fc3. First and last
        parts separated correspond to fc2 and fc3 calculators, respectively.
    order : int = 2 or 3
        2 and 3 indicate fc2 and fc3, respectively.

    """
    if fc_calculator is None:
        return None
    else:
        _fc_calculator = _split_fc_calculator_str(fc_calculator, order)
        if _fc_calculator is None:
            return None
        fc_calculator_lower = _fc_calculator.lower()
        if fc_calculator_lower not in ("traditional", "symfc", "alm"):
            raise ValueError(
                f"Unknown fc_calculator: {_fc_calculator}. "
                "Available calculators are 'traditional', 'symfc', and 'alm'."
            )
        return fc_calculator_lower


def extract_fc2_fc3_calculators_options(
    fc_calculator_opts: str | None,
    order: int,
) -> str | None:
    """Extract fc_calculator_options for fc2 and fc3.

    fc_calculator_opts : str
        FC calculator options. "|" separates fc2 and fc3. First and last
        parts separated correspond to fc2 and fc3 calculators, respectively.
    order : int = 2 or 3
        2 and 3 indicate fc2 and fc3, respectively.

    """
    if fc_calculator_opts is None:
        return None
    else:
        _fc_calculator_opts = _split_fc_calculator_str(fc_calculator_opts, order)
        return _fc_calculator_opts


def _split_fc_calculator_str(fc_calculator: str, order: int) -> str | None:
    if "|" in fc_calculator:
        _fc_calculator = fc_calculator.split("|")[order - 2]
        if _fc_calculator == "":
            return None
    else:
        if fc_calculator.strip() == "":
            return None
        else:
            _fc_calculator = fc_calculator
    return _fc_calculator


def update_cutoff_fc_calculator_options(
    fc_calc_opts: str | None,
    cutoff_pair_distance: float | None,
) -> str | None:
    """Update fc_calculator_options with cutoff distances.

    Parameters
    ----------
    fc_calc_opts : str or None
        FC calculator options.
    cutoff_pair_distance : float, optional
        Cutoff distance for pair interaction.

    """
    if cutoff_pair_distance is not None:
        if isinstance(fc_calc_opts, str) and "cutoff" not in fc_calc_opts:
            fc_calc_opts = f"{fc_calc_opts}, cutoff = {cutoff_pair_distance}"
        elif fc_calc_opts is None:
            fc_calc_opts = f"cutoff = {cutoff_pair_distance}"

    return fc_calc_opts


def get_fc_calculator_params(
    fc_calculator: str | None,
    fc_calculator_options: str | None,
    cutoff_pair_distance: float | None,
    log_level: int = 0,
) -> tuple[str | None, str | None]:
    """Compile fc_calculator and fc_calculator_options from input settings."""
    _fc_calculator = None
    fc_calculator_list = []
    if fc_calculator is not None:
        for fc_calculatr_str in fc_calculator.split("|"):
            if fc_calculatr_str == "":  # No external calculator
                fc_calculator_list.append(fc_calculatr_str.lower())
            elif fc_calculatr_str.lower() in fc_calculator_names:
                fc_calculator_list.append(fc_calculatr_str.lower())
        if fc_calculator_list:
            _fc_calculator = "|".join(fc_calculator_list)

    _fc_calculator_options = fc_calculator_options
    if cutoff_pair_distance:
        if fc_calculator_list and fc_calculator_list[-1] in ("alm", "symfc"):
            if fc_calculator_list[-1] == "alm":
                cutoff_str = f"-1 {cutoff_pair_distance}"
            if fc_calculator_list[-1] == "symfc":
                cutoff_str = f"{cutoff_pair_distance}"
            _fc_calculator_options = _set_cutoff_in_fc_calculator_options(
                _fc_calculator_options,
                cutoff_str,
                log_level,
            )

    return _fc_calculator, _fc_calculator_options


def determine_cutoff_pair_distance(
    fc_calculator: str | None = None,
    fc_calculator_options: str | None = None,
    cutoff_pair_distance: float | None = None,
    symfc_memory_size: float | None = None,
    random_displacements: int | str | None = None,
    supercell: PhonopyAtoms | None = None,
    primitive: Primitive | None = None,
    symmetry: Symmetry | None = None,
    log_level: int = 0,
) -> float | None:
    """Determine cutoff pair distance for displacements."""
    _cutoff_pair_distance, _symfc_memory_size = _get_cutoff_pair_distance(
        fc_calculator,
        fc_calculator_options,
        cutoff_pair_distance,
        symfc_memory_size,
    )
    if random_displacements is not None and random_displacements != "auto":
        _symfc_memory_size = None
    if _symfc_memory_size is not None:
        if fc_calculator is None:
            pass
        elif fc_calculator != "symfc":
            raise RuntimeError(
                "Estimation of cutoff_pair_distance by memory size is only "
                "available for symfc calculator."
            )
        symfc_options = {"memsize": {3: _symfc_memory_size}}
        if supercell is None or primitive is None or symmetry is None:
            raise RuntimeError(
                "supercell, primitive, and symmetry are required to estimate "
                "cutoff_pair_distance by memory size."
            )
        update_symfc_cutoff_by_memsize(
            symfc_options, supercell, primitive, symmetry, verbose=log_level > 0
        )
        if symfc_options["cutoff"] is not None:
            _cutoff_pair_distance = symfc_options["cutoff"][3]
    return _cutoff_pair_distance


def _set_cutoff_in_fc_calculator_options(
    fc_calculator_options: str | None,
    cutoff_str: str,
    log_level: int,
):
    str_appended = f"cutoff={cutoff_str}"
    calc_opts = fc_calculator_options
    if calc_opts is None:
        calc_opts = "|"
    if "|" in calc_opts:
        calc_opts_fc2, calc_opts_fc3 = [v.strip() for v in calc_opts.split("|")][:2]
    else:
        calc_opts_fc2 = calc_opts
        calc_opts_fc3 = calc_opts

    if calc_opts_fc3 == "":
        calc_opts_fc3 += f"{str_appended}"
        if log_level:
            print(f'Set "{str_appended}" to fc_calculator_options for fc3.')
    elif "cutoff" not in calc_opts_fc3:
        calc_opts_fc3 += f", {str_appended}"
        if log_level:
            print(f'Appended "{str_appended}" to fc_calculator_options for fc3.')

    return f"{calc_opts_fc2}|{calc_opts_fc3}"


def _get_cutoff_pair_distance(
    fc_calculator: str | None,
    fc_calculator_options: str | None,
    cutoff_pair_distance: float | None,
    symfc_memory_size: float | None,
) -> tuple[float | None, float | None]:
    """Return cutoff_pair_distance from settings."""
    _, _fc_calculator_options = get_fc_calculator_params(
        fc_calculator,
        fc_calculator_options,
        cutoff_pair_distance,
    )
    symfc_options = parse_symfc_options(
        extract_fc2_fc3_calculators_options(_fc_calculator_options, 3), 3
    )

    _cutoff_pair_distance = cutoff_pair_distance
    cutoff = symfc_options.get("cutoff")
    if cutoff is not None:
        _cutoff_pair_distance = cutoff.get(3)

    _symfc_memory_size = symfc_memory_size
    memsize = symfc_options.get("memsize")
    if memsize is not None:
        _symfc_memory_size = memsize.get(3)

    return _cutoff_pair_distance, _symfc_memory_size
