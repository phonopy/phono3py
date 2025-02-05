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
from phonopy.interface.symfc import SymfcFCSolver
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry


def get_fc3(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    displacements: np.ndarray,
    forces: np.ndarray,
    fc_calculator: Optional[str] = None,
    fc_calculator_options: Optional[str] = None,
    is_compact_fc: bool = False,
    symmetry: Optional[Symmetry] = None,
    log_level: int = 0,
):
    """Calculate 2upercell 2nd and 3rd order force constants.

    Parameters
    ----------
    supercell : PhonopyAtoms
        Supercell
    primitive : Primitive
        Primitive cell
    displacements : array_like
        Displacements of atoms in supercell.
        shape=(num_snapshots, num_atoms, 3), dtype='double', order='C'
    forces : array_like
        Forces of atoms in supercell.
        shape=(num_snapshots, num_atoms, 3), dtype='double', order='C'
    fc_calculator : str, optional
        Currently only 'alm' is supported. Default is None, meaning invoking
        'alm'.
    fc_calculator_options : str, optional
        This is arbitrary string.
    log_level : integer or bool, optional
        Verbosity level. False or 0 means quiet. True or 1 means normal level
        of log to stdout. 2 gives verbose mode.

    Returns
    -------
    (fc2, fc3) : tuple[ndarray]
        2nd and 3rd order force constants.

    """
    if fc_calculator == "alm":
        from phonopy.interface.alm import run_alm

        fc = run_alm(
            supercell,
            primitive,
            displacements,
            forces,
            2,
            options=fc_calculator_options,
            is_compact_fc=is_compact_fc,
            log_level=log_level,
        )
        return fc[2], fc[3]
    elif fc_calculator == "symfc":
        from phonopy.interface.symfc import run_symfc

        fc = run_symfc(
            supercell,
            primitive,
            displacements,
            forces,
            orders=[2, 3],
            is_compact_fc=is_compact_fc,
            symmetry=symmetry,
            options=fc_calculator_options,
            log_level=log_level,
        )
        return fc[2], fc[3]
    else:
        msg = "Force constants calculator of %s was not found ." % fc_calculator
        raise RuntimeError(msg)


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
    basis_size = symfc_solver._symfc.estimate_basis_size(orders=[3])[3]
    memsize = basis_size**2 * 3 * 8 / 10**9
    memsize2 = len(supercell) * 3 * batch_size * basis_size * 8 / 10**9
    return memsize, memsize2
