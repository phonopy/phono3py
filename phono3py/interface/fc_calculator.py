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
    """Supercell 2nd order force constants (fc2) are calculated.

    The expected shape of supercell fc3 to be returned is
        (len(atom_list), num_atoms, num_atom, 3, 3, 3),
    where atom_list is either all atoms in primitive cell or supercell,
    which is chosen by is_compact_fc=True for primitive cell and False for
    supercell.

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
    fc3 : ndarray
        3rd order force constants.
        shape=(len(atom_list), num_atoms, num_atoms, 3, 3, 3)
        dtype='double', order='C'.
        Here atom_list is either all atoms in primitive cell or supercell,
        which is chosen by is_compact_fc=True for primitive cell and False
        for supercell.

    """
    if fc_calculator == "alm":
        from phono3py.interface.alm import get_fc3 as get_fc3_alm

        return get_fc3_alm(
            supercell,
            primitive,
            displacements,
            forces,
            options=fc_calculator_options,
            is_compact_fc=is_compact_fc,
            log_level=log_level,
        )
    elif fc_calculator == "symfc":
        from phonopy.interface.symfc import run_symfc

        return run_symfc(
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
    else:
        msg = "Force constants calculator of %s was not found ." % fc_calculator
        raise RuntimeError(msg)


def extract_fc2_fc3_calculators(fc_calculator: Optional[Union[str, dict]], order: int):
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
        raise RuntimeError("fc_calculator should be str or dict.")
