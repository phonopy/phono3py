"""Utilities for lattice thermal conductivity calculation."""

# Copyright (C) 2022 Atsushi Togo
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

from typing import TYPE_CHECKING

from phono3py.file_IO import write_pp_to_hdf5
from phono3py.phonon3.interaction import Interaction
from phono3py.phonon3.triplets import get_all_triplets

if TYPE_CHECKING:
    from phono3py.conductivity.base import ConductivityBase


def select_colmat_solver(pinv_solver):
    """Return collision matrix solver id."""
    try:
        import phono3py._phono3py as phono3c

        default_solver = phono3c.default_colmat_solver()
    except ImportError:
        print("Phono3py C-routine is not compiled correctly.")
        default_solver = 4

    if not phono3c.include_lapacke():
        if pinv_solver in (1, 2, 6):
            raise RuntimeError(
                "Use pinv-solver 3, 4, or 5 because "
                "phono3py is not compiled with LAPACKE."
            )

    solver_numbers = (1, 2, 3, 4, 5, 6, 7)

    solver = pinv_solver
    if solver == 6:  # 6 must return 3 for not transposing unitary matrix.
        solver = 3
    if solver == 0:  # default solver
        if default_solver in (3, 4, 5):
            try:
                import scipy.linalg  # noqa F401
            except ImportError:
                solver = 1
            else:
                solver = default_solver
        else:
            solver = default_solver
    elif solver not in solver_numbers:
        solver = default_solver

    return solver


def write_pp_interaction(
    conductivity: "ConductivityBase",
    pp: Interaction,
    i,
    filename=None,
    compression="gzip",
):
    """Write ph-ph interaction strength in hdf5 file."""
    grid_point = conductivity.grid_points[i]
    sigmas = conductivity.sigmas
    sigma_cutoff = conductivity.sigma_cutoff_width
    mesh = conductivity.mesh_numbers
    triplets, weights, _, _ = pp.get_triplets_at_q()
    all_triplets = get_all_triplets(grid_point, pp.bz_grid)

    if len(sigmas) > 1:
        print("Multiple smearing parameters were given. The last one in ")
        print("ph-ph interaction calculations was written in the file.")

    write_pp_to_hdf5(
        mesh,
        pp=pp.interaction_strength,
        g_zero=pp.zero_value_positions,
        grid_point=grid_point,
        triplet=triplets,
        weight=weights,
        triplet_all=all_triplets,
        sigma=sigmas[-1],
        sigma_cutoff=sigma_cutoff,
        filename=filename,
        compression=compression,
    )
