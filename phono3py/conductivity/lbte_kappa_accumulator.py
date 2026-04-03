"""LBTEKappaAccumulator: thin delegation to CollisionMatrixSolver."""

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

from typing import Any

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.collision_matrix_solver import CollisionMatrixSolver
from phono3py.conductivity.grid_point_data import GridPointAggregates
from phono3py.conductivity.lbte_collision_provider import LBTECollisionResult


class LBTEKappaAccumulator:
    """Assemble global collision matrix and compute LBTE thermal conductivity.

    This is Stage 2 of the two-stage LBTE design.  Stage 1 (per-grid-point)
    is handled by LBTECollisionProvider.  LBTECalculator calls accumulate()
    once per irreducible grid point and then finalize() to assemble the full
    collision matrix, solve it, and compute kappa and kappa_RTA.

    Internally delegates all work to CollisionMatrixSolver.

    Parameters
    ----------
    solver : CollisionMatrixSolver
        Pre-configured collision matrix solver.

    """

    def __init__(self, solver: CollisionMatrixSolver) -> None:
        """Init method."""
        self._solver = solver

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Allocate global arrays before the grid-point accumulation loop."""
        self._solver.prepare()

    def accumulate(
        self,
        i_gp: int,
        collision_result: LBTECollisionResult,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Store per-grid-point Stage 1 data.

        Parameters
        ----------
        i_gp : int
            Loop index over ir_grid_points (0-based).
        collision_result : LBTECollisionResult
            Result from LBTECollisionProvider.compute().
        extra : dict or None, optional
            Plugin-specific data from the velocity provider.  Ignored by the
            standard LBTE accumulator; used by plugin accumulators (e.g.
            Wigner) to extract vm_by_vm, velocity_operator, etc.

        """
        self._solver.store(i_gp, collision_result)

    def finalize(
        self,
        aggregates: GridPointAggregates,
        *,
        suppress_kappa_log: bool = False,
    ) -> None:
        """Assemble collision matrix and compute LBTE thermal conductivity.

        Stage 2: combine diagonals, apply weights, symmetrize, solve for kappa.

        Parameters
        ----------
        aggregates : GridPointAggregates
            Aggregated per-grid-point data from the calculator.
        suppress_kappa_log : bool, optional
            When True, skip the per-temperature kappa table log so that the
            caller (e.g. WignerLBTEKappaAccumulator) can print its own format
            after computing additional terms (Stage 3).  Default False.

        """
        self._solver.solve(aggregates, suppress_kappa_log=suppress_kappa_log)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def solver(self) -> CollisionMatrixSolver:
        """Return the underlying CollisionMatrixSolver."""
        return self._solver

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return LBTE thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._solver.kappa
