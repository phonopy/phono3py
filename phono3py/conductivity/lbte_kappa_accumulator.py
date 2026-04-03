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

    def prepare(self, is_full_pp: bool = False) -> None:
        """Allocate global arrays before the grid-point accumulation loop.

        Parameters
        ----------
        is_full_pp : bool, optional
            Allocate averaged_pp_interaction array.  Default False.

        """
        self._solver.prepare(is_full_pp=is_full_pp)

    def accumulate(
        self,
        i_gp: int,
        collision_result: LBTECollisionResult,
        group_velocities: NDArray[np.double],
        heat_capacities: NDArray[np.double],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Store per-grid-point Stage 1 data.

        Parameters
        ----------
        i_gp : int
            Loop index over ir_grid_points (0-based).
        collision_result : LBTECollisionResult
            Result from LBTECollisionProvider.compute().
        group_velocities : NDArray[np.double]
            Group velocities at this grid point, shape (num_band0, 3).
        heat_capacities : NDArray[np.double]
            Mode heat capacities, shape (num_temp, num_band0).
        extra : dict or None, optional
            Plugin-specific data from the velocity provider.  Ignored by the
            standard LBTE accumulator; used by plugin accumulators (e.g.
            Wigner) to extract vm_by_vm, velocity_operator, etc.

        """
        self._solver.store(i_gp, collision_result, group_velocities, heat_capacities)

    def store_gamma_iso(self, i_gp: int, gamma_iso: NDArray[np.double]) -> None:
        """Store isotope scattering rate for one irreducible grid point.

        Parameters
        ----------
        i_gp : int
            Loop index over ir_grid_points (0-based).
        gamma_iso : NDArray[np.double]
            Isotope scattering rates, shape (num_sigma, num_band0).

        """
        self._solver.store_gamma_iso(i_gp, gamma_iso)

    def finalize(
        self,
        num_sampling_grid_points: int,
        *,
        suppress_kappa_log: bool = False,
    ) -> None:
        """Assemble collision matrix and compute LBTE thermal conductivity.

        Stage 2: combine diagonals, apply weights, symmetrize, solve for kappa.

        Parameters
        ----------
        num_sampling_grid_points : int
            Total number of sampling grid points (sum of k-star orders).
        suppress_kappa_log : bool, optional
            When True, skip the per-temperature kappa table log so that the
            caller (e.g. WignerLBTEKappaAccumulator) can print its own format
            after computing additional terms (Stage 3).  Default False.

        """
        self._solver.solve(
            num_sampling_grid_points, suppress_kappa_log=suppress_kappa_log
        )

    def get_main_diagonal(
        self, i_gp: int, i_sigma: int, i_temp: int
    ) -> NDArray[np.double]:
        """Return total scattering rate at a grid point.

        Returns the sum of ph-ph gamma, isotope gamma (if present), and
        boundary scattering (if present) at grid point i_gp for sigma i_sigma
        and temperature i_temp.  Shape is (num_band0,).

        Parameters
        ----------
        i_gp : int
            Grid point index (IR grid or reducible mesh index).
        i_sigma : int
            Sigma index.
        i_temp : int
            Temperature index.

        """
        return self._solver.get_main_diagonal(i_gp, i_sigma, i_temp)

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

    @property
    def kappa_RTA(self) -> NDArray[np.double]:
        """Return RTA thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._solver.kappa_RTA

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode LBTE kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._solver.mode_kappa

    @property
    def mode_kappa_RTA(self) -> NDArray[np.double]:
        """Return mode RTA kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._solver.mode_kappa_RTA

    @property
    def collision_matrix(self) -> NDArray[np.double] | None:
        """Return assembled collision matrix."""
        return self._solver.collision_matrix

    @collision_matrix.setter
    def collision_matrix(self, value: NDArray[np.double] | None) -> None:
        """Set collision matrix (used when reading from file)."""
        self._solver.collision_matrix = value

    @property
    def collision_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues of collision matrix."""
        return self._solver.collision_eigenvalues

    @property
    def gamma(self) -> NDArray[np.double]:
        """Return gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        return self._solver.gamma

    @gamma.setter
    def gamma(self, value: NDArray[np.double]) -> None:
        """Set gamma (used when reading from file)."""
        self._solver.gamma = value

    @property
    def gamma_iso(self) -> NDArray[np.double] | None:
        """Return isotope gamma, shape (num_sigma, num_gp, num_band0)."""
        return self._solver.gamma_iso

    @property
    def averaged_pp_interaction(self) -> NDArray[np.double] | None:
        """Return averaged ph-ph interaction, shape (num_gp, num_band0)."""
        return self._solver.averaged_pp_interaction

    @property
    def boundary_mfp(self) -> float | None:
        """Return boundary mean free path in micrometres."""
        return self._solver.boundary_mfp

    @property
    def mode_heat_capacities(self) -> NDArray[np.double]:
        """Return mode heat capacities, shape (num_temp, num_gp, num_band0)."""
        return self._solver.mode_heat_capacities

    @property
    def f_vectors(self) -> NDArray[np.double] | None:
        """Return f-vectors (experimental), shape (num_gp, num_band0, 3)."""
        return self._solver.f_vectors

    @property
    def mfp(self) -> NDArray[np.double] | None:
        """Return mean free path, shape (num_sigma, num_temp, num_gp, num_band0, 3)."""
        return self._solver.mfp

    @property
    def group_velocities(self) -> NDArray[np.double]:
        """Return group velocities, shape (num_gp, num_band0, 3)."""
        return self._solver.group_velocities

    @property
    def temperatures(self) -> NDArray[np.double]:
        """Return temperatures in Kelvin."""
        return self._solver.temperatures

    @temperatures.setter
    def temperatures(self, value: NDArray[np.double]) -> None:
        """Set temperatures and re-allocate all arrays via prepare()."""
        self._solver.temperatures = value
