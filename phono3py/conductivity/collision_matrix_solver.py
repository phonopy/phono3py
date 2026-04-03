"""CollisionMatrixSolver: global collision matrix assembly and LBTE solve.

Extracted from LBTEKappaAccumulator to serve as a shared utility for all
LBTE-based accumulators (standard, Wigner, Kubo).  Each accumulator composes
a solver instance and calls store() / solve() rather than inheriting or
delegating to an inner accumulator.

"""

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

import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.context import ConductivityContext
from phono3py.conductivity.lbte_collision_provider import LBTECollisionResult
from phono3py.conductivity.utils import (
    diagonalize_collision_matrix,
    select_colmat_solver,
)
from phono3py.phonon.grid import get_grid_points_by_rotations


@dataclass
class LBTESolveResult:
    """Result of CollisionMatrixSolver.solve().

    Attributes
    ----------
    kappa : NDArray[np.double]
        LBTE thermal conductivity, shape (num_sigma, num_temp, 6).
    kappa_RTA : NDArray[np.double]
        RTA thermal conductivity, shape (num_sigma, num_temp, 6).
    mode_kappa : NDArray[np.double]
        Mode LBTE kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6).
    mode_kappa_RTA : NDArray[np.double]
        Mode RTA kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6).
    collision_eigenvalues : NDArray[np.double] | None
        Eigenvalues of collision matrix.
    f_vectors : NDArray[np.double]
        f-vectors, shape (num_gp, num_band0, 3).
    mfp : NDArray[np.double]
        Mean free path, shape (num_sigma, num_temp, num_gp, num_band0, 3).

    """

    kappa: NDArray[np.double]
    kappa_RTA: NDArray[np.double]
    mode_kappa: NDArray[np.double]
    mode_kappa_RTA: NDArray[np.double]
    collision_eigenvalues: NDArray[np.double] | None
    f_vectors: NDArray[np.double]
    mfp: NDArray[np.double]


class CollisionMatrixSolver:
    """Assemble global collision matrix and compute LBTE thermal conductivity.

    This class holds the collision matrix assembly, in-place operations
    (symmetrization, diagonalization, degeneracy averaging, symmetry
    expansion), and kappa computation.  It is used by LBTEKappaAccumulator
    and the Wigner / Kubo LBTE accumulators.

    Parameters
    ----------
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, configuration).
    conversion_factor : float
        Unit conversion factor for thermal conductivity.
    is_reducible_collision_matrix : bool, optional
        Use the full reducible (non-symmetry-reduced) collision matrix.
        Default False.
    is_kappa_star : bool, optional
        Use k-star symmetry averaging.  Default True.
    solve_collective_phonon : bool, optional
        Use Chaput collective-phonon method (not supported; must be False).
        Default False.
    pinv_cutoff : float, optional
        Eigenvalue cutoff for pseudo-inversion.  Default 1e-8.
    pinv_solver : int, optional
        Solver selection index.  Default 0.
    pinv_method : int, optional
        Pseudo-inverse criterion (0=abs(eig)<cutoff, 1=eig<cutoff).
        Default 0.
    lang : {"C", "Python"}, optional
        Backend for C-extension operations.  Default "C".
    log_level : int, optional
        Verbosity level.  Default 0.

    """

    def __init__(
        self,
        context: ConductivityContext,
        conversion_factor: float,
        is_reducible_collision_matrix: bool = False,
        is_kappa_star: bool = True,
        solve_collective_phonon: bool = False,
        pinv_cutoff: float = 1.0e-8,
        pinv_solver: int = 0,
        pinv_method: int = 0,
        lang: Literal["C", "Python"] = "C",
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._context = context
        self._conversion_factor = conversion_factor
        self._is_reducible_collision_matrix = is_reducible_collision_matrix
        self._is_kappa_star = is_kappa_star
        self._solve_collective_phonon = solve_collective_phonon
        self._pinv_cutoff = pinv_cutoff
        self._pinv_solver = pinv_solver
        self._pinv_method = pinv_method
        self._lang: Literal["C", "Python"] = lang
        self._log_level = log_level

        # Set in prepare().
        self._is_full_pp: bool

        # Allocated in prepare().
        self._gamma: NDArray[np.double]
        self._gamma_iso: NDArray[np.double] | None = None
        self._gv: NDArray[np.double]
        self._cv: NDArray[np.double]
        self._averaged_pp_interaction: NDArray[np.double] | None = None
        self._collision_matrix: NDArray[np.double] | None = None
        self._collision_eigenvalues: NDArray[np.double] | None = None
        self._f_vectors: NDArray[np.double] | None = None
        self._mfp: NDArray[np.double] | None = None

        # Allocated in prepare() and filled in solve().
        self._kappa: NDArray[np.double]
        self._kappa_RTA: NDArray[np.double]
        self._mode_kappa: NDArray[np.double]
        self._mode_kappa_RTA: NDArray[np.double]

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
        self._is_full_pp = is_full_pp
        num_sigma = len(self._context.sigmas)
        num_temp = len(self._context.temperatures)
        num_band0 = len(self._context.band_indices)
        num_band = self._context.frequencies.shape[1]
        num_ir_gp = len(self._context.ir_grid_points)

        if self._is_reducible_collision_matrix:
            num_gp = int(np.prod(self._context.mesh_numbers))
        else:
            num_gp = num_ir_gp

        self._gamma = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0), dtype="double", order="C"
        )
        self._gv = np.zeros((num_gp, num_band0, 3), dtype="double", order="C")
        self._cv = np.zeros((num_temp, num_gp, num_band0), dtype="double", order="C")
        self._f_vectors = np.zeros((num_gp, num_band0, 3), dtype="double", order="C")
        self._mfp = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, 3),
            dtype="double",
            order="C",
        )

        self._kappa = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._kappa_RTA = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._mode_kappa = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, 6),
            dtype="double",
            order="C",
        )
        self._mode_kappa_RTA = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, 6),
            dtype="double",
            order="C",
        )

        if is_full_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_gp, num_band0), dtype="double", order="C"
            )

        if self._is_reducible_collision_matrix:
            self._collision_matrix = np.zeros(
                (
                    num_sigma,
                    num_temp,
                    num_gp,
                    num_band0,
                    num_gp,
                    num_band,
                ),
                dtype="double",
                order="C",
            )
            self._collision_eigenvalues = np.zeros(
                (num_sigma, num_temp, num_gp * num_band),
                dtype="double",
                order="C",
            )
        else:
            self._collision_matrix = np.zeros(
                (
                    num_sigma,
                    num_temp,
                    num_ir_gp,
                    num_band0,
                    3,
                    num_ir_gp,
                    num_band,
                    3,
                ),
                dtype="double",
                order="C",
            )
            self._collision_eigenvalues = np.zeros(
                (num_sigma, num_temp, num_ir_gp * num_band * 3),
                dtype="double",
                order="C",
            )

    def store(
        self,
        i_gp: int,
        collision_result: LBTECollisionResult,
        group_velocities: NDArray[np.double],
        heat_capacities: NDArray[np.double],
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

        """
        assert self._collision_matrix is not None

        if self._is_reducible_collision_matrix:
            ir_gp = self._context.ir_grid_points[i_gp]
            i_data = int(self._context.bz_grid.bzg2grg[ir_gp])
        else:
            i_data = i_gp

        self._gamma[:, :, i_data, :] = collision_result.gamma
        self._collision_matrix[:, :, i_data, :] = collision_result.collision_row
        self._gv[i_data] = group_velocities
        self._cv[:, i_data, :] = heat_capacities

        if (
            collision_result.averaged_pp is not None
            and self._averaged_pp_interaction is not None
        ):
            self._averaged_pp_interaction[i_data] = collision_result.averaged_pp

    def store_gamma_iso(self, i_gp: int, gamma_iso: NDArray[np.double]) -> None:
        """Store isotope scattering rate for one irreducible grid point.

        Parameters
        ----------
        i_gp : int
            Loop index over ir_grid_points (0-based).
        gamma_iso : NDArray[np.double]
            Isotope scattering rates, shape (num_sigma, num_band0).

        """
        if self._gamma_iso is None:
            num_sigma = len(self._context.sigmas)
            num_band0 = len(self._context.band_indices)
            if self._is_reducible_collision_matrix:
                num_gp = int(np.prod(self._context.mesh_numbers))
            else:
                num_gp = len(self._context.ir_grid_points)
            self._gamma_iso = np.zeros(
                (num_sigma, num_gp, num_band0), dtype="double", order="C"
            )

        if self._is_reducible_collision_matrix:
            ir_gp = self._context.ir_grid_points[i_gp]
            i_data = int(self._context.bz_grid.bzg2grg[ir_gp])
        else:
            i_data = i_gp
        self._gamma_iso[:, i_data, :] = gamma_iso

    def solve(
        self,
        num_sampling_grid_points: int,
        *,
        suppress_kappa_log: bool = False,
    ) -> LBTESolveResult:
        """Assemble collision matrix and compute LBTE thermal conductivity.

        Stage 2: combine diagonals, apply weights, symmetrize, solve for
        kappa.

        Parameters
        ----------
        num_sampling_grid_points : int
            Total number of sampling grid points (sum of k-star orders).
        suppress_kappa_log : bool, optional
            When True, skip the per-temperature kappa table log so that the
            caller (e.g. WignerLBTEKappaAccumulator) can print its own format
            after computing additional terms (Stage 3).  The sigma header and
            diagonalize output are still printed.  Default False.

        Returns
        -------
        LBTESolveResult

        """
        assert self._collision_matrix is not None
        if self._log_level:
            print(f"- Collision matrix shape {self._collision_matrix.shape}")

        weights = self._prepare_collision_matrix_by_type()
        self._set_kappa_and_rta_at_sigmas(
            num_sampling_grid_points,
            weights,
            suppress_kappa_log=suppress_kappa_log,
        )

        return LBTESolveResult(
            kappa=self._kappa,
            kappa_RTA=self._kappa_RTA,
            mode_kappa=self._mode_kappa,
            mode_kappa_RTA=self._mode_kappa_RTA,
            collision_eigenvalues=self._collision_eigenvalues,
            f_vectors=self._f_vectors,  # type: ignore[arg-type]
            mfp=self._mfp,  # type: ignore[arg-type]
        )

    def get_main_diagonal(
        self, i_gp: int, i_sigma: int, i_temp: int
    ) -> NDArray[np.double]:
        """Return total scattering rate at a grid point.

        Returns the sum of ph-ph gamma, isotope gamma (if present), and
        boundary scattering (if present) at grid point i_gp for sigma
        i_sigma and temperature i_temp.  Shape is (num_band0,).

        Parameters
        ----------
        i_gp : int
            Grid point index (IR grid or reducible mesh index).
        i_sigma : int
            Sigma index.
        i_temp : int
            Temperature index.

        """
        return self._get_main_diagonal(i_gp, i_sigma, i_temp)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return LBTE thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._kappa

    @property
    def kappa_RTA(self) -> NDArray[np.double]:
        """Return RTA thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._kappa_RTA

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode LBTE kappa."""
        return self._mode_kappa

    @property
    def mode_kappa_RTA(self) -> NDArray[np.double]:
        """Return mode RTA kappa."""
        return self._mode_kappa_RTA

    @property
    def collision_matrix(self) -> NDArray[np.double] | None:
        """Return assembled collision matrix."""
        return self._collision_matrix

    @collision_matrix.setter
    def collision_matrix(self, value: NDArray[np.double] | None) -> None:
        """Set collision matrix (used when reading from file)."""
        self._collision_matrix = value

    @property
    def collision_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues of collision matrix."""
        return self._collision_eigenvalues

    @property
    def gamma(self) -> NDArray[np.double]:
        """Return gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: NDArray[np.double]) -> None:
        """Set gamma (used when reading from file)."""
        self._gamma = value

    @property
    def gamma_iso(self) -> NDArray[np.double] | None:
        """Return isotope gamma, shape (num_sigma, num_gp, num_band0)."""
        return self._gamma_iso

    @property
    def averaged_pp_interaction(self) -> NDArray[np.double] | None:
        """Return averaged ph-ph interaction, shape (num_gp, num_band0)."""
        return self._averaged_pp_interaction

    @property
    def boundary_mfp(self) -> float | None:
        """Return boundary mean free path in micrometres."""
        return self._context.boundary_mfp

    @property
    def mode_heat_capacities(self) -> NDArray[np.double]:
        """Return mode heat capacities, shape (num_temp, num_gp, num_band0)."""
        return self._cv

    @property
    def f_vectors(self) -> NDArray[np.double] | None:
        """Return f-vectors, shape (num_gp, num_band0, 3)."""
        return self._f_vectors

    @property
    def mfp(self) -> NDArray[np.double] | None:
        """Return mean free path."""
        return self._mfp

    @property
    def group_velocities(self) -> NDArray[np.double]:
        """Return group velocities, shape (num_gp, num_band0, 3)."""
        return self._gv

    @property
    def temperatures(self) -> NDArray[np.double]:
        """Return temperatures in Kelvin."""
        return self._context.temperatures

    @temperatures.setter
    def temperatures(self, value: NDArray[np.double]) -> None:
        """Set temperatures and re-allocate all arrays via prepare()."""
        self._context.temperatures = np.asarray(value, dtype="double")
        self.prepare(is_full_pp=self._is_full_pp)

    # ------------------------------------------------------------------
    # Collision matrix preparation
    # ------------------------------------------------------------------

    def _prepare_collision_matrix_by_type(self) -> NDArray[np.double]:
        if self._is_reducible_collision_matrix:
            return self._prepare_reducible_collision_matrix()
        return self._prepare_ir_collision_matrix()

    def _prepare_ir_collision_matrix(self) -> NDArray[np.double]:
        self._combine_collisions()
        weights = self._apply_ir_collision_weights()
        self._average_collision_matrix_by_degeneracy()
        self._symmetrize_collision_matrix()
        return weights

    def _prepare_reducible_collision_matrix(self) -> NDArray[np.double]:
        if self._is_kappa_star:
            self._expand_reducible_collision_matrix_by_symmetry()
        self._combine_reducible_collisions()
        weights = self._get_reducible_collision_weights()
        self._symmetrize_collision_matrix()
        return weights

    # ------------------------------------------------------------------
    # Combine diagonals into off-diagonal rows
    # ------------------------------------------------------------------

    def _combine_collisions(self) -> None:
        """Add main diagonal elements to the IR collision matrix."""
        for j, k in self._iter_sigma_temp():
            for i_irgp, main_diag, rotation in self._iter_ir_diagonal_entries(j, k):
                self._add_main_diagonal_to_ir_collision(
                    i_sigma=j,
                    i_temp=k,
                    i_irgp=i_irgp,
                    main_diagonal=main_diag,
                    rotation=rotation,
                )

    def _combine_reducible_collisions(self) -> None:
        """Add main diagonal elements to the reducible collision matrix."""
        num_mesh_points = int(np.prod(self._context.mesh_numbers))
        for j, k in self._iter_sigma_temp():
            for i_mesh in range(num_mesh_points):
                main_diag = self._get_main_diagonal(i_mesh, j, k)
                self._add_main_diagonal_to_reducible_collision(
                    i_sigma=j,
                    i_temp=k,
                    i_mesh=i_mesh,
                    main_diagonal=main_diag,
                )

    def _iter_sigma_temp(self) -> Iterator[tuple[int, int]]:
        return np.ndindex(  # type: ignore[return-value]
            (len(self._context.sigmas), len(self._context.temperatures))
        )

    def _iter_ir_diagonal_entries(
        self, i_sigma: int, i_temp: int
    ) -> Iterator[tuple[int, NDArray[np.double], NDArray[np.double]]]:
        assert self._context.rot_grid_points is not None
        for i_irgp, ir_gp in enumerate(self._context.ir_grid_points):
            rot_gps = self._context.rot_grid_points[i_irgp]
            for rotation, rotated_gp in zip(
                self._context.rotations_cartesian, rot_gps, strict=True
            ):
                if ir_gp != rotated_gp:
                    continue
                main_diag = self._get_main_diagonal(i_irgp, i_sigma, i_temp)
                yield i_irgp, main_diag, rotation

    def _add_main_diagonal_to_ir_collision(
        self,
        *,
        i_sigma: int,
        i_temp: int,
        i_irgp: int,
        main_diagonal: NDArray[np.double],
        rotation: NDArray[np.double],
    ) -> None:
        assert self._collision_matrix is not None
        for ll, diag in enumerate(main_diagonal):
            self._collision_matrix[i_sigma, i_temp, i_irgp, ll, :, i_irgp, ll, :] += (
                diag * rotation
            )

    def _add_main_diagonal_to_reducible_collision(
        self,
        *,
        i_sigma: int,
        i_temp: int,
        i_mesh: int,
        main_diagonal: NDArray[np.double],
    ) -> None:
        assert self._collision_matrix is not None
        for ll, diag in enumerate(main_diagonal):
            self._collision_matrix[i_sigma, i_temp, i_mesh, ll, i_mesh, ll] += diag

    def _get_main_diagonal(self, i: int, j: int, k: int) -> NDArray[np.double]:
        """Return main diagonal of collision matrix at grid point i.

        Parameters
        ----------
        i : int
            Grid point index (IR grid or mesh index).
        j : int
            Sigma index.
        k : int
            Temperature index.

        """
        main_diagonal = self._gamma[j, k, i].copy()
        if self._gamma_iso is not None:
            main_diagonal += self._gamma_iso[j, i]
        if self._context.boundary_mfp is not None:
            main_diagonal += self._get_boundary_scattering(i)
        return main_diagonal

    def _get_boundary_scattering(self, i_gp: int) -> NDArray[np.double]:
        num_band = self._context.frequencies.shape[1]
        g_boundary = np.zeros(num_band, dtype="double")
        assert self._context.boundary_mfp is not None
        for ll in range(num_band):
            g_boundary[ll] = (
                np.linalg.norm(self._gv[i_gp, ll])
                * get_physical_units().Angstrom
                * 1e6
                / (4 * np.pi * self._context.boundary_mfp)
            )
        return g_boundary

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    def _apply_ir_collision_weights(self) -> NDArray[np.double]:
        return self._multiply_weights_to_collisions()

    def _get_reducible_collision_weights(self) -> NDArray[np.double]:
        return np.ones(int(np.prod(self._context.mesh_numbers)), dtype="double")

    def _multiply_weights_to_collisions(self) -> NDArray[np.double]:
        assert self._collision_matrix is not None
        weights = self._get_ir_weights()
        for i, w_i in enumerate(weights):
            for j, w_j in enumerate(weights):
                self._collision_matrix[:, :, i, :, :, j, :, :] *= w_i * w_j
        return weights

    def _get_ir_weights(self) -> NDArray[np.double]:
        """Return sqrt(g_k / |g|) weights for irreducible grid points.

        g_k : number of arms of k-star at each ir-qpoint.
        |g| : order of crystallographic point group.

        """
        assert self._context.rot_grid_points is not None
        weights = np.zeros(len(self._context.rot_grid_points), dtype="double")
        for i, r_gps in enumerate(self._context.rot_grid_points):
            weights[i] = np.sqrt(len(np.unique(r_gps)))
            sym_broken = False
            for gp in np.unique(r_gps):
                if len(np.where(r_gps == gp)[0]) != (
                    self._context.rot_grid_points.shape[1] // len(np.unique(r_gps))
                ):
                    sym_broken = True
            if sym_broken:
                print("=" * 26 + " Warning " + "=" * 26)
                print("Symmetry of grid is broken.")
        return weights / np.sqrt(self._context.rot_grid_points.shape[1])

    # ------------------------------------------------------------------
    # Degeneracy averaging
    # ------------------------------------------------------------------

    def _average_collision_matrix_by_degeneracy(self) -> None:
        """Average collision matrix elements within degenerate phonon subspaces."""
        start = time.time()
        if self._log_level:
            sys.stdout.write(
                "- Averaging collision matrix elements by phonon degeneracy "
            )
            sys.stdout.flush()

        assert self._collision_matrix is not None
        col_mat = self._collision_matrix
        self._average_colmat_rows_by_degeneracy(col_mat)
        self._average_colmat_cols_by_degeneracy(col_mat)

        if self._log_level:
            print("[%.3fs]" % (time.time() - start))
            sys.stdout.flush()

    def _average_colmat_rows_by_degeneracy(self, col_mat: NDArray[np.double]) -> None:
        assert self._context.frequencies is not None
        for i, gp in enumerate(self._context.ir_grid_points):
            freqs = self._context.frequencies[gp]
            for dset in degenerate_sets(freqs):
                bi_set = self._get_bi_set(freqs, dset)
                if self._is_reducible_collision_matrix:
                    i_data = int(self._context.bz_grid.bzg2grg[gp])
                    sum_col = col_mat[:, :, i_data, bi_set, :, :].sum(axis=2) / len(
                        bi_set
                    )
                    for j in bi_set:
                        col_mat[:, :, i_data, j, :, :] = sum_col
                else:
                    sum_col = col_mat[:, :, i, bi_set, :, :, :, :].sum(axis=2) / len(
                        bi_set
                    )
                    for j in bi_set:
                        col_mat[:, :, i, j, :, :, :, :] = sum_col

    def _average_colmat_cols_by_degeneracy(self, col_mat: NDArray[np.double]) -> None:
        assert self._context.frequencies is not None
        for i, gp in enumerate(self._context.ir_grid_points):
            freqs = self._context.frequencies[gp]
            for dset in degenerate_sets(freqs):
                bi_set = self._get_bi_set(freqs, dset)
                if self._is_reducible_collision_matrix:
                    i_data = int(self._context.bz_grid.bzg2grg[gp])
                    sum_col = col_mat[:, :, :, :, i_data, bi_set].sum(axis=4) / len(
                        bi_set
                    )
                    for j in bi_set:
                        col_mat[:, :, :, :, i_data, j] = sum_col
                else:
                    sum_col = col_mat[:, :, :, :, :, i, bi_set, :].sum(axis=5) / len(
                        bi_set
                    )
                    for j in bi_set:
                        col_mat[:, :, :, :, :, i, j, :] = sum_col

    @staticmethod
    def _get_bi_set(freqs: NDArray[np.double], dset: list[int]) -> list[int]:
        return [j for j in range(len(freqs)) if j in dset]

    # ------------------------------------------------------------------
    # Symmetrization
    # ------------------------------------------------------------------

    def _symmetrize_collision_matrix(self) -> None:
        """Symmetrize collision matrix as (Omega + Omega^T) / 2."""
        start = time.time()
        if self._can_use_builtin_symmetrizer():
            if self._log_level:
                sys.stdout.write("- Making collision matrix symmetric (built-in) ")
                sys.stdout.flush()
            import phono3py._phono3py as phono3c

            phono3c.symmetrize_collision_matrix(self._collision_matrix)
        else:
            if self._log_level:
                sys.stdout.write("- Making collision matrix symmetric (numpy) ")
                sys.stdout.flush()
            assert self._collision_matrix is not None
            size = self._get_symmetrization_size()
            for i in range(self._collision_matrix.shape[0]):
                for j in range(self._collision_matrix.shape[1]):
                    col_mat = self._collision_matrix[i, j].reshape(size, size)
                    col_mat += col_mat.T
                    col_mat /= 2

        if self._log_level:
            print("[%.3fs]" % (time.time() - start))
            sys.stdout.flush()

    def _can_use_builtin_symmetrizer(self) -> bool:
        try:
            import phono3py._phono3py  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_symmetrization_size(self) -> int:
        assert self._collision_matrix is not None
        if self._is_reducible_collision_matrix:
            return int(np.prod(self._collision_matrix.shape[2:4]))
        return int(np.prod(self._collision_matrix.shape[2:5]))

    # ------------------------------------------------------------------
    # Reducible: expand by symmetry
    # ------------------------------------------------------------------

    def _expand_reducible_collision_matrix_by_symmetry(self) -> None:
        self._average_collision_matrix_by_degeneracy()
        ir_gr_gps, rot_gps = self._get_reducible_rotation_maps()
        self._expand_reducible_collisions(ir_gr_gps, rot_gps)
        self._expand_local_values(ir_gr_gps, rot_gps)

    def _get_reducible_rotation_maps(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        num_mesh_points = int(np.prod(self._context.mesh_numbers))
        num_rot = len(self._context.rotations_cartesian)
        rot_grid_points = np.zeros((num_rot, num_mesh_points), dtype="int64")
        ir_gr_grid_points = np.array(
            self._context.bz_grid.bzg2grg[self._context.ir_grid_points],
            dtype="int64",
        )
        for i in range(num_mesh_points):
            rot_grid_points[:, i] = self._context.bz_grid.bzg2grg[
                get_grid_points_by_rotations(
                    self._context.bz_grid.grg2bzg[i],
                    self._context.bz_grid,
                )
            ]
        return ir_gr_grid_points, rot_grid_points

    def _expand_reducible_collisions(
        self,
        ir_gr_grid_points: NDArray[np.int64],
        rot_grid_points: NDArray[np.int64],
    ) -> None:
        """Fill full collision matrix by symmetry."""
        assert self._collision_matrix is not None
        start = time.time()
        if self._log_level:
            sys.stdout.write("- Expanding properties to all grid points ")
            sys.stdout.flush()

        if self._lang == "C":
            import phono3py._phono3py as phono3c

            phono3c.expand_collision_matrix(
                self._collision_matrix,
                ir_gr_grid_points,
                rot_grid_points,
            )
        else:
            num_mesh_points = int(np.prod(self._context.mesh_numbers))
            colmat = self._collision_matrix
            for ir_gp in ir_gr_grid_points:
                multi = (rot_grid_points[:, ir_gp] == ir_gp).sum()
                colmat_irgp = colmat[:, :, ir_gp, :, :, :].copy()
                colmat_irgp /= multi
                colmat[:, :, ir_gp, :, :, :] = 0
                for j, _ in enumerate(self._context.rotations_cartesian):
                    gp_r = rot_grid_points[j, ir_gp]
                    for k in range(num_mesh_points):
                        gp_c = rot_grid_points[j, k]
                        colmat[:, :, gp_r, :, gp_c, :] += colmat_irgp[:, :, :, k, :]

        if self._log_level:
            print("[%.3fs]" % (time.time() - start))
            sys.stdout.flush()

    def _expand_local_values(
        self,
        ir_gr_grid_points: NDArray[np.int64],
        rot_grid_points: NDArray[np.int64],
    ) -> None:
        """Expand gv, cv, gamma to all mesh grid points by symmetry."""
        for ir_gp in ir_gr_grid_points:
            multi = (rot_grid_points[:, ir_gp] == ir_gp).sum()
            gv_irgp = self._gv[ir_gp].copy()
            cv_irgp = self._cv[:, ir_gp, :].copy()
            gamma_irgp = self._gamma[:, :, ir_gp, :].copy()
            self._gv[ir_gp] = 0
            self._cv[:, ir_gp, :] = 0
            self._gamma[:, :, ir_gp, :] = 0
            if self._gamma_iso is not None:
                gamma_iso_irgp = self._gamma_iso[:, ir_gp, :].copy()
                self._gamma_iso[:, ir_gp, :] = 0
            for j, r in enumerate(self._context.rotations_cartesian):
                gp_r = rot_grid_points[j, ir_gp]
                self._gv[gp_r] += np.dot(gv_irgp, r.T) / multi
                self._cv[:, gp_r, :] += cv_irgp / multi
                self._gamma[:, :, gp_r, :] += gamma_irgp / multi
                if self._gamma_iso is not None:
                    self._gamma_iso[:, gp_r, :] += gamma_iso_irgp / multi  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------
    # Kappa computation
    # ------------------------------------------------------------------

    def _set_kappa_and_rta_at_sigmas(
        self,
        num_sampling_grid_points: int,
        weights: NDArray[np.double],
        *,
        suppress_kappa_log: bool = False,
    ) -> None:
        """Loop over sigma and temperature to compute kappa and kappa_RTA."""
        for i_sigma, sigma in enumerate(self._context.sigmas):
            if self._log_level:
                self._log_sigma_header(sigma)

            for i_temp, temperature in enumerate(self._context.temperatures):
                if temperature <= 0:
                    continue

                self._set_kappa_RTA_by_collision_type(i_sigma, i_temp, weights)

                assert self._collision_matrix is not None
                w = diagonalize_collision_matrix(
                    self._collision_matrix,
                    i_sigma=i_sigma,
                    i_temp=i_temp,
                    pinv_solver=self._pinv_solver,
                    log_level=self._log_level,
                )
                if w is not None:
                    assert self._collision_eigenvalues is not None
                    self._collision_eigenvalues[i_sigma, i_temp] = w

                self._set_kappa_by_collision_type(i_sigma, i_temp, weights)

                if self._log_level and not suppress_kappa_log:
                    self._log_kappa_at_temperature(
                        i_sigma,
                        i_temp,
                        temperature,
                        num_sampling_grid_points,
                    )

        if self._log_level and not suppress_kappa_log:
            print("", flush=True)

        n = num_sampling_grid_points
        if n > 0:
            self._kappa /= n
            self._kappa_RTA /= n

    def _set_kappa_by_collision_type(
        self, i_sigma: int, i_temp: int, weights: NDArray[np.double]
    ) -> None:
        if self._is_reducible_collision_matrix:
            self._set_kappa_reducible_colmat(i_sigma, i_temp, weights)
        else:
            self._set_kappa_ir_colmat(i_sigma, i_temp, weights)

    def _set_kappa_RTA_by_collision_type(
        self, i_sigma: int, i_temp: int, weights: NDArray[np.double]
    ) -> None:
        if self._is_reducible_collision_matrix:
            self._set_kappa_RTA_reducible_colmat(i_sigma, i_temp, weights)
        else:
            self._set_kappa_RTA_ir_colmat(i_sigma, i_temp, weights)

    def _set_kappa_ir_colmat(
        self, i_sigma: int, i_temp: int, weights: NDArray[np.double]
    ) -> None:
        if self._solve_collective_phonon:
            raise NotImplementedError(
                "solve_collective_phonon is not supported in CollisionMatrixSolver."
            )
        X = self._get_X(i_temp, weights)
        num_ir_gp = len(self._context.ir_grid_points)
        Y = self._get_Y(i_sigma, i_temp, weights, X)
        self._set_mean_free_path(i_sigma, i_temp, weights, Y)
        self._set_mode_kappa(self._mode_kappa, X, Y, num_ir_gp, i_sigma, i_temp)
        self._kappa[i_sigma, i_temp] += (
            self._mode_kappa[i_sigma, i_temp].sum(axis=0).sum(axis=0)
        )

    def _set_kappa_reducible_colmat(
        self, i_sigma: int, i_temp: int, weights: NDArray[np.double]
    ) -> None:
        X = self._get_X(i_temp, weights)
        num_mesh_points = int(np.prod(self._context.mesh_numbers))
        Y = self._get_Y(i_sigma, i_temp, weights, X)
        self._set_mean_free_path(i_sigma, i_temp, weights, Y)
        self._set_mode_kappa(self._mode_kappa, X, Y, num_mesh_points, i_sigma, i_temp)
        self._mode_kappa[i_sigma, i_temp] /= len(self._context.rotations_cartesian)
        self._kappa[i_sigma, i_temp] += (
            self._mode_kappa[i_sigma, i_temp].sum(axis=0).sum(axis=0)
        )

    def _set_kappa_RTA_ir_colmat(
        self, i_sigma: int, i_temp: int, weights: NDArray[np.double]
    ) -> None:
        X = self._get_X(i_temp, weights)
        Y = self._build_rta_Y_ir_colmat(i_sigma, i_temp, X)
        num_ir_gp = len(self._context.ir_grid_points)
        self._set_mode_kappa(self._mode_kappa_RTA, X, Y, num_ir_gp, i_sigma, i_temp)
        self._kappa_RTA[i_sigma, i_temp] += (
            self._mode_kappa_RTA[i_sigma, i_temp].sum(axis=0).sum(axis=0)
        )

    def _set_kappa_RTA_reducible_colmat(
        self, i_sigma: int, i_temp: int, weights: NDArray[np.double]
    ) -> None:
        X = self._get_X(i_temp, weights)
        num_mesh_points = int(np.prod(self._context.mesh_numbers))
        Y = self._build_rta_Y_reducible_colmat(i_sigma, i_temp, X)
        self._set_mode_kappa(
            self._mode_kappa_RTA, X, Y, num_mesh_points, i_sigma, i_temp
        )
        self._mode_kappa_RTA[i_sigma, i_temp] /= len(self._context.rotations_cartesian)
        self._kappa_RTA[i_sigma, i_temp] += (
            self._mode_kappa_RTA[i_sigma, i_temp].sum(axis=0).sum(axis=0)
        )

    def _build_rta_Y_ir_colmat(
        self, i_sigma: int, i_temp: int, X: NDArray[np.double]
    ) -> NDArray[np.double]:
        assert self._context.frequencies is not None
        Y = np.zeros_like(X)
        num_band = self._context.frequencies.shape[1]
        for i, gp in enumerate(self._context.ir_grid_points):
            g = self._get_main_diagonal(i, i_sigma, i_temp)
            frequencies = self._context.frequencies[gp]
            for j, f in enumerate(frequencies):
                if f > self._context.cutoff_frequency:
                    i_mode = i * num_band + j
                    old_settings = np.seterr(all="raise")
                    try:
                        Y[i_mode, :] = X[i_mode, :] / g[j]
                    except Exception:
                        print("=" * 26 + " Warning " + "=" * 26)
                        print(
                            " Unexpected physical condition of ph-ph "
                            "interaction calculation was found."
                        )
                        print(
                            " g[j]=%f at gp=%d, band=%d, freq=%f" % (g[j], gp, j + 1, f)
                        )
                        print("=" * 61)
                    np.seterr(**old_settings)
        return Y

    def _build_rta_Y_reducible_colmat(
        self, i_sigma: int, i_temp: int, X: NDArray[np.double]
    ) -> NDArray[np.double]:
        assert self._context.frequencies is not None
        assert self._collision_matrix is not None
        num_band = self._context.frequencies.shape[1]
        num_mesh_points = int(np.prod(self._context.mesh_numbers))
        size = num_mesh_points * num_band
        v_diag = np.diagonal(
            self._collision_matrix[i_sigma, i_temp].reshape(size, size)
        )
        Y = np.zeros_like(X)
        for gp in range(num_mesh_points):
            frequencies = self._context.frequencies[gp]
            for j, f in enumerate(frequencies):
                if f > self._context.cutoff_frequency:
                    i_mode = gp * num_band + j
                    Y[i_mode, :] = X[i_mode, :] / v_diag[i_mode]
        return Y

    # ------------------------------------------------------------------
    # X and Y vectors for kappa computation
    # ------------------------------------------------------------------

    def _get_X(self, i_temp: int, weights: NDArray[np.double]) -> NDArray[np.double]:
        """Compute X vector (Chaput's paper) at temperature index i_temp."""
        X = self._gv.copy()
        num_band = self._context.frequencies.shape[1]
        freqs = self._get_X_frequencies()
        t = self._context.temperatures[i_temp]
        freqs_factor = self._get_X_frequency_factor(freqs, t)
        self._scale_X_by_weights_and_frequency(X, weights, freqs_factor, num_band)
        if t <= 0:
            return np.zeros_like(X.reshape(-1, 3))
        return X.reshape(-1, 3)

    def _get_X_frequencies(self) -> NDArray[np.double]:
        assert self._context.frequencies is not None
        if self._is_reducible_collision_matrix:
            return self._context.frequencies[self._context.bz_grid.grg2bzg]
        return self._context.frequencies[self._context.ir_grid_points]

    def _get_X_frequency_factor(
        self, freqs: NDArray[np.double], temperature: float
    ) -> NDArray[np.double]:
        sinh = np.where(
            freqs > self._context.cutoff_frequency,
            np.sinh(
                freqs
                * get_physical_units().THzToEv
                / (2 * get_physical_units().KB * temperature)
            ),
            -1.0,
        )
        inv_sinh = np.where(sinh > 0, 1.0 / sinh, 0)
        return (
            freqs
            * get_physical_units().THzToEv
            * inv_sinh
            / (4 * get_physical_units().KB * temperature**2)
        )

    def _scale_X_by_weights_and_frequency(
        self,
        X: NDArray[np.double],
        weights: NDArray[np.double],
        freqs_factor: NDArray[np.double],
        num_band: int,
    ) -> None:
        for i, f in enumerate(freqs_factor):
            X[i] *= weights[i]
            for j in range(num_band):
                X[i, j] *= f[j]

    def _get_Y(
        self,
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
        X: NDArray[np.double],
    ) -> NDArray[np.double]:
        """Compute Y = Omega^-1 X."""
        solver = self._get_colmat_solver()
        num_grid_points, size = self._get_Y_problem_size()
        v = self._get_Y_solver_matrix(i_sigma, i_temp, size, solver)

        start = time.time()
        self._log_Y_solver(i_sigma, i_temp, solver)
        Y = self._solve_Y(solver, v, X, i_sigma, i_temp)
        self._set_f_vectors(Y, num_grid_points, weights)

        if self._log_level and solver != 7:
            print("[%.3fs]" % (time.time() - start), flush=True)
            sys.stdout.flush()

        return Y

    def _get_colmat_solver(self) -> int:
        solver = select_colmat_solver(self._pinv_solver)
        if self._pinv_solver == 6:
            return 6
        return solver

    def _get_Y_problem_size(self) -> tuple[int, int]:
        num_band = self._context.frequencies.shape[1]
        if self._is_reducible_collision_matrix:
            num_gp = int(np.prod(self._context.mesh_numbers))
            size = num_gp * num_band
        else:
            num_gp = len(self._context.ir_grid_points)
            size = num_gp * num_band * 3
        return num_gp, size

    def _get_Y_solver_matrix(
        self, i_sigma: int, i_temp: int, size: int, solver: int
    ) -> NDArray[np.double]:
        assert self._collision_matrix is not None
        v = self._collision_matrix[i_sigma, i_temp].reshape(size, size)
        if solver in [1, 2, 4, 5]:
            return v.T
        return v

    def _log_Y_solver(self, i_sigma: int, i_temp: int, solver: int) -> None:
        if not self._log_level or solver == 7:
            return
        assert self._collision_eigenvalues is not None
        eig_str = "abs(eig)" if self._pinv_method == 0 else "eig"
        w = self._collision_eigenvalues[i_sigma, i_temp]
        null_space = (np.abs(w) < self._pinv_cutoff).sum()
        print(
            f"Pinv by ignoring {null_space}/{len(w)} dims "
            f"under {eig_str}<{self._pinv_cutoff:<.1e}",
            end="",
        )

    def _solve_Y(
        self,
        solver: int,
        v: NDArray[np.double],
        X: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
    ) -> NDArray[np.double]:
        if solver in [0, 1, 2, 3, 4, 5]:
            return self._solve_Y_eigendecomp(v, X, i_sigma, i_temp)
        if solver == 6:
            return self._solve_Y_builtin_pinv(v, X, i_sigma, i_temp)
        if solver == 7:
            return self._solve_Y_direct_pinv(v, X)
        raise ValueError(f"Unknown collision matrix solver {solver}")

    def _solve_Y_eigendecomp(
        self,
        v: NDArray[np.double],
        X: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
    ) -> NDArray[np.double]:
        if self._log_level:
            print(" (np.dot) ", end="")
            sys.stdout.flush()

        e = self._get_eigvals_pinv(i_sigma, i_temp)
        if self._is_reducible_collision_matrix:
            X1 = np.dot(v.T, X)
            for i in range(3):
                X1[:, i] *= e
            return np.dot(v, X1)
        return np.dot(v, e * np.dot(v.T, X.ravel())).reshape(-1, 3)

    def _solve_Y_builtin_pinv(
        self,
        v: NDArray[np.double],
        X: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
    ) -> NDArray[np.double]:
        import phono3py._phono3py as phono3c

        if self._log_level:
            print(" (built-in-pinv) ", end="", flush=True)

        assert self._collision_eigenvalues is not None
        assert self._collision_matrix is not None
        w = self._collision_eigenvalues[i_sigma, i_temp]
        phono3c.pinv_from_eigensolution(
            self._collision_matrix,
            w,
            i_sigma,
            i_temp,
            self._pinv_cutoff,
            self._pinv_method,
        )
        return self._solve_Y_direct_pinv(v, X)

    def _solve_Y_direct_pinv(
        self, v: NDArray[np.double], X: NDArray[np.double]
    ) -> NDArray[np.double]:
        if self._is_reducible_collision_matrix:
            return np.dot(v, X)
        return np.dot(v, X.ravel()).reshape(-1, 3)

    def _get_eigvals_pinv(self, i_sigma: int, i_temp: int) -> NDArray[np.double]:
        assert self._collision_eigenvalues is not None
        w = self._collision_eigenvalues[i_sigma, i_temp]
        e = np.zeros_like(w)
        for ll, val in enumerate(w):
            _val = abs(val) if self._pinv_method == 0 else val
            if _val > self._pinv_cutoff:
                e[ll] = 1 / val
        return e

    def _set_f_vectors(
        self,
        Y: NDArray[np.double],
        num_grid_points: int,
        weights: NDArray[np.double],
    ) -> None:
        """Compute f-vectors from Y.

        Collision matrix is half of that in Chaput's paper, so Y is
        divided by 2.

        """
        assert self._f_vectors is not None
        num_band = self._context.frequencies.shape[1]
        self._f_vectors[:] = (
            (Y / 2).reshape(num_grid_points, num_band * 3).T / weights
        ).T.reshape(self._f_vectors.shape)

    def _set_mean_free_path(
        self,
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
        Y: NDArray[np.double],
    ) -> None:
        assert self._mfp is not None
        assert self._f_vectors is not None
        t = self._context.temperatures[i_temp]
        for i, f_gp in enumerate(self._f_vectors):
            for j, f in enumerate(f_gp):
                cv = self._cv[i_temp, i, j]
                if cv < 1e-10:
                    continue
                self._mfp[i_sigma, i_temp, i, j] = (
                    -2 * t * np.sqrt(get_physical_units().KB / cv) * f / (2 * np.pi)
                )

    # ------------------------------------------------------------------
    # Mode kappa
    # ------------------------------------------------------------------

    def _set_mode_kappa(
        self,
        mode_kappa: NDArray[np.double],
        X: NDArray[np.double],
        Y: NDArray[np.double],
        num_grid_points: int,
        i_sigma: int,
        i_temp: int,
    ) -> None:
        """Compute mode thermal conductivity tensor.

        kappa = A * (R*X, R*Y), where A = k_B * T^2 / V.

        Collision matrix is defined as half of that in Chaput's paper,
        so the factor of 2 is not needed here.

        """
        num_band = self._context.frequencies.shape[1]
        for i, (v_gp, f_gp) in enumerate(
            zip(
                X.reshape(num_grid_points, num_band, 3),
                Y.reshape(num_grid_points, num_band, 3),
                strict=True,
            )
        ):
            for j, (v, f) in enumerate(zip(v_gp, f_gp, strict=True)):
                # Skip three lowest modes at Gamma-point
                # (assumed no imaginary modes).
                if (self._context.bz_grid.addresses[i] == 0).all() and j < 3:
                    continue
                sum_k = self._compute_mode_kappa_tensor(v, f)
                sum_k = sum_k + sum_k.T
                for k, vxf in enumerate(
                    ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
                ):
                    mode_kappa[i_sigma, i_temp, i, j, k] = sum_k[vxf]

        t = self._context.temperatures[i_temp]
        mode_kappa[i_sigma, i_temp] *= (
            self._conversion_factor * get_physical_units().KB * t**2
        )

    def _compute_mode_kappa_tensor(
        self,
        v: NDArray[np.double],
        f: NDArray[np.double],
    ) -> NDArray[np.double]:
        sum_k = np.zeros((3, 3), dtype="double")
        for r in self._context.rotations_cartesian:
            sum_k += np.outer(np.dot(r, v), np.dot(r, f))
        return sum_k

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_sigma_header(self, sigma: float | None) -> None:
        text = "----------- Thermal conductivity (W/m-k) "
        if sigma:
            text += "for sigma=%s -----------" % sigma
        else:
            text += "with tetrahedron method -----------"
        print(text, flush=True)

    def _log_kappa_at_temperature(
        self,
        i_sigma: int,
        i_temp: int,
        temperature: float,
        num_sampling_grid_points: int,
    ) -> None:
        n = num_sampling_grid_points if num_sampling_grid_points > 0 else 1
        print(
            ("#%6s       " + " %-10s" * 6)
            % ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
        )
        print(
            ("%7.1f " + " %10.3f" * 6)
            % ((temperature,) + tuple(self._kappa[i_sigma, i_temp] / n))
        )
        print(
            (" %6s " + " %10.3f" * 6)
            % (("(RTA)",) + tuple(self._kappa_RTA[i_sigma, i_temp] / n))
        )
        print("-" * 76, flush=True)
