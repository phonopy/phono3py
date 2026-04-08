"""LBTECalculator: composition-based LBTE thermal conductivity."""

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

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.build_components import KappaSettings
from phono3py.conductivity.grid_point_data import GridPointAggregates
from phono3py.conductivity.heat_capacity_solvers import ModeHeatCapacitySolver
from phono3py.conductivity.kappa_solvers import LBTEKappaSolver
from phono3py.conductivity.lbte_collision_solver import LBTECollisionSolver
from phono3py.conductivity.scattering_solvers import IsotopeScatteringSolver
from phono3py.conductivity.utils import (
    show_grid_point_frequencies_gv,
    show_grid_point_header,
)
from phono3py.conductivity.velocity_solvers import GroupVelocitySolver
from phono3py.other.isotope import Isotope
from phono3py.phonon.grid import BZGrid, get_qpoints_from_bz_grid_points
from phono3py.phonon3.interaction import Interaction


class LBTECalculator:
    """LBTE thermal conductivity calculator using composed building blocks.

    Two-stage design:

    Stage 1 (per-grid-point, parallel-ready): for each irreducible grid point
    compute the collision matrix row via LBTECollisionSolver and store
    velocities and heat capacities via the standard providers.

    Stage 2 (global, after Stage 1 loop): LBTEKappaSolver.finalize()
    assembles the full collision matrix, symmetrizes it, diagonalizes or
    inverts it, and computes kappa and kappa_RTA.

    Results are accessed via the kappa, kappa_RTA, mode_kappa, mode_kappa_RTA,
    collision_matrix, collision_eigenvalues, gamma, and related properties.

    Parameters
    ----------
    pp : Interaction
        Ph-ph interaction object.  init_dynamical_matrix must have been called.
    velocity_solver : GroupVelocitySolver
        Computes group velocities at each grid point.
    cv_solver : ModeHeatCapacitySolver
        Computes mode heat capacities at each grid point.
    collision_solver : LBTECollisionSolver
        Computes gamma and one collision matrix row per grid point.
    kappa_solver : LBTEKappaSolver
        Owns the global collision matrix and solves for kappa at Stage 2.
    kappa_settings : KappaSettings
        Shared computation metadata (grid, phonon, symmetry, configuration).
    is_isotope : bool, optional
        Include isotope scattering.  Default False.
    mass_variances : array-like or None, optional
        Mass variances for isotope scattering.  Default None.
    is_full_pp : bool, optional
        Compute averaged ph-ph interaction.  Default False.
    log_level : int, optional
        Verbosity level.  Default 0.

    """

    def __init__(
        self,
        pp: Interaction,
        velocity_solver: GroupVelocitySolver,
        cv_solver: ModeHeatCapacitySolver,
        collision_solver: LBTECollisionSolver,
        kappa_solver: LBTEKappaSolver,
        kappa_settings: KappaSettings,
        frequencies: NDArray[np.double],
        *,
        is_isotope: bool = False,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        is_full_pp: bool = False,
        sigma_cutoff_width: float | None = None,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._pp = pp
        self._velocity_solver = velocity_solver
        self._cv_solver = cv_solver
        self._collision_solver = collision_solver
        self._kappa_solver = kappa_solver
        self._kappa_settings = kappa_settings
        self._frequencies = frequencies
        self._is_full_pp = is_full_pp
        self._sigma_cutoff_width = sigma_cutoff_width
        self._log_level = log_level

        # Isotope solver (optional).
        self._isotope_solver: IsotopeScatteringSolver | None = None
        if is_isotope or mass_variances is not None:
            self._isotope_solver = self._build_isotope_solver(mass_variances)

        self._grid_point_count = 0

        # Per-grid-point data (allocated in _allocate_values).
        self._gamma: NDArray[np.double] | None = None
        self._gv: NDArray[np.double] | None = None
        self._cv: NDArray[np.double] | None = None
        self._gamma_iso: NDArray[np.double] | None = None
        self._averaged_pp_interaction: NDArray[np.double] | None = None
        self._gamma_elph: NDArray[np.double] | None = None
        self._gamma_boundary: NDArray[np.double] | None = None
        self._vm_by_vm: NDArray[np.cdouble] | None = None
        self._heat_capacity_matrix: NDArray[np.double] | None = None

        # Allocate arrays.
        self._allocate_values()
        self._kappa_solver.prepare()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        on_grid_point: Callable[[int], None] | None = None,
    ) -> None:
        """Run all grid points and compute kappa.

        The computation is organized in separate phases:

        1. Bulk heat capacity (vectorized, single call).
        2. Velocity loop (per-GP, no interaction state dependency).
        3. Isotope loop (per-GP, no interaction state dependency).
        4. Main collision loop (per-GP, requires Interaction.set_grid_point).
        5. Finalize (assemble collision matrix and compute kappa).

        Parameters
        ----------
        on_grid_point : callable or None, optional
            Called with the grid-point loop index after each grid point is
            processed in the collision loop.  Used for per-grid-point file
            writes.

        """
        self._prepare_isotope_phonons()

        # (1) Bulk heat capacity.
        if self._log_level:
            print("Running heat capacity calculations...")
            self._compute_bulk_heat_capacities()

        # (2) Velocity loop.
        if self._log_level:
            print("Running velocity calculations...")
        self._compute_all_velocities()

        # (3) Isotope loop.
        if self._isotope_solver is not None:
            if self._log_level:
                for sigma in self._kappa_settings.sigmas:
                    print("Running isotope scattering calculations ", end="")
                    print(
                        "with tetrahedron method..."
                        if sigma is None
                        else f"sigma={sigma}..."
                    )
            self._compute_all_isotope()

        # (4) Main collision loop.
        self._grid_point_count = 0
        for i_gp in range(len(self._kappa_settings.grid_points)):
            self._compute_collision_at_grid_point(i_gp)
            self._grid_point_count = i_gp + 1
            if on_grid_point is not None:
                on_grid_point(i_gp)

        if self._log_level:
            print(
                "=================== End of collection of collisions "
                "==================="
            )

        # (5) Finalize.
        self._kappa_solver.finalize(self._build_grid_point_aggregates())

    def set_kappa_at_sigmas(self) -> None:
        """Finalize kappa from a pre-loaded collision matrix (read-from-file path).

        Calls kappa_solver.finalize().  Use this instead of run() when
        gamma and collision_matrix have been loaded externally.

        """
        aggregates = self._build_grid_point_aggregates()
        self._kappa_solver.finalize(aggregates)

    def delete_gp_collision_and_pp(self) -> None:
        """No-op: memory management compatibility method."""

    # ------------------------------------------------------------------
    # Properties — kappa settings
    # ------------------------------------------------------------------

    @property
    def kappa_settings(self) -> KappaSettings:
        """Return the kappa settings."""
        return self._kappa_settings

    # ------------------------------------------------------------------
    # Properties — grid / phonon metadata
    # ------------------------------------------------------------------

    @property
    def mesh_numbers(self) -> NDArray[np.int64]:
        """Return BZ mesh numbers."""
        return self._kappa_settings.mesh_numbers

    @property
    def bz_grid(self) -> BZGrid:
        """Return BZ grid object."""
        return self._kappa_settings.bz_grid

    @property
    def grid_points(self) -> NDArray[np.int64]:
        """Return irreducible BZ grid point indices."""
        return self._kappa_settings.grid_points

    @property
    def qpoints(self) -> NDArray[np.double]:
        """Return q-point coordinates of the irreducible grid points."""
        return np.array(
            get_qpoints_from_bz_grid_points(
                self._kappa_settings.grid_points, self._kappa_settings.bz_grid
            ),
            dtype="double",
            order="C",
        )

    @property
    def grid_weights(self) -> NDArray[np.int64]:
        """Return symmetry weights of the irreducible grid points."""
        return self._kappa_settings.grid_weights

    @property
    def temperatures(self) -> NDArray[np.double]:
        """Return temperatures in Kelvin."""
        assert self._kappa_settings.temperatures is not None
        return self._kappa_settings.temperatures

    @property
    def sigmas(self) -> list[float | None]:
        """Return smearing widths."""
        return self._kappa_settings.sigmas

    @property
    def sigma_cutoff_width(self) -> float | None:
        """Return smearing cutoff width."""
        return self._sigma_cutoff_width

    @property
    def frequencies(self) -> NDArray[np.double]:
        """Return phonon frequencies at the irreducible grid points."""
        return self._frequencies[self._kappa_settings.grid_points]

    @property
    def grid_point_count(self) -> int:
        """Return number of grid points processed so far."""
        return self._grid_point_count

    # ------------------------------------------------------------------
    # Properties — computed physical quantities
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return LBTE thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._kappa_solver.solver.kappa

    @property
    def kappa_RTA(self) -> NDArray[np.double]:
        """Return RTA thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._kappa_solver.solver.kappa_RTA

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode LBTE kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._kappa_solver.solver.mode_kappa

    @property
    def mode_kappa_RTA(self) -> NDArray[np.double]:
        """Return mode RTA kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._kappa_solver.solver.mode_kappa_RTA

    @property
    def gamma(self) -> NDArray[np.double]:
        """Return ph-ph gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        assert self._gamma is not None
        return self._gamma

    @gamma.setter
    def gamma(self, value: NDArray[np.double]) -> None:
        """Set gamma (for loading from file)."""
        self._gamma = value

    @property
    def gamma_isotope(self) -> NDArray[np.double] | None:
        """Return isotope gamma, shape (num_sigma, num_gp, num_band0)."""
        return self._gamma_iso

    @property
    def collision_matrix(self) -> NDArray[np.double] | None:
        """Return assembled collision matrix."""
        return self._kappa_solver.solver.collision_matrix

    @collision_matrix.setter
    def collision_matrix(self, value: NDArray[np.double] | None) -> None:
        """Set collision matrix (for loading from file)."""
        self._kappa_solver.solver.collision_matrix = value

    @property
    def collision_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues of collision matrix."""
        return self._kappa_solver.solver.collision_eigenvalues

    @property
    def averaged_pp_interaction(self) -> NDArray[np.double] | None:
        """Return averaged ph-ph interaction, shape (num_gp, num_band0)."""
        return self._averaged_pp_interaction

    @property
    def boundary_mfp(self) -> float | None:
        """Return boundary mean free path in micrometres."""
        return self._kappa_settings.boundary_mfp

    @property
    def group_velocities(self) -> NDArray[np.double]:
        """Return group velocities, shape (num_gp, num_band0, 3)."""
        assert self._gv is not None
        return self._gv

    @property
    def mode_heat_capacities(self) -> NDArray[np.double]:
        """Return mode heat capacities, shape (num_temp, num_gp, num_band0)."""
        assert self._cv is not None
        return self._cv

    @property
    def f_vectors(self) -> NDArray[np.double] | None:
        """Return f-vectors, shape (num_gp, num_band0, 3)."""
        return self._kappa_solver.solver.f_vectors

    @property
    def mfp(self) -> NDArray[np.double] | None:
        """Return mean free path, shape (num_sigma, num_temp, num_gp, num_band0, 3)."""
        return self._kappa_solver.solver.mfp

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute lookups to the kappa solver.

        Allows plugin-specific properties (kappa_P_exact, kappa_C, etc.)
        to be accessed directly on the calculator without hard-coding them here.

        """
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            acc = object.__getattribute__(self, "_kappa_solver")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(acc, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def get_extra_kappa_output(self) -> dict[str, Any] | None:
        """Return variant-specific kappa output from the kappa solver.

        Called by output writers to obtain plugin-defined quantities that
        are written to the hdf5 file via
        write_kappa_to_hdf5(extra_datasets=...).

        Returns None when the kappa solver does not implement this method
        (standard LBTE).

        """
        fn = getattr(self._kappa_solver, "get_extra_kappa_output", None)
        return fn() if callable(fn) else None

    def get_frequencies_all(self) -> NDArray[np.double]:
        """Return phonon frequencies on the full BZ grid."""
        return self._frequencies[self._kappa_settings.bz_grid.grg2bzg]

    # ------------------------------------------------------------------
    # Private: isotope
    # ------------------------------------------------------------------

    def _build_isotope_solver(
        self,
        mass_variances: Sequence[float] | NDArray[np.double] | None,
    ) -> IsotopeScatteringSolver:
        isotope = Isotope(
            self._pp.mesh_numbers,
            self._pp.primitive,
            mass_variances=mass_variances,
            bz_grid=self._pp.bz_grid,
            frequency_factor_to_THz=self._pp.frequency_factor_to_THz,
            symprec=self._pp.primitive_symmetry.tolerance,
            cutoff_frequency=self._pp.cutoff_frequency,
            lapack_zheev_uplo=self._pp.lapack_zheev_uplo,
        )
        return IsotopeScatteringSolver(
            isotope, self._kappa_settings.sigmas, log_level=self._log_level
        )

    def _prepare_isotope_phonons(self) -> None:
        if self._isotope_solver is None:
            return
        frequencies, eigenvectors, phonon_done = self._pp.get_phonons()
        self._isotope_solver.isotope.set_phonons(
            frequencies,
            eigenvectors,
            phonon_done,
            dm=self._pp.dynamical_matrix,
        )

    # ------------------------------------------------------------------
    # Private: per-grid-point computation
    # ------------------------------------------------------------------

    def _show_log_header(self, i_gp: int) -> None:
        if not self._log_level:
            return
        mass_variances = (
            self._isotope_solver.isotope.mass_variances
            if self._isotope_solver is not None
            else None
        )
        show_grid_point_header(
            bzgp=self._kappa_settings.grid_points[i_gp],
            i_gp=i_gp,
            num_gps=len(self._kappa_settings.grid_points),
            bz_grid=self._kappa_settings.bz_grid,
            boundary_mfp=self._kappa_settings.boundary_mfp,
            mass_variances=mass_variances,
        )

    def _show_log(self, i_gp: int, gv: NDArray[np.double]) -> None:
        bz_gp = self._kappa_settings.grid_points[i_gp]
        frequencies = self._frequencies[bz_gp][self._kappa_settings.band_indices]
        show_grid_point_frequencies_gv(
            frequencies,
            gv,
            gv_delta_q=getattr(self._velocity_solver, "gv_delta_q", None),
        )

    def _allocate_values(self) -> None:
        """Allocate per-grid-point arrays."""
        num_sigma = len(self._kappa_settings.sigmas)
        num_temp = len(self._kappa_settings.temperatures)
        num_gp = len(self._kappa_settings.grid_points)
        num_band0 = len(self._kappa_settings.band_indices)
        num_band = self._frequencies.shape[1]

        self._gamma = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0), order="C", dtype="double"
        )
        self._gv = np.zeros((num_gp, num_band0, 3), order="C", dtype="double")
        self._cv = np.zeros((num_temp, num_gp, num_band0), order="C", dtype="double")
        if self._isotope_solver is not None:
            self._gamma_iso = np.zeros(
                (num_sigma, num_gp, num_band0), order="C", dtype="double"
            )
        if self._is_full_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_gp, num_band0), order="C", dtype="double"
            )
        if self._velocity_solver.produces_vm_by_vm:
            self._vm_by_vm = np.zeros(
                (num_gp, num_band0, num_band, 6), order="C", dtype="complex128"
            )
        if self._cv_solver.produces_heat_capacity_matrix:
            self._heat_capacity_matrix = np.zeros(
                (num_temp, num_gp, num_band0, num_band), order="C", dtype="double"
            )

    def _build_grid_point_aggregates(self) -> GridPointAggregates:
        """Build GridPointAggregates for kappa_solver.finalize()."""
        return GridPointAggregates(
            group_velocities=self._gv,
            mode_heat_capacities=self._cv,
            gamma=self._gamma,
            gamma_isotope=self._gamma_iso,
            gamma_boundary=self._gamma_boundary,
            gamma_elph=self._gamma_elph,
            vm_by_vm=self._vm_by_vm,
            heat_capacity_matrix=self._heat_capacity_matrix,
        )

    def _compute_bulk_heat_capacities(self) -> None:
        """Compute heat capacities for all grid points at once."""
        cv_result = self._cv_solver.compute(self._kappa_settings.grid_points)
        self._cv = cv_result.heat_capacities
        if cv_result.heat_capacity_matrix is not None:
            self._heat_capacity_matrix = cv_result.heat_capacity_matrix

    def _compute_all_velocities(self) -> None:
        """Compute velocities for all grid points."""
        for i_gp in range(len(self._kappa_settings.grid_points)):
            grid_point = int(self._kappa_settings.grid_points[i_gp])
            vel_result = self._velocity_solver.compute(grid_point)
            self._gv[i_gp] = vel_result.group_velocities
            if vel_result.vm_by_vm is not None:
                self._vm_by_vm[i_gp] = vel_result.vm_by_vm

    def _compute_all_isotope(self) -> None:
        """Compute isotope scattering for all grid points."""
        assert self._isotope_solver is not None
        for i_gp in range(len(self._kappa_settings.grid_points)):
            grid_point = int(self._kappa_settings.grid_points[i_gp])
            gamma_iso = self._isotope_solver.compute(grid_point)
            self._gamma_iso[:, i_gp, :] = gamma_iso[
                :, self._kappa_settings.band_indices
            ]

    def _compute_collision_at_grid_point(self, i_gp: int) -> None:
        """Compute collision matrix row and gamma at a single grid point."""
        self._show_log_header(i_gp)
        grid_point = int(self._kappa_settings.grid_points[i_gp])

        collision_result = self._collision_solver.compute(grid_point)
        self._gamma[:, :, i_gp, :] = collision_result.gamma

        if self._is_full_pp and collision_result.averaged_pp is not None:
            self._averaged_pp_interaction[i_gp] = collision_result.averaged_pp

        self._kappa_solver.store(i_gp, collision_result)

        if self._log_level:
            self._show_log(i_gp, self._gv[i_gp])
