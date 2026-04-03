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

from phono3py.conductivity.grid_point_data import GridPointInput, make_grid_point_input
from phono3py.conductivity.heat_capacity_providers import ModeHeatCapacityProvider
from phono3py.conductivity.lbte_collision_provider import LBTECollisionProvider
from phono3py.conductivity.lbte_kappa_accumulator import LBTEKappaAccumulator
from phono3py.conductivity.scattering_providers import IsotopeScatteringProvider
from phono3py.conductivity.utils import (
    show_grid_point_frequencies_gv,
    show_grid_point_header,
)
from phono3py.conductivity.velocity_providers import GroupVelocityProvider
from phono3py.other.isotope import Isotope
from phono3py.phonon.grid import (
    BZGrid,
    get_ir_grid_points,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon3.interaction import Interaction


class LBTECalculator:
    """LBTE thermal conductivity calculator using composed building blocks.

    Two-stage design:

    Stage 1 (per-grid-point, parallel-ready): for each irreducible grid point
    compute the collision matrix row via LBTECollisionProvider and accumulate
    velocities and heat capacities via the standard providers.

    Stage 2 (global, after Stage 1 loop): LBTEKappaAccumulator.finalize()
    assembles the full collision matrix, symmetrizes it, diagonalizes or
    inverts it, and computes kappa and kappa_RTA.

    Results are accessed via the kappa, kappa_RTA, mode_kappa, mode_kappa_RTA,
    collision_matrix, collision_eigenvalues, gamma, and related properties.

    Parameters
    ----------
    pp : Interaction
        Ph-ph interaction object.  init_dynamical_matrix must have been called.
    velocity_provider : GroupVelocityProvider
        Computes group velocities at each grid point.
    cv_provider : ModeHeatCapacityProvider
        Computes mode heat capacities at each grid point.
    collision_provider : LBTECollisionProvider
        Computes gamma and one collision matrix row per grid point.
    accumulator : LBTEKappaAccumulator
        Owns the global collision matrix and solves for kappa at Stage 2.
    temperatures : NDArray[np.double]
        Temperatures in Kelvin, shape (num_temp,).
    sigmas : list of float or None
        Smearing widths.  Empty list selects the tetrahedron method.
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
        velocity_provider: GroupVelocityProvider,
        cv_provider: ModeHeatCapacityProvider,
        collision_provider: LBTECollisionProvider,
        accumulator: LBTEKappaAccumulator,
        temperatures: NDArray[np.double],
        sigmas: list[float | None],
        is_isotope: bool = False,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        is_full_pp: bool = False,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._pp = pp
        self._velocity_provider = velocity_provider
        self._cv_provider = cv_provider
        self._collision_provider = collision_provider
        self._accumulator = accumulator
        self._temperatures = np.asarray(temperatures, dtype="double")
        self._sigmas = sigmas
        self._is_full_pp = is_full_pp
        self._log_level = log_level

        # Solve phonons at all grid points.
        self._pp.nac_q_direction = None
        self._pp.run_phonon_solver_at_gamma()
        self._frequencies, self._eigenvectors, self._phonon_done = (
            self._pp.get_phonons()
        )
        if not self._pp.phonon_all_done:
            self._pp.run_phonon_solver()

        # Build IR grid point list.
        ir_gps, self._grid_weights = self._get_ir_grid_points()
        self._ir_grid_points = np.array(self._pp.bz_grid.grg2bzg[ir_gps], dtype="int64")

        # Isotope provider (optional).
        self._isotope_provider: IsotopeScatteringProvider | None = None
        if is_isotope or mass_variances is not None:
            self._isotope_provider = self._build_isotope_provider(mass_variances)

        self._num_sampling_grid_points = 0
        self._grid_point_count = 0

        # Allocate accumulator arrays.
        self._accumulator.prepare(is_full_pp=self._is_full_pp)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        on_grid_point: Callable[[int], None] | None = None,
    ) -> None:
        """Run all grid points and compute kappa.

        Parameters
        ----------
        on_grid_point : callable or None, optional
            Called with the grid-point loop index after each grid point is
            processed.  Used for per-grid-point file writes.

        """
        if self._log_level:
            print(
                "==================== Lattice thermal conductivity (LBTE) "
                "===================="
            )

        self._prepare_isotope_phonons()

        self._num_sampling_grid_points = 0
        self._grid_point_count = 0

        for i_gp in range(len(self._ir_grid_points)):
            self._run_at_grid_point(i_gp)
            self._grid_point_count = i_gp + 1
            if on_grid_point is not None:
                on_grid_point(i_gp)

        if self._log_level:
            print(
                "=================== End of collection of collisions "
                "==================="
            )

        self._accumulator.finalize(self._num_sampling_grid_points)

    def set_kappa_at_sigmas(self) -> None:
        """Finalize kappa from a pre-loaded collision matrix (read-from-file path).

        Calls accumulator.finalize() using the sum of IR grid weights as the
        number of sampling grid points.  Use this instead of run() when
        gamma and collision_matrix have been loaded externally.

        """
        self._accumulator.finalize(int(self._grid_weights.sum()))

    def delete_gp_collision_and_pp(self) -> None:
        """No-op: memory management compatibility method."""

    # ------------------------------------------------------------------
    # Properties — grid / phonon metadata
    # ------------------------------------------------------------------

    @property
    def mesh_numbers(self) -> NDArray[np.int64]:
        """Return BZ mesh numbers."""
        return self._pp.mesh_numbers

    @property
    def bz_grid(self) -> BZGrid:
        """Return BZ grid object."""
        return self._pp.bz_grid

    @property
    def grid_points(self) -> NDArray[np.int64]:
        """Return irreducible BZ grid point indices."""
        return self._ir_grid_points

    @property
    def qpoints(self) -> NDArray[np.double]:
        """Return q-point coordinates of the irreducible grid points."""
        return np.array(
            get_qpoints_from_bz_grid_points(self._ir_grid_points, self._pp.bz_grid),
            dtype="double",
            order="C",
        )

    @property
    def grid_weights(self) -> NDArray[np.int64]:
        """Return symmetry weights of the irreducible grid points."""
        return self._grid_weights

    @property
    def temperatures(self) -> NDArray[np.double]:
        """Return temperatures in Kelvin."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, value: Sequence[float] | NDArray[np.double]) -> None:
        """Set temperatures and re-allocate accumulator arrays.

        Used by the read-from-file path to resize arrays when the temperatures
        stored in the collision file differ from the initial default.

        """
        self._temperatures = np.asarray(value, dtype="double")
        self._accumulator.temperatures = self._temperatures

    @property
    def sigmas(self) -> list[float | None]:
        """Return smearing widths."""
        return self._sigmas

    @property
    def sigma_cutoff_width(self) -> float | None:
        """Return smearing cutoff width (from collision provider)."""
        return self._collision_provider._sigma_cutoff

    @property
    def frequencies(self) -> NDArray[np.double]:
        """Return phonon frequencies at the irreducible grid points."""
        assert self._frequencies is not None
        return self._frequencies[self._ir_grid_points]

    @property
    def grid_point_count(self) -> int:
        """Return number of grid points processed so far."""
        return self._grid_point_count

    @property
    def number_of_sampling_grid_points(self) -> int:
        """Return total BZ grid points represented (sum of k-star orders)."""
        return self._num_sampling_grid_points

    # ------------------------------------------------------------------
    # Properties — computed physical quantities
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return LBTE thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._accumulator.kappa

    @property
    def kappa_RTA(self) -> NDArray[np.double]:
        """Return RTA thermal conductivity, shape (num_sigma, num_temp, 6)."""
        return self._accumulator.kappa_RTA

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode LBTE kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._accumulator.mode_kappa

    @property
    def mode_kappa_RTA(self) -> NDArray[np.double]:
        """Return mode RTA kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._accumulator.mode_kappa_RTA

    @property
    def gamma(self) -> NDArray[np.double]:
        """Return ph-ph gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        return self._accumulator.gamma

    @gamma.setter
    def gamma(self, value: NDArray[np.double]) -> None:
        """Set gamma (for loading from file)."""
        self._accumulator.gamma = value

    @property
    def gamma_isotope(self) -> NDArray[np.double] | None:
        """Return isotope gamma, shape (num_sigma, num_gp, num_band0)."""
        return self._accumulator.gamma_iso

    @property
    def collision_matrix(self) -> NDArray[np.double] | None:
        """Return assembled collision matrix."""
        return self._accumulator.collision_matrix

    @collision_matrix.setter
    def collision_matrix(self, value: NDArray[np.double] | None) -> None:
        """Set collision matrix (for loading from file)."""
        self._accumulator.collision_matrix = value

    @property
    def collision_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues of collision matrix."""
        return self._accumulator.collision_eigenvalues

    @property
    def averaged_pp_interaction(self) -> NDArray[np.double] | None:
        """Return averaged ph-ph interaction, shape (num_gp, num_band0)."""
        return self._accumulator.averaged_pp_interaction

    @property
    def boundary_mfp(self) -> float | None:
        """Return boundary mean free path in micrometres."""
        return self._accumulator.boundary_mfp

    @property
    def mode_heat_capacities(self) -> NDArray[np.double]:
        """Return mode heat capacities, shape (num_temp, num_gp, num_band0)."""
        return self._accumulator.mode_heat_capacities

    @property
    def f_vectors(self) -> NDArray[np.double] | None:
        """Return f-vectors, shape (num_gp, num_band0, 3)."""
        return self._accumulator.f_vectors

    @property
    def mfp(self) -> NDArray[np.double] | None:
        """Return mean free path, shape (num_sigma, num_temp, num_gp, num_band0, 3)."""
        return self._accumulator.mfp

    def __getattr__(self, name: str) -> object:
        """Delegate unknown attribute lookups to the accumulator.

        Allows plugin-specific properties (kappa_P_exact, kappa_C, etc.)
        to be accessed directly on the calculator without hard-coding them here.

        """
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            acc = object.__getattribute__(self, "_accumulator")
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(acc, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None

    def get_extra_kappa_output(self) -> dict[str, Any] | None:
        """Return variant-specific kappa output from the accumulator.

        Called by output writers to obtain plugin-defined quantities (e.g.
        Wigner kappa_P_exact, kappa_C) that are written to the hdf5 file via
        write_kappa_to_hdf5(extra_datasets=...).

        Returns None when the accumulator does not implement this method
        (standard LBTE).

        """
        fn = getattr(self._accumulator, "get_extra_kappa_output", None)
        return fn() if callable(fn) else None

    def get_frequencies_all(self) -> NDArray[np.double]:
        """Return phonon frequencies on the full BZ grid."""
        assert self._frequencies is not None
        return self._frequencies[self._pp.bz_grid.grg2bzg]

    # ------------------------------------------------------------------
    # Private: grid
    # ------------------------------------------------------------------

    def _get_ir_grid_points(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        ir_gps, ir_weights, _ = get_ir_grid_points(self._pp.bz_grid)
        return np.array(ir_gps, dtype="int64"), np.array(ir_weights, dtype="int64")

    # ------------------------------------------------------------------
    # Private: isotope
    # ------------------------------------------------------------------

    def _build_isotope_provider(
        self,
        mass_variances: Sequence[float] | NDArray[np.double] | None,
    ) -> IsotopeScatteringProvider:
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
        return IsotopeScatteringProvider(
            isotope, self._sigmas, log_level=self._log_level
        )

    def _prepare_isotope_phonons(self) -> None:
        if self._isotope_provider is None:
            return
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        assert self._phonon_done is not None
        self._isotope_provider.isotope.set_phonons(
            self._frequencies,
            self._eigenvectors,
            self._phonon_done,
            dm=self._pp.dynamical_matrix,
        )

    # ------------------------------------------------------------------
    # Private: per-grid-point computation
    # ------------------------------------------------------------------

    def _make_grid_point_input(self, i_gp: int) -> GridPointInput:
        assert self._frequencies is not None
        assert self._eigenvectors is not None
        return make_grid_point_input(
            grid_point=int(self._ir_grid_points[i_gp]),
            grid_weight=int(self._grid_weights[i_gp]),
            frequencies=self._frequencies,
            eigenvectors=self._eigenvectors,
            bz_grid=self._pp.bz_grid,
            band_indices=np.asarray(self._pp.band_indices, dtype="int64"),
        )

    def _show_log_header(self, i_gp: int) -> None:
        if not self._log_level:
            return
        mass_variances = (
            self._isotope_provider.isotope.mass_variances
            if self._isotope_provider is not None
            else None
        )
        show_grid_point_header(
            bzgp=self._ir_grid_points[i_gp],
            i_gp=i_gp,
            num_gps=len(self._ir_grid_points),
            bz_grid=self._pp.bz_grid,
            boundary_mfp=self._accumulator.boundary_mfp,
            mass_variances=mass_variances,
        )

    def _show_log(self, i_gp: int, gv: NDArray[np.double]) -> None:
        bz_gp = self._ir_grid_points[i_gp]
        assert self._frequencies is not None
        frequencies = self._frequencies[bz_gp][self._pp.band_indices]
        show_grid_point_frequencies_gv(
            frequencies,
            gv,
            gv_delta_q=getattr(self._velocity_provider, "gv_delta_q", None),
        )

    def _run_at_grid_point(self, i_gp: int) -> None:
        self._show_log_header(i_gp)
        gp_input = self._make_grid_point_input(i_gp)

        # Group velocities.
        vel_result = self._velocity_provider.compute(gp_input)
        assert vel_result.group_velocities is not None
        gv = vel_result.group_velocities
        self._num_sampling_grid_points += vel_result.num_sampling_grid_points

        # Mode heat capacities.
        cv_result = self._cv_provider.compute(gp_input, self._temperatures)
        assert cv_result.heat_capacities is not None
        cv = cv_result.heat_capacities  # (num_temp, num_band0)

        # Collision matrix row + gamma (Stage 1).
        collision_result = self._collision_provider.compute(gp_input)

        # Isotope scattering (optional diagonal contribution).
        if self._isotope_provider is not None:
            iso_result = self._isotope_provider.compute_gamma_isotope(gp_input)
            assert iso_result.gamma_isotope is not None
            self._accumulator.store_gamma_iso(
                i_gp,
                iso_result.gamma_isotope[:, self._pp.band_indices],
            )

        # Accumulate into global arrays.
        extra = vel_result.extra
        if vel_result.vm_by_vm is not None:
            extra["vm_by_vm"] = vel_result.vm_by_vm
        if cv_result.heat_capacity_matrix is not None:
            extra["heat_capacity_matrix"] = cv_result.heat_capacity_matrix
        self._accumulator.accumulate(i_gp, collision_result, gv, cv, extra or None)

        if self._log_level:
            self._show_log(i_gp, gv)
