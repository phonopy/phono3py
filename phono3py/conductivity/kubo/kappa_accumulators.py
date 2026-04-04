"""Kappa accumulators for the Green-Kubo method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.collision_matrix_solver import CollisionMatrixSolver
from phono3py.conductivity.context import ConductivityContext
from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
    compute_effective_gamma,
)
from phono3py.conductivity.lbte_collision_provider import LBTECollisionResult
from phono3py.conductivity.utils import log_kappa_header, log_kappa_row

_FLOATINGPOINTERROR_THRESHOLD = 1e-12


def _compute_kubo_mode_kappa_matrix(
    frequencies: NDArray[np.double],
    gamma: NDArray[np.double],
    heat_capacity_matrix: NDArray[np.double],
    vm_by_vm: NDArray[np.cdouble],
    cutoff_frequency: float,
    conversion_factor: float,
) -> NDArray[np.double]:
    """Compute Kubo mode-kappa matrix for one grid point.

    The Green-Kubo formula for each band pair (s, s'):

        kappa_mat[s, s'] = C[s,s'] * V[s,s']*V*[s,s'] * g / (dw^2 + g^2)

    where g = gamma_s + gamma_s' is the sum of half-linewidths and all
    frequencies are in THz.

    Parameters
    ----------
    frequencies : ndarray of double, shape (num_band,)
        Phonon frequencies at this grid point in THz.
    gamma : ndarray of double, shape (num_sigma, num_temp, num_band)
        Effective scattering half-linewidths.
    heat_capacity_matrix : ndarray of double, shape (num_temp, num_band0, num_band)
        Off-diagonal heat capacity matrix C_{ss'}.
    vm_by_vm : ndarray of complex, shape (num_band0, num_band, 6)
        k-star-averaged velocity outer product in Voigt order.
    cutoff_frequency : float
        Modes with frequency below this value (in THz) are skipped.
    conversion_factor : float
        Unit conversion factor to W/(m*K).

    Returns
    -------
    mode_kappa_mat : ndarray of double,
        shape (num_sigma, num_temp, num_band0, num_band, 6)

    """
    num_sigma, num_temp, num_band = gamma.shape

    mode_kappa_mat = np.zeros(
        (num_sigma, num_temp, num_band, num_band, 6), dtype="double"
    )

    for j in range(num_sigma):
        for k in range(num_temp):
            g_sum = np.add.outer(gamma[j, k], gamma[j, k])
            delta_omega = np.subtract.outer(frequencies, frequencies)
            freqs_condisions = frequencies > cutoff_frequency
            denom = delta_omega**2 + g_sum**2
            condition_matrix = (
                np.outer(freqs_condisions, freqs_condisions) * denom
                > _FLOATINGPOINTERROR_THRESHOLD
            )
            denom = np.where(condition_matrix, denom, 1.0)
            contribution_matrix = np.where(
                condition_matrix,
                heat_capacity_matrix[k, :, :, None]
                * vm_by_vm.real
                * g_sum[:, :, None]
                / denom[:, :, None]
                * conversion_factor,
                0,
            )
            mode_kappa_mat[j, k] = contribution_matrix[:, :, :]

    return mode_kappa_mat


class KuboRTAKappaAccumulator:
    """Kappa accumulator for the Green-Kubo formula.

    Accumulates the full band-pair kappa matrix ``mode_kappa_inter``.
    The ``kappa`` property sums over all band pairs.

    The Green-Kubo formula for each band pair (s, s'):

        kappa_mat[s, s'] = C_{s s'} * GVM_{s s'} * g / ((w_s' - w_s)^2 + g^2)

    where g = gamma_s + gamma_s' is the sum of half-linewidths.

    Parameters
    ----------
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, configuration).
    conversion_factor : float
        Unit conversion factor to W/(m*K).
    is_isotope : bool
        Include isotope scattering.
    log_level : int
        Verbosity level.

    """

    def __init__(
        self,
        context: ConductivityContext,
        conversion_factor: float,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._context = context
        self._conversion_factor = conversion_factor
        self._log_level = log_level

        # Kappa arrays (allocated in prepare()).
        self._kappa: NDArray[np.double]
        self._mode_kappa_matrix: NDArray[np.double]

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
        *,
        num_band: int | None = None,
    ) -> None:
        """Allocate kappa arrays."""
        if num_band is None:
            num_band = num_band0
        self._kappa = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._kappa_intra = np.zeros(
            (num_sigma, num_temp, 6), dtype="double", order="C"
        )
        self._mode_kappa_matrix = np.zeros(
            (num_sigma, num_temp, num_gp, num_band, num_band, 6),
            dtype="double",
            order="C",
        )

    def finalize(self, aggregates: GridPointAggregates) -> None:
        """Compute kappa and kappa_intra from aggregated data."""
        assert aggregates.vm_by_vm is not None
        assert aggregates.heat_capacity_matrix is not None

        num_sampling_grid_points = aggregates.num_sampling_grid_points
        gamma_eff = compute_effective_gamma(aggregates)

        for i_gp, gp in enumerate(self._context.grid_points):
            mode_kappa_matrix = _compute_kubo_mode_kappa_matrix(
                frequencies=self._context.frequencies[gp],
                gamma=gamma_eff[:, :, i_gp, :],
                heat_capacity_matrix=aggregates.heat_capacity_matrix[:, i_gp, :, :],
                vm_by_vm=aggregates.vm_by_vm[i_gp],
                cutoff_frequency=self._context.cutoff_frequency,
                conversion_factor=self._conversion_factor,
            )
            self._mode_kappa_matrix[:, :, i_gp, :, :, :] = mode_kappa_matrix

        if num_sampling_grid_points > 0:
            self._kappa = (
                self._mode_kappa_matrix.sum(axis=(2, 3, 4)) / num_sampling_grid_points
            )
            self._kappa_intra = (
                np.einsum("abijjc->abc", self._mode_kappa_matrix)
                / num_sampling_grid_points
            )

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total kappa (kappa_intra + kappa_inter).

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa

    @property
    def kappa_intra(self) -> NDArray[np.double]:
        """Return intra-band (diagonal) kappa.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa_intra

    @property
    def kappa_inter(self) -> NDArray[np.double]:
        """Return inter-band (off-diagonal) kappa.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa - self._kappa_intra

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode kappa summed over j_band.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        # Sum over j_band axis (axis=-2 of the per-gp array, axis=4 in stored array).
        return self._mode_kappa_matrix.sum(axis=4)

    @property
    def mode_kappa_matrix(self) -> NDArray[np.double]:
        """Return full band-pair kappa matrix.

        Shape: (num_gp, num_sigma, num_temp, num_band0, num_band, 6).

        """
        return self._mode_kappa_matrix

    # ------------------------------------------------------------------
    # Properties -- per-grid-point data
    # ------------------------------------------------------------------

    def log_kappa(
        self,
        num_ignored_phonon_modes: NDArray[np.int64] | None = None,
        num_phonon_modes: int | None = None,
    ) -> None:
        """Print K_intra, K_inter, K_TOT rows for the Kubo-RTA conductivity."""
        if not self._log_level:
            return

        kappa_inter = self._kappa - self._kappa_intra
        show_ipm = (
            self._log_level > 1
            and num_ignored_phonon_modes is not None
            and num_phonon_modes is not None
        )

        for i, sigma in enumerate(self._context.sigmas):
            log_kappa_header(sigma, show_ipm=show_ipm)
            for j, t in enumerate(self._context.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row(
                    "K_intra", t, self._kappa_intra[i, j], ipm, num_phonon_modes
                )
            print(" ")
            for j, t in enumerate(self._context.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row("K_inter", t, kappa_inter[i, j], ipm, num_phonon_modes)
            print(" ")
            for j, t in enumerate(self._context.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row("K_TOT", t, self._kappa[i, j], ipm, num_phonon_modes)
            print("", flush=True)


class KuboLBTEKappaAccumulator:
    """LBTE accumulator with added Kubo inter-band kappa.

    Composes a CollisionMatrixSolver for the standard LBTE solve (intra-band)
    and adds the inter-band kappa from the Kubo formula using the collision
    matrix diagonal as effective linewidths.

    Stage 1 (per-grid-point): store() delegates collision data to the solver.

    Stage 2 (global): finalize() calls solver.solve() for the standard LBTE
    kappa (intra-band), then computes Kubo inter-band kappa.

    Parameters
    ----------
    solver : CollisionMatrixSolver
        Shared solver for the standard LBTE solve.
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, configuration).
    conversion_factor : float
        Unit conversion factor to W/(m*K).
    log_level : int, optional
        Verbosity level. Default 0.

    """

    def __init__(
        self,
        solver: CollisionMatrixSolver,
        context: ConductivityContext,
        conversion_factor: float,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._solver = solver
        self._context = context
        self._conversion_factor = conversion_factor
        self._log_level = log_level

        # Output arrays (populated in finalize).
        self._kappa_inter: NDArray[np.double] | None = None
        self._mode_kappa_matrix: NDArray[np.double] | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        """Allocate accumulator arrays."""
        self._solver.prepare()

    def store(
        self,
        i_gp: int,
        collision_result: LBTECollisionResult,
    ) -> None:
        """Store collision matrix row for this grid point.

        Parameters
        ----------
        i_gp : int
            Loop index over ir_grid_points (0-based).
        collision_result : LBTECollisionResult
            Result from LBTECollisionProvider.compute().

        """
        self._solver.store(i_gp, collision_result)

    def finalize(
        self,
        aggregates: GridPointAggregates,
        suppress_kappa_log: bool = False,
    ) -> None:
        """Finalize LBTE solve, then compute Kubo kappa with LBTE linewidths."""
        self._solver.solve(
            aggregates,
            suppress_kappa_log=bool(self._log_level),
        )
        self._compute_kubo_kappa(aggregates)
        if self._log_level:
            self._log_kubo_kappa()

    # ------------------------------------------------------------------
    # Properties — delegated to solver (LBTE intra-band results)
    # ------------------------------------------------------------------

    @property
    def solver(self) -> CollisionMatrixSolver:
        """Return the underlying CollisionMatrixSolver."""
        return self._solver

    # ------------------------------------------------------------------
    # Properties — Kubo-specific
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total kappa (kappa_intra_exact + kappa_inter).

        Shape: (num_sigma, num_temp, 6).

        """
        if self._kappa_inter is not None:
            return self._solver.kappa + self._kappa_inter
        return self._solver.kappa

    @property
    def kappa_intra_exact(self) -> NDArray[np.double]:
        """Return intra-band kappa from LBTE exact solve.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._solver.kappa

    @property
    def kappa_intra_RTA(self) -> NDArray[np.double]:
        """Return intra-band kappa from LBTE RTA.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._solver.kappa_RTA

    @property
    def kappa_inter(self) -> NDArray[np.double] | None:
        """Return inter-band (off-diagonal) kappa from Kubo formula.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa_inter

    @property
    def mode_kappa_intra_exact(self) -> NDArray[np.double]:
        """Return intra-band mode kappa from LBTE exact solve.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._solver.mode_kappa

    @property
    def mode_kappa_intra_RTA(self) -> NDArray[np.double]:
        """Return intra-band mode kappa from LBTE RTA.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._solver.mode_kappa_RTA

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return intra-band mode kappa from LBTE exact solve.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).

        """
        return self._solver.mode_kappa

    @property
    def mode_kappa_matrix(self) -> NDArray[np.double] | None:
        """Return full band-pair Kubo kappa matrix.

        Shape: (num_gp, num_sigma, num_temp, num_band0, num_band, 6).

        """
        return self._mode_kappa_matrix

    def get_extra_kappa_output(self) -> dict[str, NDArray[np.double] | None]:
        """Return Kubo LBTE kappa arrays keyed by HDF5 dataset name."""
        return {
            "kappa_inter": self._kappa_inter,
            "kappa_intra_exact": self._solver.kappa,
            "kappa_intra_RTA": self._solver.kappa_RTA,
            "mode_kappa_intra_exact": self._solver.mode_kappa,
            "mode_kappa_intra_RTA": self._solver.mode_kappa_RTA,
        }

    # ------------------------------------------------------------------
    # Private: Kubo kappa computation
    # ------------------------------------------------------------------

    def _compute_kubo_kappa(self, aggregates: GridPointAggregates) -> None:
        """Compute inter-band kappa using LBTE collision matrix diagonal."""
        vm_by_vm = aggregates.vm_by_vm
        heat_capacity_matrix = aggregates.heat_capacity_matrix
        num_sampling_grid_points = aggregates.num_sampling_grid_points
        if vm_by_vm is None or heat_capacity_matrix is None:
            return

        num_sigma = len(self._context.sigmas)
        num_temp = len(self._context.temperatures)
        num_ir = len(self._context.ir_grid_points)

        kappa_inter = np.zeros((num_sigma, num_temp, 6), dtype="double")
        mode_kappa_inter_list: list[NDArray[np.double]] = []

        for i_gp in range(num_ir):
            gp = int(self._context.ir_grid_points[i_gp])
            frequencies = self._context.frequencies[gp]
            num_band = len(frequencies)

            # Build gamma from LBTE collision matrix diagonal.
            gamma = np.zeros((num_sigma, num_temp, num_band), dtype="double")
            for i_sigma in range(num_sigma):
                for i_temp in range(num_temp):
                    gamma[i_sigma, i_temp] = self._solver.get_main_diagonal(
                        i_gp, i_sigma, i_temp
                    )

            # Diagonal (intra-band) and off-diagonal (inter-band) More
            # precisely, when frequency differences are small, heat capacity
            # matrix elements are calculated as usual mode heat capacity.
            mode_kappa_matrix = _compute_kubo_mode_kappa_matrix(
                frequencies=frequencies,
                gamma=gamma,
                heat_capacity_matrix=heat_capacity_matrix[i_gp],
                vm_by_vm=vm_by_vm[i_gp],
                cutoff_frequency=self._context.cutoff_frequency,
                conversion_factor=self._conversion_factor,
            )
            # mkm: (num_sigma, num_temp, num_band0, num_band, 6)
            mode_kappa_inter_list.append(mode_kappa_matrix)

            # Off-diagonal (inter-band) sum only.
            all_sum = mode_kappa_matrix.sum(axis=(2, 3))
            diag = np.diagonal(
                mode_kappa_matrix, axis1=2, axis2=3
            )  # (..., 6, num_band)
            kappa_inter += all_sum - diag.sum(axis=-1)

        if num_sampling_grid_points > 0:
            kappa_inter /= num_sampling_grid_points

        self._kappa_inter = kappa_inter
        self._mode_kappa_matrix = np.array(mode_kappa_inter_list)

    def _log_kubo_kappa(self) -> None:
        """Print Kubo LBTE kappa table."""
        kappa_intra_exact = self._solver.kappa
        kappa_intra_RTA = self._solver.kappa_RTA
        kappa_inter = self._kappa_inter

        for i_sigma in range(len(self._context.sigmas)):
            for i_temp, t in enumerate(self._context.temperatures):
                if t <= 0:
                    continue
                print(
                    ("#%6s       " + " %-10s" * 6)
                    % ("           T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                )
                print(
                    "K_intra_exact"
                    + ("%7.1f " + " %10.3f" * 6)
                    % ((t,) + tuple(kappa_intra_exact[i_sigma, i_temp]))
                )
                print(
                    "(K_intra_RTA)"
                    + ("%7.1f " + " %10.3f" * 6)
                    % ((t,) + tuple(kappa_intra_RTA[i_sigma, i_temp]))
                )
                if kappa_inter is not None:
                    print(
                        "K_inter      "
                        + ("%7.1f " + " %10.3f" * 6)
                        % ((t,) + tuple(kappa_inter[i_sigma, i_temp]))
                    )
                    print(" ")
                    kappa_tot = (
                        kappa_intra_exact[i_sigma, i_temp]
                        + kappa_inter[i_sigma, i_temp]
                    )
                    print(
                        "K_TOT=K_intra+K_inter"
                        + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_tot))
                    )
                print("-" * 76, flush=True)
