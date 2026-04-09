"""Kappa solvers for the Green-Kubo method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.build_components import KappaSettings
from phono3py.conductivity.collision_matrix_kernel import CollisionMatrixKernel
from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
    compute_effective_gamma,
)
from phono3py.conductivity.lbte_collision_solver import LBTECollisionResult
from phono3py.conductivity.utils import (
    log_kappa_header,
    log_kappa_row,
    log_sigma_header,
)


def _compute_kubo_mode_kappa_matrix(
    frequencies: NDArray[np.double],
    gamma: NDArray[np.double],
    heat_capacity_matrix: NDArray[np.double],
    vm_by_vm: NDArray[np.cdouble],
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
    heat_capacity_matrix : ndarray of double, shape (num_temp, num_band, num_band)
        Off-diagonal heat capacity matrix C_{ss'}.
    vm_by_vm : ndarray of complex, shape (num_band, num_band, 6)
        k-star-averaged velocity outer product in Voigt order.
    conversion_factor : float
        Unit conversion factor to W/(m*K).

    Returns
    -------
    mode_kappa_mat : ndarray of double,
        shape (num_sigma, num_temp, num_band, num_band, 6)

    """
    g_sum = gamma[:, :, :, None] + gamma[:, :, None, :]  # stbb
    delta_omega = frequencies[None, None, :, None] - frequencies[None, None, None, :]
    denom = delta_omega**2 + g_sum**2  # stbb

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kappa_mat = (  # stbb6
            heat_capacity_matrix[None, :, :, :, None]
            * vm_by_vm.real[None, None, :, :, :]
            * g_sum[:, :, :, :, None]
            / denom[:, :, :, :, None]
            * conversion_factor
        )
        mode_kappa_mat = np.where(np.isfinite(kappa_mat), kappa_mat, 0)

    return mode_kappa_mat


class KuboRTAKappaSolver:
    """Kappa solver for the Green-Kubo formula.

    Computes the full band-pair kappa matrix ``mode_kappa_inter``.
    The ``kappa`` property sums over all band pairs.

    See the formula in _compute_kubo_mode_kappa_matrix().

    Parameters
    ----------
    kappa_settings : KappaSettings
        Shared computation metadata (grid, phonon, symmetry, configuration).
    log_level : int
        Verbosity level.

    """

    def __init__(
        self,
        kappa_settings: KappaSettings,
        frequencies: NDArray[np.double],
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._kappa_settings = kappa_settings
        self._frequencies = frequencies
        self._log_level = log_level

        self._kappa: NDArray[np.double] | None = None
        self._kappa_intra: NDArray[np.double] | None = None
        self._mode_kappa_matrix: NDArray[np.double] | None = None

    def finalize(self, aggregates: GridPointAggregates) -> None:
        """Compute kappa and kappa_intra from aggregated data."""
        assert aggregates.vm_by_vm is not None
        assert aggregates.heat_capacity_matrix is not None

        gamma_eff = compute_effective_gamma(aggregates)
        num_sigma, num_temp, num_gp, num_band = gamma_eff.shape

        self._mode_kappa_matrix = np.zeros(
            (num_sigma, num_temp, num_gp, num_band, num_band, 6),
            dtype="double",
            order="C",
        )

        for i_gp, gp in enumerate(self._kappa_settings.grid_points):
            mode_kappa_matrix = _compute_kubo_mode_kappa_matrix(
                frequencies=self._frequencies[gp],
                gamma=gamma_eff[:, :, i_gp, :],
                heat_capacity_matrix=aggregates.heat_capacity_matrix[:, i_gp, :, :],
                vm_by_vm=aggregates.vm_by_vm[i_gp],
                conversion_factor=self._kappa_settings.conversion_factor,
            )
            self._mode_kappa_matrix[:, :, i_gp, :, :, :] = mode_kappa_matrix

        num_mesh_points = int(np.prod(self._kappa_settings.mesh_numbers))
        self._kappa = self._mode_kappa_matrix.sum(axis=(2, 3, 4)) / num_mesh_points
        self._kappa_intra = (
            np.einsum("abijjc->abc", self._mode_kappa_matrix) / num_mesh_points
        )

    @property
    def kappa(self) -> NDArray[np.double] | None:
        """Return total kappa (kappa_intra + kappa_inter).

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa

    @property
    def kappa_intra(self) -> NDArray[np.double] | None:
        """Return intra-band (diagonal) kappa.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa_intra

    @property
    def kappa_inter(self) -> NDArray[np.double] | None:
        """Return inter-band (off-diagonal) kappa.

        Shape: (num_sigma, num_temp, 6).

        """
        if self._kappa is None or self._kappa_intra is None:
            return None

        return self._kappa - self._kappa_intra

    @property
    def mode_kappa_matrix(self) -> NDArray[np.double] | None:
        """Return full band-pair kappa matrix.

        Shape: (num_gp, num_sigma, num_temp, num_band0, num_band, 6).

        """
        return self._mode_kappa_matrix

    def get_extra_kappa_output(self) -> dict[str, NDArray[np.double] | None]:
        """Return Kubo LBTE kappa arrays keyed by HDF5 dataset name."""
        return {
            "kappa_intra": self._kappa_intra,
            "kappa_inter": self.kappa_inter,
        }

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

        for i, sigma in enumerate(self._kappa_settings.sigmas):
            log_kappa_header(sigma, show_ipm=show_ipm)
            for j, t in enumerate(self._kappa_settings.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row(
                    "K_intra", t, self._kappa_intra[i, j], ipm, num_phonon_modes
                )
            print(" ")
            for j, t in enumerate(self._kappa_settings.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row("K_inter", t, kappa_inter[i, j], ipm, num_phonon_modes)
            print(" ")
            for j, t in enumerate(self._kappa_settings.temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                log_kappa_row("K_TOT", t, self._kappa[i, j], ipm, num_phonon_modes)
            print("", flush=True)


class KuboLBTEKappaSolver:
    """LBTE kappa solver with added Kubo inter-band kappa.

    Composes a CollisionMatrixKernel for the standard LBTE solve (intra-band)
    and adds the inter-band kappa from the Kubo formula using the collision
    matrix diagonal as effective linewidths.

    Stage 1 (per-grid-point): store() delegates collision data to the solver.

    Stage 2 (global): finalize() calls solver.solve() for the standard LBTE
    kappa (intra-band), then computes Kubo inter-band kappa.

    Parameters
    ----------
    solver : CollisionMatrixKernel
        Shared solver for the standard LBTE solve.
    kappa_settings : KappaSettings
        Shared computation metadata (grid, phonon, symmetry, configuration).
    log_level : int, optional
        Verbosity level. Default 0.

    """

    def __init__(
        self,
        solver: CollisionMatrixKernel,
        kappa_settings: KappaSettings,
        frequencies: NDArray[np.double],
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._solver = solver
        self._kappa_settings = kappa_settings
        self._frequencies = frequencies
        self._log_level = log_level

        num_sigma = len(kappa_settings.sigmas)
        num_temp = len(kappa_settings.temperatures)
        num_gp = len(kappa_settings.grid_points)
        num_band = frequencies.shape[1]
        self._mode_kappa_matrix = np.zeros(
            (num_sigma, num_temp, num_gp, num_band, num_band, 6),
            dtype="double",
            order="C",
        )
        self._kappa_inter = np.zeros((num_sigma, num_temp, 6), dtype="double")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
            Result from LBTECollisionSolver.compute().

        """
        self._solver.store(i_gp, collision_result)

    def finalize(
        self,
        aggregates: GridPointAggregates,
    ) -> None:
        """Finalize LBTE solve, then compute Kubo kappa with LBTE linewidths."""
        self._gamma_eff = compute_effective_gamma(aggregates)
        prev_sigma = -1
        for i_sigma, i_temp in self._solver.solve_iter(aggregates):
            self._compute_kubo_kappa_at(aggregates, i_sigma, i_temp)
            if self._log_level:
                if i_sigma != prev_sigma:
                    log_sigma_header(self._kappa_settings.sigmas[i_sigma])
                    prev_sigma = i_sigma
                self._log_kubo_kappa_at(i_sigma, i_temp)

    # ------------------------------------------------------------------
    # Properties — delegated to solver (LBTE intra-band results)
    # ------------------------------------------------------------------

    @property
    def solver(self) -> CollisionMatrixKernel:
        """Return the underlying CollisionMatrixKernel."""
        return self._solver

    # ------------------------------------------------------------------
    # Properties — Kubo-specific
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total kappa (kappa_intra_exact + kappa_inter).

        Shape: (num_sigma, num_temp, 6).

        """
        return self._solver.kappa + self._kappa_inter

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
        }

    # ------------------------------------------------------------------
    # Private: Kubo kappa computation
    # ------------------------------------------------------------------

    def _compute_kubo_kappa_at(
        self,
        aggregates: GridPointAggregates,
        i_sigma: int,
        i_temp: int,
    ) -> None:
        """Compute Kubo kappa for one (sigma, temperature) pair."""
        assert aggregates.vm_by_vm is not None
        assert aggregates.heat_capacity_matrix is not None

        for i_gp, gp in enumerate(self._kappa_settings.grid_points):
            mode_kappa_matrix = _compute_kubo_mode_kappa_matrix(
                frequencies=self._frequencies[gp],
                gamma=self._gamma_eff[i_sigma : i_sigma + 1, i_temp : i_temp + 1, i_gp],
                heat_capacity_matrix=aggregates.heat_capacity_matrix[
                    i_temp : i_temp + 1, i_gp
                ],
                vm_by_vm=aggregates.vm_by_vm[i_gp],
                conversion_factor=self._kappa_settings.conversion_factor,
            )
            self._mode_kappa_matrix[i_sigma, i_temp, i_gp] = mode_kappa_matrix[0, 0]

        num_mesh_points = int(np.prod(self._kappa_settings.mesh_numbers))
        mkm = self._mode_kappa_matrix[i_sigma, i_temp]
        kappa_total = mkm.sum(axis=(0, 1, 2)) / num_mesh_points
        kappa_intra = np.einsum("ijjc->c", mkm) / num_mesh_points

        self._kappa_inter[i_sigma, i_temp] = kappa_total - kappa_intra

    def _log_kubo_kappa_at(self, i_sigma: int, i_temp: int) -> None:
        """Print Kubo LBTE kappa for one (sigma, temperature) pair."""
        t = self._kappa_settings.temperatures[i_temp]
        if t <= 0:
            return

        kappa_intra = self._solver.kappa[i_sigma, i_temp]
        kappa_intra_RTA = self._solver.kappa_RTA[i_sigma, i_temp]
        kappa_inter = self._kappa_inter[i_sigma, i_temp]

        print(
            "#"
            + " " * 14
            + "T(K)        xx         yy         zz         yz         xz         xy"
        )
        print("K_intra     " + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_intra)))
        print(
            "K_intra_RTA "
            + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_intra_RTA))
        )
        print("K_inter     " + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_inter)))
        kappa_tot = kappa_intra + kappa_inter
        print("K_TOT       " + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(kappa_tot)))
        print("-" * 76, flush=True)
