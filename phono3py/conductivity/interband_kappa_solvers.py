"""Shared kappa solvers for inter-band methods (SMM19, NJC23, etc.)."""

from __future__ import annotations

from typing import Callable

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

ComputeModeKappaFn = Callable[
    [
        NDArray[np.double],  # frequencies at grid point
        NDArray[np.double],  # gamma slice
        GridPointAggregates,  # aggregates
        int,  # i_gp
        float,  # conversion_factor
    ],
    NDArray[np.double],
]


def _fill_mode_kappa_matrix(
    kappa_settings: KappaSettings,
    frequencies: NDArray[np.double],
    gamma_eff: NDArray[np.double],
    aggregates: GridPointAggregates,
    compute_mode_kappa: ComputeModeKappaFn,
) -> NDArray[np.double]:
    """Compute mode-kappa matrix for all grid points.

    Returns
    -------
    mode_kappa_matrix : ndarray of double,
        shape (num_sigma, num_temp, num_gp, num_band, num_band, 6)

    """
    num_sigma, num_temp, num_gp, num_band = gamma_eff.shape
    mode_kappa_matrix = np.zeros(
        (num_sigma, num_temp, num_gp, num_band, num_band, 6),
        dtype="double",
        order="C",
    )
    for i_gp, gp in enumerate(kappa_settings.grid_points):
        mode_kappa_matrix[:, :, i_gp, :, :, :] = compute_mode_kappa(
            frequencies[gp],
            gamma_eff[:, :, i_gp, :],
            aggregates,
            i_gp,
            kappa_settings.conversion_factor,
        )
    return mode_kappa_matrix


class InterBandRTAKappaSolver:
    """RTA kappa solver for inter-band methods (SMM19, NJC23, etc.).

    The per-grid-point mode-kappa computation is delegated to the callable
    ``compute_mode_kappa`` passed at construction time.

    Parameters
    ----------
    kappa_settings : KappaSettings
        Shared computation metadata (grid, phonon, symmetry, configuration).
    frequencies : ndarray of double
        Phonon frequencies, shape (num_full_gp, num_band).
    compute_mode_kappa : ComputeModeKappaFn
        Callable that computes the mode-kappa matrix for one grid point.
    log_level : int
        Verbosity level.

    """

    def __init__(
        self,
        kappa_settings: KappaSettings,
        frequencies: NDArray[np.double],
        compute_mode_kappa: ComputeModeKappaFn,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._kappa_settings = kappa_settings
        self._frequencies = frequencies
        self._compute_mode_kappa = compute_mode_kappa
        self._log_level = log_level

        self._kappa: NDArray[np.double] | None = None
        self._kappa_intra: NDArray[np.double] | None = None
        self._mode_kappa_matrix: NDArray[np.double] | None = None

    def finalize(self, aggregates: GridPointAggregates) -> None:
        """Compute kappa and kappa_intra from aggregated data."""
        gamma_eff = compute_effective_gamma(aggregates)

        self._mode_kappa_matrix = _fill_mode_kappa_matrix(
            self._kappa_settings,
            self._frequencies,
            gamma_eff,
            aggregates,
            self._compute_mode_kappa,
        )

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

        Shape: (num_sigma, num_temp, num_gp, num_band0, num_band, 6).

        """
        return self._mode_kappa_matrix

    def get_extra_kappa_output(self) -> dict[str, NDArray[np.double] | None]:
        """Return extra kappa arrays keyed by HDF5 dataset name."""
        return {
            "kappa_intra": self._kappa_intra,
            "kappa_inter": self.kappa_inter,
        }

    def log_kappa(
        self,
        num_ignored_phonon_modes: NDArray[np.int64] | None = None,
        num_phonon_modes: int | None = None,
    ) -> None:
        """Print K_intra, K_inter, K_TOT rows for the RTA conductivity."""
        if not self._log_level:
            return

        assert self._kappa is not None
        assert self._kappa_intra is not None
        kappa_inter = self._kappa - self._kappa_intra
        show_ipm = (
            self._log_level > 1
            and num_ignored_phonon_modes is not None
            and num_phonon_modes is not None
        )

        rows = [
            ("K_intra ", self._kappa_intra),
            ("K_inter ", kappa_inter),
            ("K_TOT   ", self._kappa),
        ]
        for i, sigma in enumerate(self._kappa_settings.sigmas):
            log_kappa_header(sigma, show_ipm=show_ipm, num_spaces=10)
            for label, values in rows:
                for j, t in enumerate(self._kappa_settings.temperatures):
                    ipm = (
                        int(num_ignored_phonon_modes[i, j])
                        if show_ipm and num_ignored_phonon_modes is not None
                        else None
                    )
                    log_kappa_row(label, t, values[i, j], ipm, num_phonon_modes)
                print(" " if label != "K_TOT" else "", flush=label == "K_TOT")


class InterBandLBTEKappaSolver:
    """LBTE kappa solver for inter-band methods (SMM19, NJC23, etc.).

    Composes a CollisionMatrixKernel for the standard LBTE solve (intra-band)
    and adds the inter-band kappa computed by the callable
    ``compute_mode_kappa`` passed at construction time.

    Parameters
    ----------
    solver : CollisionMatrixKernel
        Shared solver for the standard LBTE solve.
    kappa_settings : KappaSettings
        Shared computation metadata (grid, phonon, symmetry, configuration).
    frequencies : ndarray of double
        Phonon frequencies, shape (num_full_gp, num_band).
    compute_mode_kappa : ComputeModeKappaFn
        Callable that computes the mode-kappa matrix for one grid point.
    log_level : int, optional
        Verbosity level. Default 0.

    """

    def __init__(
        self,
        solver: CollisionMatrixKernel,
        kappa_settings: KappaSettings,
        frequencies: NDArray[np.double],
        compute_mode_kappa: ComputeModeKappaFn,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._solver = solver
        self._kappa_settings = kappa_settings
        self._frequencies = frequencies
        self._compute_mode_kappa = compute_mode_kappa
        self._log_level = log_level

        num_sigma = len(kappa_settings.sigmas)
        num_temp = len(kappa_settings.temperatures)
        self._kappa_inter = np.zeros((num_sigma, num_temp, 6), dtype="double")
        self._mode_kappa_matrix: NDArray[np.double] | None = None

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
        """Finalize LBTE solve, then compute inter-band kappa."""
        gamma_eff = compute_effective_gamma(aggregates)

        # Pre-compute mode_kappa_matrix for all (sigma, temp, gp) at once.
        # This depends only on RTA gamma, not on the LBTE solution.
        self._mode_kappa_matrix = _fill_mode_kappa_matrix(
            self._kappa_settings,
            self._frequencies,
            gamma_eff,
            aggregates,
            self._compute_mode_kappa,
        )

        num_mesh_points = int(np.prod(self._kappa_settings.mesh_numbers))
        prev_sigma = -1
        for i_sigma, i_temp in self._solver.solve_iter(aggregates):
            mkm = self._mode_kappa_matrix[i_sigma, i_temp]
            kappa_total = mkm.sum(axis=(0, 1, 2)) / num_mesh_points
            kappa_intra = np.einsum("ijjc->c", mkm) / num_mesh_points
            self._kappa_inter[i_sigma, i_temp] = kappa_total - kappa_intra
            if self._log_level:
                if i_sigma != prev_sigma:
                    log_sigma_header(self._kappa_settings.sigmas[i_sigma])
                    prev_sigma = i_sigma
                self._log_lbte_kappa_at(i_sigma, i_temp)

    # ------------------------------------------------------------------
    # Properties -- delegated to solver (LBTE intra-band results)
    # ------------------------------------------------------------------

    @property
    def solver(self) -> CollisionMatrixKernel:
        """Return the underlying CollisionMatrixKernel."""
        return self._solver

    # ------------------------------------------------------------------
    # Properties
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
        """Return inter-band (off-diagonal) kappa.

        Shape: (num_sigma, num_temp, 6).

        """
        return self._kappa_inter

    @property
    def mode_kappa_matrix(self) -> NDArray[np.double] | None:
        """Return full band-pair kappa matrix.

        Shape: (num_sigma, num_temp, num_gp, num_band0, num_band, 6).

        """
        return self._mode_kappa_matrix

    def get_extra_kappa_output(self) -> dict[str, NDArray[np.double] | None]:
        """Return extra LBTE kappa arrays keyed by HDF5 dataset name."""
        return {
            "kappa_inter": self._kappa_inter,
            "kappa_intra_exact": self._solver.kappa,
            "kappa_intra_RTA": self._solver.kappa_RTA,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _log_lbte_kappa_at(self, i_sigma: int, i_temp: int) -> None:
        """Print LBTE kappa for one (sigma, temperature) pair."""
        t = self._kappa_settings.temperatures[i_temp]
        if t <= 0:
            return

        kappa_intra = self._solver.kappa[i_sigma, i_temp]
        kappa_intra_RTA = self._solver.kappa_RTA[i_sigma, i_temp]
        kappa_inter = self._kappa_inter[i_sigma, i_temp]

        sigma = self._kappa_settings.sigmas[i_sigma]
        log_kappa_header(sigma, num_spaces=14)
        for label, values in [
            ("K_intra     ", kappa_intra),
            ("K_intra_RTA ", kappa_intra_RTA),
            ("K_inter     ", kappa_inter),
            ("K_TOT       ", kappa_intra + kappa_inter),
        ]:
            print(label + ("%7.1f " + " %10.3f" * 6) % ((t,) + tuple(values)))
        print("-" * 76, flush=True)
