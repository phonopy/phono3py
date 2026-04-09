"""Kappa solvers for the SMM19 method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.build_components import KappaSettings
from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
    compute_effective_gamma,
)
from phono3py.conductivity.utils import log_kappa_header, log_kappa_row


def _compute_SMM19_mode_kappa_matrix(
    frequencies: NDArray[np.double],
    gamma: NDArray[np.double],
    heat_capacity: NDArray[np.double],
    vm_by_vm: NDArray[np.cdouble],
    conversion_factor: float,
) -> NDArray[np.double]:
    """Compute SMM19 mode-kappa matrix for one grid point.

    The SMM19 formula for each band pair (s, s'):

        kappa_mat[s, s'] =
          1/4 * (w_s + w_s') * (C[s]/w_s + C[s']/w_s')
          * V[s,s']*V*[s,s'] * g_sum / ((w_s - w_s')^2 + g_sum^2)

    where g = gamma_s + gamma_s' is the sum of half-linewidths and all
    frequencies are in THz.

    Parameters
    ----------
    frequencies : ndarray of double, shape (num_band,)
        Phonon frequencies at this grid point in THz.
    gamma : ndarray of double, shape (num_sigma, num_temp, num_band)
        Effective scattering half-linewidths.
    heat_capacity : ndarray of double, shape (num_temp, num_band)
        Heat capacity for each band in eV/K.
    vm_by_vm : ndarray of complex, shape (num_band0, num_band, 6)
        k-star-averaged velocity outer product in Voigt order.
    conversion_factor : float
        Unit conversion factor to W/(m*K).

    Returns
    -------
    mode_kappa_mat : ndarray of double,
        shape (num_sigma, num_temp, num_band0, num_band, 6)

    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cp_per_freq = heat_capacity / frequencies[None, :]  # tb
        cp_per_freq = np.where(np.isfinite(cp_per_freq), cp_per_freq, 0)

    freq_sum = frequencies[:, None] + frequencies[None, :]  # bb
    cp_per_freq_mat = cp_per_freq[:, :, None] + cp_per_freq[:, None, :]  # tbb
    cv_mat = cp_per_freq_mat * freq_sum[None, :, :]  # tbb

    g_sum = gamma[:, :, :, None] + gamma[:, :, None, :]  # stbb
    delta_omega = frequencies[None, None, :, None] - frequencies[None, None, None, :]
    denom = delta_omega**2 + g_sum**2  # stbb
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        kappa_mat = (
            cv_mat[None, :, :, :, None]
            * vm_by_vm.real[None, None, :, :, :]
            * g_sum[:, :, :, :, None]
            / denom[:, :, :, :, None]
            * (conversion_factor / 4.0)
        )
        kappa_mat = np.where(np.isfinite(kappa_mat), kappa_mat, 0)

    return kappa_mat


class SMM19RTAKappaSolver:
    """Kappa solver for the SMM19 formula.

    Computes the full band-pair kappa matrix ``mode_kappa_inter``.
    The ``kappa`` property sums over all band pairs.

    See the formula in _compute_SMM19_mode_kappa_matrix().

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
        assert aggregates.mode_heat_capacities is not None

        gamma_eff = compute_effective_gamma(aggregates)
        num_sigma, num_temp, num_gp, num_band = gamma_eff.shape

        self._mode_kappa_matrix = np.zeros(
            (num_sigma, num_temp, num_gp, num_band, num_band, 6),
            dtype="double",
            order="C",
        )

        for i_gp, gp in enumerate(self._kappa_settings.grid_points):
            mode_kappa_matrix = _compute_SMM19_mode_kappa_matrix(
                frequencies=self._frequencies[gp],
                gamma=gamma_eff[:, :, i_gp, :],
                heat_capacity=aggregates.mode_heat_capacities[:, i_gp, :],
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
            "mode_kappa_matrix": self._mode_kappa_matrix,
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
