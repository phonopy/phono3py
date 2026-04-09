"""Kappa solvers for the SMM19 method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import GridPointAggregates


def _compute_smm19_mode_kappa_matrix(
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


def compute_smm19_mode_kappa(
    frequencies: NDArray[np.double],
    gamma: NDArray[np.double],
    aggregates: GridPointAggregates,
    i_gp: int,
    conversion_factor: float,
) -> NDArray[np.double]:
    """Adapter for InterBandRTA/LBTEKappaSolver."""
    assert aggregates.vm_by_vm is not None
    assert aggregates.mode_heat_capacities is not None
    return _compute_smm19_mode_kappa_matrix(
        frequencies=frequencies,
        gamma=gamma,
        heat_capacity=aggregates.mode_heat_capacities[:, i_gp, :],
        vm_by_vm=aggregates.vm_by_vm[i_gp],
        conversion_factor=conversion_factor,
    )
