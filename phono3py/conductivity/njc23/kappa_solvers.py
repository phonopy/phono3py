"""Kappa solvers for the NJC23 (Green-Kubo) method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import GridPointAggregates


def _compute_njc23_mode_kappa_matrix(
    frequencies: NDArray[np.double],
    gamma: NDArray[np.double],
    heat_capacity_matrix: NDArray[np.double],
    vm_by_vm: NDArray[np.cdouble],
    conversion_factor: float,
) -> NDArray[np.double]:
    """Compute NJC23 mode-kappa matrix for one grid point.

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


def compute_njc23_mode_kappa(
    frequencies: NDArray[np.double],
    gamma: NDArray[np.double],
    aggregates: GridPointAggregates,
    i_gp: int,
    conversion_factor: float,
) -> NDArray[np.double]:
    """Adapter for InterBandRTA/LBTEKappaSolver."""
    assert aggregates.vm_by_vm is not None
    assert aggregates.heat_capacity_matrix is not None
    return _compute_njc23_mode_kappa_matrix(
        frequencies=frequencies,
        gamma=gamma,
        heat_capacity_matrix=aggregates.heat_capacity_matrix[:, i_gp, :, :],
        vm_by_vm=aggregates.vm_by_vm[i_gp],
        conversion_factor=conversion_factor,
    )
