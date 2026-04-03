"""Shared helpers for the Green-Kubo method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FLOATINGPOINTERROR_THRESHOLD = 1e-12


def compute_kubo_mode_kappa_matrix(
    frequencies: NDArray[np.double],
    gamma: NDArray[np.double],
    heat_capacity_matrix: NDArray[np.double],
    vm_by_vm: NDArray[np.cdouble],
    cutoff_frequency: float,
    conversion_factor: float,
) -> NDArray[np.double]:
    """Compute Kubo mode-kappa matrix for one grid point.

    Core band-pair loop shared by RTA (KuboRTAKappaAccumulator) and LBTE
    (KuboLBTEKappaAccumulator).

    The Green-Kubo formula for each band pair (s, s'):

        kappa_mat[s, s'] = C[s,s'] * V[s,s']*V*[s,s'] * g / (dw^2 + g^2)

    where g = gamma_s + gamma_s' is the sum of half-linewidths and all
    frequencies are in THz.

    Parameters
    ----------
    frequencies : ndarray of double, shape (num_band,)
        Phonon frequencies at this grid point in THz.
    gamma : ndarray of double, shape (num_sigma, num_temp, num_band)
        Effective scattering half-linewidths. For RTA this is the
        sum of ph-ph, isotope, boundary, and elph contributions.
        For LBTE this comes from the collision matrix diagonal.
    heat_capacity_matrix : ndarray of double, shape (num_temp, num_band0, num_band)
        Off-diagonal heat capacity matrix C_{ss'}.
    vm_by_vm : ndarray of complex, shape (num_band0, num_band, 6)
        k-star-averaged velocity outer product in Voigt order.
    cutoff_frequency : float
        Modes with frequency below this value (in THz) are skipped.
    conversion_factor : float
        Unit conversion factor to W/(m*K).
    grid_point : int, optional
        Grid point index for warning messages. Default -1.

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
                > FLOATINGPOINTERROR_THRESHOLD
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
