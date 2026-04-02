"""Kappa formula for the Green-Kubo method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import (
    GridPointResult,
    compute_effective_gamma,
)


def compute_kubo_mode_kappa_mat(
    frequencies: NDArray[np.double],
    gamma: NDArray[np.double],
    heat_capacity_matrix: NDArray[np.double],
    velocity_product: NDArray[np.cdouble],
    cutoff_frequency: float,
    conversion_factor: float,
    grid_point: int = -1,
) -> NDArray[np.double]:
    """Compute Kubo mode-kappa matrix for one grid point.

    Core band-pair loop shared by RTA (KuboKappaFormula) and LBTE
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
    velocity_product : ndarray of complex, shape (num_band0, num_band, 6)
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
    num_sigma, num_temp, num_band0 = gamma.shape
    num_band = velocity_product.shape[1]

    mode_kappa_mat = np.zeros(
        (num_sigma, num_temp, num_band0, num_band, 6), dtype="double"
    )

    for j in range(num_sigma):
        for k in range(num_temp):
            for i_band in range(num_band0):
                if frequencies[i_band] < cutoff_frequency:
                    continue
                g_i = gamma[j, k, i_band]
                for j_band in range(num_band):
                    if frequencies[j_band] < cutoff_frequency:
                        break
                    g = g_i + gamma[j, k, j_band]
                    delta_omega = frequencies[j_band] - frequencies[i_band]
                    denom = delta_omega**2 + g**2
                    old_settings = np.seterr(all="raise")
                    try:
                        contribution = (
                            heat_capacity_matrix[k, i_band, j_band]
                            * velocity_product[i_band, j_band]
                            * g
                            / denom
                            * conversion_factor
                        ).real
                    except FloatingPointError:
                        contribution = None
                    except Exception:
                        print("=" * 26 + " Warning " + "=" * 26)
                        print(
                            " Unexpected physical condition of ph-ph "
                            "interaction calculation was found."
                        )
                        print(
                            " g=%f at gp=%d, band=%d, freq=%f, band=%d, freq=%f"
                            % (
                                g_i,
                                grid_point,
                                i_band + 1,
                                frequencies[i_band],
                                j_band + 1,
                                frequencies[j_band],
                            )
                        )
                        print("=" * 61)
                        contribution = None
                    finally:
                        np.seterr(**old_settings)

                    if contribution is not None:
                        mode_kappa_mat[j, k, i_band, j_band] = contribution

    return mode_kappa_mat


class KuboKappaFormula:
    """Compute mode-kappa contribution at a single grid point (Green-Kubo formula).

    The Green-Kubo formula for each band pair (s, s'):

        kappa_mat[s, s'] = C_{s s'} * GVM_{s s'} * g / ((omega_s' - omega_s)^2 + g^2)

    where g = gamma_s + gamma_s' is the sum of half-linewidths and all
    frequencies are in THz.

    Parameters
    ----------
    cutoff_frequency : float
        Modes with frequency below this value (in THz) are skipped.
    conversion_factor : float
        Unit conversion factor to W/(m*K).

    Notes
    -----
    Assumes ``num_band0 == num_band`` (all phonon branches selected), which is
    required so that ``gamma[:, :, j_band]`` can index any band.

    The effective linewidth combines all diagonal scattering contributions:

        gamma_eff[s] = gamma[s] + gamma_isotope[s] + gamma_boundary[s] + gamma_elph[s]

    """

    def __init__(self, cutoff_frequency: float, conversion_factor: float):
        """Init method."""
        self._cutoff_frequency = cutoff_frequency
        self._conversion_factor = conversion_factor

    def compute(self, result: GridPointResult) -> NDArray[np.double]:
        """Compute Kubo mode-kappa at this grid point.

        Parameters
        ----------
        result : GridPointResult
            Must have ``velocity_product`` (num_band0, num_band, 6) complex,
            ``heat_capacity_matrix`` (num_temp, num_band0, num_band), and
            ``gamma`` (num_sigma, num_temp, num_band0) populated.
            Requires ``num_band0 == num_band``.

        Returns
        -------
        mode_kappa_mat : ndarray of double,
            shape (num_sigma, num_temp, num_band0, num_band, 6)
            Per-(sigma, temp, i_band, j_band) kappa contribution in Voigt order.

        """
        assert result.velocity_product is not None
        assert result.heat_capacity_matrix is not None
        assert result.gamma is not None

        gamma = compute_effective_gamma(result)  # (num_sigma, num_temp, num_band0)
        num_band0 = gamma.shape[2]
        num_band = result.velocity_product.shape[1]
        assert num_band0 == num_band, (
            "KuboKappaFormula requires num_band0 == num_band "
            f"(got {num_band0} vs {num_band})."
        )

        return compute_kubo_mode_kappa_mat(
            frequencies=result.input.frequencies,
            gamma=gamma,
            heat_capacity_matrix=result.heat_capacity_matrix,
            velocity_product=result.velocity_product,
            cutoff_frequency=self._cutoff_frequency,
            conversion_factor=self._conversion_factor,
            grid_point=result.input.grid_point,
        )
