"""Kappa formula for the Wigner transport equation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.grid_point_data import (
    GridPointResult,
    compute_effective_gamma,
)

# Threshold in THz below which two modes are considered degenerate and treated
# as a population (diagonal) term instead of a coherence (off-diagonal) term.
DEGENERATE_FREQUENCY_THRESHOLD_THZ = 1e-4


def get_conversion_factor_WTE(volume: float) -> float:
    """Return unit conversion factor for Wigner transport equation kappa.

    Parameters
    ----------
    volume : float
        Primitive-cell volume in Angstrom^3.

    Returns
    -------
    float
        Conversion factor in W/(m*K).
    """
    u = get_physical_units()
    return (
        (u.THz * u.Angstrom) ** 2  # group velocity squared
        * u.EV  # specific heat in eV/K
        * u.Hbar  # Lorentzian eV^-1 to s
        / (volume * u.Angstrom**3)  # unit cell volume
    )


class WignerKappaFormula:
    """Compute mode-kappa contribution at a single grid point (Wigner transport).

    This formula implements the Wigner transport equation (WTE) kappa, which
    decomposes into a population term kappa_P and a coherence term kappa_C:

        kappa_TOT = kappa_P + kappa_C

    The population term kappa_P is returned directly from ``compute()``.
    The coherence term is stored in ``result.extra["wigner_mode_kappa_C"]`` as a
    side effect.

    Parameters
    ----------
    cutoff_frequency : float
        Modes with frequency below this value (in THz) are skipped.
    conversion_factor_WTE : float
        Unit conversion factor to W/(m*K) for the WTE formula; see
        ``get_conversion_factor_WTE``.

    Notes
    -----
    Assumes ``num_band0 == num_band`` (all phonon branches), which is the
    standard usage for Wigner calculations.

    The effective linewidth combines all diagonal scattering contributions:

        gamma_eff = gamma + gamma_isotope + gamma_boundary + gamma_elph

    """

    def __init__(self, cutoff_frequency: float, conversion_factor_WTE: float):
        """Init method."""
        self._cutoff_frequency = cutoff_frequency
        self._conversion_factor_WTE = conversion_factor_WTE

    def compute(self, result: GridPointResult) -> NDArray[np.double]:
        """Compute Wigner mode-kappa population and coherence terms.

        Parameters
        ----------
        result : GridPointResult
            Must have ``velocity_product`` (num_band0, num_band, 6) complex,
            ``heat_capacities`` (num_temp, num_band0), and
            ``gamma`` (num_sigma, num_temp, num_band0) populated.
            ``input.frequencies`` must contain all-band frequencies.

        Returns
        -------
        mode_kappa_P : ndarray of double, shape (num_sigma, num_temp, num_band0, 6)
            Population contribution to mode kappa.

        Side effects
        ------------
        Sets ``result.extra["wigner_mode_kappa_C"]`` to the coherence
        contribution, shape (num_sigma, num_temp, num_band0, num_band, 6).

        """
        assert result.velocity_product is not None
        assert result.heat_capacities is not None
        assert result.gamma is not None

        frequencies = result.input.frequencies  # (num_band,) all bands
        gv_by_gv = result.velocity_product  # (num_band0, num_band, 6) complex
        cv = result.heat_capacities  # (num_temp, num_band0)

        gamma = compute_effective_gamma(result)  # (num_sigma, num_temp, num_band0)
        num_sigma, num_temp, num_band0 = gamma.shape
        num_band = len(frequencies)
        THzToEv = get_physical_units().THzToEv

        mode_kappa_P = np.zeros((num_sigma, num_temp, num_band0, 6), dtype="double")
        mode_kappa_C = np.zeros(
            (num_sigma, num_temp, num_band0, num_band, 6), dtype="double"
        )

        for j in range(num_sigma):
            for k in range(num_temp):
                g = gamma[j, k]  # (num_band0,)
                cv_k = cv[k]  # (num_band0,)
                for s1 in range(num_band0):
                    freq_s1 = frequencies[s1]
                    if freq_s1 <= self._cutoff_frequency:
                        continue
                    for s2 in range(num_band):
                        freq_s2 = frequencies[s2]
                        if freq_s2 <= self._cutoff_frequency:
                            continue
                        pair = self._get_pair_contribution(
                            freq_s1=freq_s1,
                            freq_s2=freq_s2,
                            g_s1=g[s1],
                            g_s2=g[s2],
                            cv_s1=cv_k[s1],
                            cv_s2=cv_k[s2],
                            gv_by_gv_s1s2=gv_by_gv[s1, s2],
                            THzToEv=THzToEv,
                        )
                        if pair is None:
                            continue
                        contribution, is_population = pair
                        if is_population:
                            mode_kappa_P[j, k, s1] += 0.5 * contribution
                            mode_kappa_P[j, k, s2] += 0.5 * contribution
                        else:
                            mode_kappa_C[j, k, s1, s2] += contribution

        result.extra["wigner_mode_kappa_C"] = mode_kappa_C
        return mode_kappa_P

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_pair_contribution(
        self,
        *,
        freq_s1: float,
        freq_s2: float,
        g_s1: float,
        g_s2: float,
        cv_s1: float,
        cv_s2: float,
        gv_by_gv_s1s2: NDArray[np.cdouble],
        THzToEv: float,
    ) -> tuple[NDArray[np.double], bool] | None:
        """Return the kappa contribution of a (s1, s2) mode pair.

        Returns None when either mode is below the cutoff (handled by the
        caller) or when the contribution is identically zero.

        Returns
        -------
        (contribution, is_population) or None
            contribution : (6,) real array
            is_population : True when the pair is treated as a population
                            term (|omega_s1 - omega_s2| <
                            DEGENERATE_FREQUENCY_THRESHOLD_THZ).

        """
        hbar_omega_s1 = freq_s1 * THzToEv
        hbar_omega_s2 = freq_s2 * THzToEv
        hbar_gamma_s1 = 2.0 * g_s1 * THzToEv
        hbar_gamma_s2 = 2.0 * g_s2 * THzToEv

        gamma_sum = hbar_gamma_s1 + hbar_gamma_s2
        delta_omega = hbar_omega_s1 - hbar_omega_s2
        denominator = delta_omega**2 + 0.25 * gamma_sum**2
        if denominator == 0.0:
            return None
        lorentzian_div_hbar = (0.5 * gamma_sum) / denominator

        prefactor = (
            0.25
            * (hbar_omega_s1 + hbar_omega_s2)
            * (cv_s1 / hbar_omega_s1 + cv_s2 / hbar_omega_s2)
        )
        contribution = (
            gv_by_gv_s1s2
            * prefactor
            * lorentzian_div_hbar
            * self._conversion_factor_WTE
        ).real
        is_population = abs(freq_s1 - freq_s2) < DEGENERATE_FREQUENCY_THRESHOLD_THZ
        return contribution, is_population
