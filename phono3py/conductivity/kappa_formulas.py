"""Kappa formula building blocks for conductivity calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import (
    GridPointResult,
    compute_effective_gamma,
)


class KappaFormula:
    """Compute mode-kappa contribution at a single grid point (standard BTE).

    This formula implements the ``KappaFormula`` protocol for the standard
    Boltzmann transport equation:

        kappa_mode = Cv * (v x v) * tau / 2
                   = Cv * gv_by_gv / (2 * gamma_eff) * unit_conversion

    where ``gamma_eff`` is the total effective linewidth (phonon-phonon + isotope
    + electron-phonon + boundary scattering) stored in
    ``GridPointResult.gamma``.

    Parameters
    ----------
    cutoff_frequency : float
        Modes with frequency below this value (in THz) are skipped.
    conversion_factor : float
        Unit conversion factor to W/(m*K).

    Notes
    -----
    The effective linewidth is computed internally as:

        gamma_eff = gamma + gamma_isotope + gamma_boundary + gamma_elph

    Only ``gamma`` is required; the other contributions are optional and
    added when present.

    """

    def __init__(self, cutoff_frequency: float, conversion_factor: float):
        """Init method."""
        self._cutoff_frequency = cutoff_frequency
        self._conversion_factor = conversion_factor

    def compute(self, result: GridPointResult) -> NDArray[np.double]:
        """Compute mode-kappa contribution at this grid point.

        Parameters
        ----------
        result : GridPointResult
            Must have ``velocity_product`` (num_band0, 6),
            ``heat_capacities`` (num_temp, num_band0), and
            ``gamma`` (num_sigma, num_temp, num_band0) populated.

        Returns
        -------
        mode_kappa : ndarray of double, shape (num_sigma, num_temp, num_band0, 6)
            Mode-resolved kappa contribution.  Modes below cutoff are zero.

        """
        assert result.velocity_product is not None
        assert result.heat_capacities is not None
        assert result.gamma is not None

        gv_by_gv = result.velocity_product  # (num_band0, 6)
        cv = result.heat_capacities  # (num_temp, num_band0)
        frequencies = result.input.frequencies[result.input.band_indices]

        gamma = compute_effective_gamma(result)  # (num_sigma, num_temp, num_band0)
        num_sigma, num_temp, num_band0 = gamma.shape
        mode_kappa = np.zeros((num_sigma, num_temp, num_band0, 6), dtype="double")

        for ll in range(num_band0):
            if frequencies[ll] < self._cutoff_frequency:
                continue
            for j in range(num_sigma):
                for k in range(num_temp):
                    g = gamma[j, k, ll]
                    old_settings = np.seterr(all="raise")
                    try:
                        mode_kappa[j, k, ll] = (
                            gv_by_gv[ll] * cv[k, ll] / (g * 2) * self._conversion_factor
                        )
                    except FloatingPointError:
                        # g ≈ 0 and |gv| = 0: contribution is zero
                        pass
                    except Exception:
                        print("=" * 26 + " Warning " + "=" * 26)
                        print(
                            " Unexpected physical condition of ph-ph "
                            "interaction calculation was found."
                        )
                        print(
                            " g=%f at gp=%d, band=%d, freq=%f"
                            % (g, result.input.grid_point, ll + 1, frequencies[ll])
                        )
                        print("=" * 61)
                    finally:
                        np.seterr(**old_settings)

        return mode_kappa
