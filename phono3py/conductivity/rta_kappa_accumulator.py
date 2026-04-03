"""RTAKappaAccumulator: kappa accumulator for the standard BTE-RTA method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.context import ConductivityContext
from phono3py.conductivity.grid_point_data import (
    GridPointAggregates,
    GridPointResult,
    compute_effective_gamma,
)
from phono3py.conductivity.utils import log_kappa_header, log_kappa_row


class RTAKappaAccumulator:
    """Kappa accumulator for the standard BTE diagonal formula.

    Computes mode kappa using the standard Boltzmann transport equation:

        kappa_mode = Cv * (v x v) * tau / 2
                   = Cv * gv_by_gv / (2 * gamma_eff) * unit_conversion

    where gamma_eff is the total effective linewidth (phonon-phonon + isotope
    + electron-phonon + boundary scattering).

    Parameters
    ----------
    context : ConductivityContext
        Shared computation metadata (grid, phonon, symmetry, configuration).
    conversion_factor : float
        Unit conversion factor to W/(m*K).
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

        # Allocated in prepare().
        self._kappa: NDArray[np.double]
        self._mode_kappa: NDArray[np.double]

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
        *,
        num_band: int | None = None,
    ) -> None:
        """Allocate per-grid-point and kappa arrays."""
        self._kappa = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._mode_kappa = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, 6), dtype="double", order="C"
        )

    def accumulate(self, i_gp: int, result: GridPointResult) -> None:
        """Store per-grid-point data and compute mode kappa at ``i_gp``."""
        assert result.group_velocities is not None
        assert result.gv_by_gv is not None
        assert result.heat_capacities is not None
        assert result.gamma is not None

        # Compute mode kappa.
        gv_by_gv = result.gv_by_gv
        cv = result.heat_capacities  # (num_temp, num_band0)
        frequencies = result.input.frequencies[result.input.band_indices]

        gamma = compute_effective_gamma(
            result.gamma,
            gamma_isotope=result.gamma_isotope,
            gamma_boundary=result.gamma_boundary,
            gamma_elph=result.gamma_elph,
        )  # (num_sigma, num_temp, num_band0)
        num_sigma, num_temp, num_band0 = gamma.shape
        mode_kappa = np.zeros((num_sigma, num_temp, num_band0, 6), dtype="double")

        for ll in range(num_band0):
            if frequencies[ll] < self._context.cutoff_frequency:
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
                        # g ~ 0 and |gv| = 0: contribution is zero
                        pass
                    except Exception:
                        print("=" * 26 + " Warning " + "=" * 26)
                        print(
                            " Unexpected physical condition of ph-ph "
                            "interaction calculation was found."
                        )
                        print(
                            " g=%f at gp=%d, band=%d, freq=%f"
                            % (
                                g,
                                result.input.grid_point,
                                ll + 1,
                                frequencies[ll],
                            )
                        )
                        print("=" * 61)
                    finally:
                        np.seterr(**old_settings)

        self._mode_kappa[:, :, i_gp, :, :] = mode_kappa

    def finalize(self, aggregates: GridPointAggregates) -> None:
        """Compute kappa from mode_kappa."""
        num_sampling_grid_points = aggregates.num_sampling_grid_points
        if num_sampling_grid_points > 0:
            self._kappa = (
                np.sum(self._mode_kappa, axis=(2, 3)) / num_sampling_grid_points
            )

    def log_kappa(
        self,
        num_ignored_phonon_modes: NDArray[np.int64] | None = None,
        num_phonon_modes: int | None = None,
    ) -> None:
        """Print kappa table after finalization."""
        if not self._log_level:
            return
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
                log_kappa_row("", t, self._kappa[i, j], ipm, num_phonon_modes)
            print("", flush=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return kappa tensor, shape (num_sigma, num_temp, 6)."""
        return self._kappa

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._mode_kappa
