"""RTAKappaAccumulator: kappa accumulator for the standard BTE-RTA method."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import (
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
    cutoff_frequency : float
        Modes with frequency below this value (in THz) are skipped.
    conversion_factor : float
        Unit conversion factor to W/(m*K).
    temperatures : array-like
        Temperature values in Kelvin.
    sigmas : sequence
        Smearing widths (None for tetrahedron method).
    log_level : int
        Verbosity level.

    """

    def __init__(
        self,
        cutoff_frequency: float,
        conversion_factor: float,
        temperatures: NDArray[np.double] | Sequence[float] | None = None,
        sigmas: Sequence[float | None] | None = None,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._cutoff_frequency = cutoff_frequency
        self._conversion_factor = conversion_factor
        self._temperatures = temperatures
        self._sigmas: list[float | None] = [] if sigmas is None else list(sigmas)
        self._log_level = log_level

        # Allocated in prepare().
        self._gv: NDArray[np.double]
        self._gv_by_gv: NDArray[np.double]
        self._cv: NDArray[np.double]
        self._gamma: NDArray[np.double]
        self._gamma_iso: NDArray[np.double] | None = None
        self._averaged_pp_interaction: NDArray[np.double] | None = None
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
        is_full_pp: bool = False,
    ) -> None:
        """Allocate per-grid-point and kappa arrays."""
        self._gv = np.zeros((num_gp, num_band0, 3), dtype="double", order="C")
        self._gv_by_gv = np.zeros((num_gp, num_band0, 6), dtype="double", order="C")
        self._cv = np.zeros((num_temp, num_gp, num_band0), dtype="double", order="C")
        self._gamma = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0), dtype="double", order="C"
        )
        if is_full_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_gp, num_band0), dtype="double", order="C"
            )
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

        # Store per-grid-point data.
        self._gv[i_gp] = result.group_velocities
        self._gv_by_gv[i_gp] = result.gv_by_gv  # (num_band0, 6)
        self._cv[:, i_gp, :] = result.heat_capacities
        self._gamma[:, :, i_gp, :] = result.gamma
        if result.gamma_isotope is not None:
            if self._gamma_iso is None:
                ns, _, nb = (
                    result.gamma_isotope.shape[0],
                    self._gv.shape[0],
                    self._gv.shape[1],
                )
                self._gamma_iso = np.zeros(
                    (ns, self._gv.shape[0], nb), dtype="double", order="C"
                )
            self._gamma_iso[:, i_gp, :] = result.gamma_isotope
        if (
            result.averaged_pp_interaction is not None
            and self._averaged_pp_interaction is not None
        ):
            self._averaged_pp_interaction[i_gp] = result.averaged_pp_interaction

        # Compute mode kappa.
        gv_by_gv = result.gv_by_gv
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

    def finalize(self, num_sampling_grid_points: int) -> None:
        """Compute kappa from mode_kappa."""
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
        if not self._log_level or self._temperatures is None:
            return
        show_ipm = (
            self._log_level > 1
            and num_ignored_phonon_modes is not None
            and num_phonon_modes is not None
        )
        for i, sigma in enumerate(self._sigmas):
            log_kappa_header(sigma, show_ipm=show_ipm)
            for j, t in enumerate(self._temperatures):
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
    def group_velocities(self) -> NDArray[np.double]:
        """Return group velocities, shape (num_gp, num_band0, 3)."""
        return self._gv

    @property
    def gv_by_gv(self) -> NDArray[np.double]:
        """Return symmetrised v-outer-v, shape (num_gp, num_band0, 6)."""
        return self._gv_by_gv

    @property
    def mode_heat_capacities(self) -> NDArray[np.double]:
        """Return mode heat capacities, shape (num_temp, num_gp, num_band0)."""
        return self._cv

    @property
    def gamma(self) -> NDArray[np.double]:
        """Return ph-ph gamma, shape (num_sigma, num_temp, num_gp, num_band0)."""
        return self._gamma

    @gamma.setter
    def gamma(self, value: NDArray[np.double]) -> None:
        self._gamma = value

    @property
    def gamma_isotope(self) -> NDArray[np.double] | None:
        """Return isotope gamma, shape (num_sigma, num_gp, num_band0)."""
        return self._gamma_iso

    @gamma_isotope.setter
    def gamma_isotope(self, value: NDArray[np.double] | None) -> None:
        self._gamma_iso = value

    @property
    def averaged_pp_interaction(self) -> NDArray[np.double] | None:
        """Return averaged ph-ph interaction, shape (num_gp, num_band0)."""
        return self._averaged_pp_interaction

    @averaged_pp_interaction.setter
    def averaged_pp_interaction(self, value: NDArray[np.double] | None) -> None:
        self._averaged_pp_interaction = value

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return kappa tensor, shape (num_sigma, num_temp, 6)."""
        return self._kappa

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._mode_kappa
