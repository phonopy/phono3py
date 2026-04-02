"""Kappa accumulator building blocks for conductivity calculations.

An accumulator owns the kappa formula and the BZ-summation arrays.
``ConductivityCalculator`` calls ``accumulate()`` once per grid point and
``finalize()`` at the end; it delegates output properties to the accumulator
via ``__getattr__``.

Adding a new transport variant requires only a new accumulator class registered
via ``register_calculator()`` — ``ConductivityCalculator`` and the core factory
do not need to be modified.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from phono3py.conductivity.grid_point_data import (
    GridPointResult,
    compute_effective_gamma,
)


@runtime_checkable
class KappaAccumulator(Protocol):
    """Protocol for BZ-summation of kappa contributions.

    Implementations
    ---------------
    RTAKappaAccumulator
        Standard BTE diagonal formula; exposes ``kappa``, ``mode_kappa``.

    """

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
    ) -> None:
        """Allocate output arrays before the grid-point loop starts.

        Called by ``ConductivityCalculator._allocate_values()``.

        """
        ...

    def accumulate(self, i_gp: int, result: GridPointResult) -> None:
        """Compute and store the kappa contribution for grid point ``i_gp``.

        The formula is called inside this method; ``result`` must already
        have ``velocity_product``, ``heat_capacities``, ``gamma`` (and
        optional ``gamma_isotope``, ``gamma_boundary``, ``gamma_elph``)
        populated.

        """
        ...

    def finalize(self, num_sampling_grid_points: int) -> None:
        """Normalise accumulated kappa by the total number of sampling points."""
        ...

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return total kappa tensor, shape (num_sigma, num_temp, 6)."""
        ...

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode-resolved kappa.

        Shape: (num_sigma, num_temp, num_gp, num_band0, 6).
        """
        ...


def _log_kappa_header(
    sigma: float | None,
    show_ipm: bool = False,
) -> None:
    """Print the kappa table header line for a given sigma."""
    text = "----------- Thermal conductivity (W/m-k) "
    if sigma:
        text += "for sigma=%s -----------" % sigma
    else:
        text += "with tetrahedron method -----------"
    print(text)
    if show_ipm:
        print(
            ("#%6s       " + " %-10s" * 6 + "#ipm")
            % ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
        )
    else:
        print(
            ("#%6s       " + " %-10s" * 6)
            % ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
        )


def _log_kappa_row(
    label: str,
    temperature: float,
    kappa_row: NDArray[np.double],
    num_ignored: int | None = None,
    num_phonon_modes: int | None = None,
) -> None:
    """Print one row of the kappa table."""
    if num_ignored is not None and num_phonon_modes is not None:
        print(
            label
            + ("%7.1f" + " %10.3f" * 6 + " %d/%d")
            % ((temperature,) + tuple(kappa_row) + (num_ignored, num_phonon_modes))
        )
    else:
        print(label + ("%7.1f " + " %10.3f" * 6) % ((temperature,) + tuple(kappa_row)))


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
        self._kappa: NDArray[np.double]
        self._mode_kappa: NDArray[np.double]

    def prepare(
        self,
        num_sigma: int,
        num_temp: int,
        num_gp: int,
        num_band0: int,
    ) -> None:
        """Allocate kappa and mode_kappa arrays."""
        self._kappa = np.zeros((num_sigma, num_temp, 6), dtype="double", order="C")
        self._mode_kappa = np.zeros(
            (num_sigma, num_temp, num_gp, num_band0, 6), dtype="double", order="C"
        )

    def accumulate(self, i_gp: int, result: GridPointResult) -> None:
        """Compute and accumulate mode kappa at grid point ``i_gp``."""
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
        # gv_by_gv already encodes the k-star order (sum over rotations
        # divided by site multiplicity). Do NOT multiply by grid_weight;
        # the normalization is done once via num_sampling_grid_points.
        self._kappa += np.sum(mode_kappa, axis=2)

    def finalize(self, num_sampling_grid_points: int) -> None:
        """Normalise by the total number of sampling grid points."""
        if num_sampling_grid_points > 0:
            self._kappa /= num_sampling_grid_points

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
            _log_kappa_header(sigma, show_ipm=show_ipm)
            for j, t in enumerate(self._temperatures):
                ipm = (
                    int(num_ignored_phonon_modes[i, j])
                    if show_ipm and num_ignored_phonon_modes is not None
                    else None
                )
                _log_kappa_row("", t, self._kappa[i, j], ipm, num_phonon_modes)
            print("", flush=True)

    @property
    def kappa(self) -> NDArray[np.double]:
        """Return kappa tensor, shape (num_sigma, num_temp, 6)."""
        return self._kappa

    @property
    def mode_kappa(self) -> NDArray[np.double]:
        """Return mode kappa, shape (num_sigma, num_temp, num_gp, num_band0, 6)."""
        return self._mode_kappa
