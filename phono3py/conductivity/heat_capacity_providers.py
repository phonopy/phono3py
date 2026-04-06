"""Heat capacity provider building blocks for conductivity calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.grid_point_data import GridPointInput, HeatCapacityResult
from phono3py.phonon.heat_capacity_matrix import mode_cv_matrix
from phono3py.phonon3.interaction import Interaction


def _get_heat_capacities(
    grid_point: int,
    pp: Interaction,
    temperatures: NDArray[np.double],
) -> NDArray[np.double]:
    """Return mode heat capacity.

    cv returned should be given to self._cv by

        self._cv[:, i_data, :] = cv

    """
    if not pp.phonon_all_done:
        raise RuntimeError(
            "Phonon calculation has not been done yet. "
            "Run phono3py.run_phonon_solver() before this method."
        )

    frequencies, _, _ = pp.get_phonons()
    assert frequencies is not None
    freqs_ev = frequencies[grid_point] * get_physical_units().THzToEv
    cv = np.zeros((len(temperatures), len(freqs_ev)), dtype="double")
    cutoff = pp.cutoff_frequency * get_physical_units().THzToEv
    condition = freqs_ev > cutoff

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cv_vals = mode_cv(temperatures, freqs_ev)
        cv = np.where(np.isfinite(cv_vals) & condition[None, :], cv_vals, 0.0)
    return cv


class ModeHeatCapacityProvider:
    """Compute scalar mode heat capacities at a grid point.

    This provider implements the ``HeatCapacityProvider`` protocol.  It
    computes the mode heat capacity Cv (per mode, per unit cell) at the
    requested temperatures using the standard Einstein/harmonic-oscillator
    formula, delegating to ``get_heat_capacities`` from ``conductivity.base``.

    The returned ``HeatCapacityResult`` contains ``heat_capacities``
    with shape ``(num_temp, num_band0)``.

    Parameters
    ----------
    pp : Interaction
        Interaction instance.  Phonon solver must have been run
        (``phonon_all_done == True``) before calling ``compute``.
    """

    def __init__(self, pp: Interaction):
        """Init method."""
        self._pp = pp

    def compute(
        self,
        gp: GridPointInput,
        temperatures: NDArray[np.double],
    ) -> HeatCapacityResult:
        """Compute mode heat capacities at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.
        temperatures : ndarray of double, shape (num_temp,)
            Temperatures in Kelvin.

        Returns
        -------
        HeatCapacityResult
            ``heat_capacities`` (num_temp, num_band0) is set.
        """
        cv = _get_heat_capacities(gp.grid_point, self._pp, temperatures)
        return HeatCapacityResult(heat_capacities=cv[:, self._pp.band_indices])


class HeatCapacityMatrixProvider:
    """Compute heat capacity matrix at a grid point for the Kubo formula.

    This provider implements the ``HeatCapacityProvider`` protocol for the
    Green-Kubo formula.  It computes the off-diagonal heat capacity matrix
    ``C_{qjj'}`` using ``mode_cv_matrix`` from
    ``phono3py.phonon.heat_capacity_matrix``.

    The returned ``HeatCapacityResult`` contains:
    - ``heat_capacities`` (num_temp, num_band0): diagonal (standard) mode heat
      capacities; set for compatibility with ``RTACalculator``.
    - ``heat_capacity_matrix`` (num_temp, num_band0, num_band): full heat
      capacity matrix for selected bands (rows) vs all bands (columns).

    Parameters
    ----------
    pp : Interaction
        Interaction instance.  Phonon solver must have been run before calling
        ``compute``.

    """

    def __init__(self, pp: Interaction):
        """Init method."""
        self._pp = pp

    def compute(
        self,
        gp: GridPointInput,
        temperatures: NDArray[np.double],
    ) -> HeatCapacityResult:
        """Compute heat capacity matrix at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.
        temperatures : ndarray of double, shape (num_temp,)
            Temperatures in Kelvin.

        Returns
        -------
        HeatCapacityResult
            ``heat_capacities`` (num_temp, num_band0) and
            ``heat_capacity_matrix`` (num_temp, num_band0, num_band) are set.

        """
        frequencies, _, _ = self._pp.get_phonons()
        assert frequencies is not None
        freqs_ev = frequencies[gp.grid_point] * get_physical_units().THzToEv
        cv = _get_heat_capacities(gp.grid_point, self._pp, temperatures)

        cutoff = self._pp.cutoff_frequency * get_physical_units().THzToEv
        condition = freqs_ev > cutoff
        condition = np.logical_and.outer(condition, condition)

        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cvm = mode_cv_matrix(temperatures, freqs_ev)
            cv_mat = np.where(
                condition[None, :, :],
                np.where(np.isfinite(cvm), cvm, cv[:, None, :]),
                0.0,
            )

        return HeatCapacityResult(
            heat_capacities=cv[:, self._pp.band_indices],
            heat_capacity_matrix=cv_mat[:, self._pp.band_indices, :],
        )
