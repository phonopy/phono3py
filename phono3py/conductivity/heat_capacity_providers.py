"""Heat capacity provider building blocks for conductivity calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.grid_point_data import GridPointInput, GridPointResult
from phono3py.phonon.heat_capacity_matrix import mode_cv_matrix
from phono3py.phonon3.interaction import Interaction


def get_heat_capacities(
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
    freqs = (
        frequencies[grid_point][pp.band_indices]  # type: ignore
        * get_physical_units().THzToEv
    )
    cutoff = pp.cutoff_frequency * get_physical_units().THzToEv
    cv = np.zeros((len(temperatures), len(freqs)), dtype="double")
    # x=freq/T has to be small enough to avoid overflow of exp(x).
    # x < 100 is the hard-corded criterion.
    # Otherwise just set 0.
    for i, f in enumerate(freqs):
        if f > cutoff:
            condition = f < 100 * temperatures * get_physical_units().KB
            cv[:, i] = np.where(
                condition,
                mode_cv(np.where(condition, temperatures, 10000), f),  # type: ignore
                0,
            )
    return cv


class ModeHeatCapacityProvider:
    """Compute scalar mode heat capacities at a grid point.

    This provider implements the ``HeatCapacityProvider`` protocol.  It
    computes the mode heat capacity Cv (per mode, per unit cell) at the
    requested temperatures using the standard Einstein/harmonic-oscillator
    formula, delegating to ``get_heat_capacities`` from ``conductivity.base``.

    The returned ``GridPointResult`` field ``heat_capacities`` has shape
    ``(num_temp, num_band0)``.

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
    ) -> GridPointResult:
        """Compute mode heat capacities at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.
        temperatures : ndarray of double, shape (num_temp,)
            Temperatures in Kelvin.

        Returns
        -------
        GridPointResult
            ``heat_capacities`` (num_temp, num_band0) is set.
        """
        cv = get_heat_capacities(gp.grid_point, self._pp, temperatures)
        result = GridPointResult(input=gp)
        result.heat_capacities = cv
        return result


class HeatCapacityMatrixProvider:
    """Compute heat capacity matrix at a grid point for the Kubo formula.

    This provider implements the ``HeatCapacityProvider`` protocol for the
    Green-Kubo formula.  It computes the off-diagonal heat capacity matrix
    ``C_{qjj'}`` using ``mode_cv_matrix`` from
    ``phono3py.phonon.heat_capacity_matrix``.

    The returned ``GridPointResult`` fields:
    - ``heat_capacities`` (num_temp, num_band0): diagonal (standard) mode heat
      capacities; set for compatibility with ``ConductivityCalculator``.
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
    ) -> GridPointResult:
        """Compute heat capacity matrix at a grid point.

        Parameters
        ----------
        gp : GridPointInput
            Per-grid-point phonon data.
        temperatures : ndarray of double, shape (num_temp,)
            Temperatures in Kelvin.

        Returns
        -------
        GridPointResult
            ``heat_capacities`` (num_temp, num_band0) and
            ``heat_capacity_matrix`` (num_temp, num_band0, num_band) are set.
        """
        result = GridPointResult(input=gp)

        # Scalar heat capacities (diagonal) for ConductivityCalculator storage.
        result.heat_capacities = get_heat_capacities(
            gp.grid_point, self._pp, temperatures
        )

        frequencies, _, _ = self._pp.get_phonons()
        freqs_ev = frequencies[gp.grid_point] * get_physical_units().THzToEv
        cutoff_ev = self._pp.cutoff_frequency * get_physical_units().THzToEv

        num_temp = len(temperatures)
        num_band0 = len(gp.band_indices)
        num_band = len(freqs_ev)
        cv_mat = np.zeros((num_temp, num_band0, num_band), dtype="double")

        for i_temp, temp in enumerate(temperatures):
            if temp == 0.0:
                continue
            # Skip temperature if any x = freq/(KB*T) > 100 (overflow guard).
            if (freqs_ev / (temp * get_physical_units().KB) > 100).any():
                continue
            cvm = mode_cv_matrix(temp, freqs_ev, cutoff=cutoff_ev)
            cv_mat[i_temp] = cvm[gp.band_indices, :]

        result.heat_capacity_matrix = cv_mat
        return result
