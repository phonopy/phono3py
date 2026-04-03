"""Heat capacity provider for the Green-Kubo formula."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.grid_point_data import GridPointInput, GridPointResult
from phono3py.conductivity.heat_capacity_providers import (
    get_heat_capacities,
    get_temperature_condition,
)
from phono3py.phonon.heat_capacity_matrix import mode_cv_matrix
from phono3py.phonon3.interaction import Interaction

DIV_BY_ZERO_THRESHOLD_EV = 1e-10


class HeatCapacityMatrixProvider:
    """Compute heat capacity matrix at a grid point for the Kubo formula.

    This provider implements the ``HeatCapacityProvider`` protocol for the
    Green-Kubo formula.  It computes the off-diagonal heat capacity matrix
    ``C_{qjj'}`` using ``mode_cv_matrix`` from
    ``phono3py.phonon.heat_capacity_matrix``.

    The returned ``GridPointResult`` fields:
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

        # Scalar heat capacities (diagonal) for RTACalculator storage.
        result.heat_capacities = get_heat_capacities(
            gp.grid_point, self._pp, temperatures
        )

        frequencies, _, _ = self._pp.get_phonons()
        freqs_ev = frequencies[gp.grid_point] * get_physical_units().THzToEv

        num_temp = len(temperatures)
        num_band0 = len(gp.band_indices)
        num_band = len(freqs_ev)
        cv_mat = np.zeros((num_temp, num_band0, num_band), dtype="double")

        for i_temp, temp in enumerate(temperatures):
            if temp == 0.0:
                continue
            if not get_temperature_condition(freqs_ev, temp):
                continue
            cvm = mode_cv_matrix(temp, freqs_ev, cutoff=DIV_BY_ZERO_THRESHOLD_EV)
            cv_mat[i_temp] = cvm[gp.band_indices, :]

        result.heat_capacity_matrix = cv_mat
        return result
