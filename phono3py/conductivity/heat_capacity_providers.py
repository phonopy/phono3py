"""Heat capacity provider building blocks for conductivity calculations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.grid_point_data import GridPointInput, GridPointResult
from phono3py.phonon3.interaction import Interaction


def get_temperature_condition(
    freqs: NDArray[np.double] | float, temperatures: NDArray[np.double] | float
) -> bool:
    """Return condition for computing mode heat capacity.

    One of arguments has to be float.

    freqs:
        Phonon frequencies in eV.
    temperatures:
        Temperatures in K.

    """
    return bool((freqs < 100 * temperatures * get_physical_units().KB).any())


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
            condition = get_temperature_condition(f, temperatures)
            # To avoid numpy warning
            _temperatures = np.where(condition, temperatures, 10000)
            cv[:, i] = np.where(condition, mode_cv(_temperatures, f), 0.0)  # type: ignore
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
