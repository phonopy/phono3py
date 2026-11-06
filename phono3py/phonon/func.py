# SPDX-License-Identifier: BSD-3-Clause
"""Mathematical functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.thermal_properties import bose_einstein_dist
from phonopy.physical_units import get_physical_units


def gaussian(x: NDArray[np.double], sigma: float) -> NDArray[np.double]:
    """Return normal distribution."""
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x**2) / 2 / sigma**2)


def sigma_squared(x: NDArray[np.double], T: float) -> NDArray[np.double]:
    """Return mode length.

    sigma^2 = (0.5 + n) hbar / omega

    Note
    ----
    FloatingPointError is raised at zero frequency when T > 0.

    Parameters
    ----------
    x : ndarray
        Phonon frequency in THz (without 2pi).
    T : float
        Temperature in K

    Returns
    -------
    Values in [AMU * Angstrom^2]

    """
    # exp(-x) underflows to zero at low temperature, which correctly gives n = 0.
    with np.errstate(divide="raise", invalid="raise", under="ignore"):
        n = bose_einstein_dist(x, T)
    # factor=1.0107576777968994
    factor = (
        get_physical_units().Hbar
        * get_physical_units().EV
        / (2 * np.pi * get_physical_units().THz)
        / get_physical_units().AMU
        / get_physical_units().Angstrom ** 2
    )

    return (0.5 + n) / x * factor


def mode_cv(temps: NDArray[np.double], freqs: NDArray[np.double]) -> NDArray[np.double]:
    """Return mode heat capacity.

    With x = E / (k_B T), the expression k_B x^2 exp(-|x|) / expm1(-|x|)^2
    equals k_B x^2 exp(x) / (exp(x) - 1)^2, and exp(-|x|) does not overflow at
    low temperature.

    Note
    ----
    The value is nan at T=0 and at zero frequency. Callers have to handle these
    cases.

    Parameters
    ----------
    temps : ndarray
        Temperatures in K.
    freqs : ndarray
        Phonon frequencies in eV.

    Returns
    -------
    ndarray
        Mode heat capacities in eV/K.
        shape=(num_temps, num_freqs), dtype='double'.

    """
    kb = get_physical_units().KB
    x = np.abs(np.divide.outer(freqs, kb * temps).T)
    return kb * x**2 * np.exp(-x) / np.expm1(-x) ** 2
