# SPDX-License-Identifier: BSD-3-Clause
"""Mathematical functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units


def gaussian(x: NDArray[np.double], sigma: float) -> NDArray[np.double]:
    """Return normal distribution."""
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x**2) / 2 / sigma**2)


def bose_einstein(x: NDArray[np.double], T: float) -> NDArray[np.double]:
    """Return Bose-Einstein distribution.

    Note
    ----
    RuntimeWarning (divide by zero encountered in true_divide) will be emitted
    when t=0 for x as ndarray and x=0 for x as ndarray.

    This RuntimeWarning can be changed to error by np.seterr(all='raise') and
    Then FloatingPointError is emitted.

    Parameters
    ----------
    x : ndarray
        Phonon frequency in THz (without 2pi).
    T : float
        Temperature in K

    """
    return 1.0 / np.expm1(
        get_physical_units().THzToEv * x / (get_physical_units().KB * T)
    )


def sigma_squared(x: NDArray[np.double], T: float) -> NDArray[np.double]:
    """Return mode length.

    sigma^2 = (0.5 + n) hbar / omega

    Note
    ----
    RuntimeWarning (invalid value encountered in sqrt) will be emitted
    when x < 0 for x as ndarray.

    This RuntimeWarning can be changed to error by np.seterr(all='raise') and
    Then FloatingPointError is emitted.

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
    #####################################
    old_settings = np.seterr(all="raise")
    #####################################

    n = bose_einstein(x, T)
    # factor=1.0107576777968994
    factor = (
        get_physical_units().Hbar
        * get_physical_units().EV
        / (2 * np.pi * get_physical_units().THz)
        / get_physical_units().AMU
        / get_physical_units().Angstrom ** 2
    )

    #########################
    np.seterr(**old_settings)
    #########################

    return (0.5 + n) / x * factor
