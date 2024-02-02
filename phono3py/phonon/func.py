"""Mathematical functions."""

# Copyright (C) 2020 Atsushi Togo
# All rights reserved.
#
# This file is part of phono3py.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the phonopy project nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from phonopy.units import AMU, EV, Angstrom, Hbar, Kb, THz, THzToEv


def gaussian(x, sigma):
    """Return normal distribution."""
    return 1.0 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x**2) / 2 / sigma**2)


def bose_einstein(x, T):
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
    return 1.0 / (np.exp(THzToEv * x / (Kb * T)) - 1)


def sigma_squared(x, T):
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
    factor = Hbar * EV / (2 * np.pi * THz) / AMU / Angstrom**2

    #########################
    np.seterr(**old_settings)
    #########################

    return (0.5 + n) / x * factor
