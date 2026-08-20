"""Calculate group velocity matrix."""

# Copyright (C) 2021 Atsushi Togo
# All rights reserved.
#
# This file is part of phonopy.
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

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.physical_units import get_physical_units

# Threshold on x = E / (kB T) below which two modes are treated as
# degenerate and (n_j - n_j') / (x_j - x_j') is replaced by its analytic
# limit instead of the quotient.
_DEGENERACY_TOLERANCE = 1e-6


def _bose_einstein_difference_ratio(
    x: NDArray[np.double], n: NDArray[np.double]
) -> NDArray[np.double]:
    r"""Return (n_j - n_j') / (x_j - x_j') with x = E / (k_B T).

    Evaluated as a quotient, this is 0/0 for degenerate modes and loses all
    significant digits when two frequencies differ only by round-off, which
    happens at every q-point where bands are degenerate. The exact identity

    n_{\mathbf{q}j} - n_{\mathbf{q}j'} =
    -n_{\mathbf{q}j} (n_{\mathbf{q}j'} + 1)
    \left( e^{x_{\mathbf{q}j} - x_{\mathbf{q}j'}} - 1 \right)

    removes the cancellation and gives the finite limit
    -n_{\mathbf{q}j} (n_{\mathbf{q}j} + 1) at x_{\mathbf{q}j} = x_{\mathbf{q}j'}.
    It is used for the nearly degenerate pairs, for which the exponential is
    safely expanded to first order.

    """
    d = x[:, :, None] - x[:, None, :]
    n_j = n[:, :, None]
    n_jp = n[:, None, :]
    is_deg = np.abs(d) < _DEGENERACY_TOLERANCE
    return np.where(
        is_deg,
        -n_j * (n_jp + 1) * (1 + d / 2),
        (n_j - n_jp) / np.where(is_deg, 1, d),
    )


def _mode_cv_matrix(
    temps: NDArray[np.double],
    freqs: NDArray[np.double],
    prefactor: NDArray[np.double],
) -> NDArray[np.double]:
    r"""Assemble mode heat capacity matrix from a frequency prefactor.

    Both NJC23 and IBDB19 heat capacity matrices share the common form

    C_{\mathbf{q}jj'} = -\frac{P_{\mathbf{q}jj'}}{T}
    \frac{n_{\mathbf{q}j} - n_{\mathbf{q}j'}}
    {\omega_{\mathbf{q}j} - \omega_{\mathbf{q}j'}}

    and differ only in the frequency prefactor P_{\mathbf{q}jj'}.

    Parameters
    ----------
    temps : NDArray[np.double]
        Temperatures in K.
    freqs : NDArray[np.double]
        Phonon frequencies at a q-point in eV.
    prefactor : NDArray[np.double]
        Frequency prefactor P_{\mathbf{q}jj'} in eV^2, shape (num_band, num_band).

    Returns
    -------
    ndarray
        Heat capacity matrix in eV/K.
        shape=(num_temps, num_band, num_band), dtype='double', order='C'.

    """
    kb_temps = get_physical_units().KB * temps
    x = np.divide.outer(freqs, kb_temps).T
    n = 1.0 / np.expm1(x)
    ratio = _bose_einstein_difference_ratio(x, n)
    cvm = -prefactor[None, :, :] * ratio / (temps * kb_temps)[:, None, None]
    return np.ascontiguousarray(cvm)


def mode_cv_matrix_njc23(
    temps: NDArray[np.double],
    freqs: NDArray[np.double],
) -> NDArray[np.double]:
    r"""Calculate mode heat capacity matrix, Cqjj', of NJC23.

    This implements the heat capacity matrix used by M. Ndour, P. Jund,
    and L. Chaput, J. Non-Cryst. Solids 621, 122618 (2023).

    C_{\mathbf{q}jj'} =
    k_\text{B} \frac{e^{x_{\mathbf{q}j} - x_{\mathbf{q}j'}} - 1}
    {x_{\mathbf{q}j} - x_{\mathbf{q}j'}} \left( \frac{
    x_{\mathbf{q}j} + x_{\mathbf{q}j'}}{2}
    \right)^2 n_{\mathbf{q}j}(n_{\mathbf{q}j'} + 1)

    This is reduced to

    C_{\mathbf{q}jj'} = k_\text{B}
    -\left( \frac{x_{\mathbf{q}j} + x_{\mathbf{q}j'}}{2} \right)^2
    \frac{n_{\mathbf{q}j} - n_{\mathbf{q}j'}}
    {x_{\mathbf{q}j} - x_{\mathbf{q}j'}}

    With x = \frac{\hbar \omega}{k_\text{B} T},

    C_{\mathbf{q}jj'} =
    -\frac{\hbar(\omega_{\mathbf{q}j} + \omega_{\mathbf{q}j'})^2}
    {4T(\omega_{\mathbf{q}j} - \omega_{\mathbf{q}j'})}
    (n_{\mathbf{q}j} - n_{\mathbf{q}j'})

    Note
    ----
    Diagonal (j=j') terms reduce to normal mode heat capacity.

    Parameters
    ----------
    temps: NDArray[np.double]
        Temperatures in K.
    freqs : NDArray[np.double]
        Phonon frequencies at a q-point in eV.

    Returns
    -------
    ndarray
        Heat capacity matrix in eV/K.
        shape=(num_temps, num_band, num_band), dtype='double', order='C'.

    """
    f_add = np.add.outer(freqs, freqs)
    return _mode_cv_matrix(temps, freqs, f_add**2 / 4)


def mode_cv_matrix_ibdb19(
    temps: NDArray[np.double],
    freqs: NDArray[np.double],
) -> NDArray[np.double]:
    r"""Calculate mode heat capacity matrix, Cqjj', of Eq. (9) of IBDB19.

    This implements the heat capacity matrix of the quasi-harmonic
    Green-Kubo formula, Eq. (9) of L. Isaeva, G. Barbalinardo, D. Donadio,
    and S. Baroni, Nat. Commun. 10, 3853 (2019),

    c_{\mathbf{q}jj'} =
    -\frac{\hbar \omega_{\mathbf{q}j} \omega_{\mathbf{q}j'}}{T}
    \frac{n_{\mathbf{q}j} - n_{\mathbf{q}j'}}
    {\omega_{\mathbf{q}j} - \omega_{\mathbf{q}j'}}

    With the energy E = \hbar \omega (in the same unit as ``freqs``), the
    factor \hbar cancels and this becomes

    c_{\mathbf{q}jj'} =
    -\frac{E_{\mathbf{q}j} E_{\mathbf{q}j'}}{T}
    \frac{n_{\mathbf{q}j} - n_{\mathbf{q}j'}}
    {E_{\mathbf{q}j} - E_{\mathbf{q}j'}}

    Note
    ----
    Diagonal (j=j') terms reduce to normal mode heat capacity.

    Parameters
    ----------
    temps: NDArray[np.double]
        Temperatures in K.
    freqs : NDArray[np.double]
        Phonon frequencies at a q-point in eV.

    Returns
    -------
    ndarray
        Heat capacity matrix in eV/K.
        shape=(num_temps, num_band, num_band), dtype='double', order='C'.

    """
    f_mul = np.multiply.outer(freqs, freqs)
    return _mode_cv_matrix(temps, freqs, f_mul)


def mode_cv_matrix_smm19(
    temps: NDArray[np.double],
    freqs: NDArray[np.double],
) -> NDArray[np.double]:
    r"""Calculate effective mode heat capacity matrix, Cqjj', of SMM19.

    This implements the effective heat capacity matrix of the Wigner /
    unified-theory coherence term of G. Simoncelli, N. Marzari, and
    F. Mauri, Nat. Phys. 15, 809 (2019), built from the scalar mode heat
    capacities C_{\mathbf{q}j},

    C^{\mathrm{SMM}}_{\mathbf{q}jj'} =
    \frac{1}{4}(\omega_{\mathbf{q}j} + \omega_{\mathbf{q}j'})
    \left( \frac{C_{\mathbf{q}j}}{\omega_{\mathbf{q}j}}
    + \frac{C_{\mathbf{q}j'}}{\omega_{\mathbf{q}j'}} \right)

    This expression is invariant to the frequency unit, since the unit
    factor cancels between the sum and the ratios.

    Note
    ----
    Diagonal (j=j') terms reduce to normal mode heat capacity.

    Parameters
    ----------
    temps: NDArray[np.double]
        Temperatures in K.
    freqs : NDArray[np.double]
        Phonon frequencies at a q-point in eV.

    Returns
    -------
    ndarray
        Heat capacity matrix in eV/K.
        shape=(num_temps, num_band, num_band), dtype='double', order='C'.

    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        cv = mode_cv(temps, freqs)  # (num_temp, num_band), eV/K
        cv_per_freq = cv / freqs[None, :]
        cv_per_freq = np.where(np.isfinite(cv_per_freq), cv_per_freq, 0.0)
    f_add = np.add.outer(freqs, freqs)
    cv_per_freq_sum = cv_per_freq[:, :, None] + cv_per_freq[:, None, :]
    cvm = 0.25 * f_add[None, :, :] * cv_per_freq_sum
    return np.ascontiguousarray(cvm)
