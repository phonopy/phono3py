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
from phonopy.physical_units import get_physical_units


def mode_cv_matrix(
    temps: NDArray[np.double],
    freqs: NDArray[np.double],
) -> NDArray[np.double]:
    r"""Calculate mode heat capacity matrix, Cqjj'.

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
    cutoff : float
        This is used to check the degeneracy in eV.

    Returns
    -------
    ndarray
        Heat capacity matrix in eV/K.
        shape=(num_temps, num_band, num_band), dtype='double', order='C'.

    """

    def bose_einstein(
        freqs: NDArray[np.double], temps: NDArray[np.double]
    ) -> NDArray[np.double]:
        """Re-implemented here because of different physical units."""
        x = np.divide.outer(freqs, get_physical_units().KB * temps).T
        return 1.0 / (np.exp(x) - 1)

    n = bose_einstein(freqs, temps)
    cvm = np.zeros((len(temps), len(freqs), len(freqs)), dtype="double", order="C")
    f_sub = np.subtract.outer(freqs, freqs)
    f_add = np.add.outer(freqs, freqs)
    n_sub = n[:, :, None] - n[:, None, :]
    cvm = -(f_add**2)[None, :, :] / 4 / temps[:, None, None] * n_sub / f_sub[None, :, :]
    return cvm
