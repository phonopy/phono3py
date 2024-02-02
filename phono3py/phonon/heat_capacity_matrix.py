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

import numpy as np
from phonopy.phonon.thermal_properties import mode_cv
from phonopy.units import Kb


def mode_cv_matrix(temp, freqs, cutoff=1e-4):
    r"""Calculate mode heat capacity matrix, Cqjj'.

    C_{\mathbf{q}jj'} = K_B \frac{e^{x_{\mathbf{q}j} - x_{\mathbf{q}j'}} - 1}
    {x_{\mathbf{q}j} - x_{\mathbf{q}j'}} \left( \frac{
    x_{\mathbf{q}j} + x_{\mathbf{q}j'}}{2}
    \right)^2 n_{\mathbf{q}j}(n_{\mathbf{q}j'} + 1)

    Note
    ----
    Diagonal (j=j') terms reduce to normal mode heat capacity.

    Parameters
    ----------
    temp : float
        Temperature in K.
    freqs : ndarray
        Phonon frequencies at a q-point in eV.
    cutoff : float
        This is used to check the degeneracy.

    Returns
    -------
    ndarray
        Heat capacity matrix in eV/K.
        shape=(num_band, num_band), dtype='double', order='C'.

    """
    x = freqs / Kb / temp
    shape = (len(freqs), len(freqs))
    cvm = np.zeros(shape, dtype="double", order="C")
    for i, j in np.ndindex(shape):
        if abs(freqs[i] - freqs[j]) < cutoff:
            cvm[i, j] = mode_cv(temp, freqs[i])
            continue
        sub = x[i] - x[j]
        add = x[i] + x[j]
        n_inv = np.exp([x[i], x[j], sub]) - 1
        cvm[i, j] = Kb * n_inv[2] / sub * (add / 2) ** 2 / n_inv[0] * (1 / n_inv[1] + 1)
    return cvm
