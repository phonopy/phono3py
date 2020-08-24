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

"""SSCHA calculation

Formulae implemented are based on these papers:

    Bianco et al. https://doi.org/10.1103/PhysRevB.96.014111
    Aseginolaza et al. https://doi.org/10.1103/PhysRevB.100.214307

"""

import numpy as np
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phono3py.phonon.func import mode_length, bose_einstein


class LambdaTensor(object):
    def __init__(self, frequencies, eigenvectors, masses):
        """

        Parameters
        ----------
        frequencies : ndarray
            Supercell phonon frequencies in THz (without 2pi).
            shape=(grid_point, band), dtype='double', order='C'
        eigenvectors : ndarray
            Supercell phonon eigenvectors.
            shape=(grid_point, band, band), dtype='double', order='C'
        massses : ndarray
            Atomic masses of supercell in AMU.

        """

        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._masses = masses
        self._occ = None

    def run_Lambda(self, f, T):
        r"""Tensor with four legs of atomic indices

        Sum over phonon modes mu and nu

        .. math::

           - \frac{\hbar^2}{8} \sum_{\mu\nu}
           \frac{F(z, \omega_\mu, \omega_\nu)}{\omega_\mu \omega_\nu}
           \frac{e^a_\mu}{\sqrt{M_a}} \frac{e^b_\nu}{\sqrt{M_b}}
           \frac{e^c_\mu}{\sqrt{M_c}} \frac{e^d_\nu}{\sqrt{M_d}}

        Parameters
        ----------
        f : float
            Phonon freuqency in THz (without 2pi)
        T : float
            Temperature in K

        """

        freqs = self._frequencies
        self._occ = bose_einstein(freqs, T)
        sqrt_masses = np.sqrt(self._masses)
        dtype = "c%d" % (np.dtype('double').itemsize * 2)
        N = len(self._masses)
        lambda_tensor = np.zeros((N, N, N, N), dtype=dtype, order='C')

        for a, m_a in enumerate(sqrt_masses):
            for b, m_b in enumerate(sqrt_masses):
                for c, m_c in enumerate(sqrt_masses):
                    for d, m_d in enumerate(sqrt_masses):
                        ph_sum = self._sum_over_phonons(f, freqs, a, b, c, d)
                        ph_sum /= m_a * m_b * m_c * m_d
                        lambda_tensor[a, b, c, d] = ph_sum

    def _sum_over_phonons(self, f, freqs, a, b, c, d):
        N = len(self._masses)
        lambda_abcd = 0
        for i, f_i in freqs:
            e_i = self._eigenvectors[i // N, :, i % N]
            for j, f_j in freqs:
                e_j = self._eigenvectors[j // N, :, j % N]
                fz = self._get_Fz(f, f_i, f_j, self._occ[i], self._occ[j])
                fz /= f_i * f_j
                fz *= e_i[a] * e_j[b] * e_i[c] * e_j[d]
                lambda_abcd += fz
        return lambda_abcd

    def _get_Fz(self, f, f_1, f_2, n_1, n_2):
        r"""Function in Lambda

        .. math::

           \frac{2}{\hbar} \left\{ \frac{(\omega_\mu + \omega_\nu)
           (1 + n_\mu + n_\nu)]}{(\omega_\mu + \omega_\nu)^2 - z^2}
           - \frac{(\omega_\mu - \omega_\nu)(n_\mu - n_\nu)]}
           {(\omega_\mu - \omega_\nu)^2 - z^2} \right \}

        Parameters
        ----------
        f : float
            Phonon freuqency in THz (without 2pi)
        T : float
            Temperature in K

        Returns
        -------
        Calculated value without hbar / 2.

        """

        f_sum = f_1 + f_2
        f_diff = f_1 - f_2
        fz = f_sum * (1 + n_1 + n_2) / (f_sum ** 2 - f ** 2)
        fz -= f_diff * (n_1 - n_2) / (f_diff ** 2 - f ** 2)
        return fz


class DispCorrMatrix(object):
    """Calculate gamma matrix

    This calculation is similar to the transformation from
    dynamical matrices to force constants. Instead of creating
    dynamcial matrices from eigenvalues and eigenvectors,
    1/a and eigenvectors are used, where a is mode length.

    """

    def __init__(self, primitive, supercell):
        self._d2f = DynmatToForceConstants(primitive,
                                           supercell,
                                           is_full_fc=True)

    @property
    def commensurate_points(self):
        return self._d2f.commensurate_points

    def create_gamma_matrix(self, frequencies, eigenvectors, T):
        a = mode_length(frequencies, T)
        self._d2f.create_dynamical_matrices(1.0 / a ** 2, eigenvectors)

    def run(self):
        self._d2f.run()

    @property
    def gamma_matrix(self):
        return self._d2f.force_constants
