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
from phonopy.units import VaspToTHz
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phono3py.phonon.func import mode_length


class SupercellPhonon(object):
    def __init__(self, supercell, force_constants, factor=VaspToTHz):
        self._supercell = supercell
        _fc2 = np.swapaxes(force_constants, 1, 2)
        _fc2 = _fc2.reshape(-1, np.prod(_fc2.shape[-2:]))
        masses = np.repeat(supercell.masses, 3)
        dynmat = np.array(_fc2 / np.sqrt(np.outer(masses, masses)),
                          dtype='double', order='C')
        eigvals, eigvecs = np.linalg.eigh(dynmat)
        freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals) * factor
        self._eigenvalues = np.array(eigvals, dtype='double', order='C')
        self._eigenvectors = np.array(eigvecs, dtype='double', order='C')
        self._frequencies = np.array(freqs, dtype='double', order='C')
        self._dynamical_matrix = dynmat

    @property
    def eigenvalues(self):
        return self._eigenvalues

    @property
    def eigenvectors(self):
        return self._eigenvectors

    @property
    def frequencies(self):
        return self._frequencies

    @property
    def dynamical_matrix(self):
        return self._dynamical_matrix

    @property
    def supercell(self):
        return self._supercell


class DispCorrMatrix(object):
    """Calculate gamma matrix"""

    def __init__(self, supercell_phonon):
        self._supercell_phonon = supercell_phonon
        self._gamma_matrix = None

    def run(self, T):
        freqs = self._supercell_phonon.frequencies
        eigvecs = self._supercell_phonon.eigenvectors
        a = mode_length(freqs, T)
        masses = np.repeat(self._supercell_phonon.supercell.masses, 3)
        gamma = np.dot(eigvecs, np.dot(np.diag(1.0 / a ** 2), eigvecs.T))
        self._gamma_matrix = gamma * np.sqrt(np.outer(masses, masses))

    @property
    def gamma_matrix(self):
        return self._gamma_matrix


class DispCorrMatrixMesh(object):
    """Calculate gamma matrix

    This calculation is similar to the transformation from
    dynamical matrices to force constants. Instead of creating
    dynamcial matrices from eigenvalues and eigenvectors,
    1/a and eigenvectors are used, where a is mode length.

    """

    def __init__(self, primitive, supercell):
        self._d2f = DynmatToForceConstants(
            primitive, supercell, is_full_fc=True)

    @property
    def commensurate_points(self):
        return self._d2f.commensurate_points

    def create_gamma_matrix(self, frequencies, eigenvectors, T):
        """

        Parameters
        ----------
        frequencies : ndarray
            Supercell phonon frequencies in THz (without 2pi).
            shape=(grid_point, band), dtype='double', order='C'
        eigenvectors : ndarray
            Supercell phonon eigenvectors.
            shape=(grid_point, band, band), dtype='double', order='C'

        """

        a = mode_length(frequencies, T)
        self._d2f.create_dynamical_matrices(1.0 / a ** 2, eigenvectors)

    def run(self):
        self._d2f.run()

    @property
    def gamma_matrix(self):
        return self._d2f.force_constants
