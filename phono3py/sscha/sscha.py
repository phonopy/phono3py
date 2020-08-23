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

from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phono3py.phonon.func import mode_length


class LambdaTensor(object):
    def __init__(self):
        pass

    def run_Fz(self):
        pass


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
        self._d2f.create_dynamical_matrices(1.0 / a, eigenvectors)

    def run(self):
        self._d2f.run()

    @property
    def gamma_matrix(self):
        return self._d2f.force_constants
