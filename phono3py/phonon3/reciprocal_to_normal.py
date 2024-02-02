"""Transform fc3 in reciprocal space to phonon space."""

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


class ReciprocalToNormal:
    """Class to transform fc3 in reciprocal space to phonon space.

    This is an implementation in python for prototyping and the test.
    Equivalent routine is implementated in C, and this is what usually
    we use.

    """

    def __init__(
        self, primitive, frequencies, eigenvectors, band_indices, cutoff_frequency=0
    ):
        """Init method."""
        self._primitive = primitive
        self._frequencies = frequencies
        self._eigenvectors = eigenvectors
        self._band_indices = band_indices
        self._cutoff_frequency = cutoff_frequency

        self._masses = self._primitive.masses

        self._fc3_normal = None
        self._fc3_reciprocal = None

    def run(self, fc3_reciprocal, grid_triplet):
        """Calculate fc3 in phonon coordinates."""
        num_band = len(self._primitive) * 3
        self._fc3_reciprocal = fc3_reciprocal
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._fc3_normal = np.zeros(
            (len(self._band_indices), num_band, num_band), dtype=dtype
        )
        self._reciprocal_to_normal(grid_triplet)

    def get_reciprocal_to_normal(self):
        """Return fc3 in phonon coordinates."""
        return self._fc3_normal

    def _reciprocal_to_normal(self, grid_triplet):
        e1, e2, e3 = self._eigenvectors[grid_triplet]
        f1, f2, f3 = self._frequencies[grid_triplet]
        num_band = len(f1)
        cutoff = self._cutoff_frequency
        for i, j, k in list(np.ndindex(len(self._band_indices), num_band, num_band)):
            bi = self._band_indices[i]
            if f1[bi] > cutoff and f2[j] > cutoff and f3[k] > cutoff:
                fc3_elem = self._sum_in_atoms((bi, j, k), (e1, e2, e3))
                fff = np.sqrt(f1[bi] * f2[j] * f3[k])
                self._fc3_normal[i, j, k] = fc3_elem / fff

    def _sum_in_atoms(self, band_indices, eigvecs):
        num_atom = self._primitive.get_number_of_atoms()
        (e1, e2, e3) = eigvecs
        (b1, b2, b3) = band_indices

        sum_fc3 = 0j
        for i, j, k in list(np.ndindex((num_atom,) * 3)):
            sum_fc3_cart = 0
            for ll, m, n in list(np.ndindex((3, 3, 3))):
                sum_fc3_cart += (
                    e1[i * 3 + ll, b1]
                    * e2[j * 3 + m, b2]
                    * e3[k * 3 + n, b3]
                    * self._fc3_reciprocal[i, j, k, ll, m, n]
                )
            mass_sqrt = np.sqrt(np.prod(self._masses[[i, j, k]]))
            sum_fc3 += sum_fc3_cart / mass_sqrt

        return sum_fc3
