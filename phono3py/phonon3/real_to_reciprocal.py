"""Transform fc3 in real space to reciprocal space."""

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


class RealToReciprocal:
    """Transform fc3 in real space to reciprocal space."""

    def __init__(self, fc3, primitive, mesh, symprec=1e-5):
        """Init method."""
        self._fc3 = fc3
        self._primitive = primitive
        self._mesh = mesh
        self._symprec = symprec

        self._p2s_map = primitive.p2s_map
        self._s2p_map = primitive.s2p_map
        # Reduce supercell atom index to primitive index
        self._svecs, self._multi = self._primitive.get_smallest_vectors()
        self._fc3_reciprocal = None

    def run(self, triplet):
        """Run at each triplet of q-vectors."""
        self._triplet = triplet
        num_patom = len(self._primitive)
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._fc3_reciprocal = np.zeros(
            (num_patom, num_patom, num_patom, 3, 3, 3), dtype=dtype
        )
        self._real_to_reciprocal()

    def get_fc3_reciprocal(self):
        """Return fc3 in reciprocal space."""
        return self._fc3_reciprocal

    def _real_to_reciprocal(self):
        num_patom = len(self._primitive)
        sum_triplets = np.where(
            np.all(self._triplet != 0, axis=0), self._triplet.sum(axis=0), 0
        )
        sum_q = sum_triplets.astype("double") / self._mesh
        for i in range(num_patom):
            for j in range(num_patom):
                for k in range(num_patom):
                    self._fc3_reciprocal[i, j, k] = self._real_to_reciprocal_elements(
                        (i, j, k)
                    )

            prephase = self._get_prephase(sum_q, i)
            self._fc3_reciprocal[i] *= prephase

    def _real_to_reciprocal_elements(self, patom_indices):
        num_satom = len(self._s2p_map)
        pi = patom_indices
        i = self._p2s_map[pi[0]]
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        fc3_reciprocal = np.zeros((3, 3, 3), dtype=dtype)
        for j in range(num_satom):
            if self._s2p_map[j] != self._p2s_map[pi[1]]:
                continue
            for k in range(num_satom):
                if self._s2p_map[k] != self._p2s_map[pi[2]]:
                    continue
                phase = self._get_phase((j, k), pi[0])
                fc3_reciprocal += self._fc3[i, j, k] * phase
        return fc3_reciprocal

    def _get_prephase(self, sum_q, patom_index):
        r = self._primitive.scaled_positions[patom_index]
        return np.exp(2j * np.pi * np.dot(sum_q, r))

    def _get_phase(self, satom_indices, patom0_index):
        si = satom_indices
        p0 = patom0_index
        phase = 1 + 0j
        for i in (0, 1):
            svecs_adrs = self._multi[si[i], p0, 1]
            multi = self._multi[si[i], p0, 0]
            vs = self._svecs[svecs_adrs : (svecs_adrs + multi)]
            phase *= (
                np.exp(
                    2j
                    * np.pi
                    * np.dot(vs, self._triplet[i + 1].astype("double") / self._mesh)
                ).sum()
                / multi
            )
        return phase
