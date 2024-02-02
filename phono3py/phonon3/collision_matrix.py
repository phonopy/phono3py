"""Calculate collision matrix of direct solution of LBTE."""

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

from typing import Optional

import numpy as np
from phonopy.units import Kb, THzToEv

from phono3py.phonon3.imag_self_energy import ImagSelfEnergy
from phono3py.phonon3.interaction import Interaction


class CollisionMatrix(ImagSelfEnergy):
    """Collision matrix of direct solution of LBTE for one grid point.

    Main diagonal part (imag-self-energy) and
    the other part are separately stored.

    """

    def __init__(
        self,
        interaction: Interaction,
        rotations_cartesian=None,
        num_ir_grid_points=Optional[int],
        rot_grid_points=None,
        is_reducible_collision_matrix=False,
        log_level=0,
        lang="C",
    ):
        """Init method."""
        self._pp: Interaction
        self._is_collision_matrix: bool
        self._sigma = None
        self._frequency_points = None
        self._temperature = None
        self._grid_point = None
        self._lang = None
        self._imag_self_energy = None
        self._collision_matrix = None
        self._pp_strength = None
        self._frequencies = None
        self._triplets_at_q = None
        self._triplets_map_at_q = None
        self._weights_at_q = None
        self._band_indices = None
        self._unit_conversion = None
        self._cutoff_frequency = None
        self._g = None
        self._unit_conversion = None
        self._log_level = log_level

        super().__init__(interaction, lang=lang)

        self._is_reducible_collision_matrix = is_reducible_collision_matrix
        self._is_collision_matrix = True

        if not self._is_reducible_collision_matrix:
            self._num_ir_grid_points = num_ir_grid_points
            self._rot_grid_points = np.array(
                self._pp.bz_grid.bzg2grg[rot_grid_points], dtype="int_", order="C"
            )
            self._rotations_cartesian = rotations_cartesian

    def run(self):
        """Calculate collision matrix at a grid point."""
        if self._pp_strength is None:
            self.run_interaction()

        num_band0 = self._pp_strength.shape[1]
        num_band = self._pp_strength.shape[2]
        self._imag_self_energy = np.zeros(num_band0, dtype="double")

        if self._is_reducible_collision_matrix:
            num_mesh_points = np.prod(self._pp.mesh_numbers)
            self._collision_matrix = np.zeros(
                (num_band0, num_mesh_points, num_band), dtype="double"
            )
        else:
            self._collision_matrix = np.zeros(
                (num_band0, 3, self._num_ir_grid_points, num_band, 3), dtype="double"
            )
        self._run_with_band_indices()
        self._run_collision_matrix()

    def get_collision_matrix(self):
        """Return collision matrix at a grid point."""
        return self._collision_matrix

    def set_grid_point(self, grid_point=None):
        """Set a grid point and prepare for collision matrix calculation."""
        if grid_point is None:
            self._grid_point = None
        else:
            self._pp.set_grid_point(grid_point, store_triplets_map=True)
            self._pp_strength = None
            (
                self._triplets_at_q,
                self._weights_at_q,
                self._triplets_map_at_q,
                self._ir_map_at_q,
            ) = self._pp.get_triplets_at_q()
            self._grid_point = grid_point
            self._frequencies, self._eigenvectors, _ = self._pp.get_phonons()

    def _run_collision_matrix(self):
        if self._temperature > 0:
            if self._lang == "C":
                if self._is_reducible_collision_matrix:
                    self._run_c_reducible_collision_matrix()
                else:
                    self._run_c_collision_matrix()
            else:
                if self._is_reducible_collision_matrix:
                    self._run_py_reducible_collision_matrix()
                else:
                    self._run_py_collision_matrix()

    def _run_c_collision_matrix(self):
        import phono3py._phono3py as phono3c

        phono3c.collision_matrix(
            self._collision_matrix,
            self._pp_strength,
            self._frequencies,
            self._g,
            self._triplets_at_q,
            self._triplets_map_at_q,
            self._ir_map_at_q,
            self._rot_grid_points,  # in GRGrid
            self._rotations_cartesian,
            self._temperature,
            self._unit_conversion,
            self._cutoff_frequency,
        )

    def _run_c_reducible_collision_matrix(self):
        import phono3py._phono3py as phono3c

        phono3c.reducible_collision_matrix(
            self._collision_matrix,
            self._pp_strength,
            self._frequencies,
            self._g,
            self._triplets_at_q,
            self._triplets_map_at_q,
            self._ir_map_at_q,
            self._temperature,
            self._unit_conversion,
            self._cutoff_frequency,
        )

    def _run_py_collision_matrix(self):
        r"""Sum over rotations, and q-points and bands for third phonons.

        \Omega' = \sum_R' R' \Omega_{kp,R'k'p'}

        pp_strength.shape = (num_triplets, num_band0, num_band, num_band)

        """
        num_band0 = self._pp_strength.shape[1]
        num_band = self._pp_strength.shape[2]
        gp2tp, tp2s, swapped = self._get_gp2tp_map()
        for i in range(self._num_ir_grid_points):
            r_gps = self._rot_grid_points[i]
            for r, r_gp in zip(self._rotations_cartesian, r_gps):
                inv_sinh = self._get_inv_sinh(tp2s[r_gp])
                ti = gp2tp[r_gp]
                for j, k in np.ndindex((num_band0, num_band)):
                    if swapped[r_gp]:
                        collision = (
                            self._pp_strength[ti, j, :, k]
                            * inv_sinh
                            * self._g[2, ti, j, :, k]
                        ).sum()
                    else:
                        collision = (
                            self._pp_strength[ti, j, k]
                            * inv_sinh
                            * self._g[2, ti, j, k]
                        ).sum()
                    collision *= self._unit_conversion
                    self._collision_matrix[j, :, i, k, :] += collision * r

    def _run_py_reducible_collision_matrix(self):
        r"""Sum over q-points and bands of third phonons.

        This corresponds to the second term of right hand side of
        \Omega_{q0p0, q1p1} in Chaput's paper.

        pp_strength.shape = (num_triplets, num_band0, num_band, num_band)

        """
        num_mesh_points = np.prod(self._pp.mesh_numbers)
        num_band0 = self._pp_strength.shape[1]
        num_band = self._pp_strength.shape[2]
        gp2tp, tp2s, swapped = self._get_gp2tp_map()
        for gp1 in range(num_mesh_points):
            inv_sinh = self._get_inv_sinh(tp2s[gp1])
            ti = gp2tp[gp1]
            for j, k in np.ndindex((num_band0, num_band)):
                if swapped[gp1]:
                    collision = (
                        self._pp_strength[ti, j, :, k]
                        * inv_sinh
                        * self._g[2, ti, j, :, k]
                    ).sum()
                else:
                    collision = (
                        self._pp_strength[ti, j, k] * inv_sinh * self._g[2, ti, j, k]
                    ).sum()
                collision *= self._unit_conversion
                self._collision_matrix[j, gp1, k] += collision

    def _get_gp2tp_map(self):
        """Return mapping table from grid point index to triplet index.

        triplets_map_at_q contains index mapping of q1 in (q0, q1, q2) to
        independet q1 under q0+q1+q2=G with a fixed q0.

        Note
        ----
        map_q[gp1] <= gp1.:
            Symmetry relation of grid poi nts with a stabilizer q0.
        map_triplets[gp1] <= gp1 :
            map_q[gp1] == gp1 : map_q[gp2] if map_q[gp2] < gp1 otherwise gp1.
            map_q[gp1] != gp1 : map_triplets[map_q[gp1]]



        As a rule
        1. map_triplets[gp1] == gp1 : [gp0, gp1, gp2]
        2. map_triplets[gp1] != gp1 : [gp0, map_q[gp2], gp1'],
                                      map_triplets[gp1] == map_q[gp2]

        """
        map_triplets = self._triplets_map_at_q
        map_q = self._ir_map_at_q
        gp2tp = -np.ones(len(map_triplets), dtype="int_")
        tp2s = -np.ones(len(map_triplets), dtype="int_")
        swapped = np.zeros(len(map_triplets), dtype="bytes")
        num_tps = 0

        bzg2grg = self._pp.bz_grid.bzg2grg

        for gp1, tp_gp1 in enumerate(map_triplets):
            if map_q[gp1] == gp1:
                if gp1 == tp_gp1:
                    gp2tp[gp1] = num_tps
                    tp2s[gp1] = self._triplets_at_q[num_tps][2]
                    assert bzg2grg[self._triplets_at_q[num_tps][1]] == gp1
                    num_tps += 1
                else:  # q1 <--> q2 swap if swappable.
                    gp2tp[gp1] = gp2tp[tp_gp1]
                    tp2s[gp1] = self._triplets_at_q[gp2tp[gp1]][1]
                    swapped[gp1] = 1
                    assert map_q[bzg2grg[self._triplets_at_q[gp2tp[gp1]][2]]] == gp1
            else:  # q1 is not in ir-q1s.
                gp2tp[gp1] = gp2tp[map_q[gp1]]
                tp2s[gp1] = tp2s[map_q[gp1]]
                swapped[gp1] = swapped[map_q[gp1]]

            # Alternative implementation of tp2s
            # grg2bzg = self._pp.bz_grid.grg2bzg
            # addresses = self._pp.bz_grid.addresses
            # q0 = addresses[self._triplets_at_q[0][0]]
            # q1 = addresses[grg2bzg[gp1]]
            # q2 = -q0 - q1
            # gp2 = get_grid_point_from_address(q2, self._pp.bz_grid.D_diag)
            # tp2s[gp1] = self._pp.bz_grid.grg2bzg[gp2]

        return gp2tp, tp2s, swapped

    def _get_inv_sinh(self, gp):
        """Return sinh term for bands at a q-point."""
        freqs = self._frequencies[gp]
        sinh = np.where(
            freqs > self._cutoff_frequency,
            np.sinh(freqs * THzToEv / (2 * Kb * self._temperature)),
            -1.0,
        )
        inv_sinh = np.where(sinh > 0, 1.0 / sinh, 0)

        return inv_sinh
