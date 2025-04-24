"""Wigner thermal conductivity base class."""

# Copyright (C) 2022 Michele Simoncelli
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

import textwrap

import numpy as np
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.units import EV, Angstrom, Hbar, THz

from phono3py.conductivity.base import HeatCapacityMixIn
from phono3py.phonon.grid import get_grid_points_by_rotations
from phono3py.phonon.velocity_operator import VelocityOperator


class ConductivityWignerMixIn(HeatCapacityMixIn):
    """Thermal conductivity mix-in for velocity operator.

    This mix-in is included in `ConductivityWignerRTA` and `ConductivityWignerLBTE`.

    """

    @property
    def kappa_TOT_RTA(self):
        """Return kappa."""
        return self._kappa_TOT_RTA

    @property
    def kappa_P_RTA(self):
        """Return kappa."""
        return self._kappa_P_RTA

    @property
    def kappa_C(self):
        """Return kappa."""
        return self._kappa_C

    @property
    def mode_kappa_P_RTA(self):
        """Return mode_kappa."""
        return self._mode_kappa_P_RTA

    @property
    def mode_kappa_C(self):
        """Return mode_kappa."""
        return self._mode_kappa_C

    @property
    def velocity_operator(self):
        """Return velocity operator at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._gv_operator

    @property
    def gv_by_gv_operator(self):
        """Return gv_by_gv operator at grid points where mode kappa are calculated."""
        return self._gv_operator_sum2

    def _init_velocity(self, gv_delta_q):
        self._velocity_obj = VelocityOperator(
            self._pp.dynamical_matrix,
            q_length=gv_delta_q,
            symmetry=self._pp.primitive_symmetry,
            frequency_factor_to_THz=self._pp.frequency_factor_to_THz,
        )

    def _set_velocities(self, i_gp, i_data):
        self._set_gv_operator(i_gp, i_data)
        self._set_gv_by_gv_operator(i_gp, i_data)

    def _set_gv_operator(self, i_irgp, i_data):
        """Set velocity operator."""
        irgp = self._grid_points[i_irgp]
        self._velocity_obj.run([self._get_qpoint_from_gp_index(irgp)])
        gv_operator = self._velocity_obj.velocity_operators[0, :, :, :]
        self._gv_operator[i_data] = gv_operator[self._pp.band_indices, :, :]
        #
        gv = np.einsum("iij->ij", gv_operator).real
        deg_sets = degenerate_sets(self._frequencies[irgp])
        # group velocities in the degenerate subspace are obtained diagonalizing the
        # velocity operator in the subspace of degeneracy.
        for id_dir in range(3):
            pos = 0
            for deg in deg_sets:
                if len(deg) > 1:
                    matrix_deg = gv_operator[
                        pos : pos + len(deg), pos : pos + len(deg), id_dir
                    ]
                    eigvals_deg = np.linalg.eigvalsh(matrix_deg)
                    gv[pos : pos + len(deg), id_dir] = eigvals_deg
                pos += len(deg)
        #
        self._gv[i_data] = gv[self._pp.band_indices, :]

    def _set_gv_by_gv_operator(self, i_irgp, i_data):
        """Outer product of group velocities.

        (v x v) [num_k*, num_freqs, 3, 3]

        """
        gv_by_gv_operator_tensor, order_kstar = self._get_gv_by_gv_operator(
            i_irgp, i_data
        )
        # gv_by_gv_tensor, order_kstar = self._get_gv_by_gv(i_irgp, i_data)
        self._num_sampling_grid_points += order_kstar

        # Sum all vxv at k*
        for j, vxv in enumerate(([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
            # self._gv_sum2[i_data, :, j] = gv_by_gv_tensor[:, vxv[0], vxv[1]]
            # here it is storing the 6 independent components of the v^i x v^j tensor
            # i_data is the q-point index
            # j indexes the 6 independent component of the symmetric tensor  v^i x v^j
            self._gv_operator_sum2[i_data, :, :, j] = gv_by_gv_operator_tensor[
                :, :, vxv[0], vxv[1]
            ]
            # self._gv_sum2[i_data, :, j] = gv_by_gv_tensor[:, vxv[0], vxv[1]]

    def _get_gv_by_gv_operator(self, i_irgp, i_data):
        if self._is_kappa_star:
            rotation_map = get_grid_points_by_rotations(
                self._grid_points[i_irgp], self._pp.bz_grid
            )
        else:
            rotation_map = get_grid_points_by_rotations(
                self._grid_points[i_irgp],
                self._pp.bz_grid,
                reciprocal_rotations=self._point_operations,
            )

        gv_operator = self._gv_operator[i_data]
        nat3 = len(self._pp.primitive) * 3
        nbands = np.shape(gv_operator)[0]

        gv_by_gv_operator = np.zeros((nbands, nat3, 3, 3), dtype=self._complex_dtype)

        for r in self._rotations_cartesian:
            # can be optimized
            gvs_rot_operator = np.zeros((nbands, nat3, 3), dtype=self._complex_dtype)
            for s in range(0, nbands):
                for s_p in range(0, nat3):
                    for i in range(0, 3):
                        for j in range(0, 3):
                            gvs_rot_operator[s, s_p, i] += (
                                gv_operator[s, s_p, j] * r.T[j, i]
                            )
            #
            for s in range(0, nbands):
                for s_p in range(0, nat3):
                    for i in range(0, 3):
                        for j in range(0, 3):
                            gv_by_gv_operator[s, s_p, i, j] += gvs_rot_operator[
                                s, s_p, i
                            ] * np.conj(gvs_rot_operator[s, s_p, j])
                            # note np.conj(gvs_rot_operator[s,s_p,j]) =
                            # gvs_rot_operator[s_p,s,j] since Vel op. is hermitian

        order_kstar = len(np.unique(rotation_map))
        gv_by_gv_operator /= len(rotation_map) // len(np.unique(rotation_map))

        if self._grid_weights is not None:
            if order_kstar != self._grid_weights[i_irgp]:
                if self._log_level:
                    text = (
                        "Number of elements in k* is unequal "
                        "to number of equivalent grid-points. "
                        "This means that the mesh sampling grids break "
                        "symmetry. Please check carefully "
                        "the convergence over grid point densities."
                    )
                    msg = textwrap.fill(
                        text, initial_indent=" ", subsequent_indent=" ", width=70
                    )
                    print("*" * 30 + "Warning" + "*" * 30)
                    print(msg)
                    print("*" * 67)

        return gv_by_gv_operator, order_kstar


def get_conversion_factor_WTE(volume):
    """Return conversion factor of thermal conductivity."""
    return (
        (THz * Angstrom) ** 2  # ----> group velocity
        * EV  # ----> specific heat is in eV/
        * Hbar  # ----> transform lorentzian_div_hbar from eV^-1 to s
        / (volume * Angstrom**3)
    )  # ----> unit cell volume
