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
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.base import ConductivityComponentsBase, get_heat_capacities
from phono3py.phonon.grid import (
    get_grid_points_by_rotations,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon.velocity_operator import VelocityOperator
from phono3py.phonon3.interaction import Interaction


def get_conversion_factor_WTE(volume):
    """Return conversion factor of thermal conductivity."""
    return (
        (get_physical_units().THz * get_physical_units().Angstrom)
        ** 2  # --> group velocity
        * get_physical_units().EV  # --> specific heat is in eV/
        * get_physical_units().Hbar  # --> transform lorentzian_div_hbar from eV^-1 to s
        / (volume * get_physical_units().Angstrom ** 3)
    )  # --> unit cell volume


class ConductivityWignerComponents(ConductivityComponentsBase):
    """Thermal conductivity components for velocity operator.

    Used by `ConductivityWignerRTA` and `ConductivityWignerLBTE`.

    """

    def __init__(
        self,
        pp: Interaction,
        grid_points: NDArray[np.int64],
        grid_weights: NDArray[np.int64],
        point_operations: NDArray[np.int64],
        rotations_cartesian: NDArray[np.int64],
        temperatures: Optional[NDArray[np.float64]] = None,
        is_kappa_star: bool = True,
        gv_delta_q: Optional[float] = None,
        is_reducible_collision_matrix: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        super().__init__(
            pp,
            grid_points=grid_points,
            grid_weights=grid_weights,
            point_operations=point_operations,
            rotations_cartesian=rotations_cartesian,
            temperatures=temperatures,
            is_kappa_star=is_kappa_star,
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            log_level=log_level,
        )

        self._gv_operator: np.ndarray
        self._gv_operator_sum2: np.ndarray

        if self._pp.dynamical_matrix is None:
            raise RuntimeError("Interaction.init_dynamical_matrix() has to be called.")
        self._velocity_obj = VelocityOperator(
            self._pp.dynamical_matrix,
            q_length=gv_delta_q,
            symmetry=self._pp.primitive_symmetry,
            frequency_factor_to_THz=self._pp.frequency_factor_to_THz,
        )

        self._num_sampling_grid_points = 0
        self._complex_dtype = "c%d" % (np.dtype("double").itemsize * 2)

        if self._temperatures is not None:
            self._allocate_values()

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

    def set_velocities(self, i_gp, i_data):
        """Set velocities at a grid point."""
        self._set_gv_operator(i_gp, i_data)
        self._set_gv_by_gv_operator(i_gp, i_data)

    def set_heat_capacities(self, i_gp: int, i_data: int):
        """Set heat capacity at grid point and at data location."""
        if self._temperatures is None:
            raise RuntimeError(
                "Temperatures have not been set yet. "
                "Set temperatures before this method."
            )

        cv = get_heat_capacities(self._grid_points[i_gp], self._pp, self._temperatures)
        self._cv[:, i_data, :] = cv

    def _set_gv_operator(self, i_irgp, i_data):
        """Set velocity operator."""
        irgp = self._grid_points[i_irgp]
        frequencies, _, _ = self._pp.get_phonons()
        self._velocity_obj.run(
            [get_qpoints_from_bz_grid_points(irgp, self._pp.bz_grid)]
        )
        gv_operator = self._velocity_obj.velocity_operators[0, :, :, :]
        self._gv_operator[i_data] = gv_operator[self._pp.band_indices, :, :]
        #
        gv = np.einsum("iij->ij", gv_operator).real
        deg_sets = degenerate_sets(frequencies[irgp])
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
            # self._gv_by_gv[i_data, :, j] = gv_by_gv_tensor[:, vxv[0], vxv[1]]
            # here it is storing the 6 independent components of the v^i x v^j tensor
            # i_data is the q-point index
            # j indexes the 6 independent component of the symmetric tensor  v^i x v^j
            self._gv_operator_sum2[i_data, :, :, j] = gv_by_gv_operator_tensor[
                :, :, vxv[0], vxv[1]
            ]
            # self._gv_by_gv[i_data, :, j] = gv_by_gv_tensor[:, vxv[0], vxv[1]]

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

    def _allocate_values(self):
        super()._allocate_values()

        num_band0 = len(self._pp.band_indices)
        if self._is_reducible_collision_matrix:
            num_grid_points = np.prod(self._pp.mesh_numbers)
        else:
            num_grid_points = len(self._grid_points)
        num_band = len(self._pp.primitive) * 3

        self._gv_operator = np.zeros(
            (num_grid_points, num_band0, num_band, 3),
            order="C",
            dtype=self._complex_dtype,
        )
        self._gv_operator_sum2 = np.zeros(
            (num_grid_points, num_band0, num_band, 6),
            order="C",
            dtype=self._complex_dtype,
        )
