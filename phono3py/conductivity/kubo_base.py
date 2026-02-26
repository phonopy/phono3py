"""Kubo thermal conductivity components."""

# Copyright (C) 2022 Atsushi Togo
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

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.base import (
    ConductivityComponentsBase,
    get_kstar_order,
    get_multiplicity_at_q,
)
from phono3py.phonon.grid import get_qpoints_from_bz_grid_points
from phono3py.phonon.group_velocity_matrix import GroupVelocityMatrix
from phono3py.phonon.heat_capacity_matrix import mode_cv_matrix
from phono3py.phonon3.interaction import Interaction


class ConductivityKuboComponents(ConductivityComponentsBase):
    """Thermal conductivity components for Kubo RTA."""

    def __init__(
        self,
        pp: Interaction,
        grid_points: NDArray[np.int64],
        grid_weights: NDArray[np.int64],
        point_operations: NDArray[np.int64],
        rotations_cartesian: NDArray[np.int64],
        temperatures: NDArray[np.float64] | None = None,
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,
        log_level: int = 0,
    ):
        """Init method."""
        super().__init__(
            pp,
            grid_points,
            grid_weights,
            point_operations,
            rotations_cartesian,
            temperatures=temperatures,
            is_kappa_star=is_kappa_star,
            gv_delta_q=gv_delta_q,
            log_level=log_level,
        )

        self._complex_dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._cv_mat: NDArray[np.float64]
        self._gv_mat: NDArray[np.complex128]
        self._gv_mat_sum2: NDArray[np.complex128]

        self._velocity_obj = GroupVelocityMatrix(
            self._pp.dynamical_matrix,
            q_length=gv_delta_q,
            symmetry=self._pp.primitive_symmetry,
            frequency_factor_to_THz=self._pp.frequency_factor_to_THz,
        )

        if self._temperatures is not None:
            self._allocate_values()

    @property
    def heat_capacity_matrices(self) -> NDArray[np.float64]:
        """Return heat capacity matrices at sampling grid points."""
        return self._cv_mat

    @property
    def gv_matrix_sum2(self) -> NDArray[np.complex128]:
        """Return summed products of velocity matrices over k-star."""
        return self._gv_mat_sum2

    def set_heat_capacities(self, i_gp, i_data):
        """Set heat capacity matrix.

        x=freq/T has to be small enough to avoid overflow of exp(x).
        x < 100 is the hard-corded criterion.
        Otherwise just set 0.

        """
        super().set_heat_capacities(i_gp, i_data)

        if self._temperatures is None:
            raise RuntimeError(
                "Temperatures have not been set yet. "
                "Set temperatures before this method."
            )

        irgp = self._grid_points[i_gp]
        frequencies, _, _ = self._pp.get_phonons()
        freqs = frequencies[irgp] * get_physical_units().THzToEv
        cutoff = self._pp.cutoff_frequency * get_physical_units().THzToEv

        for i_temp, temp in enumerate(self._temperatures):
            if (freqs / (temp * get_physical_units().KB) > 100).any():
                continue
            cvm = mode_cv_matrix(temp, freqs, cutoff=cutoff)
            self._cv_mat[i_temp, i_data] = cvm[self._pp.band_indices, :]

    def set_velocities(self, i_gp, i_data):
        """Set and cache group-velocity-related quantities for a selected grid point.

        This method computes velocity-derived tensors/vectors at the grid point index
        ``i_gp`` and stores them at data-slot index ``i_data``. It also updates the
        running count of sampled grid points using the k-star multiplicity.

        Parameters
        ----------
        i_gp : int
            Grid-point index used to compute velocity quantities.
        i_data : int
            Destination index in internal storage arrays where computed values are
            written.

        Notes
        -----
        Updates the following internal attributes in-place:
        - ``self._num_sampling_grid_points`` (incremented by k-star order)
        - ``self._gv_mat[i_data]`` (group velocity matrix)
        - ``self._gv_mat_sum2[i_data]`` (summed second-order velocity matrix term)
        - ``self._gv[i_data]`` (group velocity vector)

        """
        gvm, gv = self._get_gv_matrix(i_gp)
        gvm_sum2, kstar_order = self._get_gvm_by_gvm(i_gp)
        self._num_sampling_grid_points += kstar_order
        self._gv_mat[i_data] = gvm
        self._gv_mat_sum2[i_data] = gvm_sum2
        self._gv[i_data] = gv

    def _get_gv_matrix(self, i_gp):
        """Get group velocity matrix.

        Returns
        -------
        gv_mat : ndarray
            Group velocity matrix at grid point of `i_gp`.
            shape=(num_band0, num_band, 3), dtype=complex
        gv : ndarray
            Group velocity at grid point of `i_gp`.
            shape=(num_band0, 3), dtype=double

        """
        irgp = self._grid_points[i_gp]
        self._velocity_obj.run(
            [get_qpoints_from_bz_grid_points(irgp, self._pp.bz_grid)]
        )
        gvm = np.zeros(self._gv_mat.shape[1:], dtype=self._complex_dtype, order="C")
        gv = np.zeros(self._gv.shape[1:], dtype="double", order="C")
        for i in range(3):
            gvm[:, :, i] = self._velocity_obj.group_velocity_matrices[
                0, i, self._pp.band_indices, :
            ]
            gv[:, i] = (
                self._velocity_obj.group_velocity_matrices[0, i]
                .diagonal()[self._pp.band_indices]
                .real
            )
        return gvm, gv

    def _get_gvm_by_gvm(self, i_gp):
        r"""Compute sum of gvm over k-star.

        q is an irreducible q-point. k_q is an arm of k-star of q.
        R are the elements of the reciprocal point group.
        m is the multiplicity of q, i.e., order of site-point symmetry.

        \sum_{k_q} v^\alpha_{kjj'} v^\beta_{kj'j}
        = 1/m \sum_{R} v^\alpha_{Rq jj'} v^\beta_{Rq j'j}

        Returns
        -------
        gvm_sum2 : ndarray
            sum of gvm x gvm over kstar arms.
            shape=(num_band0, num_band, 6).
            The last 6 gives 3x3 tensor element positions, xx, yy, zz, yz, xz, xy.
        order_kstar : int
            Number of kstar arms.

        """
        if self._is_kappa_star:
            multi = get_multiplicity_at_q(
                self._grid_points[i_gp],
                self._pp,
                self._point_operations,
            )
        else:
            multi = 1
        q = get_qpoints_from_bz_grid_points(self._grid_points[i_gp], self._pp.bz_grid)
        qpoints = [np.dot(r, q) for r in self._point_operations]
        self._velocity_obj.run(qpoints)

        gvm_sum2 = np.zeros(
            self._gv_mat.shape[1:-1] + (6,), dtype=self._complex_dtype, order="C"
        )
        for i_pair, (a, b) in enumerate(
            ([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])
        ):
            for gvm in self._velocity_obj.group_velocity_matrices:
                gvm_by_gvm = np.multiply(gvm[a], gvm[b].T)
                gvm_sum2[:, :, i_pair] += gvm_by_gvm[self._pp.band_indices, :]
        gvm_sum2 /= multi
        kstar_order = get_kstar_order(
            self._grid_weights[i_gp],
            multi,
            self._point_operations,
            verbose=self._log_level > 0,
        )
        return gvm_sum2, kstar_order

    def _allocate_values(self):
        super()._allocate_values()

        if self._temperatures is None:
            raise RuntimeError(
                "Temperatures have not been set yet. "
                "Set temperatures before this method."
            )

        num_band0 = len(self._pp.band_indices)
        num_band = len(self._pp.primitive) * 3
        num_temp = len(self._temperatures)
        num_grid_points = len(self._grid_points)

        self._cv_mat = np.zeros(
            (num_temp, num_grid_points, num_band0, num_band), dtype="double", order="C"
        )
        self._gv_mat = np.zeros(
            (num_grid_points, num_band0, num_band, 3),
            dtype=self._complex_dtype,
            order="C",
        )
        self._gv_mat_sum2 = np.zeros(
            (num_grid_points, num_band0, num_band, 6),
            dtype=self._complex_dtype,
            order="C",
        )
