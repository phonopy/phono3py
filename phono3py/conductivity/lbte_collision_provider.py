"""LBTECollisionProvider: per-grid-point LBTE collision matrix row."""

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

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from phono3py.file_IO import read_pp_from_hdf5
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.interaction import Interaction


@dataclass
class LBTECollisionResult:
    """Per-grid-point LBTE collision data.

    gamma : NDArray[np.double]
        Imaginary self-energy (diagonal of collision matrix).
        Shape: (num_sigma, num_temp, num_band0).
    collision_row : NDArray[np.double]
        Off-diagonal part of the collision matrix row at this grid point.
        Shape (IR mode): (num_sigma, num_temp, num_band0, 3, num_ir_gp, num_band, 3).
        Shape (reducible mode): (num_sigma, num_temp, num_band0, num_mesh, num_band).
    averaged_pp : NDArray[np.double] or None
        Averaged ph-ph interaction, shape (num_band0,). Set when is_full_pp=True.
    """

    gamma: NDArray[np.double]
    collision_row: NDArray[np.double]
    averaged_pp: NDArray[np.double] | None = None


class LBTECollisionProvider:
    """Computes collision matrix row and gamma at one irreducible grid point.

    Wraps CollisionMatrix (which must be pre-initialized with the global grid
    structure: rotations_cartesian, num_ir_grid_points, rot_grid_points for
    IR mode, or is_reducible_collision_matrix=True for reducible mode).

    Parameters
    ----------
    pp : Interaction
        Ph-ph interaction object.
    collision : CollisionMatrix
        Pre-initialized CollisionMatrix instance.
    sigmas : list
        Smearing widths. Empty list selects the tetrahedron method.
    sigma_cutoff : float or None
        Smearing cutoff in units of sigma.
    temperatures : NDArray[np.double]
        Temperatures in Kelvin.
    is_full_pp : bool, optional
        Store averaged ph-ph interaction. Default False.
    read_pp : bool, optional
        Read ph-ph interaction from file. Default False.
    pp_filename : path or None, optional
        Filename for ph-ph interaction file. Default None.
    log_level : int, optional
        Verbosity. Default 0.
    """

    def __init__(
        self,
        pp: Interaction,
        collision: CollisionMatrix,
        sigmas: list[float | None],
        sigma_cutoff: float | None,
        temperatures: NDArray[np.double],
        is_full_pp: bool = False,
        read_pp: bool = False,
        pp_filename: str | os.PathLike | None = None,
        log_level: int = 0,
    ) -> None:
        """Init method."""
        self._pp = pp
        self._collision = collision
        self._sigmas = sigmas
        self._sigma_cutoff = sigma_cutoff
        self._temperatures = temperatures
        self._is_full_pp = is_full_pp
        self._read_pp = read_pp
        self._pp_filename = pp_filename
        self._log_level = log_level

    def compute(self, grid_point: int) -> LBTECollisionResult:
        """Compute gamma and collision matrix row at one grid point.

        Parameters
        ----------
        grid_point : int
            BZ grid point index.

        Returns
        -------
        LBTECollisionResult
            gamma shape: (num_sigma, num_temp, num_band0)
            collision_row shape: (num_sigma, num_temp, <row_shape>)
        """
        self._collision.set_grid_point(grid_point)

        if self._log_level:
            triplets = self._pp.get_triplets_at_q()[0]
            assert triplets is not None
            print("Number of triplets: %d" % len(triplets))

        num_sigma = len(self._sigmas)
        num_temp = len(self._temperatures)
        num_band0 = len(self._pp.band_indices)

        gamma_all = np.zeros((num_sigma, num_temp, num_band0), dtype="double")
        collision_rows: list[list[NDArray[np.double]]] = []
        averaged_pp: NDArray[np.double] | None = None

        for i_sigma, sigma in enumerate(self._sigmas):
            self._collision.set_sigma(sigma, sigma_cutoff=self._sigma_cutoff)
            self._collision.run_integration_weights()
            self._run_interaction(i_sigma, sigma, grid_point)

            if self._is_full_pp and i_sigma == 0:
                averaged_pp = np.array(self._pp.averaged_interaction, dtype="double")

            row_per_temp: list[NDArray[np.double]] = []
            for k, t in enumerate(self._temperatures):
                self._collision.temperature = t
                self._collision.run()
                ise = self._collision.imag_self_energy
                assert ise is not None
                gamma_all[i_sigma, k] = ise
                row = self._collision.get_collision_matrix()
                assert row is not None
                row_per_temp.append(row.copy())
            collision_rows.append(row_per_temp)

        collision_row = np.array(collision_rows, dtype="double")

        return LBTECollisionResult(
            gamma=gamma_all,
            collision_row=collision_row,
            averaged_pp=averaged_pp,
        )

    def _run_interaction(
        self, i_sigma: int, sigma: float | None, grid_point: int
    ) -> None:
        """Compute or reuse ph-ph interaction strength."""
        if self._read_pp:
            self._read_interaction_from_file(sigma, grid_point)
        elif i_sigma != 0 and (self._is_full_pp or self._sigma_cutoff is None):
            if self._log_level:
                print("Existing ph-ph interaction is used.")
            self._collision.run_interaction(is_full_pp=self._is_full_pp)

    def _read_interaction_from_file(self, sigma: float | None, grid_point: int) -> None:
        """Read ph-ph interaction from hdf5 file."""
        pp_strength, g_zero_from_file = read_pp_from_hdf5(
            self._pp.mesh_numbers,
            grid_point=grid_point,
            sigma=sigma,
            sigma_cutoff=self._sigma_cutoff,
            filename=self._pp_filename,  # type: ignore[arg-type]
            verbose=(self._log_level > 0),
        )
        _, g_zero_runtime = self._collision.get_integration_weights()
        if (
            g_zero_from_file is not None
            and g_zero_runtime is not None
            and (g_zero_from_file != g_zero_runtime).any()
        ):
            self._collision.set_interaction_strength(
                pp_strength, g_zero=g_zero_from_file
            )
        else:
            self._collision.set_interaction_strength(pp_strength)
