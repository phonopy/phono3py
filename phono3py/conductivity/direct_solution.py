"""Calculate lattice thermal conductivity with direct solution."""

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

import numpy as np
from numpy.typing import ArrayLike, NDArray

from phono3py.conductivity.base import ConductivityComponents
from phono3py.conductivity.direct_solution_base import (
    ConductivityLBTEBase,
    diagonalize_collision_matrix,
)
from phono3py.phonon3.interaction import Interaction


class ConductivityLBTE(ConductivityLBTEBase):
    """Lattice thermal conductivity calculation by direct solution."""

    def __init__(
        self,
        interaction: Interaction,
        grid_points: ArrayLike | None = None,
        temperatures: ArrayLike | None = None,
        sigmas: ArrayLike | None = None,
        sigma_cutoff: float | None = None,
        is_isotope: bool = False,
        mass_variances: ArrayLike | None = None,
        boundary_mfp: float | None = None,  # in micrometer
        solve_collective_phonon: bool = False,
        is_reducible_collision_matrix: bool = False,
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,
        is_full_pp: bool = False,
        read_pp: bool = False,
        pp_filename: str | os.PathLike | None = None,
        pinv_cutoff: float = 1.0e-8,
        pinv_solver: int = 0,
        pinv_method: int = 0,
        log_level: int = 0,
        lang: str = "C",
    ):
        """Init method."""
        self._kappa = None
        self._kappa_RTA = None
        self._mode_kappa = None
        self._mode_kappa_RTA = None

        super().__init__(
            interaction,
            grid_points=grid_points,
            temperatures=temperatures,
            sigmas=sigmas,
            sigma_cutoff=sigma_cutoff,
            is_isotope=is_isotope,
            mass_variances=mass_variances,
            boundary_mfp=boundary_mfp,
            solve_collective_phonon=solve_collective_phonon,
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            is_kappa_star=is_kappa_star,
            is_full_pp=is_full_pp,
            read_pp=read_pp,
            pp_filename=pp_filename,
            pinv_cutoff=pinv_cutoff,
            pinv_solver=pinv_solver,
            pinv_method=pinv_method,
            log_level=log_level,
            lang=lang,
        )

        self._conductivity_components: ConductivityComponents = ConductivityComponents(
            self._pp,
            self._grid_points,
            self._grid_weights,
            self._point_operations,
            self._rotations_cartesian,
            temperatures=self._temperatures,
            average_gv_over_kstar=self._average_gv_over_kstar,
            is_kappa_star=self._is_kappa_star,
            gv_delta_q=gv_delta_q,
            is_reducible_collision_matrix=self._is_reducible_collision_matrix,
            log_level=self._log_level,
        )

    @property
    def kappa(self) -> NDArray | None:
        """Return kappa."""
        return self._kappa

    @property
    def mode_kappa(self) -> NDArray | None:
        """Return mode_kappa."""
        return self._mode_kappa

    @property
    def kappa_RTA(self) -> NDArray | None:
        """Return RTA lattice thermal conductivity."""
        return self._kappa_RTA

    @property
    def mode_kappa_RTA(self) -> NDArray | None:
        """Return RTA mode lattice thermal conductivities."""
        return self._mode_kappa_RTA

    @property
    def gv_by_gv(self) -> NDArray:
        """Return gv_by_gv at grid points where mode kappa are calculated."""
        return self._conductivity_components.gv_by_gv

    def _set_cv(self, i_gp, i_data):
        """Set cv for conductivity components."""
        self._conductivity_components.set_heat_capacities(i_gp, i_data)

    def _set_velocities(self, i_gp, i_data):
        """Set velocities for conductivity components."""
        self._conductivity_components.set_velocities(i_gp, i_data)

    def _allocate_local_values(self, num_grid_points):
        """Allocate grid point local arrays.

        For full collision matrix, `num_grid_points` equals to the number of
        grid points in GRGrid, i.e., `prod(D_diag)`. Otherwise, number of
        grid points to be iterated over.

        """
        if self._temperatures is None:
            raise RuntimeError(
                "Temperatures have not been set yet. "
                "Set temperatures before this method."
            )

        num_band0 = len(self._pp.band_indices)
        num_temp = len(self._temperatures)
        super()._allocate_local_values(num_grid_points)

        self._kappa = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._kappa_RTA = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._mode_kappa = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band0, 6), dtype="double"
        )
        self._mode_kappa_RTA = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band0, 6), dtype="double"
        )

    def _set_kappa_at_sigmas(self, weights):
        """Calculate thermal conductivity from collision matrix."""
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "----------- Thermal conductivity (W/m-k) "
                if sigma:
                    text += "for sigma=%s -----------" % sigma
                else:
                    text += "with tetrahedron method -----------"
                print(text, flush=True)

            for k, t in enumerate(self._temperatures):
                if t > 0:
                    self._set_kappa_RTA(j, k, weights)

                    w = diagonalize_collision_matrix(
                        self._collision_matrix,
                        i_sigma=j,
                        i_temp=k,
                        pinv_solver=self._pinv_solver,
                        log_level=self._log_level,
                    )
                    if w is not None:
                        self._collision_eigenvalues[j, k] = w

                    self._set_kappa(j, k, weights)

                    if self._log_level:
                        print(
                            ("#%6s       " + " %-10s" * 6)
                            % ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                        )
                        print(
                            ("%7.1f " + " %10.3f" * 6)
                            % ((t,) + tuple(self._kappa[j, k]))
                        )
                        print(
                            (" %6s " + " %10.3f" * 6)
                            % (("(RTA)",) + tuple(self._kappa_RTA[j, k]))
                        )
                        print("-" * 76, flush=True)

        if self._log_level:
            print("", flush=True)

    def _set_kappa(self, i_sigma, i_temp, weights):
        if self._is_reducible_collision_matrix:
            self._set_kappa_reducible_colmat(
                self._kappa, self._mode_kappa, i_sigma, i_temp, weights
            )
        else:
            self._set_kappa_ir_colmat(
                self._kappa, self._mode_kappa, i_sigma, i_temp, weights
            )

    def _set_kappa_RTA(self, i_sigma, i_temp, weights):
        if self._is_reducible_collision_matrix:
            self._set_kappa_RTA_reducible_colmat(
                self._kappa_RTA, self._mode_kappa_RTA, i_sigma, i_temp, weights
            )
        else:
            self._set_kappa_RTA_ir_colmat(
                self._kappa_RTA, self._mode_kappa_RTA, i_sigma, i_temp, weights
            )
