"""Lattice thermal conductivity calculation with RTA."""

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

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from phono3py.conductivity.base import ConductivityComponents
from phono3py.conductivity.rta_base import ConductivityRTABase
from phono3py.phonon3.interaction import Interaction


class ConductivityRTA(ConductivityRTABase):
    """Lattice thermal conductivity calculation with RTA."""

    def __init__(
        self,
        interaction: Interaction,
        grid_points: ArrayLike | None = None,
        temperatures: ArrayLike | None = None,
        sigmas: Sequence[float | None] | None = None,
        sigma_cutoff: float | None = None,
        is_isotope: bool = False,
        mass_variances: ArrayLike | None = None,
        boundary_mfp: float | None = None,  # in micrometer
        use_ave_pp: bool = False,
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,
        is_full_pp: bool = False,
        read_pp: bool = False,
        store_pp: bool = False,
        pp_filename: str | None = None,
        is_N_U: bool = False,
        is_gamma_detail: bool = False,
        is_frequency_shift_by_bubble: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._kappa = None
        self._mode_kappa = None

        super().__init__(
            interaction,
            grid_points=grid_points,
            temperatures=temperatures,
            sigmas=sigmas,
            sigma_cutoff=sigma_cutoff,
            is_isotope=is_isotope,
            mass_variances=mass_variances,
            boundary_mfp=boundary_mfp,
            use_ave_pp=use_ave_pp,
            is_kappa_star=is_kappa_star,
            is_full_pp=is_full_pp,
            read_pp=read_pp,
            store_pp=store_pp,
            pp_filename=pp_filename,
            is_N_U=is_N_U,
            is_gamma_detail=is_gamma_detail,
            is_frequency_shift_by_bubble=is_frequency_shift_by_bubble,
            log_level=log_level,
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
    def gv_by_gv(self) -> NDArray:
        """Return gv_by_gv at grid points where mode kappa are calculated."""
        return self._conductivity_components.gv_by_gv

    def _set_cv(self, i_gp, i_data):
        """Set cv for conductivity components."""
        self._conductivity_components.set_heat_capacities(i_gp, i_data)

    def _set_velocities(self, i_gp, i_data):
        """Set velocities for conductivity components."""
        self._conductivity_components.set_velocities(i_gp, i_data)

    def set_kappa_at_sigmas(self):
        """Calculate kappa from ph-ph interaction results."""
        if not self._pp.phonon_all_done:
            raise RuntimeError(
                "Phonon calculation has not been done yet. "
                "Run phono3py.run_phonon_solver() before this method."
            )
        if self._temperatures is None:
            raise RuntimeError(
                "Temperatures have not been set yet. "
                "Set temperatures before this method."
            )

        num_band = len(self._pp.primitive) * 3
        mode_heat_capacities = self._conductivity_components.mode_heat_capacities
        gv_by_gv = self._conductivity_components.gv_by_gv
        for i, _ in enumerate(self._grid_points):
            cv = mode_heat_capacities[:, i, :]
            gp = self._grid_points[i]
            frequencies = self._frequencies[gp]  # type: ignore

            # Kappa
            for j in range(len(self._sigmas)):
                for k in range(len(self._temperatures)):
                    g_sum = self._get_main_diagonal(i, j, k)
                    for ll in range(num_band):
                        if frequencies[ll] < self._pp.cutoff_frequency:
                            self._num_ignored_phonon_modes[j, k] += 1  # type: ignore
                            continue

                        old_settings = np.seterr(all="raise")
                        try:
                            self._mode_kappa[j, k, i, ll] = (  # type: ignore
                                gv_by_gv[i, ll]
                                * cv[k, ll]
                                / (g_sum[ll] * 2)
                                * self._conversion_factor
                            )
                        except FloatingPointError:
                            # supposed that g is almost 0 and |gv|=0
                            pass
                        except Exception:
                            print("=" * 26 + " Warning " + "=" * 26)
                            print(
                                " Unexpected physical condition of ph-ph "
                                "interaction calculation was found."
                            )
                            print(
                                " g=%f at gp=%d, band=%d, freq=%f"
                                % (g_sum[ll], gp, ll + 1, frequencies[ll])
                            )
                            print("=" * 61)
                        np.seterr(**old_settings)

        N = self.number_of_sampling_grid_points
        self._kappa = self._mode_kappa.sum(axis=2).sum(axis=2) / N

    def _allocate_values(self):
        if self._temperatures is None:
            raise RuntimeError(
                "Temperatures have not been set yet. "
                "Set temperatures before this method."
            )

        super()._allocate_values()

        num_band = len(self._pp.primitive) * 3
        num_grid_points = len(self._grid_points)
        num_temp = len(self._temperatures)

        # kappa* and mode_kappa* are accessed when all bands exist, i.e.,
        # num_band0==num_band.
        self._kappa = np.zeros(
            (len(self._sigmas), num_temp, 6), order="C", dtype="double"
        )
        self._mode_kappa = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band, 6),
            order="C",
            dtype="double",
        )
