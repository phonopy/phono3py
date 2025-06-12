"""Wigner thermal conductivity RTA class."""

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

from __future__ import annotations

import numpy as np
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.rta_base import ConductivityRTABase
from phono3py.conductivity.wigner_base import (
    ConductivityWignerComponents,
    get_conversion_factor_WTE,
)
from phono3py.phonon3.interaction import Interaction


class ConductivityWignerRTA(ConductivityRTABase):
    """Class of Wigner lattice thermal conductivity under RTA.

    Authors
    -------
    Michele Simoncelli

    """

    def __init__(
        self,
        interaction: Interaction,
        grid_points: np.ndarray | None = None,
        temperatures: list | np.ndarray | None = None,
        sigmas: list | np.ndarray | None = None,
        sigma_cutoff: float | None = None,
        is_isotope: bool = False,
        mass_variances: list | np.ndarray | None = None,
        boundary_mfp: float | None = None,  # in micrometer
        use_ave_pp: bool = False,
        is_kappa_star: bool = True,
        gv_delta_q: float | None = None,
        is_full_pp: bool = False,
        read_pp: bool = False,
        store_pp: bool = False,
        pp_filename: float | None = None,
        is_N_U: bool = False,
        is_gamma_detail: bool = False,
        is_frequency_shift_by_bubble: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._cv = None
        self._kappa_TOT_RTA = None
        self._kappa_P_RTA = None
        self._kappa_C = None
        self._mode_kappa_P_RTA = None
        self._mode_kappa_C = None

        self._gv_operator = None
        self._gv_operator_sum2 = None

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
        self._conversion_factor_WTE = get_conversion_factor_WTE(
            self._pp.primitive.volume
        )

        self._conductivity_components = ConductivityWignerComponents(
            self._pp,
            self._grid_points,
            self._grid_weights,
            self._point_operations,
            self._rotations_cartesian,
            temperatures=self._temperatures,
            is_kappa_star=self._is_kappa_star,
            gv_delta_q=gv_delta_q,
            log_level=self._log_level,
        )

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

    def set_kappa_at_sigmas(self):
        """Calculate the Wigner thermal conductivity.

        k_P + k_C using the scattering operator in the RTA approximation.

        """
        num_band = len(self._pp.primitive) * 3
        THzToEv = get_physical_units().THzToEv
        gv_by_gv = self._conductivity_components.gv_by_gv_operator
        cv = self._conductivity_components.mode_heat_capacities

        for i, _ in enumerate(self._grid_points):
            gp = self._grid_points[i]
            frequencies = self._frequencies[gp]
            # Kappa
            for j in range(len(self._sigmas)):
                for k in range(len(self._temperatures)):
                    g_sum = self._get_main_diagonal(
                        i, j, k
                    )  # phonon HWHM: q-point, sigma, temperature
                    for s1 in range(num_band):
                        for s2 in range(num_band):
                            hbar_omega_eV_s1 = (
                                frequencies[s1] * THzToEv
                            )  # hbar*omega=h*nu in eV
                            hbar_omega_eV_s2 = (
                                frequencies[s2] * THzToEv
                            )  # hbar*omega=h*nu in eV
                            if (frequencies[s1] > self._pp.cutoff_frequency) and (
                                frequencies[s2] > self._pp.cutoff_frequency
                            ):
                                hbar_gamma_eV_s1 = 2.0 * g_sum[s1] * THzToEv
                                hbar_gamma_eV_s2 = 2.0 * g_sum[s2] * THzToEv
                                #
                                lorentzian_divided_by_hbar = (
                                    0.5 * (hbar_gamma_eV_s1 + hbar_gamma_eV_s2)
                                ) / (
                                    (hbar_omega_eV_s1 - hbar_omega_eV_s2) ** 2
                                    + 0.25
                                    * ((hbar_gamma_eV_s1 + hbar_gamma_eV_s2) ** 2)
                                )
                                #
                                prefactor = (
                                    0.25
                                    * (hbar_omega_eV_s1 + hbar_omega_eV_s2)
                                    * (
                                        cv[k, i, s1] / hbar_omega_eV_s1
                                        + cv[k, i, s2] / hbar_omega_eV_s2
                                    )
                                )
                                if np.abs(frequencies[s1] - frequencies[s2]) < 1e-4:
                                    # degenerate or diagonal s1=s2 modes contribution
                                    # determine k_P
                                    contribution = (
                                        (gv_by_gv[i, s1, s2])
                                        * prefactor
                                        * lorentzian_divided_by_hbar
                                        * self._conversion_factor_WTE
                                    ).real
                                    #
                                    self._mode_kappa_P_RTA[j, k, i, s1] += (
                                        0.5 * contribution
                                    )
                                    self._mode_kappa_P_RTA[j, k, i, s2] += (
                                        0.5 * contribution
                                    )
                                    # prefactor 0.5 arises from the fact that degenerate
                                    # modes have the same specific heat, hence they give
                                    # the same contribution to the populations
                                    # conductivity
                                else:
                                    self._mode_kappa_C[j, k, i, s1, s2] += (
                                        (gv_by_gv[i, s1, s2])
                                        * prefactor
                                        * lorentzian_divided_by_hbar
                                        * self._conversion_factor_WTE
                                    ).real

                            elif s1 == s2:
                                self._num_ignored_phonon_modes[j, k] += 1

        N = self.number_of_sampling_grid_points
        self._kappa_P_RTA = self._mode_kappa_P_RTA.sum(axis=2).sum(axis=2) / N
        #
        self._kappa_C = self._mode_kappa_C.sum(axis=2).sum(axis=2).sum(axis=2) / N
        #
        self._kappa_TOT_RTA = self._kappa_P_RTA + self._kappa_C

    def _set_cv(self, i_gp, i_data):
        """Set cv for conductivity components."""
        self._conductivity_components.set_heat_capacities(i_gp, i_data)

    def _set_velocities(self, i_gp, i_data):
        """Set velocities for conductivity components."""
        self._conductivity_components.set_velocities(i_gp, i_data)

    def _allocate_values(self):
        super()._allocate_values()

        num_band0 = len(self._pp.band_indices)
        num_grid_points = len(self._grid_points)
        num_temp = len(self._temperatures)
        nat3 = len(self._pp.primitive) * 3

        # kappa* and mode_kappa* are accessed when all bands exist, i.e.,
        # num_band0==num_band.
        self._kappa_TOT_RTA = np.zeros(
            (len(self._sigmas), num_temp, 6), order="C", dtype="double"
        )
        self._kappa_P_RTA = np.zeros(
            (len(self._sigmas), num_temp, 6), order="C", dtype="double"
        )
        self._kappa_C = np.zeros(
            (len(self._sigmas), num_temp, 6), order="C", dtype="double"
        )

        self._mode_kappa_P_RTA = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, nat3, 6),
            order="C",
            dtype="double",
        )

        # one more index because we have off-diagonal terms (second index not
        # parallelized)
        self._mode_kappa_C = np.zeros(
            (
                len(self._sigmas),
                num_temp,
                num_grid_points,
                num_band0,
                nat3,
                6,
            ),
            order="C",
            dtype="double",
        )
