"""Wigner thermal conductivity direct solution class."""

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

from phono3py.conductivity.direct_solution import (
    ConductivityLBTEBase,
    diagonalize_collision_matrix,
)
from phono3py.conductivity.wigner_base import (
    ConductivityWignerComponents,
    get_conversion_factor_WTE,
)
from phono3py.phonon3.interaction import Interaction


class ConductivityWignerLBTE(ConductivityLBTEBase):
    """Class of Wigner lattice thermal conductivity under direct-solution.

    Authors
    -------
    Michele Simoncelli

    """

    def __init__(
        self,
        interaction: Interaction,
        grid_points=None,
        temperatures=None,
        sigmas=None,
        sigma_cutoff=None,
        is_isotope=False,
        mass_variances=None,
        boundary_mfp=None,
        solve_collective_phonon=False,
        is_reducible_collision_matrix=False,
        is_kappa_star=True,
        gv_delta_q=None,
        is_full_pp=False,
        read_pp=False,
        pp_filename=None,
        pinv_cutoff=1.0e-8,
        pinv_solver=0,
        pinv_method=0,
        log_level=0,
        lang="C",
    ):
        """Init method."""
        self._kappa_TOT_exact = None
        self._kappa_TOT_RTA = None
        self._kappa_P_exact = None
        self._mode_kappa_P_exact = None
        self._kappa_P_RTA = None
        self._mode_kappa_P_RTA = None
        self._kappa_C = None
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
            is_reducible_collision_matrix=self._is_reducible_collision_matrix,
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

    @property
    def kappa_TOT_exact(self):
        """Return kappa."""
        return self._kappa_TOT_exact

    @property
    def kappa_P_exact(self):
        """Return kappa."""
        return self._kappa_P_exact

    @property
    def mode_kappa_P_exact(self):
        """Return mode_kappa."""
        return self._mode_kappa_P_exact

    def _set_cv(self, i_gp, i_data):
        """Set cv for conductivity components."""
        self._conductivity_components.set_heat_capacities(i_gp, i_data)

    def _set_velocities(self, i_gp, i_data):
        """Set velocities for conductivity components."""
        self._conductivity_components.set_velocities(i_gp, i_data)

    def _allocate_local_values(self, num_grid_points):
        """Allocate grid point local arrays."""
        num_band0 = len(self._pp.band_indices)
        num_temp = len(self._temperatures)
        super()._allocate_local_values(num_grid_points)

        nat3 = len(self._pp.primitive) * 3
        self._kappa_TOT_RTA = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._kappa_TOT_exact = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._kappa_P_exact = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._kappa_P_RTA = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._kappa_C = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._mode_kappa_P_exact = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band0, 6), dtype="double"
        )
        self._mode_kappa_P_RTA = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band0, 6), dtype="double"
        )

        complex_dtype = "c%d" % (np.dtype("double").itemsize * 2)
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
            dtype=complex_dtype,
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
                    self._collision_eigenvalues[j, k] = w

                    self._set_kappa(j, k, weights)

                    if self._log_level:
                        print(
                            ("#%6s       " + " %-10s" * 6)
                            % (
                                "         \t\t  T(K)",
                                "xx",
                                "yy",
                                "zz",
                                "yz",
                                "xz",
                                "xy",
                            )
                        )
                        print(
                            "K_P_exact\t\t"
                            + ("%7.1f " + " %10.3f" * 6)
                            % ((t,) + tuple(self._kappa_P_exact[j, k]))
                        )
                        print(
                            "(K_P_RTA)\t\t"
                            + ("%7.1f " + " %10.3f" * 6)
                            % ((t,) + tuple(self._kappa_P_RTA[j, k]))
                        )
                        print(
                            "K_C      \t\t"
                            + ("%7.1f " + " %10.3f" * 6)
                            % ((t,) + tuple(self._kappa_C[j, k].real))
                        )
                        print(" ")
                        print(
                            "K_TOT=K_P_exact+K_C\t"
                            + ("%7.1f " + " %10.3f" * 6)
                            % (
                                (t,)
                                + tuple(self._kappa_C[j, k] + self._kappa_P_exact[j, k])
                            )
                        )
                        print("-" * 76, flush=True)

        if self._log_level:
            print("", flush=True)

    def _set_kappa(self, i_sigma, i_temp, weights):
        if self._is_reducible_collision_matrix:
            self._set_kappa_reducible_colmat(
                self._kappa_P_exact, self._mode_kappa_P_exact, i_sigma, i_temp, weights
            )
        else:
            self._set_kappa_ir_colmat(
                self._kappa_P_exact, self._mode_kappa_P_exact, i_sigma, i_temp, weights
            )

    def _set_kappa_RTA(self, i_sigma, i_temp, weights):
        if self._is_reducible_collision_matrix:
            self._set_kappa_RTA_reducible_colmat(
                self._kappa_P_RTA, self._mode_kappa_P_RTA, i_sigma, i_temp, weights
            )
            print(
                " WARNING: Coherences conductivity not implemented for "
                "is_reducible_collision_matrix=True "
            )
        else:
            self._set_kappa_RTA_ir_colmat(
                self._kappa_P_RTA, self._mode_kappa_P_RTA, i_sigma, i_temp, weights
            )
            self._set_kappa_C_ir_colmat(i_sigma, i_temp)

    def _set_kappa_C_ir_colmat(self, i_sigma, i_temp):
        """Calculate coherence term of the thermal conductivity."""
        N = self.number_of_sampling_grid_points
        num_band = len(self._pp.primitive) * 3
        # num_ir_grid_points = len(self._ir_grid_points)
        THzToEv = get_physical_units().THzToEv
        for i, gp in enumerate(self._ir_grid_points):
            # linewidths at qpoint i, sigma i_sigma, and temperature i_temp
            g = self._get_main_diagonal(i, i_sigma, i_temp) * 2.0  # linewidth (FWHM)
            frequencies = self._frequencies[gp]
            cv = self._conductivity_components.mode_heat_capacities[i_temp, i, :]
            for s1 in range(num_band):
                for s2 in range(num_band):
                    hbar_omega_eV_s1 = (
                        frequencies[s1] * THzToEv
                    )  # hbar*omega=h*nu in eV
                    hbar_omega_eV_s2 = (
                        frequencies[s2] * THzToEv
                    )  # hbar*omega=h*nu in eV
                    if (
                        (frequencies[s1] > self._pp.cutoff_frequency)
                        and (frequencies[s2] > self._pp.cutoff_frequency)
                    ) and np.abs(frequencies[s1] - frequencies[s2]) > 1e-4:
                        # frequencies must be non-degenerate to contribute to k_C,
                        # otherwise they contribute to k_P
                        hbar_gamma_eV_s1 = g[s1] * THzToEv
                        hbar_gamma_eV_s2 = g[s2] * THzToEv
                        lorentzian_divided_by_hbar = (
                            0.5 * (hbar_gamma_eV_s1 + hbar_gamma_eV_s2)
                        ) / (
                            (hbar_omega_eV_s1 - hbar_omega_eV_s2) ** 2
                            + 0.25 * ((hbar_gamma_eV_s1 + hbar_gamma_eV_s2) ** 2)
                        )
                        #
                        prefactor = (
                            0.25
                            * (hbar_omega_eV_s1 + hbar_omega_eV_s2)
                            * (cv[s1] / hbar_omega_eV_s1 + cv[s2] / hbar_omega_eV_s2)
                        )
                        self._mode_kappa_C[i_sigma, i_temp, i, s1, s2] = (
                            (self._conductivity_components.gv_by_gv_operator[i, s1, s2])
                            * prefactor
                            * lorentzian_divided_by_hbar
                            * self._conversion_factor_WTE
                        )

        self._kappa_C[i_sigma, i_temp] = (
            self._mode_kappa_C[i_sigma, i_temp].sum(axis=0).sum(axis=0).sum(axis=0) / N
        ).real
