"""Kubo thermal conductivity RTA class."""

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

import numpy as np

from phono3py.conductivity.kubo_base import ConductivityKuboMixIn
from phono3py.conductivity.rta_base import ConductivityRTABase
from phono3py.phonon3.interaction import Interaction


class ConductivityKuboRTA(ConductivityKuboMixIn, ConductivityRTABase):
    """Class of Kubo lattice thermal conductivity under RTA."""

    def __init__(
        self,
        interaction: Interaction,
        grid_points=None,
        temperatures=None,
        sigmas=None,
        sigma_cutoff=None,
        is_isotope=False,
        mass_variances=None,
        boundary_mfp=None,  # in micrometer
        use_ave_pp=False,
        is_kappa_star=True,
        gv_delta_q=None,
        is_full_pp=False,
        read_pp=False,
        store_pp=False,
        pp_filename=None,
        is_N_U=False,
        is_gamma_detail=False,
        is_frequency_shift_by_bubble=False,
        log_level=0,
    ):
        """Init method."""
        self._cv_mat = None
        self._gv_mat = None
        self._kappa = None

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
            gv_delta_q=gv_delta_q,
            is_full_pp=is_full_pp,
            read_pp=read_pp,
            store_pp=store_pp,
            pp_filename=pp_filename,
            is_N_U=is_N_U,
            is_gamma_detail=is_gamma_detail,
            is_frequency_shift_by_bubble=is_frequency_shift_by_bubble,
            log_level=log_level,
        )

    def set_kappa_at_sigmas(self):
        """Calculate kappa from ph-ph interaction results."""
        for i_gp, _ in enumerate(self._grid_points):
            frequencies = self._frequencies[self._grid_points[i_gp]]
            for j in range(len(self._sigmas)):
                for k in range(len(self._temperatures)):
                    g_sum = self._get_main_diagonal(i_gp, j, k)
                    for i_band, freq in enumerate(frequencies):
                        if freq < self._pp.cutoff_frequency:
                            self._num_ignored_phonon_modes[j, k] += 1
                            continue
                        self._set_kappa_at_sigmas(
                            j, k, i_gp, i_band, g_sum, frequencies
                        )
        N = self._num_sampling_grid_points
        self._kappa = self._mode_kappa_mat.sum(axis=2).sum(axis=2).sum(axis=2).real / N

    def _set_kappa_at_sigmas(self, j, k, i_gp, i_band, g_sum, frequencies):
        gvm_sum2 = self._gv_mat_sum2[i_gp]
        cvm = self._cv_mat[k, i_gp]
        for j_band, freq in enumerate(frequencies):
            if freq < self._pp.cutoff_frequency:
                return

            g = g_sum[i_band] + g_sum[j_band]
            for i_pair, _ in enumerate(
                ([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])
            ):
                old_settings = np.seterr(all="raise")
                try:
                    self._mode_kappa_mat[j, k, i_gp, i_band, j_band, i_pair] = (
                        cvm[i_band, j_band]
                        * gvm_sum2[i_band, j_band, i_pair]
                        * g
                        / ((frequencies[j_band] - frequencies[i_band]) ** 2 + g**2)
                        * self._conversion_factor
                    )
                except FloatingPointError:
                    # supposed that g is almost 0 and |gv|=0
                    pass
                except Exception:
                    gp = self._grid_points[i_gp]
                    print("=" * 26 + " Warning " + "=" * 26)
                    print(
                        " Unexpected physical condition of ph-ph "
                        "interaction calculation was found."
                    )
                    print(
                        " g=%f at gp=%d, band=%d, freq=%f, band=%d, freq=%f"
                        % (
                            g_sum[i_band],
                            gp,
                            i_band + 1,
                            frequencies[i_band],
                            j_band + 1,
                            frequencies[j_band],
                        )
                    )
                    print("=" * 61)
                np.seterr(**old_settings)

    def _allocate_values(self):
        super()._allocate_values()

        num_band0 = len(self._pp.band_indices)
        num_band = len(self._pp.primitive) * 3
        num_grid_points = len(self._grid_points)
        num_temp = len(self._temperatures)

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

        # kappa and mode_kappa_mat are accessed when all bands exist, i.e.,
        # num_band0==num_band.
        self._kappa = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._mode_kappa_mat = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band, num_band, 6),
            dtype=self._complex_dtype,
            order="C",
        )
