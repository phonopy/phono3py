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

import numpy as np
from phono3py.phonon3.imag_self_energy import get_imag_self_energy
from phono3py.phonon3.frequency_shift import imag_to_real


class SpectralFunction(object):
    """Calculate spectral function"""

    def __init__(self,
                 interaction,
                 grid_points,
                 frequency_points=None,
                 frequency_step=None,
                 num_frequency_points=None,
                 temperatures=None):
        self._interaction = interaction
        self._grid_points = grid_points
        self._frequency_points_in = frequency_points
        self._frequency_step = frequency_step
        self._num_frequency_points = num_frequency_points
        self._temperatures = temperatures
        self._spectral_functions = None
        self._gammas = None
        self._deltas = None
        self._frequency_points = None

    def run(self):
        self._run_gamma()
        self._run_delta()
        self._run_spectral_function()

    @property
    def spectral_functions(self):
        return self._spectral_functions

    @property
    def shifts(self):
        return self._deltas

    @property
    def half_linewidths(self):
        return self._gammas

    @property
    def frequency_points(self):
        return self._frequency_points

    def _run_gamma(self):
        # gammas[grid_points, sigmas, temps, freq_points, band_indices]
        gammas, self._frequency_points = get_imag_self_energy(
            self._interaction,
            self._grid_points,
            frequency_points=self._frequency_points_in,
            frequency_step=self._frequency_step,
            num_frequency_points=self._num_frequency_points,
            temperatures=self._temperatures)
        self._gammas = np.array(gammas[:, 0], dtype='double', order='C')

    def _run_delta(self):
        self._deltas = np.zeros_like(self._gammas)
        for i, gp in enumerate(self._grid_points):
            for j, temp in enumerate(self._temperatures):
                for k, bi in enumerate(self._interaction.band_indices):
                    re_part, fpoints = imag_to_real(
                        self._gammas[i, j, :, k], self._frequency_points)
                    self._deltas[i, j, :, k] = -re_part
        assert (np.abs(self._frequency_points - fpoints) < 1e-8).all()

    def _run_spectral_function(self):
        frequencies = self._interaction.get_phonons()[0]
        self._spectral_functions = np.zeros_like(self._gammas)
        for i, gp in enumerate(self._grid_points):
            for j, temp in enumerate(self._temperatures):
                for k, bi in enumerate(self._interaction.band_indices):
                    freq = frequencies[gp, bi]
                    gammas = self._gammas[i, j, :, k]
                    deltas = self._deltas[i, j, :, k]
                    vals = self._get_spectral_function(gammas, deltas, freq)
                    self._spectral_functions[i, j, :, k] = vals

    def _get_spectral_function(self, gammas, deltas, freq):
        fpoints = self._frequency_points
        vals = (4 * freq ** 2 * gammas) / (
            (fpoints - freq ** 2 - 2 * freq * deltas) ** 2
            + (2 * freq * gammas) ** 2)
        return vals
