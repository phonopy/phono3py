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
from phono3py.phonon3.imag_self_energy import (
    get_imag_self_energy, get_frequency_points)
from phono3py.phonon3.real_self_energy import imag_to_real
from phono3py.file_IO import write_spectral_function_at_grid_point


def write_spectral_function(spectral_functions,
                            mesh,
                            grid_points,
                            band_indices,
                            frequency_points,
                            temperatures,
                            sigmas=None,
                            output_filename=None,
                            is_mesh_symmetry=True,
                            log_level=0):
    if sigmas is None:
        _sigmas = [None, ]
    else:
        _sigmas = sigmas

    if spectral_functions.ndim == 4:
        _spf = spectral_functions[:, None, :, :, :]
    else:
        _spf = spectral_functions

    for gp, spf_sigmas in zip(grid_points, _spf):
        for sigma, spf_temps in zip(_sigmas, spf_sigmas):
            for t, spf in zip(temperatures, spf_temps):
                for i, bi in enumerate(band_indices):
                    pos = 0
                    for j in range(i):
                        pos += len(band_indices[j])
                    filename = write_spectral_function_at_grid_point(
                        gp,
                        bi,
                        frequency_points,
                        spf[pos:(pos + len(bi))].sum(axis=0) / len(bi),
                        mesh,
                        t,
                        filename=output_filename,
                        is_mesh_symmetry=is_mesh_symmetry)
                    if log_level:
                        print("Spectral functions were "
                              "written to \"%s\"." % filename)


class SpectralFunction(object):
    """Calculate spectral function"""

    def __init__(self,
                 interaction,
                 grid_points,
                 frequency_points=None,
                 frequency_step=None,
                 num_frequency_points=None,
                 temperatures=None,
                 log_level=0):
        self._interaction = interaction
        self._grid_points = grid_points
        self._frequency_points_in = frequency_points
        self._frequency_step = frequency_step
        self._num_frequency_points = num_frequency_points
        self._temperatures = temperatures
        self._log_level = log_level

        self._spectral_functions = None
        self._gammas = None
        self._deltas = None
        self._frequency_points = None

    def run(self):
        self._set_frequency_points()

        self._gammas = np.zeros(
            (len(self._grid_points), len(self._temperatures),
             len(self._interaction.band_indices), len(self._frequency_points)),
            dtype='double', order='C')
        self._deltas = np.zeros_like(self._gammas)
        self._spectral_functions = np.zeros_like(self._gammas)

        if self._log_level:
            print("-" * 28 + " Spectral function " + "-" * 28)

        for i, gp in enumerate(self._grid_points):
            self._run_gamma(i, gp)
            if self._log_level:
                print("-" * 11 +
                      " Real part of self energy by Kramers-Kronig relation "
                      + "-" * 11)
            self._run_delta(i)
            self._run_spectral_function(i, gp)
        if self._log_level:
            print("-" * 75)

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

    @property
    def grid_points(self):
        return self._grid_points

    def _run_gamma(self, i, grid_point):
        # gammas[grid_points, sigmas, temps, band_indices, freq_points]
        fpoints, gammas = get_imag_self_energy(
            self._interaction,
            [grid_point, ],
            frequency_points=self._frequency_points_in,
            frequency_step=self._frequency_step,
            num_frequency_points=self._num_frequency_points,
            temperatures=self._temperatures,
            log_level=self._log_level)
        self._gammas[i] = np.array(gammas[0, 0], dtype='double', order='C')
        assert (np.abs(self._frequency_points - fpoints) < 1e-8).all()

    def _run_delta(self, i):
        for j, temp in enumerate(self._temperatures):
            for k, bi in enumerate(self._interaction.band_indices):
                re_part, fpoints = imag_to_real(
                    self._gammas[i, j, k], self._frequency_points)
                self._deltas[i, j, k] = -re_part
        assert (np.abs(self._frequency_points - fpoints) < 1e-8).all()

    def _run_spectral_function(self, i, grid_point):
        frequencies = self._interaction.get_phonons()[0]
        for j, temp in enumerate(self._temperatures):
            for k, bi in enumerate(self._interaction.band_indices):
                freq = frequencies[grid_point, bi]
                gammas = self._gammas[i, j, k]
                deltas = self._deltas[i, j, k]
                vals = self._get_spectral_function(gammas, deltas, freq)
                self._spectral_functions[i, j, k] = vals

    def _get_spectral_function(self, gammas, deltas, freq):
        fpoints = self._frequency_points
        nums = 4 * freq ** 2 * gammas
        denoms = ((fpoints ** 2 - freq ** 2 - 2 * freq * deltas) ** 2
                  + (2 * freq * gammas) ** 2)
        vals = np.where(denoms > 0, nums / denoms, 0)
        return vals

    def _set_frequency_points(self):
        if (self._interaction.get_phonons()[2] == 0).any():
            if self._log_level:
                print("Running harmonic phonon calculations...")
            self._interaction.run_phonon_solver()
        max_phonon_freq = np.amax(self._interaction.get_phonons()[0])
        self._frequency_points = get_frequency_points(
            max_phonon_freq=max_phonon_freq,
            frequency_points=self._frequency_points_in,
            frequency_step=self._frequency_step,
            num_frequency_points=self._num_frequency_points)
