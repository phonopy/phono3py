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

import sys
import numpy as np
from phono3py.phonon3.imag_self_energy import (
    run_ise_at_frequency_points_batch, get_frequency_points, ImagSelfEnergy)
from phono3py.phonon3.real_self_energy import imag_to_real
from phono3py.file_IO import (
    write_spectral_function_at_grid_point, write_spectral_function_to_hdf5)


def run_spectral_function(interaction,
                          grid_points,
                          frequency_points=None,
                          frequency_step=None,
                          num_frequency_points=None,
                          num_points_in_batch=None,
                          sigmas=None,
                          temperatures=None,
                          band_indices=None,
                          write_txt=False,
                          write_hdf5=False,
                          output_filename=None,
                          log_level=0):
    spf = SpectralFunction(interaction,
                           grid_points,
                           frequency_points=frequency_points,
                           frequency_step=frequency_step,
                           num_frequency_points=num_frequency_points,
                           num_points_in_batch=num_points_in_batch,
                           sigmas=sigmas,
                           temperatures=temperatures,
                           log_level=log_level)
    for i, gp in enumerate(spf):
        frequencies = interaction.get_phonons()[0]
        for sigma_i, sigma in enumerate(spf.sigmas):
            for t, spf_at_t in zip(
                    temperatures, spf.spectral_functions[sigma_i, i]):
                for j, bi in enumerate(band_indices):
                    pos = 0
                    for k in range(j):
                        pos += len(band_indices[k])
                    filename = write_spectral_function_at_grid_point(
                        gp,
                        bi,
                        spf.frequency_points,
                        spf_at_t[pos:(pos + len(bi))].sum(axis=0) / len(bi),
                        interaction.mesh_numbers,
                        t,
                        sigma=sigma,
                        filename=output_filename,
                        is_mesh_symmetry=interaction.is_mesh_symmetry)
                    if log_level:
                        print("Spectral functions were written to")
                        print("\"%s\"." % filename)

            filename = write_spectral_function_to_hdf5(
                gp,
                band_indices,
                temperatures,
                spf.spectral_functions[sigma_i, i],
                spf.shifts[sigma_i, i],
                spf.half_linewidths[sigma_i, i],
                interaction.mesh_numbers,
                sigma=sigma,
                frequency_points=spf.frequency_points,
                frequencies=frequencies[gp],
                filename=output_filename)

            if log_level:
                print("Spectral functions were stored in \"%s\"." % filename)
                sys.stdout.flush()

    return spf


class SpectralFunction(object):
    """Calculate spectral function"""

    def __init__(self,
                 interaction,
                 grid_points,
                 frequency_points=None,
                 frequency_step=None,
                 num_frequency_points=None,
                 num_points_in_batch=None,
                 sigmas=None,
                 temperatures=None,
                 log_level=0):
        self._interaction = interaction
        self._grid_points = grid_points
        self._frequency_points_in = frequency_points
        self._num_points_in_batch = num_points_in_batch
        self._frequency_step = frequency_step
        self._num_frequency_points = num_frequency_points
        self._temperatures = temperatures
        self._log_level = log_level

        if sigmas is None:
            self._sigmas = [None, ]
        else:
            self._sigmas = sigmas
        self._frequency_points = None
        self._gammas = None
        self._deltas = None
        self._spectral_functions = None
        self._gp_index = None

    def run(self):
        for gp_index in self:
            pass

    def __iter__(self):
        self._prepare()

        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self._gp_index >= len(self._grid_points):
            if self._log_level:
                print("-" * 74)
            raise StopIteration

        if self._log_level:
            print(("-" * 24 + " Spectral function (%d/%d) " + "-" * 24)
                  % (self._gp_index + 1, len(self._grid_points)))

        gp = self._grid_points[self._gp_index]
        ise = ImagSelfEnergy(self._interaction)
        ise.set_grid_point(gp)

        if self._log_level:
            print("Running ph-ph interaction calculation...")
            sys.stdout.flush()

        ise.run_interaction()

        for sigma_i, sigma in enumerate(self._sigmas):
            self._run_gamma(ise, self._gp_index, sigma, sigma_i)
            self._run_delta(self._gp_index, sigma_i)
            self._run_spectral_function(self._gp_index, gp, sigma_i)

        self._gp_index += 1
        return gp

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

    @property
    def sigmas(self):
        return self._sigmas

    def _prepare(self):
        self._set_frequency_points()
        self._gammas = np.zeros(
            (len(self._sigmas), len(self._grid_points),
             len(self._temperatures), len(self._interaction.band_indices),
             len(self._frequency_points)),
            dtype='double', order='C')
        self._deltas = np.zeros_like(self._gammas)
        self._spectral_functions = np.zeros_like(self._gammas)
        self._gp_index = 0

    def _run_gamma(self, ise, i, sigma, sigma_i):
        if self._log_level:
            print("* Imaginary part of self energy")

        ise.set_sigma(sigma)
        run_ise_at_frequency_points_batch(
            self._frequency_points,
            ise,
            self._temperatures,
            self._gammas[sigma_i, i],
            nelems_in_batch=self._num_points_in_batch,
            log_level=self._log_level)

    def _run_delta(self, i, sigma_i):
        if self._log_level:
            print("* Real part of self energy")
            print("Running Kramers-Kronig relation integration...")

        for j, temp in enumerate(self._temperatures):
            for k, bi in enumerate(self._interaction.band_indices):
                re_part, fpoints = imag_to_real(
                    self._gammas[sigma_i, i, j, k], self._frequency_points)
                self._deltas[sigma_i, i, j, k] = -re_part
        assert (np.abs(self._frequency_points - fpoints) < 1e-8).all()

    def _run_spectral_function(self, i, grid_point, sigma_i):
        """Compute spectral functions from self-energies

        Note
        ----
        Spectral function A is defined by

                -1        G_0
            A = -- Im -----------
                pi    1 - G_0 Pi

        where pi = 3.14..., and Pi is the self energy, Pi = Delta - iGamma.
        It is expected that the integral of A over frequency is
        approximately 1 for each phonon mode.

        """

        if self._log_level:
            print("* Spectral function")
        frequencies = self._interaction.get_phonons()[0]
        for j, temp in enumerate(self._temperatures):
            for k, bi in enumerate(self._interaction.band_indices):
                freq = frequencies[grid_point, bi]
                gammas = self._gammas[sigma_i, i, j, k]
                deltas = self._deltas[sigma_i, i, j, k]
                vals = self._get_spectral_function(gammas, deltas, freq)
                self._spectral_functions[sigma_i, i, j, k] = vals

    def _get_spectral_function(self, gammas, deltas, freq):
        fpoints = self._frequency_points
        nums = 4 * freq ** 2 * gammas / np.pi
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
            sigmas=self._sigmas,
            frequency_points=self._frequency_points_in,
            frequency_step=self._frequency_step,
            num_frequency_points=self._num_frequency_points)
