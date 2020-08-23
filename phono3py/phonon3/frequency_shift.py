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
from phonopy.units import Hbar, EV, THz
from phonopy.phonon.degeneracy import degenerate_sets
from phono3py.phonon.func import bose_einstein
from phono3py.file_IO import write_frequency_shift, write_Delta_to_hdf5


def get_frequency_shift(interaction,
                        grid_points,
                        epsilons=None,
                        temperatures=None,
                        output_filename=None,
                        write_Delta_hdf5=True,
                        log_level=0):
    if epsilons is None:
        _epsilons = [None, ]
    else:
        _epsilons = epsilons

    if temperatures is None:
        _temperatures = [0.0, 300.0]
    else:
        _temperatures = temperatures
    _temperatures = np.array(_temperatures, dtype='double')

    band_indices = interaction.band_indices
    fst = FrequencyShift(interaction)
    mesh = interaction.mesh_numbers

    all_deltas = np.zeros((len(_epsilons), len(grid_points),
                           len(_temperatures), len(band_indices)),
                          dtype='double', order='C')

    for j, gp in enumerate(grid_points):
        fst.grid_point = gp
        if log_level:
            weights = interaction.get_triplets_at_q()[1]
            print("------ Frequency shift -o- at grid point %d ------" % gp)
            print("Number of ir-triplets: %d / %d"
                  % (len(weights), weights.sum()))

        fst.run_interaction()
        frequencies = interaction.get_phonons()[0][gp]

        if log_level:
            qpoint = interaction.grid_address[gp] / mesh.astype(float)
            print("Phonon frequencies at (%4.2f, %4.2f, %4.2f):"
                  % tuple(qpoint))
            for bi in band_indices:
                print("%3d  %f" % (bi + 1, frequencies[bi]))

        for i, epsilon in enumerate(_epsilons):
            fst.epsilon = epsilon
            delta = np.zeros((len(_temperatures), len(band_indices)),
                             dtype='double')
            for k, t in enumerate(_temperatures):
                fst.temperature = t
                fst.run()
                delta[k] = fst.frequency_shift

            all_deltas[i, j] = delta

            if write_Delta_hdf5:
                # pos = 0
                # for bi in band_indices:
                #     filename = write_frequency_shift(
                #         gp,
                #         bi,
                #         _temperatures,
                #         delta[:, pos:(pos + len(bi))],
                #         mesh,
                #         epsilon,
                #         filename=output_filename)
                #     pos += len(bi)
                #     print(filename, interaction.get_phonons()[0][gp][bi])
                #     sys.stdout.flush()

                filename = write_Delta_to_hdf5(
                    gp,
                    band_indices,
                    _temperatures,
                    delta,
                    mesh,
                    fst.epsilon,
                    frequencies=frequencies,
                    filename=output_filename)

                if log_level:
                    print("Frequency shfits with epsilon=%f were stored in"
                          % fst.epsilon)
                    print("\"%s\"." % filename)
                    sys.stdout.flush()

    return all_deltas


class FrequencyShift(object):

    default_epsilon = 0.05

    """

    About the parameter epsilon
    ---------------------------
    epsilon is the value to approximate 1/x by

        x / (x^2 + epsilon^2)

    where 1/x appears in Cauchy principal value, so it is expected to be
    around 1/x except near x=0, and zero near x=0.

    How to test epsilon
    -------------------
    At a sampling mesh, choose one band and calcualte frequency shifts at
    various epsilon values and plot over the epsilons. Decreasing epsinon,
    at some point, discontinous change may be found.

    """

    def __init__(self,
                 interaction,
                 grid_point=None,
                 temperature=None,
                 epsilon=None,
                 lang='C'):
        """

        Parameters
        ----------
        interaction : Interaction
            Instance of Interaction class that is ready to use, i.e., phonons
            are set properly, etc.
        grid_point : int, optional
            A grid point on a sampling mesh.
        temperature : float, optional
            Temperature in K.
        epsilon : float, optional
            Parameter explained above. The unit is consisered as THz.

        """

        self._pp = interaction
        self.epsilon = epsilon
        if temperature is None:
            self.temperature = 300.0
        else:
            self.temperature = temperature
        self.grid_point = grid_point

        self._lang = lang
        self._frequency_ = None
        self._pp_strength = None
        self._frequencies = None
        self._triplets_at_q = None
        self._weights_at_q = None
        self._band_indices = None
        self._unit_conversion = None
        self._cutoff_frequency = interaction.cutoff_frequency
        self._frequency_shifts = None

        # Unit to THz of Delta
        self._unit_conversion = (18 / (Hbar * EV) ** 2
                                 / (2 * np.pi * THz) ** 2
                                 * EV ** 2)

    def run(self):
        if self._pp_strength is None:
            self.run_interaction()

        num_band0 = self._pp_strength.shape[1]
        self._frequency_shifts = np.zeros(num_band0, dtype='double')
        self._run_with_band_indices()

    def run_interaction(self):
        self._pp.run(lang=self._lang)
        self._pp_strength = self._pp.interaction_strength
        (self._frequencies,
         self._eigenvectors) = self._pp.get_phonons()[:2]
        (self._triplets_at_q,
         self._weights_at_q) = self._pp.get_triplets_at_q()[:2]
        self._band_indices = self._pp.band_indices

    @property
    def frequency_shift(self):
        if self._cutoff_frequency is None:
            return self._frequency_shifts
        else:  # Averaging frequency shifts by degenerate bands
            shifts = np.zeros_like(self._frequency_shifts)
            freqs = self._frequencies[self._grid_point]
            deg_sets = degenerate_sets(freqs)  # like [[0,1], [2], [3,4,5]]
            for dset in deg_sets:
                bi_set = []
                for i, bi in enumerate(self._band_indices):
                    if bi in dset:
                        bi_set.append(i)
                if len(bi_set) > 0:
                    for i in bi_set:
                        shifts[i] = (self._frequency_shifts[bi_set].sum() /
                                     len(bi_set))
            return shifts

    @property
    def grid_point(self):
        return self._grid_point

    @grid_point.setter
    def grid_point(self, grid_point=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._pp.set_grid_point(grid_point)
            self._pp_strength = None
            (self._triplets_at_q,
             self._weights_at_q) = self._pp.get_triplets_at_q()[:2]
            self._grid_point = self._triplets_at_q[0, 0]

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        if epsilon is None:
            self._epsilon = self.default_epsilon
        else:
            self._epsilon = float(epsilon)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)

    def _run_with_band_indices(self):
        if self._lang == 'C':
            self._run_c_with_band_indices()
        else:
            self._run_py_with_band_indices()

    def _run_c_with_band_indices(self):
        import phono3py._phono3py as phono3c
        phono3c.frequency_shift_at_bands(self._frequency_shifts,
                                         self._pp_strength,
                                         self._triplets_at_q,
                                         self._weights_at_q,
                                         self._frequencies,
                                         self._band_indices,
                                         self._temperature,
                                         self._epsilon,
                                         self._unit_conversion,
                                         self._cutoff_frequency)

    def _run_py_with_band_indices(self):
        for i, (triplet, w, interaction) in enumerate(
            zip(self._triplets_at_q,
                self._weights_at_q,
                self._pp_strength)):

            freqs = self._frequencies[triplet]
            for j, bi in enumerate(self._band_indices):
                if self._temperature > 0:
                    self._frequency_shifts[j] += (
                        self._frequency_shifts_at_bands(
                            j, bi, freqs, interaction, w))
                else:
                    self._frequency_shifts[j] += (
                        self._frequency_shifts_at_bands_0K(
                            j, bi, freqs, interaction, w))

        self._frequency_shifts *= self._unit_conversion

    def _frequency_shifts_at_bands(self, i, bi, freqs, interaction, weight):
        sum_d = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            if (freqs[1, j] > self._cutoff_frequency and
                freqs[2, k] > self._cutoff_frequency):
                d = 0.0
                n2 = bose_einstein(freqs[1, j], self._temperature)
                n3 = bose_einstein(freqs[2, k], self._temperature)
                f1 = freqs[0, bi] + freqs[1, j] + freqs[2, k]
                f2 = freqs[0, bi] - freqs[1, j] - freqs[2, k]
                f3 = freqs[0, bi] - freqs[1, j] + freqs[2, k]
                f4 = freqs[0, bi] + freqs[1, j] - freqs[2, k]

                # if abs(f1) > self._epsilon:
                #     d -= (n2 + n3 + 1) / f1
                # if abs(f2) > self._epsilon:
                #     d += (n2 + n3 + 1) / f2
                # if abs(f3) > self._epsilon:
                #     d -= (n2 - n3) / f3
                # if abs(f4) > self._epsilon:
                #     d += (n2 - n3) / f4
                d -= (n2 + n3 + 1) * f1 / (f1 ** 2 + self._epsilon ** 2)
                d += (n2 + n3 + 1) * f2 / (f2 ** 2 + self._epsilon ** 2)
                d -= (n2 - n3) * f3 / (f3 ** 2 + self._epsilon ** 2)
                d += (n2 - n3) * f4 / (f4 ** 2 + self._epsilon ** 2)

                sum_d += d * interaction[i, j, k] * weight
        return sum_d

    def _frequency_shifts_at_bands_0K(self, i, bi, freqs, interaction, weight):
        sum_d = 0
        for (j, k) in list(np.ndindex(interaction.shape[1:])):
            if (freqs[1, j] > self._cutoff_frequency and
                freqs[2, k] > self._cutoff_frequency):
                d = 0.0
                f1 = freqs[0, bi] + freqs[1, j] + freqs[2, k]
                f2 = freqs[0, bi] - freqs[1, j] - freqs[2, k]

                # if abs(f1) > self._epsilon:
                #     d -= 1.0 / f1
                # if abs(f2) > self._epsilon:
                #     d += 1.0 / f2
                d -= 1.0 * f1 / (f1 ** 2 + self._epsilon ** 2)
                d += 1.0 * f2 / (f2 ** 2 + self._epsilon ** 2)

                sum_d += d * interaction[i, j, k] * weight
        return sum_d
