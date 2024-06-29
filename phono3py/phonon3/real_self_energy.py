"""Calculate real-part of self-energy of bubble diagram."""

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
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.units import EV, Hbar, THz

from phono3py.file_IO import (
    write_real_self_energy_at_grid_point,
    write_real_self_energy_to_hdf5,
)
from phono3py.phonon.func import bose_einstein
from phono3py.phonon3.imag_self_energy import get_frequency_points
from phono3py.phonon3.interaction import Interaction


class RealSelfEnergy:
    """Class to calculate real-part of self-energy of bubble diagram.

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

    default_epsilon = 0.05

    def __init__(
        self,
        interaction: Interaction,
        grid_point=None,
        temperature=None,
        epsilon=None,
        lang="C",
    ):
        """Init method.

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
        self._frequency_points = None
        self._real_self_energies = None

        # Unit to THz of Delta
        self._unit_conversion = 18 / (Hbar * EV) ** 2 / (2 * np.pi * THz) ** 2 * EV**2

    def run(self):
        """Calculate real-part of self-energies."""
        if self._pp_strength is None:
            self.run_interaction()

        num_band0 = len(self._pp.band_indices)
        if self._frequency_points is None:
            self._real_self_energies = np.zeros(num_band0, dtype="double")
            self._run_with_band_indices()
        else:
            self._real_self_energies = np.zeros(
                (len(self._frequency_points), num_band0), dtype="double"
            )
            self._run_with_frequency_points()

    def run_interaction(self):
        """Calculate ph-ph interaction strength."""
        self._pp.run(lang=self._lang)
        self._pp_strength = self._pp.interaction_strength
        (self._frequencies, self._eigenvectors) = self._pp.get_phonons()[:2]
        (self._triplets_at_q, self._weights_at_q) = self._pp.get_triplets_at_q()[:2]
        self._band_indices = self._pp.band_indices

    @property
    def real_self_energy(self):
        """Return calculated real-part of self-energies."""
        if self._cutoff_frequency is None:
            return self._real_self_energies
        else:  # Averaging frequency shifts by degenerate bands
            shifts = np.zeros_like(self._real_self_energies)
            freqs = self._frequencies[self._grid_point]
            deg_sets = degenerate_sets(freqs)  # like [[0,1], [2], [3,4,5]]
            for dset in deg_sets:
                bi_set = []
                for i, bi in enumerate(self._band_indices):
                    if bi in dset:
                        bi_set.append(i)
                if len(bi_set) > 0:
                    for i in bi_set:
                        if self._frequency_points is None:
                            shifts[i] = self._real_self_energies[bi_set].sum() / len(
                                bi_set
                            )
                        else:
                            shifts[:, i] = self._real_self_energies[:, bi_set].sum(
                                axis=1
                            ) / len(bi_set)
            return shifts

    @property
    def grid_point(self):
        """Setter and getter of a grid point."""
        return self._grid_point

    @grid_point.setter
    def grid_point(self, grid_point=None):
        if grid_point is None:
            self._grid_point = None
        else:
            self._pp.set_grid_point(grid_point)
            self._pp_strength = None
            (self._triplets_at_q, self._weights_at_q) = self._pp.get_triplets_at_q()[:2]
            self._grid_point = self._triplets_at_q[0, 0]

    @property
    def epsilon(self):
        """Setter and getter of epsilon.

        See the detail about epsilon at docstring of this class.

        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        if epsilon is None:
            self._epsilon = self.default_epsilon
        else:
            self._epsilon = float(epsilon)

    @property
    def temperature(self):
        """Setter and getter of a temperature point."""
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)

    @property
    def frequency_points(self):
        """Setter and getter of frequency points."""
        return self._frequency_points

    @frequency_points.setter
    def frequency_points(self, frequency_points):
        self._frequency_points = np.array(frequency_points, dtype="double")

    def _run_with_band_indices(self):
        if self._lang == "C":
            self._run_c_with_band_indices()
        else:
            self._run_py_with_band_indices()

    def _run_with_frequency_points(self):
        if self._lang == "C":
            self._run_c_with_frequency_points()
        else:
            self._run_py_with_frequency_points()

    def _run_c_with_band_indices(self):
        import phono3py._phono3py as phono3c

        phono3c.real_self_energy_at_bands(
            self._real_self_energies,
            self._pp_strength,
            self._triplets_at_q,
            self._weights_at_q,
            self._frequencies,
            self._band_indices,
            self._temperature,
            self._epsilon,
            self._unit_conversion,
            self._cutoff_frequency,
        )

    def _run_py_with_band_indices(self):
        for triplet, w, interaction in zip(
            self._triplets_at_q, self._weights_at_q, self._pp_strength
        ):
            freqs = self._frequencies[triplet]
            for j, bi in enumerate(self._band_indices):
                fpoint = freqs[0, bi]
                if self._temperature > 0:
                    self._real_self_energies[j] += self._real_self_energies_at_bands(
                        j, fpoint, freqs, interaction, w
                    )
                else:
                    self._real_self_energies[j] += self._real_self_energies_at_bands_0K(
                        j, fpoint, freqs, interaction, w
                    )

        self._real_self_energies *= self._unit_conversion

    def _run_c_with_frequency_points(self):
        import phono3py._phono3py as phono3c

        for i, fpoint in enumerate(self._frequency_points):
            shifts = np.zeros(self._real_self_energies.shape[1], dtype="double")
            phono3c.real_self_energy_at_frequency_point(
                shifts,
                fpoint,
                self._pp_strength,
                self._triplets_at_q,
                self._weights_at_q,
                self._frequencies,
                self._band_indices,
                self._temperature,
                self._epsilon,
                self._unit_conversion,
                self._cutoff_frequency,
            )
            self._real_self_energies[i][:] = shifts

    def _run_py_with_frequency_points(self):
        for k, fpoint in enumerate(self._frequency_points):
            for triplet, w, interaction in zip(
                self._triplets_at_q, self._weights_at_q, self._pp_strength
            ):
                freqs = self._frequencies[triplet]
                for j, _ in enumerate(self._band_indices):
                    if self._temperature > 0:
                        self._real_self_energies[k, j] += (
                            self._real_self_energies_at_bands(
                                j, fpoint, freqs, interaction, w
                            )
                        )
                    else:
                        self._real_self_energies[k, j] += (
                            self._real_self_energies_at_bands_0K(
                                j, fpoint, freqs, interaction, w
                            )
                        )

        self._real_self_energies *= self._unit_conversion

    def _real_self_energies_at_bands(self, i, fpoint, freqs, interaction, weight):
        if fpoint < self._cutoff_frequency:
            return 0

        sum_d = 0
        for j, k in list(np.ndindex(interaction.shape[1:])):
            if fpoint < self._cutoff_frequency:
                continue

            if (
                freqs[1, j] > self._cutoff_frequency
                and freqs[2, k] > self._cutoff_frequency
            ):
                d = 0.0
                n2 = bose_einstein(freqs[1, j], self._temperature)
                n3 = bose_einstein(freqs[2, k], self._temperature)
                f1 = fpoint + freqs[1, j] + freqs[2, k]
                f2 = fpoint - freqs[1, j] - freqs[2, k]
                f3 = fpoint - freqs[1, j] + freqs[2, k]
                f4 = fpoint + freqs[1, j] - freqs[2, k]

                # if abs(f1) > self._epsilon:
                #     d -= (n2 + n3 + 1) / f1
                # if abs(f2) > self._epsilon:
                #     d += (n2 + n3 + 1) / f2
                # if abs(f3) > self._epsilon:
                #     d -= (n2 - n3) / f3
                # if abs(f4) > self._epsilon:
                #     d += (n2 - n3) / f4
                d -= (n2 + n3 + 1) * f1 / (f1**2 + self._epsilon**2)
                d += (n2 + n3 + 1) * f2 / (f2**2 + self._epsilon**2)
                d -= (n2 - n3) * f3 / (f3**2 + self._epsilon**2)
                d += (n2 - n3) * f4 / (f4**2 + self._epsilon**2)

                sum_d += d * interaction[i, j, k] * weight
        return sum_d

    def _real_self_energies_at_bands_0K(self, i, fpoint, freqs, interaction, weight):
        if fpoint < self._cutoff_frequency:
            return 0

        sum_d = 0
        for j, k in list(np.ndindex(interaction.shape[1:])):
            if (
                freqs[1, j] > self._cutoff_frequency
                and freqs[2, k] > self._cutoff_frequency
            ):
                d = 0.0
                f1 = fpoint + freqs[1, j] + freqs[2, k]
                f2 = fpoint - freqs[1, j] - freqs[2, k]

                # if abs(f1) > self._epsilon:
                #     d -= 1.0 / f1
                # if abs(f2) > self._epsilon:
                #     d += 1.0 / f2
                d -= 1.0 * f1 / (f1**2 + self._epsilon**2)
                d += 1.0 * f2 / (f2**2 + self._epsilon**2)

                sum_d += d * interaction[i, j, k] * weight
        return sum_d


class ImagToReal:
    """Calculate real part of self-energy using Kramers-Kronig relation."""

    def __init__(self, im_part, frequency_points, diagram="bubble"):
        """Init method.

        Parameters
        ----------
        im_part : array_like
            Imaginary part of self-energy at frequency points.
            shape=(frequency_points,), dtype='double'
        frequency_points : array_like
            Frequency points sampled at constant intervale increasing
            order starting at 0 and ending around maximum phonon frequency
            in Brillouin zone.
            shape=(frequency_points,), dtype='double'
        diagram : str
            Only bubble diagram is implemented currently.

        """
        if diagram == "bubble":
            (
                self._im_part,
                self._all_frequency_points,
                self._df,
            ) = self._expand_bubble_im_part(im_part, frequency_points)
        else:
            raise RuntimeError("Only daigram='bubble' is implemented.")

        self._re_part = None
        self._frequency_points = None

    @property
    def re_part(self):
        """Return real part."""
        return self._re_part

    @property
    def frequency_points(self):
        """Return frequency points."""
        return self._frequency_points

    def run(self, method="pick_one"):
        """Calculate real part."""
        if method == "pick_one":
            self._re_part, self._frequency_points = self._pick_one()
        elif method == "half_shift":
            self._re_part, self._frequency_points = self._half_shift()
        else:
            raise RuntimeError("No method is found.")

    def _pick_one(self):
        """Calculate real-part with same frequency points excluding one point."""
        re_part = []
        fpoints = []
        coef = self._df / np.pi
        i_zero = (len(self._im_part) - 1) // 2
        for i, im_part_at_i in enumerate(self._im_part):
            if i < i_zero:
                continue
            fpoint = self._all_frequency_points[i]
            freqs = self._all_frequency_points - fpoint
            freqs[i] = 1
            val = ((self._im_part / freqs).sum() - im_part_at_i) * coef
            re_part.append(val)
            fpoints.append(fpoint)
        return (np.array(re_part, dtype="double"), np.array(fpoints, dtype="double"))

    def _half_shift(self):
        """Calculate real-part with half-shifted frequency points."""
        re_part = []
        fpoints = []
        coef = self._df / np.pi
        i_zero = (len(self._im_part) - 1) // 2
        for i, _ in enumerate(self._im_part):
            if i < i_zero:
                continue
            fpoint = self._all_frequency_points[i] + self._df / 2
            freqs = self._all_frequency_points - fpoint
            val = (self._im_part / freqs).sum() * coef
            re_part.append(val)
            fpoints.append(fpoint)
        return (np.array(re_part, dtype="double"), np.array(fpoints, dtype="double"))

    def _expand_bubble_im_part(self, im_part, frequency_points):
        if (np.abs(frequency_points[0]) > 1e-8).any():
            raise RuntimeError("The first element of frequency_points is not zero.")

        all_df = np.subtract(frequency_points[1:], frequency_points[:-1])
        df = np.mean(all_df)
        if (np.abs(all_df - df) > 1e-6).any():
            print(all_df)
            raise RuntimeError("Frequency interval of frequency_points is not uniform.")

        # im_part is inverted at omega < 0.
        _im_part = np.hstack([-im_part[::-1], im_part[1:]])
        _frequency_points = np.hstack([-frequency_points[::-1], frequency_points[1:]])

        return _im_part, _frequency_points, df


def get_real_self_energy(
    interaction: Interaction,
    grid_points,
    temperatures,
    epsilons=None,
    frequency_points=None,
    frequency_step=None,
    num_frequency_points=None,
    frequency_points_at_bands=False,
    write_hdf5=True,
    output_filename=None,
    log_level=0,
):
    """Real part of self energy at frequency points.

    Band indices to be calculated at are kept in Interaction instance.

    Parameters
    ----------
    interaction : Interaction
        Ph-ph interaction.
    grid_points : array_like
        Grid-point indices where imag-self-energeis are caclculated.
        dtype=int, shape=(grid_points,)
    temperatures : array_like
        Temperatures where imag-self-energies are calculated.
        dtype=float, shape=(temperatures,)
    epsilons : array_like
        Smearing widths to computer principal part. When multiple values
        are given frequency shifts for those values are returned.
        dtype=float, shape=(epsilons,)
    frequency_points : array_like, optional
        Frequency sampling points. Default is None. With
        frequency_points_at_bands=False and frequency_points is None,
        num_frequency_points or frequency_step is used to generate uniform
        frequency sampling points.
        dtype=float, shape=(frequency_points,)
    frequency_step : float, optional
        Uniform pitch of frequency sampling points. Default is None. This
        results in using num_frequency_points.
    num_frequency_points: int, optional
        Number of sampling sampling points to be used instead of
        frequency_step. This number includes end points. Default is None,
        which gives 201.
    frequency_points_at_bands : bool, optional
        Phonon band frequencies are used as frequency points when True.
        Default is False.
    num_points_in_batch: int, optional
        Number of sampling points in one batch. This is for the frequency
        sampling mode and the sampling points are divided into batches.
        Lager number provides efficient use of multi-cores but more
        memory demanding. Default is None, which give the number of 10.
    log_level: int
        Log level. Default is 0.

    Returns
    -------
    tuple :
        (frequency_points, all_deltas) are returned.

        When frequency_points_at_bands is True,

            all_deltas.shape = (epsilons, temperatures, grid_points,
                                band_indices)

        otherwise

            all_deltas.shape = (epsilons, temperatures, grid_points,
                                band_indices, frequency_points)

    """
    if epsilons is None:
        _epsilons = [
            None,
        ]
    else:
        _epsilons = epsilons

    _temperatures = np.array(temperatures, dtype="double")

    if (interaction.get_phonons()[2] == 0).any():
        if log_level:
            print("Running harmonic phonon calculations...")
        interaction.run_phonon_solver()

    fst = RealSelfEnergy(interaction)
    mesh = interaction.mesh_numbers
    bz_grid = interaction.bz_grid

    # Set phonon at Gamma without NAC for finding max_phonon_freq.
    interaction.run_phonon_solver_at_gamma()
    max_phonon_freq = np.amax(interaction.get_phonons()[0])
    interaction.run_phonon_solver_at_gamma(is_nac=True)

    band_indices = interaction.band_indices

    if frequency_points_at_bands:
        _frequency_points = None
        all_deltas = np.zeros(
            (len(_epsilons), len(_temperatures), len(grid_points), len(band_indices)),
            dtype="double",
            order="C",
        )
    else:
        _frequency_points = get_frequency_points(
            max_phonon_freq=max_phonon_freq,
            sigmas=epsilons,
            frequency_points=frequency_points,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
        )
        all_deltas = np.zeros(
            (
                len(_epsilons),
                len(_temperatures),
                len(grid_points),
                len(band_indices),
                len(_frequency_points),
            ),
            dtype="double",
            order="C",
        )
        fst.frequency_points = _frequency_points

    for j, gp in enumerate(grid_points):
        fst.grid_point = gp
        if log_level:
            weights = interaction.get_triplets_at_q()[1]
            if len(grid_points) > 1:
                print(
                    "------------------- Real part of self energy -o- (%d/%d) "
                    "-------------------" % (j + 1, len(grid_points))
                )
            else:
                print(
                    "----------------------- Real part of self energy -o- "
                    "-----------------------"
                )
            print("Grid point: %d" % gp)
            print("Number of ir-triplets: %d / %d" % (len(weights), weights.sum()))

        fst.run_interaction()
        frequencies = interaction.get_phonons()[0][gp]

        if log_level:
            bz_grid = interaction.bz_grid
            qpoint = np.dot(bz_grid.QDinv, bz_grid.addresses[gp])
            print("Phonon frequencies at (%4.2f, %4.2f, %4.2f):" % tuple(qpoint))
            for bi, freq in enumerate(frequencies):
                print("%3d  %f" % (bi + 1, freq))
            sys.stdout.flush()

        for i, epsilon in enumerate(_epsilons):
            fst.epsilon = epsilon
            for k, t in enumerate(_temperatures):
                fst.temperature = t
                fst.run()
                all_deltas[i, k, j] = fst.real_self_energy.T

                # if not frequency_points_at_bands:
                #     pos = 0
                #     for bi_set in [[bi, ] for bi in band_indices]:
                #         filename = write_real_self_energy(
                #             gp,
                #             bi_set,
                #             _frequency_points,
                #             all_deltas[i, k, j, pos:(pos + len(bi_set))],
                #             mesh,
                #             fst.epsilon,
                #             t,
                #             filename=output_filename)
                #         pos += len(bi_set)

                #         if log_level:
                #             print("Real part of self energies were stored in "
                #                   "\"%s\"." % filename)
                #         sys.stdout.flush()

            if write_hdf5:
                filename = write_real_self_energy_to_hdf5(
                    gp,
                    band_indices,
                    _temperatures,
                    all_deltas[i, :, j],
                    mesh,
                    fst.epsilon,
                    bz_grid=bz_grid,
                    frequency_points=_frequency_points,
                    frequencies=frequencies,
                    filename=output_filename,
                )

                if log_level:
                    print('Real part of self energies were stored in "%s".' % filename)
                    sys.stdout.flush()

    return _frequency_points, all_deltas


def write_real_self_energy(
    real_self_energy,
    mesh,
    grid_points,
    band_indices,
    frequency_points,
    temperatures,
    epsilons,
    output_filename=None,
    is_mesh_symmetry=True,
    log_level=0,
):
    """Write real-part of self-energies into files."""
    if epsilons is None:
        _epsilons = [
            RealSelfEnergy.default_epsilon,
        ]
    else:
        _epsilons = epsilons

    for epsilon, rse_temps in zip(_epsilons, real_self_energy):
        for t, rse_gps in zip(temperatures, rse_temps):
            for gp, rse in zip(grid_points, rse_gps):
                for i, bi in enumerate(band_indices):
                    pos = 0
                    for j in range(i):
                        pos += len(band_indices[j])
                    filename = write_real_self_energy_at_grid_point(
                        gp,
                        bi,
                        frequency_points,
                        rse[pos : (pos + len(bi))].sum(axis=0) / len(bi),
                        mesh,
                        epsilon,
                        t,
                        filename=output_filename,
                        is_mesh_symmetry=is_mesh_symmetry,
                    )
                    if log_level:
                        print(
                            "Real parts of self-energies were "
                            'written to "%s".' % filename
                        )


def imag_to_real(im_part, frequency_points):
    """Calculate real-part of self-energy from the imaginary-part."""
    i2r = ImagToReal(im_part, frequency_points)
    i2r.run()
    return i2r.re_part, i2r.frequency_points
