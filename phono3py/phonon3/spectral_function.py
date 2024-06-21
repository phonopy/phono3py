"""Calculate spectral function due to bubble diagram."""

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

from phono3py.file_IO import (
    write_spectral_function_at_grid_point,
    write_spectral_function_to_hdf5,
)
from phono3py.phonon3.imag_self_energy import (
    ImagSelfEnergy,
    get_frequency_points,
    run_ise_at_frequency_points_batch,
)
from phono3py.phonon3.interaction import Interaction, all_bands_exist
from phono3py.phonon3.real_self_energy import imag_to_real


def run_spectral_function(
    interaction: Interaction,
    grid_points,
    temperatures=None,
    sigmas=None,
    frequency_points=None,
    frequency_step=None,
    num_frequency_points=None,
    num_points_in_batch=None,
    band_indices=None,
    write_txt=False,
    write_hdf5=False,
    output_filename=None,
    log_level=0,
):
    """Spectral function of self energy at frequency points.

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
    sigmas : array_like, optional
        A set of sigmas. simgas=[None, ] means to use tetrahedron method,
        otherwise smearing method with real positive value of sigma.
        For example, sigmas=[None, 0.01, 0.03] is possible. Default is None,
        which results in [None, ].
        dtype=float, shape=(sigmas,)
    frequency_points : array_like, optional
        Frequency sampling points. Default is None. With
        frequency_points_at_bands=False and frequency_points is None,
        num_frequency_points or frequency_step is used to generate uniform
        frequency sampling points.
        dtype=float, shape=(frequency_points,)
    frequency_step : float, optional
        Uniform pitch of frequency sampling points. Default is None. This
        results in using num_frequency_points.
    num_frequency_points : int, optional
        Number of sampling sampling points to be used instead of
        frequency_step. This number includes end points. Default is None,
        which gives 201.
    num_points_in_batch : int, optional
        Number of sampling points in one batch. This is for the frequency
        sampling mode and the sampling points are divided into batches.
        Lager number provides efficient use of multi-cores but more
        memory demanding. Default is None, which give the number of 10.
    band_indices : list
        Lists of list. Each list in list contains band indices.
    log_level: int
        Log level. Default is 0.

    Returns
    -------
    SpectralFunction
        spf.spectral_functions.shape = (sigmas, temperatures, grid_points,
                                        band_indices, frequency_points)
        spf.half_linewidths, spf.shifts have the same shape as above.

    """
    spf = SpectralFunction(
        interaction,
        grid_points,
        frequency_points=frequency_points,
        frequency_step=frequency_step,
        num_frequency_points=num_frequency_points,
        num_points_in_batch=num_points_in_batch,
        sigmas=sigmas,
        temperatures=temperatures,
        log_level=log_level,
    )
    for i, gp in enumerate(spf):
        frequencies = interaction.get_phonons()[0]
        for sigma_i, sigma in enumerate(spf.sigmas):
            for t, spf_at_t in zip(temperatures, spf.spectral_functions[sigma_i, :, i]):
                for j, bi in enumerate(band_indices):
                    pos = 0
                    for k in range(j):
                        pos += len(band_indices[k])
                    if write_txt:
                        filename = write_spectral_function_at_grid_point(
                            gp,
                            bi,
                            spf.frequency_points,
                            spf_at_t[pos : (pos + len(bi))].sum(axis=0) / len(bi),
                            interaction.mesh_numbers,
                            t,
                            sigma=sigma,
                            filename=output_filename,
                            is_mesh_symmetry=interaction.is_mesh_symmetry,
                        )
                    if log_level:
                        print(f'Spectral functions were written to "{filename}".')

            if write_hdf5:
                filename = write_spectral_function_to_hdf5(
                    gp,
                    bi,
                    temperatures,
                    spf.spectral_functions[sigma_i, :, i],
                    spf.shifts[sigma_i, :, i],
                    spf.half_linewidths[sigma_i, :, i],
                    interaction.mesh_numbers,
                    interaction.bz_grid,
                    sigma=sigma,
                    frequency_points=spf.frequency_points,
                    frequencies=frequencies[gp],
                    all_band_exist=all_bands_exist(interaction),
                    filename=output_filename,
                )

            if log_level:
                print(f'Spectral functions were stored in "{filename}".')
                sys.stdout.flush()

    return spf


class SpectralFunction:
    """Calculate spectral function due to bubble diagram."""

    def __init__(
        self,
        interaction: Interaction,
        grid_points,
        frequency_points=None,
        frequency_step=None,
        num_frequency_points=None,
        num_points_in_batch=None,
        sigmas=None,
        temperatures=None,
        log_level=0,
    ):
        """Init method."""
        self._pp = interaction
        self._grid_points = grid_points
        self._frequency_points_in = frequency_points
        self._num_points_in_batch = num_points_in_batch
        self._frequency_step = frequency_step
        self._num_frequency_points = num_frequency_points
        self._temperatures = temperatures
        self._log_level = log_level

        if sigmas is None:
            self._sigmas = [
                None,
            ]
        else:
            self._sigmas = sigmas
        self._frequency_points = None
        self._gammas = None
        self._deltas = None
        self._spectral_functions = None
        self._gp_index = None

    def run(self):
        """Calculate spectral function over grid points."""
        for _ in self:
            pass

    def __iter__(self):
        """Initialize iterator."""
        self._prepare()

        return self

    def __next__(self):
        """Calculate at next grid point."""
        if self._gp_index >= len(self._grid_points):
            if self._log_level:
                print("-" * 74)
            raise StopIteration

        gp = self._grid_points[self._gp_index]
        qpoint = np.dot(
            self._pp.bz_grid.addresses[gp],
            self._pp.bz_grid.QDinv.T,
        )
        if self._log_level:
            print(
                ("-" * 24 + " Spectral function %d (%d/%d) " + "-" * 24)
                % (gp, self._gp_index + 1, len(self._grid_points))
            )
            print("q-point: (%5.2f %5.2f %5.2f)" % tuple(qpoint))

        ise = ImagSelfEnergy(self._pp)
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
        """Return calculated spectral functions."""
        return self._spectral_functions

    @property
    def shifts(self):
        """Return real part of self energies."""
        return self._deltas

    @property
    def half_linewidths(self):
        """Return imaginary part of self energies."""
        return self._gammas

    @property
    def frequency_points(self):
        """Return frequency points."""
        return self._frequency_points

    @property
    def grid_points(self):
        """Return grid points."""
        return self._grid_points

    @property
    def sigmas(self):
        """Return sigmas."""
        return self._sigmas

    def _prepare(self):
        self._set_frequency_points()
        self._gammas = np.zeros(
            (
                len(self._sigmas),
                len(self._temperatures),
                len(self._grid_points),
                len(self._pp.band_indices),
                len(self._frequency_points),
            ),
            dtype="double",
            order="C",
        )
        self._deltas = np.zeros_like(self._gammas)
        self._spectral_functions = np.zeros_like(self._gammas)
        self._gp_index = 0

    def _run_gamma(self, ise, i, sigma, sigma_i):
        if self._log_level:
            print("* Imaginary part of self energy")

        ise.set_sigma(sigma)
        run_ise_at_frequency_points_batch(
            i,
            sigma_i,
            self._frequency_points,
            ise,
            self._temperatures,
            self._gammas,
            nelems_in_batch=self._num_points_in_batch,
            log_level=self._log_level,
        )

    def _run_delta(self, i, sigma_i):
        if self._log_level:
            print("* Real part of self energy")
            print("Running Kramers-Kronig relation integration...")

        for j, _ in enumerate(self._temperatures):
            for k, _ in enumerate(self._pp.band_indices):
                re_part, fpoints = imag_to_real(
                    self._gammas[sigma_i, j, i, k], self._frequency_points
                )
                self._deltas[sigma_i, j, i, k] = -re_part
        assert (np.abs(self._frequency_points - fpoints) < 1e-8).all()

    def _run_spectral_function(self, i, grid_point, sigma_i):
        """Compute spectral functions from self-energies.

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
        frequencies = self._pp.get_phonons()[0]
        for j, _ in enumerate(self._temperatures):
            for k, bi in enumerate(self._pp.band_indices):
                freq = frequencies[grid_point, bi]
                gammas = self._gammas[sigma_i, j, i, k]
                deltas = self._deltas[sigma_i, j, i, k]
                vals = self._get_spectral_function(gammas, deltas, freq)
                self._spectral_functions[sigma_i, j, i, k] = vals

    def _get_spectral_function(self, gammas, deltas, freq):
        fpoints = self._frequency_points
        nums = 4 * freq**2 * gammas / np.pi
        denoms = (fpoints**2 - freq**2 - 2 * freq * deltas) ** 2 + (
            2 * freq * gammas
        ) ** 2
        vals = np.where(denoms > 0, nums / denoms, 0)
        return vals

    def _set_frequency_points(self):
        if (self._pp.get_phonons()[2] == 0).any():
            if self._log_level:
                print("Running harmonic phonon calculations...")
            self._pp.run_phonon_solver()

        # Set phonon at Gamma without NAC for finding max_phonon_freq.
        self._pp.run_phonon_solver_at_gamma()
        max_phonon_freq = np.amax(self._pp.get_phonons()[0])
        self._pp.run_phonon_solver_at_gamma(is_nac=True)

        self._frequency_points = get_frequency_points(
            max_phonon_freq=max_phonon_freq,
            sigmas=self._sigmas,
            frequency_points=self._frequency_points_in,
            frequency_step=self._frequency_step,
            num_frequency_points=self._num_frequency_points,
        )
