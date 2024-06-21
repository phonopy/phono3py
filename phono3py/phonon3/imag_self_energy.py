"""Calculation of imaginary-part of self-energy of bubble diagram."""

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
import warnings
from typing import List, Optional

import numpy as np
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.units import EV, Hbar, THz

from phono3py.file_IO import (
    write_gamma_detail_to_hdf5,
    write_imag_self_energy_at_grid_point,
)
from phono3py.phonon.func import bose_einstein
from phono3py.phonon3.interaction import Interaction
from phono3py.phonon3.triplets import get_triplets_integration_weights


class ImagSelfEnergy:
    """Class for imaginary-part of self-energy of bubble diagram."""

    def __init__(
        self,
        interaction: Interaction,
        with_detail=False,
        lang="C",
    ):
        """Init method.

        Band indices to be calculated at are kept in Interaction instance.

        Parameters
        ----------
        interaction : Interaction
            Class instance of ph-ph interaction.
        with_detail : bool, optional
            Contributions to gammas for each triplets are computed. Default is
            False.
        lang : str, optional
            This is used for debugging purpose.

        """
        self._pp = interaction
        self._sigma = None
        self._temperature = None
        self._frequency_points = None
        self._grid_point = None

        self._lang = lang
        self._imag_self_energy = None
        self._detailed_imag_self_energy = None
        self._pp_strength = None
        self._frequencies = None
        self._triplets_at_q = None
        self._weights_at_q = None
        self._with_detail = with_detail
        self._cutoff_frequency = interaction.cutoff_frequency

        self._g = None  # integration weights
        self._g_zero = None  # Necessary elements of interaction strength
        self._is_collision_matrix = False

        # Unit to THz of Gamma
        self._unit_conversion = (
            18 * np.pi / (Hbar * EV) ** 2 / (2 * np.pi * THz) ** 2 * EV**2
        )

    def run(self):
        """Calculate imaginary-part of self-energies."""
        if self._pp_strength is None:
            self.run_interaction()

        num_band0 = self._pp_strength.shape[1]

        if self._frequency_points is None:
            self._imag_self_energy = np.zeros(num_band0, dtype="double")
            if self._with_detail:
                self._detailed_imag_self_energy = np.empty_like(self._pp_strength)
                self._detailed_imag_self_energy[:] = 0
                self._ise_N = np.zeros_like(self._imag_self_energy)
                self._ise_U = np.zeros_like(self._imag_self_energy)
            self._run_with_band_indices()
        else:
            self._imag_self_energy = np.zeros(
                (len(self._frequency_points), num_band0), order="C", dtype="double"
            )
            if self._with_detail:
                self._detailed_imag_self_energy = np.zeros(
                    (len(self._frequency_points),) + self._pp_strength.shape,
                    order="C",
                    dtype="double",
                )
                self._ise_N = np.zeros_like(self._imag_self_energy)
                self._ise_U = np.zeros_like(self._imag_self_energy)
            self._run_with_frequency_points()

    def run_interaction(self, is_full_pp=True):
        """Calculate ph-ph interaction."""
        if is_full_pp or self._frequency_points is not None:
            self._pp.run(lang=self._lang)
        else:
            self._pp.run(lang=self._lang, g_zero=self._g_zero)
        self._pp_strength = self._pp.interaction_strength

    def run_integration_weights(self, scattering_event_class=None):
        """Compute integration weights at grid points."""
        if self._frequency_points is None:
            bi = self._pp.band_indices
            f_points = self._frequencies[self._grid_point][bi]
        else:
            f_points = self._frequency_points

        self._g, self._g_zero = get_triplets_integration_weights(
            self._pp,
            np.array(f_points, dtype="double"),
            self._sigma,
            self._sigma_cutoff,
            is_collision_matrix=self._is_collision_matrix,
        )

        if scattering_event_class == 1 or scattering_event_class == 2:
            self._g[scattering_event_class - 1] = 0

    @property
    def imag_self_energy(self):
        """Return calculated imaginary-part of self-energies."""
        if self._cutoff_frequency is None:
            return self._imag_self_energy
        else:
            return self._average_by_degeneracy(self._imag_self_energy)

    def get_imag_self_energy(self):
        """Return calculated imaginary-part of self-energies."""
        warnings.warn(
            "Use attribute, ImagSelfEnergy.imag_self_energy "
            "instead of ImagSelfEnergy.get_imag_self_energy().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.imag_self_energy

    def get_imag_self_energy_N_and_U(self):
        """Return normal and Umklapp contributions.

        Three-phonon scatterings are categorized into normal and Umklapp and
        the contributions of the triplets to imaginary-part of self-energies
        are returned.

        """
        if self._cutoff_frequency is None:
            return self._ise_N, self._ise_U
        else:
            return (
                self._average_by_degeneracy(self._ise_N),
                self._average_by_degeneracy(self._ise_U),
            )

    @property
    def detailed_imag_self_energy(self):
        """Return triplets contributions to imaginary-part of self-energies."""
        return self._detailed_imag_self_energy

    def get_detailed_imag_self_energy(self):
        """Return triplets contributions to imaginary-part of self-energies."""
        warnings.warn(
            "Use attribute, ImagSelfEnergy.detailed_imag_self_energy "
            "instead of ImagSelfEnergy.get_detailed_imag_self_energy().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.detailed_imag_self_energy

    def get_integration_weights(self):
        """Return integration weights.

        See the details of returns at ``get_triplets_integration_weights``.

        """
        return self._g, self._g_zero

    @property
    def unit_conversion_factor(self):
        """Return unit conversion factor of gamma."""
        return self._unit_conversion

    def get_unit_conversion_factor(self):
        """Return unit conversion factor of gamma."""
        warnings.warn(
            "Use attribute, ImagSelfEnergy.unit_conversion_factor "
            "instead of ImagSelfEnergy.get_unit_conversion_factor().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.unit_conversion_factor

    def set_grid_point(self, grid_point=None):
        """Set a grid point at which calculation will be performed."""
        if grid_point is None:
            self._grid_point = None
        else:
            self._pp.set_grid_point(grid_point)
            self._pp_strength = None
            (self._triplets_at_q, self._weights_at_q) = self._pp.get_triplets_at_q()[:2]
            self._grid_point = grid_point
            self._frequencies, self._eigenvectors, _ = self._pp.get_phonons()

    def set_sigma(self, sigma, sigma_cutoff=None):
        """Set sigma value. None means tetrahedron method."""
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = float(sigma)

        if sigma_cutoff is None:
            self._sigma_cutoff = None
        else:
            self._sigma_cutoff = float(sigma_cutoff)

        self.delete_integration_weights()

    @property
    def frequency_points(self):
        """Getter and setter of sampling frequency points."""
        return self._frequency_points

    @frequency_points.setter
    def frequency_points(self, frequency_points):
        if frequency_points is None:
            self._frequency_points = None
        else:
            self._frequency_points = np.array(frequency_points, dtype="double")

    def set_frequency_points(self, frequency_points):
        """Set frequency points where spectrum calculation will be performed."""
        warnings.warn(
            "Use attribute, ImagSelfEnergy.frequency_points "
            "instead of ImagSelfEnergy.set_frequency_points().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.frequency_points = frequency_points

    @property
    def temperature(self):
        """Getter and setter of temperature."""
        return self._temperature

    @temperature.setter
    def temperature(self, temperature):
        if temperature is None:
            self._temperature = None
        else:
            self._temperature = float(temperature)

    def set_temperature(self, temperature):
        """Set temperatures where calculation will be peformed."""
        warnings.warn(
            "Use attribute, ImagSelfEnergy.temperature "
            "instead of ImagSelfEnergy.set_temperature().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.temperature = temperature

    def set_averaged_pp_interaction(self, ave_pp):
        """Set averaged ph-ph interactions.

        This is used for analysis of the calculated results by introducing
        averaged value as an approximation.
        Setting this, ph-ph interaction calculation will not be executed.

        """
        num_triplets = len(self._triplets_at_q)
        num_band = len(self._pp.primitive) * 3
        num_grid = np.prod(self._pp.mesh_numbers)
        bi = self._pp.get_band_indices()
        self._pp_strength = np.zeros(
            (num_triplets, len(bi), num_band, num_band), dtype="double"
        )

        for i, v_ave in enumerate(ave_pp):
            self._pp_strength[:, i, :, :] = v_ave / num_grid

    def set_interaction_strength(self, pp_strength, g_zero=None):
        """Set ph-ph interaction strengths."""
        self._pp_strength = pp_strength
        if g_zero is not None:
            self._g_zero = g_zero
        self._pp.set_interaction_strength(pp_strength, g_zero=g_zero)

    def delete_integration_weights(self):
        """Delete large ndarray's."""
        self._g = None
        self._g_zero = None
        self._pp_strength = None

    def _run_with_band_indices(self):
        if self._g is not None:
            if self._lang == "C":
                if self._with_detail:
                    # self._detailed_imag_self_energy.shape =
                    #    (num_triplets, num_band0, num_band, num_band)
                    # self._imag_self_energy is also set.
                    self._run_c_detailed_with_band_indices_with_g()
                else:
                    # self._imag_self_energy.shape = (num_band0,)
                    self._run_c_with_band_indices_with_g()
            else:
                print("Running into _run_py_with_band_indices_with_g()")
                print("This routine is super slow and only for the test.")
                self._run_py_with_band_indices_with_g()
        else:
            print(
                "get_triplets_integration_weights must be executed "
                "before calling this method."
            )
            import sys

            sys.exit(1)

    def _run_with_frequency_points(self):
        if self._g is not None:
            if self._lang == "C":
                if self._with_detail:
                    self._run_c_detailed_with_frequency_points_with_g()
                else:
                    self._run_c_with_frequency_points_with_g()
            else:
                print("Running into _run_py_with_frequency_points_with_g()")
                print("This routine is super slow and only for the test.")
                self._run_py_with_frequency_points_with_g()
        else:
            print(
                "get_triplets_integration_weights must be executed "
                "before calling this method."
            )
            import sys

            sys.exit(1)

    def _run_c_with_band_indices_with_g(self):
        import phono3py._phono3py as phono3c

        if self._g_zero is None:
            _g_zero = np.zeros(self._pp_strength.shape, dtype="byte", order="C")
        else:
            _g_zero = self._g_zero

        phono3c.imag_self_energy_with_g(
            self._imag_self_energy,
            self._pp_strength,
            self._triplets_at_q,
            self._weights_at_q,
            self._frequencies,
            self._temperature,
            self._g,
            _g_zero,
            self._cutoff_frequency,
            -1,
        )
        self._imag_self_energy *= self._unit_conversion

    def _run_c_detailed_with_band_indices_with_g(self):
        import phono3py._phono3py as phono3c

        if self._g_zero is None:
            _g_zero = np.zeros(self._pp_strength.shape, dtype="byte", order="C")
        else:
            _g_zero = self._g_zero

        phono3c.detailed_imag_self_energy_with_g(
            self._detailed_imag_self_energy,
            self._ise_N,  # Normal
            self._ise_U,  # Umklapp
            self._pp_strength,
            self._triplets_at_q,
            self._weights_at_q,
            self._pp.bz_grid.addresses,
            self._frequencies,
            self._temperature,
            self._g,
            _g_zero,
            self._cutoff_frequency,
        )

        self._detailed_imag_self_energy *= self._unit_conversion
        self._ise_N *= self._unit_conversion
        self._ise_U *= self._unit_conversion
        self._imag_self_energy = self._ise_N + self._ise_U

    def _run_c_with_frequency_points_with_g(self):
        import phono3py._phono3py as phono3c

        num_band0 = self._pp_strength.shape[1]
        ise_at_f = np.zeros(num_band0, dtype="double")

        for i in range(len(self._frequency_points)):
            phono3c.imag_self_energy_with_g(
                ise_at_f,
                self._pp_strength,
                self._triplets_at_q,
                self._weights_at_q,
                self._frequencies,
                self._temperature,
                self._g,
                self._g_zero,
                self._cutoff_frequency,
                i,
            )
            self._imag_self_energy[i] = ise_at_f
        self._imag_self_energy *= self._unit_conversion

    def _run_c_detailed_with_frequency_points_with_g(self):
        import phono3py._phono3py as phono3c

        num_band0 = self._pp_strength.shape[1]
        g_shape = list(self._g.shape)
        g_shape[2] = num_band0
        g = np.zeros((2,) + self._pp_strength.shape, order="C", dtype="double")
        detailed_ise_at_f = np.zeros(
            self._detailed_imag_self_energy.shape[1:5], order="C", dtype="double"
        )
        ise_at_f_N = np.zeros(num_band0, dtype="double")
        ise_at_f_U = np.zeros(num_band0, dtype="double")
        _g_zero = np.zeros(g_shape, dtype="byte", order="C")

        for i in range(len(self._frequency_points)):
            for j in range(g.shape[2]):
                g[:, :, j, :, :] = self._g[:, :, i, :, :]
                phono3c.detailed_imag_self_energy_with_g(
                    detailed_ise_at_f,
                    ise_at_f_N,
                    ise_at_f_U,
                    self._pp_strength,
                    self._triplets_at_q,
                    self._weights_at_q,
                    self._pp.bz_grid.addresses,
                    self._frequencies,
                    self._temperature,
                    g,
                    _g_zero,
                    self._cutoff_frequency,
                )
            self._detailed_imag_self_energy[i] = (
                detailed_ise_at_f * self._unit_conversion
            )
            self._ise_N[i] = ise_at_f_N * self._unit_conversion
            self._ise_U[i] = ise_at_f_U * self._unit_conversion
            self._imag_self_energy[i] = self._ise_N[i] + self._ise_U[i]

    def _run_py_with_band_indices_with_g(self):
        if self._temperature > 0:
            self._ise_thm_with_band_indices()
        else:
            self._ise_thm_with_band_indices_0K()

    def _ise_thm_with_band_indices(self):
        freqs = self._frequencies[self._triplets_at_q[:, [1, 2]]]
        freqs = np.where(freqs > self._cutoff_frequency, freqs, 1)
        n = bose_einstein(freqs, self._temperature)
        for i, (tp, w, interaction) in enumerate(
            zip(self._triplets_at_q, self._weights_at_q, self._pp_strength)
        ):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                f1 = self._frequencies[tp[1]][j]
                f2 = self._frequencies[tp[2]][k]
                if f1 > self._cutoff_frequency and f2 > self._cutoff_frequency:
                    n2 = n[i, 0, j]
                    n3 = n[i, 1, k]
                    g1 = self._g[0, i, :, j, k]
                    g2_g3 = self._g[1, i, :, j, k]  # g2 - g3
                    self._imag_self_energy[:] += (
                        ((n2 + n3 + 1) * g1 + (n2 - n3) * (g2_g3))
                        * interaction[:, j, k]
                        * w
                    )

        self._imag_self_energy *= self._unit_conversion

    def _ise_thm_with_band_indices_0K(self):
        for i, (w, interaction) in enumerate(
            zip(self._weights_at_q, self._pp_strength)
        ):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                self._imag_self_energy[:] += g1 * interaction[:, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _run_py_with_frequency_points_with_g(self):
        if self._temperature > 0:
            self._ise_thm_with_frequency_points()
        else:
            self._ise_thm_with_frequency_points_0K()

    def _ise_thm_with_frequency_points(self):
        for i, (tp, w, interaction) in enumerate(
            zip(self._triplets_at_q, self._weights_at_q, self._pp_strength)
        ):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                f1 = self._frequencies[tp[1]][j]
                f2 = self._frequencies[tp[2]][k]
                if f1 > self._cutoff_frequency and f2 > self._cutoff_frequency:
                    n2 = bose_einstein(f1, self._temperature)
                    n3 = bose_einstein(f2, self._temperature)
                    g1 = self._g[0, i, :, j, k]
                    g2_g3 = self._g[1, i, :, j, k]  # g2 - g3
                    for ll in range(len(interaction)):
                        self._imag_self_energy[:, ll] += (
                            ((n2 + n3 + 1) * g1 + (n2 - n3) * (g2_g3))
                            * interaction[ll, j, k]
                            * w
                        )

        self._imag_self_energy *= self._unit_conversion

    def _ise_thm_with_frequency_points_0K(self):
        for i, (w, interaction) in enumerate(
            zip(self._weights_at_q, self._pp_strength)
        ):
            for j, k in list(np.ndindex(interaction.shape[1:])):
                g1 = self._g[0, i, :, j, k]
                for ll in range(len(interaction)):
                    self._imag_self_energy[:, ll] += g1 * interaction[ll, j, k] * w

        self._imag_self_energy *= self._unit_conversion

    def _average_by_degeneracy(self, imag_self_energy):
        return average_by_degeneracy(
            imag_self_energy, self._pp.band_indices, self._frequencies[self._grid_point]
        )


def get_imag_self_energy(
    interaction: Interaction,
    grid_points,
    temperatures,
    sigmas=None,
    frequency_points=None,
    frequency_step=None,
    num_frequency_points=None,
    frequency_points_at_bands=False,
    num_points_in_batch=None,
    scattering_event_class=None,  # class 1 or 2
    write_gamma_detail=False,
    return_gamma_detail=False,
    output_filename=None,
    log_level=0,
):
    """Imaginary-part of self-energy at frequency points.

    Band indices to be calculated at are found in Interaction instance.

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
    scattering_event_class : int, optional
        Specific choice of scattering event class, 1 or 2 that is specified
        1 or 2, respectively. The result is stored in gammas. Therefore
        usual gammas are not stored in the variable. Default is None, which
        doesn't specify scattering_event_class.
    write_gamma_detail : bool, optional
        Detailed gammas are written into a file in hdf5. Default is False.
    return_gamma_detail : bool, optional
        With True, detailed gammas are returned. Default is False.
    log_level: int
        Log level. Default is 0.

    Returns
    -------
    tuple :
        (frequency_points, gammas) are returned. With return_gamma_detail=True,
        (frequency_points, gammas, detailed_gamma) are returned.
        detailed_gamma is a list of detailed_gamma_at_gp's.

        When frequency_points_at_bands is True,

            gamma.shape = (sigmas, temperatures, grid_points, band_indices)
            detailed_gamma_at_gp.shape = (sigmas, temperatures, triplets,
                                          band_indices, num_band, num_band)
        else:
            detailed_gamma_at_gp = np.zeros(
                (len(_sigmas), len(temperatures), _num_frequency_points,
                 len(weights), num_band0, num_band, num_band),
                dtype='double')

        otherwise

            gamma.shape = (sigmas, temperatures, grid_points,
                           band_indices, frequency_points)
            detailed_gamma_at_gp.shape = (sigmas, temperatures, triplets,
                                          frequency_points,
                                          band_indices, num_band, num_band)

    """
    if sigmas is None:
        _sigmas = [
            None,
        ]
    else:
        _sigmas = sigmas

    if (interaction.get_phonons()[2] == 0).any():
        if log_level:
            print("Running harmonic phonon calculations...")
        interaction.run_phonon_solver()

    # Set phonon at Gamma without NAC for finding max_phonon_freq.
    interaction.run_phonon_solver_at_gamma()
    max_phonon_freq = np.amax(interaction.get_phonons()[0])
    interaction.run_phonon_solver_at_gamma(is_nac=True)

    num_band0 = len(interaction.band_indices)

    if frequency_points_at_bands:
        _frequency_points = None
        _num_frequency_points = num_band0
        gamma = np.zeros(
            (len(_sigmas), len(temperatures), len(grid_points), _num_frequency_points),
            dtype="double",
            order="C",
        )
    else:
        _frequency_points = get_frequency_points(
            max_phonon_freq=max_phonon_freq,
            sigmas=_sigmas,
            frequency_points=frequency_points,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
        )
        _num_frequency_points = len(_frequency_points)
        gamma = np.zeros(
            (
                len(_sigmas),
                len(temperatures),
                len(grid_points),
                num_band0,
                _num_frequency_points,
            ),
            dtype="double",
            order="C",
        )

    detailed_gamma: List[Optional[np.ndarray]] = []

    ise = ImagSelfEnergy(
        interaction, with_detail=(write_gamma_detail or return_gamma_detail)
    )
    for i, gp in enumerate(grid_points):
        ise.set_grid_point(gp)

        if log_level:
            bz_grid = interaction.bz_grid
            weights = interaction.get_triplets_at_q()[1]
            if len(grid_points) > 1:
                print(
                    "---------------- Imaginary part of self energy -o- (%d/%d) "
                    "----------------" % (i + 1, len(grid_points))
                )
            else:
                print(
                    "-------------------- Imaginary part of self energy -o- "
                    "--------------------"
                )
            print("Grid point: %d" % gp)
            print("Number of ir-triplets: " "%d / %d" % (len(weights), weights.sum()))

        ise.run_interaction()
        frequencies = interaction.get_phonons()[0][gp]

        if log_level:
            qpoint = np.dot(bz_grid.QDinv, bz_grid.addresses[gp])
            print("Phonon frequencies at (%4.2f, %4.2f, %4.2f):" % tuple(qpoint))
            for bi, freq in enumerate(frequencies):
                print("%3d  %f" % (bi + 1, freq))
            sys.stdout.flush()

        _get_imag_self_energy_at_gp(
            gamma,
            detailed_gamma,
            i,
            gp,
            _sigmas,
            temperatures,
            _frequency_points,
            _num_frequency_points,
            scattering_event_class,
            num_points_in_batch,
            interaction,
            ise,
            write_gamma_detail,
            return_gamma_detail,
            output_filename,
            log_level,
        )

    if return_gamma_detail:
        return _frequency_points, gamma, detailed_gamma
    else:
        return _frequency_points, gamma


def _get_imag_self_energy_at_gp(
    gamma,
    detailed_gamma,
    i,
    gp,
    _sigmas,
    temperatures,
    _frequency_points,
    _num_frequency_points,
    scattering_event_class,
    num_points_in_batch,
    interaction: Interaction,
    ise,
    write_gamma_detail,
    return_gamma_detail,
    output_filename,
    log_level,
):
    num_band0 = len(interaction.band_indices)
    frequencies = interaction.get_phonons()[0]
    mesh = interaction.mesh_numbers
    bz_grid = interaction.bz_grid

    if write_gamma_detail or return_gamma_detail:
        triplets, weights, _, _ = interaction.get_triplets_at_q()
        num_band = frequencies.shape[1]
        if _frequency_points is None:
            detailed_gamma_at_gp = np.zeros(
                (
                    len(_sigmas),
                    len(temperatures),
                    len(weights),
                    num_band0,
                    num_band,
                    num_band,
                ),
                dtype="double",
            )
        else:
            detailed_gamma_at_gp = np.zeros(
                (
                    len(_sigmas),
                    len(temperatures),
                    _num_frequency_points,
                    len(weights),
                    num_band0,
                    num_band,
                    num_band,
                ),
                dtype="double",
            )
    else:
        detailed_gamma_at_gp = None

    for j, sigma in enumerate(_sigmas):
        if log_level:
            if sigma:
                print("Sigma: %s" % sigma)
            else:
                print("Tetrahedron method is used for BZ integration.")

        ise.set_sigma(sigma)
        _get_imag_self_energy_at_sigma(
            gamma,
            detailed_gamma_at_gp,
            i,
            j,
            temperatures,
            _frequency_points,
            scattering_event_class,
            num_points_in_batch,
            ise,
            write_gamma_detail,
            return_gamma_detail,
            log_level,
        )

        if write_gamma_detail:
            full_filename = write_gamma_detail_to_hdf5(
                temperatures,
                mesh,
                bz_grid=bz_grid,
                gamma_detail=detailed_gamma_at_gp[j],
                grid_point=gp,
                triplet=triplets,
                weight=weights,
                sigma=sigma,
                frequency_points=_frequency_points,
                filename=output_filename,
            )

            if log_level:
                print(
                    "Contribution of each triplet to imaginary part of "
                    'self energy is written in\n"%s".' % full_filename
                )

        if return_gamma_detail:
            detailed_gamma.append(detailed_gamma_at_gp)


def _get_imag_self_energy_at_sigma(
    gamma,
    detailed_gamma_at_gp,
    i,
    j,
    temperatures,
    _frequency_points,
    scattering_event_class,
    num_points_in_batch,
    ise: ImagSelfEnergy,
    write_gamma_detail,
    return_gamma_detail,
    log_level,
):
    # Run one by one at frequency points
    if detailed_gamma_at_gp is None:
        detailed_gamma_at_gp_at_j = None
    else:
        detailed_gamma_at_gp_at_j = detailed_gamma_at_gp[j]

    if _frequency_points is None:
        ise.run_integration_weights(scattering_event_class=scattering_event_class)
        for k, t in enumerate(temperatures):
            ise.temperature = t
            ise.run()
            gamma[j, k, i] = ise.imag_self_energy
            if write_gamma_detail or return_gamma_detail:
                detailed_gamma_at_gp[k] = ise.detailed_imag_self_energy
    else:
        run_ise_at_frequency_points_batch(
            i,
            j,
            _frequency_points,
            ise,
            temperatures,
            gamma,
            write_gamma_detail=write_gamma_detail,
            return_gamma_detail=return_gamma_detail,
            detailed_gamma_at_gp=detailed_gamma_at_gp_at_j,
            scattering_event_class=scattering_event_class,
            nelems_in_batch=num_points_in_batch,
            log_level=log_level,
        )


def get_frequency_points(
    max_phonon_freq=None,
    sigmas=None,
    frequency_points=None,
    frequency_step=None,
    num_frequency_points=None,
):
    """Generate frequency points.

    This function may be mostly used for the phonon frequency axis of
    spectrum-like calculations.

    """
    if frequency_points is None:
        if sigmas is not None:
            sigma_vals = [sigma for sigma in sigmas if sigma is not None]
        else:
            sigma_vals = []
        if sigma_vals:
            fmax = max_phonon_freq * 2 + np.max(sigma_vals) * 4
        else:
            fmax = max_phonon_freq * 2
        fmax *= 1.005
        fmin = 0
        _frequency_points = _sample_frequency_points(
            fmin,
            fmax,
            frequency_step=frequency_step,
            num_frequency_points=num_frequency_points,
        )
    else:
        _frequency_points = np.array(frequency_points, dtype="double")

    return _frequency_points


def _sample_frequency_points(
    f_min, f_max, frequency_step=None, num_frequency_points=None
):
    if num_frequency_points is None:
        if frequency_step is not None:
            frequency_points = np.arange(f_min, f_max, frequency_step, dtype="double")
        else:
            frequency_points = np.array(np.linspace(f_min, f_max, 201), dtype="double")
    else:
        frequency_points = np.array(
            np.linspace(f_min, f_max, num_frequency_points), dtype="double"
        )

    return frequency_points


def write_imag_self_energy(
    imag_self_energy,
    mesh,
    grid_points,
    band_indices,
    frequency_points,
    temperatures,
    sigmas,
    scattering_event_class=None,
    output_filename=None,
    is_mesh_symmetry=True,
    log_level=0,
):
    """Write imaginary-part of self-energies into text files."""
    for sigma, ise_temps in zip(sigmas, imag_self_energy):
        for t, ise_gps in zip(temperatures, ise_temps):
            for gp, ise in zip(grid_points, ise_gps):
                for i, bi in enumerate(band_indices):
                    pos = 0
                    for j in range(i):
                        pos += len(band_indices[j])
                    filename = write_imag_self_energy_at_grid_point(
                        gp,
                        bi,
                        mesh,
                        frequency_points,
                        ise[pos : (pos + len(bi))].sum(axis=0) / len(bi),
                        sigma=sigma,
                        temperature=t,
                        scattering_event_class=scattering_event_class,
                        filename=output_filename,
                        is_mesh_symmetry=is_mesh_symmetry,
                    )
                    if log_level:
                        print(
                            "Imaginary part of self-energies were "
                            'written to "%s".' % filename
                        )


def average_by_degeneracy(imag_self_energy, band_indices, freqs_at_gp):
    """Take averages of values of energetically degenerated bands."""
    deg_sets = degenerate_sets(freqs_at_gp)
    imag_se = np.zeros_like(imag_self_energy)
    for dset in deg_sets:
        bi_set = []
        for i, bi in enumerate(band_indices):
            if bi in dset:
                bi_set.append(i)
        for i in bi_set:
            if imag_self_energy.ndim == 1:
                imag_se[i] = imag_self_energy[bi_set].sum() / len(bi_set)
            else:
                imag_se[:, i] = imag_self_energy[:, bi_set].sum(axis=1) / len(bi_set)
    return imag_se


def run_ise_at_frequency_points_batch(
    i,
    j,
    _frequency_points,
    ise: ImagSelfEnergy,
    temperatures,
    gamma,
    write_gamma_detail=False,
    return_gamma_detail=False,
    detailed_gamma_at_gp=None,
    scattering_event_class=None,
    nelems_in_batch=50,
    log_level=0,
):
    """Run calculations at frequency points batch by batch.

    See the details about batch in docstring of ``get_imag_self_energy``.

    """
    if nelems_in_batch is None:
        _nelems_in_batch = 10
    else:
        _nelems_in_batch = nelems_in_batch

    batches = get_freq_points_batches(len(_frequency_points), _nelems_in_batch)

    if log_level:
        print(
            "Calculations at %d frequency points are devided into "
            "%d batches." % (len(_frequency_points), len(batches))
        )

    for bi, fpts_batch in enumerate(batches):
        if log_level:
            print("%d/%d: %s" % (bi + 1, len(batches), fpts_batch + 1))
            sys.stdout.flush()

        ise.frequency_points = _frequency_points[fpts_batch]
        ise.run_integration_weights(scattering_event_class=scattering_event_class)
        for ll, t in enumerate(temperatures):
            ise.temperature = t
            ise.run()
            gamma[j, ll, i, :, fpts_batch] = ise.imag_self_energy
            if write_gamma_detail or return_gamma_detail:
                detailed_gamma_at_gp[ll, fpts_batch] = ise.detailed_imag_self_energy


def get_freq_points_batches(tot_nelems, nelems=None):
    """Divide frequency points into batches."""
    if nelems is None:
        _nelems = 10
    else:
        _nelems = nelems
    nbatch = tot_nelems // _nelems
    batches = [np.arange(i * _nelems, (i + 1) * _nelems) for i in range(nbatch)]
    if tot_nelems % _nelems > 0:
        batches.append(np.arange(_nelems * nbatch, tot_nelems))
    return batches
