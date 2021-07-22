"""Thermal conductivity base class."""
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

import warnings
import textwrap
import numpy as np
from phonopy.phonon.group_velocity import GroupVelocity
from phonopy.harmonic.force_constants import similarity_transformation
from phonopy.phonon.thermal_properties import mode_cv as get_mode_cv
from phonopy.units import THzToEv, EV, THz, Angstrom
from phono3py.file_IO import write_pp_to_hdf5
from phono3py.phonon3.triplets import get_all_triplets
from phono3py.other.isotope import Isotope
from phono3py.phonon.grid import (get_ir_grid_points,
                                  get_grid_points_by_rotations)

unit_to_WmK = ((THz * Angstrom) ** 2 / (Angstrom ** 3) * EV / THz /
               (2 * np.pi))  # 2pi comes from definition of lifetime.


def all_bands_exist(interaction):
    """Return if all bands are selected or not."""
    band_indices = interaction.band_indices
    num_band = len(interaction.primitive) * 3
    if len(band_indices) == num_band:
        if (band_indices - np.arange(num_band) == 0).all():
            return True
    return False


def write_pp(conductivity,
             pp,
             i,
             filename=None,
             compression="gzip"):
    """Write ph-ph interaction strength in hdf5 file."""
    grid_point = conductivity.grid_points[i]
    sigmas = conductivity.sigmas
    sigma_cutoff = conductivity.sigma_cutoff_width
    mesh = conductivity.mesh_numbers
    triplets, weights, _, _ = pp.get_triplets_at_q()
    all_triplets = get_all_triplets(grid_point, pp.bz_grid)

    if len(sigmas) > 1:
        print("Multiple smearing parameters were given. The last one in ")
        print("ph-ph interaction calculations was written in the file.")

    write_pp_to_hdf5(mesh,
                     pp=pp.interaction_strength,
                     g_zero=pp.zero_value_positions,
                     grid_point=grid_point,
                     triplet=triplets,
                     weight=weights,
                     triplet_all=all_triplets,
                     sigma=sigmas[-1],
                     sigma_cutoff=sigma_cutoff,
                     filename=filename,
                     compression=compression)


class Conductivity(object):
    """Thermal conductivity base class."""

    def __init__(self,
                 interaction,
                 grid_points=None,
                 temperatures=None,
                 sigmas=None,
                 sigma_cutoff=None,
                 is_isotope=False,
                 mass_variances=None,
                 boundary_mfp=None,  # in micrometre
                 is_kappa_star=True,
                 gv_delta_q=None,  # finite difference for group veolocity
                 is_full_pp=False,
                 log_level=0):
        """Init method."""
        if sigmas is None:
            self._sigmas = []
        else:
            self._sigmas = sigmas
        self._sigma_cutoff = sigma_cutoff
        self._pp = interaction
        self._is_full_pp = is_full_pp
        self._collision = None  # has to be set derived class
        if temperatures is None:
            self._temperatures = None
        else:
            self._temperatures = np.array(temperatures, dtype='double')
        self._is_kappa_star = is_kappa_star
        self._gv_delta_q = gv_delta_q
        self._log_level = log_level
        self._primitive = self._pp.primitive
        self._dm = self._pp.dynamical_matrix
        self._frequency_factor_to_THz = self._pp.frequency_factor_to_THz
        self._cutoff_frequency = self._pp.cutoff_frequency
        self._boundary_mfp = boundary_mfp

        if not self._is_kappa_star:
            self._point_operations = np.array([np.eye(3, dtype='int_')],
                                              dtype='int_', order='C')
        else:
            self._point_operations = np.array(
                self._pp.bz_grid.reciprocal_operations,
                dtype='int_', order='C')
        rec_lat = np.linalg.inv(self._primitive.cell)
        self._rotations_cartesian = np.array(
            [similarity_transformation(rec_lat, r)
             for r in self._point_operations], dtype='double', order='C')

        self._grid_points = None
        self._grid_weights = None
        self._ir_grid_points = None
        self._ir_grid_weights = None
        self._bz_grid = self._pp.bz_grid

        self._read_gamma = False
        self._read_gamma_iso = False

        self._kappa = None
        self._mode_kappa = None

        self._frequencies = None
        self._cv = None
        self._gv = None
        self._gv_sum2 = None
        self._gamma = None
        self._gamma_iso = None
        self._num_sampling_grid_points = 0

        volume = self._primitive.volume
        self._conversion_factor = unit_to_WmK / volume

        self._isotope = None
        self._mass_variances = None
        self._is_isotope = is_isotope
        if mass_variances is not None:
            self._is_isotope = True
        if self._is_isotope:
            self._set_isotope(mass_variances)

        self._grid_point_count = None
        self._set_grid_properties(grid_points)

        if (self._dm.is_nac() and
            self._dm.nac_method == 'gonze' and
            self._gv_delta_q is None):  # noqa E129
            self._gv_delta_q = 1e-5
            if self._log_level:
                msg = "Group velocity calculation:\n"
                text = ("Analytical derivative of dynamical matrix is not "
                        "implemented for NAC by Gonze et al. Instead "
                        "numerical derivative of it is used with dq=1e-5 "
                        "for group velocity calculation.")
                msg += textwrap.fill(text,
                                     initial_indent="  ",
                                     subsequent_indent="  ",
                                     width=70)
                print(msg)
        self._gv_obj = GroupVelocity(
            self._dm,
            q_length=self._gv_delta_q,
            symmetry=self._pp.primitive_symmetry,
            frequency_factor_to_THz=self._frequency_factor_to_THz)
        # gv_delta_q may be changed.
        self._gv_delta_q = self._gv_obj.get_q_length()

    def __iter__(self):
        """Calculate mode kappa at each grid point."""
        return self

    def __next__(self):
        """Return grid point count for mode kappa."""
        if self._grid_point_count == len(self._grid_points):
            if self._log_level:
                print("=================== End of collection of collisions "
                      "===================")
            raise StopIteration
        else:
            self._run_at_grid_point()
            self._grid_point_count += 1
            return self._grid_point_count - 1

    def next(self):
        """For backward compatibility."""
        return self.__next__()

    @property
    def mesh_numbers(self):
        """Return mesh numbers of GR-grid."""
        return self._pp.mesh_numbers

    def get_mesh_numbers(self):
        """Return mesh numbers of GR-grid."""
        warnings.warn("Use attribute, Conductivity.mesh_numbers "
                      "instead of Conductivity.get_mesh_numbers().",
                      DeprecationWarning)
        return self.mesh_numbers

    @property
    def mode_heat_capacities(self):
        """Return mode heat capacity at constant volume at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._cv

    def get_mode_heat_capacities(self):
        """Return mode heat capacity at constant volume at grid points.

        Grid points are those at mode kappa are calculated.

        """
        warnings.warn("Use attribute, Conductivity.mode_heat_capacities "
                      "instead of Conductivity.get_mode_heat_capacities().",
                      DeprecationWarning)
        return self.mode_heat_capacities

    @property
    def group_velocities(self):
        """Return group velocities at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._gv

    def get_group_velocities(self):
        """Return group velocities at grid points.

        Grid points are those at mode kappa are calculated.

        """
        warnings.warn("Use attribute, Conductivity.group_velocities "
                      "instead of Conductivity.get_group_velocities().",
                      DeprecationWarning)
        return self.group_velocities

    @property
    def gv_by_gv(self):
        """Return gv_by_gv at grid points where mode kappa are calculated."""
        return self._gv_sum2

    def get_gv_by_gv(self):
        """Return gv_by_gv at grid points where mode kappa are calculated."""
        warnings.warn("Use attribute, Conductivity.gv_by_gv "
                      "instead of Conductivity.get_gv_by_gv().",
                      DeprecationWarning)
        return self.gv_by_gv

    @property
    def frequencies(self):
        """Return frequencies at grid points.

        Grid points are those at mode kappa are calculated.

        """
        return self._frequencies[self._grid_points]

    def get_frequencies(self):
        """Return frequencies at grid points.

        Grid points are those at mode kappa are calculated.

        """
        warnings.warn("Use attribute, Conductivity.frequencies "
                      "instead of Conductivity.get_frequencies().",
                      DeprecationWarning)
        return self.frequencies

    @property
    def qpoints(self):
        """Return q-points where mode kappa are calculated."""
        return self._qpoints

    def get_qpoints(self):
        """Return q-points where mode kappa are calculated."""
        warnings.warn("Use attribute, Conductivity.qpoints "
                      "instead of Conductivity.get_qpoints().",
                      DeprecationWarning)
        return self.qpoints

    @property
    def grid_points(self):
        """Return grid point indices where mode kappa are calculated.

        Grid point indices are given in BZ-grid.

        """
        return self._grid_points

    def get_grid_points(self):
        """Return grid point indices where mode kappa are calculated.

        Grid point indices are given in BZ-grid.

        """
        warnings.warn("Use attribute, Conductivity.grid_points "
                      "instead of Conductivity.get_grid_points().",
                      DeprecationWarning)
        return self.grid_points

    @property
    def grid_weights(self):
        """Return grid point weights where mode kappa are calculated."""
        return self._grid_weights

    def get_grid_weights(self):
        """Return grid point weights where mode kappa are calculated."""
        warnings.warn("Use attribute, Conductivity.grid_weights "
                      "instead of Conductivity.get_grid_weights().",
                      DeprecationWarning)
        return self.grid_weights

    @property
    def temperatures(self):
        """Setter and getter of temperatures."""
        return self._temperatures

    @temperatures.setter
    def temperatures(self, temperatures):
        self._temperatures = temperatures
        self._allocate_values()

    def get_temperatures(self):
        """Return temperatures."""
        warnings.warn("Use attribute, Conductivity.temperatures "
                      "instead of Conductivity.get_temperatures().",
                      DeprecationWarning)
        return self.temperatures

    def set_temperatures(self, temperatures):
        """Set temperatures."""
        warnings.warn("Use attribute, Conductivity.temperatures "
                      "instead of Conductivity.set_temperatures().",
                      DeprecationWarning)
        self.temperatures = temperatures

    @property
    def gamma(self):
        """Setter and getter of gamma."""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma
        self._read_gamma = True

    def get_gamma(self):
        """Return gamma."""
        warnings.warn("Use attribute, Conductivity.gamma "
                      "instead of Conductivity.get_gamma().",
                      DeprecationWarning)
        return self.gamma

    def set_gamma(self, gamma):
        """Set gamma."""
        warnings.warn("Use attribute, Conductivity.gamma "
                      "instead of Conductivity.set_gamma().",
                      DeprecationWarning)
        self.gamma = gamma

    @property
    def gamma_isotope(self):
        """Setter and getter of gamma from isotope."""
        return self._gamma_iso

    @gamma_isotope.setter
    def gamma_isotope(self, gamma_iso):
        self._gamma_iso = gamma_iso
        self._read_gamma_iso = True

    def get_gamma_isotope(self):
        """Return gamma from isotope."""
        warnings.warn("Use attribute, Conductivity.gamma_isotope "
                      "instead of Conductivity.get_gamma_isotope().",
                      DeprecationWarning)
        return self.gamma_isotope

    def set_gamma_isotope(self, gamma_iso):
        """Set gamma from isotope."""
        warnings.warn("Use attribute, Conductivity.gamma_isotope "
                      "instead of Conductivity.set_gamma_isotope().",
                      DeprecationWarning)
        self.gamma_isotope = gamma_iso

    @property
    def kappa(self):
        """Return kappa."""
        return self._kappa

    def get_kappa(self):
        """Return kappa."""
        warnings.warn("Use attribute, Conductivity.kappa "
                      "instead of Conductivity.get_kappa().",
                      DeprecationWarning)
        return self.kappa

    @property
    def mode_kappa(self):
        """Return mode_kappa."""
        return self._mode_kappa

    def get_mode_kappa(self):
        """Return mode_kappa."""
        warnings.warn("Use attribute, Conductivity.mode_kappa "
                      "instead of Conductivity.get_mode_kappa().",
                      DeprecationWarning)
        return self.mode_kappa

    @property
    def sigmas(self):
        """Return sigmas."""
        return self._sigmas

    def get_sigmas(self):
        """Return sigmas."""
        warnings.warn("Use attribute, Conductivity.sigmas "
                      "instead of Conductivity.get_sigmas().",
                      DeprecationWarning)
        return self.sigmas

    @property
    def sigma_cutoff_width(self):
        """Return smearing width cutoff."""
        return self._sigma_cutoff

    def get_sigma_cutoff_width(self):
        """Return smearing width cutoff."""
        warnings.warn("Use attribute, Conductivity.sigma_cutoff_width "
                      "instead of Conductivity.get_sigma_cutoff_width().",
                      DeprecationWarning)
        return self.sigma_cutoff_width

    @property
    def grid_point_count(self):
        """Return interator count of self."""
        return self._grid_point_count

    def get_grid_point_count(self):
        """Return interator count of self."""
        warnings.warn("Use attribute, Conductivity.grid_point_count "
                      "instead of Conductivity.get_grid_point_count().",
                      DeprecationWarning)
        return self.grid_point_count

    @property
    def averaged_pp_interaction(self):
        """Return averaged pp strength."""
        return self._averaged_pp_interaction

    def get_averaged_pp_interaction(self):
        """Return averaged pp interaction strength."""
        warnings.warn("Use attribute, Conductivity.averaged_pp_interaction "
                      "instead of Conductivity.get_averaged_pp_interaction().",
                      DeprecationWarning)
        return self.averaged_pp_interaction

    def get_number_of_sampling_grid_points(self):
        """Return number of grid points.

        This is calculated by the sum of numbers of arms of k-start in
        `Conductivity._set_gv_by_gv`.

        """
        return self._num_sampling_grid_points

    def _run_at_grid_point(self):
        """Must be implementated in the inherited class."""
        raise NotImplementedError()

    def _allocate_values(self):
        """Must be implementated in the inherited class."""
        raise NotImplementedError()

    def _set_grid_properties(self, grid_points):
        self._pp.set_nac_q_direction(nac_q_direction=None)

        if grid_points is not None:  # Specify grid points
            self._grid_points = grid_points
            (self._ir_grid_points,
             self._ir_grid_weights) = self._get_ir_grid_points()
        elif not self._is_kappa_star:  # All grid points
            self._grid_points = self._bz_grid.grg2bzg
            self._grid_weights = np.ones(len(self._grid_points), dtype='int_')
            self._ir_grid_points = self._grid_points
            self._ir_grid_weights = self._grid_weights
        else:  # Automatic sampling
            self._grid_points, self._grid_weights = self._get_ir_grid_points()
            self._ir_grid_points = self._grid_points
            self._ir_grid_weights = self._grid_weights

        self._qpoints = np.array(
            np.dot(self._bz_grid.addresses[self._grid_points],
                   self._bz_grid.QDinv.T), dtype='double', order='C')
        self._grid_point_count = 0
        (self._frequencies,
         self._eigenvectors,
         phonon_done) = self._pp.get_phonons()
        if (phonon_done == 0).any():
            self._pp.run_phonon_solver()

    def _get_gamma_isotope_at_sigmas(self, i):
        gamma_iso = []
        pp_freqs, pp_eigvecs, pp_phonon_done = self._pp.get_phonons()

        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "Calculating Gamma of ph-isotope with "
                if sigma is None:
                    text += "tetrahedron method"
                else:
                    text += "sigma=%s" % sigma
                print(text)

            self._isotope.sigma = sigma
            self._isotope.set_phonons(pp_freqs,
                                      pp_eigvecs,
                                      pp_phonon_done,
                                      dm=self._dm)
            gp = self._grid_points[i]
            self._isotope.set_grid_point(gp)
            self._isotope.run()
            gamma_iso.append(self._isotope.gamma)

        return np.array(gamma_iso, dtype='double', order='C')

    def _get_ir_grid_points(self):
        """Find irreducible grid points."""
        ir_grid_points, ir_grid_weights, _ = get_ir_grid_points(self._bz_grid)
        ir_grid_points = np.array(
            self._bz_grid.grg2bzg[ir_grid_points], dtype='int_')

        return ir_grid_points, ir_grid_weights

    def _set_isotope(self, mass_variances):
        if mass_variances is True:
            mv = None
        else:
            mv = mass_variances
        self._isotope = Isotope(
            self._pp.mesh_numbers,
            self._primitive,
            mass_variances=mv,
            bz_grid=self._bz_grid,
            frequency_factor_to_THz=self._frequency_factor_to_THz,
            symprec=self._pp.primitive_symmetry.tolerance,
            cutoff_frequency=self._cutoff_frequency,
            lapack_zheev_uplo=self._pp.lapack_zheev_uplo)
        self._mass_variances = self._isotope.mass_variances

    def _set_harmonic_properties(self, i_irgp, i_data):
        """Set group velocity and mode heat capacity."""
        grid_point = self._grid_points[i_irgp]
        freqs = self._frequencies[grid_point][self._pp.band_indices]
        self._cv[:, i_data, :] = self._get_cv(freqs)
        self._gv_obj.run([self._qpoints[i_irgp], ])
        gv = self._gv_obj.get_group_velocity()[0, self._pp.band_indices, :]
        self._gv[i_data] = gv

    def _get_cv(self, freqs):
        cv = np.zeros((len(self._temperatures), len(freqs)), dtype='double')
        # T/freq has to be large enough to avoid divergence.
        # Otherwise just set 0.
        for i, f in enumerate(freqs):
            finite_t = (self._temperatures > f / 100)
            if f > self._cutoff_frequency:
                cv[:, i] = np.where(
                    finite_t, get_mode_cv(
                        np.where(finite_t, self._temperatures, 10000),
                        f * THzToEv), 0)
        return cv

    def _set_gv_by_gv(self, i_irgp, i_data):
        """Outer product of group velocities.

        (v x v) [num_k*, num_freqs, 3, 3]

        """
        gv_by_gv_tensor, order_kstar = self._get_gv_by_gv(i_irgp, i_data)
        self._num_sampling_grid_points += order_kstar

        # Sum all vxv at k*
        for j, vxv in enumerate(
                ([0, 0], [1, 1], [2, 2], [1, 2], [0, 2], [0, 1])):
            self._gv_sum2[i_data, :, j] = gv_by_gv_tensor[:, vxv[0], vxv[1]]

    def _get_gv_by_gv(self, i_irgp, i_data):
        if self._is_kappa_star:
            rotation_map = get_grid_points_by_rotations(
                self._grid_points[i_irgp],
                self._bz_grid)
        else:
            rotation_map = get_grid_points_by_rotations(
                self._grid_points[i_irgp],
                self._bz_grid,
                reciprocal_rotations=self._point_operations)

        gv = self._gv[i_data]
        gv_by_gv = np.zeros((len(gv), 3, 3), dtype='double')

        for r in self._rotations_cartesian:
            gvs_rot = np.dot(gv, r.T)
            gv_by_gv += [np.outer(r_gv, r_gv) for r_gv in gvs_rot]
        gv_by_gv /= len(rotation_map) // len(np.unique(rotation_map))
        order_kstar = len(np.unique(rotation_map))

        if self._grid_weights is not None:
            if order_kstar != self._grid_weights[i_irgp]:
                if self._log_level:
                    text = ("Number of elements in k* is unequal "
                            "to number of equivalent grid-points. "
                            "This means that the mesh sampling grids break "
                            "symmetry. Please check carefully "
                            "the convergence over grid point densities.")
                    msg = textwrap.fill(text,
                                        initial_indent=" ",
                                        subsequent_indent=" ",
                                        width=70)
                    print("*" * 30 + "Warning" + "*" * 30)
                    print(msg)
                    print("*" * 67)

        return gv_by_gv, order_kstar

    def _get_main_diagonal(self, i, j, k):
        main_diagonal = self._gamma[j, k, i].copy()
        if self._gamma_iso is not None:
            main_diagonal += self._gamma_iso[j, i]
        if self._boundary_mfp is not None:
            main_diagonal += self._get_boundary_scattering(i)

        # num_band = len(self._primitive) * 3
        # if self._boundary_mfp is not None:
        #     for l in range(num_band):
        #         # Acoustic modes at Gamma are avoided.
        #         if i == 0 and l < 3:
        #             continue
        #         gv_norm = np.linalg.norm(self._gv[i, l])
        #         mean_free_path = (gv_norm * Angstrom * 1e6 /
        #                           (4 * np.pi * main_diagonal[l]))
        #         if mean_free_path > self._boundary_mfp:
        #             main_diagonal[l] = (
        #                 gv_norm / (4 * np.pi * self._boundary_mfp))

        return main_diagonal

    def _get_boundary_scattering(self, i):
        num_band = len(self._primitive) * 3
        g_boundary = np.zeros(num_band, dtype='double')
        for ll in range(num_band):
            g_boundary[ll] = (np.linalg.norm(self._gv[i, ll]) * Angstrom * 1e6
                              / (4 * np.pi * self._boundary_mfp))
        return g_boundary

    def _show_log_header(self, i):
        if self._log_level:
            gp = self._grid_points[i]
            print("======================= Grid point %d (%d/%d) "
                  "=======================" %
                  (gp, i + 1, len(self._grid_points)))
            print("q-point: (%5.2f %5.2f %5.2f)" % tuple(self._qpoints[i]))
            if self._boundary_mfp is not None:
                if self._boundary_mfp > 1000:
                    print("Boundary mean free path (millimetre): %.3f" %
                          (self._boundary_mfp / 1000.0))
                else:
                    print("Boundary mean free path (micrometre): %.5f" %
                          self._boundary_mfp)
            if self._is_isotope:
                print(("Mass variance parameters: " +
                       "%5.2e " * len(self._mass_variances)) %
                      tuple(self._mass_variances))
