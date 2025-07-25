"""Lattice thermal conductivity calculation base class with RTA."""

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

from __future__ import annotations

import warnings
from abc import abstractmethod

import numpy as np
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.base import ConductivityBase
from phono3py.file_IO import read_pp_from_hdf5
from phono3py.other.tetrahedron_method import get_tetrahedra_relative_grid_address
from phono3py.phonon.grid import (
    get_grid_points_by_rotations,
    get_qpoints_from_bz_grid_points,
)
from phono3py.phonon3.imag_self_energy import ImagSelfEnergy, average_by_degeneracy
from phono3py.phonon3.interaction import Interaction


class ConductivityRTABase(ConductivityBase):
    """Base class of ConductivityRTA*.

    This is a base class of RTA classes.

    """

    def __init__(
        self,
        interaction: Interaction,
        grid_points: np.ndarray | None = None,
        temperatures: list | np.ndarray | None = None,
        sigmas: list | np.ndarray | None = None,
        sigma_cutoff: float | None = None,
        is_isotope: bool = False,
        mass_variances: list | np.ndarray | None = None,
        boundary_mfp: float | None = None,  # in micrometer
        use_ave_pp: bool = False,
        is_kappa_star: bool = True,
        is_full_pp: bool = False,
        read_pp: bool = False,
        store_pp: bool = False,
        pp_filename: float | None = None,
        is_N_U: bool = False,
        is_gamma_detail: bool = False,
        is_frequency_shift_by_bubble: bool = False,
        log_level: int = 0,
    ):
        """Init method."""
        self._is_N_U = is_N_U
        self._is_gamma_detail = is_gamma_detail
        self._is_frequency_shift_by_bubble = is_frequency_shift_by_bubble

        self._gamma_N = None
        self._gamma_U = None
        self._gamma_detail_at_q = None
        self._use_ave_pp = use_ave_pp
        self._use_const_ave_pp = None
        self._num_ignored_phonon_modes = None

        super().__init__(
            interaction,
            grid_points=grid_points,
            temperatures=temperatures,
            sigmas=sigmas,
            sigma_cutoff=sigma_cutoff,
            is_isotope=is_isotope,
            mass_variances=mass_variances,
            boundary_mfp=boundary_mfp,
            is_kappa_star=is_kappa_star,
            is_full_pp=is_full_pp,
            log_level=log_level,
        )

        self._use_const_ave_pp = self._pp.constant_averaged_interaction
        self._read_pp = read_pp
        self._store_pp = store_pp
        self._pp_filename = pp_filename

        if self._temperatures is not None:
            self._allocate_values()

        self._collision = ImagSelfEnergy(
            self._pp, with_detail=(self._is_gamma_detail or self._is_N_U)
        )

    def get_gamma_N_U(self):
        """Return N and U parts of gamma."""
        return (self._gamma_N, self._gamma_U)

    def set_gamma_N_U(self, gamma_N, gamma_U):
        """Set N and U parts of gamma."""
        self._gamma_N = gamma_N
        self._gamma_U = gamma_U

    def get_gamma_detail_at_q(self):
        """Return contribution of each triplet to gamma at current q-point."""
        return self._gamma_detail_at_q

    def get_number_of_ignored_phonon_modes(self):
        """Return number of ignored phonon modes."""
        warnings.warn(
            "Use attribute, number_of_ignored_phonon_modes",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.number_of_ignored_phonon_modes

    @property
    def number_of_ignored_phonon_modes(self):
        """Return number of ignored phonon modes."""
        return self._num_ignored_phonon_modes

    def set_averaged_pp_interaction(self, ave_pp):
        """Set averaged ph-ph interaction."""
        self._averaged_pp_interaction = ave_pp

    @abstractmethod
    def set_kappa_at_sigmas(self):
        """Must be implemented in the inherited class."""
        raise NotImplementedError()

    def _allocate_values(self):
        if self._temperatures is None:
            raise RuntimeError(
                "Temperatures have not been set yet. "
                "Set temperatures before this method."
            )

        num_band0 = len(self._pp.band_indices)
        num_grid_points = len(self._grid_points)
        num_temp = len(self._temperatures)
        if not self._read_gamma:
            self._gamma = np.zeros(
                (len(self._sigmas), num_temp, num_grid_points, num_band0),
                order="C",
                dtype="double",
            )
            if self._is_gamma_detail or self._is_N_U:
                self._gamma_N = np.zeros_like(self._gamma)
                self._gamma_U = np.zeros_like(self._gamma)

        if self._is_isotope:
            self._gamma_iso = np.zeros(
                (len(self._sigmas), num_grid_points, num_band0),
                order="C",
                dtype="double",
            )
        if self._is_full_pp or self._use_ave_pp or self._use_const_ave_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_grid_points, num_band0), order="C", dtype="double"
            )
        self._num_ignored_phonon_modes = np.zeros(
            (len(self._sigmas), num_temp), order="C", dtype="intc"
        )

    def _run_at_grid_point(self):
        i_gp = self._grid_point_count
        self._show_log_header(i_gp)
        grid_point = self._grid_points[i_gp]
        self._set_cv(i_gp, i_gp)
        self._set_velocities(i_gp, i_gp)

        if self._read_gamma:
            if self._use_ave_pp:
                self._collision.set_grid_point(grid_point)
                self._set_gamma_at_sigmas(i_gp)
        else:
            self._collision.set_grid_point(grid_point)
            num_triplets = len(self._pp.get_triplets_at_q()[0])
            if self._log_level:
                print("Number of triplets: %d" % num_triplets, flush=True)

            if (
                self._is_full_pp
                or self._read_pp
                or self._store_pp
                or self._use_ave_pp
                or self._use_const_ave_pp
                or self._is_gamma_detail
            ):
                self._set_gamma_at_sigmas(i_gp)
            else:  # can save memory space
                self._set_gamma_at_sigmas_lowmem(i_gp)

        if self._is_isotope and not self._read_gamma_iso:
            gamma_iso = self._get_gamma_isotope_at_sigmas(i_gp)
            self._gamma_iso[:, i_gp, :] = gamma_iso[:, self._pp.band_indices]

        if self._log_level:
            self._show_log(i_gp)

    def _set_gamma_at_sigmas(self, i):
        for j, sigma in enumerate(self._sigmas):
            self._collision.set_sigma(sigma, sigma_cutoff=self._sigma_cutoff)
            self._collision.run_integration_weights()

            if self._log_level:
                text = "Collisions will be calculated with "
                if sigma is None:
                    text += "tetrahedron method."
                else:
                    text += "sigma=%s" % sigma
                    if self._sigma_cutoff is None:
                        text += "."
                    else:
                        text += "(%4.2f SD)." % self._sigma_cutoff
                print(text)

            if self._read_pp:
                pp, _g_zero = read_pp_from_hdf5(
                    self._pp.mesh_numbers,
                    grid_point=self._grid_points[i],
                    sigma=sigma,
                    sigma_cutoff=self._sigma_cutoff,
                    filename=self._pp_filename,
                    verbose=(self._log_level > 0),
                )
                _, g_zero = self._collision.get_integration_weights()
                if self._log_level:
                    if len(self._sigmas) > 1:
                        print(
                            "Multiple sigmas or mixing smearing and "
                            "tetrahedron method is not supported."
                        )
                if _g_zero is not None and (_g_zero != g_zero).any():
                    raise ValueError("Inconsistency found in g_zero.")
                self._collision.set_interaction_strength(pp)
            elif self._use_ave_pp:
                self._collision.set_averaged_pp_interaction(
                    self._averaged_pp_interaction[i]
                )
            elif self._use_const_ave_pp:
                if self._log_level:
                    print(
                        "Constant ph-ph interaction of %6.3e is used."
                        % self._pp.constant_averaged_interaction
                    )
                self._collision.run_interaction()
                self._averaged_pp_interaction[i] = self._pp.averaged_interaction
            elif j != 0 and (self._is_full_pp or self._sigma_cutoff is None):
                if self._log_level:
                    print("Existing ph-ph interaction is used.")
            else:
                if self._log_level:
                    print("Calculating ph-ph interaction...")
                self._collision.run_interaction(is_full_pp=self._is_full_pp)
                if self._is_full_pp:
                    self._averaged_pp_interaction[i] = self._pp.averaged_interaction

            # Number of triplets depends on q-point.
            # So this is allocated each time.
            if self._is_gamma_detail:
                num_temp = len(self._temperatures)
                self._gamma_detail_at_q = np.empty(
                    ((num_temp,) + self._pp.interaction_strength.shape),
                    dtype="double",
                    order="C",
                )
                self._gamma_detail_at_q[:] = 0

            if self._log_level:
                print("Calculating collisions at temperatures...")
            for k, t in enumerate(self._temperatures):
                self._collision.temperature = t
                self._collision.run()
                self._gamma[j, k, i] = self._collision.imag_self_energy
                if self._is_N_U or self._is_gamma_detail:
                    g_N, g_U = self._collision.get_imag_self_energy_N_and_U()
                    self._gamma_N[j, k, i] = g_N
                    self._gamma_U[j, k, i] = g_U
                if self._is_gamma_detail:
                    self._gamma_detail_at_q[k] = (
                        self._collision.get_detailed_imag_self_energy()
                    )

    def _set_gamma_at_sigmas_lowmem(self, i):
        """Calculate gamma without storing ph-ph interaction strength.

        `svecs` and `multi` below must not be simply replaced by
        `self._pp.primitive.get_smallest_vectors()` because they must be in
        dense format as always so in Interaction class instance.
        `p2s`, `s2p`, and `masses` have to be also given from Interaction
        class instance.

        """
        num_band = len(self._pp.primitive) * 3
        band_indices = self._pp.band_indices
        (
            svecs,
            multi,
            p2s,
            s2p,
            masses,
        ) = self._pp.get_primitive_and_supercell_correspondence()
        triplets_at_q, weights_at_q, _, _ = self._pp.get_triplets_at_q()

        if None in self._sigmas:
            tetrahedra = get_tetrahedra_relative_grid_address(
                self._pp.bz_grid.microzone_lattice
            )

        # It is assumed that self._sigmas = [None].
        temperatures_THz = np.array(
            self._temperatures * get_physical_units().KB / get_physical_units().THzToEv,
            dtype="double",
        )
        for j, sigma in enumerate(self._sigmas):
            self._collision.set_sigma(sigma)
            if self._is_N_U:
                collisions = np.zeros(
                    (2, len(self._temperatures), len(band_indices)),
                    dtype="double",
                    order="C",
                )
            else:
                collisions = np.zeros(
                    (len(self._temperatures), len(band_indices)),
                    dtype="double",
                    order="C",
                )
            import phono3py._phono3py as phono3c

            # True: OpenMP over triplets
            # False: OpenMP over bands
            if self._pp.openmp_per_triplets is None:
                if len(triplets_at_q) > num_band:
                    openmp_per_triplets = True
                else:
                    openmp_per_triplets = False
            else:
                openmp_per_triplets = self._pp.openmp_per_triplets

            if sigma is None:
                phono3c.pp_collision(
                    collisions,
                    np.array(
                        np.dot(tetrahedra, self._pp.bz_grid.P.T),
                        dtype="int64",
                        order="C",
                    ),
                    self._frequencies,
                    self._eigenvectors,
                    triplets_at_q,
                    weights_at_q,
                    self._pp.bz_grid.addresses,
                    self._pp.bz_grid.gp_map,
                    self._pp.bz_grid.store_dense_gp_map * 1 + 1,
                    self._pp.bz_grid.D_diag,
                    self._pp.bz_grid.Q,
                    self._pp.fc3,
                    self._pp.fc3_nonzero_indices,
                    svecs,
                    multi,
                    masses,
                    p2s,
                    s2p,
                    band_indices,
                    temperatures_THz,
                    self._is_N_U * 1,
                    self._pp.symmetrize_fc3q * 1,
                    self._pp.make_r0_average * 1,
                    self._pp.all_shortest,
                    self._pp.cutoff_frequency,
                    openmp_per_triplets * 1,
                )
            else:
                if self._sigma_cutoff is None:
                    sigma_cutoff = -1
                else:
                    sigma_cutoff = float(self._sigma_cutoff)
                phono3c.pp_collision_with_sigma(
                    collisions,
                    sigma,
                    sigma_cutoff,
                    self._frequencies,
                    self._eigenvectors,
                    triplets_at_q,
                    weights_at_q,
                    self._pp.bz_grid.addresses,
                    self._pp.bz_grid.D_diag,
                    self._pp.bz_grid.Q,
                    self._pp.fc3,
                    self._pp.fc3_nonzero_indices,
                    svecs,
                    multi,
                    masses,
                    p2s,
                    s2p,
                    band_indices,
                    temperatures_THz,
                    self._is_N_U * 1,
                    self._pp.symmetrize_fc3q * 1,
                    self._pp.make_r0_average * 1,
                    self._pp.all_shortest,
                    self._pp.cutoff_frequency,
                    openmp_per_triplets * 1,
                )
            col_unit_conv = self._collision.unit_conversion_factor
            pp_unit_conv = self._pp.unit_conversion_factor
            if self._is_N_U:
                col = collisions.sum(axis=0)
                col_N = collisions[0]
                col_U = collisions[1]
            else:
                col = collisions
            for k in range(len(self._temperatures)):
                self._gamma[j, k, i, :] = average_by_degeneracy(
                    col[k] * col_unit_conv * pp_unit_conv,
                    band_indices,
                    self._frequencies[self._grid_points[i]],
                )
                if self._is_N_U:
                    self._gamma_N[j, k, i, :] = average_by_degeneracy(
                        col_N[k] * col_unit_conv * pp_unit_conv,
                        band_indices,
                        self._frequencies[self._grid_points[i]],
                    )
                    self._gamma_U[j, k, i, :] = average_by_degeneracy(
                        col_U[k] * col_unit_conv * pp_unit_conv,
                        band_indices,
                        self._frequencies[self._grid_points[i]],
                    )

    def _show_log(self, i_gp):
        q = get_qpoints_from_bz_grid_points(i_gp, self._pp.bz_grid)
        gp = self._grid_points[i_gp]
        frequencies = self._frequencies[gp][self._pp.band_indices]
        gv = self._conductivity_components.group_velocities[i_gp]

        if self._averaged_pp_interaction is not None:
            ave_pp = self._averaged_pp_interaction[i_gp]
        else:
            ave_pp = None
        self._show_log_value_names()

        if self._log_level > 2:
            self._show_log_values_on_kstar(frequencies, gv, ave_pp, gp, q)
        else:
            self._show_log_values(frequencies, gv, ave_pp)

        print("", end="", flush=True)

    def _show_log_values(self, frequencies, gv, ave_pp):
        if self._is_full_pp or self._use_ave_pp or self._use_const_ave_pp:
            for f, v, pp in zip(frequencies, gv, ave_pp):
                print(
                    "%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e"
                    % (f, v[0], v[1], v[2], np.linalg.norm(v), pp)
                )
        else:
            for f, v in zip(frequencies, gv):
                print(
                    "%8.3f   (%8.3f %8.3f %8.3f) %8.3f"
                    % (f, v[0], v[1], v[2], np.linalg.norm(v))
                )

    def _show_log_values_on_kstar(self, frequencies, gv, ave_pp, gp, q):
        rotation_map = get_grid_points_by_rotations(gp, self._pp.bz_grid)
        for i, j in enumerate(np.unique(rotation_map)):
            for k, (rot, rot_c) in enumerate(
                zip(self._point_operations, self._rotations_cartesian)
            ):
                if rotation_map[k] != j:
                    continue

                print(
                    " k*%-2d (%5.2f %5.2f %5.2f)" % ((i + 1,) + tuple(np.dot(rot, q)))
                )
                if self._is_full_pp or self._use_ave_pp or self._use_const_ave_pp:
                    for f, v, pp in zip(frequencies, np.dot(rot_c, gv.T).T, ave_pp):
                        print(
                            "%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e"
                            % (f, v[0], v[1], v[2], np.linalg.norm(v), pp)
                        )
                else:
                    for f, v in zip(frequencies, np.dot(rot_c, gv.T).T):
                        print(
                            "%8.3f   (%8.3f %8.3f %8.3f) %8.3f"
                            % (f, v[0], v[1], v[2], np.linalg.norm(v))
                        )
        print("")

    def _show_log_value_names(self):
        if self._is_full_pp or self._use_ave_pp or self._use_const_ave_pp:
            text = "Frequency     group velocity (x, y, z)     |gv|       Pqj"
        else:
            text = "Frequency     group velocity (x, y, z)     |gv|"
        if self._conductivity_components.gv_delta_q is None:
            pass
        else:
            text += "  (dq=%3.1e)" % self._conductivity_components.gv_delta_q
        print(text)
