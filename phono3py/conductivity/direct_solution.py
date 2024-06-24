"""Calculate lattice thermal conductivity by direct solution."""

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
import time
import warnings
from abc import abstractmethod
from typing import Type, Union

import numpy as np
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.units import Kb, THzToEv

from phono3py.conductivity.base import ConductivityBase, ConductivityMixIn
from phono3py.conductivity.utils import (
    ConductivityLBTEWriter,
    select_colmat_solver,
    set_collision_from_file,
)
from phono3py.conductivity.utils import write_pp as write_phph
from phono3py.conductivity.wigner import (
    ConductivityWignerMixIn,
    get_conversion_factor_WTE,
)
from phono3py.file_IO import read_pp_from_hdf5
from phono3py.phonon.grid import get_grid_points_by_rotations
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.interaction import Interaction, all_bands_exist


class ConductivityLBTEBase(ConductivityBase):
    """Base class of ConductivityLBTE*.

    This is a base class for direct-solution classes.

    """

    def __init__(
        self,
        interaction: Interaction,
        grid_points=None,
        temperatures=None,
        sigmas=None,
        sigma_cutoff=None,
        is_isotope=False,
        mass_variances=None,
        boundary_mfp=None,  # in micrometre
        solve_collective_phonon=False,
        is_reducible_collision_matrix=False,
        is_kappa_star=True,
        gv_delta_q=None,  # finite difference for group veolocity
        is_full_pp=False,
        read_pp=False,
        pp_filename=None,
        pinv_cutoff=1.0e-8,
        pinv_solver=0,
        pinv_method=0,
        log_level=0,
        lang="C",
    ):
        """Init method."""
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

        self._init_velocity(gv_delta_q)

        self._lang = lang
        self._collision_eigenvalues = None
        self._is_reducible_collision_matrix = is_reducible_collision_matrix
        self._solve_collective_phonon = solve_collective_phonon
        # if not self._is_kappa_star:
        #     self._is_reducible_collision_matrix = True
        self._collision_matrix = None
        self._read_pp = read_pp
        self._pp_filename = pp_filename
        self._pinv_cutoff = pinv_cutoff
        self._pinv_method = pinv_method
        self._pinv_solver = pinv_solver

        self._cv = None
        self._f_vectors = None  # experimental
        self._mfp = None  # experimental

        if grid_points is None:
            self._all_grid_points = True
        else:
            self._all_grid_points = False
        self._rot_grid_points = None

        if self._is_reducible_collision_matrix:
            self._collision = CollisionMatrix(
                self._pp, is_reducible_collision_matrix=True, log_level=self._log_level
            )
        else:
            self._rot_grid_points = self._get_rot_grid_points()
            self._collision = CollisionMatrix(
                self._pp,
                rotations_cartesian=self._rotations_cartesian,
                num_ir_grid_points=len(self._ir_grid_points),
                rot_grid_points=self._rot_grid_points,
                log_level=self._log_level,
            )

        if self._temperatures is not None:
            self._allocate_values()

    @property
    def collision_matrix(self):
        """Setter and getter of collision matrix."""
        return self._collision_matrix

    @collision_matrix.setter
    def collision_matrix(self, collision_matrix):
        self._collision_matrix = collision_matrix

    def get_collision_matrix(self):
        """Return collision matrix."""
        warnings.warn(
            "Use attribute, Conductivity_LBTE.collision_matrix "
            "instead of Conductivity_LBTE.get_collision_matrix().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.collision_matrix

    def set_collision_matrix(self, collision_matrix):
        """Set collision matrix."""
        warnings.warn(
            "Use attribute, Conductivity_LBTE.collision_matrix "
            "instead of Conductivity_LBTE.set_collision_matrix().",
            DeprecationWarning,
            stacklevel=2,
        )
        self.collision_matrix = collision_matrix

    def get_collision_eigenvalues(self):
        """Return eigenvalues of collision matrix."""
        return self._collision_eigenvalues

    def get_frequencies_all(self):
        """Return phonon frequencies on GR-grid."""
        return self._frequencies[self._pp.bz_grid.grg2bzg]

    def get_f_vectors(self):
        """Return f vectors.

        This is experimental.

        """
        return self._f_vectors

    def get_mean_free_path(self):
        """Return mean free path.

        This is experimental and not well defined.

        """
        return self._mfp

    def delete_gp_collision_and_pp(self):
        """Deallocate large arrays."""
        self._collision.delete_integration_weights()
        self._pp.delete_interaction_strength()

    def set_kappa_at_sigmas(self):
        """Calculate lattice thermal conductivity from collision matrix.

        This method is called after all elements of collision matrix are filled.

        """
        if len(self._grid_points) != len(self._ir_grid_points):
            print("Collision matrix is not well created.")
            import sys

            sys.exit(1)
        else:
            weights = self._prepare_collision_matrix()
            self._set_kappa_at_sigmas(weights)

    @abstractmethod
    def _set_kappa(self, i_sigma, i_temp, weights):
        raise NotImplementedError

    @abstractmethod
    def _set_kappa_at_sigmas(weights):
        raise NotImplementedError

    def _set_kappa_ir_colmat(self, kappa, mode_kappa, i_sigma, i_temp, weights):
        """Calculate direct solution thermal conductivity of ir colmat.

        kappa and mode_kappa are overwritten.

        """
        N = self._num_sampling_grid_points
        if self._solve_collective_phonon:
            self._set_mode_kappa_Chaput(mode_kappa, i_sigma, i_temp, weights)
        else:
            X = self._get_X(i_temp, weights)
            num_ir_grid_points = len(self._ir_grid_points)
            Y = self._get_Y(i_sigma, i_temp, weights, X)
            self._set_mean_free_path(i_sigma, i_temp, weights, Y)
            self._set_mode_kappa(
                mode_kappa,
                X,
                Y,
                num_ir_grid_points,
                self._rotations_cartesian,
                i_sigma,
                i_temp,
            )
            # self._set_mode_kappa_from_mfp(weights,
            #                               self._rotations_cartesian,
            #                               i_sigma,
            #                               i_temp)

        kappa[i_sigma, i_temp] = mode_kappa[i_sigma, i_temp].sum(axis=0).sum(axis=0) / N

    def _set_kappa_reducible_colmat(self, kappa, mode_kappa, i_sigma, i_temp, weights):
        """Calculate direct solution thermal conductivity of full colmat.

        kappa and mode_kappa are overwritten.

        """
        N = self._num_sampling_grid_points
        X = self._get_X(i_temp, weights)
        num_mesh_points = np.prod(self._pp.mesh_numbers)
        Y = self._get_Y(i_sigma, i_temp, weights, X)
        self._set_mean_free_path(i_sigma, i_temp, weights, Y)
        # Putting self._rotations_cartesian is to symmetrize kappa.
        # None can be put instead for watching pure information.
        self._set_mode_kappa(
            mode_kappa,
            X,
            Y,
            num_mesh_points,
            self._rotations_cartesian,
            i_sigma,
            i_temp,
        )
        mode_kappa[i_sigma, i_temp] /= len(self._rotations_cartesian)
        kappa[i_sigma, i_temp] = mode_kappa[i_sigma, i_temp].sum(axis=0).sum(axis=0) / N

    def _get_rot_grid_points(self):
        num_ir_grid_points = len(self._ir_grid_points)
        rot_grid_points = np.zeros(
            (num_ir_grid_points, len(self._point_operations)), dtype="int_"
        )
        if self._is_kappa_star:
            rotations = self._pp.bz_grid.rotations  # rotations of GR-grid
        else:
            rotations = self._point_operations  # only identity
        for i, ir_gp in enumerate(self._ir_grid_points):
            rot_grid_points[i] = get_grid_points_by_rotations(
                ir_gp, self._pp.bz_grid, reciprocal_rotations=rotations
            )
        return rot_grid_points

    def _allocate_values(self):
        """Allocate arrays."""
        if self._is_reducible_collision_matrix:
            self._allocate_reducible_colmat_values()
        else:
            self._allocate_ir_colmat_values()

    def _allocate_local_values(self, num_grid_points):
        """Allocate grid point local arrays."""
        num_band0 = len(self._pp.band_indices)
        num_temp = len(self._temperatures)
        self._gv = np.zeros((num_grid_points, num_band0, 3), dtype="double", order="C")
        self._cv = np.zeros(
            (num_temp, num_grid_points, num_band0), dtype="double", order="C"
        )
        self._gamma = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band0),
            dtype="double",
            order="C",
        )
        if self._is_full_pp:
            self._averaged_pp_interaction = np.zeros(
                (num_grid_points, num_band0), dtype="double", order="C"
            )
        if self._is_isotope:
            self._gamma_iso = np.zeros(
                (len(self._sigmas), num_grid_points, num_band0),
                dtype="double",
                order="C",
            )
        self._f_vectors = np.zeros(
            (num_grid_points, num_band0, 3), dtype="double", order="C"
        )
        self._mfp = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band0, 3),
            dtype="double",
            order="C",
        )

    def _run_at_grid_point(self):
        """Calculate properties at a grid point."""
        i_gp = self._grid_point_count
        self._show_log_header(i_gp)
        gp = self._grid_points[i_gp]

        if not self._all_grid_points:
            self._collision_matrix[:] = 0

        if not self._read_gamma:
            self._collision.set_grid_point(gp)

            if self._log_level:
                print("Number of triplets: %d" % len(self._pp.get_triplets_at_q()[0]))

            self._set_collision_matrix_at_sigmas(i_gp)

        if self._is_reducible_collision_matrix:
            i_data = self._pp.bz_grid.bzg2grg[gp]
        else:
            i_data = i_gp
        self._set_velocities(i_gp, i_data)
        self._set_cv(i_gp, i_data)
        if self._is_isotope:
            gamma_iso = self._get_gamma_isotope_at_sigmas(i_gp)
            band_indices = self._pp.band_indices
            self._gamma_iso[:, i_data, :] = gamma_iso[:, band_indices]

        if self._log_level:
            self._show_log(i_gp)

    def _allocate_reducible_colmat_values(self):
        """Allocate arrays for reducilble collision matrix."""
        num_band0 = len(self._pp.band_indices)
        num_band = len(self._pp.primitive) * 3
        num_temp = len(self._temperatures)
        num_mesh_points = np.prod(self._pp.mesh_numbers)
        if self._all_grid_points:
            num_stored_grid_points = num_mesh_points
        else:
            num_stored_grid_points = 1
        self._allocate_local_values(num_mesh_points)
        if self._collision_matrix is None:
            self._collision_matrix = np.empty(
                (
                    len(self._sigmas),
                    num_temp,
                    num_stored_grid_points,
                    num_band0,
                    num_mesh_points,
                    num_band,
                ),
                dtype="double",
                order="C",
            )
            self._collision_matrix[:] = 0
        self._collision_eigenvalues = np.zeros(
            (len(self._sigmas), num_temp, num_mesh_points * num_band),
            dtype="double",
            order="C",
        )

    def _allocate_ir_colmat_values(self):
        """Allocate arrays for ir collision matrix."""
        num_band0 = len(self._pp.band_indices)
        num_band = len(self._pp.primitive) * 3
        num_temp = len(self._temperatures)
        num_ir_grid_points = len(self._ir_grid_points)
        num_grid_points = len(self._grid_points)
        if self._all_grid_points:
            num_stored_grid_points = num_grid_points
        else:
            num_stored_grid_points = 1

        self._allocate_local_values(num_grid_points)
        if self._collision_matrix is None:
            self._collision_matrix = np.empty(
                (
                    len(self._sigmas),
                    num_temp,
                    num_stored_grid_points,
                    num_band0,
                    3,
                    num_ir_grid_points,
                    num_band,
                    3,
                ),
                dtype="double",
                order="C",
            )
            self._collision_matrix[:] = 0
        self._collision_eigenvalues = np.zeros(
            (len(self._sigmas), num_temp, num_ir_grid_points * num_band * 3),
            dtype="double",
            order="C",
        )

    def _set_collision_matrix_at_sigmas(self, i_gp):
        """Calculate collision matrices at grid point.

        i_gp : int
            Grid point count.

        """
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "Calculating collision matrix with "
                if sigma is None:
                    text += "tetrahedron method."
                else:
                    text += "sigma=%s" % sigma
                    if self._sigma_cutoff is None:
                        text += "."
                    else:
                        text += "(%4.2f SD)." % self._sigma_cutoff
                print(text)

            self._collision.set_sigma(sigma, sigma_cutoff=self._sigma_cutoff)
            self._collision.run_integration_weights()

            if self._read_pp:
                pp_strength, _g_zero = read_pp_from_hdf5(
                    self._pp.mesh_numbers,
                    grid_point=self._grid_points[i_gp],
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
                    print("=" * 26 + " Warning " + "=" * 26)
                    print("Inconsistency found in g_zero.")
                    print(
                        "The inconsistency may come from slight numerical "
                        "calculator difference between hardwares or linear algebra "
                        "libraries. "
                        "To avoid the inconsistency, it is recommended to use the same "
                        "phonon-*.hdf5 for generating pp-*.hdf5 because phonon "
                        "frequencies are used to determine g_zero. "
                        "If significant difference of values below is found, it can be "
                        "a sign of that something is really wrong. Otherwise, this "
                        "warning may be ignored."
                    )
                    print(_g_zero.shape, g_zero.shape)
                    for i, (_v, v) in enumerate(zip(_g_zero, g_zero)):
                        if (_v != v).any():
                            print(f"{i + 1} {_v.sum()} {v.sum()}")
                    self._collision.set_interaction_strength(
                        pp_strength, g_zero=_g_zero
                    )
                else:
                    self._collision.set_interaction_strength(pp_strength)
            elif j != 0 and (self._is_full_pp or self._sigma_cutoff is None):
                if self._log_level:
                    print("Existing ph-ph interaction is used.")
            else:
                if self._log_level:
                    print("Calculating ph-ph interaction...")
                self._collision.run_interaction(is_full_pp=self._is_full_pp)

            if self._is_full_pp and j == 0:
                self._averaged_pp_interaction[i_gp] = self._pp.averaged_interaction

            for k, t in enumerate(self._temperatures):
                self._collision.temperature = t
                self._collision.run()
                if self._all_grid_points:
                    if self._is_reducible_collision_matrix:
                        i_data = self._pp.bz_grid.bzg2grg[self._grid_points[i_gp]]
                    else:
                        i_data = i_gp
                else:
                    i_data = 0
                self._gamma[j, k, i_data] = self._collision.imag_self_energy
                self._collision_matrix[j, k, i_data] = (
                    self._collision.get_collision_matrix()
                )

    def _prepare_collision_matrix(self):
        """Collect pieces and construct collision matrix."""
        if self._is_reducible_collision_matrix:
            if self._is_kappa_star:
                self._average_collision_matrix_by_degeneracy()
                num_mesh_points = np.prod(self._pp.mesh_numbers)
                num_rot = len(self._point_operations)
                rot_grid_points = np.zeros((num_rot, num_mesh_points), dtype="int_")
                # Ir-grid points and rot_grid_points in generalized regular grid
                ir_gr_grid_points = np.array(
                    self._pp.bz_grid.bzg2grg[self._ir_grid_points], dtype="int_"
                )
                for i in range(num_mesh_points):
                    rot_grid_points[:, i] = self._pp.bz_grid.bzg2grg[
                        get_grid_points_by_rotations(
                            self._pp.bz_grid.grg2bzg[i], self._pp.bz_grid
                        )
                    ]
                self._expand_reducible_collisions(ir_gr_grid_points, rot_grid_points)
                self._expand_local_values(ir_gr_grid_points, rot_grid_points)
            self._combine_reducible_collisions()
            weights = np.ones(np.prod(self._pp.mesh_numbers), dtype="int_")
            self._symmetrize_collision_matrix()
        else:
            self._combine_collisions()
            weights = self._multiply_weights_to_collisions()
            self._average_collision_matrix_by_degeneracy()
            self._symmetrize_collision_matrix()

        return weights

    def _multiply_weights_to_collisions(self):
        weights = self._get_weights()
        for i, w_i in enumerate(weights):
            for j, w_j in enumerate(weights):
                self._collision_matrix[:, :, i, :, :, j, :, :] *= w_i * w_j
        return weights

    def _combine_collisions(self):
        """Include diagonal elements into collision matrix."""
        num_band = len(self._pp.primitive) * 3
        for j, k in np.ndindex((len(self._sigmas), len(self._temperatures))):
            for i, ir_gp in enumerate(self._ir_grid_points):
                for r, r_gp in zip(self._rotations_cartesian, self._rot_grid_points[i]):
                    if ir_gp != r_gp:
                        continue

                    main_diagonal = self._get_main_diagonal(i, j, k)
                    for ll in range(num_band):
                        self._collision_matrix[j, k, i, ll, :, i, ll, :] += (
                            main_diagonal[ll] * r
                        )

    def _combine_reducible_collisions(self):
        """Include diagonal elements into collision matrix."""
        num_band = len(self._pp.primitive) * 3
        num_mesh_points = np.prod(self._pp.mesh_numbers)

        for j, k in np.ndindex((len(self._sigmas), len(self._temperatures))):
            for i in range(num_mesh_points):
                main_diagonal = self._get_main_diagonal(i, j, k)
                for ll in range(num_band):
                    self._collision_matrix[j, k, i, ll, i, ll] += main_diagonal[ll]

    def _expand_reducible_collisions(self, ir_gr_grid_points, rot_grid_points):
        """Fill elements of full collision matrix by symmetry."""
        start = time.time()
        if self._log_level:
            sys.stdout.write("- Expanding properties to all grid points ")
            sys.stdout.flush()

        if self._lang == "C":
            import phono3py._phono3py as phono3c

            phono3c.expand_collision_matrix(
                self._collision_matrix, ir_gr_grid_points, rot_grid_points
            )
        else:
            num_mesh_points = np.prod(self._pp.mesh_numbers)
            colmat = self._collision_matrix
            for ir_gp in ir_gr_grid_points:
                multi = (rot_grid_points[:, ir_gp] == ir_gp).sum()
                colmat_irgp = colmat[:, :, ir_gp, :, :, :].copy()
                colmat_irgp /= multi
                colmat[:, :, ir_gp, :, :, :] = 0
                for j, _ in enumerate(self._rotations_cartesian):
                    gp_r = rot_grid_points[j, ir_gp]
                    for k in range(num_mesh_points):
                        gp_c = rot_grid_points[j, k]
                        colmat[:, :, gp_r, :, gp_c, :] += colmat_irgp[:, :, :, k, :]

        if self._log_level:
            print("[%.3fs]" % (time.time() - start))
            sys.stdout.flush()

    def _expand_local_values(self, ir_gr_grid_points, rot_grid_points):
        """Fill elements of local properties at grid points."""
        for ir_gp in ir_gr_grid_points:
            cv_irgp = self._cv[:, ir_gp, :].copy()
            self._cv[:, ir_gp, :] = 0
            gv_irgp = self._gv[ir_gp].copy()
            self._gv[ir_gp] = 0
            gamma_irgp = self._gamma[:, :, ir_gp, :].copy()
            self._gamma[:, :, ir_gp, :] = 0
            multi = (rot_grid_points[:, ir_gp] == ir_gp).sum()
            if self._is_isotope:
                gamma_iso_irgp = self._gamma_iso[:, ir_gp, :].copy()
                self._gamma_iso[:, ir_gp, :] = 0
            for j, r in enumerate(self._rotations_cartesian):
                gp_r = rot_grid_points[j, ir_gp]
                self._gamma[:, :, gp_r, :] += gamma_irgp / multi
                if self._is_isotope:
                    self._gamma_iso[:, gp_r, :] += gamma_iso_irgp / multi
                self._cv[:, gp_r, :] += cv_irgp / multi
                self._gv[gp_r] += np.dot(gv_irgp, r.T) / multi

    def _get_weights(self):
        """Return weights used for collision matrix and |X> and |f>.

        For symmetry compressed collision matrix.

        self._rot_grid_points : ndarray
            Grid points generated by applying point group to ir-grid-points
            in BZ-grid.
            shape=(ir_grid_points, point_operations), dtype='int_'

        r_gps : grid points of arms of k-star with duplicates
            len(r_gps) == order of crystallographic point group
            len(unique(r_gps)) == number of arms of the k-star

        Returns
        -------
        weights : ndarray
            sqrt(g_k/|g|), where g is the crystallographic point group and
            g_k is the number of arms of k-star at each ir-qpoint.
            shape=(ir_grid_points,), dtype='double'

        """
        weights = np.zeros(len(self._rot_grid_points), dtype="double")
        for i, r_gps in enumerate(self._rot_grid_points):
            weights[i] = np.sqrt(len(np.unique(r_gps)))

            sym_broken = False
            for gp in np.unique(r_gps):
                if len(np.where(r_gps == gp)[0]) != self._rot_grid_points.shape[
                    1
                ] // len(np.unique(r_gps)):
                    sym_broken = True

            if sym_broken:
                print("=" * 26 + " Warning " + "=" * 26)
                print("Symmetry of grid is broken.")

        return weights / np.sqrt(self._rot_grid_points.shape[1])

    def _symmetrize_collision_matrix(self):
        r"""Symmetrize collision matrix.

        (\Omega + \Omega^T) / 2.

        """
        start = time.time()

        try:
            import phono3py._phono3py as phono3c

            if self._log_level:
                sys.stdout.write("- Making collision matrix symmetric " "(built-in) ")
                sys.stdout.flush()
            phono3c.symmetrize_collision_matrix(self._collision_matrix)
        except ImportError:
            if self._log_level:
                sys.stdout.write("- Making collision matrix symmetric " "(numpy) ")
                sys.stdout.flush()

            if self._is_reducible_collision_matrix:
                size = np.prod(self._collision_matrix.shape[2:4])
            else:
                size = np.prod(self._collision_matrix.shape[2:5])
            for i in range(self._collision_matrix.shape[0]):
                for j in range(self._collision_matrix.shape[1]):
                    col_mat = self._collision_matrix[i, j].reshape(size, size)
                    col_mat += col_mat.T
                    col_mat /= 2

        if self._log_level:
            print("[%.3fs]" % (time.time() - start))
            sys.stdout.flush()

    def _average_collision_matrix_by_degeneracy(self):
        """Average symmetrically equivalent elemetns of collision matrix."""
        start = time.time()

        # Average matrix elements belonging to degenerate bands
        if self._log_level:
            sys.stdout.write(
                "- Averaging collision matrix elements " "by phonon degeneracy "
            )
            sys.stdout.flush()

        col_mat = self._collision_matrix
        for i, gp in enumerate(self._ir_grid_points):
            freqs = self._frequencies[gp]
            deg_sets = degenerate_sets(freqs)
            for dset in deg_sets:
                bi_set = []
                for j in range(len(freqs)):
                    if j in dset:
                        bi_set.append(j)

                if self._is_reducible_collision_matrix:
                    i_data = self._pp.bz_grid.bzg2grg[gp]
                    sum_col = col_mat[:, :, i_data, bi_set, :, :].sum(axis=2) / len(
                        bi_set
                    )
                    for j in bi_set:
                        col_mat[:, :, i_data, j, :, :] = sum_col
                else:
                    sum_col = col_mat[:, :, i, bi_set, :, :, :, :].sum(axis=2) / len(
                        bi_set
                    )
                    for j in bi_set:
                        col_mat[:, :, i, j, :, :, :, :] = sum_col

        for i, gp in enumerate(self._ir_grid_points):
            freqs = self._frequencies[gp]
            deg_sets = degenerate_sets(freqs)
            for dset in deg_sets:
                bi_set = []
                for j in range(len(freqs)):
                    if j in dset:
                        bi_set.append(j)
                if self._is_reducible_collision_matrix:
                    i_data = self._pp.bz_grid.bzg2grg[gp]
                    sum_col = col_mat[:, :, :, :, i_data, bi_set].sum(axis=4) / len(
                        bi_set
                    )
                    for j in bi_set:
                        col_mat[:, :, :, :, i_data, j] = sum_col
                else:
                    sum_col = col_mat[:, :, :, :, :, i, bi_set, :].sum(axis=5) / len(
                        bi_set
                    )
                    for j in bi_set:
                        col_mat[:, :, :, :, :, i, j, :] = sum_col

        if self._log_level:
            print("[%.3fs]" % (time.time() - start))
            sys.stdout.flush()

    def _get_X(self, i_temp, weights):
        """Calculate X in Chaput's paper."""
        num_band = len(self._pp.primitive) * 3
        X = self._gv.copy()
        if self._is_reducible_collision_matrix:
            freqs = self._frequencies[self._pp.bz_grid.grg2bzg]
        else:
            freqs = self._frequencies[self._ir_grid_points]

        t = self._temperatures[i_temp]
        sinh = np.where(
            freqs > self._pp.cutoff_frequency,
            np.sinh(freqs * THzToEv / (2 * Kb * t)),
            -1.0,
        )
        inv_sinh = np.where(sinh > 0, 1.0 / sinh, 0)
        freqs_sinh = freqs * THzToEv * inv_sinh / (4 * Kb * t**2)

        for i, f in enumerate(freqs_sinh):
            X[i] *= weights[i]
            for j in range(num_band):
                X[i, j] *= f[j]

        if t > 0:
            return X.reshape(-1, 3)
        else:
            return np.zeros_like(X.reshape(-1, 3))

    def _get_Y(self, i_sigma, i_temp, weights, X):
        r"""Calculate Y = (\Omega^-1, X)."""
        solver = select_colmat_solver(self._pinv_solver)
        num_band = len(self._pp.primitive) * 3

        if self._is_reducible_collision_matrix:
            num_grid_points = np.prod(self._pp.mesh_numbers)
            size = num_grid_points * num_band
        else:
            num_grid_points = len(self._ir_grid_points)
            size = num_grid_points * num_band * 3
        v = self._collision_matrix[i_sigma, i_temp].reshape(size, size)
        # Transpose eigvecs because colmat was solved by column major order
        if solver in [1, 2, 4, 5]:
            v = v.T

        start = time.time()

        if self._pinv_method == 0:
            eig_str = "abs(eig)"
        else:
            eig_str = "eig"
        print(
            f"Calculating pseudo-inv by ignoring {eig_str}<{self._pinv_cutoff:<.1e}",
            end="",
        )
        if solver in [0, 1, 2, 3, 4, 5]:
            if self._log_level:
                print(" (np.dot) ", end="")
                sys.stdout.flush()
            e = self._get_eigvals_pinv(i_sigma, i_temp)
            if self._is_reducible_collision_matrix:
                X1 = np.dot(v.T, X)
                for i in range(3):
                    X1[:, i] *= e
                Y = np.dot(v, X1)
            else:
                Y = np.dot(v, e * np.dot(v.T, X.ravel())).reshape(-1, 3)
        else:  # solver=6 This is slower as far as tested.
            import phono3py._phono3py as phono3c

            if self._log_level:
                print(" (built-in) ", end="")
                sys.stdout.flush()

            w = self._collision_eigenvalues[i_sigma, i_temp]
            phono3c.pinv_from_eigensolution(
                self._collision_matrix,
                w,
                i_sigma,
                i_temp,
                self._pinv_cutoff,
                self._pinv_method,
            )
            if self._is_reducible_collision_matrix:
                Y = np.dot(v, X)
            else:
                Y = np.dot(v, X.ravel()).reshape(-1, 3)

        self._set_f_vectors(Y, num_grid_points, weights)

        if self._log_level:
            print("[%.3fs]" % (time.time() - start))
            sys.stdout.flush()

        return Y

    def _set_f_vectors(self, Y, num_grid_points, weights):
        """Calculate f-vectors.

        Collision matrix is half of that defined in Chaput's paper.
        Therefore Y is divided by 2.

        """
        num_band = len(self._pp.primitive) * 3
        self._f_vectors[:] = (
            (Y / 2).reshape(num_grid_points, num_band * 3).T / weights
        ).T.reshape(self._f_vectors.shape)

    def _get_eigvals_pinv(self, i_sigma, i_temp):
        """Return inverse eigenvalues of eigenvalues > epsilon."""
        w = self._collision_eigenvalues[i_sigma, i_temp]
        e = np.zeros_like(w)

        for ll, val in enumerate(w):
            if self._pinv_method == 0:
                _val = abs(val)
            else:
                _val = val
            if _val > self._pinv_cutoff:
                e[ll] = 1 / val
        return e

    def _get_I(self, a, b, size, plus_transpose=True):
        """Return I matrix in Chaput's PRL paper.

        None is returned if I is zero matrix.

        """
        r_sum = np.zeros((3, 3), dtype="double", order="C")
        for r in self._rotations_cartesian:
            for i in range(3):
                for j in range(3):
                    r_sum[i, j] += r[a, i] * r[b, j]
        if plus_transpose:
            r_sum += r_sum.T

        # Return None not to consume computer for diagonalization
        if (np.abs(r_sum) < 1e-10).all():
            return None

        # Same as np.kron(np.eye(size), r_sum), but writen as below
        # to be sure the values in memory C-congiguous with 'double'.
        I_mat = np.zeros((3 * size, 3 * size), dtype="double", order="C")
        for i in range(size):
            I_mat[(i * 3) : ((i + 1) * 3), (i * 3) : ((i + 1) * 3)] = r_sum

        return I_mat

    @abstractmethod
    def _set_kappa_RTA(self, i_sigma, i_temp, weights):
        raise NotImplementedError

        """Calculate RTA thermal conductivity with either ir or full colmat."""
        if self._is_reducible_collision_matrix:
            self._set_kappa_RTA_reducible_colmat(i_sigma, i_temp, weights)
        else:
            self._set_kappa_RTA_ir_colmat(i_sigma, i_temp, weights)

    def _set_kappa_RTA_ir_colmat(
        self, kappa_RTA, mode_kappa_RTA, i_sigma, i_temp, weights
    ):
        """Calculate RTA thermal conductivity.

        This RTA is supposed to be the same as conductivity_RTA.

        """
        N = self._num_sampling_grid_points
        num_band = len(self._pp.primitive) * 3
        X = self._get_X(i_temp, weights)
        Y = np.zeros_like(X)
        num_ir_grid_points = len(self._ir_grid_points)
        for i, gp in enumerate(self._ir_grid_points):
            g = self._get_main_diagonal(i, i_sigma, i_temp)
            frequencies = self._frequencies[gp]
            for j, f in enumerate(frequencies):
                if f > self._pp.cutoff_frequency:
                    i_mode = i * num_band + j
                    old_settings = np.seterr(all="raise")
                    try:
                        Y[i_mode, :] = X[i_mode, :] / g[j]
                    except Exception:
                        print("=" * 26 + " Warning " + "=" * 26)
                        print(
                            " Unexpected physical condition of ph-ph "
                            "interaction calculation was found."
                        )
                        print(
                            " g[j]=%f at gp=%d, band=%d, freq=%f" % (g[j], gp, j + 1, f)
                        )
                        print("=" * 61)
                    np.seterr(**old_settings)

        self._set_mode_kappa(
            mode_kappa_RTA,
            X,
            Y,
            num_ir_grid_points,
            self._rotations_cartesian,
            i_sigma,
            i_temp,
        )
        kappa_RTA[i_sigma, i_temp] = (
            mode_kappa_RTA[i_sigma, i_temp].sum(axis=0).sum(axis=0) / N
        )

    def _set_kappa_RTA_reducible_colmat(
        self, kappa_RTA, mode_kappa_RTA, i_sigma, i_temp, weights
    ):
        """Calculate RTA thermal conductivity.

        This RTA is not equivalent to conductivity_RTA.
        The lifetime is defined from the diagonal part of collision matrix.

        `kappa` and `mode_kappa` are overwritten.

        """
        N = self._num_sampling_grid_points
        num_band = len(self._pp.primitive) * 3
        X = self._get_X(i_temp, weights)
        Y = np.zeros_like(X)

        num_mesh_points = np.prod(self._pp.mesh_numbers)
        size = num_mesh_points * num_band
        v_diag = np.diagonal(
            self._collision_matrix[i_sigma, i_temp].reshape(size, size)
        )

        for gp in range(num_mesh_points):
            frequencies = self._frequencies[gp]
            for j, f in enumerate(frequencies):
                if f > self._pp.cutoff_frequency:
                    i_mode = gp * num_band + j
                    Y[i_mode, :] = X[i_mode, :] / v_diag[i_mode]
        # Putting self._rotations_cartesian is to symmetrize kappa.
        # None can be put instead for watching pure information.
        self._set_mode_kappa(
            mode_kappa_RTA,
            X,
            Y,
            num_mesh_points,
            self._rotations_cartesian,
            i_sigma,
            i_temp,
        )
        g = len(self._rotations_cartesian)
        mode_kappa_RTA[i_sigma, i_temp] /= g
        kappa_RTA[i_sigma, i_temp] = (
            mode_kappa_RTA[i_sigma, i_temp].sum(axis=0).sum(axis=0) / N
        )

    def _set_mode_kappa(
        self, mode_kappa, X, Y, num_grid_points, rotations_cartesian, i_sigma, i_temp
    ):
        """Calculate mode thermal conductivity.

        kappa = A*(RX, RY) = A*(RX, R omega^-1 X), where A = k_B T^2 / V.

        Note
        ----
        Collision matrix is defined as a half of that in Chaput's paper.
        Therefore here 2 is not necessary multiplied.
        sum_k = sum_k + sum_k.T is equivalent to I(a,b) + I(b,a).

        """
        num_band = len(self._pp.primitive) * 3
        for i, (v_gp, f_gp) in enumerate(
            zip(
                X.reshape(num_grid_points, num_band, 3),
                Y.reshape(num_grid_points, num_band, 3),
            )
        ):
            for j, (v, f) in enumerate(zip(v_gp, f_gp)):
                # Do not consider three lowest modes at Gamma-point
                # It is assumed that there are no imaginary modes.
                if (self._pp.bz_grid.addresses[i] == 0).all() and j < 3:
                    continue

                if rotations_cartesian is None:
                    sum_k = np.outer(v, f)
                else:
                    sum_k = np.zeros((3, 3), dtype="double")
                    for r in rotations_cartesian:
                        sum_k += np.outer(np.dot(r, v), np.dot(r, f))
                sum_k = sum_k + sum_k.T
                for k, vxf in enumerate(
                    ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
                ):
                    mode_kappa[i_sigma, i_temp, i, j, k] = sum_k[vxf]

        t = self._temperatures[i_temp]
        mode_kappa[i_sigma, i_temp] *= self._conversion_factor * Kb * t**2

    def _set_mode_kappa_Chaput(self, mode_kappa, i_sigma, i_temp, weights):
        """Calculate mode kappa by the way in Laurent Chaput's PRL paper.

        This gives the different result from _set_mode_kappa and requires more
        memory space.

        """
        X = self._get_X(i_temp, weights).ravel()
        num_ir_grid_points = len(self._ir_grid_points)
        num_band = len(self._pp.primitive) * 3
        size = num_ir_grid_points * num_band * 3
        v = self._collision_matrix[i_sigma, i_temp].reshape(size, size)
        solver = select_colmat_solver(self._pinv_solver)
        if solver in [1, 2, 4, 5]:
            v = v.T
        e = self._get_eigvals_pinv(i_sigma, i_temp)
        t = self._temperatures[i_temp]

        omega_inv = np.empty(v.shape, dtype="double", order="C")
        np.dot(v, (e * v).T, out=omega_inv)
        Y = np.dot(omega_inv, X)
        self._set_f_vectors(Y, num_ir_grid_points, weights)
        elems = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        for i, vxf in enumerate(elems):
            mat = self._get_I(vxf[0], vxf[1], num_ir_grid_points * num_band)
            mode_kappa[i_sigma, i_temp, :, :, i] = 0
            if mat is not None:
                np.dot(mat, omega_inv, out=mat)
                # vals = (X ** 2 * np.diag(mat)).reshape(-1, 3).sum(axis=1)
                # vals = vals.reshape(num_ir_grid_points, num_band)
                # self._mode_kappa[i_sigma, i_temp, :, :, i] = vals
                w = diagonalize_collision_matrix(
                    mat, pinv_solver=self._pinv_solver, log_level=self._log_level
                )
                if solver in [1, 2, 4, 5]:
                    mat = mat.T
                spectra = np.dot(mat.T, X) ** 2 * w
                for s, eigvec in zip(spectra, mat.T):
                    vals = s * (eigvec**2).reshape(-1, 3).sum(axis=1)
                    vals = vals.reshape(num_ir_grid_points, num_band)
                    mode_kappa[i_sigma, i_temp, :, :, i] += vals

        factor = self._conversion_factor * Kb * t**2
        mode_kappa[i_sigma, i_temp] *= factor

    def _set_mode_kappa_from_mfp(self, weights, rotations_cartesian, i_sigma, i_temp):
        for i, (v_gp, mfp_gp, cv_gp) in enumerate(
            zip(self._gv, self._mfp[i_sigma, i_temp], self._cv[i_temp])
        ):
            for j, (v, mfp, cv) in enumerate(zip(v_gp, mfp_gp, cv_gp)):
                sum_k = np.zeros((3, 3), dtype="double")
                for r in rotations_cartesian:
                    sum_k += np.outer(np.dot(r, v), np.dot(r, mfp))
                sum_k = (sum_k + sum_k.T) / 2 * cv * weights[i] ** 2 * 2 * np.pi
                for k, vxf in enumerate(
                    ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
                ):
                    self._mode_kappa[i_sigma, i_temp, i, j, k] = sum_k[vxf]
        self._mode_kappa *= -self._conversion_factor

    def _set_mean_free_path(self, i_sigma, i_temp, weights, Y):
        t = self._temperatures[i_temp]
        # shape = (num_grid_points, num_band, 3),
        for i, f_gp in enumerate(self._f_vectors):
            for j, f in enumerate(f_gp):
                cv = self._cv[i_temp, i, j]
                if cv < 1e-10:
                    continue
                self._mfp[i_sigma, i_temp, i, j] = (
                    -2 * t * np.sqrt(Kb / cv) * f / (2 * np.pi)
                )

    def _show_log(self, i):
        gp = self._grid_points[i]
        frequencies = self._frequencies[gp]
        if self._is_reducible_collision_matrix:
            gv = self._gv[self._pp.bz_grid.bzg2grg[gp]]
        else:
            gv = self._gv[i]
        if self._is_full_pp:
            ave_pp = self._averaged_pp_interaction[i]
            text = "Frequency     group velocity (x, y, z)     |gv|       Pqj"
        else:
            text = "Frequency     group velocity (x, y, z)     |gv|"

        if self._velocity_obj.q_length is None:
            pass
        else:
            text += "  (dq=%3.1e)" % self._velocity_obj.q_length
        print(text)
        if self._is_full_pp:
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

        sys.stdout.flush()

    def _py_symmetrize_collision_matrix(self):
        num_band = len(self._pp.primitive) * 3
        num_ir_grid_points = len(self._ir_grid_points)
        for i in range(num_ir_grid_points):
            for j in range(num_band):
                for k in range(3):
                    for ll in range(num_ir_grid_points):
                        for m in range(num_band):
                            for n in range(3):
                                self._py_set_symmetrized_element(i, j, k, ll, m, n)

    def _py_set_symmetrized_element(self, i, j, k, ll, m, n):
        sym_val = (
            self._collision_matrix[:, :, i, j, k, ll, m, n]
            + self._collision_matrix[:, :, ll, m, n, i, j, k]
        ) / 2
        self._collision_matrix[:, :, i, j, k, ll, m, n] = sym_val
        self._collision_matrix[:, :, ll, m, n, i, j, k] = sym_val

    def _py_symmetrize_collision_matrix_no_kappa_stars(self):
        num_band = len(self._pp.primitive) * 3
        num_ir_grid_points = len(self._ir_grid_points)
        for i in range(num_ir_grid_points):
            for j in range(num_band):
                for k in range(num_ir_grid_points):
                    for ll in range(num_band):
                        self._py_set_symmetrized_element_no_kappa_stars(i, j, k, ll)

    def _py_set_symmetrized_element_no_kappa_stars(self, i, j, k, ll):
        sym_val = (
            self._collision_matrix[:, :, i, j, k, ll]
            + self._collision_matrix[:, :, k, ll, i, j]
        ) / 2
        self._collision_matrix[:, :, i, j, k, ll] = sym_val
        self._collision_matrix[:, :, k, ll, i, j] = sym_val


class ConductivityLBTE(ConductivityMixIn, ConductivityLBTEBase):
    """Lattice thermal conductivity calculation by direct solution."""

    def __init__(
        self,
        interaction: Interaction,
        grid_points=None,
        temperatures=None,
        sigmas=None,
        sigma_cutoff=None,
        is_isotope=False,
        mass_variances=None,
        boundary_mfp=None,
        solve_collective_phonon=False,
        is_reducible_collision_matrix=False,
        is_kappa_star=True,
        gv_delta_q=None,
        is_full_pp=False,
        read_pp=False,
        pp_filename=None,
        pinv_cutoff=1.0e-8,
        pinv_solver=0,
        pinv_method=0,
        log_level=0,
        lang="C",
    ):
        """Init method."""
        self._kappa = None
        self._kappa_RTA = None
        self._mode_kappa = None
        self._mode_kappa_RTA = None
        self._gv_sum2 = None

        super().__init__(
            interaction,
            grid_points=grid_points,
            temperatures=temperatures,
            sigmas=sigmas,
            sigma_cutoff=sigma_cutoff,
            is_isotope=is_isotope,
            mass_variances=mass_variances,
            boundary_mfp=boundary_mfp,
            solve_collective_phonon=solve_collective_phonon,
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            is_kappa_star=is_kappa_star,
            gv_delta_q=gv_delta_q,
            is_full_pp=is_full_pp,
            read_pp=read_pp,
            pp_filename=pp_filename,
            pinv_cutoff=pinv_cutoff,
            pinv_solver=pinv_solver,
            pinv_method=pinv_method,
            log_level=log_level,
            lang=lang,
        )

    @property
    def kappa_RTA(self):
        """Return RTA lattice thermal conductivity."""
        return self._kappa_RTA

    def get_kappa_RTA(self):
        """Return RTA lattice thermal conductivity."""
        warnings.warn(
            "Use attribute, Conductivity_LBTE.kappa_RTA "
            "instead of Conductivity_LBTE.get_kappa_RTA().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.kappa_RTA

    @property
    def mode_kappa_RTA(self):
        """Return RTA mode lattice thermal conductivities."""
        return self._mode_kappa_RTA

    def get_mode_kappa_RTA(self):
        """Return RTA mode lattice thermal conductivities."""
        warnings.warn(
            "Use attribute, Conductivity_LBTE.mode_kappa_RTA "
            "instead of Conductivity_LBTE.get_mode_kappa_RTA().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mode_kappa_RTA

    def _allocate_local_values(self, num_grid_points):
        """Allocate grid point local arrays.

        For full collision matrix, `num_grid_points` equals to the number of
        grid points in GRGrid, i.e., `prod(D_diag)`. Otherwise, number of
        grid points to be iterated over.

        """
        num_band0 = len(self._pp.band_indices)
        num_temp = len(self._temperatures)
        super()._allocate_local_values(num_grid_points)

        self._kappa = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._kappa_RTA = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._gv_sum2 = np.zeros(
            (num_grid_points, num_band0, 6), dtype="double", order="C"
        )
        self._mode_kappa = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band0, 6), dtype="double"
        )
        self._mode_kappa_RTA = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band0, 6), dtype="double"
        )

    def _set_kappa_at_sigmas(self, weights):
        """Calculate thermal conductivity from collision matrix."""
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "----------- Thermal conductivity (W/m-k) "
                if sigma:
                    text += "for sigma=%s -----------" % sigma
                else:
                    text += "with tetrahedron method -----------"
                print(text)
                sys.stdout.flush()

            for k, t in enumerate(self._temperatures):
                if t > 0:
                    self._set_kappa_RTA(j, k, weights)

                    w = diagonalize_collision_matrix(
                        self._collision_matrix,
                        i_sigma=j,
                        i_temp=k,
                        pinv_solver=self._pinv_solver,
                        log_level=self._log_level,
                    )
                    self._collision_eigenvalues[j, k] = w

                    self._set_kappa(j, k, weights)

                    if self._log_level:
                        print(
                            ("#%6s       " + " %-10s" * 6)
                            % ("T(K)", "xx", "yy", "zz", "yz", "xz", "xy")
                        )
                        print(
                            ("%7.1f " + " %10.3f" * 6)
                            % ((t,) + tuple(self._kappa[j, k]))
                        )
                        print(
                            (" %6s " + " %10.3f" * 6)
                            % (("(RTA)",) + tuple(self._kappa_RTA[j, k]))
                        )
                        print("-" * 76)
                        sys.stdout.flush()

        if self._log_level:
            print("")

    def _set_kappa(self, i_sigma, i_temp, weights):
        if self._is_reducible_collision_matrix:
            self._set_kappa_reducible_colmat(
                self._kappa, self._mode_kappa, i_sigma, i_temp, weights
            )
        else:
            self._set_kappa_ir_colmat(
                self._kappa, self._mode_kappa, i_sigma, i_temp, weights
            )

    def _set_kappa_RTA(self, i_sigma, i_temp, weights):
        if self._is_reducible_collision_matrix:
            self._set_kappa_RTA_reducible_colmat(
                self._kappa_RTA, self._mode_kappa_RTA, i_sigma, i_temp, weights
            )
        else:
            self._set_kappa_RTA_ir_colmat(
                self._kappa_RTA, self._mode_kappa_RTA, i_sigma, i_temp, weights
            )


class ConductivityWignerLBTE(ConductivityWignerMixIn, ConductivityLBTEBase):
    """Class of Wigner lattice thermal conductivity under direct-solution.

    Authors
    -------
    Michele Simoncelli

    """

    def __init__(
        self,
        interaction: Interaction,
        grid_points=None,
        temperatures=None,
        sigmas=None,
        sigma_cutoff=None,
        is_isotope=False,
        mass_variances=None,
        boundary_mfp=None,
        solve_collective_phonon=False,
        is_reducible_collision_matrix=False,
        is_kappa_star=True,
        gv_delta_q=None,
        is_full_pp=False,
        read_pp=False,
        pp_filename=None,
        pinv_cutoff=1.0e-8,
        pinv_solver=0,
        pinv_method=0,
        log_level=0,
        lang="C",
    ):
        """Init method."""
        self._kappa_TOT_exact = None
        self._kappa_TOT_RTA = None
        self._kappa_P_exact = None
        self._mode_kappa_P_exact = None
        self._kappa_P_RTA = None
        self._mode_kappa_P_RTA = None
        self._kappa_C = None
        self._mode_kappa_C = None
        self._gv_operator = None
        self._gv_operator_sum2 = None

        super().__init__(
            interaction,
            grid_points=grid_points,
            temperatures=temperatures,
            sigmas=sigmas,
            sigma_cutoff=sigma_cutoff,
            is_isotope=is_isotope,
            mass_variances=mass_variances,
            boundary_mfp=boundary_mfp,
            solve_collective_phonon=solve_collective_phonon,
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            is_kappa_star=is_kappa_star,
            gv_delta_q=gv_delta_q,
            is_full_pp=is_full_pp,
            read_pp=read_pp,
            pp_filename=pp_filename,
            pinv_cutoff=pinv_cutoff,
            pinv_solver=pinv_solver,
            pinv_method=pinv_method,
            log_level=log_level,
            lang=lang,
        )
        self._conversion_factor_WTE = get_conversion_factor_WTE(
            self._pp.primitive.volume
        )

    @property
    def kappa_TOT_exact(self):
        """Return kappa."""
        return self._kappa_TOT_exact

    @property
    def kappa_P_exact(self):
        """Return kappa."""
        return self._kappa_P_exact

    @property
    def mode_kappa_P_exact(self):
        """Return mode_kappa."""
        return self._mode_kappa_P_exact

    def _allocate_local_values(self, num_grid_points):
        """Allocate grid point local arrays."""
        num_band0 = len(self._pp.band_indices)
        num_temp = len(self._temperatures)
        super()._allocate_local_values(num_grid_points)

        nat3 = len(self._pp.primitive) * 3
        self._kappa_TOT_RTA = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._kappa_TOT_exact = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._kappa_P_exact = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._kappa_P_RTA = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._kappa_C = np.zeros(
            (len(self._sigmas), num_temp, 6), dtype="double", order="C"
        )
        self._mode_kappa_P_exact = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band0, 6), dtype="double"
        )
        self._mode_kappa_P_RTA = np.zeros(
            (len(self._sigmas), num_temp, num_grid_points, num_band0, 6), dtype="double"
        )

        complex_dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._gv_operator = np.zeros(
            (num_grid_points, num_band0, nat3, 3), order="C", dtype=complex_dtype
        )
        self._gv_operator_sum2 = np.zeros(
            (num_grid_points, num_band0, nat3, 6), order="C", dtype=complex_dtype
        )
        # one more index because we have off-diagonal terms (second index not
        # parallelized)
        self._mode_kappa_C = np.zeros(
            (
                len(self._sigmas),
                num_temp,
                num_grid_points,
                num_band0,
                nat3,
                6,
            ),
            order="C",
            dtype=complex_dtype,
        )

    def _set_kappa_at_sigmas(self, weights):
        """Calculate thermal conductivity from collision matrix."""
        for j, sigma in enumerate(self._sigmas):
            if self._log_level:
                text = "----------- Thermal conductivity (W/m-k) "
                if sigma:
                    text += "for sigma=%s -----------" % sigma
                else:
                    text += "with tetrahedron method -----------"
                print(text)
                sys.stdout.flush()

            for k, t in enumerate(self._temperatures):
                if t > 0:
                    self._set_kappa_RTA(j, k, weights)

                    w = diagonalize_collision_matrix(
                        self._collision_matrix,
                        i_sigma=j,
                        i_temp=k,
                        pinv_solver=self._pinv_solver,
                        log_level=self._log_level,
                    )
                    self._collision_eigenvalues[j, k] = w

                    self._set_kappa(j, k, weights)

                    if self._log_level:
                        print(
                            ("#%6s       " + " %-10s" * 6)
                            % (
                                "         \t\t  T(K)",
                                "xx",
                                "yy",
                                "zz",
                                "yz",
                                "xz",
                                "xy",
                            )
                        )
                        print(
                            "K_P_exact\t\t"
                            + ("%7.1f " + " %10.3f" * 6)
                            % ((t,) + tuple(self._kappa_P_exact[j, k]))
                        )
                        print(
                            "(K_P_RTA)\t\t"
                            + ("%7.1f " + " %10.3f" * 6)
                            % ((t,) + tuple(self._kappa_P_RTA[j, k]))
                        )
                        print(
                            "K_C      \t\t"
                            + ("%7.1f " + " %10.3f" * 6)
                            % ((t,) + tuple(self._kappa_C[j, k].real))
                        )
                        print(" ")
                        print(
                            "K_TOT=K_P_exact+K_C\t"
                            + ("%7.1f " + " %10.3f" * 6)
                            % (
                                (t,)
                                + tuple(self._kappa_C[j, k] + self._kappa_P_exact[j, k])
                            )
                        )
                        print("-" * 76)
                        sys.stdout.flush()

        if self._log_level:
            print("")

    def _set_kappa(self, i_sigma, i_temp, weights):
        if self._is_reducible_collision_matrix:
            self._set_kappa_reducible_colmat(
                self._kappa_P_exact, self._mode_kappa_P_exact, i_sigma, i_temp, weights
            )
        else:
            self._set_kappa_ir_colmat(
                self._kappa_P_exact, self._mode_kappa_P_exact, i_sigma, i_temp, weights
            )

    def _set_kappa_RTA(self, i_sigma, i_temp, weights):
        if self._is_reducible_collision_matrix:
            self._set_kappa_RTA_reducible_colmat(
                self._kappa_P_RTA, self._mode_kappa_P_RTA, i_sigma, i_temp, weights
            )
            print(
                " WARNING: Coherences conductivity not implemented for "
                "is_reducible_collision_matrix=True "
            )
        else:
            self._set_kappa_RTA_ir_colmat(
                self._kappa_P_RTA, self._mode_kappa_P_RTA, i_sigma, i_temp, weights
            )
            self._set_kappa_C_ir_colmat(i_sigma, i_temp)

    def _set_kappa_C_ir_colmat(self, i_sigma, i_temp):
        """Calculate coherence term of the thermal conductivity."""
        N = self._num_sampling_grid_points
        num_band = len(self._pp.primitive) * 3
        # num_ir_grid_points = len(self._ir_grid_points)
        for i, gp in enumerate(self._ir_grid_points):
            # linewidths at qpoint i, sigma i_sigma, and temperature i_temp
            g = self._get_main_diagonal(i, i_sigma, i_temp) * 2.0  # linewidth (FWHM)
            frequencies = self._frequencies[gp]
            cv = self._cv[i_temp, i, :]
            for s1 in range(num_band):
                for s2 in range(num_band):
                    hbar_omega_eV_s1 = (
                        frequencies[s1] * THzToEv
                    )  # hbar*omega=h*nu in eV
                    hbar_omega_eV_s2 = (
                        frequencies[s2] * THzToEv
                    )  # hbar*omega=h*nu in eV
                    if (
                        (frequencies[s1] > self._pp.cutoff_frequency)
                        and (frequencies[s2] > self._pp.cutoff_frequency)
                    ) and np.abs(frequencies[s1] - frequencies[s2]) > 1e-4:
                        # frequencies must be non-degenerate to contribute to k_C,
                        # otherwise they contribute to k_P
                        hbar_gamma_eV_s1 = g[s1] * THzToEv
                        hbar_gamma_eV_s2 = g[s2] * THzToEv
                        lorentzian_divided_by_hbar = (
                            0.5 * (hbar_gamma_eV_s1 + hbar_gamma_eV_s2)
                        ) / (
                            (hbar_omega_eV_s1 - hbar_omega_eV_s2) ** 2
                            + 0.25 * ((hbar_gamma_eV_s1 + hbar_gamma_eV_s2) ** 2)
                        )
                        #
                        prefactor = (
                            0.25
                            * (hbar_omega_eV_s1 + hbar_omega_eV_s2)
                            * (cv[s1] / hbar_omega_eV_s1 + cv[s2] / hbar_omega_eV_s2)
                        )
                        self._mode_kappa_C[i_sigma, i_temp, i, s1, s2] = (
                            (self._gv_operator_sum2[i, s1, s2])
                            * prefactor
                            * lorentzian_divided_by_hbar
                            * self._conversion_factor_WTE
                        )

        self._kappa_C[i_sigma, i_temp] = (
            self._mode_kappa_C[i_sigma, i_temp].sum(axis=0).sum(axis=0).sum(axis=0) / N
        ).real


def get_thermal_conductivity_LBTE(
    interaction: Interaction,
    temperatures=None,
    sigmas=None,
    sigma_cutoff=None,
    is_isotope=False,
    mass_variances=None,
    grid_points=None,
    boundary_mfp=None,  # in micrometre
    solve_collective_phonon=False,
    is_reducible_collision_matrix=False,
    is_kappa_star=True,
    gv_delta_q=None,
    is_full_pp=False,
    conductivity_type=None,
    pinv_cutoff=1.0e-8,
    pinv_solver=0,  # default: dsyev in lapacke
    pinv_method=0,  # default: abs(eig) < cutoff
    write_collision=False,
    read_collision=False,
    write_kappa=False,
    write_pp=False,
    read_pp=False,
    write_LBTE_solution=False,
    compression="gzip",
    input_filename=None,
    output_filename=None,
    log_level=0,
):
    """Calculate lattice thermal conductivity by direct solution."""
    if temperatures is None:
        _temperatures = [
            300,
        ]
    else:
        _temperatures = temperatures
    if sigmas is None:
        sigmas = []
    if log_level:
        print("-" * 19 + " Lattice thermal conducitivity (LBTE) " + "-" * 19)
        print(
            "Cutoff frequency of pseudo inversion of collision matrix: %s" % pinv_cutoff
        )

    if read_collision:
        temps = None
    else:
        temps = _temperatures

    conductivity_LBTE_class: Type[Union[ConductivityLBTE, ConductivityWignerLBTE]]
    if conductivity_type == "wigner":
        conductivity_LBTE_class = ConductivityWignerLBTE
    else:
        conductivity_LBTE_class = ConductivityLBTE

    lbte = conductivity_LBTE_class(
        interaction,
        grid_points=grid_points,
        temperatures=temps,
        sigmas=sigmas,
        sigma_cutoff=sigma_cutoff,
        is_isotope=is_isotope,
        mass_variances=mass_variances,
        boundary_mfp=boundary_mfp,
        solve_collective_phonon=solve_collective_phonon,
        is_reducible_collision_matrix=is_reducible_collision_matrix,
        is_kappa_star=is_kappa_star,
        gv_delta_q=gv_delta_q,
        is_full_pp=is_full_pp,
        read_pp=read_pp,
        pp_filename=input_filename,
        pinv_cutoff=pinv_cutoff,
        pinv_solver=pinv_solver,
        pinv_method=pinv_method,
        log_level=log_level,
    )

    if read_collision:
        read_from = set_collision_from_file(
            lbte,
            indices=read_collision,
            is_reducible_collision_matrix=is_reducible_collision_matrix,
            filename=input_filename,
            log_level=log_level,
        )
        if not read_from:
            print("Reading collision failed.")
            return False
        if log_level:
            temps_read = lbte.temperatures
            if len(temps_read) > 5:
                text = (" %.1f " * 5 + "...") % tuple(temps_read[:5])
                text += " %.1f" % temps_read[-1]
            else:
                text = (" %.1f " * len(temps_read)) % tuple(temps_read)
            print("Temperature: " + text)

    # This computes pieces of collision matrix sequentially.
    for i in lbte:
        if write_pp:
            write_phph(
                lbte, interaction, i, filename=output_filename, compression=compression
            )

        if write_collision:
            ConductivityLBTEWriter.write_collision(
                lbte,
                interaction,
                i=i,
                is_reducible_collision_matrix=is_reducible_collision_matrix,
                is_one_gp_colmat=(grid_points is not None),
                filename=output_filename,
            )

        lbte.delete_gp_collision_and_pp()

    # Write full collision matrix
    if write_LBTE_solution:
        if (
            read_collision
            and all_bands_exist(interaction)
            and read_from == "grid_points"
            and grid_points is None
        ) or (not read_collision):
            ConductivityLBTEWriter.write_collision(
                lbte, interaction, filename=output_filename
            )

    if grid_points is None and all_bands_exist(interaction):
        lbte.set_kappa_at_sigmas()

        if write_kappa:
            ConductivityLBTEWriter.write_kappa(
                lbte,
                interaction.primitive.volume,
                is_reducible_collision_matrix=is_reducible_collision_matrix,
                write_LBTE_solution=write_LBTE_solution,
                pinv_solver=pinv_solver,
                compression=compression,
                filename=output_filename,
                log_level=log_level,
            )

    return lbte


def diagonalize_collision_matrix(
    collision_matrices, i_sigma=None, i_temp=None, pinv_solver=0, log_level=0
):
    """Diagonalize collision matrices.

    Note
    ----
    collision_matricies is overwritten by eigenvectors.

    Parameters
    ----------
    collision_matricies : ndarray, optional
        Collision matrix. This ndarray has to have the following size and
        flags.
        shapes:
            (sigmas, temperatures, prod(mesh), num_band, prod(mesh), num_band)
            (sigmas, temperatures, ir_grid_points, num_band, 3,
                                   ir_grid_points, num_band, 3)
            (size, size)
        dtype='double', order='C'
    i_sigma : int, optional
        Index of BZ integration methods, tetrahedron method and smearing
        method with widths. Default is None.
    i_temp : int, optional
        Index of temperature. Default is None.
    pinv_solver : int, optional
        Diagnalization solver choice.
    log_level : int, optional
        Verbosity level. Smaller is more quiet. Default is 0.

    Returns
    -------
    w : ndarray, optional
        Eigenvalues.
        shape=(size_of_collision_matrix,), dtype='double'

    """
    start = time.time()

    # Matrix size of collision matrix to be diagonalized.
    # The following value is expected:
    #   ir-colmat:  num_ir_grid_points * num_band * 3
    #   red-colmat: num_mesh_points * num_band

    shape = collision_matrices.shape
    if len(shape) == 6:
        size = shape[2] * shape[3]
        assert size == shape[4] * shape[5]
    elif len(shape) == 8:
        size = np.prod(shape[2:5])
        assert size == np.prod(shape[5:8])
    elif len(shape) == 2:
        size = shape[0]
        assert size == shape[1]

    solver = select_colmat_solver(pinv_solver)

    # [1] dsyev: safer and slower than dsyevd and smallest memory usage
    # [2] dsyevd: faster than dsyev and largest memory usage
    if solver in [1, 2]:
        if log_level:
            routine = ["dsyev", "dsyevd"][solver - 1]
            sys.stdout.write("Diagonalizing by lapacke %s... " % routine)
            sys.stdout.flush()
        import phono3py._phono3py as phono3c

        w = np.zeros(size, dtype="double")
        if i_sigma is None:
            _i_sigma = 0
        else:
            _i_sigma = i_sigma
        if i_temp is None:
            _i_temp = 0
        else:
            _i_temp = i_temp
        phono3c.diagonalize_collision_matrix(
            collision_matrices, w, _i_sigma, _i_temp, 0.0, (solver + 1) % 2, 0
        )  # only diagonalization
    elif solver == 3:  # np.linalg.eigh depends on dsyevd.
        if log_level:
            sys.stdout.write("Diagonalizing by np.linalg.eigh... ")
            sys.stdout.flush()
        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        w, col_mat[:] = np.linalg.eigh(col_mat)

    elif solver == 4:  # fully scipy dsyev
        if log_level:
            sys.stdout.write("Diagonalizing by " "scipy.linalg.lapack.dsyev... ")
            sys.stdout.flush()
        import scipy.linalg

        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        w, _, info = scipy.linalg.lapack.dsyev(col_mat.T, overwrite_a=1)
    elif solver == 5:  # fully scipy dsyevd
        if log_level:
            sys.stdout.write("Diagonalizing by " "scipy.linalg.lapack.dsyevd... ")
            sys.stdout.flush()
        import scipy.linalg

        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        w, _, info = scipy.linalg.lapack.dsyevd(col_mat.T, overwrite_a=1)

    if log_level:
        print("[%.3fs]" % (time.time() - start))
        sys.stdout.flush()

    return w
