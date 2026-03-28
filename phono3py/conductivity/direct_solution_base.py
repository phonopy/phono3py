"""Calculate lattice thermal conductivity base class with direct solution."""

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

import os
import sys
import time
from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.physical_units import get_physical_units

from phono3py.conductivity.base import ConductivityBase
from phono3py.conductivity.utils import select_colmat_solver
from phono3py.file_IO import read_pp_from_hdf5
from phono3py.phonon.grid import get_grid_points_by_rotations
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.interaction import Interaction


class ConductivityLBTEBase(ConductivityBase):
    """Base class of ConductivityLBTE*.

    This is a base class for direct-solution classes.

    """

    def __init__(
        self,
        interaction: Interaction,
        grid_points: Sequence[int] | NDArray[np.int64] | None = None,
        temperatures: Sequence[float] | NDArray[np.double] | None = None,
        sigmas: Sequence[float | None] | None = None,
        sigma_cutoff: float | None = None,
        is_isotope: bool = False,
        mass_variances: Sequence[float] | NDArray[np.double] | None = None,
        boundary_mfp: float | None = None,  # in micrometer
        solve_collective_phonon: bool = False,
        is_reducible_collision_matrix: bool = False,
        is_kappa_star: bool = True,
        is_full_pp: bool = False,
        read_pp: bool = False,
        pp_filename: str | os.PathLike | None = None,
        pinv_cutoff: float = 1.0e-8,
        pinv_solver: int = 0,
        pinv_method: int = 0,
        log_level: int = 0,
        lang: Literal["C", "Python"] = "C",
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

        self._lang: Literal["C", "Python"] = lang
        self._collision_eigenvalues: NDArray[np.double] | None = None
        self._is_reducible_collision_matrix = is_reducible_collision_matrix
        self._solve_collective_phonon = solve_collective_phonon
        # if not self._is_kappa_star:
        #     self._is_reducible_collision_matrix = True
        self._collision_matrix: NDArray[np.double] | None = None
        self._read_pp = read_pp
        self._pp_filename = pp_filename
        self._pinv_cutoff = pinv_cutoff
        self._pinv_method = pinv_method
        self._pinv_solver = pinv_solver

        self._cv: NDArray[np.double] | None = None
        self._f_vectors: NDArray[np.double] | None = None  # experimental
        self._mfp: NDArray[np.double] | None = None  # experimental

        if grid_points is None:
            self._all_grid_points = True
        else:
            self._all_grid_points = False
        self._rot_grid_points: NDArray[np.int64] | None = None

        if self._is_reducible_collision_matrix:
            self._collision = CollisionMatrix(
                self._pp,
                is_reducible_collision_matrix=True,
                log_level=self._log_level,
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
    def collision_matrix(self) -> NDArray[np.double] | None:
        """Setter and getter of collision matrix."""
        return self._collision_matrix

    @collision_matrix.setter
    def collision_matrix(self, collision_matrix: NDArray[np.double] | None) -> None:
        self._collision_matrix = collision_matrix

    @property
    def collision_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues of collision matrix."""
        return self._collision_eigenvalues

    def get_frequencies_all(self) -> NDArray[np.double]:
        """Return phonon frequencies on GR-grid."""
        assert self._frequencies is not None
        return self._frequencies[self._pp.bz_grid.grg2bzg]

    def get_f_vectors(self) -> NDArray[np.double] | None:
        """Return f vectors.

        This is experimental.

        """
        return self._f_vectors

    def get_mean_free_path(self) -> NDArray[np.double] | None:
        """Return mean free path.

        This is experimental and not well defined.

        """
        return self._mfp

    def delete_gp_collision_and_pp(self) -> None:
        """Deallocate large arrays."""
        self._collision.delete_integration_weights()
        self._pp.delete_interaction_strength()

    def set_kappa_at_sigmas(self) -> None:
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
    def _set_kappa(
        self, i_sigma: int, i_temp: int, weights: NDArray[np.double]
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _set_kappa_at_sigmas(self, weights: NDArray[np.double]) -> None:
        raise NotImplementedError()

    def _set_kappa_ir_colmat(
        self,
        kappa: NDArray[np.double],
        mode_kappa: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
    ) -> None:
        """Calculate direct solution thermal conductivity of ir colmat.

        kappa and mode_kappa are overwritten.

        """
        N = self.number_of_sampling_grid_points
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
                i_sigma,
                i_temp,
            )
            # self._set_mode_kappa_from_mfp(weights, i_sigma, i_temp)

        kappa[i_sigma, i_temp] = mode_kappa[i_sigma, i_temp].sum(axis=0).sum(axis=0) / N

    def _set_kappa_reducible_colmat(
        self,
        kappa: NDArray[np.double],
        mode_kappa: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
    ) -> None:
        """Calculate direct solution thermal conductivity of full colmat.

        kappa and mode_kappa are overwritten.

        """
        N = self.number_of_sampling_grid_points
        X = self._get_X(i_temp, weights)
        num_mesh_points = int(np.prod(self._pp.mesh_numbers))
        Y = self._get_Y(i_sigma, i_temp, weights, X)
        self._set_mean_free_path(i_sigma, i_temp, weights, Y)
        # Putting self._rotations_cartesian is to symmetrize kappa.
        # None can be put instead for watching pure information.
        self._set_mode_kappa(
            mode_kappa,
            X,
            Y,
            num_mesh_points,
            i_sigma,
            i_temp,
        )
        mode_kappa[i_sigma, i_temp] /= len(self._rotations_cartesian)
        kappa[i_sigma, i_temp] = mode_kappa[i_sigma, i_temp].sum(axis=0).sum(axis=0) / N

    def _get_rot_grid_points(self) -> NDArray[np.int64]:
        num_ir_grid_points = len(self._ir_grid_points)
        rot_grid_points = np.zeros(
            (num_ir_grid_points, len(self._point_operations)), dtype="int64"
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

    def _allocate_values(self) -> None:
        """Allocate arrays."""
        if self._is_reducible_collision_matrix:
            self._allocate_reducible_colmat_values()
        else:
            self._allocate_ir_colmat_values()

    def _allocate_local_values(self, num_grid_points: int) -> None:
        """Allocate grid point local arrays."""
        assert self._temperatures is not None
        num_band0 = len(self._pp.band_indices)
        num_temp = len(self._temperatures)
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

    def _run_at_grid_point(self) -> None:
        """Calculate properties at a grid point."""
        i_gp = self._grid_point_count
        self._show_log_header(i_gp)
        gp = self._grid_points[i_gp]

        self._prepare_collisions_at_grid_point(i_gp, gp)
        i_data = self._get_data_index(i_gp, gp)
        self._set_local_properties_at_grid_point(i_gp, i_data)

        if self._log_level:
            self._show_log(i_gp)

    def _prepare_collisions_at_grid_point(self, i_gp: int, gp: int) -> None:
        self._reset_collision_matrix_if_needed()

        if self._read_gamma:
            return

        self._collision.set_grid_point(gp)
        if self._log_level:
            triplets = self._pp.get_triplets_at_q()[0]
            assert triplets is not None
            print("Number of triplets: %d" % len(triplets))
        self._set_collision_matrix_at_sigmas(i_gp)

    def _reset_collision_matrix_if_needed(self) -> None:
        assert self._collision_matrix is not None
        if not self._all_grid_points:
            self._collision_matrix[:] = 0

    def _set_local_properties_at_grid_point(self, i_gp: int, i_data: int) -> None:
        self._set_velocities(i_gp, i_data)
        self._set_cv(i_gp, i_data)
        if self._is_isotope:
            self._set_isotope_gamma_at_grid_point(i_gp, i_data)

    def _get_data_index(self, i_gp: int, gp: int) -> int:
        if self._is_reducible_collision_matrix:
            return self._pp.bz_grid.bzg2grg[gp]
        return i_gp

    def _set_isotope_gamma_at_grid_point(self, i_gp: int, i_data: int) -> None:
        gamma_iso = self._get_gamma_isotope_at_sigmas(i_gp)
        band_indices = self._pp.band_indices
        assert self._gamma_iso is not None
        self._gamma_iso[:, i_data, :] = gamma_iso[:, band_indices]

    def _allocate_reducible_colmat_values(self) -> None:
        """Allocate arrays for reducilble collision matrix."""
        assert self._temperatures is not None
        num_band0 = len(self._pp.band_indices)
        num_band = len(self._pp.primitive) * 3
        num_temp = len(self._temperatures)
        num_mesh_points = int(np.prod(self._pp.mesh_numbers))
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

    def _allocate_ir_colmat_values(self) -> None:
        """Allocate arrays for ir collision matrix."""
        assert self._temperatures is not None
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

    def _set_collision_matrix_at_sigmas(self, i_gp: int) -> None:
        """Calculate collision matrices at grid point.

        i_gp : int
            Grid point count.

        """
        for j, sigma in enumerate(self._sigmas):
            self._run_collision_matrix_at_sigma(i_gp, j, sigma)

    def _run_collision_matrix_at_sigma(
        self, i_gp: int, i_sigma: int, sigma: float | None
    ) -> None:
        self._show_collision_matrix_sigma_log(sigma)
        self._collision.set_sigma(sigma, sigma_cutoff=self._sigma_cutoff)
        self._collision.run_integration_weights()

        self._set_interaction_strength_at_sigma(i_gp, i_sigma, sigma)
        self._store_averaged_pp_if_needed(i_gp, i_sigma)
        self._store_collision_results_at_sigma(i_gp, i_sigma)

    def _store_averaged_pp_if_needed(self, i_gp: int, i_sigma: int) -> None:
        if self._is_full_pp and i_sigma == 0:
            assert self._averaged_pp_interaction is not None
            self._averaged_pp_interaction[i_gp] = self._pp.averaged_interaction

    def _show_collision_matrix_sigma_log(self, sigma: float | None) -> None:
        if not self._log_level:
            return

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

    def _set_interaction_strength_at_sigma(
        self, i_gp: int, i_sigma: int, sigma: float | None
    ) -> None:
        if self._read_pp:
            self._set_interaction_strength_from_file(i_gp, sigma)
        elif i_sigma != 0 and (self._is_full_pp or self._sigma_cutoff is None):
            if self._log_level:
                print("Existing ph-ph interaction is used.")
        else:
            if self._log_level:
                print("Calculating ph-ph interaction...")
            self._collision.run_interaction(is_full_pp=self._is_full_pp)

    def _set_interaction_strength_from_file(
        self, i_gp: int, sigma: float | None
    ) -> None:
        pp_strength, _g_zero = read_pp_from_hdf5(
            self._pp.mesh_numbers,
            grid_point=self._grid_points[i_gp],
            sigma=sigma,
            sigma_cutoff=self._sigma_cutoff,
            filename=self._pp_filename,  # type: ignore[arg-type]
            verbose=(self._log_level > 0),
        )
        _, g_zero = self._collision.get_integration_weights()
        if self._log_level and len(self._sigmas) > 1:
            print(
                "Multiple sigmas or mixing smearing and "
                "tetrahedron method is not supported."
            )
        if _g_zero is not None and g_zero is not None and (_g_zero != g_zero).any():
            self._show_g_zero_inconsistency_warning(_g_zero, g_zero)
            self._collision.set_interaction_strength(pp_strength, g_zero=_g_zero)
        else:
            self._collision.set_interaction_strength(pp_strength)

    def _show_g_zero_inconsistency_warning(
        self, g_zero_from_file: NDArray[np.int8], g_zero_runtime: NDArray[np.int8]
    ) -> None:
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
        print(g_zero_from_file.shape, g_zero_runtime.shape)
        for i, (_v, v) in enumerate(zip(g_zero_from_file, g_zero_runtime, strict=True)):
            if (_v != v).any():
                print(f"{i + 1} {_v.sum()} {v.sum()}")

    def _store_collision_results_at_sigma(self, i_gp: int, i_sigma: int) -> None:
        assert self._temperatures is not None
        assert self._collision_matrix is not None
        i_data = self._get_collision_storage_index(i_gp)
        for k, t in enumerate(self._temperatures):
            self._collision.temperature = t
            self._collision.run()
            self._gamma[i_sigma, k, i_data] = self._collision.imag_self_energy
            self._collision_matrix[i_sigma, k, i_data] = (
                self._collision.get_collision_matrix()  # type: ignore[union-attr]
            )

    def _get_collision_storage_index(self, i_gp: int) -> int:
        if not self._all_grid_points:
            return 0
        if self._is_reducible_collision_matrix:
            gp = int(self._grid_points[i_gp])
            return self._pp.bz_grid.bzg2grg[gp]
        return i_gp

    def _prepare_collision_matrix(self) -> NDArray[np.double]:
        """Collect pieces and construct collision matrix."""
        assert self._collision_matrix is not None
        if self._log_level:
            print(f"- Collision matrix shape {self._collision_matrix.shape}")

        return self._prepare_collision_matrix_by_type()

    def _prepare_collision_matrix_by_type(self) -> NDArray[np.double]:
        if self._is_reducible_collision_matrix:
            return self._prepare_reducible_collision_matrix()
        return self._prepare_ir_collision_matrix()

    def _prepare_reducible_collision_matrix(self) -> NDArray[np.double]:
        if self._is_kappa_star:
            self._expand_reducible_collision_matrix_by_symmetry()

        self._combine_reducible_collisions()
        weights = self._get_reducible_collision_weights()
        self._symmetrize_collision_matrix()
        return weights

    def _expand_reducible_collision_matrix_by_symmetry(self) -> None:
        self._average_collision_matrix_by_degeneracy()
        ir_gr_grid_points, rot_grid_points = self._get_reducible_rotation_maps()
        self._expand_reducible_collisions(ir_gr_grid_points, rot_grid_points)
        self._expand_local_values(ir_gr_grid_points, rot_grid_points)

    def _get_reducible_collision_weights(self) -> NDArray[np.double]:
        return np.ones(np.prod(self._pp.mesh_numbers), dtype="double")

    def _prepare_ir_collision_matrix(self) -> NDArray[np.double]:
        self._combine_collisions()
        weights = self._apply_ir_collision_weights()
        self._average_collision_matrix_by_degeneracy()
        self._symmetrize_collision_matrix()
        return weights

    def _apply_ir_collision_weights(self) -> NDArray[np.double]:
        return self._multiply_weights_to_collisions()

    def _get_reducible_rotation_maps(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        num_mesh_points = np.prod(self._pp.mesh_numbers)
        num_rot = len(self._point_operations)
        rot_grid_points = np.zeros((num_rot, num_mesh_points), dtype="int64")
        ir_gr_grid_points = np.array(
            self._pp.bz_grid.bzg2grg[self._ir_grid_points], dtype="int64"
        )
        for i in range(num_mesh_points):
            rot_grid_points[:, i] = self._pp.bz_grid.bzg2grg[
                get_grid_points_by_rotations(
                    self._pp.bz_grid.grg2bzg[i], self._pp.bz_grid
                )
            ]
        return ir_gr_grid_points, rot_grid_points

    def _multiply_weights_to_collisions(self) -> NDArray[np.double]:
        assert self._collision_matrix is not None
        weights = self._get_weights()
        for i, w_i in enumerate(weights):
            for j, w_j in enumerate(weights):
                self._collision_matrix[:, :, i, :, :, j, :, :] *= w_i * w_j
        return weights

    def _combine_collisions(self) -> None:
        """Include diagonal elements into collision matrix."""
        for j, k in self._iter_sigma_temperature_indices():
            for (
                i_irgp,
                main_diagonal,
                rotation,
            ) in self._iter_ir_collision_diagonal_entries(j, k):
                self._add_main_diagonal_to_ir_collision(
                    i_sigma=j,
                    i_temp=k,
                    i_irgp=i_irgp,
                    main_diagonal=main_diagonal,
                    rotation=rotation,
                )

    def _combine_reducible_collisions(self) -> None:
        """Include diagonal elements into collision matrix."""
        for j, k in self._iter_sigma_temperature_indices():
            entries = self._iter_reducible_collision_diagonal_entries(j, k)
            for i_mesh, main_diagonal in entries:
                self._add_main_diagonal_to_reducible_collision(
                    i_sigma=j,
                    i_temp=k,
                    i_mesh=i_mesh,
                    main_diagonal=main_diagonal,
                )

    def _iter_sigma_temperature_indices(self) -> Iterator[tuple[int, int]]:
        assert self._temperatures is not None
        return np.ndindex((len(self._sigmas), len(self._temperatures)))  # type: ignore[return-value]

    def _iter_ir_collision_diagonal_entries(
        self, i_sigma: int, i_temp: int
    ) -> Iterator[tuple[int, NDArray[np.double], NDArray[np.double]]]:
        assert self._rot_grid_points is not None
        for i_irgp, ir_gp in enumerate(self._ir_grid_points):
            for rotation, rotated_gp in zip(
                self._rotations_cartesian, self._rot_grid_points[i_irgp], strict=True
            ):
                if ir_gp != rotated_gp:
                    continue

                main_diagonal = self._get_main_diagonal(i_irgp, i_sigma, i_temp)
                yield i_irgp, main_diagonal, rotation

    def _iter_reducible_collision_diagonal_entries(
        self, i_sigma: int, i_temp: int
    ) -> Iterator[tuple[int, NDArray[np.double]]]:
        num_mesh_points = np.prod(self._pp.mesh_numbers)
        for i_mesh in range(num_mesh_points):
            main_diagonal = self._get_main_diagonal(i_mesh, i_sigma, i_temp)
            yield i_mesh, main_diagonal

    def _add_main_diagonal_to_ir_collision(
        self,
        *,
        i_sigma: int,
        i_temp: int,
        i_irgp: int,
        main_diagonal: NDArray[np.double],
        rotation: NDArray[np.double],
    ) -> None:
        assert self._collision_matrix is not None
        for ll, diag in enumerate(main_diagonal):
            self._collision_matrix[i_sigma, i_temp, i_irgp, ll, :, i_irgp, ll, :] += (
                diag * rotation
            )

    def _add_main_diagonal_to_reducible_collision(
        self,
        *,
        i_sigma: int,
        i_temp: int,
        i_mesh: int,
        main_diagonal: NDArray[np.double],
    ) -> None:
        assert self._collision_matrix is not None
        for ll, diag in enumerate(main_diagonal):
            self._collision_matrix[i_sigma, i_temp, i_mesh, ll, i_mesh, ll] += diag

    def _expand_reducible_collisions(
        self,
        ir_gr_grid_points: NDArray[np.int64],
        rot_grid_points: NDArray[np.int64],
    ) -> None:
        """Fill elements of full collision matrix by symmetry."""
        assert self._collision_matrix is not None
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

    def _expand_local_values(
        self,
        ir_gr_grid_points: NDArray[np.int64],
        rot_grid_points: NDArray[np.int64],
    ) -> None:
        """Fill elements of local properties at grid points.

        Note
        ----
        Internal state of self._conductivity_components is updated.

        """
        assert self._gamma is not None
        cv = self._conductivity_components.mode_heat_capacities
        gv = self._conductivity_components.group_velocities
        for ir_gp in ir_gr_grid_points:
            cv_irgp = cv[:, ir_gp, :].copy()
            cv[:, ir_gp, :] = 0
            gv_irgp = gv[ir_gp].copy()
            gv[ir_gp] = 0
            gamma_irgp = self._gamma[:, :, ir_gp, :].copy()
            self._gamma[:, :, ir_gp, :] = 0
            multi = (rot_grid_points[:, ir_gp] == ir_gp).sum()
            if self._is_isotope:
                assert self._gamma_iso is not None
                gamma_iso_irgp = self._gamma_iso[:, ir_gp, :].copy()
                self._gamma_iso[:, ir_gp, :] = 0
            for j, r in enumerate(self._rotations_cartesian):
                gp_r = rot_grid_points[j, ir_gp]
                self._gamma[:, :, gp_r, :] += gamma_irgp / multi
                if self._is_isotope:
                    assert self._gamma_iso is not None
                    self._gamma_iso[:, gp_r, :] += gamma_iso_irgp / multi  # type: ignore[possibly-undefined]
                cv[:, gp_r, :] += cv_irgp / multi
                gv[gp_r] += np.dot(gv_irgp, r.T) / multi

    def _get_weights(self) -> NDArray[np.double]:
        """Return weights used for collision matrix and |X> and |f>.

        For symmetry compressed collision matrix.

        self._rot_grid_points : ndarray
            Grid points generated by applying point group to ir-grid-points
            in BZ-grid.
            shape=(ir_grid_points, point_operations), dtype='int64'

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
        assert self._rot_grid_points is not None
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

    def _symmetrize_collision_matrix(self) -> None:
        r"""Symmetrize collision matrix.

        (\Omega + \Omega^T) / 2.

        """
        start = time.time()

        symmetrizer = self._select_collision_matrix_symmetrizer()
        symmetrizer()

        if self._log_level:
            print("[%.3fs]" % (time.time() - start))
            sys.stdout.flush()

    def _select_collision_matrix_symmetrizer(self) -> Callable[[], None]:
        """Return available symmetrization backend."""
        if self._can_use_builtin_collision_symmetrizer():
            return self._symmetrize_collision_matrix_with_builtin
        return self._symmetrize_collision_matrix_with_numpy

    def _can_use_builtin_collision_symmetrizer(self) -> bool:
        """Return True when C-extension symmetrizer is available."""
        try:
            import phono3py._phono3py  # noqa: F401

            return True
        except ImportError:
            return False

    def _symmetrize_collision_matrix_with_builtin(self) -> None:
        """Symmetrize collision matrix using C-extension backend."""
        import phono3py._phono3py as phono3c

        if self._log_level:
            sys.stdout.write("- Making collision matrix symmetric (built-in) ")
            sys.stdout.flush()
        phono3c.symmetrize_collision_matrix(self._collision_matrix)

    def _symmetrize_collision_matrix_with_numpy(self) -> None:
        """Symmetrize collision matrix by numpy fallback."""
        if self._log_level:
            sys.stdout.write("- Making collision matrix symmetric (numpy) ")
            sys.stdout.flush()

        assert self._collision_matrix is not None
        size = self._get_symmetrization_matrix_size()
        for i in range(self._collision_matrix.shape[0]):
            for j in range(self._collision_matrix.shape[1]):
                col_mat = self._collision_matrix[i, j].reshape(size, size)
                col_mat += col_mat.T
                col_mat /= 2

    def _get_symmetrization_matrix_size(self) -> int:
        """Return flattened matrix size used in symmetrization."""
        assert self._collision_matrix is not None
        if self._is_reducible_collision_matrix:
            return int(np.prod(self._collision_matrix.shape[2:4]))
        return int(np.prod(self._collision_matrix.shape[2:5]))

    def _average_collision_matrix_by_degeneracy(self) -> None:
        """Average symmetrically equivalent elements of collision matrix."""
        start = time.time()

        # Average matrix elements belonging to degenerate bands
        if self._log_level:
            sys.stdout.write(
                "- Averaging collision matrix elements by phonon degeneracy "
            )
            sys.stdout.flush()

        assert self._collision_matrix is not None
        col_mat = self._collision_matrix
        self._average_collision_matrix_rows_by_degeneracy(col_mat)
        self._average_collision_matrix_columns_by_degeneracy(col_mat)

        if self._log_level:
            print("[%.3fs]" % (time.time() - start))
            sys.stdout.flush()

    def _average_collision_matrix_rows_by_degeneracy(
        self, col_mat: NDArray[np.double]
    ) -> None:
        assert self._frequencies is not None
        for i, gp in enumerate(self._ir_grid_points):
            freqs = self._frequencies[gp]
            for dset in degenerate_sets(freqs):
                bi_set = self._get_bi_set_from_degenerate_set(freqs, dset)

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

    def _average_collision_matrix_columns_by_degeneracy(
        self, col_mat: NDArray[np.double]
    ) -> None:
        assert self._frequencies is not None
        for i, gp in enumerate(self._ir_grid_points):
            freqs = self._frequencies[gp]
            for dset in degenerate_sets(freqs):
                bi_set = self._get_bi_set_from_degenerate_set(freqs, dset)

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

    @staticmethod
    def _get_bi_set_from_degenerate_set(
        freqs: NDArray[np.double], dset: list[int]
    ) -> list[int]:
        bi_set = []
        for j in range(len(freqs)):
            if j in dset:
                bi_set.append(j)
        return bi_set

    def _get_X(self, i_temp: int, weights: NDArray[np.double]) -> NDArray[np.double]:
        """Calculate X in Chaput's paper."""
        X = self._conductivity_components.group_velocities.copy()
        num_band = len(self._pp.primitive) * 3
        assert self._temperatures is not None
        freqs = self._get_X_frequencies()
        t = self._temperatures[i_temp]
        freqs_factor = self._get_X_frequency_factor(freqs, t)
        self._scale_X_by_weights_and_frequency(X, weights, freqs_factor, num_band)

        if t <= 0:
            return np.zeros_like(X.reshape(-1, 3))
        return X.reshape(-1, 3)

    def _get_X_frequencies(self) -> NDArray[np.double]:
        assert self._frequencies is not None
        if self._is_reducible_collision_matrix:
            return self._frequencies[self._pp.bz_grid.grg2bzg]
        return self._frequencies[self._ir_grid_points]

    def _get_X_frequency_factor(
        self, freqs: NDArray[np.double], temperature: float
    ) -> NDArray[np.double]:
        sinh = np.where(
            freqs > self._pp.cutoff_frequency,
            np.sinh(
                freqs
                * get_physical_units().THzToEv
                / (2 * get_physical_units().KB * temperature)
            ),
            -1.0,
        )
        inv_sinh = np.where(sinh > 0, 1.0 / sinh, 0)
        return (
            freqs
            * get_physical_units().THzToEv
            * inv_sinh
            / (4 * get_physical_units().KB * temperature**2)
        )

    def _scale_X_by_weights_and_frequency(
        self,
        X: NDArray[np.double],
        weights: NDArray[np.double],
        freqs_factor: NDArray[np.double],
        num_band: int,
    ) -> None:
        for i, f in enumerate(freqs_factor):
            X[i] *= weights[i]
            for j in range(num_band):
                X[i, j] *= f[j]

    def _get_Y(
        self,
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
        X: NDArray[np.double],
    ) -> NDArray[np.double]:
        r"""Calculate Y = (\Omega^-1, X)."""
        solver = self._get_colmat_solver()
        num_grid_points, size = self._get_Y_problem_size()
        v = self._get_Y_solver_matrix(i_sigma, i_temp, size, solver)

        start = time.time()
        self._show_Y_solver_log(i_sigma, i_temp, solver)
        Y = self._solve_Y_by_solver(solver, v, X, i_sigma, i_temp)

        self._set_f_vectors(Y, num_grid_points, weights)

        if self._log_level and solver != 7:
            print("[%.3fs]" % (time.time() - start), flush=True)
            sys.stdout.flush()

        return Y

    def _get_colmat_solver(self) -> int:
        solver = select_colmat_solver(self._pinv_solver)
        if self._pinv_solver == 6:
            return 6
        return solver

    def _get_Y_problem_size(self) -> tuple[int, int]:
        num_band = len(self._pp.primitive) * 3
        if self._is_reducible_collision_matrix:
            num_grid_points = int(np.prod(self._pp.mesh_numbers))
            size = num_grid_points * num_band
        else:
            num_grid_points = len(self._ir_grid_points)
            size = num_grid_points * num_band * 3
        return num_grid_points, size

    def _get_Y_solver_matrix(
        self, i_sigma: int, i_temp: int, size: int, solver: int
    ) -> NDArray[np.double]:
        assert self._collision_matrix is not None
        v = self._collision_matrix[i_sigma, i_temp].reshape(size, size)
        if solver in [1, 2, 4, 5]:
            return v.T
        return v

    def _show_Y_solver_log(self, i_sigma: int, i_temp: int, solver: int) -> None:
        if not self._log_level or solver == 7:
            return

        assert self._collision_eigenvalues is not None
        eig_str = "abs(eig)" if self._pinv_method == 0 else "eig"
        w = self._collision_eigenvalues[i_sigma, i_temp]
        null_space = (np.abs(w) < self._pinv_cutoff).sum()
        print(
            f"Pinv by ignoring {null_space}/{len(w)} dims "
            f"under {eig_str}<{self._pinv_cutoff:<.1e}",
            end="",
        )

    def _solve_Y_by_solver(
        self,
        solver: int,
        v: NDArray[np.double],
        X: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
    ) -> NDArray[np.double]:
        if solver in [0, 1, 2, 3, 4, 5]:
            return self._solve_Y_with_eigendecomposition(v, X, i_sigma, i_temp)
        if solver == 6:
            return self._solve_Y_with_builtin_pinv(v, X, i_sigma, i_temp)
        if solver == 7:
            return self._solve_Y_with_direct_pinv(v, X)
        raise ValueError(f"Unknown collision matrix solver {solver}")

    def _solve_Y_with_eigendecomposition(
        self,
        v: NDArray[np.double],
        X: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
    ) -> NDArray[np.double]:
        if self._log_level:
            print(" (np.dot) ", end="")
            sys.stdout.flush()

        e = self._get_eigvals_pinv(i_sigma, i_temp)
        if self._is_reducible_collision_matrix:
            X1 = np.dot(v.T, X)
            for i in range(3):
                X1[:, i] *= e
            return np.dot(v, X1)
        return np.dot(v, e * np.dot(v.T, X.ravel())).reshape(-1, 3)

    def _solve_Y_with_builtin_pinv(
        self,
        v: NDArray[np.double],
        X: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
    ) -> NDArray[np.double]:
        import phono3py._phono3py as phono3c

        if self._log_level:
            print(" (built-in-pinv) ", end="", flush=True)

        assert self._collision_eigenvalues is not None
        assert self._collision_matrix is not None
        w = self._collision_eigenvalues[i_sigma, i_temp]
        phono3c.pinv_from_eigensolution(
            self._collision_matrix,
            w,
            i_sigma,
            i_temp,
            self._pinv_cutoff,
            self._pinv_method,
        )
        return self._solve_Y_with_direct_pinv(v, X)

    def _solve_Y_with_direct_pinv(
        self, v: NDArray[np.double], X: NDArray[np.double]
    ) -> NDArray[np.double]:
        if self._is_reducible_collision_matrix:
            return np.dot(v, X)
        return np.dot(v, X.ravel()).reshape(-1, 3)

    def _set_f_vectors(
        self, Y: NDArray[np.double], num_grid_points: int, weights: NDArray[np.double]
    ) -> None:
        """Calculate f-vectors.

        Collision matrix is half of that defined in Chaput's paper.
        Therefore Y is divided by 2.

        """
        assert self._f_vectors is not None
        num_band = len(self._pp.primitive) * 3
        self._f_vectors[:] = (
            (Y / 2).reshape(num_grid_points, num_band * 3).T / weights
        ).T.reshape(self._f_vectors.shape)

    def _get_eigvals_pinv(self, i_sigma: int, i_temp: int) -> NDArray[np.double]:
        """Return inverse eigenvalues of eigenvalues > epsilon."""
        assert self._collision_eigenvalues is not None
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

    def _get_I(
        self, a: int, b: int, size: int, plus_transpose: bool = True
    ) -> NDArray[np.double] | None:
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

        # Same as np.kron(np.eye(size), r_sum), but written as below
        # to be sure the values in memory C-contiguous with 'double'.
        I_mat = np.zeros((3 * size, 3 * size), dtype="double", order="C")
        for i in range(size):
            I_mat[(i * 3) : ((i + 1) * 3), (i * 3) : ((i + 1) * 3)] = r_sum

        return I_mat

    @abstractmethod
    def _set_kappa_RTA(
        self, i_sigma: int, i_temp: int, weights: NDArray[np.double]
    ) -> None:
        raise NotImplementedError()

    def _set_kappa_at_sigmas_common(self, weights: NDArray[np.double]) -> None:
        """Run common conductivity loop over sigma and temperature."""
        assert self._temperatures is not None
        for i_sigma, sigma in enumerate(self._sigmas):
            if self._log_level:
                self._show_kappa_sigma_header(sigma)

            for i_temp, temperature in enumerate(self._temperatures):
                self._set_kappa_at_sigma_and_temperature_common(
                    i_sigma,
                    i_temp,
                    temperature,
                    weights,
                )

        if self._log_level:
            print("", flush=True)

    def _show_kappa_sigma_header(self, sigma: float | None) -> None:
        text = "----------- Thermal conductivity (W/m-k) "
        if sigma:
            text += "for sigma=%s -----------" % sigma
        else:
            text += "with tetrahedron method -----------"
        print(text, flush=True)

    def _set_kappa_at_sigma_and_temperature_common(
        self, i_sigma: int, i_temp: int, temperature: float, weights: NDArray[np.double]
    ) -> None:
        """Run common per-(sigma, temperature) conductivity workflow."""
        if temperature <= 0:
            return

        self._set_kappa_RTA(i_sigma, i_temp, weights)
        assert self._collision_matrix is not None
        w = diagonalize_collision_matrix(
            self._collision_matrix,
            i_sigma=i_sigma,
            i_temp=i_temp,
            pinv_solver=self._pinv_solver,
            log_level=self._log_level,
        )
        if w is not None:
            assert self._collision_eigenvalues is not None
            self._collision_eigenvalues[i_sigma, i_temp] = w

        self._set_kappa(i_sigma, i_temp, weights)

        if self._log_level:
            self._show_kappa_at_temperature(i_sigma, i_temp, temperature)

    @abstractmethod
    def _show_kappa_at_temperature(
        self, i_sigma: int, i_temp: int, temperature: float
    ) -> None:
        raise NotImplementedError()

    def _set_kappa_by_collision_type(
        self,
        kappa: NDArray[np.double],
        mode_kappa: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
    ) -> None:
        """Dispatch kappa calculation by collision-matrix representation."""
        if self._is_reducible_collision_matrix:
            self._set_kappa_reducible_colmat(
                kappa,
                mode_kappa,
                i_sigma,
                i_temp,
                weights,
            )
        else:
            self._set_kappa_ir_colmat(
                kappa,
                mode_kappa,
                i_sigma,
                i_temp,
                weights,
            )

    def _set_kappa_RTA_by_collision_type(
        self,
        kappa_RTA: NDArray[np.double],
        mode_kappa_RTA: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
    ) -> None:
        """Dispatch RTA kappa calculation by collision-matrix representation."""
        if self._is_reducible_collision_matrix:
            self._set_kappa_RTA_reducible_colmat(
                kappa_RTA,
                mode_kappa_RTA,
                i_sigma,
                i_temp,
                weights,
            )
        else:
            self._set_kappa_RTA_ir_colmat(
                kappa_RTA,
                mode_kappa_RTA,
                i_sigma,
                i_temp,
                weights,
            )

    def _set_kappa_RTA_ir_colmat(
        self,
        kappa_RTA: NDArray[np.double],
        mode_kappa_RTA: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
    ) -> None:
        """Calculate RTA thermal conductivity.

        This RTA is supposed to be the same as conductivity_RTA.

        """
        X = self._get_X(i_temp, weights)
        Y = self._build_rta_Y_ir_colmat(i_sigma, i_temp, X)
        num_ir_grid_points = len(self._ir_grid_points)

        self._set_mode_kappa_and_accumulate(
            kappa_RTA,
            mode_kappa_RTA,
            X,
            Y,
            num_ir_grid_points,
            i_sigma,
            i_temp,
        )

    def _set_kappa_RTA_reducible_colmat(
        self,
        kappa_RTA: NDArray[np.double],
        mode_kappa_RTA: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
    ) -> None:
        """Calculate RTA thermal conductivity.

        This RTA is not equivalent to conductivity_RTA.
        The lifetime is defined from the diagonal part of collision matrix.

        `kappa` and `mode_kappa` are overwritten.

        """
        X = self._get_X(i_temp, weights)
        num_mesh_points = int(np.prod(self._pp.mesh_numbers))
        Y = self._build_rta_Y_reducible_colmat(i_sigma, i_temp, X)
        rotation_norm = float(len(self._rotations_cartesian))
        self._set_mode_kappa_and_accumulate(
            kappa_RTA,
            mode_kappa_RTA,
            X,
            Y,
            num_mesh_points,
            i_sigma,
            i_temp,
            rotation_normalization=rotation_norm,
        )

    def _build_rta_Y_ir_colmat(
        self, i_sigma: int, i_temp: int, X: NDArray[np.double]
    ) -> NDArray[np.double]:
        assert self._frequencies is not None
        Y = np.zeros_like(X)
        num_band = len(self._pp.primitive) * 3
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
        return Y

    def _build_rta_Y_reducible_colmat(
        self, i_sigma: int, i_temp: int, X: NDArray[np.double]
    ) -> NDArray[np.double]:
        assert self._frequencies is not None
        assert self._collision_matrix is not None
        num_band = len(self._pp.primitive) * 3
        num_mesh_points = int(np.prod(self._pp.mesh_numbers))
        size = num_mesh_points * num_band
        v_diag = np.diagonal(
            self._collision_matrix[i_sigma, i_temp].reshape(size, size)
        )
        Y = np.zeros_like(X)
        for gp in range(num_mesh_points):
            frequencies = self._frequencies[gp]
            for j, f in enumerate(frequencies):
                if f > self._pp.cutoff_frequency:
                    i_mode = gp * num_band + j
                    Y[i_mode, :] = X[i_mode, :] / v_diag[i_mode]
        return Y

    def _set_mode_kappa_and_accumulate(
        self,
        kappa: NDArray[np.double],
        mode_kappa: NDArray[np.double],
        X: NDArray[np.double],
        Y: NDArray[np.double],
        num_grid_points: int,
        i_sigma: int,
        i_temp: int,
        rotation_normalization: float = 1.0,
    ) -> None:
        """Set mode kappa and accumulate into kappa."""
        self._set_mode_kappa(
            mode_kappa,
            X,
            Y,
            num_grid_points,
            i_sigma,
            i_temp,
        )
        if rotation_normalization != 1.0:
            mode_kappa[i_sigma, i_temp] /= rotation_normalization
        self._accumulate_kappa_from_mode_kappa(kappa, mode_kappa, i_sigma, i_temp)

    def _accumulate_kappa_from_mode_kappa(
        self,
        kappa: NDArray[np.double],
        mode_kappa: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
    ) -> None:
        n_sampling = self.number_of_sampling_grid_points
        kappa[i_sigma, i_temp] = (
            mode_kappa[i_sigma, i_temp].sum(axis=0).sum(axis=0) / n_sampling
        )

    def _set_mode_kappa(
        self,
        mode_kappa: NDArray[np.double],
        X: NDArray[np.double],
        Y: NDArray[np.double],
        num_grid_points: int,
        i_sigma: int,
        i_temp: int,
    ) -> None:
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
                strict=True,
            )
        ):
            for j, (v, f) in enumerate(zip(v_gp, f_gp, strict=True)):
                # Do not consider three lowest modes at Gamma-point
                # It is assumed that there are no imaginary modes.
                if (self._pp.bz_grid.addresses[i] == 0).all() and j < 3:
                    continue

                sum_k = self._compute_mode_kappa_tensor(v, f)
                sum_k = sum_k + sum_k.T
                for k, vxf in enumerate(
                    ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
                ):
                    mode_kappa[i_sigma, i_temp, i, j, k] = sum_k[vxf]

        assert self._temperatures is not None
        t = self._temperatures[i_temp]
        mode_kappa[i_sigma, i_temp] *= (
            self._conversion_factor * get_physical_units().KB * t**2
        )

    def _compute_mode_kappa_tensor(
        self,
        v: NDArray[np.double],
        f: NDArray[np.double],
    ) -> NDArray[np.double]:
        # if rotations_cartesian is None:
        #     return np.outer(v, f)
        return self._compute_mode_kappa_tensor_with_rotations(v, f)

    def _compute_mode_kappa_tensor_with_rotations(
        self,
        v: NDArray[np.double],
        f: NDArray[np.double],
    ) -> NDArray[np.double]:
        sum_k = np.zeros((3, 3), dtype="double")
        for r in self._rotations_cartesian:
            sum_k += np.outer(np.dot(r, v), np.dot(r, f))
        return sum_k

    def _set_mode_kappa_Chaput(
        self,
        mode_kappa: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
    ) -> None:
        """Calculate mode kappa by the way in Laurent Chaput's PRL paper.

        This gives the different result from _set_mode_kappa and requires more
        memory space.

        """
        X, num_ir_grid_points, num_band, solver, v = self._prepare_chaput_inputs(
            i_sigma, i_temp, weights
        )
        assert self._temperatures is not None
        t = self._temperatures[i_temp]

        omega_inv = self._build_chaput_omega_inverse(v, i_sigma, i_temp)
        self._set_f_vectors(np.dot(omega_inv, X), num_ir_grid_points, weights)
        elems = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        for i, vxf in enumerate(elems):
            mode_kappa[i_sigma, i_temp, :, :, i] = 0
            self._accumulate_chaput_mode_kappa_component(
                mode_kappa,
                i_sigma,
                i_temp,
                i,
                vxf,
                omega_inv,
                X,
                num_ir_grid_points,
                num_band,
                solver,
            )

        factor = self._conversion_factor * get_physical_units().KB * t**2
        mode_kappa[i_sigma, i_temp] *= factor

    def _prepare_chaput_inputs(
        self, i_sigma: int, i_temp: int, weights: NDArray[np.double]
    ) -> tuple[NDArray[np.double], int, int, int, NDArray[np.double]]:
        assert self._collision_matrix is not None
        X = self._get_X(i_temp, weights).ravel()
        num_ir_grid_points = len(self._ir_grid_points)
        num_band = len(self._pp.primitive) * 3
        size = num_ir_grid_points * num_band * 3
        v = self._collision_matrix[i_sigma, i_temp].reshape(size, size)
        solver = select_colmat_solver(self._pinv_solver)
        if solver in [1, 2, 4, 5]:
            v = v.T
        return X, num_ir_grid_points, num_band, solver, v

    def _build_chaput_omega_inverse(
        self, v: NDArray[np.double], i_sigma: int, i_temp: int
    ) -> NDArray[np.double]:
        e = self._get_eigvals_pinv(i_sigma, i_temp)
        omega_inv = np.empty(v.shape, dtype="double", order="C")
        np.dot(v, (e * v).T, out=omega_inv)
        return omega_inv

    def _accumulate_chaput_mode_kappa_component(
        self,
        mode_kappa: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
        i_elem: int,
        vxf: tuple[int, int],
        omega_inv: NDArray[np.double],
        X: NDArray[np.double],
        num_ir_grid_points: int,
        num_band: int,
        solver: int,
    ) -> None:
        mat = self._get_I(vxf[0], vxf[1], num_ir_grid_points * num_band)
        if mat is None:
            return

        np.dot(mat, omega_inv, out=mat)
        w = diagonalize_collision_matrix(
            mat,
            pinv_solver=self._pinv_solver,
            log_level=self._log_level,
        )
        if solver in [1, 2, 4, 5]:
            mat = mat.T
        spectra = np.dot(mat.T, X) ** 2 * w
        for s, eigvec in zip(spectra, mat.T, strict=True):
            vals = s * (eigvec**2).reshape(-1, 3).sum(axis=1)
            vals = vals.reshape(num_ir_grid_points, num_band)
            mode_kappa[i_sigma, i_temp, :, :, i_elem] += vals

    def _set_mode_kappa_from_mfp(
        self,
        weights: NDArray[np.double],
        i_sigma: int,
        i_temp: int,
    ) -> None:
        assert self._mfp is not None
        assert self._cv is not None
        for i, (v_gp, mfp_gp, cv_gp) in enumerate(
            zip(self._gv, self._mfp[i_sigma, i_temp], self._cv[i_temp], strict=True)  # type: ignore[attr-defined]
        ):
            for j, (v, mfp, cv) in enumerate(zip(v_gp, mfp_gp, cv_gp, strict=True)):
                sum_k = np.zeros((3, 3), dtype="double")
                for r in self._rotations_cartesian:
                    sum_k += np.outer(np.dot(r, v), np.dot(r, mfp))
                sum_k = (sum_k + sum_k.T) / 2 * cv * weights[i] ** 2 * 2 * np.pi
                for k, vxf in enumerate(
                    ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
                ):
                    self._mode_kappa[i_sigma, i_temp, i, j, k] = sum_k[vxf]  # type: ignore[attr-defined]
        self._mode_kappa *= -self._conversion_factor  # type: ignore[attr-defined]

    def _set_mean_free_path(
        self,
        i_sigma: int,
        i_temp: int,
        weights: NDArray[np.double],
        Y: NDArray[np.double],
    ) -> None:
        assert self._temperatures is not None
        assert self._f_vectors is not None
        assert self._mfp is not None
        t = self._temperatures[i_temp]
        # shape = (num_grid_points, num_band, 3),
        for i, f_gp in enumerate(self._f_vectors):
            for j, f in enumerate(f_gp):
                cv = self._conductivity_components.mode_heat_capacities[i_temp, i, j]
                if cv < 1e-10:
                    continue
                self._mfp[i_sigma, i_temp, i, j] = (
                    -2 * t * np.sqrt(get_physical_units().KB / cv) * f / (2 * np.pi)
                )

    def _show_log(self, i: int) -> None:
        assert self._frequencies is not None
        gp = self._grid_points[i]
        frequencies = self._frequencies[gp]
        if self._is_reducible_collision_matrix:
            gv = self._conductivity_components.group_velocities[
                self._pp.bz_grid.bzg2grg[gp]
            ]
        else:
            gv = self._conductivity_components.group_velocities[i]
        if self._is_full_pp:
            assert self._averaged_pp_interaction is not None
            ave_pp = self._averaged_pp_interaction[i]
            text = "Frequency     group velocity (x, y, z)     |gv|       Pqj"
        else:
            text = "Frequency     group velocity (x, y, z)     |gv|"

        if self._conductivity_components.gv_delta_q is None:
            pass
        else:
            text += "  (dq=%3.1e)" % self._conductivity_components.gv_delta_q
        print(text)
        if self._is_full_pp:
            for f, v, pp in zip(frequencies, gv, ave_pp, strict=True):
                print(
                    "%8.3f   (%8.3f %8.3f %8.3f) %8.3f %11.3e"
                    % (f, v[0], v[1], v[2], np.linalg.norm(v), pp)
                )
        else:
            for f, v in zip(frequencies, gv, strict=True):
                print(
                    "%8.3f   (%8.3f %8.3f %8.3f) %8.3f"
                    % (f, v[0], v[1], v[2], np.linalg.norm(v))
                )

        sys.stdout.flush()

    def _py_symmetrize_collision_matrix(self) -> None:
        num_band = len(self._pp.primitive) * 3
        num_ir_grid_points = len(self._ir_grid_points)
        for i in range(num_ir_grid_points):
            for j in range(num_band):
                for k in range(3):
                    for ll in range(num_ir_grid_points):
                        for m in range(num_band):
                            for n in range(3):
                                self._py_set_symmetrized_element(i, j, k, ll, m, n)

    def _py_set_symmetrized_element(
        self, i: int, j: int, k: int, ll: int, m: int, n: int
    ) -> None:
        assert self._collision_matrix is not None
        sym_val = (
            self._collision_matrix[:, :, i, j, k, ll, m, n]
            + self._collision_matrix[:, :, ll, m, n, i, j, k]
        ) / 2
        self._collision_matrix[:, :, i, j, k, ll, m, n] = sym_val
        self._collision_matrix[:, :, ll, m, n, i, j, k] = sym_val

    def _py_symmetrize_collision_matrix_no_kappa_stars(self) -> None:
        num_band = len(self._pp.primitive) * 3
        num_ir_grid_points = len(self._ir_grid_points)
        for i in range(num_ir_grid_points):
            for j in range(num_band):
                for k in range(num_ir_grid_points):
                    for ll in range(num_band):
                        self._py_set_symmetrized_element_no_kappa_stars(i, j, k, ll)

    def _py_set_symmetrized_element_no_kappa_stars(
        self, i: int, j: int, k: int, ll: int
    ) -> None:
        assert self._collision_matrix is not None
        sym_val = (
            self._collision_matrix[:, :, i, j, k, ll]
            + self._collision_matrix[:, :, k, ll, i, j]
        ) / 2
        self._collision_matrix[:, :, i, j, k, ll] = sym_val
        self._collision_matrix[:, :, k, ll, i, j] = sym_val


def diagonalize_collision_matrix(
    collision_matrices: NDArray[np.double],
    i_sigma: int | None = None,
    i_temp: int | None = None,
    pinv_solver: int = 0,
    log_level: int = 0,
) -> NDArray[np.double] | None:
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
        When pinv_solve==7, None is returned.

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
    trace = np.trace(collision_matrices[i_sigma, i_temp].reshape(size, size))

    # [1] dsyev: safer and slower than dsyevd and smallest memory usage
    # [2] dsyevd: faster than dsyev and largest memory usage
    if solver in [1, 2]:
        if log_level:
            routine = ["dsyev", "dsyevd"][solver - 1]
            print("Diagonalizing by lapacke %s ... " % routine, end="", flush=True)
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
            print("Diagonalize by np.linalg.eigh ", end="", flush=True)
        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        w, col_mat[:] = np.linalg.eigh(col_mat)  # type: ignore[assignment]

    elif solver == 4:  # fully scipy dsyev
        if log_level:
            print("Diagonalize by scipy.linalg.lapack.dsyev ", end="", flush=True)
        import scipy.linalg

        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        w, _, info = scipy.linalg.lapack.dsyev(col_mat.T, overwrite_a=1)  # type: ignore
    elif solver == 5:  # fully scipy dsyevd
        if log_level:
            print("Diagnalize by scipy.linalg.lapack.dsyevd ", end="", flush=True)
        import scipy.linalg

        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        w, _, info = scipy.linalg.lapack.dsyevd(col_mat.T, overwrite_a=1)  # type: ignore
    elif solver == 7:
        if log_level:
            print(
                "Pseudo inversion using np.linalg.pinv(a, hermitian=False) ",
                end="",
                flush=True,
            )
        col_mat = collision_matrices[i_sigma, i_temp].reshape(size, size)
        # hermitian=True calls eigh, which is not what we want.
        col_mat[:, :] = np.linalg.pinv(col_mat, hermitian=False)
        w = None

    if log_level:
        if w is not None:
            print(f"sum={w.sum():<.1e} d={trace - w.sum():<.1e} ", end="")
        print("[%.3fs]" % (time.time() - start), flush=True)

    return w
