"""Calculate ph-ph interaction and phonons on grid."""

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

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, get_dynamical_matrix
from phonopy.physical_units import get_physical_units
from phonopy.structure.cells import Primitive, compute_all_sg_permutations
from phonopy.structure.symmetry import Symmetry

from phono3py.phonon.grid import (
    BZGrid,
    get_grid_points_by_rotations,
    get_ir_grid_points,
)
from phono3py.phonon.solver import run_phonon_solver_c, run_phonon_solver_py
from phono3py.phonon3.real_to_reciprocal import RealToReciprocal
from phono3py.phonon3.reciprocal_to_normal import ReciprocalToNormal
from phono3py.phonon3.triplets import get_nosym_triplets_at_q, get_triplets_at_q


class Interaction:
    """Calculate ph-ph interaction and phonons on grid.

    This class instance is the heart of phono3py calculation.
    Many data are stored.

    The following three steps have to be done manually.
    1) init_dynamical_matrix
    2) set_grid_point
    3) run

    Attributes
    ----------
    interaction_strength
    mesh_numbers
    is_mesh_symmetry
    fc3
    dynamical_matrix
    primitive
    primitive_symmetry
    bz_grid
    band_indices
    nac_params
    nac_q_direction
    zero_value_positions
    frequency_factor_to_THz
    lapack_zheev_uplo
    cutoff_frequency
    symmetrize_fc3q
    make_r0_average

    """

    def __init__(
        self,
        primitive: Primitive,
        bz_grid: BZGrid,
        primitive_symmetry: Symmetry,
        fc3: NDArray | None = None,
        fc3_nonzero_indices: NDArray | None = None,
        band_indices: NDArray | Sequence | None = None,
        constant_averaged_interaction: float | None = None,
        frequency_factor_to_THz: float | None = None,
        frequency_scale_factor: float | None = None,
        unit_conversion: float | None = None,
        is_mesh_symmetry: bool = True,
        symmetrize_fc3q: bool = False,
        make_r0_average: bool = False,
        cutoff_frequency: float | None = None,
        lapack_zheev_uplo: Literal["L", "U"] = "L",
        openmp_per_triplets: bool | None = None,
    ):
        """Init method."""
        self._primitive = primitive
        self._bz_grid = bz_grid
        self._primitive_symmetry = primitive_symmetry

        self._band_indices = self._get_band_indices(band_indices)
        self._constant_averaged_interaction = constant_averaged_interaction
        if frequency_factor_to_THz is None:
            self._frequency_factor_to_THz = get_physical_units().DefaultToTHz
        else:
            self._frequency_factor_to_THz = frequency_factor_to_THz
        self._frequency_scale_factor = frequency_scale_factor

        if fc3 is not None:
            self._set_fc3(fc3, fc3_nonzero_indices=fc3_nonzero_indices)

        # Unit to eV^2
        if unit_conversion is None:
            num_grid = np.prod(self.mesh_numbers)
            self._unit_conversion = float(
                (get_physical_units().Hbar * get_physical_units().EV) ** 3
                / 36
                / 8
                * get_physical_units().EV ** 2
                / get_physical_units().Angstrom ** 6
                / (2 * np.pi * get_physical_units().THz) ** 3
                / get_physical_units().AMU ** 3
                / num_grid
                / get_physical_units().EV ** 2
            )
        else:
            self._unit_conversion = unit_conversion
        if cutoff_frequency is None:
            self._cutoff_frequency = 0
        else:
            self._cutoff_frequency = cutoff_frequency
        self._is_mesh_symmetry = is_mesh_symmetry
        self._symmetrize_fc3q = symmetrize_fc3q
        self._make_r0_average = make_r0_average
        self._lapack_zheev_uplo: Literal["L", "U"] = lapack_zheev_uplo
        self._openmp_per_triplets = openmp_per_triplets

        self._symprec = self._primitive_symmetry.tolerance

        self._triplets_at_q = None
        self._weights_at_q = None
        self._triplets_map_at_q = None
        self._ir_map_at_q = None
        self._interaction_strength = None
        self._g_zero = None

        self._phonon_done = None
        self._phonon_all_done = False
        self._done_nac_at_gamma = False  # Phonon at Gamma is calculated with NAC.
        self._frequencies = None
        self._eigenvectors = None
        self._frequencies_at_gamma = None
        self._eigenvectors_at_gamma = None
        self._dm = None
        self._nac_params = None
        self._nac_q_direction = None

        self._band_index_count = 0

        self._svecs, self._multi = self._primitive.get_smallest_vectors()
        self._masses = np.array(self._primitive.masses, dtype="double")
        self._p2s = np.array(self._primitive.p2s_map, dtype="int64")
        self._s2p = np.array(self._primitive.s2p_map, dtype="int64")
        n_satom, n_patom, _ = self._multi.shape
        self._all_shortest = np.zeros(
            (n_patom, n_satom, n_satom), dtype="byte", order="C"
        )
        self._get_all_shortest()

    def run(self, lang: Literal["C", "Python"] = "C", g_zero: NDArray | None = None):
        """Run ph-ph interaction calculation."""
        if self._phonon_all_done:
            self.run_phonon_solver()

        if self._triplets_at_q is None:
            raise RuntimeError("Set grid point first by set_grid_point().")

        num_band = len(self._primitive) * 3
        num_triplets = len(self._triplets_at_q)

        self._interaction_strength = np.empty(
            (num_triplets, len(self._band_indices), num_band, num_band), dtype="double"
        )
        if self._constant_averaged_interaction is None:
            self._interaction_strength[:] = 0
            if lang == "C":
                self._run_c(g_zero)
            else:
                self._run_py()
        else:
            num_grid = np.prod(self.mesh_numbers)
            self._interaction_strength[:] = (
                self._constant_averaged_interaction / num_grid
            )

    @property
    def interaction_strength(self) -> NDArray | None:
        """Return ph-ph interaction strength.

        Returns
        -------
        ndarray
            shape=(num_ir_grid_points, num_specified_band, num_band, num_band),
            dtype='double', order='C'

        """
        return self._interaction_strength

    @property
    def mesh_numbers(self) -> NDArray:
        """Return mesh numbers.

        Returns
        -------
        ndarray
           shape=(3, ), dtype='int64'

        """
        return self._bz_grid.D_diag

    @property
    def is_mesh_symmetry(self) -> bool:
        """Whether symmetry of grid is utilized or not."""
        return self._is_mesh_symmetry

    @property
    def fc3(self) -> NDArray:
        """Return fc3."""
        return self._fc3

    @property
    def fc3_nonzero_indices(self) -> NDArray:
        """Return fc3_nonzero_indices."""
        return self._fc3_nonzero_indices

    @property
    def dynamical_matrix(self) -> DynamicalMatrix | None:
        """Return DynamicalMatrix class instance."""
        return self._dm

    @property
    def primitive(self) -> Primitive:
        """Return Primitive class instance."""
        return self._primitive

    @property
    def primitive_symmetry(self) -> Symmetry:
        """Return Symmetry class instance of primitive cell."""
        return self._primitive_symmetry

    @property
    def phonon_all_done(self) -> bool:
        """Return whether phonon calculation is done at all grid points.

        True value is set in run_phonon_solver(). Even when False, it is
        possible that all phonons are already calculated, but it is safer to
        think some of phonons are not calculated yet. To be sure, check
        (self._phonon_done == 0).any().

        """
        return self._phonon_all_done

    def get_triplets_at_q(
        self,
    ) -> tuple[
        NDArray | None,
        NDArray | None,
        NDArray | None,
        NDArray | None,
    ]:
        """Return grid point triplets information.

        triplets_at_q is in BZ-grid.
        triplets_map_at_q is in GR-grid.
        ir_map_at_q is in GR-grid.
        See details at ``get_triplets_at_q``.

        """
        return (
            self._triplets_at_q,
            self._weights_at_q,
            self._triplets_map_at_q,
            self._ir_map_at_q,
        )

    @property
    def bz_grid(self) -> BZGrid:
        """Return BZGrid class instance."""
        return self._bz_grid

    @property
    def band_indices(self) -> NDArray:
        """Return band indices.

        Returns
        -------
        ndarray
            shape=(num_specified_bands, ), dtype='int64'

        """
        return self._band_indices

    @property
    def nac_params(self) -> dict | None:
        """Return NAC params."""
        return self._nac_params

    @property
    def nac_q_direction(self) -> NDArray | None:
        """Return q-direction used for NAC at q->0.

        Direction of q-vector watching from Gamma point used for
        non-analytical term correction. This is effective only at q=0
        (physically q->0). The direction is given in crystallographic
        (fractional) coordinates.
        shape=(3,), dtype='double'.
        Default value is None, which means this feature is not used.

        """
        return self._nac_q_direction

    @nac_q_direction.setter
    def nac_q_direction(self, nac_q_direction):
        if nac_q_direction is None:
            self._nac_q_direction = None
        else:
            self._nac_q_direction = np.array(nac_q_direction, copy=True, dtype="double")

    @property
    def zero_value_positions(self) -> NDArray | None:
        """Return zero ph-ph interaction elements information.

        Returns
        -------
        shape is same as that of interaction_strength, dtype='byte', order='C'

        """
        return self._g_zero

    def get_phonons(
        self,
    ) -> tuple[NDArray | None, NDArray | None, NDArray | None]:
        """Return phonons on grid.

        Returns
        -------
        tuple
            frequencies : ndarray
                Phonon frequencies on grid.
                shape=(num_bz_grid, num_band), dtype='double', order='C'
            eigenvectors : ndarray
                Phonon eigenvectors on grid.
                shape=(num_bz_grid, num_band, num_band),
                dtype="c%d" % (np.dtype('double').itemsize * 2), order='C'
            phonon_done : ndarray
                1 if phonon at a grid point is calculated, otherwise 0.
                shape=(num_bz_grid, ), dtype='byte'

        """
        return self._frequencies, self._eigenvectors, self._phonon_done

    @property
    def frequency_factor_to_THz(self) -> float:
        """Return phonon frequency conversion factor to THz."""
        return self._frequency_factor_to_THz

    @property
    def lapack_zheev_uplo(self) -> Literal["L", "U"]:
        """Return U or L for lapack zheev solver."""
        return self._lapack_zheev_uplo

    @property
    def cutoff_frequency(self) -> float:
        """Return cutoff phonon frequency to judge imaginary phonon."""
        return self._cutoff_frequency

    @property
    def openmp_per_triplets(self) -> bool | None:
        """Return whether OpenMP distribution over triplets or bands."""
        return self._openmp_per_triplets

    @property
    def symmetrize_fc3q(self) -> bool:
        """Return boolean of symmetrize_fc3q."""
        return self._symmetrize_fc3q

    @property
    def make_r0_average(self) -> bool:
        """Return boolean of make_r0_average.

        This flag is used to activate averaging of fc3 transformation
        from real space to reciprocal space around three atoms. With False,
        it is done at the first atom. With True, it is done at three atoms
        and averaged.

        """
        return self._make_r0_average

    @property
    def all_shortest(self) -> NDArray:
        """Return boolean of make_r0_average.

        This flag is used to activate averaging of fc3 transformation
        from real space to reciprocal space around three atoms. With False,
        it is done at the first atom. With True, it is done at three atoms
        and averaged.

        """
        return self._all_shortest

    @property
    def averaged_interaction(self) -> NDArray:
        """Return sum over phonon triplets of interaction strength.

        See Eq.(21) of PRB 91, 094306 (2015)

        """
        if self._interaction_strength is None or self._weights_at_q is None:
            raise RuntimeError(
                "Interaction strength and weights at q are not set. "
                "Run Interaction.run() first."
            )

        # v[triplet, band0, band, band]
        v = self._interaction_strength
        w = self._weights_at_q
        v_sum = np.dot(w, v.sum(axis=2).sum(axis=2))
        return v_sum / np.prod(v.shape[2:])

    def get_primitive_and_supercell_correspondence(
        self,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """Return atomic pair information."""
        return (self._svecs, self._multi, self._p2s, self._s2p, self._masses)

    @property
    def unit_conversion_factor(self) -> float:
        """Return unit conversion factor."""
        return self._unit_conversion

    @property
    def constant_averaged_interaction(self) -> float | None:
        """Return constant averaged interaction."""
        return self._constant_averaged_interaction

    def set_interaction_strength(self, pp_strength, g_zero=None):
        """Set interaction strength."""
        self._interaction_strength = pp_strength
        self._g_zero = g_zero

    def set_grid_point(self, grid_point, store_triplets_map=False):
        """Set grid point and prepare grid point triplets."""
        if not self._is_mesh_symmetry:
            (
                triplets_at_q,
                weights_at_q,
                triplets_map_at_q,
                ir_map_at_q,
            ) = get_nosym_triplets_at_q(grid_point, self._bz_grid)
        else:
            # Special treatment of symmetry is applied when q_direction is used
            # at Gamma point = (0 0 0).
            if (
                self._bz_grid.addresses[grid_point] == 0
            ).all() and self._nac_q_direction is not None:
                rotations = []
                for i, r in enumerate(self._bz_grid.reciprocal_operations):
                    dq = self._nac_q_direction
                    dq /= np.linalg.norm(dq)
                    diff = np.dot(r, dq) - dq
                    if (abs(diff) < 1e-5).all():
                        rotations.append(self._bz_grid.rotations[i])
                (
                    triplets_at_q,
                    weights_at_q,
                    triplets_map_at_q,
                    ir_map_at_q,
                ) = get_triplets_at_q(
                    grid_point,
                    self._bz_grid,
                    reciprocal_rotations=rotations,
                    is_time_reversal=False,
                )
            else:
                (
                    triplets_at_q,
                    weights_at_q,
                    triplets_map_at_q,
                    ir_map_at_q,
                ) = get_triplets_at_q(grid_point, self._bz_grid)

            # Re-calculate phonon at Gamma-point when q-direction is given.
            if (self._bz_grid.addresses[grid_point] == 0).all():
                self.run_phonon_solver_at_gamma(is_nac=True)
            elif self._done_nac_at_gamma:
                if self._nac_q_direction is None:
                    self.run_phonon_solver_at_gamma()
                else:
                    msg = (
                        "Phonons at Gamma has been calculated with NAC, "
                        "but ph-ph interaction is expected to calculate at "
                        "non-Gamma point. Setting Interaction.nac_q_direction = "
                        "None, can avoid raising this exception to re-run phonon "
                        "calculation at Gamma without NAC."
                    )
                    raise RuntimeError(msg)

        reciprocal_lattice = np.linalg.inv(self._primitive.cell)
        for triplet in triplets_at_q:
            sum_q = (self._bz_grid.addresses[triplet]).sum(axis=0)
            if (sum_q % self.mesh_numbers != 0).any():
                print("============= Warning ==================")
                print("%s" % triplet)
                for tp in triplet:
                    print(
                        "%s %s"
                        % (
                            self._bz_grid.addresses[tp],
                            np.linalg.norm(
                                np.dot(
                                    reciprocal_lattice,
                                    self._bz_grid.addresses[tp]
                                    / self.mesh_numbers.astype("double"),
                                )
                            ),
                        )
                    )
                print("%s" % sum_q)
                print("============= Warning ==================")

        self._triplets_at_q = triplets_at_q
        self._weights_at_q = weights_at_q

        if store_triplets_map:
            self._triplets_map_at_q = triplets_map_at_q
            self._ir_map_at_q = ir_map_at_q

    def init_dynamical_matrix(
        self,
        fc2,
        supercell,
        primitive,
        nac_params=None,
        decimals=None,
    ):
        """Prepare for phonon calculation on grid.

        solve_dynamical_matrices : bool
           When False, phonon calculation will be postponed.

        """
        self._nac_params = nac_params
        self._dm = get_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=self._frequency_scale_factor,
            decimals=decimals,
        )
        self._allocate_phonon()

    def set_phonon_data(self, frequencies, eigenvectors, bz_grid_addresses):
        """Set phonons on grid."""
        if bz_grid_addresses.shape != self._bz_grid.addresses.shape:
            raise RuntimeError(
                "Input grid address size is inconsistent. Setting phonons failed."
            )

        if (self._bz_grid.addresses - bz_grid_addresses).all():
            raise RuntimeError(
                "Input grid addresses are inconsistent. Setting phonons failed."
            )
        else:
            if (
                self._phonon_done is None
                or self._frequencies is None
                or self._eigenvectors is None
            ):
                raise RuntimeError(
                    "Phonons are not initialized. Call init_dynamical_matrix() first."
                )
            self._phonon_done[:] = 1
            self._frequencies[:] = frequencies
            self._eigenvectors[:] = eigenvectors
            gp_Gamma = self._bz_grid.gp_Gamma
            self._frequencies_at_gamma = self._frequencies[gp_Gamma].copy()
            self._eigenvectors_at_gamma = self._eigenvectors[gp_Gamma].copy()

    def run_phonon_solver(self, grid_points=None, solve_by_rotation=False):
        """Run phonon solver at BZ-grid points."""
        if grid_points is None:
            if solve_by_rotation:
                self.run_phonon_solver_with_eigvec_rotation()
            else:
                self._run_phonon_solver_c(
                    np.arange(len(self._bz_grid.addresses), dtype="int64")
                )
            self._phonon_all_done = True
        else:
            self._run_phonon_solver_c(grid_points)

    def run_phonon_solver_at_gamma(self, is_nac=False):
        """Run phonon solver at Gamma point.

        Run phonon solver at Gamma point with/without NAC. When `self._nac_q_direction`
        is None, always without NAC. `self._nac_q_direction` will be unchanged in any
        case.

        Parameters
        ----------
        is_nac : bool, optional
            With NAC when is_nac is True and `self._nac_q_direction` is not None,
            otherwise without NAC. Default is False.

        """
        if (
            self._phonon_done is None
            or self._frequencies is None
            or self._eigenvectors is None
        ):
            raise RuntimeError(
                "Phonons are not initialized. Call init_dynamical_matrix() first."
            )

        if not is_nac and self._frequencies_at_gamma is not None:
            gp_Gamma = self._bz_grid.gp_Gamma
            self._frequencies[gp_Gamma] = self._frequencies_at_gamma
            self._eigenvectors[gp_Gamma] = self._eigenvectors_at_gamma
            return

        self._phonon_done[self._bz_grid.gp_Gamma] = 0
        if is_nac:
            self._done_nac_at_gamma = True
            self.run_phonon_solver(np.array([self._bz_grid.gp_Gamma], dtype="int64"))
        else:
            self._done_nac_at_gamma = False
            _nac_q_direction = self._nac_q_direction
            self._nac_q_direction = None
            self.run_phonon_solver(np.array([self._bz_grid.gp_Gamma], dtype="int64"))
            self._nac_q_direction = _nac_q_direction

    def run_phonon_solver_with_eigvec_rotation(self):
        """Phonons at ir-grid-points are copied by proper rotations.

        Some phonons that are not covered by rotations are solved.

        The following data are updated.
            self._frequencies
            self._eigenvectors
            self._phonon_done

        """
        if self._phonon_done is None:
            raise RuntimeError(
                "Phonons are not initialized. Call init_dynamical_matrix() first."
            )

        self._phonon_done[:] = 0
        ir_grid_points, _, _ = get_ir_grid_points(self._bz_grid)
        ir_bz_grid_points = self._bz_grid.grg2bzg[ir_grid_points]
        self.run_phonon_solver(grid_points=ir_bz_grid_points)

        d2r_map = self._get_reciprocal_rotations_in_space_group_operations()

        # perms.shape = (len(spg_ops), len(primitive)), dtype='intc'
        perms = compute_all_sg_permutations(
            self._primitive.scaled_positions,
            self._bz_grid.symmetry_dataset.rotations,  # type: ignore
            self._bz_grid.symmetry_dataset.translations,  # type: ignore
            np.array(self._primitive.cell.T, dtype="double", order="C"),
            symprec=self._symprec,
        )

        for d_i, r_i in enumerate(d2r_map):
            r = self._bz_grid.rotations[r_i]
            r_cart = self._bz_grid.rotations_cartesian[r_i]
            for irgp in ir_bz_grid_points:
                bzgp = get_grid_points_by_rotations(
                    irgp,
                    self._bz_grid,
                    reciprocal_rotations=[
                        r,
                    ],
                    with_surface=True,
                )[0]
                if self._phonon_done[bzgp]:
                    continue

                self._rotate_eigvecs(irgp, bzgp, r_cart, perms[d_i], d_i)

        bz_grid_points_solved = self._get_phonons_at_minus_q()
        if bz_grid_points_solved:
            print("DEBUG: BZ-grid points additionally solved than ir-grid-points.")
            qpoints = np.dot(
                self._bz_grid.addresses[bz_grid_points_solved]
                / self._bz_grid.D_diag.astype("double"),
                self._bz_grid.Q.T,
            )
            distances = np.linalg.norm(
                np.dot(qpoints, np.linalg.inv(self._primitive.cell).T), axis=1
            )
            for qpt, dist in zip(qpoints, distances, strict=True):
                print(qpt, dist)

    def _get_reciprocal_rotations_in_space_group_operations(self):
        """Collect reciprocal rotations that belong to space group operations.

        Exclude reciprocal rotations that are made by time reversal symmetry.

        Returns
        -------
        d2r_map : list
            Indices of reciprocal rotations.

        """
        d2r_map = []
        for r in self._bz_grid.symmetry_dataset.rotations:  # type: ignore
            for i, rec_r in enumerate(self._bz_grid.reciprocal_operations):
                if (rec_r.T == r).all():
                    d2r_map.append(i)
                    break

        assert len(d2r_map) == len(self._bz_grid.symmetry_dataset.rotations)  # type: ignore

        return d2r_map

    def _rotate_eigvecs(self, orig_gp, bzgp, r_cart, perm, t_i):
        r"""Rotate eigenvectors at q to those Rq.

        e_j'(Rq) = R e_j(q) exp(-iRq.\tau)

        """
        assert self._phonon_done is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None

        Rq = np.dot(self._bz_grid.QDinv, self._bz_grid.addresses[bzgp])
        tau = self._bz_grid.symmetry_dataset.translations[t_i]  # type: ignore
        phase_factor = np.exp(-2j * np.pi * np.dot(Rq, tau))
        self._phonon_done[bzgp] = 1
        self._frequencies[bzgp, :] = self._frequencies[orig_gp, :]
        eigvecs = self._eigenvectors[orig_gp, :, :] * phase_factor
        for i, vec in enumerate(eigvecs.T):
            vec_perm = vec.reshape(-1, 3)[perm, :].T
            vec_rot = np.dot(r_cart, vec_perm).T.ravel()
            self._eigenvectors[bzgp, :, i] = vec_rot

    def _get_phonons_at_minus_q(self):
        """Phonons at -q are given by phonons at q.

        A few points may be uncovered by rotations. Those points are counted.

        Returns
        -------
        bz_grid_points_solved : list of int
            BZ-grid points where phonons that were additionally solved
            in this method.

        """
        assert self._phonon_done is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None

        r_inv = -np.eye(3, dtype="int64")
        bz_grid_points_solved = []
        for bzgp, done in enumerate(self._phonon_done):
            if done:
                continue

            # Get grid point at -q.
            bzgp_mq = get_grid_points_by_rotations(
                bzgp,
                self._bz_grid,
                reciprocal_rotations=[
                    r_inv,
                ],
                with_surface=True,
            )[0]

            if self._phonon_done[bzgp_mq] == 0:
                self.run_phonon_solver(
                    grid_points=np.array(
                        [
                            bzgp_mq,
                        ],
                        dtype="int64",
                    )
                )
                bz_grid_points_solved.append(bzgp_mq)

            self._phonon_done[bzgp] = 1
            self._frequencies[bzgp, :] = self._frequencies[bzgp_mq, :]
            self._eigenvectors[bzgp, :, :] = np.conj(self._eigenvectors[bzgp_mq, :, :])

        assert (self._phonon_done == 1).all()

        return bz_grid_points_solved

    def delete_interaction_strength(self):
        """Delete large arrays loosely.

        Memory deallocation would rely on garbage collector of python.
        So this may not work as expected.

        """
        self._interaction_strength = None
        self._g_zero = None

    def _set_fc3(self, fc3: NDArray, fc3_nonzero_indices: NDArray | None = None):
        if (
            isinstance(fc3, np.ndarray)
            and fc3.dtype == np.dtype("double")
            and fc3.flags.aligned
            and fc3.flags.owndata
            and fc3.flags.c_contiguous
            and self._frequency_scale_factor is None
        ):
            self._fc3 = fc3
        elif self._frequency_scale_factor is None:
            self._fc3 = np.array(fc3, dtype="double", order="C")
        else:
            self._fc3 = np.array(
                fc3 * self._frequency_scale_factor**2, dtype="double", order="C"
            )

        if fc3_nonzero_indices is None:
            self._fc3_nonzero_indices = np.ones(
                self._fc3.shape[:3], dtype="byte", order="C"
            )
        elif (
            isinstance(fc3_nonzero_indices, np.ndarray)
            and fc3_nonzero_indices.dtype == np.dtype("byte")
            and fc3_nonzero_indices.flags.aligned
            and fc3_nonzero_indices.flags.owndata
            and fc3_nonzero_indices.flags.c_contiguous
        ):
            self._fc3_nonzero_indices = fc3_nonzero_indices
        else:
            self._fc3_nonzero_indices = np.array(
                fc3_nonzero_indices, dtype="byte", order="C"
            )

    def _get_band_indices(self, band_indices) -> NDArray:
        num_band = len(self._primitive) * 3
        if band_indices is None:
            return np.arange(num_band, dtype="int64")
        else:
            return np.array(band_indices, dtype="int64")

    def _run_c(self, g_zero):
        import phono3py._phono3py as phono3c

        assert self._interaction_strength is not None
        assert self._triplets_at_q is not None

        num_band = len(self._primitive) * 3
        if g_zero is None or self._symmetrize_fc3q:
            _g_zero = np.zeros(
                self._interaction_strength.shape,
                dtype="byte",
                order="C",
            )
        else:
            _g_zero = g_zero

        # True: OpenMP over triplets
        # False: OpenMP over bands
        if self._openmp_per_triplets is None:
            if len(self._triplets_at_q) > num_band:
                openmp_per_triplets = True
            else:
                openmp_per_triplets = False
        else:
            openmp_per_triplets = self._openmp_per_triplets

        phono3c.interaction(
            self._interaction_strength,
            _g_zero,
            self._frequencies,
            self._eigenvectors,
            self._triplets_at_q,
            self._bz_grid.addresses,
            self._bz_grid.D_diag,
            self._bz_grid.Q,
            self._fc3,
            self._fc3_nonzero_indices,
            self._svecs,
            self._multi,
            self._masses,
            self._p2s,
            self._s2p,
            self._band_indices,
            self._symmetrize_fc3q * 1,
            self._make_r0_average * 1,
            self._all_shortest,
            self._cutoff_frequency,
            openmp_per_triplets * 1,
        )
        self._interaction_strength *= self._unit_conversion
        self._g_zero = g_zero

    def _run_phonon_solver_c(self, grid_points):
        run_phonon_solver_c(
            self._dm,
            self._frequencies,
            self._eigenvectors,
            self._phonon_done,
            grid_points,
            self._bz_grid.addresses,
            self._bz_grid.QDinv,
            frequency_conversion_factor=self._frequency_factor_to_THz,
            nac_q_direction=self._nac_q_direction,
            lapack_zheev_uplo=self._lapack_zheev_uplo,
        )

    def _run_py(self):
        assert self._interaction_strength is not None
        assert self._triplets_at_q is not None
        assert self._frequencies is not None
        assert self._eigenvectors is not None

        r2r = RealToReciprocal(
            self._fc3, self._primitive, self.mesh_numbers, symprec=self._symprec
        )
        r2n = ReciprocalToNormal(
            self._primitive,
            self._frequencies,
            self._eigenvectors,
            self._band_indices,
            cutoff_frequency=self._cutoff_frequency,
        )

        for i, grid_triplet in enumerate(self._triplets_at_q):
            print("%d / %d" % (i + 1, len(self._triplets_at_q)))
            r2r.run(self._bz_grid.addresses[grid_triplet])
            fc3_reciprocal = r2r.get_fc3_reciprocal()
            for gp in grid_triplet:
                self._run_phonon_solver_py(gp)
            r2n.run(fc3_reciprocal, grid_triplet)
            fc3_normal = r2n.get_reciprocal_to_normal()
            assert fc3_normal is not None
            self._interaction_strength[i] = (
                np.abs(fc3_normal) ** 2 * self._unit_conversion
            )

    def _run_phonon_solver_py(self, grid_point):
        run_phonon_solver_py(
            grid_point,
            self._phonon_done,
            self._frequencies,
            self._eigenvectors,
            self._bz_grid.addresses,
            self._bz_grid.QDinv,
            self._dm,
            self._frequency_factor_to_THz,
            self._lapack_zheev_uplo,
        )

    def _allocate_phonon(self):
        """Allocate phonon arrays.

        Phonons at Gamma point without NAC are stored in `self._frequencies_at_gamma`
        and `self._eigenvectors_at_gamma`.

        """
        num_band = len(self._primitive) * 3
        num_grid = len(self._bz_grid.addresses)
        self._phonon_done = np.zeros(num_grid, dtype="byte")
        self._phonon_all_done = False
        self._frequencies = np.zeros((num_grid, num_band), dtype="double", order="C")
        complex_dtype = "c%d" % (np.dtype("double").itemsize * 2)
        self._eigenvectors = np.zeros(
            (num_grid, num_band, num_band), dtype=complex_dtype, order="C"
        )
        gp_Gamma = self._bz_grid.gp_Gamma
        self.run_phonon_solver_at_gamma()
        self._frequencies_at_gamma = self._frequencies[gp_Gamma].copy()
        self._eigenvectors_at_gamma = self._eigenvectors[gp_Gamma].copy()
        self._phonon_done[gp_Gamma] = 0

    def _get_all_shortest(self):
        """Return array indicating distances among three atoms are all shortest.

        multi.shape = (n_satom, n_patom)
        svecs : distance with respect to primitive cell basis
        perms.shape = (n_pure_trans, n_satom)

        """
        svecs = self._svecs
        multi = self._multi
        n_satom, n_patom, _ = multi.shape
        perms = self._primitive.atomic_permutations
        s2pp_map = [self._primitive.p2p_map[i] for i in self._s2p]
        lattice = self._primitive.cell

        for i_patom, j_atom in np.ndindex((n_patom, n_satom)):
            if multi[j_atom, i_patom, 0] > 1:
                continue
            j_patom = s2pp_map[j_atom]
            i_perm = np.where(perms[:, j_atom] == self._p2s[j_patom])[0]
            assert len(i_perm) == 1
            for k_atom in range(n_satom):
                if multi[k_atom, i_patom, 0] > 1:
                    continue
                k_atom_mapped = perms[i_perm[0], k_atom]
                if multi[k_atom_mapped, j_patom, 0] > 1:
                    continue
                vec_jk = (
                    svecs[multi[k_atom, i_patom, 1]] - svecs[multi[j_atom, i_patom, 1]]
                )
                d_jk = np.linalg.norm(vec_jk @ lattice)
                d_jk_mapped = np.linalg.norm(
                    svecs[multi[k_atom_mapped, j_patom, 1]] @ lattice
                )
                if abs(d_jk_mapped - d_jk) < self._symprec:
                    self._all_shortest[i_patom, j_atom, k_atom] = 1


def all_bands_exist(interaction: Interaction):
    """Return if all bands are selected or not."""
    band_indices = interaction.band_indices
    num_band = len(interaction.primitive) * 3
    if len(band_indices) == num_band:
        if (band_indices - np.arange(num_band) == 0).all():
            return True
    return False
