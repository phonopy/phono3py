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

import warnings

import numpy as np
from phonopy.harmonic.dynamical_matrix import get_dynamical_matrix
from phonopy.structure.cells import (
    Primitive,
    compute_all_sg_permutations,
    sparse_to_dense_svecs,
)
from phonopy.structure.symmetry import Symmetry
from phonopy.units import AMU, EV, Angstrom, Hbar, THz, VaspToTHz

from phono3py.phonon3.real_to_reciprocal import RealToReciprocal
from phono3py.phonon3.reciprocal_to_normal import ReciprocalToNormal
from phono3py.phonon3.triplets import get_nosym_triplets_at_q, get_triplets_at_q
from phono3py.phonon.grid import (
    BZGrid,
    get_grid_points_by_rotations,
    get_ir_grid_points,
)
from phono3py.phonon.solver import run_phonon_solver_c, run_phonon_solver_py


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

    """

    def __init__(
        self,
        primitive: Primitive,
        bz_grid: BZGrid,
        primitive_symmetry: Symmetry,
        fc3=None,
        band_indices=None,
        constant_averaged_interaction=None,
        frequency_factor_to_THz=VaspToTHz,
        frequency_scale_factor=None,
        unit_conversion=None,
        is_mesh_symmetry=True,
        symmetrize_fc3q=False,
        cutoff_frequency=None,
        lapack_zheev_uplo="L",
    ):
        """Init method."""
        self._primitive = primitive
        self._bz_grid = bz_grid
        self._primitive_symmetry = primitive_symmetry

        self._band_indices = None
        self._set_band_indices(band_indices)
        self._constant_averaged_interaction = constant_averaged_interaction
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._frequency_scale_factor = frequency_scale_factor

        if fc3 is not None:
            self._set_fc3(fc3)

        # Unit to eV^2
        if unit_conversion is None:
            num_grid = np.prod(self.mesh_numbers)
            self._unit_conversion = (
                (Hbar * EV) ** 3
                / 36
                / 8
                * EV**2
                / Angstrom**6
                / (2 * np.pi * THz) ** 3
                / AMU**3
                / num_grid
                / EV**2
            )
        else:
            self._unit_conversion = unit_conversion
        if cutoff_frequency is None:
            self._cutoff_frequency = 0
        else:
            self._cutoff_frequency = cutoff_frequency
        self._is_mesh_symmetry = is_mesh_symmetry
        self._symmetrize_fc3q = symmetrize_fc3q
        self._lapack_zheev_uplo = lapack_zheev_uplo

        self._symprec = self._primitive_symmetry.tolerance

        self._triplets_at_q = None
        self._weights_at_q = None
        self._triplets_map_at_q = None
        self._ir_map_at_q = None
        self._interaction_strength = None
        self._g_zero = None

        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
        self._dm = None
        self._nac_params = None
        self._nac_q_direction = None

        self._band_index_count = 0

        svecs, multi = self._primitive.get_smallest_vectors()
        if self._primitive.store_dense_svecs:
            self._svecs = svecs
            self._multi = multi
        else:
            self._svecs, self._multi = sparse_to_dense_svecs(svecs, multi)

        self._masses = np.array(self._primitive.masses, dtype="double")
        self._p2s = np.array(self._primitive.p2s_map, dtype="int_")
        self._s2p = np.array(self._primitive.s2p_map, dtype="int_")

    def run(self, lang="C", g_zero=None):
        """Run ph-ph interaction calculation."""
        if (self._phonon_done == 0).any():
            self.run_phonon_solver()

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
    def interaction_strength(self):
        """Return ph-ph interaction strength.

        Returns
        -------
        ndarray
            shape=(num_ir_grid_points, num_specified_band, num_band, num_band),
            dtype='double', order='C'

        """
        return self._interaction_strength

    def get_interaction_strength(self):
        """Return ph-ph interaction strength."""
        warnings.warn(
            "Use attribute, Interaction.interaction_strength "
            "instead of Interaction.get_interaction_strength().",
            DeprecationWarning,
        )
        return self.interaction_strength

    @property
    def mesh_numbers(self):
        """Return mesh numbers.

        Returns
        -------
        ndarray
           shape=(3, ), dtype='int_'

        """
        return self._bz_grid.D_diag

    def get_mesh_numbers(self):
        """Return mesh numbers."""
        warnings.warn(
            "Use attribute, Interaction.mesh_numbers "
            "instead of Interaction.get_mesh_numbers().",
            DeprecationWarning,
        )
        return self.mesh_numbers

    @property
    def is_mesh_symmetry(self):
        """Whether symmetry of grid is utilized or not."""
        return self._is_mesh_symmetry

    @property
    def fc3(self):
        """Return fc3."""
        return self._fc3

    def get_fc3(self):
        """Return fc3."""
        warnings.warn(
            "Use attribute, Interaction.fc3 " "instead of Interaction.get_fc3().",
            DeprecationWarning,
        )
        return self.fc3

    @property
    def dynamical_matrix(self):
        """Return DynamicalMatrix class instance."""
        return self._dm

    def get_dynamical_matrix(self):
        """Return DynamicalMatrix class instance."""
        warnings.warn(
            "Use attribute, Interaction.dynamical_matrix "
            "instead of Interaction.get_dynamical_matrix().",
            DeprecationWarning,
        )
        return self.dynamical_matrix

    @property
    def primitive(self):
        """Return Primitive class instance."""
        return self._primitive

    def get_primitive(self):
        """Return Primitive class instance."""
        warnings.warn(
            "Use attribute, Interaction.primitive "
            "instead of Interaction.get_primitive().",
            DeprecationWarning,
        )
        return self.primitive

    @property
    def primitive_symmetry(self):
        """Return Symmetry class instance of primitive cell."""
        return self._primitive_symmetry

    def get_triplets_at_q(self):
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
    def bz_grid(self):
        """Return BZGrid class instance."""
        return self._bz_grid

    @property
    def band_indices(self):
        """Return band indices.

        Returns
        -------
        ndarray
            shape=(num_specified_bands, ), dtype='int_'

        """
        return self._band_indices

    def get_band_indices(self):
        """Return band indices."""
        warnings.warn(
            "Use attribute, Interaction.band_indices "
            "instead of Interaction.get_band_indices().",
            DeprecationWarning,
        )
        return self.band_indices

    @property
    def nac_params(self):
        """Return NAC params."""
        return self._nac_params

    @property
    def nac_q_direction(self):
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
            self._nac_q_direction = np.array(nac_q_direction, dtype="double")

    def get_nac_q_direction(self):
        """Return q-direction used for NAC at q->0."""
        warnings.warn(
            "Use attribute, Interaction.nac_q_direction "
            "instead of Interaction.get_nac_q_direction().",
            DeprecationWarning,
        )
        return self.nac_q_direction

    def set_nac_q_direction(self, nac_q_direction=None):
        """Set NAC q-point direction valid at q->0."""
        warnings.warn(
            "Use attribute, Interaction.nac_q_direction "
            "instead of Interaction.set_nac_q_direction().",
            DeprecationWarning,
        )
        self.nac_q_direction = nac_q_direction

    @property
    def zero_value_positions(self):
        """Return zero ph-ph interaction elements information.

        Returns
        -------
        shape is same as that of interaction_strength, dtype='byte', order='C'

        """
        return self._g_zero

    def get_zero_value_positions(self):
        """Return zero ph-ph interaction elements information."""
        warnings.warn(
            "Use attribute, Interaction.zero_value_positions "
            "instead of Interaction.get_zero_value_positions().",
            DeprecationWarning,
        )
        return self.zero_value_positions

    def get_phonons(self):
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
                1 if phonon at a grid point is calcualted, otherwise 0.
                shape=(num_bz_grid, ), dtype='byte'

        """
        return self._frequencies, self._eigenvectors, self._phonon_done

    @property
    def frequency_factor_to_THz(self):
        """Return phonon frequency conversion factor to THz."""
        return self._frequency_factor_to_THz

    def get_frequency_factor_to_THz(self):
        """Return phonon frequency conversion factor to THz."""
        warnings.warn(
            "Use attribute, Interaction.frequency_factor_to_THz ",
            "instead of Interaction.get_frequency_factor_to_THz().",
            DeprecationWarning,
        )
        return self.frequency_factor_to_THz

    @property
    def lapack_zheev_uplo(self):
        """Return U or L for lapack zheev solver."""
        return self._lapack_zheev_uplo

    def get_lapack_zheev_uplo(self):
        """Return U or L for lapack zheev solver."""
        warnings.warn(
            "Use attribute, Interaction.lapack_zheev_uplo "
            "instead of Interaction.get_lapack_zheev_uplo().",
            DeprecationWarning,
        )
        return self.lapack_zheev_uplo

    @property
    def cutoff_frequency(self):
        """Return cutoff phonon frequency to judge imaginary phonon."""
        return self._cutoff_frequency

    def get_cutoff_frequency(self):
        """Return cutoff phonon frequency to judge imaginary phonon."""
        warnings.warn(
            "Use attribute, Interaction.cutoff_frequency "
            "instead of Interaction.get_cutoff_frequency().",
            DeprecationWarning,
        )
        return self.cutoff_frequency

    def get_averaged_interaction(self):
        """Return sum over phonon triplets of interaction strength.

        See Eq.(21) of PRB 91, 094306 (2015)

        """
        # v[triplet, band0, band, band]
        v = self._interaction_strength
        w = self._weights_at_q
        v_sum = np.dot(w, v.sum(axis=2).sum(axis=2))
        return v_sum / np.prod(v.shape[2:])

    def get_primitive_and_supercell_correspondence(self):
        """Return atomic pair information."""
        return (self._svecs, self._multi, self._p2s, self._s2p, self._masses)

    def get_unit_conversion_factor(self):
        """Return unit conversion factor."""
        return self._unit_conversion

    def get_constant_averaged_interaction(self):
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
            (
                triplets_at_q,
                weights_at_q,
                triplets_map_at_q,
                ir_map_at_q,
            ) = get_triplets_at_q(grid_point, self._bz_grid, swappable=True)

            # Special treatment of symmetry is applied when q_direction is
            # used.
            if self._nac_q_direction is not None:
                if (self._bz_grid.addresses[grid_point] == 0).all():
                    self._phonon_done[grid_point] = 0
                    self.run_phonon_solver(
                        np.array(
                            [
                                grid_point,
                            ],
                            dtype="int_",
                        )
                    )
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
        solve_dynamical_matrices=True,
        decimals=None,
    ):
        """Prepare for phonon calculation on grid.

        solve_dynamical_matrices : bool
           When False, phonon calculation will be postponed.

        """
        self._allocate_phonon()
        self._nac_params = nac_params
        self._dm = get_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=self._frequency_scale_factor,
            decimals=decimals,
            symprec=self._symprec,
        )

        self._phonon_done[0] = 0
        if solve_dynamical_matrices:
            self.run_phonon_solver()
        else:
            self.run_phonon_solver(np.array([0], dtype="int_"))

        if (self._bz_grid.addresses[0] == 0).all():
            if np.sum(self._frequencies[0] < self._cutoff_frequency) < 3:
                for i, f in enumerate(self._frequencies[0, :3]):
                    if not (f < self._cutoff_frequency):
                        self._frequencies[0, i] = 0
                        print("=" * 26 + " Warning " + "=" * 26)
                        print(
                            " Phonon frequency of band index %d at Gamma "
                            "is calculated to be %f." % (i + 1, f)
                        )
                        print(" But this frequency is forced to be zero.")
                        print("=" * 61)

    def set_phonon_data(self, frequencies, eigenvectors, bz_grid_addresses):
        """Set phonons on grid."""
        if bz_grid_addresses.shape != self._bz_grid.addresses.shape:
            raise RuntimeError(
                "Input grid address size is inconsistent. " "Setting phonons faild."
            )

        if (self._bz_grid.addresses - bz_grid_addresses).all():
            raise RuntimeError(
                "Input grid addresses are inconsistent. " "Setting phonons faild."
            )
        else:
            self._phonon_done[:] = 1
            self._frequencies[:] = frequencies
            self._eigenvectors[:] = eigenvectors

    def run_phonon_solver(self, grid_points=None):
        """Run phonon solver at BZ-grid points."""
        if grid_points is None:
            self._get_phonons_at_all_bz_grid_points()
        else:
            _grid_points = grid_points
            self._run_phonon_solver_c(_grid_points)

    def _get_phonons_at_all_bz_grid_points(self, expand_phonons=False):
        """Get phonons at all BZ-grid points."""
        if expand_phonons:
            self._expand_phonons()
        else:
            grid_points = np.arange(len(self._bz_grid.addresses), dtype="int_")
            self._run_phonon_solver_c(grid_points)

    def _expand_phonons(self):
        """Phonons at ir-grid-points are copied by proper rotations.

        Some phonons that are not covered by rotations are solved.

        The following data are updated.
            self._frequencies
            self._eigenvectors
            self._phonon_done

        """
        ir_grid_points, _, _ = get_ir_grid_points(self._bz_grid)
        ir_bz_grid_points = self._bz_grid.grg2bzg[ir_grid_points]
        self._run_phonon_solver_c(ir_bz_grid_points)

        d2r_map = self._get_reciprocal_rotations_in_space_group_operations()

        # perms.shape = (len(spg_ops), len(primitive)), dtype='intc'
        perms = compute_all_sg_permutations(
            self._primitive.scaled_positions,
            self._bz_grid.symmetry_dataset["rotations"],
            self._bz_grid.symmetry_dataset["translations"],
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
            print("DEBUG: BZ-grid points additionally solved " "than ir-grid-points.")
            print(bz_grid_points_solved)

    def _get_reciprocal_rotations_in_space_group_operations(self):
        """Collect reciprocal rotations that belong to space group operations.

        Exclude reciprocal rotations that are made by time reversal symmetry.

        Returns
        -------
        d2r_map : list
            Indices of reciprocal rotations.

        """
        d2r_map = []
        for r in self._bz_grid.symmetry_dataset["rotations"]:
            for i, rec_r in enumerate(self._bz_grid.reciprocal_operations):
                if (rec_r.T == r).all():
                    d2r_map.append(i)
                    break

        assert len(d2r_map) == len(self._bz_grid.symmetry_dataset["rotations"])

        return d2r_map

    def _rotate_eigvecs(self, orig_gp, bzgp, r_cart, perm, t_i):
        r"""Rotate eigenvectors at q to those Rq.

        e_j'(Rq) = R e_j(q) exp(-iRq.\tau)

        """
        Rq = np.dot(self._bz_grid.QDinv, self._bz_grid.addresses[bzgp])
        tau = self._bz_grid.symmetry_dataset["translations"][t_i]
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
        r_inv = -np.eye(3, dtype="int_")
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
                self._run_phonon_solver_c(
                    np.array(
                        [
                            bzgp_mq,
                        ],
                        dtype="int_",
                    )
                )
                bz_grid_points_solved.append(bzgp_mq)

            self._phonon_done[bzgp] = 1
            self._frequencies[bzgp, :] = self._frequencies[bzgp_mq, :]
            self._eigenvectors[bzgp, :, :] = np.conj(self._eigenvectors[bzgp_mq, :, :])

        return bz_grid_points_solved

    def delete_interaction_strength(self):
        """Delete large arrays loosely.

        Memory deallocation would rely on garbage collector of python.
        So this may not work as expected.

        """
        self._interaction_strength = None
        self._g_zero = None

    def _set_fc3(self, fc3):
        if (
            type(fc3) == np.ndarray
            and fc3.dtype == np.dtype("double")
            and fc3.flags.aligned
            and fc3.flags.owndata
            and fc3.flags.c_contiguous
            and self._frequency_scale_factor is None
        ):  # noqa E129
            self._fc3 = fc3
        elif self._frequency_scale_factor is None:
            self._fc3 = np.array(fc3, dtype="double", order="C")
        else:
            self._fc3 = np.array(
                fc3 * self._frequency_scale_factor**2, dtype="double", order="C"
            )

    def _set_band_indices(self, band_indices):
        num_band = len(self._primitive) * 3
        if band_indices is None:
            self._band_indices = np.arange(num_band, dtype="int_")
        else:
            self._band_indices = np.array(band_indices, dtype="int_")

    def _run_c(self, g_zero):
        import phono3py._phono3py as phono3c

        if g_zero is None or self._symmetrize_fc3q:
            _g_zero = np.zeros(
                self._interaction_strength.shape, dtype="byte", order="C"
            )
        else:
            _g_zero = g_zero

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
            self._svecs,
            self._multi,
            self._masses,
            self._p2s,
            self._s2p,
            self._band_indices,
            self._symmetrize_fc3q,
            self._cutoff_frequency,
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
            self._interaction_strength[i] = (
                np.abs(r2n.get_reciprocal_to_normal()) ** 2 * self._unit_conversion
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
        num_band = len(self._primitive) * 3
        num_grid = len(self._bz_grid.addresses)
        self._phonon_done = np.zeros(num_grid, dtype="byte")
        self._frequencies = np.zeros((num_grid, num_band), dtype="double", order="C")
        itemsize = self._frequencies.itemsize
        self._eigenvectors = np.zeros(
            (num_grid, num_band, num_band), dtype=("c%d" % (itemsize * 2)), order="C"
        )
