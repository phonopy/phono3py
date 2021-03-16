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
from phonopy.units import VaspToTHz, Hbar, EV, Angstrom, THz, AMU
from phono3py.phonon.solver import run_phonon_solver_c, run_phonon_solver_py
from phono3py.phonon3.real_to_reciprocal import RealToReciprocal
from phono3py.phonon3.reciprocal_to_normal import ReciprocalToNormal
from phono3py.phonon3.triplets import (get_triplets_at_q,
                                       get_nosym_triplets_at_q,
                                       get_bz_grid_address)


class Interaction(object):
    def __init__(self,
                 supercell,
                 primitive,
                 mesh,
                 symmetry,
                 fc3=None,
                 band_indices=None,
                 constant_averaged_interaction=None,
                 frequency_factor_to_THz=VaspToTHz,
                 frequency_scale_factor=None,
                 unit_conversion=None,
                 is_mesh_symmetry=True,
                 symmetrize_fc3q=False,
                 cutoff_frequency=None,
                 lapack_zheev_uplo='L'):
        self._supercell = supercell
        self._primitive = primitive
        self._mesh = np.array(mesh, dtype='intc')
        self._symmetry = symmetry

        self._band_indices = None
        self._set_band_indices(band_indices)
        self._constant_averaged_interaction = constant_averaged_interaction
        self._frequency_factor_to_THz = frequency_factor_to_THz
        self._frequency_scale_factor = frequency_scale_factor

        if fc3 is not None:
            self._set_fc3(fc3)

        # Unit to eV^2
        if unit_conversion is None:
            num_grid = np.prod(self._mesh)
            self._unit_conversion = ((Hbar * EV) ** 3 / 36 / 8
                                     * EV ** 2 / Angstrom ** 6
                                     / (2 * np.pi * THz) ** 3
                                     / AMU ** 3 / num_grid
                                     / EV ** 2)
        else:
            self._unit_conversion = unit_conversion
        if cutoff_frequency is None:
            self._cutoff_frequency = 0
        else:
            self._cutoff_frequency = cutoff_frequency
        self._is_mesh_symmetry = is_mesh_symmetry
        self._symmetrize_fc3q = symmetrize_fc3q
        self._lapack_zheev_uplo = lapack_zheev_uplo

        self._symprec = symmetry.get_symmetry_tolerance()

        self._grid_point = None
        self._triplets_at_q = None
        self._weights_at_q = None
        self._triplets_map_at_q = None
        self._ir_map_at_q = None
        self._grid_address = None
        self._bz_map = None
        self._interaction_strength = None
        self._g_zero = None

        self._phonon_done = None
        self._frequencies = None
        self._eigenvectors = None
        self._dm = None
        self._nac_params = None
        self._nac_q_direction = None

        self._band_index_count = 0

        svecs, multiplicity = self._primitive.get_smallest_vectors()
        self._smallest_vectors = svecs
        self._multiplicity = multiplicity
        self._masses = np.array(self._primitive.masses, dtype='double')
        self._p2s = self._primitive.p2s_map
        self._s2p = self._primitive.s2p_map

        self._allocate_phonon()

    def run(self, lang='C', g_zero=None):
        num_band = len(self._primitive) * 3
        num_triplets = len(self._triplets_at_q)

        self._interaction_strength = np.empty(
            (num_triplets, len(self._band_indices), num_band, num_band),
            dtype='double')
        if self._constant_averaged_interaction is None:
            self._interaction_strength[:] = 0
            if lang == 'C':
                self._run_c(g_zero)
            else:
                self._run_py()
        else:
            num_grid = np.prod(self._mesh)
            self._interaction_strength[:] = (
                self._constant_averaged_interaction / num_grid)

    @property
    def interaction_strength(self):
        return self._interaction_strength

    def get_interaction_strength(self):
        warnings.warn("Use attribute, interaction_strength.",
                      DeprecationWarning)
        return self.interaction_strength

    @property
    def mesh_numbers(self):
        return self._mesh

    def get_mesh_numbers(self):
        warnings.warn("Use attribute, mesh_numbers.", DeprecationWarning)
        return self.mesh_numbers

    @property
    def is_mesh_symmetry(self):
        return self._is_mesh_symmetry

    @property
    def fc3(self):
        return self._fc3

    def get_fc3(self):
        warnings.warn("Use attribute, fc3.", DeprecationWarning)
        return self.fc3

    @property
    def dynamical_matrix(self):
        return self._dm

    def get_dynamical_matrix(self):
        warnings.warn("Use attribute, dynamical_matrix.", DeprecationWarning)
        return self.dynamical_matrix

    @property
    def primitive(self):
        return self._primitive

    def get_primitive(self):
        warnings.warn("Use attribute, primitive.", DeprecationWarning)
        return self.primitive

    @property
    def supercell(self):
        return self._supercell

    def get_supercell(self):
        warnings.warn("Use attribute, supercell.", DeprecationWarning)
        return self.supercell

    def get_triplets_at_q(self):
        return (self._triplets_at_q,
                self._weights_at_q,
                self._triplets_map_at_q,
                self._ir_map_at_q)

    @property
    def grid_address(self):
        return self._grid_address

    def get_grid_address(self):
        warnings.warn("Use attribute, grid_address.", DeprecationWarning)
        return self.grid_address

    @property
    def bz_map(self):
        return self._bz_map

    def get_bz_map(self):
        warnings.warn("Use attribute, bz_map.", DeprecationWarning)
        return self.bz_map

    @property
    def band_indices(self):
        return self._band_indices

    def get_band_indices(self):
        warnings.warn("Use attribute, band_indices.", DeprecationWarning)
        return self.band_indices

    @property
    def nac_params(self):
        return self._nac_params

    @property
    def nac_q_direction(self):
        return self._nac_q_direction

    def get_nac_q_direction(self):
        warnings.warn("Use attribute, nac_q_direction.", DeprecationWarning)
        return self.nac_q_direction

    @property
    def zero_value_positions(self):
        return self._g_zero

    def get_zero_value_positions(self):
        warnings.warn("Use attribute, zero_value_positions.",
                      DeprecationWarning)
        return self.zero_value_positions

    def get_phonons(self):
        return self._frequencies, self._eigenvectors, self._phonon_done

    @property
    def frequency_factor_to_THz(self):
        return self._frequency_factor_to_THz

    def get_frequency_factor_to_THz(self):
        warnings.warn("Use attribute, frequency_factor_to_THz.",
                      DeprecationWarning)
        return self.frequency_factor_to_THz

    @property
    def lapack_zheev_uplo(self):
        return self._lapack_zheev_uplo

    def get_lapack_zheev_uplo(self):
        warnings.warn("Use attribute, lapack_zheev_uplo.", DeprecationWarning)
        return self.lapack_zheev_uplo

    @property
    def cutoff_frequency(self):
        return self._cutoff_frequency

    def get_cutoff_frequency(self):
        warnings.warn("Use attribute, cutoff_frequency.", DeprecationWarning)
        return self.cutoff_frequency

    def get_averaged_interaction(self):
        """Return sum over phonon triplets of interaction strength

        See Eq.(21) of PRB 91, 094306 (2015)

        """

        # v[triplet, band0, band, band]
        v = self._interaction_strength
        w = self._weights_at_q
        v_sum = np.dot(w, v.sum(axis=2).sum(axis=2))
        return v_sum / np.prod(v.shape[2:])

    def get_primitive_and_supercell_correspondence(self):
        return (self._smallest_vectors,
                self._multiplicity,
                self._p2s,
                self._s2p,
                self._masses)

    def get_unit_conversion_factor(self):
        return self._unit_conversion

    def get_constant_averaged_interaction(self):
        return self._constant_averaged_interaction

    def set_interaction_strength(self, pp_strength, g_zero=None):
        self._interaction_strength = pp_strength
        self._g_zero = g_zero

    def set_grid_point(self, grid_point, stores_triplets_map=False):
        reciprocal_lattice = np.linalg.inv(self._primitive.cell)
        if not self._is_mesh_symmetry:
            (triplets_at_q,
             weights_at_q,
             grid_address,
             bz_map,
             triplets_map_at_q,
             ir_map_at_q) = get_nosym_triplets_at_q(
                 grid_point,
                 self._mesh,
                 reciprocal_lattice,
                 stores_triplets_map=stores_triplets_map)
        else:
            (triplets_at_q,
             weights_at_q,
             grid_address,
             bz_map,
             triplets_map_at_q,
             ir_map_at_q) = get_triplets_at_q(
                 grid_point,
                 self._mesh,
                 self._symmetry.get_pointgroup_operations(),
                 reciprocal_lattice,
                 stores_triplets_map=stores_triplets_map)

        # Special treatment of symmetry is applied when q_direction is used.
        if self._nac_q_direction is not None:
            if (grid_address[grid_point] == 0).all():
                self._phonon_done[grid_point] = 0
                self.run_phonon_solver(np.array([grid_point], dtype='uintp'))
                rotations = []
                for r in self._symmetry.get_pointgroup_operations():
                    dq = self._nac_q_direction
                    dq /= np.linalg.norm(dq)
                    diff = np.dot(dq, r) - dq
                    if (abs(diff) < 1e-5).all():
                        rotations.append(r)
                (triplets_at_q,
                 weights_at_q,
                 grid_address,
                 bz_map,
                 triplets_map_at_q,
                 ir_map_at_q) = get_triplets_at_q(
                     grid_point,
                     self._mesh,
                     np.array(rotations, dtype='intc', order='C'),
                     reciprocal_lattice,
                     is_time_reversal=False,
                     stores_triplets_map=stores_triplets_map)

        for triplet in triplets_at_q:
            sum_q = (grid_address[triplet]).sum(axis=0)
            if (sum_q % self._mesh != 0).any():
                print("============= Warning ==================")
                print("%s" % triplet)
                for tp in triplet:
                    print("%s %s" %
                          (grid_address[tp],
                           np.linalg.norm(
                               np.dot(reciprocal_lattice,
                                      grid_address[tp] /
                                      self._mesh.astype('double')))))
                print("%s" % sum_q)
                print("============= Warning ==================")

        self._grid_point = grid_point
        self._triplets_at_q = triplets_at_q
        self._weights_at_q = weights_at_q
        self._triplets_map_at_q = triplets_map_at_q
        # self._grid_address = grid_address
        # self._bz_map = bz_map
        self._ir_map_at_q = ir_map_at_q

    def init_dynamical_matrix(self,
                              fc2,
                              supercell,
                              primitive,
                              nac_params=None,
                              solve_dynamical_matrices=True,
                              decimals=None,
                              verbose=False):
        self._nac_params = nac_params
        self._dm = get_dynamical_matrix(
            fc2,
            supercell,
            primitive,
            nac_params=nac_params,
            frequency_scale_factor=self._frequency_scale_factor,
            decimals=decimals,
            symprec=self._symprec)

        self._phonon_done[0] = 0
        if solve_dynamical_matrices:
            self.run_phonon_solver(verbose=verbose)
        else:
            self.run_phonon_solver(np.array([0], dtype='uintp'),
                                   verbose=verbose)

        if (self._grid_address[0] == 0).all():
            if np.sum(self._frequencies[0] < self._cutoff_frequency) < 3:
                for i, f in enumerate(self._frequencies[0, :3]):
                    if not (f < self._cutoff_frequency):
                        self._frequencies[0, i] = 0
                        print("=" * 26 + " Warning " + "=" * 26)
                        print(" Phonon frequency of band index %d at Gamma "
                              "is calculated to be %f." % (i + 1, f))
                        print(" But this frequency is forced to be zero.")
                        print("=" * 61)

    def set_nac_q_direction(self, nac_q_direction=None):
        if nac_q_direction is not None:
            self._nac_q_direction = np.array(nac_q_direction, dtype='double')

    def set_phonon_data(self, frequencies, eigenvectors, grid_address):
        if grid_address.shape != self._grid_address.shape:
            raise RuntimeError("Input grid address size is inconsistent. "
                               "Setting phonons faild.")

        if (self._grid_address - grid_address).all():
            raise RuntimeError("Input grid addresses are inconsistent. "
                               "Setting phonons faild.")
        else:
            self._phonon_done[:] = 1
            self._frequencies[:] = frequencies
            self._eigenvectors[:] = eigenvectors

    def set_phonons(self, grid_points=None, verbose=False):
        msg = ("Interaction.set_phonons is deprecated at v2.0. "
               "Use Interaction.run_phonon_solver intead.")
        warnings.warn(msg, DeprecationWarning)

        self.run_phonon_solver(grid_points=grid_points, verbose=verbose)

    def run_phonon_solver(self, grid_points=None, verbose=False):
        if grid_points is None:
            _grid_points = np.arange(len(self._grid_address), dtype='uintp')
        else:
            _grid_points = grid_points
        self._run_phonon_solver_c(_grid_points, verbose=verbose)

    def delete_interaction_strength(self):
        self._interaction_strength = None
        self._g_zero = None

    def _set_fc3(self, fc3):
        if (type(fc3) == np.ndarray and
            fc3.dtype == np.dtype('double') and
            fc3.flags.aligned and
            fc3.flags.owndata and
            fc3.flags.c_contiguous and
            self._frequency_scale_factor is None):
            self._fc3 = fc3
        elif self._frequency_scale_factor is None:
            self._fc3 = np.array(fc3, dtype='double', order='C')
        else:
            self._fc3 = np.array(fc3 * self._frequency_scale_factor ** 2,
                                 dtype='double', order='C')

    def _set_band_indices(self, band_indices):
        num_band = len(self._primitive) * 3
        if band_indices is None:
            self._band_indices = np.arange(num_band, dtype='intc')
        else:
            self._band_indices = np.array(band_indices, dtype='intc')

    def _run_c(self, g_zero):
        import phono3py._phono3py as phono3c

        if g_zero is None or self._symmetrize_fc3q:
            _g_zero = np.zeros(self._interaction_strength.shape,
                               dtype='byte', order='C')
        else:
            _g_zero = g_zero

        phono3c.interaction(self._interaction_strength,
                            _g_zero,
                            self._frequencies,
                            self._eigenvectors,
                            self._triplets_at_q,
                            self._grid_address,
                            self._mesh,
                            self._fc3,
                            self._smallest_vectors,
                            self._multiplicity,
                            self._masses,
                            self._p2s,
                            self._s2p,
                            self._band_indices,
                            self._symmetrize_fc3q,
                            self._cutoff_frequency)
        self._interaction_strength *= self._unit_conversion
        self._g_zero = g_zero

    def _run_phonon_solver_c(self, grid_points, verbose=False):
        run_phonon_solver_c(self._dm,
                            self._frequencies,
                            self._eigenvectors,
                            self._phonon_done,
                            grid_points,
                            self._grid_address,
                            self._mesh,
                            self._frequency_factor_to_THz,
                            self._nac_q_direction,
                            self._lapack_zheev_uplo,
                            verbose=verbose)

    def _run_py(self):
        r2r = RealToReciprocal(self._fc3,
                               self._supercell,
                               self._primitive,
                               self._mesh,
                               symprec=self._symprec)
        r2n = ReciprocalToNormal(self._primitive,
                                 self._frequencies,
                                 self._eigenvectors,
                                 self._band_indices,
                                 cutoff_frequency=self._cutoff_frequency)

        for i, grid_triplet in enumerate(self._triplets_at_q):
            print("%d / %d" % (i + 1, len(self._triplets_at_q)))
            r2r.run(self._grid_address[grid_triplet])
            fc3_reciprocal = r2r.get_fc3_reciprocal()
            for gp in grid_triplet:
                self._run_phonon_solver_py(gp)
            r2n.run(fc3_reciprocal, grid_triplet)
            self._interaction_strength[i] = np.abs(
                r2n.get_reciprocal_to_normal()) ** 2 * self._unit_conversion

    def _run_phonon_solver_py(self, grid_point):
        run_phonon_solver_py(grid_point,
                             self._phonon_done,
                             self._frequencies,
                             self._eigenvectors,
                             self._grid_address,
                             self._mesh,
                             self._dm,
                             self._frequency_factor_to_THz,
                             self._lapack_zheev_uplo)

    def _allocate_phonon(self):
        primitive_lattice = np.linalg.inv(self._primitive.cell)
        self._grid_address, self._bz_map = get_bz_grid_address(
            self._mesh, primitive_lattice, with_boundary=True)
        num_band = len(self._primitive) * 3
        num_grid = len(self._grid_address)
        self._phonon_done = np.zeros(num_grid, dtype='byte')
        self._frequencies = np.zeros((num_grid, num_band), dtype='double')
        itemsize = self._frequencies.itemsize
        self._eigenvectors = np.zeros((num_grid, num_band, num_band),
                                      dtype=("c%d" % (itemsize * 2)))
