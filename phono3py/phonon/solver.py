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

import numpy as np


def run_phonon_solver_c(dm,
                        frequencies,
                        eigenvectors,
                        phonon_done,
                        grid_points,
                        grid_address,
                        mesh,
                        frequency_conversion_factor,
                        nac_q_direction,  # in reduced coordinates
                        lapack_zheev_uplo,
                        verbose=False):
    import phono3py._phononmod as phononmod

    (svecs,
     multiplicity,
     masses,
     rec_lattice,  # column vectors
     positions,
     born,
     nac_factor,
     dielectric) = _extract_params(dm)

    if dm.is_nac() and dm.nac_method == 'gonze':
        gonze_nac_dataset = dm.Gonze_nac_dataset
        if gonze_nac_dataset[0] is None:
            dm.make_Gonze_nac_dataset()
            gonze_nac_dataset = dm.Gonze_nac_dataset
        (gonze_fc,  # fc where the dipole-diple contribution is removed.
         dd_q0,     # second term of dipole-dipole expression.
         G_cutoff,  # Cutoff radius in reciprocal space. This will not be used.
         G_list,    # List of G points where d-d interactions are integrated.
         Lambda) = gonze_nac_dataset  # Convergence parameter
        fc = gonze_fc
    else:
        positions = None
        dd_q0 = None
        G_list = None
        Lambda = 0
        fc = dm.force_constants

    # assert grid_points.dtype == 'int_'
    # assert grid_points.flags.c_contiguous

    fc_p2s, fc_s2p = _get_fc_elements_mapping(dm, fc)
    phononmod.phonons_at_gridpoints(
        frequencies,
        eigenvectors,
        phonon_done,
        np.array(grid_points, dtype='int_'),
        np.array(grid_address, dtype='int_', order='C'),
        np.array(mesh, dtype='int_'),
        fc,
        svecs,
        np.array(multiplicity, dtype='int_', order='C'),
        positions,
        masses,
        np.array(fc_p2s, dtype='int_'),
        np.array(fc_s2p, dtype='int_'),
        frequency_conversion_factor,
        born,
        dielectric,
        rec_lattice,
        nac_q_direction,
        nac_factor,
        dd_q0,
        G_list,
        Lambda,
        lapack_zheev_uplo)


def run_phonon_solver_py(grid_point,
                         phonon_done,
                         frequencies,
                         eigenvectors,
                         grid_address,
                         mesh,
                         dynamical_matrix,
                         frequency_conversion_factor,
                         lapack_zheev_uplo):
    gp = grid_point
    if phonon_done[gp] == 0:
        phonon_done[gp] = 1
        q = grid_address[gp].astype('double') / mesh
        dynamical_matrix.run(q)
        dm = dynamical_matrix.dynamical_matrix
        eigvals, eigvecs = np.linalg.eigh(dm, UPLO=lapack_zheev_uplo)
        eigvals = eigvals.real
        frequencies[gp] = (np.sqrt(np.abs(eigvals)) * np.sign(eigvals)
                           * frequency_conversion_factor)
        eigenvectors[gp] = eigvecs


def _extract_params(dm):
    svecs, multiplicity = dm.primitive.get_smallest_vectors()
    masses = np.array(dm.primitive.masses, dtype='double')
    rec_lattice = np.array(np.linalg.inv(dm.primitive.cell),
                           dtype='double', order='C')
    positions = np.array(dm.primitive.positions, dtype='double', order='C')
    if dm.is_nac():
        born = dm.born
        nac_factor = dm.nac_factor
        dielectric = dm.dielectric_constant
    else:
        born = None
        nac_factor = 0
        dielectric = None

    return (svecs,
            multiplicity,
            masses,
            rec_lattice,
            positions,
            born,
            nac_factor,
            dielectric)


def _get_fc_elements_mapping(dm, fc):
    p2s_map = dm.primitive.p2s_map
    s2p_map = dm.primitive.s2p_map
    if fc.shape[0] == fc.shape[1]:  # full fc
        fc_p2s = p2s_map
        fc_s2p = s2p_map
    else:  # compact fc
        primitive = dm.primitive
        p2p_map = primitive.p2p_map
        s2pp_map = np.array([p2p_map[s2p_map[i]] for i in range(len(s2p_map))],
                            dtype='intc')
        fc_p2s = np.arange(len(p2s_map), dtype='intc')
        fc_s2p = s2pp_map

    return fc_p2s, fc_s2p
